# Copyright 2025 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from inference_perf.client.metricsclient.base import StageRuntimeInfo, StageStatus
from inference_perf.utils.request_queue import RequestQueue
from inference_perf.datagen import DataGenerator, SyntheticDataGenerator
from inference_perf.client.modelserver import ModelServerClient
from inference_perf.config import (
    APIConfig,
    DataConfig,
    DataGenType,
    Distribution,
    FairnessConfig,
    LoadConfig,
    ProgramConfig,
)
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from .load_generator import RequestQueueData, Worker

from typing import Dict, List, Optional
from types import FrameType
import time
import multiprocessing as mp
from multiprocessing.synchronize import Event as SyncEvent
from multiprocessing.sharedctypes import Synchronized
from asyncio import sleep
import logging
import signal

logger = logging.getLogger(__name__)


def _create_program_datagen(
    program: ProgramConfig,
    api_config: APIConfig,
    tokenizer: Optional[CustomTokenizer],
) -> DataGenerator:
    """Create a SyntheticDataGenerator for a program with fixed token counts."""
    data_config = DataConfig(
        type=DataGenType.Synthetic,
        input_distribution=Distribution(
            min=program.prompt_tokens,
            max=program.prompt_tokens,
            mean=float(program.prompt_tokens),
            std_dev=0,
            total_count=program.total_requests,
        ),
        output_distribution=Distribution(
            min=program.max_tokens,
            max=program.max_tokens,
            mean=float(program.max_tokens),
            std_dev=0,
            total_count=program.total_requests,
        ),
    )
    return SyntheticDataGenerator(api_config, data_config, tokenizer)


class MultiProgramLoadGenerator:
    """
    Runs multiple independent programs (tenants) concurrently.
    Each program has its own request count, concurrency limit, token profile,
    start time offset, and optional fairness header.

    Reuses the existing Worker class from load_generator.py — one Worker per program.
    """

    def __init__(
        self,
        programs: List[ProgramConfig],
        api_config: APIConfig,
        fairness_config: Optional[FairnessConfig],
        load_config: LoadConfig,
        tokenizer: Optional[CustomTokenizer] = None,
    ) -> None:
        self.programs = programs
        self.api_config = api_config
        self.fairness_config = fairness_config
        self.load_config = load_config
        self.tokenizer = tokenizer
        self.stage_runtime_info: Dict[int, StageRuntimeInfo] = {}
        self.workers: List[Worker] = []
        self.interrupt_sig = False
        signal.signal(signal.SIGINT, self._sigint_handler)

    def _sigint_handler(self, _signum: int, _frame: Optional[FrameType]) -> None:
        self.interrupt_sig = True

    async def run(self, client: ModelServerClient) -> None:
        num_programs = len(self.programs)

        # One queue channel per program (each program gets a dedicated worker)
        request_queue: RequestQueue[RequestQueueData] = RequestQueue(num_programs)
        finished_requests_counter: "Synchronized[int]" = mp.Value("i", 0)
        active_requests_counter: "Synchronized[int]" = mp.Value("i", 0)
        request_phase: SyncEvent = mp.Event()
        stop_signal: SyncEvent = mp.Event()
        cancel_signal: SyncEvent = mp.Event()

        request_phase.set()

        # Create one Worker per program with program-specific concurrency
        for idx, program in enumerate(self.programs):
            datagen = _create_program_datagen(program, self.api_config, self.tokenizer)

            worker = Worker(
                id=idx,
                client=client,
                request_queue=request_queue.get_channel(idx),
                datagen=datagen,
                max_concurrency=program.concurrency,
                stop_signal=stop_signal,
                cancel_signal=cancel_signal,
                request_phase=request_phase,
                finished_requests_counter=finished_requests_counter,
                active_requests_counter=active_requests_counter,
                shared_max_concurrency=None,
                base_seed=self.load_config.base_seed + idx,
            )
            self.workers.append(worker)
            worker.start()

        # Compute total requests for progress tracking
        total_requests = sum(p.total_requests for p in self.programs)

        # Small delay to let workers start their event loops
        start_time = time.perf_counter() + 1
        start_time_epoch = time.time()

        # Enqueue requests for all programs
        for idx, program in enumerate(self.programs):
            extra_headers: Optional[dict[str, str]] = None
            if not program.no_fairness_header and self.fairness_config:
                extra_headers = {self.fairness_config.header_key: program.name}

            # Compute the request time offset for this program's start_time
            program_start = start_time + program.start_time

            datagen = _create_program_datagen(program, self.api_config, self.tokenizer)
            data_gen = datagen.get_data()

            for req_idx in range(program.total_requests):
                request_data = next(data_gen)
                # All requests for a program are enqueued at program_start;
                # concurrency is controlled by the Worker's semaphore
                request_queue.put(
                    RequestQueueData(
                        stage_id=idx,
                        request_data=request_data,
                        request_time=program_start,
                        lora_adapter=None,
                        extra_headers=extra_headers,
                        program_id=program.name,
                    ),
                    channel_id=idx,
                )

        # Wait for all requests to complete
        logger.info("All program requests enqueued, waiting for completion (%d total)", total_requests)
        while finished_requests_counter.value < total_requests:
            if self.interrupt_sig:
                logger.info("MultiProgramLoadGenerator received SIGINT")
                cancel_signal.set()
                await sleep(1)
                break
            await sleep(1)
            logger.debug(
                "Progress: %d/%d requests completed, %d active",
                finished_requests_counter.value, total_requests, active_requests_counter.value,
            )

        # Wait for queues to drain
        request_phase.clear()
        request_queue.join()

        end_time_epoch = time.time()

        # Record stage_runtime_info per program
        for idx, program in enumerate(self.programs):
            self.stage_runtime_info[idx] = StageRuntimeInfo(
                stage_id=idx,
                rate=float(program.concurrency),  # closest analog to QPS for program-based load
                start_time=start_time_epoch + program.start_time,
                end_time=end_time_epoch,
                status=StageStatus.COMPLETED if not self.interrupt_sig else StageStatus.FAILED,
                concurrency_level=program.concurrency,
            )

        # Stop workers
        stop_signal.set()
        request_phase.set()  # Wake workers stuck in wait

    async def stop(self) -> None:
        for worker in self.workers:
            worker.join(timeout=1.0)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=0.0)
