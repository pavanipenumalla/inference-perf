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

"""Load generator that replays agentic traces with DAG-driven request scheduling."""

import asyncio
import logging
import random
import signal
import time
from typing import Dict, Optional

from inference_perf.apis.chat import ChatCompletionAPIData, ChatMessage
from inference_perf.client.metricsclient.base import StageRuntimeInfo, StageStatus
from inference_perf.client.modelserver import ModelServerClient
from inference_perf.config import AgenticTraceConfig, FairnessConfig
from inference_perf.utils.agentic_trace_reader import AgenticTraceReader, TracedCall

logger = logging.getLogger(__name__)

# Default per-request timeout in seconds
DEFAULT_REQUEST_TIMEOUT = 120.0


class AgenticTraceLoadGenerator:
    """
    Replays agentic trace DAGs against a real LLM server.

    Each program is an async DAG executor: LLM calls send real HTTP requests,
    tool calls simulate latency via asyncio.sleep, and children fire concurrently
    on parent completion.

    Fan-in (merge) nodes wait for ALL parents to complete before executing,
    using a pending_parents counter as a barrier.
    """

    SEED = 42  # Fixed seed for reproducible program sampling

    def __init__(
        self,
        config: AgenticTraceConfig,
        client: ModelServerClient,
        fairness_config: Optional[FairnessConfig] = None,
    ) -> None:
        self.config = config
        self._client = client
        self.fairness_config = fairness_config
        self.reader = AgenticTraceReader(config.trace_file)
        self.stage_runtime_info: Dict[int, StageRuntimeInfo] = {}
        self.rng = random.Random(self.SEED)
        self._completed_requests = 0
        self._failed_requests = 0
        self._total_llm_calls = 0
        self.interrupt_sig = False
        signal.signal(signal.SIGINT, self._sigint_handler)

    def _sigint_handler(self, _signum: int, _frame: Optional[object]) -> None:
        self.interrupt_sig = True

    async def _execute_call(
        self,
        call: TracedCall,
        client: ModelServerClient,
        semaphore: asyncio.Semaphore,
        program_id: str,
        stage_id: int,
        pending_parents: Dict[int, int],
        scheduled_time: float = 0.0,
    ) -> None:
        """Execute a single call in the DAG, then signal children.

        After this call completes, decrements pending_parents for each child.
        When a child's pending_parents reaches 0 (all parents done), it fires.
        This implements barrier semantics for fan-in merge nodes.
        """
        if self.interrupt_sig:
            return

        if call.type == "llm":
            await semaphore.acquire()
            try:
                # Build request data based on content mode
                if self.config.content_mode == "synthetic":
                    token_count = call.prefill_tokens or 256
                    dummy_text = "x " * (token_count * 2)
                    content = dummy_text
                else:
                    content = call.input or ""
                # Prepend program_id to make each program instance unique,
                # preventing the server from shortcutting via prefix caching
                # when the same program template is sampled multiple times.
                content = f"Program: {program_id}\n{content}"
                request_data = ChatCompletionAPIData(
                    messages=[ChatMessage(role="user", content=content)],
                    max_tokens=call.decode_tokens or 128,
                )
                extra_headers: Optional[dict[str, str]] = None
                if self.fairness_config:
                    extra_headers = {self.fairness_config.header_key: program_id}
                try:
                    await asyncio.wait_for(
                        client.process_request(
                            request_data,
                            stage_id,
                            scheduled_time,
                            extra_headers=extra_headers,
                            program_id=program_id,
                        ),
                        timeout=DEFAULT_REQUEST_TIMEOUT,
                    )
                    self._completed_requests += 1
                except asyncio.TimeoutError:
                    self._failed_requests += 1
                    logger.warning(
                        "Request timed out after %.0fs for program %s call %s",
                        DEFAULT_REQUEST_TIMEOUT, program_id, call.id,
                    )
                    return  # Abandon subtree on timeout
            except Exception as e:
                self._failed_requests += 1
                logger.warning("Request failed for program %s call %s: %s", program_id, call.id, e)
                return  # Abandon subtree on failure
            finally:
                semaphore.release()

        elif call.type == "tool":
            # Simulate tool latency
            delay_s = (call.runtime_ms or 0) / 1000.0
            if delay_s > 0:
                await asyncio.sleep(delay_s)

        # Decrement pending_parents for each child, fire those that are ready.
        # Capture scheduled_time now — the moment the child becomes runnable —
        # so schedule_delay reflects real queuing/semaphore wait.
        ready_children = []
        child_scheduled_time = time.perf_counter()
        for child in call.children:
            child_key = id(child)
            pending_parents[child_key] -= 1
            if pending_parents[child_key] == 0:
                ready_children.append(child)

        if ready_children:
            await asyncio.gather(
                *(
                    self._execute_call(
                        child, client, semaphore, program_id, stage_id,
                        pending_parents, scheduled_time=child_scheduled_time,
                    )
                    for child in ready_children
                )
            )

    async def _execute_program(
        self,
        program_idx: int,
        client: ModelServerClient,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Sample a program from the pool and execute its DAG.

        Each program gets its own stage_id (= program_idx) so the report
        generator produces per-program metrics, matching MultiProgramLoadGenerator.
        """
        stage_id = program_idx
        program = self.rng.choice(self.reader.programs)
        program_id = f"{program.id}_{program_idx}"
        llm_calls = sum(1 for c in self._iter_calls(program.root) if c.type == "llm")
        logger.info(
            "Starting program %s (template %s, %d LLM calls)",
            program_id, program.id, llm_calls,
        )

        # Initialize pending parent counts for barrier synchronization.
        # Each call starts with its parent_count; root has 0 so it fires immediately.
        pending_parents: Dict[int, int] = {}
        for call in self._iter_calls(program.root):
            pending_parents[id(call)] = call.parent_count

        start_time_epoch = time.time()
        root_scheduled_time = time.perf_counter()
        await self._execute_call(
            program.root, client, semaphore, program_id, stage_id,
            pending_parents, scheduled_time=root_scheduled_time,
        )
        end_time_epoch = time.time()

        self.stage_runtime_info[stage_id] = StageRuntimeInfo(
            stage_id=stage_id,
            rate=self.config.arrival_rate,
            start_time=start_time_epoch,
            end_time=end_time_epoch,
            status=StageStatus.COMPLETED if not self.interrupt_sig else StageStatus.FAILED,
            concurrency_level=self.config.max_concurrency,
        )
        logger.info("Finished program %s (%.1fs)", program_id, end_time_epoch - start_time_epoch)

    @staticmethod
    def _iter_calls(root: TracedCall):
        """BFS iterate all calls in a DAG."""
        visited = set()
        queue = [root]
        while queue:
            call = queue.pop(0)
            if id(call) in visited:
                continue
            visited.add(id(call))
            yield call
            queue.extend(call.children)

    async def _log_progress(self) -> None:
        """Periodically log progress."""
        while True:
            await asyncio.sleep(10)
            logger.info(
                "Progress: %d/%d requests completed, %d failed, %d in-flight",
                self._completed_requests,
                self._total_llm_calls,
                self._failed_requests,
                self._total_llm_calls - self._completed_requests - self._failed_requests,
            )

    async def run(self, client: ModelServerClient) -> None:
        if not self.reader.programs:
            logger.warning("No programs in trace file %s", self.config.trace_file)
            return

        semaphore = asyncio.Semaphore(self.config.max_concurrency)
        start_time_epoch = time.time()
        interval = 1.0 / self.config.arrival_rate

        # Pre-compute total LLM calls for progress reporting
        # (approximate — actual programs are randomly sampled, but this gives a rough count)
        avg_llm_calls = sum(
            sum(1 for c in self._iter_calls(p.root) if c.type == "llm")
            for p in self.reader.programs
        ) / len(self.reader.programs)
        self._total_llm_calls = int(avg_llm_calls * self.config.total_programs)
        logger.info(
            "Starting agentic trace replay: %d programs, ~%d LLM requests, arrival_rate=%.1f/s, max_concurrency=%d",
            self.config.total_programs, self._total_llm_calls,
            self.config.arrival_rate, self.config.max_concurrency,
        )

        # Start progress logger
        progress_task = asyncio.create_task(self._log_progress())

        tasks = []
        for i in range(self.config.total_programs):
            if self.interrupt_sig:
                break
            task = asyncio.create_task(
                self._execute_program(i, client, semaphore)
            )
            tasks.append(task)
            # Wait for inter-arrival time before spawning next program
            if i < self.config.total_programs - 1:
                await asyncio.sleep(interval)

        # Wait for all programs to complete
        await asyncio.gather(*tasks, return_exceptions=True)

        # Stop progress logger
        progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass

        end_time_epoch = time.time()
        duration = end_time_epoch - start_time_epoch
        logger.info(
            "Agentic trace replay finished: %d completed, %d failed in %.1fs",
            self._completed_requests, self._failed_requests, duration,
        )

    async def stop(self) -> None:
        # Close the client session to avoid "Unclosed client session" errors.
        # The existing multi-process generators don't need this because each
        # worker process manages its own session. Our generator uses the client
        # directly in the main process.
        if hasattr(self._client, "close"):
            await self._client.close()
