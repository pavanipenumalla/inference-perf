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

"""Reader for agentic trace JSON files produced by agentix's convert_traces.py."""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TracedCall:
    id: str
    type: str  # "llm" or "tool"
    # LLM fields
    prefill_tokens: Optional[int] = None
    total_prefill_tokens: Optional[int] = None
    decode_tokens: Optional[int] = None
    input: Optional[str] = None
    output: Optional[str] = None
    # Tool fields
    tool_name: Optional[str] = None
    runtime_ms: Optional[float] = None
    # DAG
    children: List["TracedCall"] = field(default_factory=list)
    # Number of parents — used for fan-in barrier synchronization.
    # A call with parent_count > 1 is a merge node that should wait for
    # all parents to complete before executing.
    parent_count: int = 0


@dataclass
class TracedProgram:
    id: str
    root: TracedCall
    total_calls: int
    total_decode_tokens: int
    total_prefill_tokens: int


class AgenticTraceReader:
    """Reads an agentic trace JSON file and produces TracedProgram objects."""

    def __init__(self, trace_file: str) -> None:
        self.trace_file = trace_file
        self.trace_name: str = ""
        self.programs: List[TracedProgram] = []
        self._load()

    def _load(self) -> None:
        with open(self.trace_file, "r") as f:
            data = json.load(f)

        self.trace_name = data.get("trace_name", "")

        for prog_data in data.get("programs", []):
            program = self._parse_program(prog_data)
            if program is not None:
                self.programs.append(program)

        logger.info("Loaded %d programs from %s", len(self.programs), self.trace_file)

    def _parse_program(self, prog_data: dict) -> Optional[TracedProgram]:
        calls_data = prog_data.get("calls", [])
        if not calls_data:
            return None

        # Build call objects indexed by id
        calls_by_id: Dict[str, TracedCall] = {}
        for call_data in calls_data:
            call = TracedCall(
                id=call_data["id"],
                type=call_data["type"],
                prefill_tokens=call_data.get("prefill_tokens"),
                total_prefill_tokens=call_data.get("total_prefill_tokens"),
                decode_tokens=call_data.get("decode_tokens"),
                input=call_data.get("input"),
                output=call_data.get("output"),
                tool_name=call_data.get("tool_name"),
                runtime_ms=call_data.get("runtime_ms"),
            )
            calls_by_id[call.id] = call

        # Wire up children references and compute parent counts
        for call_data in calls_data:
            call = calls_by_id[call_data["id"]]
            for child_id in call_data.get("children_ids", []):
                if child_id in calls_by_id:
                    call.children.append(calls_by_id[child_id])
                    calls_by_id[child_id].parent_count += 1

        # Find root (parent_id is None)
        root = None
        for call_data in calls_data:
            if call_data.get("parent_id") is None:
                root = calls_by_id[call_data["id"]]
                break

        if root is None:
            return None

        return TracedProgram(
            id=str(prog_data["id"]),
            root=root,
            total_calls=prog_data.get("total_calls", len(calls_data)),
            total_decode_tokens=prog_data.get("total_decode_tokens", 0),
            total_prefill_tokens=prog_data.get("total_prefill_tokens", 0),
        )
