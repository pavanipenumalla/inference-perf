import asyncio
import json
import time
from unittest.mock import AsyncMock

import pytest

from inference_perf.config import AgenticTraceConfig
from inference_perf.loadgen.agentic_trace_runner import AgenticTraceLoadGenerator


SAMPLE_TRACE = {
    "trace_name": "test",
    "programs": [
        {
            "id": "prog1",
            "total_calls": 2,
            "total_decode_tokens": 192,
            "total_prefill_tokens": 512,
            "calls": [
                {
                    "id": "llm-1",
                    "type": "llm",
                    "prefill_tokens": 256,
                    "total_prefill_tokens": 256,
                    "decode_tokens": 128,
                    "input": "Hello world",
                    "output": "Response text",
                    "parent_id": None,
                    "children_ids": ["llm-2"],
                },
                {
                    "id": "llm-2",
                    "type": "llm",
                    "prefill_tokens": 128,
                    "total_prefill_tokens": 512,
                    "decode_tokens": 64,
                    "input": "Follow up",
                    "output": "More response",
                    "parent_id": "llm-1",
                    "children_ids": [],
                },
            ],
        }
    ],
}


TRACE_WITH_TOOL = {
    "trace_name": "test_tool",
    "programs": [
        {
            "id": "prog2",
            "total_calls": 3,
            "total_decode_tokens": 192,
            "total_prefill_tokens": 868,
            "calls": [
                {
                    "id": "llm-1",
                    "type": "llm",
                    "prefill_tokens": 256,
                    "total_prefill_tokens": 256,
                    "decode_tokens": 128,
                    "input": "query",
                    "output": "use tool",
                    "parent_id": None,
                    "children_ids": ["tool-1"],
                },
                {
                    "id": "tool-1",
                    "type": "tool",
                    "tool_name": "wiki_env",
                    "runtime_ms": 37.5,
                    "input": "search",
                    "output": "result",
                    "parent_id": "llm-1",
                    "children_ids": ["llm-2"],
                },
                {
                    "id": "llm-2",
                    "type": "llm",
                    "prefill_tokens": 100,
                    "total_prefill_tokens": 612,
                    "decode_tokens": 64,
                    "input": "continue",
                    "output": "done",
                    "parent_id": "tool-1",
                    "children_ids": [],
                },
            ],
        }
    ],
}


@pytest.fixture
def trace_file(tmp_path):
    path = tmp_path / "test_trace.json"
    path.write_text(json.dumps(SAMPLE_TRACE))
    return str(path)


@pytest.fixture
def tool_trace_file(tmp_path):
    path = tmp_path / "tool_trace.json"
    path.write_text(json.dumps(TRACE_WITH_TOOL))
    return str(path)


@pytest.fixture
def config(trace_file):
    return AgenticTraceConfig(
        trace_file=trace_file,
        arrival_rate=100.0,  # fast for testing
        total_programs=1,
        max_concurrency=64,
        content_mode="original",
    )


@pytest.fixture
def tool_config(tool_trace_file):
    return AgenticTraceConfig(
        trace_file=tool_trace_file,
        arrival_rate=100.0,
        total_programs=1,
        max_concurrency=64,
        content_mode="original",
    )


@pytest.fixture
def mock_client():
    client = AsyncMock()
    client.process_request = AsyncMock()
    return client


def test_load_generator_init(config, mock_client):
    gen = AgenticTraceLoadGenerator(config=config, client=mock_client)
    assert len(gen.reader.programs) == 1


def test_load_generator_run_sends_requests(config, mock_client):
    gen = AgenticTraceLoadGenerator(config=config, client=mock_client)
    asyncio.run(gen.run(mock_client))
    # Should have sent 2 LLM requests (sequential chain)
    assert mock_client.process_request.call_count == 2


def test_load_generator_tool_call_delay(tool_config, mock_client):
    gen = AgenticTraceLoadGenerator(config=tool_config, client=mock_client)
    start = time.monotonic()
    asyncio.run(gen.run(mock_client))
    elapsed = time.monotonic() - start
    # Should have sent 2 LLM requests (tool call in between adds ~37.5ms delay)
    assert mock_client.process_request.call_count == 2
    # The tool delay should be at least 30ms (allowing some slack)
    assert elapsed >= 0.030


def test_load_generator_stage_runtime_info(config, mock_client):
    gen = AgenticTraceLoadGenerator(config=config, client=mock_client)
    asyncio.run(gen.run(mock_client))
    # Each program gets its own stage (stage_id = program_idx)
    # Config has total_programs=1, so stage 0 should exist
    assert 0 in gen.stage_runtime_info
    info = gen.stage_runtime_info[0]
    assert info.stage_id == 0


def test_load_generator_reproducible(config, mock_client):
    """Two runs with same seed produce identical request sequences."""
    calls_run1 = []
    calls_run2 = []

    def capture_calls(storage, *args, **kwargs):
        storage.append(kwargs.get("program_id", args[3] if len(args) > 3 else None))

    client1 = AsyncMock()
    client1.process_request = AsyncMock(side_effect=lambda *a, **kw: capture_calls(calls_run1, *a, **kw))
    gen1 = AgenticTraceLoadGenerator(config=config, client=client1)
    asyncio.run(gen1.run(client1))

    client2 = AsyncMock()
    client2.process_request = AsyncMock(side_effect=lambda *a, **kw: capture_calls(calls_run2, *a, **kw))
    gen2 = AgenticTraceLoadGenerator(config=config, client=client2)
    asyncio.run(gen2.run(client2))

    assert calls_run1 == calls_run2


# Diamond/merge pattern: two branches fan out then merge back
#
#   root (llm) ──→ branch-a (llm) ──→ merge (tool) ──→ leaf (llm)
#              └──→ branch-b (llm) ──┘
#
# merge has 2 parents; leaf should only execute once, after both branches complete.
TRACE_WITH_MERGE = {
    "trace_name": "test_merge",
    "programs": [
        {
            "id": "prog_merge",
            "total_calls": 5,
            "total_decode_tokens": 256,
            "total_prefill_tokens": 512,
            "calls": [
                {
                    "id": "root",
                    "type": "llm",
                    "prefill_tokens": 64,
                    "total_prefill_tokens": 64,
                    "decode_tokens": 32,
                    "input": "start",
                    "output": "go",
                    "parent_id": None,
                    "children_ids": ["branch-a", "branch-b"],
                },
                {
                    "id": "branch-a",
                    "type": "llm",
                    "prefill_tokens": 64,
                    "total_prefill_tokens": 128,
                    "decode_tokens": 32,
                    "input": "branch a",
                    "output": "done a",
                    "parent_id": "root",
                    "children_ids": ["merge"],
                },
                {
                    "id": "branch-b",
                    "type": "llm",
                    "prefill_tokens": 64,
                    "total_prefill_tokens": 128,
                    "decode_tokens": 32,
                    "input": "branch b",
                    "output": "done b",
                    "parent_id": "root",
                    "children_ids": ["merge"],
                },
                {
                    "id": "merge",
                    "type": "tool",
                    "tool_name": "ReflectionMerge",
                    "runtime_ms": 0,
                    "input": "",
                    "output": "",
                    "parent_id": "branch-a",
                    "children_ids": ["leaf"],
                },
                {
                    "id": "leaf",
                    "type": "llm",
                    "prefill_tokens": 128,
                    "total_prefill_tokens": 256,
                    "decode_tokens": 64,
                    "input": "final",
                    "output": "result",
                    "parent_id": "merge",
                    "children_ids": [],
                },
            ],
        }
    ],
}


def test_merge_node_executes_once(tmp_path):
    """Merge node with 2 parents should cause its child to execute exactly once."""
    path = tmp_path / "merge_trace.json"
    path.write_text(json.dumps(TRACE_WITH_MERGE))
    config = AgenticTraceConfig(
        trace_file=str(path),
        arrival_rate=100.0,
        total_programs=1,
        max_concurrency=64,
        content_mode="original",
    )
    client = AsyncMock()
    client.process_request = AsyncMock()
    gen = AgenticTraceLoadGenerator(config=config, client=client)
    asyncio.run(gen.run(client))
    # 4 LLM calls: root + branch-a + branch-b + leaf (merge is a tool, no LLM request)
    # leaf must execute exactly once, not twice (one per parent of merge)
    assert client.process_request.call_count == 4


def test_merge_node_waits_for_all_parents(tmp_path):
    """Merge node's child should not fire until both parents complete."""
    path = tmp_path / "merge_trace.json"
    path.write_text(json.dumps(TRACE_WITH_MERGE))
    config = AgenticTraceConfig(
        trace_file=str(path),
        arrival_rate=100.0,
        total_programs=1,
        max_concurrency=64,
        content_mode="original",
    )

    # Track the order of call IDs to verify leaf comes after both branches
    call_order = []

    async def track_order(*args, **kwargs):
        # Extract the content from the request to identify which call this is
        data = args[0]
        content = data.messages[0].content
        call_order.append(content)

    client = AsyncMock()
    client.process_request = AsyncMock(side_effect=track_order)
    gen = AgenticTraceLoadGenerator(config=config, client=client)
    asyncio.run(gen.run(client))

    # "final" (leaf) must come after both "branch a" and "branch b"
    # Content has "Program: <id>\n" prefix, so use substring matching
    final_idx = next(i for i, c in enumerate(call_order) if "final" in c)
    assert any("branch a" in c for c in call_order[:final_idx])
    assert any("branch b" in c for c in call_order[:final_idx])
