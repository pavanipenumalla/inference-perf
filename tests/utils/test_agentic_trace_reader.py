import json

import pytest

from inference_perf.utils.agentic_trace_reader import AgenticTraceReader


SAMPLE_TRACE = {
    "trace_name": "test_trace",
    "programs": [
        {
            "id": "prog1",
            "total_calls": 3,
            "total_decode_tokens": 192,
            "total_prefill_tokens": 868,
            "calls": [
                {
                    "id": "call-1",
                    "type": "llm",
                    "prefill_tokens": 256,
                    "total_prefill_tokens": 256,
                    "decode_tokens": 128,
                    "input": "Hello, what is Python?",
                    "output": "Python is a programming language.",
                    "parent_id": None,
                    "children_ids": ["call-2"],
                },
                {
                    "id": "call-2",
                    "type": "tool",
                    "tool_name": "wiki_env",
                    "runtime_ms": 37.5,
                    "input": "search python",
                    "output": "Python (programming language)...",
                    "parent_id": "call-1",
                    "children_ids": ["call-3"],
                },
                {
                    "id": "call-3",
                    "type": "llm",
                    "prefill_tokens": 100,
                    "total_prefill_tokens": 612,
                    "decode_tokens": 64,
                    "input": "Based on the search results...",
                    "output": "Python was created by Guido...",
                    "parent_id": "call-2",
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


def test_reader_loads_programs(trace_file):
    reader = AgenticTraceReader(trace_file)
    programs = reader.programs
    assert len(programs) == 1
    assert programs[0].id == "prog1"
    assert programs[0].total_calls == 3
    assert programs[0].total_decode_tokens == 192


def test_reader_builds_dag(trace_file):
    reader = AgenticTraceReader(trace_file)
    root = reader.programs[0].root
    assert root.id == "call-1"
    assert root.type == "llm"
    assert root.prefill_tokens == 256
    assert root.decode_tokens == 128
    assert len(root.children) == 1

    tool_call = root.children[0]
    assert tool_call.type == "tool"
    assert tool_call.tool_name == "wiki_env"
    assert tool_call.runtime_ms == 37.5
    assert len(tool_call.children) == 1

    leaf = tool_call.children[0]
    assert leaf.type == "llm"
    assert leaf.decode_tokens == 64
    assert len(leaf.children) == 0


def test_reader_trace_name(trace_file):
    reader = AgenticTraceReader(trace_file)
    assert reader.trace_name == "test_trace"


def test_reader_fan_out(tmp_path):
    """Test that a program with fan-out (multiple children) is loaded correctly."""
    trace = {
        "trace_name": "fanout_test",
        "programs": [
            {
                "id": "prog2",
                "total_calls": 3,
                "total_decode_tokens": 256,
                "total_prefill_tokens": 512,
                "calls": [
                    {
                        "id": "root",
                        "type": "llm",
                        "prefill_tokens": 256,
                        "total_prefill_tokens": 256,
                        "decode_tokens": 128,
                        "input": "prompt",
                        "output": "response",
                        "parent_id": None,
                        "children_ids": ["child-a", "child-b"],
                    },
                    {
                        "id": "child-a",
                        "type": "llm",
                        "prefill_tokens": 128,
                        "total_prefill_tokens": 384,
                        "decode_tokens": 64,
                        "input": "branch a",
                        "output": "result a",
                        "parent_id": "root",
                        "children_ids": [],
                    },
                    {
                        "id": "child-b",
                        "type": "llm",
                        "prefill_tokens": 128,
                        "total_prefill_tokens": 384,
                        "decode_tokens": 64,
                        "input": "branch b",
                        "output": "result b",
                        "parent_id": "root",
                        "children_ids": [],
                    },
                ],
            }
        ],
    }
    path = tmp_path / "fanout.json"
    path.write_text(json.dumps(trace))
    reader = AgenticTraceReader(str(path))
    root = reader.programs[0].root
    assert len(root.children) == 2
    assert {c.id for c in root.children} == {"child-a", "child-b"}
