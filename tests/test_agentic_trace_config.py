import pytest

from inference_perf.config import Config, LoadType


def test_agentic_trace_config_parsing():
    config = Config.model_validate(
        {
            "load": {
                "type": "agentic_trace",
                "agentic_trace": {
                    "trace_file": "/tmp/test_trace.json",
                    "arrival_rate": 2.0,
                    "total_programs": 50,
                    "max_concurrency": 64,
                    "content_mode": "original",
                },
            }
        }
    )
    assert config.load.type == LoadType.AGENTIC_TRACE
    assert config.load.agentic_trace is not None
    assert config.load.agentic_trace.trace_file == "/tmp/test_trace.json"
    assert config.load.agentic_trace.arrival_rate == 2.0
    assert config.load.agentic_trace.total_programs == 50
    assert config.load.agentic_trace.max_concurrency == 64
    assert config.load.agentic_trace.content_mode == "original"


def test_agentic_trace_requires_config():
    """AGENTIC_TRACE load type must have agentic_trace config."""
    with pytest.raises(ValueError, match="agentic_trace"):
        Config.model_validate(
            {
                "load": {
                    "type": "agentic_trace",
                }
            }
        )


def test_non_agentic_trace_rejects_agentic_config():
    """Non-AGENTIC_TRACE load types should not have agentic_trace config."""
    with pytest.raises(ValueError, match="agentic_trace"):
        Config.model_validate(
            {
                "load": {
                    "type": "constant",
                    "agentic_trace": {
                        "trace_file": "/tmp/test.json",
                        "arrival_rate": 1.0,
                        "total_programs": 10,
                        "max_concurrency": 32,
                        "content_mode": "original",
                    },
                    "stages": [{"rate": 1.0, "duration": 10}],
                }
            }
        )
