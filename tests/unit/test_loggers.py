import json
import logging

import pytest
import structlog

import agent.loggers
from agent.config import LoggingConfig
from agent.loggers import get_logger, setup_logging


@pytest.fixture(autouse=True)
def _reset_logging_state():
    yield
    logger = logging.getLogger("agent")
    logger.handlers = []
    logger.propagate = True
    structlog.reset_defaults()
    agent.loggers._log_setup_done = False


def test_disabled_writes_nothing(tmp_path):
    cfg = LoggingConfig(enabled=False, level="DEBUG", file=str(tmp_path / "a.jsonl"))
    setup_logging(cfg)
    get_logger("test").info("hello")
    assert list(tmp_path.glob("*.jsonl")) == []


def test_enabled_writes_jsonl_lines(tmp_path):
    log_file = tmp_path / "agent.jsonl"
    cfg = LoggingConfig(enabled=True, level="DEBUG", file=str(log_file))
    setup_logging(cfg)
    get_logger("test").info("hello", extra=1)
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event"] == "hello"
    assert record["extra"] == 1
    assert record["logger"] == "agent.test"


def test_stdout_when_file_is_dash(tmp_path, capsys):
    cfg = LoggingConfig(enabled=True, level="INFO", file="-")
    setup_logging(cfg)
    get_logger("test").info("to stdout")
    assert "to stdout" in capsys.readouterr().out


def test_setup_logging_is_idempotent(tmp_path):
    log_file = tmp_path / "agent.jsonl"
    cfg = LoggingConfig(enabled=True, level="INFO", file=str(log_file))
    setup_logging(cfg)
    setup_logging(cfg)
    get_logger("test").info("one line")
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
