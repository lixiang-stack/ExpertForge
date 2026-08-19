from __future__ import annotations

import logging
import sys

import structlog

from .config import LoggingConfig

_log_setup_done = False


def _configure_structlog() -> None:
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", key="ts"),
            structlog.stdlib.add_logger_name,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _configure_agent_stdlib_logger() -> None:
    logger = logging.getLogger("agent")
    logger.propagate = False
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())


def setup_logging(cfg: LoggingConfig) -> None:
    """(Re)configure the 'agent' logger and structlog. Idempotent: safe to call repeatedly."""
    global _log_setup_done

    if not _log_setup_done:
        _configure_structlog()
        _configure_agent_stdlib_logger()
        _log_setup_done = True

    logger = logging.getLogger("agent")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        if isinstance(handler, logging.FileHandler):
            handler.close()

    if not cfg.enabled:
        logger.addHandler(logging.NullHandler())
        return

    formatter = logging.Formatter("%(message)s")
    if cfg.file in ("-", "stdout"):
        handler = logging.StreamHandler(sys.stdout)
    else:
        handler = logging.FileHandler(cfg.file, encoding="utf-8")
    handler.setFormatter(formatter)
    logger.setLevel(getattr(logging, cfg.level.upper()))
    logger.addHandler(handler)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound to the 'agent.*' namespace."""
    return structlog.get_logger(f"agent.{name}")
