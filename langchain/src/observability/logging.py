from __future__ import annotations

import logging
import os
import sys
from contextvars import ContextVar
from typing import Any

import structlog
from structlog.types import EventDict, Processor

from src.observability.redactor import redact, hash_user_id

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
user_id_var: ContextVar[str] = ContextVar("user_id", default="-")


def _bind_context(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    event_dict.setdefault("request_id", request_id_var.get())
    raw_user = user_id_var.get()
    event_dict.setdefault("user_hash", hash_user_id(raw_user) if raw_user != "-" else "-")
    return event_dict


def _redact_processor(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Last line of defence — scrub every value in the dict."""
    return redact(event_dict)


def configure_logging() -> None:
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _bind_context,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        _redact_processor,                    
        structlog.processors.JSONRenderer(),
    ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, log_level)),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "cv-builder") -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
