"""Structured JSON logging.

One JSON object per line, on stderr by default and optionally to a rotating
file under ``<data_root>/logs``. No network handlers, no telemetry.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import threading
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_LOGGER_NAME = "app"
_context = threading.local()

_RESERVED = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


def safe_extra(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Rename fields that would collide with ``LogRecord``'s own attributes.

    ``logging`` raises ``KeyError`` if ``extra`` contains e.g. ``name`` or
    ``message``. Structured fields come from many call sites, so the collision
    is renamed (``name`` -> ``name_``) rather than left to blow up at runtime.
    """
    out: dict[str, Any] = {}
    for key, value in fields.items():
        clean = f"{key}_" if key in _RESERVED or key in {"message", "msg", "asctime"} else key
        out[clean] = value
    return out


def _current_context() -> dict[str, Any]:
    return getattr(_context, "data", {})


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    """Attach ``fields`` to every log record emitted inside the block."""
    previous = dict(_current_context())
    merged = {**previous, **{k: v for k, v in fields.items() if v is not None}}
    _context.data = merged
    try:
        yield
    finally:
        _context.data = previous


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON with UTC timestamps."""

    @staticmethod
    def _timestamp(record: logging.LogRecord) -> str:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
        return f"{stamp}.{int(record.msecs):03d}Z"

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self._timestamp(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(_current_context())
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_type"] = getattr(record.exc_info[0], "__name__", "Exception")
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(
    level: str = "INFO",
    *,
    log_dir: str | os.PathLike[str] | None = None,
    filename: str = "app.jsonl",
    stream: Any = None,
    force: bool = True,
) -> logging.Logger:
    """Install the JSON handlers on the ``app`` logger."""
    logger = logging.getLogger(_LOGGER_NAME)
    numeric = getattr(logging, str(level).upper(), logging.INFO)
    logger.setLevel(numeric)
    logger.propagate = False
    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
    elif logger.handlers:
        return logger

    formatter = JsonFormatter()
    console = logging.StreamHandler(stream or sys.stderr)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_dir is not None:
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            directory / filename,
            maxBytes=16 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child of the ``app`` logger."""
    if not name or name == _LOGGER_NAME:
        return logging.getLogger(_LOGGER_NAME)
    suffix = name[len(_LOGGER_NAME) + 1 :] if name.startswith(_LOGGER_NAME + ".") else name
    return logging.getLogger(_LOGGER_NAME).getChild(suffix)


def new_correlation_id() -> str:
    return uuid.uuid4().hex[:16]


def log_event(logger: logging.Logger, event: str, /, **fields: Any) -> None:
    """Emit a named event at INFO with structured fields."""
    logger.info(event, extra=safe_extra({"event": event, **fields}))


__all__ = [
    "JsonFormatter",
    "configure_logging",
    "get_logger",
    "log_context",
    "log_event",
    "new_correlation_id",
    "safe_extra",
]
