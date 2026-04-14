"""Structured JSON logging for automated error parsing on long runs."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


_RESERVED_LOG_KEYS: frozenset[str] = frozenset(
    {
        "name",
        "msg",
        "args",
        "created",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "exc_info",
        "exc_text",
        "thread",
        "threadName",
        "taskName",
    }
)


class JsonFormatter(logging.Formatter):
    """One JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for key, val in record.__dict__.items():
            if key not in _RESERVED_LOG_KEYS and not key.startswith("_"):
                payload[key] = val
        return json.dumps(payload, default=str)


def setup_json_logging(level: int = logging.INFO) -> logging.Logger:
    """Root logger: stdout, JSON lines."""
    root = logging.getLogger("neuroclaw")
    root.setLevel(level)
    root.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    root.addHandler(h)
    return root


def get_logger(name: str = "neuroclaw") -> logging.Logger:
    return logging.getLogger(name)


class RunContextAdapter(logging.LoggerAdapter):
    """Inject run_uuid into every log record."""

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        extra = kwargs.setdefault("extra", {})
        if isinstance(extra, dict):
            extra.setdefault("run_uuid", self.extra.get("run_uuid") if self.extra else None)
        return msg, kwargs
