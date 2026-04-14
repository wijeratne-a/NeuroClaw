"""Logger adapter injecting run_uuid into every record."""

from __future__ import annotations

import logging
from typing import Any


class RunContextAdapter(logging.LoggerAdapter):
    """Inject run_uuid into kwargs['extra'] for JsonFormatter."""

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        extra = kwargs.setdefault("extra", {})
        if isinstance(extra, dict) and self.extra:
            extra.setdefault("run_uuid", self.extra.get("run_uuid"))
        return msg, kwargs
