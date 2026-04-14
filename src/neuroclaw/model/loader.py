"""Staged tower loading with RSS watermark (13.5 GB on 16 GB Mac)."""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any, TypeVar

import psutil

from neuroclaw.config import Settings

logger = logging.getLogger("neuroclaw.model.loader")

T = TypeVar("T")


def rss_gb() -> float:
    """Current process RSS in GB."""
    proc = psutil.Process()
    return proc.memory_info().rss / (1024.0**3)


@contextmanager
def staged_stage(
    name: str,
    settings: Settings,
    *,
    run_uuid: str | None = None,
) -> Generator[None, None, None]:
    """Log memory before/after a stage; warn if above watermark."""
    before = rss_gb()
    logger.info(
        "stage_start",
        extra={
            "component": name,
            "rss_gb": round(before, 3),
            "run_uuid": run_uuid,
        },
    )
    try:
        yield
    finally:
        after = rss_gb()
        logger.info(
            "stage_end",
            extra={
                "component": name,
                "rss_gb": round(after, 3),
                "run_uuid": run_uuid,
            },
        )
        if after > settings.MAX_MEMORY_GB:
            logger.warning(
                "memory_watermark_exceeded",
                extra={
                    "rss_gb": round(after, 3),
                    "max_gb": settings.MAX_MEMORY_GB,
                    "run_uuid": run_uuid,
                },
            )
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                mps_backend = getattr(torch.backends, "mps", None)
                if mps_backend is not None and mps_backend.is_available():
                    torch.mps.empty_cache()
            except ImportError:
                pass


def run_staged(
    name: str,
    fn: Callable[[], T],
    settings: Settings,
    run_uuid: str | None = None,
) -> T:
    """Run a callable inside staged_stage."""
    with staged_stage(name, settings, run_uuid=run_uuid):
        return fn()
