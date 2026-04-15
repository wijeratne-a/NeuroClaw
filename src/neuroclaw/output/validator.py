"""Validate Cortical Four tensor layout, z-scored stats, drift gates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import zstandard
from safetensors.numpy import load_file

from neuroclaw.extractor.alignment import DriftStats
from neuroclaw.output.schema import (
    CORTICAL_FOUR_ROI_KEYS,
    INFERENCE_LAYOUT_CORTICAL_MARKETING_FOUR,
    KEY_MODEL_METADATA,
    KEY_TIMESTAMPS,
)


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str]
    details: dict[str, Any]


def validate_drift(stats: DriftStats) -> list[str]:
    errs: list[str] = []
    if stats.max_abs_drift_ms > 50.0:
        errs.append(f"max_abs_drift_ms {stats.max_abs_drift_ms} > 50")
    if stats.p95_drift_ms >= 20.0:
        errs.append(f"p95_drift_ms {stats.p95_drift_ms} >= 20")
    return errs


def _parse_embedded_metadata(tensors: dict[str, Any]) -> dict[str, Any]:
    meta_bytes = tensors.get(KEY_MODEL_METADATA)
    if meta_bytes is None:
        return {}
    try:
        return json.loads(meta_bytes.tobytes().decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {}


def _load_safetensors_from_zst(zst_path: str) -> dict[str, Any]:
    import tempfile
    from pathlib import Path

    p = Path(zst_path)
    raw = p.read_bytes()
    dctx = zstandard.ZstdDecompressor()
    data = dctx.decompress(raw)
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tf:
        tf.write(data)
        tmp = tf.name
    try:
        return dict(load_file(tmp))
    finally:
        Path(tmp).unlink(missing_ok=True)


def validate_cortical_four_tensors(
    tensors: dict[str, Any],
    drift_stats: DriftStats,
) -> ValidationResult:
    """Validate in-memory tensors for ``cortical_marketing_four`` layout."""
    errs: list[str] = []

    meta = _parse_embedded_metadata(tensors)
    inference_layout = meta.get("inference_layout", "unknown")

    if KEY_TIMESTAMPS not in tensors:
        errs.append("missing timestamps tensor")
    else:
        ts = np.asarray(tensors[KEY_TIMESTAMPS])
        if ts.ndim != 1:
            errs.append(f"timestamps expected 1D, got shape {ts.shape}")
        tlen = int(ts.shape[0])
        if tlen < 1:
            errs.append("timestamps length must be >= 1")

    for k in CORTICAL_FOUR_ROI_KEYS:
        if k not in tensors:
            errs.append(f"missing ROI tensor {k!r}")
            continue
        a = np.asarray(tensors[k])
        if a.ndim != 1:
            errs.append(f"{k} expected 1D float16 series, got shape {a.shape}")
        elif a.dtype != np.float16:
            errs.append(f"{k} expected dtype float16, got {a.dtype}")
        elif KEY_TIMESTAMPS in tensors and a.shape[0] != np.asarray(tensors[KEY_TIMESTAMPS]).shape[0]:
            errs.append(f"{k} length mismatch vs timestamps")

    errs.extend(validate_drift(drift_stats))

    return ValidationResult(
        ok=len(errs) == 0,
        errors=errs,
        details={
            "layout": "cortical_marketing_four",
            "inference_layout": inference_layout,
            "roi_keys": sorted(CORTICAL_FOUR_ROI_KEYS),
        },
    )


def validate_artifact_zst(
    zst_path: str,
    drift_stats: DriftStats,
) -> ValidationResult:
    """Decompress, load safetensors, validate Cortical Four ROI layout."""
    tensors = _load_safetensors_from_zst(zst_path)

    meta = _parse_embedded_metadata(tensors)
    inf = meta.get("inference_layout", "unknown")
    is_c4 = inf == INFERENCE_LAYOUT_CORTICAL_MARKETING_FOUR or CORTICAL_FOUR_ROI_KEYS.issubset(
        tensors.keys()
    )
    if is_c4:
        return validate_cortical_four_tensors(tensors, drift_stats)

    errs = [f"unknown artifact layout (expected {INFERENCE_LAYOUT_CORTICAL_MARKETING_FOUR})"]
    return ValidationResult(
        ok=False,
        errors=errs,
        details={"inference_layout": inf},
    )
