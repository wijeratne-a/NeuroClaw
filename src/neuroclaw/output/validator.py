"""Validate tensor shape, z-scored stats, drift gates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import zstandard
from safetensors.numpy import load_file

from neuroclaw.extractor.alignment import DriftStats
from neuroclaw.model.tribe_wrapper import CORTICAL_DIM
from neuroclaw.output.schema import (
    DUAL_PASS_ROI_KEYS,
    INFERENCE_LAYOUT_DUAL_PASS_AGGREGATED,
    KEY_MODEL_METADATA,
    KEY_TIMESTAMPS,
    KEY_VOXELS_ALL,
    VOXEL_DIM,
)


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str]
    details: dict[str, Any]


def validate_voxel_stats(voxels_all: np.ndarray) -> list[str]:
    """
    Sanity-check finite-scale predictions. TRIBE outputs are not globally z-scored;
    subcortical columns may be zero-padded, so stats use the cortical slice only.
    """
    errs: list[str] = []
    x = np.asarray(voxels_all, dtype=np.float64)
    if x.ndim == 2 and x.shape[1] == VOXEL_DIM:
        x = x[:, :CORTICAL_DIM]
    m = float(np.mean(x))
    s = float(np.std(x))
    if abs(m) > 0.5:
        errs.append(f"mean {m} exceeds |0.5| (cortical slice)")
    if not (0.01 <= s <= 5.0):
        errs.append(f"std {s} not in [0.01, 5.0] (cortical slice)")
    return errs


def validate_shape(voxels_all: np.ndarray) -> list[str]:
    errs: list[str] = []
    if voxels_all.ndim != 2:
        errs.append(f"expected 2D voxels_all, got shape {voxels_all.shape}")
        return errs
    if voxels_all.shape[1] != VOXEL_DIM:
        errs.append(f"expected dim 2 = {VOXEL_DIM}, got {voxels_all.shape[1]}")
    return errs


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


def validate_dual_pass_tensors(
    tensors: dict[str, Any],
    drift_stats: DriftStats,
) -> ValidationResult:
    """Validate in-memory tensors for ``dual_pass_aggregated`` layout."""
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

    for k in DUAL_PASS_ROI_KEYS:
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
            "layout": "dual_pass_aggregated",
            "inference_layout": inference_layout,
            "roi_keys": sorted(DUAL_PASS_ROI_KEYS),
        },
    )


def validate_dual_pass_artifact_zst(
    zst_path: str,
    drift_stats: DriftStats,
) -> ValidationResult:
    """Validate ROI-only dual-pass ``.safetensors.zst``."""
    tensors = _load_safetensors_from_zst(zst_path)
    return validate_dual_pass_tensors(tensors, drift_stats)


def validate_artifact_zst(
    zst_path: str,
    drift_stats: DriftStats,
) -> ValidationResult:
    """Decompress, load safetensors, run all gates (legacy grid or dual-pass ROI)."""
    tensors = _load_safetensors_from_zst(zst_path)

    meta = _parse_embedded_metadata(tensors)
    inf = meta.get("inference_layout", "unknown")
    is_dual = inf == INFERENCE_LAYOUT_DUAL_PASS_AGGREGATED or (
        KEY_VOXELS_ALL not in tensors and DUAL_PASS_ROI_KEYS.issubset(tensors.keys())
    )
    if is_dual:
        return validate_dual_pass_tensors(tensors, drift_stats)

    errs: list[str] = []
    va = np.asarray(tensors[KEY_VOXELS_ALL])
    errs.extend(validate_shape(va))
    errs.extend(validate_voxel_stats(va))
    errs.extend(validate_drift(drift_stats))

    inference_layout = inf
    if not inference_layout or inference_layout == "unknown":
        inference_layout = meta.get("inference_layout", "unknown")

    return ValidationResult(
        ok=len(errs) == 0,
        errors=errs,
        details={"shape": va.shape, "inference_layout": inference_layout},
    )
