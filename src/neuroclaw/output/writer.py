"""Write .safetensors then zstd-compress to .safetensors.zst; optional 300s chunking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import zstandard
from safetensors.numpy import save_file

from neuroclaw.output.schema import (
    DUAL_PASS_ROI_KEYS,
    KEY_MODEL_METADATA,
    KEY_TIMESTAMPS,
    KEY_VOXELS_ALL,
    KEY_VOXELS_M4,
)


def _chunk_ranges(duration_s: float, chunk_s: float = 300.0) -> list[tuple[float, float]]:
    """Hard splits on time boundaries [t0, t1)."""
    if duration_s <= chunk_s:
        return [(0.0, duration_s)]
    ranges: list[tuple[float, float]] = []
    t = 0.0
    while t < duration_s:
        t1 = min(t + chunk_s, duration_s)
        ranges.append((t, t1))
        t = t1
    return ranges


def write_safetensors_zst(
    voxels_all: np.ndarray,
    voxels_m4: np.ndarray,
    timestamps: np.ndarray,
    model_metadata: dict[str, Any],
    out_path: Path,
    *,
    compress_level: int = 3,
) -> Path:
    """
    Save uncompressed .safetensors next to final path, compress, delete intermediate.
    out_path should end with .safetensors.zst
    """
    out_path = Path(out_path)
    if not str(out_path).endswith(".safetensors.zst"):
        msg = "out_path must end with .safetensors.zst"
        raise ValueError(msg)
    tmp_safe = Path(str(out_path).removesuffix(".zst"))

    meta_json = json.dumps(model_metadata, sort_keys=True)
    tensors: dict[str, np.ndarray] = {
        KEY_VOXELS_ALL: np.asarray(voxels_all, dtype=np.float16),
        KEY_VOXELS_M4: np.asarray(voxels_m4, dtype=np.float16),
        KEY_TIMESTAMPS: np.asarray(timestamps, dtype=np.float64),
        KEY_MODEL_METADATA: np.frombuffer(meta_json.encode("utf-8"), dtype=np.uint8),
    }
    save_file(tensors, str(tmp_safe))
    raw = tmp_safe.read_bytes()
    cctx = zstandard.ZstdCompressor(level=compress_level)
    out_path.write_bytes(cctx.compress(raw))
    tmp_safe.unlink(missing_ok=True)
    return out_path


def write_transcript_sidecar(transcript: dict[str, Any], clip_id: str, out_dir: Path) -> Path:
    """Write ``{clip_id}_transcript.json`` next to artifacts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{clip_id}_transcript.json"
    p.write_text(json.dumps(transcript, indent=2, sort_keys=True), encoding="utf-8")
    return p


def write_dual_pass_safetensors_zst(
    roi_series: dict[str, np.ndarray],
    timestamps: np.ndarray,
    model_metadata: dict[str, Any],
    out_path: Path,
    *,
    compress_level: int = 3,
) -> Path:
    """ROI-only safetensors (no ``voxels_all`` / ``voxels_marketing4``)."""
    out_path = Path(out_path)
    if not str(out_path).endswith(".safetensors.zst"):
        msg = "out_path must end with .safetensors.zst"
        raise ValueError(msg)
    tmp_safe = Path(str(out_path).removesuffix(".zst"))
    meta_json = json.dumps(model_metadata, sort_keys=True)
    tensors: dict[str, np.ndarray] = {}
    for k in DUAL_PASS_ROI_KEYS:
        if k not in roi_series:
            msg = f"dual-pass artifact missing ROI key {k!r}"
            raise ValueError(msg)
        tensors[k] = np.asarray(roi_series[k], dtype=np.float16).ravel()
    tensors[KEY_TIMESTAMPS] = np.asarray(timestamps, dtype=np.float64).ravel()
    tensors[KEY_MODEL_METADATA] = np.frombuffer(meta_json.encode("utf-8"), dtype=np.uint8)
    save_file(tensors, str(tmp_safe))
    raw = tmp_safe.read_bytes()
    cctx = zstandard.ZstdCompressor(level=compress_level)
    out_path.write_bytes(cctx.compress(raw))
    tmp_safe.unlink(missing_ok=True)
    return out_path


def write_dual_pass_artifact(
    *,
    clip_id: str,
    run_uuid: str,
    out_root: Path,
    roi_series: dict[str, np.ndarray],
    timestamps: np.ndarray,
    model_metadata: dict[str, Any],
    duration_s: float,
    transcript: dict[str, Any] | None = None,
    chunk_seconds: float = 300.0,
) -> list[Path]:
    """Write dual-pass ROI tensors + optional transcript JSON sidecar."""
    out_root = Path(out_root)
    ranges = _chunk_ranges(duration_s, chunk_seconds)
    paths: list[Path] = []
    subdir = out_root / clip_id / run_uuid
    subdir.mkdir(parents=True, exist_ok=True)
    if transcript is not None:
        write_transcript_sidecar(transcript, clip_id, subdir)

    if len(ranges) == 1:
        t0, t1 = ranges[0]
        m = (timestamps >= t0) & (timestamps < t1)
        roi_sub = {k: np.asarray(v)[m] for k, v in roi_series.items()}
        p = subdir / f"{clip_id}_voxels.safetensors.zst"
        paths.append(
            write_dual_pass_safetensors_zst(
                roi_sub,
                timestamps[m],
                model_metadata,
                p,
            )
        )
        return paths

    for i, (t0, t1) in enumerate(ranges):
        m = (timestamps >= t0) & (timestamps < t1)
        if not m.any():
            continue
        meta = dict(model_metadata)
        meta["chunk_index"] = i
        meta["chunk_t0_s"] = t0
        meta["chunk_t1_s"] = t1
        roi_sub = {k: np.asarray(v)[m] for k, v in roi_series.items()}
        p = subdir / f"{clip_id}_voxels_part{i:02d}.safetensors.zst"
        paths.append(
            write_dual_pass_safetensors_zst(
                roi_sub,
                timestamps[m],
                meta,
                p,
            )
        )
    return paths


def write_artifact(
    *,
    clip_id: str,
    run_uuid: str,
    out_root: Path,
    voxels_all: np.ndarray,
    voxels_m4: np.ndarray,
    timestamps: np.ndarray,
    model_metadata: dict[str, Any],
    duration_s: float,
    chunk_seconds: float = 300.0,
) -> list[Path]:
    """
    One file per clip for short-form; chunked paths for long-form (> chunk_seconds).
    Path: artifacts/{clip_id}/{run_uuid}/{clip_id}_voxels.safetensors.zst (or _partNN).
    """
    out_root = Path(out_root)
    ranges = _chunk_ranges(duration_s, chunk_seconds)
    paths: list[Path] = []
    if len(ranges) == 1:
        t0, t1 = ranges[0]
        m = (timestamps >= t0) & (timestamps < t1)
        subdir = out_root / clip_id / run_uuid
        subdir.mkdir(parents=True, exist_ok=True)
        p = subdir / f"{clip_id}_voxels.safetensors.zst"
        paths.append(
            write_safetensors_zst(
                voxels_all[m],
                voxels_m4[m],
                timestamps[m],
                model_metadata,
                p,
            )
        )
        return paths

    for i, (t0, t1) in enumerate(ranges):
        m = (timestamps >= t0) & (timestamps < t1)
        if not m.any():
            continue
        subdir = out_root / clip_id / run_uuid
        subdir.mkdir(parents=True, exist_ok=True)
        meta = dict(model_metadata)
        meta["chunk_index"] = i
        meta["chunk_t0_s"] = t0
        meta["chunk_t1_s"] = t1
        p = subdir / f"{clip_id}_voxels_part{i:02d}.safetensors.zst"
        paths.append(
            write_safetensors_zst(
                voxels_all[m],
                voxels_m4[m],
                timestamps[m],
                meta,
                p,
            )
        )
    return paths
