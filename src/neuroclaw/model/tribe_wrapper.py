"""TRIBE v2 TribeModel — load, predict, native 29,286-dim output."""

from __future__ import annotations

import logging
import os
import inspect
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

logger = logging.getLogger("neuroclaw.model.tribe")

CORTICAL_DIM = 20484
SUBCORTICAL_DIM = 8802
VOXEL_DIM = CORTICAL_DIM + SUBCORTICAL_DIM  # 29286

DUAL_PASS_SUBCORTICAL_DIM = SUBCORTICAL_DIM  # 8802
DUAL_PASS_CORTICAL_DIM = CORTICAL_DIM  # 20484

TRIBEV2_PINNED_COMMIT = "72399081ed3f1040c4d996cefb2864a4c46f5b8e"

# From tribev2/grids/run_subcortical.py — forces native 8802-dim subcortical output.
SUBCORTICAL_CONFIG_UPDATE: dict[str, Any] = {
    "data.neuro": {
        "projection": {
            "name": "MaskProjector",
            "mask": "subcortical",
            "=replace=": True,
        }
    }
}

# Default fsaverage5 cortical surface (no override).
CORTICAL_CONFIG_UPDATE: dict[str, Any] = {}


class VoxelResult(NamedTuple):
    """TRIBE predict output; subcortical block may be zero-padded."""

    voxels: np.ndarray  # (T, VOXEL_DIM) float16
    cortical_only: bool  # True if model returned CORTICAL_DIM and subcortical was padded with zeros


class DualPassResult(NamedTuple):
    """Dual-pass ROI time series (bilateral NAcc pooled)."""

    nacc: np.ndarray  # (T,) float16
    amygdala: np.ndarray
    ffa: np.ndarray
    vmpfc: np.ndarray
    timestamps: np.ndarray  # (T,) float64


_TRIBE_AVAILABLE = False
TribeModel: Any = None

try:
    from tribev2 import TribeModel as _TribeModel  # type: ignore[import-not-found]

    TribeModel = _TribeModel
    _TRIBE_AVAILABLE = True
except ImportError:
    _TRIBE_AVAILABLE = False


def use_mock() -> bool:
    return os.environ.get("NEUROCLAW_USE_MOCK_TRIBE", "").lower() in ("1", "true", "yes") or not _TRIBE_AVAILABLE


def neuralset_feature_device(resolved_tribe_device: str) -> str:
    """
    Device string for neuralset HuggingFace extractors (text/audio/image).

    The published ``facebook/tribev2`` config pins ``device: cuda`` on those
    modules; that overrides the brain checkpoint's ``from_pretrained(device=)``
    and breaks CPU-only or MPS-only PyTorch builds. Allowed values are only
    ``cpu`` or ``cuda`` (neuralset's HuggingFaceMixin has no ``mps``).
    """
    import torch

    r = (resolved_tribe_device or "cpu").strip().lower()
    if r == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def tribev2_extractor_config_update(feature_device: str) -> dict[str, str]:
    """Keys to merge into TRIBE ConfDict so HF towers match *feature_device*."""
    return {
        "data.text_feature.device": feature_device,
        "data.audio_feature.device": feature_device,
        "data.image_feature.image.device": feature_device,
        "data.video_feature.image.device": feature_device,
    }


def merge_tribe_config_update(
    feature_device: str,
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge HF tower device keys with optional neuro/projection overrides."""
    base: dict[str, Any] = dict(tribev2_extractor_config_update(feature_device))
    if extra:
        base.update(extra)
    return base


def resolve_tribe_device(requested: str = "auto") -> str:
    """
    Map NEUROCLAW_DEVICE / CLI preference to a concrete torch device string.

    tribev2's ``device="auto"`` can pick CUDA even when this PyTorch build has no
    CUDA support, which raises at model.to(cuda). We only select cuda/mps when
    the runtime reports them as available.
    """
    import torch

    r = (requested or "auto").strip().lower()
    if r in ("", "auto"):
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    if r == "cuda":
        if not torch.cuda.is_available():
            logger.warning(
                "tribe_device_cuda_unavailable",
                extra={"fallback": "cpu"},
            )
            return "cpu"
        return "cuda"
    if r == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            logger.warning(
                "tribe_device_mps_unavailable",
                extra={"fallback": "cpu"},
            )
            return "cpu"
        return "mps"
    if r == "cpu":
        return "cpu"
    logger.warning("tribe_device_unknown", extra={"requested": requested, "fallback": "cpu"})
    return "cpu"


class _MockTribeModel:
    """Deterministic z-scored mock for CI / dev without tribev2 wheels."""

    mask_token = 0

    @classmethod
    def from_pretrained(cls, *_a: Any, **_k: Any) -> _MockTribeModel:
        return cls()

    def get_events_dataframe(self, video_path: str) -> pd.DataFrame:
        """Minimal multimodal rows for pipeline tests."""
        return pd.DataFrame(
            {
                "type": ["Video", "Audio", "Text"],
                "start": [0.0, 0.0, 0.0],
                "duration": [120.0, 120.0, 1.0],
                "text": ["", "", "mock"],
                "context": ["", "", ""],
                "filepath": [video_path, video_path, ""],
            }
        )

    def predict(self, events: pd.DataFrame | None = None, **kwargs: Any) -> tuple[Any, Any]:
        if len(events) > 0:
            dur = float(events["start"].max() + events["duration"].max())
        else:
            dur = 10.0
        n = max(1, int(np.ceil(dur)))
        rng = np.random.default_rng(42)
        mo = getattr(self, "mock_native_o", None)
        if mo == DUAL_PASS_SUBCORTICAL_DIM:
            raw = rng.standard_normal((n, DUAL_PASS_SUBCORTICAL_DIM))
            preds = (raw / (raw.std(axis=1, keepdims=True) + 1e-8)).astype(np.float16)
            return preds, []
        if mo == DUAL_PASS_CORTICAL_DIM:
            raw = rng.standard_normal((n, DUAL_PASS_CORTICAL_DIM))
            preds = (raw / (raw.std(axis=1, keepdims=True) + 1e-8)).astype(np.float16)
            return preds, []
        cort = rng.standard_normal((n, CORTICAL_DIM))
        sub = rng.standard_normal((n, SUBCORTICAL_DIM))
        full = np.concatenate([cort, sub], axis=1)
        full = (full - full.mean(axis=1, keepdims=True)) / (full.std(axis=1, keepdims=True) + 1e-8)
        preds = full.astype(np.float16)
        segments: list[Any] = []
        return preds, segments


def _extract_prediction_array(preds: Any) -> np.ndarray:
    """Strip xarray wrapper if present."""
    if hasattr(preds, "values"):
        preds = preds.values
    return np.asarray(preds)


def _coerce_time_major_voxels(arr: np.ndarray) -> np.ndarray:
    """
    Ensure shape (T, O) for legacy VoxelResult path.
    Tribev2 may return (T, O) or (O, T); normalize using known output sizes.
    """
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        msg = f"Expected 2D prediction array, got shape {arr.shape}"
        raise RuntimeError(msg)
    o_dims = (
        CORTICAL_DIM,
        VOXEL_DIM,
        SUBCORTICAL_DIM,
        DUAL_PASS_SUBCORTICAL_DIM,
        DUAL_PASS_CORTICAL_DIM,
    )
    if arr.shape[-1] in o_dims:
        return arr
    if arr.shape[0] in o_dims:
        return arr.T
    # Heuristic: feature dimension is usually larger than timestep count for brain outputs
    if arr.shape[0] > arr.shape[1]:
        return arr.T
    return arr


def load_tribe(
    device: str = "auto",
    hf_token: str | None = None,
    *,
    config_override: dict[str, Any] | None = None,
    mock_native_o: int | None = None,
) -> Any:
    """Load TRIBE or mock."""
    resolved = resolve_tribe_device(device)
    if (device or "auto").strip().lower() in ("auto", ""):
        logger.info("tribe_device_resolved", extra={"resolved": resolved})
    if use_mock():
        logger.info("using_mock_tribe_model")
        m = _MockTribeModel.from_pretrained("facebook/tribev2", device=resolved)
        if mock_native_o is not None:
            m.mock_native_o = mock_native_o  # type: ignore[attr-defined]
        return m
    assert TribeModel is not None
    feat_dev = neuralset_feature_device(resolved)
    merged_cfg = merge_tribe_config_update(feat_dev, config_override)
    kwargs: dict[str, Any] = {
        "device": resolved,
        "config_update": merged_cfg,
    }
    logger.info(
        "tribe_extractor_devices",
        extra={"neuralset_hf_device": feat_dev, "brain_device": resolved},
    )
    if hf_token:
        # tribev2 API differs across versions; map token only when supported.
        sig = inspect.signature(TribeModel.from_pretrained)
        params = set(sig.parameters.keys())
        if "token" in params:
            kwargs["token"] = hf_token
        elif "hf_token" in params:
            kwargs["hf_token"] = hf_token
        elif "use_auth_token" in params:
            kwargs["use_auth_token"] = hf_token
    try:
        return TribeModel.from_pretrained("facebook/tribev2", **kwargs)
    except TypeError as e:
        logger.warning("tribe_from_pretrained_signature_mismatch", extra={"error": str(e)})
        cu = merge_tribe_config_update(feat_dev, config_override)
        try:
            return TribeModel.from_pretrained("facebook/tribev2", device=resolved, config_update=cu)
        except TypeError:
            pass
        try:
            return TribeModel.from_pretrained("facebook/tribev2", config_update=cu)
        except TypeError:
            pass
        return TribeModel.from_pretrained("facebook/tribev2")


def normalize_prediction_to_ot(preds: Any, expected_o: int) -> np.ndarray:
    """
    Return float16 array shaped (O, T) where O == expected_o (voxels, time).
    Handles xarray, batch dimension, and (T,O) vs (O,T) orientation.
    """
    arr = _extract_prediction_array(preds)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        msg = f"Expected 2D native prediction, got shape {arr.shape}"
        raise RuntimeError(msg)
    if arr.shape[0] == expected_o:
        out = arr
    elif arr.shape[1] == expected_o:
        out = arr.T
    else:
        msg = f"Expected one axis == {expected_o}, got shape {arr.shape}"
        raise RuntimeError(msg)
    if out.shape[0] != expected_o:
        msg = f"native output dim mismatch: want O={expected_o}, got {out.shape}"
        raise RuntimeError(msg)
    return out.astype(np.float16)


def _trim_pad_time_ot(ot: np.ndarray, n_expect: int) -> np.ndarray:
    """Trim or pad along time axis (axis 1) of (O, T)."""
    o, t = ot.shape
    if t < n_expect:
        pad = np.repeat(ot[:, -1:], n_expect - t, axis=1)
        return np.concatenate([ot, pad], axis=1)
    if t > n_expect:
        return ot[:, :n_expect]
    return ot


def predict_native_ot(
    model: Any,
    events_df: pd.DataFrame,
    clip_duration_s: float,
    expected_o: int,
) -> np.ndarray:
    """
    Run model.predict; return (O, T) float16 with O == expected_o (8802 or 20484).
    """
    preds, _segments = model.predict(events=events_df)
    ot = normalize_prediction_to_ot(preds, expected_o)
    n_expect = max(1, int(np.ceil(clip_duration_s)))
    return _trim_pad_time_ot(ot, n_expect)


def predict_voxels(
    model: Any,
    events_df: pd.DataFrame,
    clip_duration_s: float,
) -> VoxelResult:
    """
    Run model.predict(events=df); return VoxelResult with (n_bins, 29286) float16 voxels.

    Sliding window / padding is handled inside tribev2; mock uses row count heuristic.
    cortical_only is True when the checkpoint returned only cortical vertices and
    the subcortical block was zero-padded to VOXEL_DIM.
    """
    preds, _segments = model.predict(events=events_df)
    arr = _coerce_time_major_voxels(_extract_prediction_array(preds))
    last = arr.shape[-1]
    cortical_only = False
    if last == CORTICAL_DIM:
        cortical_only = True
        # TRIBE v2 can return cortical vertices only; native NeuroClaw layout is
        # cortical (20484) then subcortical (8802).
        logger.warning(
            "tribe_predict_cortical_only",
            extra={
                "shape": arr.shape,
                "pad_subcortical_zeros": SUBCORTICAL_DIM,
            },
        )
        sub = np.zeros((arr.shape[0], SUBCORTICAL_DIM), dtype=arr.dtype)
        arr = np.concatenate([arr, sub], axis=1)
    elif last != VOXEL_DIM:
        msg = f"Expected last dim {VOXEL_DIM} (or cortical-only {CORTICAL_DIM}), got {arr.shape}"
        raise RuntimeError(msg)
    n_expect = max(1, int(np.ceil(clip_duration_s)))
    if arr.shape[0] < n_expect:
        pad = np.repeat(arr[-1:], n_expect - arr.shape[0], axis=0)
        arr = np.concatenate([arr, pad], axis=0)
    elif arr.shape[0] > n_expect:
        arr = arr[:n_expect]
    out = arr.astype(np.float16)
    return VoxelResult(voxels=out, cortical_only=cortical_only)
