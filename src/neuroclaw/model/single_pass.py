"""Single-pass TRIBE inference: fsaverage5 cortical 20484 -> Cortical Marketing Four ROI series."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from neuroclaw.model.tribe_wrapper import CORTICAL_DIM, predict_native_ot


class CorticalFourResult(NamedTuple):
    """Four Destrieux-pooled cortical ROI time series (length T)."""

    ffa: np.ndarray  # (T,) float16
    vmpfc: np.ndarray  # (T,) float16
    ifg: np.ndarray  # (T,) float16
    insula: np.ndarray  # (T,) float16
    timestamps: np.ndarray  # (T,) float64


def prepare_model_for_extract(model: Any) -> None:
    """TRIBE: keep all timeline segments so T stays aligned with video bins."""
    if hasattr(model, "remove_empty_segments"):
        model.remove_empty_segments = False


def _align_ot_to_bin_starts(ot: np.ndarray, bin_starts_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    ot: (O, T_model). Align T to bin_starts (trim if T > len(bs); pad last column if T < len(bs)).
    """
    bs = np.asarray(bin_starts_s, dtype=np.float64).ravel()
    t, nbs = ot.shape[1], bs.size
    if t > nbs and nbs > 0:
        ot = ot[:, :nbs]
    elif t < nbs:
        pad = np.repeat(ot[:, -1:], nbs - t, axis=1)
        ot = np.concatenate([ot, pad], axis=1)
    ts = bs[: ot.shape[1]].copy() if nbs > 0 else np.arange(ot.shape[1], dtype=np.float64)
    return ot, ts


def _pool_roi(ot: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Mean over ROI vertices -> (T,) float16."""
    if idx.size == 0:
        raise ValueError("empty ROI index array")
    part = ot[idx.astype(np.int64), :]
    return part.mean(axis=0).astype(np.float16)


def run_cortical_marketing_four(
    model: Any,
    events_df: pd.DataFrame,
    duration_s: float,
    bin_starts_s: np.ndarray,
    roi_vertices: dict[str, np.ndarray],
) -> CorticalFourResult:
    """
    One TRIBE load, native (20484, T), mean-pool four Destrieux ROIs.
    """
    prepare_model_for_extract(model)
    ot = predict_native_ot(model, events_df, duration_s, CORTICAL_DIM)
    ot, ts = _align_ot_to_bin_starts(ot, bin_starts_s)

    ffa = _pool_roi(ot, roi_vertices["FFA"])
    vmpfc = _pool_roi(ot, roi_vertices["vmPFC"])
    ifg = _pool_roi(ot, roi_vertices["IFG"])
    insula = _pool_roi(ot, roi_vertices["Insula"])

    return CorticalFourResult(
        ffa=ffa,
        vmpfc=vmpfc,
        ifg=ifg,
        insula=insula,
        timestamps=ts,
    )
