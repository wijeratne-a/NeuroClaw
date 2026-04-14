"""Sequential dual-pass TRIBE inference: subcortical 8802 then cortical 20484."""

from __future__ import annotations

import gc
import logging
import sys
from typing import Any

import numpy as np
import pandas as pd

from neuroclaw.atlas.dual_pass_rois import (
    KEY_AMYGDALA,
    KEY_FFA,
    KEY_NACC,
    KEY_VPFC,
)
from neuroclaw.config import Settings
from neuroclaw.model.tribe_wrapper import (
    CORTICAL_CONFIG_UPDATE,
    DualPassResult,
    DUAL_PASS_CORTICAL_DIM,
    DUAL_PASS_SUBCORTICAL_DIM,
    SUBCORTICAL_CONFIG_UPDATE,
    load_tribe,
    predict_native_ot,
)

logger = logging.getLogger("neuroclaw.model.dual_pass")


def _clear_torch_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass


def _pool_roi(ot: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Mean over ROI vertices -> (T,) float16."""
    if idx.size == 0:
        raise ValueError("empty ROI index array")
    part = ot[idx.astype(np.int64), :]
    return part.mean(axis=0).astype(np.float16)


def run_dual_pass(
    events_df: pd.DataFrame,
    duration_s: float,
    bin_starts_s: np.ndarray,
    roi_map: dict[str, np.ndarray],
    settings: Settings,
    run_uuid: str,
    device: str,
    hf_token: str | None,
    *,
    mock_native_o: tuple[int, int] | None = None,
) -> DualPassResult:
    """
    Pass 1: subcortical MaskProjector (8802).
    Pass 2: default cortical fsaverage5 (20484).
    Full model reload between passes.
    """
    _ = settings, run_uuid
    def _load(sub: bool) -> Any:
        if mock_native_o is not None:
            o = mock_native_o[0] if sub else mock_native_o[1]
            return load_tribe(device, hf_token, mock_native_o=o)
        ov = SUBCORTICAL_CONFIG_UPDATE if sub else CORTICAL_CONFIG_UPDATE
        return load_tribe(device, hf_token, config_override=ov)

    model1 = _load(True)
    try:
        ot_sub = predict_native_ot(
            model1,
            events_df,
            duration_s,
            DUAL_PASS_SUBCORTICAL_DIM,
        )
    finally:
        del model1
        gc.collect()
        _clear_torch_cache()

    model2 = _load(False)
    try:
        ot_cor = predict_native_ot(
            model2,
            events_df,
            duration_s,
            DUAL_PASS_CORTICAL_DIM,
        )
    finally:
        del model2
        gc.collect()
        _clear_torch_cache()

    if ot_sub.shape[1] != ot_cor.shape[1]:
        logger.error(
            "dual_pass_time_mismatch",
            extra={"T_sub": ot_sub.shape[1], "T_cort": ot_cor.shape[1]},
        )
        sys.exit(
            "FAIL: temporal dimension mismatch between subcortical and cortical passes "
            f"({ot_sub.shape[1]} vs {ot_cor.shape[1]}). Aborting."
        )

    t_bins = ot_sub.shape[1]
    bs = np.asarray(bin_starts_s, dtype=np.float64).ravel()
    if bs.size >= t_bins:
        ts = bs[:t_bins].copy()
    else:
        pad = np.arange(t_bins - bs.size, dtype=np.float64) + (bs[-1] if bs.size else 0.0) + 1.0
        ts = np.concatenate([bs, pad])

    nacc = _pool_roi(ot_sub, roi_map[KEY_NACC])
    amy = _pool_roi(ot_sub, roi_map[KEY_AMYGDALA])
    ffa = _pool_roi(ot_cor, roi_map[KEY_FFA])
    vmpfc = _pool_roi(ot_cor, roi_map[KEY_VPFC])

    return DualPassResult(
        nacc=nacc,
        amygdala=amy,
        ffa=ffa,
        vmpfc=vmpfc,
        timestamps=ts,
    )
