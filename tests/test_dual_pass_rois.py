"""Dual-pass ROI map construction (placeholder / offset logic)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from neuroclaw.atlas import dual_pass_rois as dpr
from neuroclaw.model.tribe_wrapper import DUAL_PASS_SUBCORTICAL_DIM


def test_placeholder_roi_map_has_all_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEUROCLAW_PLACEHOLDER_ATLAS", "1")
    m = dpr.build_dual_pass_roi_map()
    for k in dpr.DUAL_PASS_ROI_MAP_KEYS:
        assert k in m
        assert m[k].dtype == np.int64
        assert m[k].size > 0


def test_subcortical_indices_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEUROCLAW_PLACEHOLDER_ATLAS", raising=False)
    if os.environ.get("SKIP_TRIBEV2_ROI", ""):
        pytest.skip("tribev2 not required for this unit test")
    try:
        idx = dpr.get_subcortical_indices_relative(dpr.NACC_QUERY)
    except Exception:
        pytest.skip("tribev2.subcortical not available")
    assert idx.max() < DUAL_PASS_SUBCORTICAL_DIM
    assert idx.min() >= 0
