"""Destrieux Cortical Marketing Four vertex maps."""

from __future__ import annotations

import numpy as np
import pytest

from neuroclaw.atlas import cortical_marketing_four as cm4


def test_placeholder_roi_map_disjoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEUROCLAW_PLACEHOLDER_ATLAS", "1")
    m, meta = cm4.build_cortical_marketing_four_roi_vertices()
    assert meta.get("mode") == "placeholder"
    for k in ("FFA", "vmPFC", "IFG", "Insula"):
        assert k in m
        assert m[k].dtype == np.int64
        assert m[k].min() >= 0
        assert m[k].max() < 20484
    # disjoint ranges in placeholder
    all_idx = np.concatenate([m[k] for k in m])
    assert all_idx.size == np.unique(all_idx).size


def test_missing_label_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEUROCLAW_PLACEHOLDER_ATLAS", raising=False)
    bad = dict(cm4.DESTRIEUX_ROI_LABELS_V0)
    bad["FFA"] = ("__nonexistent_destrieux_label__",)
    monkeypatch.setattr(cm4, "DESTRIEUX_ROI_LABELS_V0", bad)
    with pytest.raises(ValueError, match="Destrieux atlas missing"):
        cm4.build_cortical_marketing_four_roi_vertices()
