"""predict_voxels handles TRIBE cortical-only output."""

from __future__ import annotations

import numpy as np
import pandas as pd

from neuroclaw.model.tribe_wrapper import (
    CORTICAL_DIM,
    SUBCORTICAL_DIM,
    VOXEL_DIM,
    VoxelResult,
    predict_voxels,
)


class _CorticalOnlyModel:
    def predict(self, events: pd.DataFrame | None = None, **kwargs):
        n = 3
        preds = np.ones((n, CORTICAL_DIM), dtype=np.float32)
        return preds, []


def test_predict_voxels_pads_cortical_only_to_voxel_dim():
    m = _CorticalOnlyModel()
    df = pd.DataFrame({"type": ["Video"], "start": [0.0], "duration": [5.0]})
    out = predict_voxels(m, df, clip_duration_s=5.0)
    assert isinstance(out, VoxelResult)
    assert out.cortical_only is True
    assert out.voxels.shape == (5, VOXEL_DIM)
    assert out.voxels.dtype == np.float16
    assert np.allclose(out.voxels[:, :CORTICAL_DIM], 1.0)
    assert np.all(out.voxels[:, CORTICAL_DIM:] == 0)


class _WholeBrainModel:
    def predict(self, events: pd.DataFrame | None = None, **kwargs):
        n = 2
        cort = np.ones((n, CORTICAL_DIM), dtype=np.float32)
        sub = np.ones((n, SUBCORTICAL_DIM), dtype=np.float32) * 2.0
        full = np.concatenate([cort, sub], axis=1)
        return full, []


def test_predict_voxels_whole_brain_sets_cortical_only_false():
    m = _WholeBrainModel()
    df = pd.DataFrame({"type": ["Video"], "start": [0.0], "duration": [2.0]})
    out = predict_voxels(m, df, clip_duration_s=2.0)
    assert out.cortical_only is False
    assert out.voxels.shape == (2, VOXEL_DIM)
    assert np.all(out.voxels[:, CORTICAL_DIM:] > 0)
