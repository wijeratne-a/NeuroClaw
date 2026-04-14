"""Marketing Four assembly with cortical-only gating."""

from __future__ import annotations

import numpy as np

from neuroclaw.atlas.marketing_four import SUBCORTICAL_ROIS, assemble_marketing4
from neuroclaw.model.tribe_wrapper import CORTICAL_DIM, VOXEL_DIM


def _idx_map() -> dict[str, np.ndarray]:
    s0 = CORTICAL_DIM
    return {
        "NAcc_L": np.arange(s0, s0 + 4, dtype=np.int64),
        "NAcc_R": np.arange(s0 + 4, s0 + 8, dtype=np.int64),
        "Amy_L": np.arange(s0 + 8, s0 + 12, dtype=np.int64),
        "Amy_R": np.arange(s0 + 12, s0 + 16, dtype=np.int64),
        "vmPFC": np.arange(100, 104, dtype=np.int64),
        "mOFC": np.arange(200, 204, dtype=np.int64),
        "FFA_R": np.arange(300, 304, dtype=np.int64),
    }


def test_assemble_marketing4_cortical_only_zeros_subcortical_slots():
    t = 3
    rng = np.random.default_rng(0)
    vox = rng.standard_normal((t, VOXEL_DIM)).astype(np.float32)
    idx = _idx_map()
    m4, region_map, ffa = assemble_marketing4(vox, idx, cortical_only=True)
    assert ffa["cortical_only"] is True
    assert set(ffa["invalid_subcortical_rois"]) == SUBCORTICAL_ROIS
    # Subcortical ROIs are first four in ORDER
    for name in ("NAcc_L", "NAcc_R", "Amy_L", "Amy_R"):
        sl = region_map[name]
        assert np.allclose(m4[:, sl[0] : sl[1] + 1], 0.0)
    # Cortical ROIs still from voxels
    sl_v = region_map["vmPFC"]
    assert not np.allclose(m4[:, sl_v[0] : sl_v[1] + 1], 0.0)


def test_assemble_marketing4_whole_brain_extracts_subcortical():
    t = 2
    vox = np.zeros((t, VOXEL_DIM), dtype=np.float32)
    s0 = CORTICAL_DIM
    vox[:, s0 : s0 + 8] = 3.14
    idx = _idx_map()
    m4, _region_map, ffa = assemble_marketing4(vox, idx, cortical_only=False)
    assert ffa["cortical_only"] is False
    assert ffa["invalid_subcortical_rois"] == []
    sl = _region_map["NAcc_L"]
    assert np.allclose(m4[:, sl[0] : sl[1] + 1], 3.14)
