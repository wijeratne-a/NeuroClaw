"""Assemble Marketing Four dense vector: NAcc_L, NAcc_R, Amy_L, Amy_R, vmPFC, mOFC, FFA_R."""

from __future__ import annotations

from typing import Any

import numpy as np

ORDER = ("NAcc_L", "NAcc_R", "Amy_L", "Amy_R", "vmPFC", "mOFC", "FFA_R")

SUBCORTICAL_ROIS = frozenset({"NAcc_L", "NAcc_R", "Amy_L", "Amy_R"})


def assemble_marketing4(
    voxels_all: np.ndarray,
    indices: dict[str, np.ndarray],
    *,
    cortical_only: bool = False,
) -> tuple[np.ndarray, dict[str, list[int]], dict[str, Any]]:
    """
    voxels_all: (T, 29286)
    Returns:
        marketing4: (T, M) float16
        region_map: name -> [start, end] inclusive slice in marketing4 row
        ffa_meta: bilateral FFA index ranges for metadata
    """
    blocks: list[np.ndarray] = []
    region_map: dict[str, list[int]] = {}
    offset = 0
    for name in ORDER:
        idx = indices[name]
        if cortical_only and name in SUBCORTICAL_ROIS:
            part = np.zeros((voxels_all.shape[0], len(idx)), dtype=np.float32)
        else:
            part = voxels_all[:, idx.astype(np.int64)]
        blocks.append(part)
        m = part.shape[1]
        region_map[name] = [offset, offset + m - 1]
        offset += m
    m4 = np.concatenate(blocks, axis=1).astype(np.float16)

    ffa_meta: dict[str, Any] = {
        "FFA_L_indices": indices.get("FFA_L", np.array([], dtype=np.int64)).tolist(),
        "FFA_R_indices": indices.get("FFA_R", np.array([], dtype=np.int64)).tolist(),
        "pfc_bilateral_note": "vmPFC/mOFC blocks include bilateral vertices per placeholder atlas",
        "cortical_only": cortical_only,
        "invalid_subcortical_rois": sorted(SUBCORTICAL_ROIS) if cortical_only else [],
    }
    return m4, region_map, ffa_meta
