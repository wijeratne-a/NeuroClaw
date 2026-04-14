"""
TRIBE v2 inference wrapper (Giant V-JEPA 2.1 weights via `facebook/tribev2`).

Stage 1 entry point for programmatic use; CLI lives in `neuroclaw.cli`.
"""

from __future__ import annotations

from neuroclaw.model.tribe_wrapper import (
    CORTICAL_DIM,
    SUBCORTICAL_CONFIG_UPDATE,
    SUBCORTICAL_DIM,
    TRIBEV2_PINNED_COMMIT,
    VOXEL_DIM,
    DualPassResult,
    VoxelResult,
    load_tribe,
    normalize_prediction_to_ot,
    predict_native_ot,
    predict_voxels,
    use_mock,
)

__all__ = [
    "CORTICAL_DIM",
    "SUBCORTICAL_CONFIG_UPDATE",
    "SUBCORTICAL_DIM",
    "TRIBEV2_PINNED_COMMIT",
    "VOXEL_DIM",
    "DualPassResult",
    "VoxelResult",
    "load_tribe",
    "normalize_prediction_to_ot",
    "predict_native_ot",
    "predict_voxels",
    "use_mock",
]
