"""
TRIBE v2 inference wrapper (Giant V-JEPA 2.1 weights via `facebook/tribev2`).

Stage 1 entry point for programmatic use; CLI lives in `neuroclaw.cli`.
"""

from __future__ import annotations

from neuroclaw.atlas.cortical_marketing_four import build_cortical_marketing_four_roi_vertices
from neuroclaw.model.single_pass import CorticalFourResult, run_cortical_marketing_four
from neuroclaw.model.tribe_wrapper import (
    CORTICAL_DIM,
    TRIBEV2_PINNED_COMMIT,
    load_tribe,
    normalize_prediction_to_ot,
    predict_native_ot,
    use_mock,
)

__all__ = [
    "CORTICAL_DIM",
    "TRIBEV2_PINNED_COMMIT",
    "CorticalFourResult",
    "build_cortical_marketing_four_roi_vertices",
    "load_tribe",
    "normalize_prediction_to_ot",
    "predict_native_ot",
    "run_cortical_marketing_four",
    "use_mock",
]
