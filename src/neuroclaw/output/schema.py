"""Safetensors key contract and metadata keys."""

from __future__ import annotations

from typing import Any

DTYPE = "float16"

KEY_TIMESTAMPS = "timestamps"
KEY_MODEL_METADATA = "model_metadata"

# Cortical Marketing Four: ROI series (1-D float16 each, length T)
KEY_ROI_FFA = "FFA"
KEY_ROI_VMPFC = "vmPFC"
KEY_ROI_IFG = "IFG"
KEY_ROI_INSULA = "Insula"

CORTICAL_FOUR_ROI_KEYS = frozenset(
    {
        KEY_ROI_FFA,
        KEY_ROI_VMPFC,
        KEY_ROI_IFG,
        KEY_ROI_INSULA,
    }
)

INFERENCE_LAYOUT_CORTICAL_MARKETING_FOUR = "cortical_marketing_four"

CORTICAL_FOUR_EXTRA_METADATA_KEYS = frozenset({"tribev2_context"})

MANDATORY_METADATA_KEYS = frozenset(
    {
        "model_id",
        "tribev2_version",
        "created_utc",
        "run_uuid",
        "git_sha",
        "device",
        "dtype",
        "hemodynamic_offset_s",
        "roi_ordering",
        "inference_layout",
        "atlas_versions",
        "drift_stats",
        "transparency_label",
        "use_case_audit",
        "region_map",
    }
)


def validate_metadata(meta: dict[str, Any]) -> None:
    missing = MANDATORY_METADATA_KEYS - set(meta.keys())
    if missing:
        msg = f"model_metadata missing keys: {sorted(missing)}"
        raise ValueError(msg)
    if meta.get("inference_layout") == INFERENCE_LAYOUT_CORTICAL_MARKETING_FOUR:
        missing2 = CORTICAL_FOUR_EXTRA_METADATA_KEYS - set(meta.keys())
        if missing2:
            msg = f"cortical_marketing_four metadata missing keys: {sorted(missing2)}"
            raise ValueError(msg)
