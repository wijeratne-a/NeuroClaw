"""Safetensors key contract and metadata keys."""

from __future__ import annotations

from typing import Any

VOXEL_DIM = 29286
DTYPE = "float16"

KEY_VOXELS_ALL = "voxels_all"
KEY_VOXELS_M4 = "voxels_marketing4"
KEY_TIMESTAMPS = "timestamps"
KEY_MODEL_METADATA = "model_metadata"

# Dual-pass aggregated ROI series (1-D float16 per key, length T)
KEY_ROI_NACC = "NAcc"
KEY_ROI_AMYGDALA = "Amygdala"
KEY_ROI_FFA = "FFA"
KEY_ROI_VPFC = "vmPFC"

DUAL_PASS_ROI_KEYS = frozenset(
    {
        KEY_ROI_NACC,
        KEY_ROI_AMYGDALA,
        KEY_ROI_FFA,
        KEY_ROI_VPFC,
    }
)

INFERENCE_LAYOUT_DUAL_PASS_AGGREGATED = "dual_pass_aggregated"

DUAL_PASS_EXTRA_METADATA_KEYS = frozenset({"tribev2_context"})

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
        "voxel_ordering",
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
    if meta.get("inference_layout") == INFERENCE_LAYOUT_DUAL_PASS_AGGREGATED:
        missing2 = DUAL_PASS_EXTRA_METADATA_KEYS - set(meta.keys())
        if missing2:
            msg = f"dual_pass_aggregated metadata missing keys: {sorted(missing2)}"
            raise ValueError(msg)
