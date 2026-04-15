"""Destrieux fsaverage5 vertex indices for the Cortical Marketing Four (20484 surface)."""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger("neuroclaw.atlas")

# Nilearn Destrieux label strings (fetch_atlas_surf_destrieux('fsaverage5').labels).
DESTRIEUX_ROI_LABELS_V0: dict[str, tuple[str, ...]] = {
    "FFA": ("G_oc-temp_lat-fusifor",),
    "vmPFC": ("G_rectus", "G_subcallosal"),
    "IFG": (
        "G_front_inf-Opercular",
        "G_front_inf-Triangul",
        "G_front_inf-Orbital",
    ),
    "Insula": ("G_insular_short", "G_Ins_lg_and_S_cent_ins"),
}

CORTICAL_VERTICES_PER_HEM = 10242


def _normalize_label_list(raw: Any) -> list[str]:
    out: list[str] = []
    for x in raw:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def _placeholder_roi_vertices() -> dict[str, np.ndarray]:
    """Deterministic disjoint index ranges for CI (NEUROCLAW_PLACEHOLDER_ATLAS)."""
    return {
        "FFA": np.arange(0, 48, dtype=np.int64),
        "vmPFC": np.arange(200, 200 + 64, dtype=np.int64),
        "IFG": np.arange(2000, 2000 + 96, dtype=np.int64),
        "Insula": np.arange(5000, 5000 + 80, dtype=np.int64),
    }


def _vertices_for_label_index(
    map_left: np.ndarray,
    map_right: np.ndarray,
    label_idx: int,
) -> np.ndarray:
    """Combine lh+rh vertex indices into 0..20483 (lh || rh+10242)."""
    li = np.where(map_left == label_idx)[0].astype(np.int64)
    ri = np.where(map_right == label_idx)[0].astype(np.int64) + CORTICAL_VERTICES_PER_HEM
    return np.concatenate([li, ri])


def build_cortical_marketing_four_roi_vertices() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """
    Map each logical ROI to native fsaverage5 vertex indices (0..20483).

    Returns:
        indices: region name -> int64 array of vertex indices
        atlas_versions: metadata for model_metadata
    """
    if os.environ.get("NEUROCLAW_PLACEHOLDER_ATLAS", "").lower() in ("1", "true", "yes"):
        return _placeholder_roi_vertices(), {
            "cortical": "destrieux-fsaverage5",
            "mode": "placeholder",
        }

    try:
        from nilearn.datasets import fetch_atlas_surf_destrieux
    except ImportError as e:
        raise RuntimeError(
            "nilearn is required for Destrieux ROI mapping. "
            "Install nilearn or set NEUROCLAW_PLACEHOLDER_ATLAS=1 for tests."
        ) from e

    atlas = fetch_atlas_surf_destrieux("fsaverage5")
    labels = _normalize_label_list(atlas.labels)
    label_set = set(labels)
    map_left = np.asarray(atlas.map_left)
    map_right = np.asarray(atlas.map_right)

    out: dict[str, np.ndarray] = {}
    for roi_name, wanted in DESTRIEUX_ROI_LABELS_V0.items():
        missing = [s for s in wanted if s not in label_set]
        if missing:
            msg = (
                f"Destrieux atlas missing label(s) for ROI {roi_name!r}: {missing}. "
                f"Update DESTRIEUX_ROI_LABELS_V0 in cortical_marketing_four.py. "
                f"Available labels ({len(labels)}): {sorted(label_set)}"
            )
            raise ValueError(msg)
        parts: list[np.ndarray] = []
        for s in wanted:
            li = labels.index(s)
            parts.append(_vertices_for_label_index(map_left, map_right, li))
        combined = np.unique(np.concatenate(parts)) if parts else np.array([], dtype=np.int64)
        out[roi_name] = combined.astype(np.int64)
        logger.info(
            "cortical_roi_vertices",
            extra={"roi": roi_name, "n_vertices": int(combined.shape[0])},
        )

    meta = {
        "cortical": "destrieux-fsaverage5",
        "mode": "nilearn",
        "nilearn_atlas": getattr(atlas, "atlas_type", "destrieux_surface"),
    }
    return out, meta
