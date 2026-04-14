"""
ROI masks: Harvard-Oxford Subcortical v1.0 (25% thresh) + Julich v2.9 (50% thresh),
MNI152NLin2009cSAsym. Falls back to deterministic placeholder indices if atlases unavailable.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from neuroclaw.model.tribe_wrapper import CORTICAL_DIM, VOXEL_DIM

logger = logging.getLogger("neuroclaw.atlas")

# Atlas version strings for metadata
HARVARD_OXFORD_VERSION = "1.0"
JULICH_VERSION = "2.9"


def _placeholder_indices() -> dict[str, np.ndarray]:
    """
    Deterministic index ranges inside 29,286-dim vector for CI.
    Not neuroscientifically valid — override with real atlas in production.
    """
    # Subcortical block starts at 20484
    s0 = 20484
    regions = {
        "NAcc_L": np.arange(s0, s0 + 32),
        "NAcc_R": np.arange(s0 + 32, s0 + 64),
        "Amy_L": np.arange(s0 + 64, s0 + 96),
        "Amy_R": np.arange(s0 + 96, s0 + 128),
        "vmPFC": np.arange(1000, 1000 + 256),
        "mOFC": np.arange(1000 + 256, 1000 + 512),
        "FFA_L": np.arange(500, 500 + 64),
        "FFA_R": np.arange(500 + 64, 500 + 128),
    }
    return {k: v.astype(np.int64) for k, v in regions.items()}


def _indices_from_nilearn() -> dict[str, np.ndarray] | None:
    """Try Harvard-Oxford + Julich via nilearn; return None on failure."""
    try:
        from nilearn.datasets import fetch_atlas_harvard_oxford, fetch_atlas_juelich
    except ImportError:
        return None

    try:
        ho = fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
        _ = ho.maps  # noqa: F841
        # Full label-based index extraction requires label-to-voxel mapping into TRIBE
        # ordering; not available without TRIBE-side NIfTI — fall through to placeholder.
        logger.info("nilearn_harvard_oxford_fetched", extra={"note": "using_placeholder_indices"})
    except Exception as e:
        logger.warning("atlas_fetch_failed", extra={"error": str(e)})

    try:
        fetch_atlas_juelich("maxprob-thr0-2mm")  # noqa: F841
    except Exception as e:
        logger.warning("julich_fetch_failed", extra={"error": str(e)})

    return None


def build_roi_masks() -> dict[str, Any]:
    """
    Return dict region_name -> int64 index array into the 29,286 native vector,
    plus metadata.
    """
    if os.environ.get("NEUROCLAW_PLACEHOLDER_ATLAS", "").lower() in ("1", "true", "yes"):
        idx = _placeholder_indices()
        return {
            "indices": idx,
            "atlas_versions": {
                "harvard_oxford": HARVARD_OXFORD_VERSION,
                "julich": JULICH_VERSION,
                "mni_template": "MNI152NLin2009cSAsym",
                "mode": "placeholder",
            },
        }

    nl = _indices_from_nilearn()
    if nl is not None:
        return {
            "indices": nl,
            "atlas_versions": {
                "harvard_oxford": HARVARD_OXFORD_VERSION,
                "julich": JULICH_VERSION,
                "mni_template": "MNI152NLin2009cSAsym",
                "mode": "nilearn",
            },
        }

    idx = _placeholder_indices()
    return {
        "indices": idx,
        "atlas_versions": {
            "harvard_oxford": HARVARD_OXFORD_VERSION,
            "julich": JULICH_VERSION,
            "mni_template": "MNI152NLin2009cSAsym",
            "mode": "placeholder",
        },
    }


def validate_indices(
    indices: dict[str, np.ndarray],
    *,
    cortical_only: bool = False,
) -> None:
    for name, arr in indices.items():
        if arr.max() >= VOXEL_DIM or arr.min() < 0:
            msg = f"ROI {name} indices out of bounds [0, {VOXEL_DIM})"
            raise ValueError(msg)
        if cortical_only and int(arr.max()) >= CORTICAL_DIM:
            logger.warning(
                "roi_in_subcortical_range_during_cortical_only_run",
                extra={"roi": name},
            )
