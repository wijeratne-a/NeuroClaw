"""ROI index maps for dual-pass TRIBE inference (subcortical 8802 + cortical 20484)."""

from __future__ import annotations

import logging
import os
import numpy as np

from neuroclaw.model.tribe_wrapper import CORTICAL_DIM, DUAL_PASS_SUBCORTICAL_DIM

logger = logging.getLogger("neuroclaw.atlas")

# Harvard-Oxford substring labels (case-insensitive) for tribev2 subcortical helper
NACC_QUERY = "Accumbens"
AMY_QUERY = "Amygdala"

# Destrieux aparc labels on fsaverage5 (nilearn)
FFA_LABELS = ("G_oc-temp_lat-fusifor",)
VPFC_LABELS = ("G_rectus", "G_subcallosal")

KEY_NACC = "NAcc"
KEY_AMYGDALA = "Amygdala"
KEY_FFA = "FFA"
KEY_VPFC = "vmPFC"

DUAL_PASS_ROI_MAP_KEYS = (KEY_NACC, KEY_AMYGDALA, KEY_FFA, KEY_VPFC)


def _placeholder_dual_pass_roi_map() -> dict[str, np.ndarray]:
    """Deterministic indices for CI when tribev2/nilearn unavailable."""
    s = 0
    return {
        KEY_NACC: np.arange(s, s + 32, dtype=np.int64),
        KEY_AMYGDALA: np.arange(s + 32, s + 64, dtype=np.int64),
        KEY_FFA: np.arange(1000, 1000 + 128, dtype=np.int64),
        KEY_VPFC: np.arange(2000, 2000 + 256, dtype=np.int64),
    }


def get_subcortical_indices_relative(roi_query: str) -> np.ndarray:
    """
    Indices into the native 8,802 subcortical output vector (Pass 1).
    Uses tribev2.plotting.subcortical.get_subcortical_roi_indices.
    """
    try:
        from tribev2.plotting.subcortical import get_subcortical_roi_indices  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("tribev2_subcortical_import_failed", extra={"roi": roi_query})
        raise
    rel = np.asarray(get_subcortical_roi_indices(roi_query), dtype=np.int64)
    if rel.max() >= DUAL_PASS_SUBCORTICAL_DIM or rel.min() < 0:
        msg = f"Subcortical indices out of range [0, {DUAL_PASS_SUBCORTICAL_DIM}) for {roi_query!r}"
        raise ValueError(msg)
    return rel


def get_cortical_indices_destrieux(labels: tuple[str, ...]) -> np.ndarray:
    """
    Vertex indices into the 20,484 fsaverage5 cortical output (Pass 2).
    Uses nilearn Destrieux surface atlas (concatenated lh+rh order).
    """
    try:
        from nilearn.datasets import fetch_atlas_surf_destrieux
    except ImportError as e:
        msg = "nilearn is required for Destrieux cortical ROI indices"
        raise RuntimeError(msg) from e

    atlas = fetch_atlas_surf_destrieux("fsaverage5")
    full = np.concatenate([atlas.map_left, atlas.map_right])
    name_list = list(atlas.labels)
    chunks: list[np.ndarray] = []
    for lab in labels:
        if lab not in name_list:
            msg = f"Destrieux label not found in atlas: {lab!r}"
            raise ValueError(msg)
        li = name_list.index(lab)
        if li == 0:
            msg = f"Destrieux label {lab!r} is background index 0"
            raise ValueError(msg)
        chunks.append(np.where(full == li)[0])
    merged = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    indices = np.unique(merged).astype(np.int64)
    if indices.size == 0:
        msg = f"No vertices matched labels {labels!r}"
        raise ValueError(msg)
    if indices.max() >= CORTICAL_DIM or indices.min() < 0:
        msg = f"Cortical indices out of bounds [0, {CORTICAL_DIM})"
        raise ValueError(msg)
    return indices


def build_dual_pass_roi_map() -> dict[str, np.ndarray]:
    """
    Returns mapping KEY -> int64 index array into Pass1 (NAcc, Amy) or Pass2 (FFA, vmPFC) vectors.
    """
    if os.environ.get("NEUROCLAW_PLACEHOLDER_ATLAS", "").lower() in ("1", "true", "yes"):
        return _placeholder_dual_pass_roi_map()

    try:
        nacc = get_subcortical_indices_relative(NACC_QUERY)
        amy = get_subcortical_indices_relative(AMY_QUERY)
    except Exception as e:
        logger.warning("subcortical_roi_fallback", extra={"error": str(e)})
        return _placeholder_dual_pass_roi_map()

    try:
        ffa = get_cortical_indices_destrieux(FFA_LABELS)
        vpfc = get_cortical_indices_destrieux(VPFC_LABELS)
    except Exception as e:
        logger.warning("cortical_roi_fallback", extra={"error": str(e)})
        ph = _placeholder_dual_pass_roi_map()
        return {
            KEY_NACC: nacc,
            KEY_AMYGDALA: amy,
            KEY_FFA: ph[KEY_FFA],
            KEY_VPFC: ph[KEY_VPFC],
        }

    return {
        KEY_NACC: nacc,
        KEY_AMYGDALA: amy,
        KEY_FFA: ffa,
        KEY_VPFC: vpfc,
    }
