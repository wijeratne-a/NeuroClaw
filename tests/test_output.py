"""Output writer, manifest, validator (Cortical Four)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from neuroclaw.extractor.alignment import DriftStats
from neuroclaw.output.manifest import build_manifest
from neuroclaw.output.schema import (
    INFERENCE_LAYOUT_CORTICAL_MARKETING_FOUR,
    KEY_ROI_FFA,
    KEY_ROI_IFG,
    KEY_ROI_INSULA,
    KEY_ROI_VMPFC,
)
from neuroclaw.output.validator import validate_artifact_zst
from neuroclaw.output.writer import write_cortical_four_safetensors_zst


def test_cortical_four_write_validate_roundtrip() -> None:
    rng = np.random.default_rng(7)
    t = 8
    ts = np.arange(t, dtype=np.float64)
    roi = {
        KEY_ROI_FFA: rng.standard_normal(t).astype(np.float16),
        KEY_ROI_VMPFC: rng.standard_normal(t).astype(np.float16),
        KEY_ROI_IFG: rng.standard_normal(t).astype(np.float16),
        KEY_ROI_INSULA: rng.standard_normal(t).astype(np.float16),
    }
    meta = {
        "model_id": "facebook/tribev2",
        "tribev2_version": "0.3.x",
        "created_utc": "2026-01-01T00:00:00Z",
        "run_uuid": "u2",
        "git_sha": "unknown",
        "device": "cpu",
        "dtype": "float16",
        "hemodynamic_offset_s": 5.0,
        "roi_ordering": "roi_only_cortical_four",
        "inference_layout": INFERENCE_LAYOUT_CORTICAL_MARKETING_FOUR,
        "tribev2_context": {
            "mask_mode": "fsaverage5_only",
            "roi_mapping_version": "destrieux-fsaverage5",
            "limitations": "test",
            "fwhm": 6.0,
            "commit_hash": "72399081ed3f1040c4d996cefb2864a4c46f5b8e",
        },
        "atlas_versions": {"mode": "placeholder"},
        "drift_stats": {
            "max_abs_drift_ms": 0.0,
            "p95_drift_ms": 0.0,
            "per_pair": {},
        },
        "transparency_label": "x",
        "use_case_audit": "commercial_content_optimization",
        "region_map": {},
    }
    from neuroclaw.output.schema import validate_metadata

    validate_metadata(meta)

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cf.safetensors.zst"
        write_cortical_four_safetensors_zst(roi, ts, meta, p)
        drift = DriftStats(0.0, 0.0, {})
        vr = validate_artifact_zst(str(p), drift)
        assert vr.ok, vr.errors
        assert vr.details.get("inference_layout") == INFERENCE_LAYOUT_CORTICAL_MARKETING_FOUR
        man = build_manifest(p)
        assert len(man.header_sha256) == 64
        assert len(man.full_file_sha256) == 64
