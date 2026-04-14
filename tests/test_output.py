"""Output writer, manifest, validator."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from neuroclaw.extractor.alignment import DriftStats
from neuroclaw.output.manifest import build_manifest
from neuroclaw.output.validator import validate_artifact_zst
from neuroclaw.output.schema import (
    INFERENCE_LAYOUT_DUAL_PASS_AGGREGATED,
    KEY_ROI_AMYGDALA,
    KEY_ROI_FFA,
    KEY_ROI_NACC,
    KEY_ROI_VPFC,
)
from neuroclaw.output.writer import write_dual_pass_safetensors_zst, write_safetensors_zst


def test_write_validate_roundtrip() -> None:
    rng = np.random.default_rng(42)
    t = 12
    va = rng.standard_normal((t, 29286)).astype(np.float32)
    va = (va - va.mean()) / (va.std() + 1e-8)
    va = va.astype(np.float16)
    m4 = rng.standard_normal((t, 32)).astype(np.float16)
    ts = np.arange(t, dtype=np.float64)
    meta = {
        "model_id": "facebook/tribev2",
        "tribev2_version": "0.3.x",
        "created_utc": "2026-01-01T00:00:00Z",
        "run_uuid": "u1",
        "git_sha": "unknown",
        "device": "cpu",
        "dtype": "float16",
        "hemodynamic_offset_s": 5.0,
        "voxel_ordering": "cortical_then_subcortical",
        "inference_layout": "whole_brain",
        "atlas_versions": {},
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
        p = Path(td) / "c_voxels.safetensors.zst"
        write_safetensors_zst(va, m4, ts, meta, p)
        drift = DriftStats(0.0, 0.0, {})
        vr = validate_artifact_zst(str(p), drift)
        assert vr.ok, vr.errors
        assert vr.details.get("inference_layout") == "whole_brain"
        man = build_manifest(p)
        assert len(man.header_sha256) == 64
        assert len(man.full_file_sha256) == 64


def test_dual_pass_write_validate_roundtrip() -> None:
    rng = np.random.default_rng(7)
    t = 8
    ts = np.arange(t, dtype=np.float64)
    roi = {
        KEY_ROI_NACC: rng.standard_normal(t).astype(np.float16),
        KEY_ROI_AMYGDALA: rng.standard_normal(t).astype(np.float16),
        KEY_ROI_FFA: rng.standard_normal(t).astype(np.float16),
        KEY_ROI_VPFC: rng.standard_normal(t).astype(np.float16),
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
        "voxel_ordering": "dual_pass_sequential",
        "inference_layout": INFERENCE_LAYOUT_DUAL_PASS_AGGREGATED,
        "tribev2_context": {
            "mask_mode": "dual_pass",
            "roi_mapping_version": "harvard-oxford-sub-2mm_and_destrieux-fsaverage5",
            "fwhm": 6.0,
            "commit_hash": "72399081ed3f1040c4d996cefb2864a4c46f5b8e",
        },
        "atlas_versions": {"mode": "dual_pass"},
        "drift_stats": {
            "max_abs_drift_ms": 0.0,
            "p95_drift_ms": 0.0,
            "per_pair": {},
        },
        "transparency_label": "x",
        "use_case_audit": "commercial_content_optimization",
        "region_map": {},
        "ffa_bilateral_metadata": {},
    }
    from neuroclaw.output.schema import validate_metadata

    validate_metadata(meta)

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "dp.safetensors.zst"
        write_dual_pass_safetensors_zst(roi, ts, meta, p)
        drift = DriftStats(0.0, 0.0, {})
        vr = validate_artifact_zst(str(p), drift)
        assert vr.ok, vr.errors
        assert vr.details.get("inference_layout") == INFERENCE_LAYOUT_DUAL_PASS_AGGREGATED
