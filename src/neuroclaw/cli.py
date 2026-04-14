"""neuroclaw-extract — multimodal ingestion and .safetensors.zst export."""

from __future__ import annotations

import logging
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.console import Console

from neuroclaw.atlas.dual_pass_rois import build_dual_pass_roi_map
from neuroclaw.atlas.marketing_four import assemble_marketing4
from neuroclaw.atlas.masks import build_roi_masks, validate_indices
from neuroclaw.compliance import TRANSPARENCY_NOTICE, validate_use_case
from neuroclaw.config import load_settings
from neuroclaw.extractor.alignment import assert_drift_limits, audit_drift
from neuroclaw.extractor.audio import extract_audio_features
from neuroclaw.extractor.asr import transcribe_clip
from neuroclaw.extractor.text import extract_ocr, ocr_events_to_dataframe_rows
from neuroclaw.extractor.video import process_video
from neuroclaw.model.dual_pass import run_dual_pass
from neuroclaw.model.events_builder import _audio_filepath_for_tribe, build_events_df
from neuroclaw.model.loader import staged_stage
from neuroclaw.model.tribe_wrapper import (
    TRIBEV2_PINNED_COMMIT,
    load_tribe,
    predict_voxels,
    resolve_tribe_device,
    use_mock,
)
from neuroclaw.output.manifest import build_manifest, write_manifest_sidecar
from neuroclaw.output.schema import (
    DTYPE,
    INFERENCE_LAYOUT_DUAL_PASS_AGGREGATED,
    KEY_ROI_AMYGDALA,
    KEY_ROI_FFA,
    KEY_ROI_NACC,
    KEY_ROI_VPFC,
    validate_metadata,
)
from neuroclaw.output.validator import validate_artifact_zst
from neuroclaw.output.writer import write_artifact, write_dual_pass_artifact
from neuroclaw.utils.adapter import RunContextAdapter
from neuroclaw.utils.determinism import set_deterministic
from neuroclaw.utils.git import get_git_sha
from neuroclaw.utils.logging import setup_json_logging

console = Console(stderr=True)
app = typer.Typer(no_args_is_help=True, add_completion=False, name="neuroclaw-extract")


@app.command()
def extract(
    input_path: Annotated[Path, typer.Option("--input", "-i", help="Video .mp4 or .mkv")],
    use_case: Annotated[str, typer.Option("--use-case", help="Allowlisted use case tag")],
    external_events: Annotated[
        Path | None,
        typer.Option("--external-events", help="Optional UTF-8 CSV TRIBE schema"),
    ] = None,
    strict: Annotated[bool, typer.Option("--strict", help="Hard fail on OCR/ASR issues")] = False,
    strict_determinism: Annotated[
        bool | None,
        typer.Option(
            "--strict-determinism/--no-strict-determinism",
            help="PyTorch deterministic kernels (default: on)",
        ),
    ] = None,
    out_root: Annotated[Path, typer.Option("--out", help="Artifact root")] = Path("artifacts"),
    clip_id: Annotated[str | None, typer.Option("--clip-id", help="Override clip id")] = None,
    dual_pass: Annotated[
        bool,
        typer.Option("--dual-pass", help="Dual-pass TRIBE (8802 subcortical + 20484 cortical) ROI export"),
    ] = False,
) -> None:
    """Extract multimodal features and write voxel tensors."""
    setup_json_logging()
    run_uuid = str(uuid.uuid4())
    log = RunContextAdapter(logging.getLogger("neuroclaw"), {"run_uuid": run_uuid})

    validate_use_case(use_case)
    use_case_norm = "commercial_content_optimization"

    settings = load_settings()
    if not use_mock():
        tok = settings.require_hf_token()
        os.environ["HF_TOKEN"] = tok
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = tok

    det = True if strict_determinism is None else bool(strict_determinism)
    set_deterministic(seed=settings.SEED, strict=det)

    video_path = Path(input_path).resolve()
    cid = clip_id or video_path.stem
    ext_events_path: Path | None = None
    if external_events is not None:
        candidate = Path(external_events).expanduser().resolve()
        if candidate.exists():
            ext_events_path = candidate
        else:
            log.warning(
                "external_events_missing",
                extra={
                    "path": str(candidate),
                    "action": "continue_without_external_events",
                    "run_uuid": run_uuid,
                },
            )

    with staged_stage("video_decode", settings, run_uuid=run_uuid):
        vpack = process_video(video_path)

    bin_starts = np.asarray(vpack["bin_starts_s"], dtype=np.float64)
    duration_s = float(vpack["duration_s"])

    with staged_stage("audio_w2v", settings, run_uuid=run_uuid):
        audio = extract_audio_features(
            video_path,
            bin_starts_s=bin_starts,
            duration_s=duration_s,
        )

    ocr_rows: list[dict] = []
    try:
        ocr_ev = extract_ocr(
            vpack["frames_1hz"],
            bin_starts,
            strict=strict,
        )
        ocr_rows = ocr_events_to_dataframe_rows(ocr_ev)
    except Exception as e:
        if strict:
            raise
        log.warning("ocr_failed", extra={"error": str(e), "run_uuid": run_uuid})

    tribe_device = resolve_tribe_device(os.environ.get("NEUROCLAW_DEVICE", "auto"))
    model = load_tribe(device=tribe_device, hf_token=settings.hf_token)

    extra_dual: list[dict] = []
    transcript: dict | None = None
    if dual_pass:
        wav = Path(_audio_filepath_for_tribe(video_path))
        transcript = transcribe_clip(wav, temperature=0.0)
        tx = (transcript.get("text") or "").strip()
        if tx:
            extra_dual.append(
                {
                    "type": "Text",
                    "start": 0.0,
                    "duration": duration_s,
                    "text": tx,
                    "context": "",
                    "filepath": "",
                    "channel": "text",
                }
            )

    merged_extra = (list(ocr_rows) if ocr_rows else []) + extra_dual
    events_df = build_events_df(
        model,
        video_path,
        external_events=ext_events_path,
        extra_rows=merged_extra or None,
    )

    drift = audit_drift(bin_starts, audio.times_s, events_df)
    assert_drift_limits(drift)

    if dual_pass:
        roi_map = build_dual_pass_roi_map()
        mock_o: tuple[int, int] | None = None
        if use_mock():
            from neuroclaw.model.tribe_wrapper import DUAL_PASS_CORTICAL_DIM, DUAL_PASS_SUBCORTICAL_DIM

            mock_o = (DUAL_PASS_SUBCORTICAL_DIM, DUAL_PASS_CORTICAL_DIM)
        with staged_stage("tribe_predict", settings, run_uuid=run_uuid):
            dp = run_dual_pass(
                events_df,
                duration_s,
                bin_starts,
                roi_map,
                settings,
                run_uuid,
                tribe_device,
                settings.hf_token,
                mock_native_o=mock_o,
            )
        t_bins = int(dp.nacc.shape[0])
        tb = max(0, t_bins - 1)
        region_map = {
            KEY_ROI_NACC: [0, tb],
            KEY_ROI_AMYGDALA: [0, tb],
            KEY_ROI_FFA: [0, tb],
            KEY_ROI_VPFC: [0, tb],
        }
        meta = {
            "model_id": "facebook/tribev2",
            "tribev2_version": os.environ.get("TRIBEV2_VERSION", "0.3.x"),
            "created_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "run_uuid": run_uuid,
            "git_sha": get_git_sha(),
            "device": tribe_device,
            "dtype": DTYPE,
            "hemodynamic_offset_s": settings.HEMODYNAMIC_OFFSET_S,
            "voxel_ordering": "dual_pass_sequential",
            "inference_layout": INFERENCE_LAYOUT_DUAL_PASS_AGGREGATED,
            "tribev2_context": {
                "mask_mode": "dual_pass",
                "roi_mapping_version": "harvard-oxford-sub-2mm_and_destrieux-fsaverage5",
                "fwhm": 6.0,
                "commit_hash": TRIBEV2_PINNED_COMMIT,
            },
            "atlas_versions": {
                "subcortical": "harvard-oxford-via-tribev2",
                "cortical": "destrieux-fsaverage5",
                "mode": "dual_pass",
            },
            "drift_stats": {
                "max_abs_drift_ms": drift.max_abs_drift_ms,
                "p95_drift_ms": drift.p95_drift_ms,
                "per_pair": drift.per_pair,
            },
            "transparency_label": TRANSPARENCY_NOTICE,
            "use_case_audit": use_case_norm,
            "region_map": region_map,
            "ffa_bilateral_metadata": {"dual_pass": True, "note": "ROI series from dual-pass pooled vertices"},
        }
        validate_metadata(meta)
        roi_series = {
            KEY_ROI_NACC: dp.nacc,
            KEY_ROI_AMYGDALA: dp.amygdala,
            KEY_ROI_FFA: dp.ffa,
            KEY_ROI_VPFC: dp.vmpfc,
        }
        paths = write_dual_pass_artifact(
            clip_id=cid,
            run_uuid=run_uuid,
            out_root=out_root.resolve(),
            roi_series=roi_series,
            timestamps=dp.timestamps,
            model_metadata=meta,
            duration_s=duration_s,
            transcript=transcript,
        )
        for p in paths:
            vr = validate_artifact_zst(str(p), drift)
            if not vr.ok:
                log.error("validation_failed", extra={"errors": vr.errors, "run_uuid": run_uuid})
                raise typer.Exit(code=1)
            man = build_manifest(Path(p))
            write_manifest_sidecar(man, cid, Path(p).parent)
            console.print(f"[green]OK[/green] {p}")

        console.print(f"run_uuid={run_uuid}")
        return

    with staged_stage("tribe_predict", settings, run_uuid=run_uuid):
        vox_result = predict_voxels(model, events_df, duration_s)
    vox = vox_result.voxels
    inference_layout = "cortical_only" if vox_result.cortical_only else "whole_brain"

    n = min(vox.shape[0], len(bin_starts))
    vox = vox[:n]
    bin_starts = bin_starts[:n]

    roi = build_roi_masks()
    idx_map = roi["indices"]
    validate_indices(idx_map, cortical_only=vox_result.cortical_only)
    m4, region_map, ffa_meta = assemble_marketing4(
        np.asarray(vox, dtype=np.float32),
        idx_map,
        cortical_only=vox_result.cortical_only,
    )

    tribev2_ver = os.environ.get("TRIBEV2_VERSION", "0.3.x")
    meta = {
        "model_id": "facebook/tribev2",
        "tribev2_version": tribev2_ver,
        "created_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "run_uuid": run_uuid,
        "git_sha": get_git_sha(),
        "device": tribe_device,
        "dtype": DTYPE,
        "hemodynamic_offset_s": settings.HEMODYNAMIC_OFFSET_S,
        "voxel_ordering": "cortical_then_subcortical",
        "inference_layout": inference_layout,
        "atlas_versions": roi["atlas_versions"],
        "drift_stats": {
            "max_abs_drift_ms": drift.max_abs_drift_ms,
            "p95_drift_ms": drift.p95_drift_ms,
            "per_pair": drift.per_pair,
        },
        "transparency_label": TRANSPARENCY_NOTICE,
        "use_case_audit": use_case_norm,
        "region_map": region_map,
        "ffa_bilateral_metadata": ffa_meta,
    }
    validate_metadata(meta)

    paths = write_artifact(
        clip_id=cid,
        run_uuid=run_uuid,
        out_root=out_root.resolve(),
        voxels_all=np.asarray(vox, dtype=np.float16),
        voxels_m4=m4,
        timestamps=bin_starts.astype(np.float64),
        model_metadata=meta,
        duration_s=duration_s,
    )

    for p in paths:
        vr = validate_artifact_zst(str(p), drift)
        if not vr.ok:
            log.error("validation_failed", extra={"errors": vr.errors, "run_uuid": run_uuid})
            raise typer.Exit(code=1)
        man = build_manifest(Path(p))
        write_manifest_sidecar(man, cid, Path(p).parent)
        console.print(f"[green]OK[/green] {p}")

    console.print(f"run_uuid={run_uuid}")


def main() -> None:
    """Console_scripts entry."""
    app()


if __name__ == "__main__":
    main()
