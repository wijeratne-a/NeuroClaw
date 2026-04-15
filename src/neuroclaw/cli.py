"""neuroclaw-extract — multimodal ingestion and Cortical Four .safetensors.zst export."""

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

from neuroclaw.atlas.cortical_marketing_four import build_cortical_marketing_four_roi_vertices
from neuroclaw.compliance import TRANSPARENCY_NOTICE, validate_use_case
from neuroclaw.config import load_settings
from neuroclaw.extractor.alignment import assert_drift_limits, audit_drift
from neuroclaw.extractor.audio import extract_audio_features
from neuroclaw.extractor.asr import transcribe_clip
from neuroclaw.extractor.text import extract_ocr, ocr_events_to_dataframe_rows
from neuroclaw.extractor.video import process_video
from neuroclaw.model.events_builder import _audio_filepath_for_tribe, build_events_df
from neuroclaw.model.loader import staged_stage
from neuroclaw.model.single_pass import run_cortical_marketing_four
from neuroclaw.model.tribe_wrapper import TRIBEV2_PINNED_COMMIT, load_tribe, resolve_tribe_device, use_mock
from neuroclaw.output.manifest import build_manifest, write_manifest_sidecar
from neuroclaw.output.schema import (
    DTYPE,
    INFERENCE_LAYOUT_CORTICAL_MARKETING_FOUR,
    KEY_ROI_FFA,
    KEY_ROI_IFG,
    KEY_ROI_INSULA,
    KEY_ROI_VMPFC,
    validate_metadata,
)
from neuroclaw.output.validator import validate_artifact_zst
from neuroclaw.output.writer import write_cortical_four_artifact
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
        typer.Option("--dual-pass", help="Deprecated: ignored; cortical-only pipeline runs"),
    ] = False,
) -> None:
    """Extract multimodal features and write Cortical Marketing Four ROI tensors."""
    setup_json_logging()
    run_uuid = str(uuid.uuid4())
    log = RunContextAdapter(logging.getLogger("neuroclaw"), {"run_uuid": run_uuid})

    if dual_pass:
        console.print(
            "[yellow]WARNING: --dual-pass is deprecated. TRIBE v2 open weights are cortical-only. "
            "Falling back to single-pass Cortical Four extraction.[/yellow]"
        )

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

    wav = Path(_audio_filepath_for_tribe(video_path))
    transcript = transcribe_clip(wav, temperature=0.0)
    extra_asr: list[dict] = []
    tx = (transcript.get("text") or "").strip()
    if tx:
        extra_asr.append(
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

    tribe_device = resolve_tribe_device(os.environ.get("NEUROCLAW_DEVICE", "auto"))
    model = load_tribe(device=tribe_device, hf_token=settings.hf_token)

    merged_extra = (list(ocr_rows) if ocr_rows else []) + extra_asr
    events_df = build_events_df(
        model,
        video_path,
        external_events=ext_events_path,
        extra_rows=merged_extra or None,
    )

    drift = audit_drift(bin_starts, audio.times_s, events_df)
    assert_drift_limits(drift)

    roi_vertices, atlas_meta = build_cortical_marketing_four_roi_vertices()

    with staged_stage("tribe_predict", settings, run_uuid=run_uuid):
        cf = run_cortical_marketing_four(
            model,
            events_df,
            duration_s,
            bin_starts,
            roi_vertices,
        )

    t_bins = int(cf.ffa.shape[0])
    tb = max(0, t_bins - 1)
    region_map = {
        KEY_ROI_FFA: [0, tb],
        KEY_ROI_VMPFC: [0, tb],
        KEY_ROI_IFG: [0, tb],
        KEY_ROI_INSULA: [0, tb],
    }

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
        "roi_ordering": "roi_only_cortical_four",
        "inference_layout": INFERENCE_LAYOUT_CORTICAL_MARKETING_FOUR,
        "tribev2_context": {
            "mask_mode": "fsaverage5_only",
            "roi_mapping_version": "destrieux-fsaverage5",
            "limitations": (
                "Subcortical paths excluded; public best.ckpt contains fsaverage5 cortical surface only."
            ),
            "fwhm": 6.0,
            "commit_hash": TRIBEV2_PINNED_COMMIT,
        },
        "atlas_versions": atlas_meta,
        "drift_stats": {
            "max_abs_drift_ms": drift.max_abs_drift_ms,
            "p95_drift_ms": drift.p95_drift_ms,
            "per_pair": drift.per_pair,
        },
        "transparency_label": TRANSPARENCY_NOTICE,
        "use_case_audit": use_case_norm,
        "region_map": region_map,
    }
    validate_metadata(meta)

    roi_series = {
        KEY_ROI_FFA: cf.ffa,
        KEY_ROI_VMPFC: cf.vmpfc,
        KEY_ROI_IFG: cf.ifg,
        KEY_ROI_INSULA: cf.insula,
    }

    paths = write_cortical_four_artifact(
        clip_id=cid,
        run_uuid=run_uuid,
        out_root=out_root.resolve(),
        roi_series=roi_series,
        timestamps=cf.timestamps,
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


def main() -> None:
    """Console_scripts entry."""
    app()


if __name__ == "__main__":
    main()
