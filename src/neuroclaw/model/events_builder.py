"""Build TRIBE events DataFrame; optional external CSV override."""

from __future__ import annotations

from pathlib import Path
import inspect
import logging
import os
import shutil
import subprocess
from typing import Any

import pandas as pd

from neuroclaw.model.local_asr import word_rows_from_wav

REQUIRED_COLS = ("type", "start", "duration")
logger = logging.getLogger("neuroclaw.model.events")

try:
    import ffmpeg
except ImportError:
    ffmpeg = None  # type: ignore[misc, assignment]


def load_external_events(path: Path) -> pd.DataFrame:
    """UTF-8 comma CSV: type, start, duration, text, context."""
    df = pd.read_csv(path, encoding="utf-8")
    for c in REQUIRED_COLS:
        if c not in df.columns:
            msg = f"external_events missing column: {c}"
            raise ValueError(msg)
    df["start"] = df["start"].astype("float64")
    df["duration"] = df["duration"].astype("float64")
    df["type"] = df["type"].astype(str)
    if "text" not in df.columns:
        df["text"] = ""
    if "context" not in df.columns:
        df["context"] = ""
    return df


def merge_external_over_asr(base: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    """External overrides ASR rows overlapping same start (±50ms)."""
    if external is None or len(external) == 0:
        return base
    ext_starts = set(external["start"].round(3))
    mask = ~base["start"].round(3).isin(ext_starts)
    rest = base.loc[mask] if len(base) else base
    return pd.concat([external, rest], ignore_index=True).sort_values("start")


def _probe_duration_s(video_path: Path) -> float:
    if ffmpeg is None:
        return 1.0
    try:
        info = ffmpeg.probe(str(video_path))
        return max(1.0, float(info.get("format", {}).get("duration", 1.0)))
    except Exception:
        return 1.0


def _audio_filepath_for_tribe(video_path: Path) -> str:
    """
    TRIBE/neuralset validates Audio events with soundfile, which cannot read mp4/mkv.
    Prefer an existing sidecar .wav; otherwise try ffmpeg to extract mono 16kHz PCM.
    """
    vp = video_path.resolve()
    suf = vp.suffix.lower()
    if suf in (".wav", ".flac", ".ogg", ".aiff", ".aif"):
        return str(vp)
    sidecar = vp.with_suffix(".wav")
    if sidecar.is_file():
        return str(sidecar)
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        logger.warning(
            "ffmpeg_missing_audio_sidecar",
            extra={
                "video": str(vp),
                "hint": "install ffmpeg or place a .wav next to the video for Audio events",
            },
        )
        return str(vp)
    try:
        subprocess.run(
            [ffmpeg_bin, "-y", "-i", str(vp), "-ac", "1", "-ar", "16000", str(sidecar)],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("audio_sidecar_extracted", extra={"wav": str(sidecar)})
        return str(sidecar)
    except Exception as e:
        logger.warning(
            "audio_sidecar_extract_failed",
            extra={"video": str(vp), "error": str(e)},
        )
        return str(vp)


def _prefer_local_asr_only() -> bool:
    return os.environ.get("NEUROCLAW_PREFER_LOCAL_ASR", "").lower() in ("1", "true", "yes")


def _local_asr_on_fallback_enabled() -> bool:
    return os.environ.get("NEUROCLAW_LOCAL_ASR_ON_FALLBACK", "1").lower() not in (
        "0",
        "false",
        "no",
    )


def _append_local_asr_words(video_path: Path, base: pd.DataFrame) -> pd.DataFrame:
    """Append Word rows from faster-whisper (same env as NeuroClaw); no-op if disabled or empty."""
    if not _local_asr_on_fallback_enabled():
        return base
    wav = Path(_audio_filepath_for_tribe(video_path))
    words = word_rows_from_wav(wav)
    if not words:
        return base
    ex = pd.DataFrame(words)
    return pd.concat([base, ex], ignore_index=True)


def _fallback_events_df(video_path: Path) -> pd.DataFrame:
    """
    Fallback used when tribev2 internal transcription pipeline fails (uvx/whisperx).
    Keep minimal multimodal rows so downstream prediction can proceed.
    """
    dur = _probe_duration_s(video_path)
    audio_fp = _audio_filepath_for_tribe(video_path)
    return pd.DataFrame(
        [
            {
                "type": "Video",
                "start": 0.0,
                "duration": dur,
                "filepath": str(video_path),
                "timeline": "default",
                "subject": "default",
                "text": "",
                "context": "",
                "channel": "video",
            },
            {
                "type": "Audio",
                "start": 0.0,
                "duration": dur,
                "filepath": audio_fp,
                "timeline": "default",
                "subject": "default",
                "text": "",
                "context": "",
                "channel": "audio",
            },
        ]
    )


def build_events_df(
    model: Any,
    video_path: Path,
    external_events: Path | None = None,
    extra_rows: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """
    Start from model.get_events_dataframe when available; merge external CSV and OCR rows.
    """
    video_path = Path(video_path)
    if hasattr(model, "get_events_dataframe") and not _prefer_local_asr_only():
        # tribev2 API differs by version; prefer explicit keyword calls.
        fn = model.get_events_dataframe
        try:
            sig = inspect.signature(fn)
            params = set(sig.parameters.keys())
        except (TypeError, ValueError):
            params = set()

        try:
            if "video_path" in params:
                base = fn(video_path=str(video_path))
            else:
                # Fallback for older signatures expecting positional first arg.
                try:
                    base = fn(str(video_path))
                except Exception:
                    # Try broad keyword fallback used by some wrappers.
                    base = fn(path=str(video_path))
        except Exception as e:
            msg = str(e)
            # Known flaky dependency path in tribev2: uvx/whisperx transcription.
            transcription_markers = (
                "whisperx failed",
                "uvx",
                "whisperx",
                "wav2vec2forctc",
                "_array_api not found",
                "compiled using numpy 1.x",
            )
            if any(m in msg.lower() for m in transcription_markers):
                logger.warning(
                    "events_fallback_used",
                    extra={"reason": "transcription_dependency_failure", "error": msg},
                )
                base = _append_local_asr_words(video_path, _fallback_events_df(video_path))
            else:
                raise
    elif hasattr(model, "get_events_dataframe") and _prefer_local_asr_only():
        logger.info(
            "events_skip_tribe_internal_asr",
            extra={"reason": "NEUROCLAW_PREFER_LOCAL_ASR", "asr": "faster_whisper"},
        )
        base = _append_local_asr_words(video_path, _fallback_events_df(video_path))
    else:
        base = pd.DataFrame(columns=list(REQUIRED_COLS) + ["text", "context", "filepath", "channel"])

    for c in REQUIRED_COLS:
        if c not in base.columns:
            if c == "type":
                base[c] = ""
            else:
                base[c] = 0.0

    if external_events is not None:
        ext = load_external_events(Path(external_events))
        base = merge_external_over_asr(base, ext)

    if extra_rows:
        ex = pd.DataFrame(extra_rows)
        base = pd.concat([base, ex], ignore_index=True)

    base = base.sort_values("start").reset_index(drop=True)
    base["start"] = base["start"].astype("float64")
    base["duration"] = base["duration"].astype("float64")
    base["type"] = base["type"].astype(str)
    # Unseen-subject population baseline (tribev2 routes None → generic embedding)
    base["subject_id"] = None
    return base
