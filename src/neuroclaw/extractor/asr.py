"""Full-clip ASR for dual-pass deterministic transcripts (faster-whisper)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from neuroclaw.model.local_asr import _resolve_asr_device, _resolve_compute_type

logger = logging.getLogger("neuroclaw.extractor.asr")


def transcribe_clip(
    wav_path: Path,
    *,
    model_id: str | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """
    Transcribe audio to full text + segments. ``temperature`` must be 0 for determinism.
    Returns schema compatible with ``{clip_id}_transcript.json`` sidecar.
    """
    import os

    wp = Path(wav_path).resolve()
    if not wp.is_file():
        logger.warning("transcribe_clip_missing_wav", extra={"path": str(wp)})
        return {
            "text": "",
            "segments": [],
            "language": "unknown",
            "asr_model": "none",
            "temperature": float(temperature),
        }

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.warning("faster_whisper_not_installed")
        return {
            "text": "",
            "segments": [],
            "language": "unknown",
            "asr_model": "none",
            "temperature": float(temperature),
        }

    mid = (model_id or os.environ.get("NEUROCLAW_ASR_MODEL") or "base").strip() or "base"
    device = _resolve_asr_device()
    compute_type = _resolve_compute_type(device)
    model = WhisperModel(mid, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        str(wp),
        temperature=temperature,
        word_timestamps=False,
        vad_filter=True,
    )
    seg_out: list[dict[str, Any]] = []
    texts: list[str] = []
    for seg in segments:
        t0 = float(getattr(seg, "start", 0.0) or 0.0)
        t1 = float(getattr(seg, "end", t0) or t0)
        tx = (getattr(seg, "text", "") or "").strip()
        texts.append(tx)
        seg_out.append({"start": t0, "end": t1, "text": tx})
    lang = getattr(info, "language", None) or "unknown"
    return {
        "text": " ".join(texts).strip(),
        "segments": seg_out,
        "language": str(lang),
        "asr_model": f"faster-whisper-{mid}",
        "temperature": float(temperature),
    }
