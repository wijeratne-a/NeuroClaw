"""Local ASR (faster-whisper) — same interpreter as NeuroClaw; bypasses tribev2 uvx/whisperx."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("neuroclaw.model.local_asr")


def _resolve_asr_device() -> str:
    r = (os.environ.get("NEUROCLAW_ASR_DEVICE") or "auto").strip().lower()
    if r not in ("auto", ""):
        return r
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "cpu"
    except Exception:
        pass
    return "cpu"


def _resolve_compute_type(device: str) -> str:
    ct = (os.environ.get("NEUROCLAW_ASR_COMPUTE_TYPE") or "").strip()
    if ct:
        return ct
    return "int8" if device == "cpu" else "float16"


def word_rows_from_wav(wav_path: Path) -> list[dict[str, Any]]:
    """
    Transcribe a mono WAV and return TRIBE-compatible Word rows (timeline/subject default).
    Empty list if faster-whisper is unavailable or transcription fails.
    """
    wp = Path(wav_path).resolve()
    if not wp.is_file():
        logger.warning("local_asr_missing_wav", extra={"path": str(wp)})
        return []

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.info(
            "local_asr_unavailable",
            extra={"reason": "faster_whisper_not_installed", "hint": "pip install 'neuroclaw[local-asr]'" },
        )
        return []

    model_id = (os.environ.get("NEUROCLAW_ASR_MODEL") or "base").strip() or "base"
    device = _resolve_asr_device()
    compute_type = _resolve_compute_type(device)

    rows: list[dict[str, Any]] = []
    try:
        model = WhisperModel(model_id, device=device, compute_type=compute_type)
        segments, _info = model.transcribe(
            str(wp),
            word_timestamps=True,
            vad_filter=True,
        )
        for seg in segments:
            words = getattr(seg, "words", None) or []
            for w in words:
                start = float(getattr(w, "start", 0.0) or 0.0)
                end = float(getattr(w, "end", start) or start)
                txt = (getattr(w, "word", "") or "").strip()
                if not txt:
                    continue
                dur = max(1e-3, end - start)
                rows.append(
                    {
                        "type": "Word",
                        "start": start,
                        "duration": dur,
                        "text": txt,
                        "context": "",
                        "filepath": "",
                        "channel": "transcript",
                        "timeline": "default",
                        "subject": "default",
                    }
                )
    except Exception as e:
        logger.warning("local_asr_failed", extra={"error": str(e), "wav": str(wp)})
        return []

    logger.info("local_asr_words", extra={"n": len(rows), "wav": str(wp), "model": model_id})
    return rows
