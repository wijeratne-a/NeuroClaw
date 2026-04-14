"""OCR (optional) + transcript channels; merge rules for TRIBE events."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger("neuroclaw.extractor.text")


@dataclass
class TextEvent:
    start_s: float
    duration_s: float
    text: str
    channel: str  # "ocr" | "transcript"


def extract_ocr(
    frames_1hz: np.ndarray,
    bin_starts_s: np.ndarray,
    *,
    strict: bool = False,
) -> list[TextEvent]:
    """
    Per-bin OCR for on-screen marketing hooks (non-critical).
    On failure: empty strings unless strict=True.
    """
    events: list[TextEvent] = []
    try:
        import cv2  # noqa: F401
    except ImportError:
        if strict:
            msg = "opencv required for OCR in strict mode"
            raise RuntimeError(msg) from None
        logger.warning("ocr_skipped", extra={"reason": "opencv_unavailable"})
        for i, t0 in enumerate(bin_starts_s):
            events.append(
                TextEvent(
                    start_s=float(t0),
                    duration_s=1.0,
                    text="",
                    channel="ocr",
                )
            )
        return events

    try:
        import pytesseract
    except ImportError:
        if strict:
            msg = "pytesseract required for OCR in strict mode"
            raise RuntimeError(msg) from None
        logger.warning("ocr_skipped", extra={"reason": "pytesseract_unavailable"})
        for i, t0 in enumerate(bin_starts_s):
            events.append(
                TextEvent(
                    start_s=float(t0),
                    duration_s=1.0,
                    text="",
                    channel="ocr",
                )
            )
        return events

    if shutil.which("tesseract") is None:
        if strict:
            msg = "tesseract binary not found on PATH (install e.g. brew install tesseract)"
            raise RuntimeError(msg) from None
        logger.warning(
            "ocr_skipped",
            extra={
                "reason": "tesseract_binary_missing",
                "hint": "install system tesseract; macOS: brew install tesseract",
            },
        )
        for i, t0 in enumerate(bin_starts_s):
            events.append(
                TextEvent(
                    start_s=float(t0),
                    duration_s=1.0,
                    text="",
                    channel="ocr",
                )
            )
        return events

    for i, t0 in enumerate(bin_starts_s):
        if i >= frames_1hz.shape[0]:
            break
        frame = frames_1hz[i]
        try:
            # Video bins are often float32 [0,255]; PIL/Tesseract need uint8 BGR.
            fr = np.asarray(frame)
            if fr.dtype.kind == "f":
                g = np.clip(fr.astype(np.float64), 0.0, 255.0)
                if float(g.max()) <= 1.0 + 1e-6:
                    g = g * 255.0
                fr = np.clip(np.round(g), 0, 255).astype(np.uint8)
            else:
                fr = np.asarray(fr, dtype=np.uint8)
            if fr.ndim == 3 and fr.shape[-1] == 3:
                bgr = np.ascontiguousarray(fr[..., ::-1])
            else:
                bgr = np.ascontiguousarray(fr)
            txt = pytesseract.image_to_string(bgr, lang="eng") or ""
            txt = " ".join(txt.split())
        except Exception as e:
            if strict:
                raise
            logger.warning("ocr_frame_failed", extra={"bin": i, "error": str(e)})
            txt = ""
        events.append(
            TextEvent(
                start_s=float(t0),
                duration_s=1.0,
                text=txt,
                channel="ocr",
            )
        )
    return events


def merge_text_streams(
    ocr_events: list[TextEvent],
    transcript_events: list[TextEvent],
) -> list[TextEvent]:
    """
    Preserve separate channels; caller builds DataFrame rows.
    Conflict policy: transcript wins for dialogue time overlaps; OCR kept for visual-only.
    Here we simply concatenate with channel tags for downstream resolution.
    """
    return list(ocr_events) + list(transcript_events)


def ocr_events_to_dataframe_rows(events: list[TextEvent]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for e in events:
        # neuralset/tribev2 rejects Word events with empty text, and also expects
        # timeline metadata. Skip empty OCR rows rather than emitting invalid events.
        txt = (e.text or "").strip()
        if not txt:
            continue
        rows.append(
            {
                "type": "Word" if e.channel == "ocr" else "Word",
                "start": e.start_s,
                "duration": e.duration_s,
                "text": txt,
                "context": "",
                "channel": e.channel,
                "timeline": "default",
                "subject": "default",
            }
        )
    return rows
