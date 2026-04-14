"""Events builder compatibility and fallback behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from neuroclaw.model.events_builder import build_events_df


class _ModelWhisperxFailure:
    def get_events_dataframe(self, *, video_path: str) -> pd.DataFrame:
        raise RuntimeError("whisperx failed: uvx dependency mismatch")


def test_build_events_df_falls_back_on_whisperx_failure(tmp_path: Path) -> None:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    model = _ModelWhisperxFailure()
    df = build_events_df(model, video)
    assert len(df) >= 2
    assert {"Video", "Audio"}.issubset(set(df["type"].astype(str).tolist()))
    assert "start" in df.columns
    assert "duration" in df.columns
    assert "timeline" in df.columns
    assert "subject" in df.columns
    assert set(df["timeline"].astype(str).tolist()) == {"default"}
    assert set(df["subject"].astype(str).tolist()) == {"default"}


def test_build_events_df_fallback_prefers_sidecar_wav(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    sidecar = tmp_path / "clip.wav"
    sidecar.write_bytes(b"")
    model = _ModelWhisperxFailure()
    df = build_events_df(model, video)
    audio = df[df["type"].astype(str) == "Audio"].iloc[0]
    assert str(audio["filepath"]) == str(sidecar)


def test_build_events_df_fallback_appends_local_asr_words(monkeypatch: Any, tmp_path: Path) -> None:
    from neuroclaw.model import events_builder as eb

    video = tmp_path / "v.mp4"
    video.write_bytes(b"")

    def _fake_words(p: Path) -> list:
        return [
            {
                "type": "Word",
                "start": 0.1,
                "duration": 0.2,
                "text": "hello",
                "context": "",
                "filepath": "",
                "channel": "transcript",
                "timeline": "default",
                "subject": "default",
            }
        ]

    monkeypatch.setattr(eb, "word_rows_from_wav", _fake_words)
    model = _ModelWhisperxFailure()
    df = build_events_df(model, video)
    words = df[df["type"].astype(str) == "Word"]
    assert len(words) == 1
    assert str(words.iloc[0]["text"]) == "hello"


class _ModelNumpyMismatch:
    def get_events_dataframe(self, *, video_path: str) -> pd.DataFrame:
        raise RuntimeError(
            "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.0.2"
        )


def test_build_events_df_fallback_on_numpy_mismatch(tmp_path: Path) -> None:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    df = build_events_df(_ModelNumpyMismatch(), video)
    assert len(df) >= 2
    assert "Video" in set(df["type"].astype(str).tolist())

