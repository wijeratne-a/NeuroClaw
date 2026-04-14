"""local_asr helpers (no model download)."""

from __future__ import annotations

from pathlib import Path

from neuroclaw.model.local_asr import word_rows_from_wav


def test_word_rows_missing_file_returns_empty() -> None:
    assert word_rows_from_wav(Path("/nonexistent/neuroclaw_asr_test.wav")) == []
