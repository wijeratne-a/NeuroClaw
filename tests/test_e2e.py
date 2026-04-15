"""Smoke: CLI with mocks (no ffmpeg / tribev2 required)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

from neuroclaw.cli import app
from neuroclaw.extractor.audio import AudioFeatures


@pytest.mark.usefixtures("mock_env")
def test_cli_extract_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch video + audio paths; dummy input file path."""

    def fake_process_video(path: Path) -> dict:
        _ = path
        n = 10
        return {
            "probe": SimpleNamespace(
                duration_s=float(n), width=64, height=48, avg_fps=30.0
            ),
            "frames_2hz": np.zeros((n * 2, 48, 64, 3), dtype=np.uint8),
            "times_2hz": np.arange(n * 2, dtype=np.float64) * 0.5,
            "frames_1hz": np.zeros((n, 48, 64, 3), dtype=np.uint8),
            "bin_starts_s": np.arange(n, dtype=np.float64),
            "duration_s": float(n),
        }

    def fake_audio(path: Path, **kwargs: object) -> AudioFeatures:
        bs = kwargs.get("bin_starts_s")
        assert bs is not None
        arr = np.asarray(bs, dtype=np.float64)
        n = len(arr)
        return AudioFeatures(
            times_s=arr,
            features=np.zeros((n, 1024), dtype=np.float32),
        )

    monkeypatch.setattr("neuroclaw.cli.process_video", fake_process_video)
    monkeypatch.setattr("neuroclaw.cli.extract_audio_features", fake_audio)
    monkeypatch.setattr(
        "neuroclaw.cli.transcribe_clip",
        lambda *_a, **_k: {
            "text": "",
            "segments": [],
            "language": "unknown",
            "asr_model": "none",
            "temperature": 0.0,
        },
    )

    dummy = tmp_path / "dummy.mp4"
    dummy.write_bytes(b"")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--input",
            str(dummy),
            "--use-case",
            "commercial-content-optimization",
            "--out",
            str(tmp_path),
            "--no-strict-determinism",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    arts = list(tmp_path.glob("**/*.safetensors.zst"))
    assert len(arts) >= 1
