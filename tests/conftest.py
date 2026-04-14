"""Pytest fixtures: env, optional tiny video."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def mock_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEUROCLAW_USE_MOCK_TRIBE", "1")
    monkeypatch.setenv("NEUROCLAW_PLACEHOLDER_ATLAS", "1")
    monkeypatch.setenv("HF_TOKEN", "test-token-for-ci")


@pytest.fixture
def tiny_mp4(tmp_path: Path) -> Path:
    """2s black video via lavfi (requires ffmpeg in PATH)."""
    out = tmp_path / "tiny.mp4"
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=64x48:d=2",
                "-pix_fmt",
                "yuv420p",
                str(out),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        pytest.skip(f"ffmpeg not available: {e}")
    return out


@pytest.fixture
def src_path() -> Path:
    return Path(__file__).resolve().parents[1] / "src"


@pytest.fixture
def pythonpath(src_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sys.path.insert(0, str(src_path.parent))
    if "PYTHONPATH" in os.environ:
        monkeypatch.setenv("PYTHONPATH", f"{src_path.parent}{os.pathsep}{os.environ['PYTHONPATH']}")
    else:
        monkeypatch.setenv("PYTHONPATH", str(src_path.parent))
