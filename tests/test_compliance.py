"""Compliance: allowlist, exit 64."""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_allowlist_accepts_commercial_content_optimization() -> None:
    from neuroclaw.compliance import validate_use_case

    assert validate_use_case("commercial_content_optimization") == "commercial_content_optimization"


def test_allowlist_accepts_hyphen_alias() -> None:
    from neuroclaw.compliance import validate_use_case

    assert validate_use_case("commercial-content-optimization") == "commercial_content_optimization"


def test_blocked_workplace_exits_64() -> None:
    import os
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    code = """
import sys
from neuroclaw.compliance import validate_use_case
try:
    validate_use_case("workplace surveillance")
except SystemExit as e:
    sys.exit(e.code)
"""
    env = {**os.environ, "PYTHONPATH": str(src)}
    r = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 64


def test_blocked_biometric_profiling_exits_64() -> None:
    import os
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    code = """
import sys
from neuroclaw.compliance import validate_use_case
try:
    validate_use_case("biometric profiling")
except SystemExit as e:
    sys.exit(e.code)
"""
    env = {**os.environ, "PYTHONPATH": str(src)}
    r = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 64


def test_blocked_wellbeing_monitoring_exits_64() -> None:
    import os
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    code = """
import sys
from neuroclaw.compliance import validate_use_case
try:
    validate_use_case("wellbeing monitoring")
except SystemExit as e:
    sys.exit(e.code)
"""
    env = {**os.environ, "PYTHONPATH": str(src)}
    r = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 64
