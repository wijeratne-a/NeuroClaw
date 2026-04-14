"""Git SHA for reproducibility manifest; 'unknown' outside a git tree."""

from __future__ import annotations

import subprocess
from pathlib import Path


def get_git_sha(cwd: Path | None = None) -> str:
    """Return short git SHA or 'unknown'."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd or Path.cwd(),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0 and out.stdout:
            return out.stdout.strip()[:40]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "unknown"
