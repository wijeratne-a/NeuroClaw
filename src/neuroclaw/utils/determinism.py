"""Reproducibility: seed 42, optional strict PyTorch deterministic algorithms."""

from __future__ import annotations

import os
import random

import numpy as np


def set_deterministic(seed: int = 42, *, strict: bool = True) -> None:
    """Set seeds; enable torch deterministic kernels when strict=True."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # cuDNN deterministic (may reduce throughput on Hopper)
        torch.backends.cudnn.deterministic = strict
        torch.backends.cudnn.benchmark = not strict
        if strict:
            try:
                torch.use_deterministic_algorithms(True, warn_only=False)
            except Exception:
                torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
