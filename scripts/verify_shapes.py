#!/usr/bin/env python3
"""
Local smoke test: assert TRIBE single-pass native output dim is exactly 20484.

Requires HF_TOKEN, tribev2 installed, and sufficient RAM. Not run in CI.
"""

from __future__ import annotations

import gc
import os
import sys


def _need_ram() -> None:
    try:
        import psutil
    except ImportError:
        return
    total_gb = psutil.virtual_memory().total / (1024.0**3)
    if total_gb < 15.5:
        sys.exit(
            "FAIL: verify_shapes.py requires >= 16GB Unified Memory (Mac) or 24GB VRAM (NVIDIA). "
            "If this crashed with 'Killed' or 'RuntimeError: MPS backend out of memory', "
            "it is an expected hardware OOM, not a codebase bug."
        )


def _clear_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass


def main() -> None:
    _need_ram()
    if not os.environ.get("HF_TOKEN", "").strip():
        sys.exit("FAIL: set HF_TOKEN for gated weights.")

    try:
        import pandas as pd

        from tribev2.demo_utils import TribeModel
    except ImportError as e:
        sys.exit(f"FAIL: import error: {e}")

    try:
        events = pd.DataFrame(
            [
                {
                    "type": "Video",
                    "start": 0.0,
                    "duration": 5.0,
                    "filepath": "",
                    "text": "",
                    "context": "",
                    "subject_id": None,
                },
                {
                    "type": "Audio",
                    "start": 0.0,
                    "duration": 5.0,
                    "filepath": "",
                    "text": "",
                    "context": "",
                    "subject_id": None,
                },
            ]
        )
    except Exception as e:
        sys.exit(f"FAIL: {e}")

    device = os.environ.get("NEUROCLAW_DEVICE", "cpu")

    print(f"Loading facebook/tribev2 on {device}...")
    try:
        model = TribeModel.from_pretrained("facebook/tribev2", device=device)
        model.remove_empty_segments = False
        preds, _ = model.predict(events=events)
        if hasattr(preds, "values"):
            preds = preds.values
        arr = __import__("numpy").asarray(preds)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim != 2:
            sys.exit(f"FAIL: expected 2D preds, got {arr.shape}")
        o20484 = 20484
        if o20484 in (arr.shape[0], arr.shape[1]):
            print("SHAPES OK (20484, T) — Ready for Destrieux slicing.")
        else:
            sys.exit(f"FAIL: Expected one axis == 20484, got shape {arr.shape}")
    except Exception as e:
        sys.exit(f"FAIL: {e}")
    finally:
        _clear_cache()


if __name__ == "__main__":
    main()
