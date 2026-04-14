#!/usr/bin/env python3
"""
Local smoke test: assert TRIBE dual-pass native output dims (8802 and 20484).

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

        from neuroclaw.model.tribe_wrapper import (
            CORTICAL_CONFIG_UPDATE,
            SUBCORTICAL_CONFIG_UPDATE,
            load_tribe,
            normalize_prediction_to_ot,
        )
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

    m1 = load_tribe(device=device, hf_token=os.environ.get("HF_TOKEN"), config_override=SUBCORTICAL_CONFIG_UPDATE)
    try:
        p1, _ = m1.predict(events=events)
        o1 = normalize_prediction_to_ot(p1, 8802)
        assert o1.shape[0] == 8802, o1.shape
    finally:
        del m1
        _clear_cache()

    m2 = load_tribe(device=device, hf_token=os.environ.get("HF_TOKEN"), config_override=CORTICAL_CONFIG_UPDATE)
    try:
        p2, _ = m2.predict(events=events)
        o2 = normalize_prediction_to_ot(p2, 20484)
        assert o2.shape[0] == 20484, o2.shape
    finally:
        del m2
        _clear_cache()

    if o1.shape[1] != o2.shape[1]:
        sys.exit(f"FAIL: time mismatch T_sub={o1.shape[1]} T_cort={o2.shape[1]}")

    print("SHAPES OK", o1.shape, o2.shape)


if __name__ == "__main__":
    main()
