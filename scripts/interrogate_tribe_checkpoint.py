#!/usr/bin/env python3
"""
Inspect a tribev2 / Tribe PyTorch checkpoint without opening it in chat.

Prints ``model_build_args`` (especially ``n_outputs``) and tensor shapes for
likely output / predictor layers in ``state_dict``.

Usage::

    python scripts/interrogate_tribe_checkpoint.py path/to/best.ckpt

Typical Hugging Face cache layout::

    ~/.cache/huggingface/hub/models--facebook--tribev2/snapshots/<hash>/best.ckpt

Requires ``torch`` (NeuroClaw declares ``torch>=2.1``). Run with the **same
Python** you use for ``neuroclaw-extract`` / ``pip install -e ".[dev]"`` —
conda ``(base)`` often has no PyTorch unless you installed it there.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_TORCH_IMPORT_ERROR = """\
No module named 'torch' — this script needs PyTorch.

Use the same environment where NeuroClaw runs (the one that can `import torch`), e.g.:
  cd /path/to/NeuroClaw && pip install -e ".[dev]"

Or install PyTorch into the current interpreter:
  pip install 'torch>=2.1'

Then verify:
  python -c "import torch; print(torch.__version__)"
"""


def _load_ckpt(ckpt_path: Path) -> dict:
    import torch

    kwargs: dict = {"map_location": "cpu", "weights_only": True}
    try:
        return torch.load(ckpt_path, mmap=True, **kwargs)  # type: ignore[arg-type]
    except TypeError:
        return torch.load(ckpt_path, **kwargs)


def interrogate_checkpoint(ckpt_path: Path) -> None:
    import torch

    if not ckpt_path.is_file():
        sys.exit(f"Not a file: {ckpt_path}")

    print(f"Loading {ckpt_path} (CPU, mmap if supported)...")
    ckpt = _load_ckpt(ckpt_path)

    if not isinstance(ckpt, dict):
        sys.exit(f"Expected checkpoint dict, got {type(ckpt)}")

    print("\n--- TOP-LEVEL KEYS ---")
    for k in sorted(ckpt.keys()):
        print(k)

    print("\n--- MODEL BUILD ARGS ---")
    build_args = ckpt.get("model_build_args", {})
    if not build_args:
        print("(missing or empty model_build_args)")
    else:
        for k in sorted(build_args.keys()):
            print(f"{k}: {build_args[k]}")
        if "n_outputs" in build_args:
            print(f"\n>>> n_outputs = {build_args['n_outputs']}")

    print("\n--- STATE_DICT: keys containing predictor / head / proj / output / mask / linear ---")
    state_dict = ckpt.get("state_dict", {})
    if not state_dict:
        print("(missing or empty state_dict)")
        return

    needles = ("predictor", "head", "proj", "output", "mask", "linear")
    matched: list[tuple[str, torch.Size]] = []
    for key, tensor in state_dict.items():
        if hasattr(tensor, "shape") and any(n in key.lower() for n in needles):
            matched.append((key, tensor.shape))
    matched.sort(key=lambda x: x[0])

    if not matched:
        print("No keys matched; last 40 keys in state_dict:")
        for k in list(state_dict.keys())[-40:]:
            t = state_dict[k]
            sh = getattr(t, "shape", None)
            print(f"  {k}: {sh}")
    else:
        for k, sh in matched:
            print(f"{k}: {sh}")

    print("\n--- LARGER 2D WEIGHTS (often final layers; top 30 by element count) ---")
    big2d: list[tuple[str, torch.Size]] = []
    for key, tensor in state_dict.items():
        if hasattr(tensor, "shape") and len(tensor.shape) == 2:
            big2d.append((key, tensor.shape))
    big2d.sort(key=lambda x: -(x[1][0] * x[1][1]))
    for k, sh in big2d[:30]:
        print(f"{k}: {sh}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Print model_build_args and predictor-related tensor shapes from a .ckpt file."
    )
    p.add_argument(
        "ckpt",
        nargs="?",
        default="best.ckpt",
        type=Path,
        help="Path to checkpoint (default: ./best.ckpt)",
    )
    args = p.parse_args()
    try:
        import torch  # noqa: F401
    except ImportError:
        sys.exit(_TORCH_IMPORT_ERROR)
    interrogate_checkpoint(args.ckpt)


if __name__ == "__main__":
    main()
