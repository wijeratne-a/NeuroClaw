"""TRIBE wrapper: normalize_prediction_to_ot and mock cortical-only output."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neuroclaw.model.tribe_wrapper import CORTICAL_DIM, normalize_prediction_to_ot


def test_normalize_prediction_to_ot_time_major() -> None:
    t, o = 4, CORTICAL_DIM
    arr = np.random.default_rng(0).standard_normal((t, o)).astype(np.float32)
    out = normalize_prediction_to_ot(arr, CORTICAL_DIM)
    assert out.shape == (o, t)
    assert out.dtype == np.float16


def test_normalize_prediction_to_ot_voxel_major() -> None:
    o, t = CORTICAL_DIM, 3
    arr = np.random.default_rng(1).standard_normal((o, t)).astype(np.float32)
    out = normalize_prediction_to_ot(arr, CORTICAL_DIM)
    assert out.shape == (o, t)


def test_normalize_prediction_to_ot_wrong_dim_raises() -> None:
    arr = np.zeros((10, 100), dtype=np.float32)
    with pytest.raises(RuntimeError, match="Expected one axis"):
        normalize_prediction_to_ot(arr, CORTICAL_DIM)


def test_mock_predict_returns_cortical_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEUROCLAW_USE_MOCK_TRIBE", "1")
    from neuroclaw.model.tribe_wrapper import load_tribe

    m = load_tribe(device="cpu", hf_token=None)
    df = pd.DataFrame(
        {
            "type": ["Video", "Audio"],
            "start": [0.0, 0.0],
            "duration": [5.0, 5.0],
            "filepath": ["/tmp/x.mp4", "/tmp/x.wav"],
            "text": ["", ""],
            "context": ["", ""],
        }
    )
    preds, _ = m.predict(events=df)
    p = np.asarray(preds)
    assert p.ndim == 2
    assert p.shape[1] == CORTICAL_DIM
