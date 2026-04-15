"""Single-pass Cortical Four orchestration."""

from __future__ import annotations

import numpy as np
import pandas as pd
from neuroclaw.model.single_pass import prepare_model_for_extract, run_cortical_marketing_four
from neuroclaw.model.tribe_wrapper import CORTICAL_DIM


class _Model:
    remove_empty_segments = True

    def predict(self, events=None, **kwargs):
        n = 5
        return np.ones((n, CORTICAL_DIM), dtype=np.float32), []


def test_prepare_model_clears_empty_segments() -> None:
    m = _Model()
    prepare_model_for_extract(m)
    assert m.remove_empty_segments is False


def test_run_cortical_marketing_four_shapes() -> None:
    m = _Model()
    df = pd.DataFrame(
        {
            "type": ["Video", "Audio"],
            "start": [0.0, 0.0],
            "duration": [5.0, 5.0],
            "filepath": ["/x.mp4", "/x.wav"],
            "text": ["", ""],
            "context": ["", ""],
        }
    )
    bs = np.arange(5, dtype=np.float64)
    roi = {
        "FFA": np.array([0, 1, 2], dtype=np.int64),
        "vmPFC": np.array([10, 11], dtype=np.int64),
        "IFG": np.array([100, 101], dtype=np.int64),
        "Insula": np.array([500], dtype=np.int64),
    }
    out = run_cortical_marketing_four(m, df, 5.0, bs, roi)
    assert out.ffa.shape == (5,)
    assert out.vmpfc.shape == (5,)
    assert out.ifg.shape == (5,)
    assert out.insula.shape == (5,)
    assert out.timestamps.shape == (5,)
    assert np.allclose(out.timestamps, bs)
