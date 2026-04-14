"""Dual-pass orchestration with mock TRIBE."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neuroclaw.atlas.dual_pass_rois import build_dual_pass_roi_map
from neuroclaw.config import load_settings
from neuroclaw.model.dual_pass import run_dual_pass


@pytest.fixture
def minimal_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "type": "Video",
                "start": 0.0,
                "duration": 5.0,
                "filepath": "/tmp/x.mp4",
                "text": "",
                "context": "",
                "subject_id": None,
            },
            {
                "type": "Audio",
                "start": 0.0,
                "duration": 5.0,
                "filepath": "/tmp/x.wav",
                "text": "",
                "context": "",
                "subject_id": None,
            },
        ]
    )


def test_run_dual_pass_mock_shapes(minimal_events: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEUROCLAW_USE_MOCK_TRIBE", "1")
    monkeypatch.setenv("NEUROCLAW_PLACEHOLDER_ATLAS", "1")
    roi = build_dual_pass_roi_map()
    bs = np.arange(5, dtype=np.float64)
    dp = run_dual_pass(
        minimal_events,
        5.0,
        bs,
        roi,
        load_settings(),
        "run-uuid",
        "cpu",
        None,
        mock_native_o=(8802, 20484),
    )
    assert dp.nacc.shape == (5,)
    assert dp.amygdala.shape == (5,)
    assert dp.ffa.shape == (5,)
    assert dp.vmpfc.shape == (5,)
    assert dp.timestamps.shape == (5,)
