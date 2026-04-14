"""Drift audit thresholds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neuroclaw.extractor.alignment import DriftStats, assert_drift_limits, audit_drift


def test_audit_identical_bins_zero_drift() -> None:
    v = np.arange(10.0)
    s = audit_drift(v, v, pd.DataFrame())
    assert s.max_abs_drift_ms == 0.0
    assert s.p95_drift_ms == 0.0
    assert_drift_limits(s)


def test_audit_fails_over_50ms() -> None:
    v = np.zeros(5)
    a = np.ones(5) * 0.1  # 100ms
    s = audit_drift(v, a, None)
    assert s.max_abs_drift_ms > 50
    with pytest.raises(RuntimeError):
        assert_drift_limits(s)
