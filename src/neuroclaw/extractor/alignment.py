"""PTS-anchored drift audit between video, audio, and text event times."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DriftStats:
    max_abs_drift_ms: float
    p95_drift_ms: float
    per_pair: dict[str, float]


def _pairwise_drift_ms(a: np.ndarray, b: np.ndarray) -> float:
    """Min mean abs delta after aligning same-length samples by sorting."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    a = np.sort(np.asarray(a, dtype=np.float64).ravel())
    b = np.sort(np.asarray(b, dtype=np.float64).ravel())
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    d = np.abs(a[:n] - b[:n]) * 1000.0
    return float(np.mean(d))


def audit_drift(
    video_bin_starts_s: np.ndarray,
    audio_bin_starts_s: np.ndarray,
    events_df: pd.DataFrame | None,
) -> DriftStats:
    """
    Compare modality timestamps; fail thresholds in validator / CLI if exceeded.

    max_abs_drift_ms: max deviation between aligned video vs audio bin starts.
    p95_drift_ms: 95th percentile of |v - a| in ms for overlapping bins.
    """
    v = np.asarray(video_bin_starts_s, dtype=np.float64).ravel()
    a = np.asarray(audio_bin_starts_s, dtype=np.float64).ravel()
    n = min(len(v), len(a))
    if n == 0:
        return DriftStats(
            max_abs_drift_ms=0.0,
            p95_drift_ms=0.0,
            per_pair={"video_audio_max_ms": 0.0},
        )
    delta_ms = np.abs(v[:n] - a[:n]) * 1000.0
    max_abs = float(np.max(delta_ms))
    p95 = float(np.percentile(delta_ms, 95))
    per_pair: dict[str, float] = {
        "video_audio_max_ms": max_abs,
        "video_audio_p95_ms": p95,
    }

    if events_df is not None and len(events_df) > 0 and "start" in events_df.columns:
        ev_t = events_df["start"].astype(np.float64).values
        if len(ev_t) and len(v):
            idx = np.searchsorted(v, ev_t, side="right") - 1
            idx = np.clip(idx, 0, len(v) - 1)
            nearest = v[idx]
            d_ev = np.abs(ev_t - nearest) * 1000.0
            per_pair["text_video_mean_ms"] = float(np.mean(d_ev))
            per_pair["text_video_p95_ms"] = float(np.percentile(d_ev, 95))
            # Critical gates: video vs audio only (text logged for audit)

    return DriftStats(
        max_abs_drift_ms=max_abs,
        p95_drift_ms=p95,
        per_pair=per_pair,
    )


def assert_drift_limits(stats: DriftStats) -> None:
    """Raise if critical alignment thresholds violated."""
    if stats.max_abs_drift_ms > 50.0:
        msg = f"max_abs_drift_ms {stats.max_abs_drift_ms} > 50ms"
        raise RuntimeError(msg)
    if stats.p95_drift_ms >= 20.0:
        msg = f"p95_drift_ms {stats.p95_drift_ms} >= 20ms"
        raise RuntimeError(msg)
