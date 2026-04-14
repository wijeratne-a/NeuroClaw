"""Video decode (ffmpeg PTS), 2 Hz decimation, 1 Hz closed-open binning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import ffmpeg
except ImportError:
    ffmpeg = None  # type: ignore[misc, assignment]


@dataclass
class VideoProbe:
    duration_s: float
    width: int
    height: int
    avg_fps: float


def probe_video(path: Path) -> VideoProbe:
    """Duration and dimensions from ffmpeg (PTS clock for sync)."""
    if ffmpeg is None:
        msg = "ffmpeg-python is required for video.probe_video"
        raise RuntimeError(msg)
    try:
        info = ffmpeg.probe(str(path))
    except ffmpeg.Error as e:
        err = (getattr(e, "stderr", None) or b"").decode("utf-8", errors="replace").strip()
        tail = err if err else "(no stderr from ffprobe)"
        raise RuntimeError(
            f"ffprobe failed for {path}: {tail} "
            "Install ffmpeg so `ffprobe` is on PATH (e.g. `brew install ffmpeg`). "
            "Confirm the file exists and contains a video stream."
        ) from e
    fmt = info.get("format", {})
    duration_s = float(fmt.get("duration", 0.0))
    vs = next(s for s in info.get("streams", []) if s.get("codec_type") == "video")
    width = int(vs.get("width", 0))
    height = int(vs.get("height", 0))
    afr = vs.get("avg_frame_rate", "30/1")
    if "/" in str(afr):
        num, den = str(afr).split("/", 1)
        avg_fps = float(num) / float(den) if float(den) else 30.0
    else:
        avg_fps = float(afr)
    return VideoProbe(duration_s=duration_s, width=width, height=height, avg_fps=avg_fps)


def decode_frames_2hz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode video to RGB frames at exactly 2 Hz using ffmpeg fps filter.
    Returns:
        frames: (T, H, W, 3) uint8
        times_s: (T,) float64 — center time of each frame in seconds (0, 0.5, 1, ...)
    """
    if ffmpeg is None:
        raise RuntimeError("ffmpeg-python is required")
    p = probe_video(path)
    if p.width <= 0 or p.height <= 0:
        msg = f"Invalid video dimensions: {path}"
        raise RuntimeError(msg)

    out, _ = (
        ffmpeg.input(str(path))
        .filter("fps", fps=2)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .run(capture_stdout=True, quiet=True)
    )
    frame_bytes = p.width * p.height * 3
    n = len(out) // frame_bytes
    if n == 0:
        msg = "No frames decoded at 2 Hz"
        raise RuntimeError(msg)
    frames_flat = np.frombuffer(out[: n * frame_bytes], dtype=np.uint8)
    frames = frames_flat.reshape((n, p.height, p.width, 3))
    # 2 Hz sample times: 0, 0.5, 1.0, ...
    times_s = np.arange(n, dtype=np.float64) * 0.5
    return frames, times_s


def bin_2hz_to_1hz(
    frames_2hz: np.ndarray,
    times_2hz: np.ndarray,
    duration_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mean-pool 2 Hz snapshots into 1 Hz bins [k, k+1) with weighted tail.

    Closed-open intervals [k, k+1) clipped to [0, duration).
    Last partial bin uses overlap-weighted mean of 2 Hz samples whose
    time centers fall in the clipped interval.
    """
    if frames_2hz.shape[0] != len(times_2hz):
        msg = "frames/times length mismatch"
        raise ValueError(msg)
    if duration_s <= 0:
        msg = "duration_s must be positive"
        raise ValueError(msg)

    n_bins = int(np.ceil(duration_s))
    if n_bins == 0:
        n_bins = 1
    h, w, c = frames_2hz.shape[1:]
    out = np.zeros((n_bins, h, w, c), dtype=np.float32)
    bin_starts = np.arange(n_bins, dtype=np.float64)

    for k in range(n_bins):
        t0 = float(k)
        t1 = min(float(k + 1), duration_s)
        if t0 >= duration_s:
            break
        if t1 <= t0:
            continue
        # 2 Hz samples at t in {0, 0.5, ...} with value covering half-second window;
        # include sample if its center t is in [t0, t1)
        mask = (times_2hz >= t0) & (times_2hz < t1)
        idx = np.nonzero(mask)[0]
        if len(idx) == 0:
            continue
        # Weight by overlap of [t0,t1) with implicit sample support [t-0.25, t+0.25)
        weights = np.zeros(len(idx), dtype=np.float64)
        for j, i in enumerate(idx):
            tc = float(times_2hz[i])
            a, b = tc - 0.25, tc + 0.25
            ov = max(0.0, min(t1, b) - max(t0, a))
            weights[j] = ov
        wsum = weights.sum()
        if wsum <= 0:
            out[k] = frames_2hz[idx].mean(axis=0).astype(np.float32)
        else:
            wn = weights / wsum
            out[k] = np.tensordot(wn, frames_2hz[idx].astype(np.float32), axes=(0, 0))

    bin_times = bin_starts[:n_bins]
    return out, bin_times


def process_video(path: Path) -> dict[str, Any]:
    """Full pipeline: probe, 2 Hz frames, 1 Hz bins. Primary clock = ffmpeg duration."""
    path = Path(path)
    probe = probe_video(path)
    frames_2hz, times_2hz = decode_frames_2hz(path)
    dur = min(probe.duration_s, float(times_2hz[-1] + 0.5) if len(times_2hz) else 0.0)
    if probe.duration_s > 0:
        dur = probe.duration_s
    frames_1hz, bin_starts = bin_2hz_to_1hz(frames_2hz, times_2hz, dur)
    return {
        "probe": probe,
        "frames_2hz": frames_2hz,
        "times_2hz": times_2hz,
        "frames_1hz": frames_1hz,
        "bin_starts_s": bin_starts,
        "duration_s": dur,
    }
