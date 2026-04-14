"""Wav2Vec-BERT 2.0 (facebook/w2v-bert-2.0) — Layer 24 hidden states, 1 Hz pooling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import ffmpeg
except ImportError:
    ffmpeg = None  # type: ignore[misc, assignment]


@dataclass
class AudioFeatures:
    """Per 1 Hz bin: mean-pooled last-hidden (layer 24) vector."""

    times_s: np.ndarray  # (T,) bin starts
    features: np.ndarray  # (T, D) float32


def _extract_wav_mono(path: Path, sample_rate: int = 16000) -> tuple[np.ndarray, float]:
    """PCM mono float32 via ffmpeg; duration from probe."""
    if ffmpeg is None:
        raise RuntimeError("ffmpeg-python required for audio extraction")
    info = ffmpeg.probe(str(path))
    dur = float(info.get("format", {}).get("duration", 0.0))
    out, _ = (
        ffmpeg.input(str(path))
        .output(
            "pipe:",
            format="f32le",
            acodec="pcm_f32le",
            ac=1,
            ar=sample_rate,
        )
        .run(capture_stdout=True, quiet=True)
    )
    audio = np.frombuffer(out, dtype=np.float32)
    return audio, dur


def extract_audio_features_1hz(
    video_path: Path,
    *,
    device: str | None = None,
    bin_starts_s: np.ndarray | None = None,
    duration_s: float | None = None,
) -> AudioFeatures:
    """
    Load Wav2Vec-BERT 2.0, run on full waveform, pool hidden states to 1 Hz bins.

    Uses Hugging Face transformers; layer index 24 (0-based) last hidden state.
    Unloads model from GPU/CPU after extraction when possible.
    """
    from transformers import AutoFeatureExtractor, AutoModel

    video_path = Path(video_path)
    wav, dur = _extract_wav_mono(video_path)
    if duration_s is not None:
        dur = duration_s
    sr = 16000
    if bin_starts_s is None:
        n_bins = int(np.ceil(dur)) if dur > 0 else 1
        bin_starts_s = np.arange(n_bins, dtype=np.float64)
    else:
        n_bins = len(bin_starts_s)

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Use auto classes so HF resolves the correct architecture for w2v-bert checkpoints.
    feat_ext = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    model = AutoModel.from_pretrained("facebook/w2v-bert-2.0")
    model.eval()
    model.to(dev)

    # Wav2Vec-BERT outputs last_hidden_state (B, T_frames, D)
    with torch.no_grad():
        inputs = feat_ext(wav, sampling_rate=sr, return_tensors="pt", padding=True)
        model_inputs = {
            k: v.to(dev)
            for k, v in inputs.items()
            if isinstance(v, torch.Tensor)
        }
        out = model(**model_inputs, output_hidden_states=True)
        # Layer 24 (last encoder): prefer index 24 if present, else last layer
        hs = out.hidden_states
        if hs is not None and len(hs) > 24:
            hidden = hs[24]
        elif hs is not None:
            hidden = hs[-1]
        else:
            hidden = out.last_hidden_state
        # hidden: (1, T', D)
        h = hidden.squeeze(0).cpu().numpy()
        frame_times = np.linspace(0.0, dur, num=h.shape[0], endpoint=False)

    # Pool to 1 Hz: for each bin [k, k+1), mean of frames in [k, min(k+1, dur))
    D = h.shape[1]
    pooled = np.zeros((n_bins, D), dtype=np.float32)
    for k in range(n_bins):
        t0 = float(bin_starts_s[k]) if k < len(bin_starts_s) else float(k)
        t1 = min(t0 + 1.0, dur)
        if t0 >= dur:
            break
        m = (frame_times >= t0) & (frame_times < t1)
        if m.any():
            pooled[k] = h[m].mean(axis=0)
        elif k < h.shape[0]:
            pooled[k] = h[min(k, h.shape[0] - 1)]

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return AudioFeatures(times_s=np.asarray(bin_starts_s[:n_bins], dtype=np.float64), features=pooled)


def extract_audio_features(path: Path, **kwargs: Any) -> AudioFeatures:
    """Alias for staged loader."""
    return extract_audio_features_1hz(path, **kwargs)
