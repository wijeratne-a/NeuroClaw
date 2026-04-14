# syntax=docker/dockerfile:1
# GPU runtime (H100 / A100 rehearsal): CUDA 12.1 + cuDNN 8
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS gpu-runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg ca-certificates curl git tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system -e ".[dev]"
RUN uv pip install --system \
    "git+https://github.com/facebookresearch/tribev2.git@72399081ed3f1040c4d996cefb2864a4c46f5b8e"
RUN python -c "from nilearn.datasets import fetch_atlas_surf_destrieux; fetch_atlas_surf_destrieux('fsaverage5')"

# Runtime: map HF + nilearn caches, e.g.:
# docker run -e HF_TOKEN=... \
#   -v ~/.cache/huggingface:/root/.cache/huggingface -e HF_HOME=/root/.cache/huggingface \
#   -v ~/.cache/nilearn:/root/nilearn_data -e NILEARN_DATA=/root/nilearn_data ...

CMD ["neuroclaw-extract", "--help"]

# --- CPU / Mac-style dev (no CUDA in image; MLX install optional) ---
FROM python:3.11-slim-bookworm AS dev-cpu
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg ca-certificates curl git tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system -e ".[dev]"
RUN uv pip install --system \
    "git+https://github.com/facebookresearch/tribev2.git@72399081ed3f1040c4d996cefb2864a4c46f5b8e"
RUN python -c "from nilearn.datasets import fetch_atlas_surf_destrieux; fetch_atlas_surf_destrieux('fsaverage5')"

# Runtime: map HF + nilearn caches (see gpu-runtime target comments).

CMD ["neuroclaw-extract", "--help"]
