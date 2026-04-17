# NeuroClaw

Visual Voxel Micro-Engine — **Stage 1** multimodal extractor: decode video/audio/text, run [TRIBE v2](https://huggingface.co/facebook/tribev2) inference on the public **fsaverage5 (20,484-vertex)** cortical head, pool **four Destrieux ROIs** (FFA, vmPFC, IFG, Insula) to `.safetensors.zst` with compliance metadata and a `transcript.json` sidecar (faster-whisper, temp=0).

See [NeuroClaw.md](NeuroClaw.md) for product context.

## Requirements

- **Python 3.11+**
- **ffmpeg** on `PATH` (decode `.mp4` / `.mkv`)
- **Hugging Face token** for gated weights (`HF_TOKEN` or `HUGGINGFACEHUB_API_TOKEN`)

## Install

```bash
pip install -e ".[dev]"
# Optional: real TRIBE package (when available from your index)
pip install -e ".[tribev2]"
```

Copy [.env.example](.env.example) to `.env` and set `HF_TOKEN`.

## CLI

```bash
neuroclaw-extract \
  --input path/to/clip.mp4 \
  --use-case commercial-content-optimization \
  --out artifacts
```

- **Allowlist** (default): `commercial_content_optimization`, `commercial-optimization`, `marketing-optimization`, `creative-testing`, `brand-engagement-analysis`.
- **Blocked** contexts exit with code **64** (EU AI Act guardrails).
- **Mock mode** (no `tribev2` / offline): `export NEUROCLAW_USE_MOCK_TRIBE=1`
- **Placeholder atlas** (CI): `export NEUROCLAW_PLACEHOLDER_ATLAS=1`

## Layout

- `frontend/` — Phase 2 React + Vite viewer for `.safetensors.zst`, transcript, and video (`npm run dev` inside `frontend/`)
- `scripts/export_brain.py` — exports Nilearn fsaverage5 pial surface to `frontend/public/brain.glb` (requires `trimesh`, `nilearn`)
- `src/neuroclaw/` — package (`neuroclaw-extract` entrypoint)
- `src/neuroclaw/inference_wrapper.py` — programmatic TRIBE load/predict re-exports
- `Dockerfile` — `dev-cpu` and `gpu-runtime` targets

## Tests

```bash
PYTHONPATH=src pytest tests/
```

## Regulatory

Outputs include `transparency_label` and `use_case_audit` in `model_metadata`; see `neuroclaw.compliance` for notices.
