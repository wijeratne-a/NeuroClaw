# NeuroClaw Phase 2 — Marketing Four UI

Local-only viewer: drag NeuroClaw `*.safetensors.zst` (+ optional `*_transcript.json`), choose a video, and scrub. Parsing runs in a Web Worker (`fzstd` + custom safetensors reader + float16→float32).

## Commands

```bash
cd frontend
npm install
npm run dev
```

Build: `npm run build`, preview: `npm run preview`.

## Inputs

- One or more `{clip}_voxels.safetensors.zst` (or `*_partNN` chunks); merged by `chunk_index`.
- Optional `{clip}_transcript.json` (same schema as NeuroClaw ASR output).
- Video file for playback (local blob URL).

Toggle **Demo 1Hz mock metrics** to exercise sync without an artifact.

The `<video>` element is always mounted so the sync hook gets a stable ref. By default, neural time follows **video playback time** (stimulus-aligned). Check **Apply hemodynamic delay (HRF)** to use `currentTime − hemodynamic_offset_s` from the artifact. ROI values are **min–max normalized** per run for the 3D brain so small z-scored tensors still read as visible “pulse.”
