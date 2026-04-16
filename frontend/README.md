# NeuroClaw — Cortical Marketing Intelligence (UI)

Local viewer: drag NeuroClaw `*.safetensors.zst` (+ optional `*_transcript.json`), choose a video, and scrub. Parsing runs in a Web Worker (`fzstd` + custom safetensors reader + float16→float32).

## Commands

```bash
cd frontend
npm install
npm run dev
```

Build: `npm run build`, preview: `npm run preview`.

## Phase 3 visuals

- **3D brain:** `public/brain.glb` — Khronos [BrainStem](https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0/BrainStem) glTF sample (anatomical mesh; ROI glow epicentres are **heuristic** fractional offsets in the mesh bounding box, not Destrieux-registered vertices).
- **ROI glow:** Gaussian falloff in a patched `MeshStandardMaterial` fragment shader (`brainShaderMaterial.ts`).
- **Semantic HUD:** Threshold toasts on normalized ROI values (marketing copy; not clinical claims).
- **Dev mock metrics:** append `?dev=1` to the URL to show the old 1Hz synthetic series toggle.

## Golden demo (optional auto-load)

Place these files under `public/demo/` (large binaries are gitignored; copy them locally after clone):

| File | Role |
|------|------|
| `demo_clip.mp4` | Video |
| `demo_voxels.safetensors.zst` | Artifact |
| `demo_transcript.json` | Transcript |

If all three exist, the app loads them on startup. Manual file picks cancel the auto-load.

## Inputs

- One or more `{clip}_voxels.safetensors.zst` (or `*_partNN` chunks); merged by `chunk_index`.
- Optional `{clip}_transcript.json` (same schema as NeuroClaw ASR output).
- Video file for playback (local blob URL).

The `<video>` element is always mounted so the sync hook gets a stable ref. By default, neural time follows **video playback time** (stimulus-aligned). Check **Apply hemodynamic delay (HRF)** to use `currentTime − hemodynamic_offset_s` from the artifact. ROI values are **min–max normalized** per run for the 3D shader and HUD.
