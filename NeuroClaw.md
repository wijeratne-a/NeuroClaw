# NeuroClaw

## Technical Foundation of the Visual Voxel Micro-Engine for Neuro-Align

## 1) Vision and Context

At the end of March 2026, Meta FAIR released TRIBE v2 (TRansformer for In-silico Brain Experiments), establishing a new baseline for predictive computational neuroscience. TRIBE v2 maps multimodal stimuli to approximately 70,000 cortical and subcortical voxels, representing a major increase in spatial resolution relative to prior versions.

Neuro-Align builds on this by creating the "Visual Voxel" Micro-Engine: a distilled local inference system that reduces an approximately 70GB TRIBE v2 class architecture (built on a V-JEPA2-style backbone) into a high-precision approximately 2GB runtime target suitable for 16GB machines.

The strategic goal is to move optimization away from lagging behavioral metrics (for example, click-through rate) toward leading biological indicators of intent and reward dynamics.

## 2) Phase 1: High-Compute Synthetic Brain Generation

### 2.1 Compute Prerequisite

A 16GB local system cannot host full TRIBE v2 inference and training workloads (typically requiring about 28-32GB VRAM class capacity for practical throughput). The first step is therefore a high-compute "Synthetic Brain" label generation pass to support student-teacher distillation.

### 2.2 GPU Pod Orchestration and Economics

- Target run: 8x H100 (SXM5) pod for a 24-hour labeling blitz.
- Reference pricing (as cited): Thunder Compute on-demand at about $1.38 per GPU-hour (about $11.04 per node-hour for 8 GPUs).
- Why this matters: H100-class memory bandwidth is needed to process V-JEPA2 spatial masking and latent extraction at usable throughput.

### 2.3 "Diverse Marketing Diet" Corpus Design

To avoid domain drift and preserve biological validity, corpus construction combines:

- Scientific floor:
  - Natural Scenes Dataset (NSD)
  - Algonauts 2025 naturalistic movie data
- Product ceiling:
  - Raw TikToks, UGC, and ad creatives
  - Labeled through TRIBE v2-scale inference to generate synthetic voxel targets

Target artifact: roughly 100GB of `safetensors` mapping:

`Marketing video frames -> 70,000-voxel vectors`

## 3) Phase 2: Mixed-Precision Distillation to a 2GB Micro-Engine

### 3.1 Distillation Principle

The architecture is split to avoid regression-quality collapse during aggressive quantization:

- Backbone path:
  - Distill V-JEPA2-Giant-like representation learning into a smaller student (for example DINOv2-Small class)
  - Quantize to 4-bit (q4_K_M) with Apple MLX-native kernels
- Regression heads:
  - Keep final voxel/engagement prediction heads in FP16
  - Preserve precision for reward-sensitive regions (for example NAcc dynamics)

This hybrid policy is designed to avoid the "quantization trap" where low-bit noise smears scalar regression outputs.

### 3.2 SALT Objective

Neuro-Align training uses Static-teacher Asymmetric Latent Training (SALT), minimizing distance between student predictions and stop-gradient teacher latents:

`L_latent = E_{x,y}[ || g_phi(f_theta(x), delta_y) - stop_grad(f_psi(y)) ||_1 ]`

Operational intent:

- Compute-efficient teacher supervision
- Stable latent transfer
- Reduced representational collapse risk

### 3.3 Memory Orchestration for 16GB Hardware

To stay within a strict local memory budget:

1. Load the ~1GB vision engine first (MLX-native) for frame inference.
2. Use unified KV caching to reduce multimodal stage latency.
3. Load the quantized advice LLM (for example Phi-4 or Llama-3.2 class) only when interpretation is requested.

This staged policy prevents process termination under constrained RAM/VRAM conditions.

## 4) Phase 2 Science Moat: Biological and Spatial Fidelity

### 4.1 pRF Foveal Bias Modeling

Population receptive field (pRF) modeling estimates voxel response with an isotropic Gaussian over visual space:

`r(t) = sum_{x,y} s(x,y,t) * g(x,y)`

Purpose:

- Verify biologically coherent spatial behavior
- Ensure region-selective activations (for example face salience near fovea for FFA-like responses)

### 4.2 Dopamine Proxy via Future-Past Distillation

A student constrained to past frames predicts a teacher informed by future context. The mismatch acts as a proxy for Reward Prediction Error (RPE), approximating novelty/expectation-violation signals in reward-linked circuitry (for example NAcc-like targets).

## 5) Phase 3: Explainable AI Advice Layer

### 5.1 Regression Attribution

The system adapts Grad-CAM to scalar engagement outputs (for example predicted BOLD-like reward index) rather than class logits.

Output:

- High-resolution saliency maps showing which pixels and motion patterns drive predicted biological responses.

### 5.2 "Creative Director" Auto-Fix Layer

A quantized local LLM converts saliency and trajectory diagnostics into specific editorial actions (timing, pacing, framing, motion guidance), such as reducing pan speed at critical moments to increase reward anticipation.

## 6) Real-Time Frontend and Interaction Model

### 6.1 60FPS Brain Visualization Pipeline

- Three.js rendering thread for pulsing 3D brain view
- Inference worker for model execution
- SharedArrayBuffer + Atomics for zero-copy voxel transfer between worker and renderer

This keeps UI fluid under high data throughput.

### 6.2 Temporal Rewind ("Hemodynamic Lag" Compensation)

Because biological hemodynamic response is delayed (around 5 seconds), the UI rewinds displayed neural pulses to align with the user-selected frame timestamp. This makes feedback feel frame-synchronous instead of biologically delayed.

## 7) Regulatory and Ethical Moats

### 7.1 EU AI Act Alignment (April 2026 framing)

- Positioning: "Commercial Content Optimizer"
- Article 5(1)(f) boundary: avoid prohibited workplace/education emotion AI contexts
- Article 50: include transparency labels disclosing emotion-recognition usage

### 7.2 SaMD-Oriented Governance Posture

Maintain a Predetermined Change Control Plan (PCCP)-style documentation model describing local adaptation behavior and update controls. Even in commercial deployment, this creates auditability and regulatory credibility.

## 8) Market Transformation Thesis

NeuroClaw operationalizes a shift:

- From: demographic and click/like optimization
- To: cognitive-state and neural-feedback optimization

Illustrative metric transition:

- Engagement: Clicks/Likes -> Brainwave synchrony and reward dynamics
- Personalization: Demographic-based -> Cognitive state-based
- Optimization loop: A/B tests -> Neural feedback loops
- Commercial outcome: Transactional conversion -> Emotional trust index

## 9) Core Deliverable Definition

The Visual Voxel Micro-Engine is the technical base for "market-of-one" creative optimization:

- Quantifies memory encoding peaks
- Tracks reward anticipation proxies
- Produces explainable, frame-level editing guidance
- Runs locally on constrained hardware through mixed precision and staged orchestration

In effect, NeuroClaw reframes marketing analytics as intention-aware neurocomputational inference.

