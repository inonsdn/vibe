# Model selection checklist

No model weights are selected, referenced, downloaded or required by this
repository. This document is what to evaluate when that changes.

The system is built so that choosing a model is an **integration**, not a
rewrite: a new `RendererBackend` subclass or a ComfyUI contract, plus an
`AnalysisAdapter` subclass per preprocessing component. No pipeline, API, CLI,
schema or QC code needs to change.

## Hard constraints

| Constraint | Consequence |
| --- | --- |
| RTX 5060, **8GB** VRAM | Assume the model does not fit. Require quantisation, tiling, frame windows and CPU/RAM offload. |
| 32GB system RAM | Offload has a budget; a model needing 40GB of host RAM is out. |
| Fully local | Weights must be downloadable **once, by you**, and run offline afterwards. No call-home licence checks. |
| Windows 11 | It must actually work on Windows. "Linux-only in practice" is a real failure mode for research code. |
| Identity preservation | The model influences only the garment region. Anything that regenerates a whole person is the wrong shape for this system. |

## The garment renderer

This is the model that matters. Evaluate candidates against the *actual* task:
re-dress a fixed performance, frame by frame, inside a mask.

### Must have

1. **Mask-conditioned inpainting.** It must accept an editable region and
   condition on the surrounding real pixels. A model that only does full-frame
   generation cannot be used here — the compositor would discard most of its
   output anyway.
2. **Garment reference conditioning.** A specific garment from reference images,
   not a text description of one. Text-only conditioning cannot reproduce *this*
   striped top.
3. **Temporal coherence.** Per-frame independent generation flickers. Either the
   model is temporally aware, or you supply optical-flow warping as a prior, or
   the output is unusable. Test this before anything else — it is the most
   common way a promising still-image model fails at video.
4. **8GB feasibility** with quantisation and tiling, at 1080×1920 or with a
   tiling strategy that does not produce visible seams.
5. **Determinism** for a fixed seed. Without it, resume is not bit-exact and the
   reproducibility manifest records intent rather than fact. If a candidate is
   non-deterministic, say so in the manifest instead of pretending.

### Should have

6. Pose or DensePose conditioning, so a printed pattern tracks the body instead
   of swimming.
7. Depth conditioning for occlusion reasoning.
8. Multi-view reference input (front/back/side) for turning choreography.
9. Lighting adaptation — a garment lit differently from the scene looks pasted on.

### Evaluation protocol

Do all of this on **your** master performance, not on the model's demo assets:

1. **One frame.** Does the garment look plausible, correctly scaled and lit?
2. **Ten consecutive frames.** Render each independently at the same seed and
   view them as a sequence. This is the flicker test, and it is where most
   candidates fail.
3. **Determinism.** Render the same frame twice at the same seed. Compare
   SHA-256. Not "looks the same" — identical bytes.
4. **Mask discipline.** How much output falls outside the mask? The compositor
   will discard it, but heavy overspill means the model is fighting the mask and
   the boundary will look wrong.
5. **VRAM headroom.** Watch peak usage. Leave room for the OS and the browser
   you will have open.
6. **Wall-clock per frame.** Multiply by your reveal length. 20 s/frame × 210
   frames is 70 minutes per outfit; decide whether that is acceptable *before*
   committing.
7. **Turning frames.** Pick the frames where the performer's back is visible.
   Most failures are here.
8. **A hard garment.** Something sheer, sequinned, or with a large logo. Note
   what breaks.

Record the results in `docs/implementation-status.md`. Do not claim a garment
model works until it has been integrated and tested — a plausible single frame
is not a working pipeline.

## The character animator

Needed only for the synthetic-master path. Judged separately from the garment
renderer because the job is different: it invents a whole person rather than
editing pixels inside a mask.

### Must have

1. **Pose conditioning** that actually follows the supplied skeleton. Test with
   the composed poses, not the model's demo motions.
2. **Character identity preservation** from the Hero Character references,
   across the whole sequence. Drift over 200 frames is the common failure.
3. **Chunk conditioning**: it must accept previously accepted frames as context
   and continue from them. A model that cannot will show a seam at every chunk
   boundary, and its backend must declare `ContextMode.NONE` so QC records the
   boundary honestly. Declare `LAST_FRAME` or `SEQUENCE` only for what the
   workflow genuinely reads — the pipeline gathers exactly the declared amount,
   and the manifest records exactly what was handed over.
4. **A stable background** across chunks. The garment pipeline's
   `background_preserved` check later assumes the master's background holds
   still.
5. **Fixed output framing** at the canonical profile's size. A model that crops
   or rescales breaks the whole coordinate system.
6. **8GB feasibility** at a chunk of 8–24 frames with 12–24 context frames.

### Evaluation protocol

On **your** composition, not the model's demos:

1. One chunk. Does the character look like the Hero Character, in the right pose?
2. Two consecutive chunks. Is the boundary invisible? This is the test most
   candidates fail.
3. Determinism: animate the same chunk twice at the same seed, compare frame
   hashes. Not "looks the same" — identical bytes.
4. The bridge frames specifically. A generated bridge is the least natural motion
   in the sequence; if the animator is going to break, it breaks there.
5. Background stability across the whole sequence.
6. Wall-clock per chunk × chunk count. A 214-frame composition at 24 frames per
   chunk is 9 chunks; multiply before committing.

Record the results in `docs/implementation-status.md`. Do not claim photoreal
character animation works until a workflow has been integrated and reviewed —
both shipped animator backends report `produces_photoreal: False`, and that is
the honest value until someone measures otherwise.

## Preprocessing adapters

Each has a stub at `src/app/adapters/` declaring its output contract, candidate
models and estimated VRAM. Priority order by value delivered per hour spent:

### 1. Video mask propagation — highest value

**Why first:** mask authoring is the dominant per-character cost. Propagating a
handful of keyframe annotations across the reveal removes most of it.

Candidates: SAM 2 video predictor, SAM 2.1, Cutie, DEVA, XMem.

Judge on: temporal stability of the boundary (a jittering hem produces visible
flicker), prompt ergonomics, windowed inference for 8GB, determinism for a fixed
prompt set.

Output: one grayscale PNG per frame in the project's mask conventions.

### 2. Human parsing — highest safety impact

**Why:** it produces the *protected* mask, the most safety-critical artefact
here.

Candidates: SCHP, Graphonomy, ATR/LIP-trained parsers, Sapiens-seg.

Judge on **recall around hands, hair strands and skin/garment boundaries**. A
false negative there leaks an edit onto the performer's body. Precision matters
less: a slightly oversized protected mask costs a little garment coverage, which
is the cheap direction to be wrong in.

### 3. Pose — now load-bearing, not just metadata

Candidates: DWPose, RTMPose, ViTPose, MediaPipe Pose, OpenPose.

Pose has **two roles** and they set different bars:

* **Garment pipeline**: control and QA metadata only. Temporal stability and
  Windows installability are what matter.
* **Motion Composition**: pose *is* the product. Extraction quality directly
  bounds what a synthetic master can be. Judge additionally on:
  * **wrist accuracy** — a jittering wrist is the most visible artefact at a
    motion join, and the anchor score weights hands highest;
  * **temporal stability of shoulder width** — the normalizer's scale comes from
    it, and a wandering estimate reads as the body pumping;
  * **confidence calibration** — the pipeline trusts `confidence` to decide what
    to interpolate and what to reject, so an over-confident detector is worse
    than an uncertain one.

Output must be the internal pose JSON format
(`app.motion.pose_format`). Until an adapter is installed, poses are computed
externally and imported with `app motion import-pose`, which is how the system
runs today. A deterministic mock exists for tests and is deliberately **not**
registered, so no pipeline can pick it up by accident.

### 4. Depth

Candidates: Depth Anything V2, Marigold, MiDaS, DPT.

Relative depth is sufficient. Temporal stability matters more than absolute
accuracy.

### 5. Optical flow

Candidates: RAFT, GMFlow, SEA-RAFT, OpenCV DIS (classical, no weights).

The classical CPU fallback is genuinely viable here and needs no weights — but
it is still not wired up until it has been tested, because an untested
"fallback" is just an untested code path.

### 6. Face landmarks

Candidates: MediaPipe FaceMesh, InsightFace, 3DDFA_V2, SynergyNet.

Feeds the face-region QC check. Optional: that check falls back to the protected
mask when landmarks are absent, and says so rather than silently passing.

### 7. DensePose

Candidates: DensePose (Detectron2), DensePose-CSE, Sapiens-dense.

Only needed once you want temporally consistent garment texture mapping. Skip
until the basic pipeline produces good output.

## Licensing

For each model, before downloading:

* Does the licence permit your intended use (likely commercial)?
* Are the training data provenance and licence documented?
* Are there restrictions on generated output?
* Does it require a licence server, account or activation? If yes, it breaks
  the offline guarantee — reject it.

Record your answers. This is not a formality; it is the reason the system
records usage rights for garments too.

## Integration steps

Once a model is chosen:

1. **Download weights manually** into a location outside the repository (they
   are gitignored regardless: `*.safetensors`, `*.ckpt`, `*.pt`, `*.onnx`,
   `*.gguf`, `*.engine`, …).
2. **For a ComfyUI workflow:** build and test it by hand in ComfyUI, title the
   nodes, export API format, write a contract. See
   [`comfyui-integration.md`](comfyui-integration.md).
3. **For a direct Python backend:** subclass `RendererBackend`, implement
   `healthcheck`, `capabilities`, `prepare`, `render_frame` (and
   `render_window` if temporal), `resume`, `collect_artifacts`. Register it in
   `app/backends/registry.py`. Report `deterministic` and
   `requires_model_weights` **honestly** — the manifest depends on it.
4. **For an adapter:** subclass the matching `AnalysisAdapter`, implement
   `capability()` and `run()`, register it. The pipeline picks it up with no
   further changes.
5. **Write tests** that skip cleanly when the weights are absent, so the suite
   still passes on a machine without them.
6. **Update** [`implementation-status.md`](implementation-status.md) with what
   was integrated, what was measured, and what remains.

## What not to do

* Do not add automatic weight downloads. Fetching is the operator's decision.
* Do not add a cloud fallback "for when the GPU is busy".
* Do not let a model write output frames directly. It returns an image; the
  compositor decides which pixels survive. That boundary is the identity
  guarantee.
* Do not relax the protected-mask clamp to make a model look better.
* Do not claim determinism you have not verified by comparing hashes.
