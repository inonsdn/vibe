# Implementation status and known limitations

Honest accounting of what works, what is a documented interface, and what is
not done. Nothing below claims a capability that has not been tested.

## Complete and tested

| Area | Status | Evidence |
| --- | --- | --- |
| Domain schemas (Pydantic v2, strict) | Done | `tests/unit/test_domain.py` |
| Traversal-safe path handling | Done | `tests/unit/test_paths.py` — 12 escape vectors |
| Config layering + hashing | Done | `tests/unit/test_core.py` |
| Structured JSON logging | Done | `tests/unit/test_core.py` |
| SQLite schema + forward-only migrations | Done | `tests/unit/test_core.py` |
| Repositories (versioned records, audit log) | Done | exercised throughout |
| ffprobe/ffmpeg wrappers, command building | Done | `tests/unit/test_core.py` |
| Template ingestion (probe → extract → hash → segment) | Done | `tests/integration/test_template_ingest.py` |
| Mask import + validation | Done | same |
| Mask semantics, combination, conflict detection | Done | `tests/unit/test_masks.py` |
| Compositing + identity preservation | Done | `tests/unit/test_compositor.py` |
| Garment ingestion (copy, hash, measure) | Done | `tests/integration/test_api.py`, `test_cli.py` |
| Compatibility engine (14 rules, versioned YAML) | Done | `tests/unit/test_compatibility_rules.py` |
| Compatibility gate + audited overrides | Done | `tests/integration/test_compatibility_gate.py` |
| Renderer abstraction | Done | `src/app/backends/base.py` |
| Mock backend (CPU, deterministic) | Done | `tests/unit/test_mock_backend.py` |
| Render orchestration, windows, checkpoints | Done | `tests/integration/test_mock_pipeline.py` |
| Resume without redoing work | Done | `tests/integration/test_resume.py` |
| Intro cache + assembly + transition | Done | `tests/unit/test_qc_metrics.py`, `test_mock_pipeline.py` |
| Final encode (1080×1920 H.264 yuv420p) | Done | `test_final_video_metadata_matches_the_configuration` |
| Audio copy from master | Done | `audio_video_duration` QC check |
| Render manifest + reproducibility digest | Done | `test_manifest_is_complete_and_reproducible` |
| QC (15 checks, JSON + text) | Done | `test_qc_passes_for_a_clean_mock_render` |
| Contact sheets | Done | `tests/unit/test_qc_metrics.py` |
| CLI (all documented commands) | Done | `tests/integration/test_cli.py` |
| HTTP API + minimal web page | Done | `tests/integration/test_api.py` |
| Offline verification | Done | `tests/unit/test_offline_and_env.py` |
| ComfyUI client, contract binding, remote refusal | Done | `tests/unit/test_comfyui.py` |
| Proxy isolation (`trust_env=False`) | Done | `tests/unit/test_proxy_and_deps.py` |
| DNS + socket network guard | Done | `tests/unit/test_proxy_and_deps.py` |
| Pinned, tested dependency versions | Done | `constraints/tested-py311.txt` + tests |
| **Motion Composition** | | |
| Pose JSON format + IO | Done | `tests/unit/test_motion_domain.py` |
| Motion domain schemas | Done | `tests/unit/test_motion_domain.py` |
| Motion ingestion (no frames extracted) | Done | `tests/integration/test_motion_pipeline.py` |
| Pose import + validation | Done | same |
| Canonical normalization | Done | `tests/unit/test_motion_normalize.py` |
| Deterministic anchor matching | Done | `tests/unit/test_motion_anchors_bridge.py` |
| Bridge generation + bone-length correction | Done | same |
| Composition assembly + frame arithmetic | Done | `tests/integration/test_motion_pipeline.py` |
| Skeleton preview | Done | same (ffmpeg-marked) |
| Motion QC (11 checks) | Done | same |
| **Master creation** | | |
| CharacterAnimatorBackend abstraction | Done | `src/app/backends/animator/base.py` |
| Mock animator (CPU, deterministic) | Done | `tests/integration/test_motion_pipeline.py` |
| Chunked animation, context gathered per declared `ContextMode` | Done | same + `tests/unit/test_comfyui_animator.py` |
| Chunk resume | Done | same |
| Master QC (6 checks) | Done | same |
| Master manifests | Done | same |
| Operator acceptance gate | Done | same |
| Promotion: accepted candidate → `HumanTemplate` | Done | `tests/integration/test_master_promotion.py` |
| Promoted master renders an outfit end to end | Done | `test_a_promoted_master_completes_a_mock_garment_render` |
| ComfyUI sequence contract (whole chunk, batch size, context) | Done | `tests/unit/test_comfyui_animator.py` |
| Pose adapter input semantics (video + range, not a folder) | Done | `tests/unit/test_motion_inputs.py` |
| Confidence threshold applied consistently | Done | same |
| Reusing one motion source in a composition | Done | `tests/integration/test_motion_composition_reuse.py` |
| Output-space join anchors + multi-join confirmation | Done | same |
| Motion + master CLI and API | Done | `tests/integration/test_motion_cli_api.py` |

**All tests pass** with no GPU, no model weights, no ComfyUI and no network —
and with proxy environment variables both set and unset. `ruff`, `black` and
`mypy` are clean. See the run output in the change description for the exact
count.

## Interfaces, deliberately not implemented

These are documented contracts with honest capability checks. Every one reports
`not_implemented` and **raises** when called, so nothing can mistake synthetic
data for real model output.

| Adapter | Purpose | Candidates listed in the stub |
| --- | --- | --- |
| `sam2` | Video mask propagation from keyframes | SAM 2, SAM 2.1, Cutie, DEVA, XMem |
| `human_parsing` | Derive the protected mask | SCHP, Graphonomy, Sapiens-seg |
| `pose` | Control/QA metadata | RTMPose, ViTPose, DWPose, MediaPipe |
| `densepose` | UV correspondence for texture mapping | DensePose, DensePose-CSE |
| `depth` | Occlusion reasoning, shading cues | Depth Anything V2, Marigold, MiDaS |
| `optical_flow` | Temporal consistency prior | RAFT, GMFlow, SEA-RAFT, OpenCV DIS |
| `face_landmarks` | Face-region QC | MediaPipe FaceMesh, InsightFace |

Pose has been promoted from "nice to have" to load-bearing: it is metadata for
the garment pipeline but the *product* of Motion Composition. The registered
adapter still reports `not_implemented` and raises; a deterministic
`MockPoseAdapter` exists for tests and is deliberately **not** registered, so no
pipeline can pick it up by accident. Poses are imported today.

See [`model-selection-checklist.md`](model-selection-checklist.md).

## The character animator: what is and is not proven

**Proven by tests:** the abstraction, chunk planning (chunks tile the range
exactly), that the pipeline gathers exactly the context a backend declares it
consumes, resume across process restarts, bit-exact determinism for identical
inputs, manifest completeness, the acceptance gate refusing incomplete or
un-QC'd candidates, and that an accepted candidate promotes into a
`HumanTemplate` whose frames are the generated PNGs byte for byte and which
then completes a mock garment render to a final MP4.

**Context honesty.** `AnimatorCapabilities.context_mode` is `none`,
`last_frame` or `sequence`, and the pipeline gathers exactly that much. The
mock animator declares `none` — it paints every frame independently, so it has
nothing to condition on, and claiming otherwise would put context frames in the
manifest that no renderer ever looked at. `ComfyUIAnimatorBackend` declares
`sequence` and stages every gathered frame.

**Not proven, because it cannot be:** that any real model animates a Hero
Character convincingly. The mock animator draws a schematic figure — flat
shading, a stick-and-polygon body — and is obviously not photoreal. Both shipped
animator backends report `produces_photoreal: False`, which is the honest value
until a workflow has been integrated and reviewed.

**Specifically unverified:** whether a real animator holds a background still
across chunks. The `master_background_consistency` check exists and passes
against the mock, whose background is fixed by construction — that proves the
check works, not that a real model will pass it.

## The ComfyUI backend: what is and is not proven

**Proven by tests** (via `httpx.MockTransport`, no ComfyUI required): endpoint
policy and remote refusal, contract loading and validation, binding by node
title, graph patching without mutating the source, prompt submission, progress
polling, timeout handling, vanished-prompt detection, missing-node-type
reporting, output download, and that the placeholder workflow needs only core
nodes.

For the animator specifically, the sequence contract is proven on the wire: a
`MockTransport` captures the submitted graph, and the tests assert that every
pose in a chunk is uploaded exactly once in frame order, that `batch_size` is
actually bound into the graph, that the whole context sequence is bound rather
than just its last frame, that ordinals map back to absolute frame indices via
the staged manifest, and that a short or duplicated output is refused.

**Not proven, because it cannot be:** that any real garment workflow produces
good output. No such workflow exists here. The placeholder flat-composites a
reference image into the mask and looks nothing like clothing on a body — that
is intentional; it is a wiring test. The animation placeholder is the same:
correct plumbing, visually useless output.

**Not yet verified against a live ComfyUI:** the tests use a mock transport.
The first time you point it at a real instance, expect to adjust for
installation quirks. `prepare()` is designed to fail loudly and specifically
when that happens.

## Known limitations

### Pose data is imported, not extracted

The dominant per-motion-reference cost. No pose model is installed, so pose JSON
is produced externally and imported. A pose adapter is the highest-value
integration for this phase, and its quality directly bounds what a synthetic
master can be. The adapter seam itself is settled: `estimate_sequence` receives
the video file and the exact source frame indices, and the poses it writes must
carry those same indices — which is checked, not trusted.

### The animation placeholder needs a custom node

`character_animate_placeholder` binds `pose_sequence` to a
`LoadImagesFromDirectory` node, which is not core ComfyUI. That is the honest
shape of a multi-frame workflow; `prepare()` names the missing node classes
rather than pretending. The garment placeholder is still core-nodes-only.

### The bridge is the least natural motion in a composition

It is generated, not observed. The bone-length correction keeps it physically
coherent and the QC checks bound its drift, but a 10–12 frame bridge between
genuinely dissimilar anchors will still read as a transition rather than as
continuous dancing. The anchor score is what keeps that rare; review the preview.

### Anchor matching is 2D and metric, not semantic

It scores joint geometry. It does not know that a hand is about to occlude the
face, or that a garment hem is mid-swing. Two poses can score identically and cut
differently. That is what the ranked candidates and the operator override are
for.

### One motion composition assumes consistent framing

Every segment is normalized to the same canonical profile, so a reference shot
in a wildly different framing (full body vs head-and-shoulders) will normalize
to a plausible scale but may place joints outside the target frame. The QC
framing check catches it after the fact; it is not prevented up front.

### Playback speed does not resample motion

`playback_speed` changes recorded timestamps only. Genuinely retiming motion is
a separate, lossy decision and is deliberately out of scope rather than done
badly by default.

### Masks are authored by hand

The dominant per-character cost. No mask model is installed, so `garment`,
`protected`, `occlusion` and `expansion` masks are produced externally and
imported. Mask propagation is the highest-value future integration.

### One render at a time

No job queue or worker pool. One GPU, one render, driven by an operator who is
watching. `--max-frames` plus `resume` is how a long render is driven in slices.
A queue would be misplaced complexity for a single-operator local tool.

### Disk-heavy

Lossless PNG per frame, per stage. Roughly 3–6 MB per 1080×1920 frame; a
600-frame master plus a few jobs is tens of gigabytes. Deliberate: it makes
every stage inspectable, resumable and hashable, and makes "the intro is
byte-identical" checkable. Assembly hardlinks where the filesystem allows it.

### Reproducibility is only as good as the backend

The mock backend is deterministic and tested as such. A ComfyUI workflow may not
be — samplers, custom nodes and some CUDA kernels break bit-exactness. The
manifest records seeds and hashes faithfully; whether re-running reproduces them
depends on the workflow. Verify by comparing hashes, and record the answer.

### The face QC check needs landmarks

Without a face-landmark adapter, `face_region_preserved` reports `skipped` and
says the protected-mask check covers the face region instead. That is true but
weaker: it verifies the masked region was untouched, not that the *face
specifically* was.

### Flash frames are excluded from protected-pixel checks

An optional 2–4 frame flash deliberately alters the whole frame, protected
regions included. Those frames are marked `source: "flash"` in the manifest and
excluded from the protected/background checks. Correct, but worth knowing.

### No relighting

The garment is composited into the master's existing lighting. Reflective and
metallic materials will look pasted on — which is why
`reflective_material_warning` exists rather than pretending otherwise.

### No true cloth simulation

A fixed performance cannot produce fabric that swings independently of the body.
High-flow fabric looks pinned; above `0.95` flow the compatibility engine blocks
the render instead of shipping something wrong.

### Compatibility rules are heuristics

Fourteen deterministic rules over declared attributes. They catch the common,
expensive mistakes cheaply. They will occasionally block something workable —
hence the audited override — and will occasionally pass something that looks
wrong, which is what QC and your eyes are for.

### Single-operator security model

No authentication, because there is nothing to authenticate to and a localhost
service with token theatre would be worse. Defended: operator mistakes, path
escapes, misconfiguration, backend misbehaviour. Not defended: a malicious local
operator or a compromised local ComfyUI. See
[`offline-security.md`](offline-security.md).

### Windows is the target, Linux is what was tested here

The code is OS-agnostic (`pathlib` throughout, no shell strings, no POSIX-only
calls) and the docs target Windows 11. This repository's test runs happened on
Linux. Windows-specific items — long paths, hardlinks across volumes, NTFS
behaviour — are documented in [`windows-setup.md`](windows-setup.md) but have
not been exercised on Windows hardware here.

## Remaining for the model-selection phase

**Garment replacement**

1. Choose and integrate a garment renderer; verify determinism and temporal
   coherence on your own footage before trusting it.
2. Integrate mask propagation — the biggest reduction in manual effort.
3. Integrate human parsing to generate protected masks automatically, judged on
   recall around hands, hair and skin.

**Motion Composition**

4. Integrate a pose adapter (DWPose/RTMPose class), judged on wrist accuracy,
   shoulder-width stability and confidence calibration.
5. Integrate a character animator and verify: chunk-boundary invisibility,
   identity stability over 200+ frames, background stability, and determinism by
   hash comparison. Only then may `produces_photoreal` become `True`.
6. Measure how a real animator handles the generated bridge frames specifically
   — they are the least natural motion it will be asked to render.

**Both**

7. Integrate depth, flow, DensePose and landmarks as the workflows need them.
8. Measure real per-frame and per-chunk time and VRAM on the RTX 5060 and
   record it here.
9. Re-run the full suite on Windows and fix anything platform-specific.
10. Update this document with measured results, not expectations.
