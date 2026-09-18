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

**383 tests pass** with no GPU, no model weights, no ComfyUI and no network.
`ruff`, `black` and `mypy` are clean.

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

See [`model-selection-checklist.md`](model-selection-checklist.md).

## The ComfyUI backend: what is and is not proven

**Proven by tests** (via `httpx.MockTransport`, no ComfyUI required): endpoint
policy and remote refusal, contract loading and validation, binding by node
title, graph patching without mutating the source, prompt submission, progress
polling, timeout handling, vanished-prompt detection, missing-node-type
reporting, output download, and that the placeholder workflow needs only core
nodes.

**Not proven, because it cannot be:** that any real garment workflow produces
good output. No such workflow exists here. The placeholder flat-composites a
reference image into the mask and looks nothing like clothing on a body — that
is intentional; it is a wiring test.

**Not yet verified against a live ComfyUI:** the tests use a mock transport.
The first time you point it at a real instance, expect to adjust for
installation quirks. `prepare()` is designed to fail loudly and specifically
when that happens.

## Known limitations

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

1. Choose and integrate a garment renderer; verify determinism and temporal
   coherence on your own footage before trusting it.
2. Integrate mask propagation — the biggest reduction in manual effort.
3. Integrate human parsing to generate protected masks automatically, judged on
   recall around hands, hair and skin.
4. Integrate pose, depth, flow and landmarks as the workflow needs them.
5. Measure real per-frame time and VRAM on the RTX 5060 and record it.
6. Decide whether a preview-quality fast path is worth having beside the final
   quality path.
7. Re-run the full suite on Windows and fix anything platform-specific.
8. Update this document with measured results, not expectations.
