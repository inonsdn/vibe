# Architecture

## The one idea

**The human is never regenerated.** A single immutable *Master Human
Performance* video supplies every pixel of the performer, the background, the
camera and the motion. Identity vectors, pose, depth, flow and landmarks are
auxiliary control and QA metadata — they are never substitutes for source
pixels.

Every render therefore has the shape:

```
output_pixel = source_pixel                  where the mask forbids editing
output_pixel = blend(source, rendered, α)    where the mask allows it
```

and the second case is confined to the garment region of the reveal segment.

## Layers

```
┌──────────────────────────────────────────────────────────────────────┐
│  app/cli            app/api                                          │
│  Typer commands     FastAPI routers  ← thin adapters, no logic        │
└───────────────────────────┬──────────────────────────────────────────┘
                            │  both construct a ServiceContext and call
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  app/pipeline                                                        │
│    context.py          ServiceContext (config + data root + repos)   │
│    template_ingest.py  probe → extract → hash → segment → validate   │
│    garment_ingest.py   copy → hash → measure                         │
│    compatibility.py    the rule engine (pure)                        │
│    compat_service.py   evaluate + persist + override (audited)        │
│    render.py           the orchestrator: masks, backend, composite    │
│    compose.py          intro cache + assembly + encode + manifest    │
└───────┬───────────────────────┬──────────────────────┬───────────────┘
        │                       │                      │
        ▼                       ▼                      ▼
┌───────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│ app/media     │   │ app/backends         │   │ app/qc               │
│  ffmpeg.py    │   │  base.py  (interface)│   │  metrics.py          │
│  frames.py    │   │  registry.py         │   │  checks.py           │
│  masks.py     │   │  mock/    (CPU only) │   │  report.py           │
│  compositor.py│   │  comfyui/ (localhost)│   │  contact_sheet.py    │
│  assembly.py  │   └──────────────────────┘   └──────────────────────┘
└───────────────┘
        │                       │
        ▼                       ▼
┌───────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│ app/domain    │   │ app/adapters         │   │ app/db               │
│  Pydantic v2  │   │  SAM2 / parsing /    │   │  migrations/*.sql    │
│  strict       │   │  pose / densepose /  │   │  database.py         │
│  schemas      │   │  depth / flow /      │   │  repositories.py     │
│               │   │  landmarks (STUBS)   │   │                      │
└───────────────┘   └──────────────────────┘   └──────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  app/core   config · paths (traversal-safe) · hashing · logging ·    │
│             errors · ids · determinism · provenance                  │
└──────────────────────────────────────────────────────────────────────┘
```

`app/offline/verify.py` sits beside these and reports on the whole environment.

## Invariants, and where each one is enforced

| Invariant | Enforced by | Verified by |
| --- | --- | --- |
| Source frames never change | `verify_source_immutability` before every render/compose | `test_source_frames_are_never_modified`, `test_tampered_source_frames_block_a_render` |
| Intro is reused, never rendered | `ensure_intro_cache` byte-copies; render range starts at the anchor | `test_intro_frames_are_identical_across_garment_jobs`, `intro_reuse_integrity` QC check |
| Only mask-allowed pixels change | `composite()` hard-restores `mask == 0` after blending | `test_only_allowed_garment_pixels_change`, `background_preserved` QC check |
| Protected pixels equal source | protected masks subtracted, then clamped to 0 post-feather | `test_protected_beats_garment_everywhere_they_overlap`, `protected_region_preserved` QC check |
| No off-by-one at the anchor | `HumanTemplate` model validator requires `intro.end == anchor == reveal.start` | `test_anchor_must_equal_intro_end`, `test_transition_anchor_has_no_off_by_one` |
| Resume never redoes work | per-frame rows in `job_frames`, written only after the PNG lands | `test_resume_renders_only_the_remaining_frames`, `test_completed_frames_are_not_handed_to_the_backend_again` |
| Same inputs → same output | seeds derived from `(job seed, frame index, garment key)` | `test_identical_seed_and_inputs_give_identical_output` |
| Blocked garments cannot render | `_assert_render_allowed` in `create_job` **and** `render_job` | `test_needs_input_blocks_job_creation`, `test_incompatible_blocks_job_creation` |
| Paths cannot escape the data root | `DataRoot.resolve` on every operator-supplied path | `tests/unit/test_paths.py` (12 escape vectors) |
| Remote ComfyUI refused | `assert_local_endpoint` in the client constructor | `test_remote_urls_are_rejected_by_default` (8 URL shapes) |

## Data flow for one render

1. **`create_job`** loads the template and garment at pinned versions, requires
   a `READY` (or validly overridden) compatibility report, pins the frame range
   to start exactly at the anchor, hashes every input, records the config hash,
   and persists the job.
2. **`render_job`** re-verifies source immutability, prepares the backend, then
   for each window of frames:
   - loads the source frame and the four masks,
   - builds the *effective mask* and saves it (auditable per frame),
   - rejects the frame if the editable area is empty or implausibly large,
   - derives the frame seed and asks the backend for a full frame,
   - composites through the effective mask and **fails the job** if a single
     pixel changed outside it,
   - writes the PNG, hashes it, and records the frame as complete.
3. **`compose_job`** populates the intro cache, assembles one contiguous
   directory keyed by absolute frame index, asserts the seam, copies the master
   audio, runs one explicit ffmpeg encode, and writes the manifest.
4. **`run_qc`** measures 15 checks and writes JSON + text reports plus contact
   sheets around the transition and across the reveal.

## Design decisions

**Frames on disk, not in memory.** An 8GB GPU and a long clip mean the working
set must be bounded. Lossless PNG per frame costs disk but makes every stage
independently inspectable, resumable and hashable. It is also what makes
"the intro is byte-identical" a checkable claim rather than a hope.

**Absolute frame indices everywhere.** A file named `frame_000090.png` means
frame 90 of the master video in *every* directory — source, masks, raw,
composited, assembly. Per-segment renumbering is the single largest source of
off-by-one bugs in this kind of pipeline, so it simply does not exist here. The
encoder is told `-start_number <intro.start>` once, explicitly.

**The backend returns a full frame; the pipeline owns the output.** A backend
cannot write an output frame. It returns an image, and `composite()` decides
which of its pixels survive. A misbehaving or hallucinating model can produce a
bad garment but cannot touch the performer's face — and if it tries, the job
fails with the exact pixel count.

**Protected masks win last.** The clamp `editable[protected] = 0` runs *after*
feathering and blurring, so no amount of morphological softening can reopen a
protected pixel. Overriding this requires a stored reviewer override that
explicitly names `protected_mask_override`.

**Derived seeds, never accumulated state.** Frame seeds are
`sha256(seed, frame_index, garment_key)`. This is why rendering frame 40 alone
produces exactly the pixels a sequential run would, which is what makes resume
bit-exact instead of merely plausible.

**Checkpoints in SQLite, written after the file lands.** A frame counts as done
only once its composited PNG is on disk and hashed. A hard kill mid-window
redoes at most that window — never less, which would silently ship a missing
frame.

**Compatibility rules are data, not code paths.** Severities and thresholds
live in versioned YAML; reports record the rules version *and* the file hash.
Tuning a threshold is a config change, and an old report still says which rules
it was judged by.

**The mock backend is a first-class citizen.** It has no GPU, weights or network
dependency, is deterministic, and is what the entire test suite renders with.
That keeps the orchestration honest and means the pipeline was never developed
against a model's forgiving behaviour.

**No node ids in application code.** The ComfyUI contract maps logical inputs
(`source_frame`, `mask`, `seed`, …) to node *titles*. Re-arranging a graph in
the UI does not require a code change; renaming a node fails loudly at
`prepare()` with the exact list of unresolved bindings.

**Two QC tolerance regimes.** Lossless intermediates are checked at
`max_diff == 0` — there is no excuse for a changed byte. The compressed final
file gets configurable tolerances, because H.264 and 4:2:0 chroma subsampling
perturb every pixel slightly and pretending otherwise would make QC useless.

**Adapters are interfaces, not fakes.** Every preprocessing adapter reports
`not_implemented` and *raises* when run. Nothing in the pipeline can mistake
synthetic data for real model output, and masks can be authored manually today.

## What is deliberately absent

No analytics, no cloud storage, no model hubs, no auto-updaters, no automatic
weight downloads, no background job queue (one render at a time on one GPU is
the honest model), and no authentication (a localhost single-operator tool that
pretended to be multi-tenant would be worse, not safer).
