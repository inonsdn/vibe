# Motion Composition

A second way to obtain a Master Human Performance: instead of filming one, an
original **Hero Character** is animated from motion borrowed out of reference
clips.

The hard constraint, and the reason this phase exists at all:

> The original people, faces, clothes, backgrounds and pixels must never appear
> in the resulting Master Human Performance.

Only geometry travels. A motion reference contributes joint positions; its
imagery is never copied, never blended, never previewed.

## Vocabulary

These five things are distinct, and the docs, CLI and schemas keep them
distinct:

| Term | What it is | Contains pixels from a reference? |
| --- | --- | --- |
| **Motion reference** (`MotionSource`) | A clip whose *motion* is being borrowed | No — only its path and hash are recorded |
| **Motion-control composition** (`MotionComposition`) | Normalized, joined, bridged pose data | No — pose JSON only |
| **Candidate synthetic master** (`MasterCandidate`) | Frames an animator produced from the composition + Hero Character | No — the animator draws the Hero Character |
| **Accepted master** | A candidate an operator explicitly accepted; immutable from then on | No |
| **Garment replacement job** (`RenderJob`) | The original pipeline, operating on an accepted master | No |

A candidate is **not** a master. The gate between them is a human decision.

## Pipeline

```
reference clip A ──probe+hash──▶ MotionSource A ──pose JSON──┐
reference clip B ──probe+hash──▶ MotionSource B ──pose JSON──┤
                                                             │
                              ┌──────────────────────────────┘
                              ▼
                    normalize each source INDEPENDENTLY
                    into the canonical body frame
                              │
                              ▼
                    rank anchor candidates near the
                    end of A and the start of B
                              │
                              ▼
                    generate a 10–12 frame pose bridge
                    (cubic Hermite + bone-length correction)
                              │
                              ▼
                    MotionComposition  ──▶ skeleton preview (review here!)
                              │
                              ▼
        Hero Character + composition ──▶ CharacterAnimatorBackend
                              │              (chunked, context-conditioned)
                              ▼
                    MasterCandidate ──▶ QC ──▶ operator ACCEPTS
                              │
                              ▼
                    accepted, immutable master
                              │
                              ▼
                    the existing garment pipeline, unchanged
```

## Canonical normalization

Two clips show different people, at different distances, framed differently.
Before their motions can be joined, each must be expressed in one coordinate
system so that "the same pose" means the same numbers.

Each source is normalized **independently** — never against the other — so
adding a third reference later cannot change how the first two were normalized.

The transform is a uniform scale plus a translation:

```
canonical = source * scale + offset
```

* **Uniform**, because anisotropic scaling would distort limb proportions and
  make limb-length continuity meaningless.
* The **scale** comes from robust statistics over the whole segment — the
  *median* shoulder width and torso length — not from the current frame. That
  is what stops the body pumping as the detector jitters. Both measurements are
  used because either alone is fragile: shoulder width collapses when the
  performer turns sideways, torso length shortens when they bend.
* The **root** is the midpoint of the shoulder centre and the hip centre, which
  is far less sensitive to one mis-detected hip than the hip centre alone.
* **Translation is smoothed** symmetrically (forward then backward, then
  averaged), so jitter is removed without introducing the lag a single forward
  pass would.
* Short low-confidence gaps are interpolated; anything longer than
  `max_interpolation_gap` is **rejected**, because inventing a body position is
  worse than refusing the segment.
* Source timing is preserved. `playback_speed` changes recorded timestamps only;
  resampling motion is a separate, lossy decision that should be explicit.

All of it is configured in `config/canonical_skeleton.v1.yaml`.

## Anchor matching

Given the tail of one segment and the head of the next, every candidate pairing
is scored on the differences a viewer would actually notice at a cut:

| Term | Why it is in the score |
| --- | --- |
| body joint distance | the overall shape must match |
| wrist / hand position | a hand jump is the most visible artefact |
| torso rotation | a shoulder-line flip reads as a stumble |
| head rotation | a head snap breaks the illusion instantly |
| root position | the body must not teleport |
| shoulder scale | a size change reads as a camera cut |
| incoming/outgoing velocity | matching poses moving in opposite directions still cut badly |
| joint confidence | never anchor on a guessed pose |

Weights and normalisers live in config. The search is exhaustive over the
configured windows and fully deterministic — same inputs, same ranking, with
ties broken on the earliest frame pair so ordering never depends on float noise.

Candidates are returned **ranked**, and the operator can override the choice.
An override is scored exactly like an automatic pick, so a human decision is
recorded with the same evidence rather than being exempt from measurement.

## Bridge generation

A bridge is **generated pose data**, never blended source imagery. Two clips of
two different people cannot be cross-faded without both people appearing; a
bridge sidesteps that entirely.

Interpolation is **cubic Hermite**: linear interpolation matches positions but
not velocities, so the body would arrive at the join, stop dead, and set off
again. Hermite takes the outgoing velocity of the previous segment and the
incoming velocity of the next, so motion flows through.

Interpolating each joint independently is not sufficient on its own. Two joints
sharing a bone follow different curves, so the bone stretches mid-bridge —
measured at 11% on the test fixtures, which is visibly wrong. Every interpolated
frame therefore passes through a **bone-length correction**: the skeleton is
walked outward from the shoulders and each child joint is placed at the
*interpolated* bone length along its raw direction. Limb lengths then follow the
interpolation of the two anchors' lengths by construction.

### The frame layout at a join

Stated once, because this is where off-by-one bugs live:

```
segment A ──▶ [A.start, anchor_prev)
bridge    ──▶ B frames; bridge[0] == pose(anchor_prev)
                        bridge[B-1] == pose(anchor_next)
segment B ──▶ [anchor_next + 1, B.end)
```

The anchor frames are contributed **by the bridge and by nothing else**, so
every output frame has exactly one origin: no duplication, no gap.
`MotionComposition.expected_frame_count()` computes exactly this, and a QC check
compares it against both the declared count and the files on disk.

Endpoints are **copied, not evaluated**: floating-point Hermite at t=0 and t=1
is almost exact, and "almost" is not what the contract says. The QC tolerance
for endpoint equality defaults to `0.0`.

## Chunk continuity

An 8GB card cannot hold a long sequence, so animation is chunked. Chunks are
stitched by **conditioning**, not concatenation:

* chunks tile the output range exactly — no frame is generated twice;
* each chunk receives the last 12–24 **accepted** frames as context — but
  only as many as the backend declares it consumes;
* every backend declares a `ContextMode` in its capabilities:

  | Mode | What the backend consumes | What the pipeline gathers |
  | --- | --- | --- |
  | `none` | nothing | nothing |
  | `last_frame` | the single frame before the chunk | one frame |
  | `sequence` | the whole configured tail, in order | `overlap_frames` frames |

  The number recorded per chunk is the number the backend was handed, and QC
  reads the declared mode rather than assuming. Reporting "16 context frames"
  while a workflow receives one image is the specific failure this exists to
  prevent. The mock animator declares `none`, honestly: it paints every frame
  independently and has nothing to condition on;
* the output size is fixed by the canonical profile, so no chunk can drift in
  framing.

Because context frames are read from disk, a resumed run receives exactly the
conditioning a single-pass run would — which is what makes resume bit-exact.

## The acceptance gate

A candidate master is the one artifact in this system that a model invented
wholesale. Everything downstream treats the master as ground truth, so a human
says "yes, this is our character, performing correctly" before it earns that
status.

`accept_master` refuses unless:

1. every frame is animated,
2. QC has been run,
3. QC passed (overridable with `--allow-qc-failure`, which is audited),
4. the reason is a real explanation (10+ characters).

Acceptance is written to the audit log with who, why and when, and is recorded
in the manifest. An accepted master is immutable: re-animating it is refused.

## Promotion: from accepted candidate to renderable template

Acceptance is a judgement. Promotion is the work, and it is a separate command
so that a failed promotion leaves an accepted candidate rather than a
half-built template.

`app master promote <candidate_id> --transition-anchor <frame>`:

1. requires the candidate to be accepted, complete and QC'd;
2. resolves the transition anchor — explicit, or the one the composition's join
   recommended. With more than one join it refuses to guess: pass
   `--transition-anchor`, or `--confirm-multiple-joins` to take the first
   recommendation;
3. validates the anchor leaves at least one intro frame and one reveal frame;
4. **hardlinks (or copies) the generated PNGs** into
   `templates/<id>/source_frames/`, byte for byte, and hashes them there;
5. creates the `HumanTemplate` record with `intro = [0, anchor)`,
   `reveal = [anchor, frame_count)`, the Hero Character's reference images as
   identity references, and the hero's consent record;
6. writes an archival MP4 for operators to watch, sets the candidate's
   `video_path`, `output_hashes` and `promoted_template_id`, and audits it.

**The PNGs are the source of truth.** Promotion never encodes to H.264 and
decodes back: that would quantise and chroma-subsample exactly the pixels the
garment pipeline later promises to restore outside the editable mask. The
archival video exists for review, and nothing in the render path reads it.

Promotion is idempotent — running it twice returns the same template. Promoting
with a *different* anchor is refused: a promoted master is immutable, so make a
new candidate instead. If any step fails, the template record and its directory
are removed before the error propagates.

### Where the anchor comes from

Every join records, in **output** frame numbering:

| Field | Meaning |
| --- | --- |
| `output_bridge_start` / `output_bridge_end` | the bridge's half-open range in the composed sequence |
| `recommended_transition_anchor` | where the garment reveal should begin — the bridge start, i.e. the frame after the borrowed opening motion |
| `prev_source_frame` / `next_source_frame` | the originating anchors in each source clip's own numbering |

That is what lets `master promote` be run without an explicit anchor on a
single-join composition, and what makes the operator's choice explicit on a
multi-join one.

## Reusing one motion source

A composition may use the same reference clip more than once — two ranges of one
performance is an ordinary thing to want. Normalized poses are therefore written
to `normalized_poses/segment_<NNN>_<source_id>/`, keyed by the segment's
**position** as well as its source. Keying on the source id alone made the
second segment overwrite the first, silently. Each segment keeps its own
canonical transform in the manifest and in the composition's normalization
summary.

## Operator commands

```bash
# 1. Register the references. No frames are extracted.
app motion ingest ./clip_03.mp4 --name "Motion A" --start-frame 0 --end-frame 168 \
    --motion-use-authorized --rights-holder "..." --license "..."
app motion ingest ./clip_02.mp4 --name "Motion B" --start-frame 12 --end-frame 210 \
    --motion-use-authorized --rights-holder "..." --license "..."

# 2. Attach pose data. Either extract it locally with DWPose ONNX ...
app motion extract-pose <motion_a> --adapter dwpose_onnx \
    --detector-model /models/dwpose/yolox_l.onnx \
    --pose-model /models/dwpose/dw-ll_ucoco_384.onnx \
    --provider cuda --diagnostics
# ... or import poses computed elsewhere.
app motion import-pose <motion_b> --from ./pose/clip_02
app motion inspect <motion_a>

# 3. Check that two very different sources land on one canonical scale.
app motion normalize <motion_a>
app motion normalize <motion_b>

# 4. Find where they can be joined.
app motion match-anchors --prev <motion_a> --next <motion_b> --top 10

# 5. Compose. Omit --anchor to use the best automatic match, or pin it.
app motion compose --name "Prototype v1" \
    --segment <motion_a> --segment <motion_b> \
    --bridge-frames 12 --anchor 150:20

# 6. REVIEW THE PREVIEW before spending GPU time.
app motion preview <composition_id>
app motion qc <composition_id>

# 7. Animate the Hero Character.
app master register-hero --name "Hero Character v1" --image ./hero/front.png
app master create --name "Master v1" --composition <composition_id> --hero <hero_id> \
    --origin synthetic --backend mock
app master animate <candidate_id>

# 8. QC, then accept explicitly.
app master qc <candidate_id>
app master inspect <candidate_id>
app master accept <candidate_id> --by "your name" \
    --reason "reviewed the bridge and framing; approved for production"

# 9. Promote it into a template the garment pipeline can render.
#    Omit --transition-anchor to use the composition's recommendation.
app master promote <candidate_id> --transition-anchor 150 --by "your name"

# 10. From here it is an ordinary template: masks, compatibility, render.
app template import-masks <template_id> --kind garment --from ./masks/garment
app compat check --template <template_id> --garment <garment_id>
app job create --template <template_id> --garment <garment_id>
```

Step 6 is the one that saves the most time. A bad bridge costs seconds to fix in
pose space and hours to discover after animation.

## Current prototype

`config/motion_prototype.example.yaml` documents the prototype's intended shape
— two clips, their frame ranges, a 10–12 frame bridge, vertical 9:16 framing
head-through-knees, mostly frontal motion, Hero Character v1. It is an
**example**, not a hard-coded fact: nothing in `src/` reads it, because clip
numbering and frame ranges are production decisions that change.

## What is not implemented

* **No pose model.** Pose is imported. The adapter interface and a deterministic
  *test-only* mock exist; the registered adapter reports `not_implemented` and
  raises rather than fabricating production data.
* **No character animation model.** The mock animator draws a schematic figure;
  the ComfyUI animator submits a workflow you author. Neither is photoreal, and
  both report `produces_photoreal: False`.
* **No background consistency guarantee from a real backend.** The QC check
  exists and passes against the mock, whose background is fixed by construction.
  Whether a real animator holds a background still is unknown until one is
  integrated.
