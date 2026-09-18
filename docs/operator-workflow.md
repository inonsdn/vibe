# Operator workflow

## Choosing a master origin

Everything below assumes a Master Human Performance exists. There are two ways
to get one, and they meet at the same place:

| Origin | How | When |
| --- | --- | --- |
| **`captured_master`** | Film an authorized performer, then `app template ingest` | You can shoot the performance you want |
| **`synthetic_master`** | Borrow motion from reference clips, animate an original Hero Character, then accept it | You cannot shoot it, or you want one character across many motions |

A synthetic master is built in [`motion-composition.md`](motion-composition.md)
and **requires explicit operator acceptance** before it can be used. Once
accepted it is an immutable master like any other, and everything from "Once per
character / step 3" below applies unchanged.

The rest of this document describes the captured path. For the synthetic path,
do the motion phase first, accept the master, then rejoin at step 3 (authoring
masks).

## Once per character

You are producing many outfit videos from **one** performance, so this stage is
worth doing carefully — every future render inherits its quality.

### 1. Shoot or generate the master performance

Requirements that matter downstream:

* **Vertical 1080×1920**, constant frame rate.
* **A neutral, minimal base outfit.** This is the single highest-leverage
  decision: the replacement garment must cover at least as much as the base, so
  a bulky base permanently limits which outfits are possible. A fitted neutral
  bodysuit or minimal two-piece maximises the compatible set.
* **Stable lighting and a static camera.** Every outfit inherits this lighting;
  the renderer cannot relight the scene convincingly.
* **A clear intro section** before the reveal — the part that will be reused
  verbatim in every video.
* **Hands and hair kept away from the garment where possible.** Every crossing
  needs an occlusion mask.

Record the consent/provenance paperwork now: a fully synthetic character, or a
real consenting adult performer with a document reference.

### 2. Ingest

```bash
app template ingest ./master.mp4 \
    --name "Dancer A - Routine 1" \
    --transition-anchor 90 \
    --clothing-class neutral_bodysuit \
    --subject-kind synthetic \
    --identity-ref ./refs/face_front.png
```

The anchor is the seam: intro is `[0, 90)`, reveal is `[90, end)`. Frame 90 is
the first re-dressed frame. Variable-frame-rate sources are refused unless you
pass `--allow-vfr-conversion`, which writes a new CFR file and leaves the
original untouched.

Note the template id. Everything references it.

### 3. Author masks

No mask model is installed, so this is external work — and it is the bulk of the
per-character effort. For each reveal frame you need:

| Family | Required | Contents |
| --- | --- | --- |
| `garment` | yes | The base outfit region to replace |
| `protected` | yes (for validation) | Face, hair, hands, exposed skin, legs, background |
| `occlusion` | strongly recommended | Hair strands, hands, props in front of the garment |
| `expansion` | when needed | Extra area a larger garment may occupy |

8-bit grayscale PNG, `frame_000090.png` onwards, matching the source frame size,
`0` immutable / `255` editable. See
[`mask-semantics.md`](mask-semantics.md) for authoring advice.

```bash
app template import-masks <template_id> --kind garment   --from ./masks/garment
app template import-masks <template_id> --kind protected --from ./masks/protected
app template import-masks <template_id> --kind occlusion --from ./masks/occlusion
```

Import validates everything before copying anything, so a bad batch cannot
half-land.

### 4. Validate

```bash
app template inspect <template_id>
```

Fix every problem before continuing. A missing mask at one frame fails the
render *at that frame*, twenty minutes in.

## Per outfit

### 5. Ingest the garment

```bash
app garment ingest \
    --image front=./outfit_01/front.jpg \
    --image back=./outfit_01/back.jpg \
    --image side=./outfit_01/side.jpg \
    --category top --coverage torso --silhouette fitted --material cotton \
    --sleeve short --length mid_thigh \
    --transparency 0.0 --reflectivity 0.1 --fabric-flow 0.2 \
    --product-name "Blue Striped Top" --brand "Example" \
    --source-url "https://example.com/p/123" \
    --license "licensed for this use" --rights-holder "Example Inc" \
    --pattern "vertical navy stripes"
```

The attributes are not decoration — they drive the compatibility rules. Getting
`--coverage` or `--length` wrong either blocks a workable garment or lets
through one that will leak the base outfit. `--source-url` is recorded for
attribution and never fetched.

### 6. Check compatibility

```bash
app compatibility check --template <template_id> --garment <garment_id> \
    --exposed-view front --exposed-view back --exposed-view side
```

Pass `--exposed-view` for each view the choreography actually shows. Omit them
entirely and the rules assume the performance turns (requiring back and side),
which is the safe default.

Read the output. Each rule prints its verdict and, when it fails, a remediation
line. Typical outcomes:

* **`NEEDS_INPUT`, missing back view** → add the image and re-check.
* **`NEEDS_INPUT`, expansion mask required** → author one and re-import.
* **`INCOMPATIBLE`, hem shorter than the base** → this garment cannot work on
  this performance. Pick another, or shoot a performance with a shorter base.
* **`READY` with warnings** → renderable; read the warnings, they predict what
  will look wrong.

If a block is genuinely wrong for your situation, override it — with a real
reason, because it is audited and ends up in the manifest:

```bash
app compatibility override <report_id> \
    --reviewer "your name" \
    --reason "the choreography never turns; the back is never visible" \
    --acknowledge missing_views_for_motion
```

### 7. Render

Start with the mock backend to confirm the wiring — it is instant, CPU-only and
deterministic:

```bash
app job create --template <template_id> --garment <garment_id> --backend mock --seed 1234
app job render <job_id> --backend mock
```

With a real ComfyUI workflow, render one frame first and look at it:

```bash
app job create --template <template_id> --garment <garment_id> --backend comfyui
app job render <job_id> --max-frames 1
# inspect data/jobs/<job_id>/raw_frames/ and composited_frames/
app job resume <job_id>
```

Interrupting is safe. `app job resume <job_id>` continues without re-rendering
a single completed frame.

### 8. Compose

```bash
app job compose <job_id>
```

Joins the cached intro to the rendered reveal at the anchor, copies the master
audio, encodes 1080×1920 H.264 yuv420p, and writes `manifest.json`.

Options: `--no-audio`, `--flash-frames 3` for a flash at the seam,
`--preview` for a small review encode, `--output-name my_video.mp4`.

### 9. QC

```bash
app job qc <job_id>
```

Fifteen checks. The ones that matter most:

| Check | What a failure means |
| --- | --- |
| `protected_region_preserved` | The performer's identity was altered. Stop and investigate. |
| `background_preserved` | Pixels changed outside the mask. Stop and investigate. |
| `intro_reuse_integrity` | The intro is not the cached source. Stop. |
| `transition_correctness` | The seam is wrong. Check the anchor. |
| `garment_temporal_stability` | Flicker in the garment region. Usually missing occlusion masks or a non-temporal workflow. |
| `duplicate_frames` | The render may have stalled and repeated a frame. |
| `dimensions_fps_frame_count` | The output does not match the contract. |

Then look at the contact sheets in `data/jobs/<job_id>/contact_sheets/`. The
transition sheet shows frames either side of the seam, labelled with their
origin (`intro` / `reveal` / `flash`) — the fastest way to catch an off-by-one
or a visible pop. The reveal sheet samples the whole segment for drift.

Numbers catch leakage; your eyes catch "the garment looks pasted on".

### 10. Repeat from step 5

The template, its masks and its intro cache are all reused. Only steps 5–9
repeat per outfit, and the intro is byte-identical every time — guaranteed by
construction, and re-verified by the `intro_reuse_integrity` check on every job.

## Batch loop

```bash
for outfit in ./outfits/*/; do
  gid=$(app --json garment ingest \
      --image "front=$outfit/front.jpg" --image "back=$outfit/back.jpg" \
      --category top --coverage torso --silhouette fitted --material cotton \
      --sleeve short --length mid_thigh --license licensed \
      --product-name "$(basename "$outfit")" --brand Example \
      | python -c 'import json,sys; print(json.load(sys.stdin)["garment_id"])')

  app compatibility check --template "$TPL" --garment "$gid" \
      --exposed-view front --exposed-view back || { echo "skip $gid"; continue; }

  jid=$(app --json job create --template "$TPL" --garment "$gid" --backend mock \
      | python -c 'import json,sys; print(json.load(sys.stdin)["job"]["id"])')

  app job render "$jid" && app job compose "$jid" && app job qc "$jid"
done
```

`app compatibility check` exits `3` for a blocking state, so `||` skips
incompatible outfits without rendering them. All `--json` output goes to stdout
while logs go to stderr, so piping stays clean.

## Recovering

| Situation | Do this |
| --- | --- |
| Motion animation interrupted | `app master animate <candidate_id> --resume` |
| Bridge looks wrong in the preview | Re-run `app motion compose` with a different `--anchor` or `--bridge-frames`; it costs seconds, re-animating costs hours |
| Candidate master QC failed | Fix the composition and re-animate, or `app master reject <id>` so it is not silently reused |
| Candidate accepted by mistake | An accepted master is immutable. Create a new candidate; the acceptance stays in the audit log |
| Render interrupted | `app job resume <job_id>` |
| Render failed at a frame | `app job inspect <job_id>` shows the error and frame; fix the cause, then resume |
| QC failed on protected pixels | Do not ship. Check the protected masks, then re-render |
| Source frames were edited | The template is invalid. Re-ingest as a new template |
| Wrong anchor | Re-ingest with the correct `--transition-anchor`; masks can be re-imported |
| Disk filling up | `app job delete <job_id> --yes` removes one job's artefacts; exports and templates are untouched |
