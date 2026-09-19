# garment-replacer

A fully local, model-agnostic system for producing short vertical videos of the
**same** character performing the **same** dance, with the **same** timing,
camera, lighting and background — changing only the clothing.

It has two phases. **Motion Composition** builds a Master Human Performance,
either by ingesting a filmed one or by animating an original Hero Character from
motion borrowed out of reference clips. **Garment replacement** then re-dresses
that master, as many times as you like, without ever regenerating the human.

The human is never regenerated. One immutable *Master Human Performance* video
is the single source of truth for every pixel of the performer. Each render
touches only the garment region of the reveal segment and restores the original
bytes everywhere else.

> **Model status:** no AI model weights are selected, referenced, downloaded or
> required. This repository is the complete system shell: data contracts,
> pipeline, mock backend, ComfyUI adapter, validation, QC, tests and docs. Real
> garment models are chosen in a later phase — see
> [`docs/model-selection-checklist.md`](docs/model-selection-checklist.md).
> Nothing here claims a garment model works, because none has been integrated.

---

## Two origins for a master

| Origin | How you get it | Usable when |
| --- | --- | --- |
| `captured_master` | `app template ingest` an authorized performer video | Immediately |
| `synthetic_master` | Compose motion, animate a Hero Character, **accept** it, then **promote** it | Only after an operator accepts it following QC *and* it has been promoted to a template |

A synthetic master borrows **motion only**: the original people, faces, clothes,
backgrounds and pixels never appear in the result. Motion artifacts are pose
JSON, and a QC check fails if any image file appears among them. See
[`docs/motion-composition.md`](docs/motion-composition.md).

Acceptance and promotion are two separate acts, deliberately:

* **`app master accept`** is the human judgement — "yes, this is our character,
  performing correctly". It changes status and writes an audit record. It does
  **not** make the master renderable.
* **`app master promote`** does the work — it freezes the generated PNG frames
  into a `HumanTemplate`, splits them into intro and reveal at a transition
  anchor, and records the hashes every later render is checked against.

Promotion never re-encodes: the frames the animator generated are hardlinked (or
copied) into the template byte for byte, because an H.264 round trip would
quantise exactly the pixels the pipeline later promises to preserve. An archival
MP4 is written alongside for operators to watch; nothing reads it back.

Once promoted, a synthetic master is immutable and the garment pipeline cannot
tell the difference from a captured one — deliberately.

## How it works

For every outfit, the pipeline:

1. **Reuses the cached intro segment** — byte-copied from the immutable source
   frames, never regenerated.
2. **Processes only the reveal segment** after the configured transition anchor.
3. **Replaces pixels only inside the editable garment region.**
4. **Restores original source pixels** outside the editable/feathered mask.
5. **Preserves** the face, hair, hands, exposed skin, legs, background, camera,
   timing and motion — verified numerically, not assumed.
6. **Joins** the cached intro and the newly dressed reveal at the anchor.
7. **Produces a reproducible render manifest** (input hashes, seeds, settings,
   workflow hash, dependency versions, the literal ffmpeg command, output
   hashes).

The guarantee in step 4 is arithmetic, not etiquette: the compositor
hard-restores every pixel the mask marks immutable *after* blending, and
protected masks always beat editable masks unless an explicitly stored,
reviewed override says otherwise.

```
master.mp4 ──ingest──▶ source_frames/ (immutable, hashed)
                            │
              ┌─────────────┴─────────────┐
              │                           │
     intro_cache/ (byte copy)      reveal frames + masks
              │                           │
              │                    backend renders ──▶ composite through mask
              │                           │
              └────────▶ assembly_frames/ ◀┘   (seam exactly at the anchor)
                                │
                          ffmpeg encode ──▶ exports/<job>.mp4 + manifest.json
                                                        + qc_report.{json,txt}
```

---

## Quick start (mock backend only — no GPU, no weights, no network)

Requires Python 3.11+ and FFmpeg on `PATH`. Nothing else.

```bash
# 1. Install, against the tested dependency pins
python -m pip install -e ".[dev]" -c constraints/tested-py311.txt

# 2. Check the environment (endpoints, executables, paths, GPU, adapters)
app doctor

# 3. Ingest a master performance.
#    The intro is [0, 90); the reveal is [90, end). Frame 90 is the seam.
app template ingest ./master.mp4 --name "Dancer A" --transition-anchor 90 \
    --clothing-class fitted_short

# 4. Import the masks you authored for the reveal frames.
#    (Mask files are frame_000090.png … matching the source frame size.)
app template import-masks <template_id> --kind garment   --from ./masks/garment
app template import-masks <template_id> --kind protected --from ./masks/protected
app template import-masks <template_id> --kind occlusion --from ./masks/occlusion

# 5. Verify the template is complete and renderable
app template inspect <template_id>

# 6. Ingest a garment's reference images and attributes
app garment ingest \
    --image front=./outfit/front.jpg \
    --image back=./outfit/back.jpg \
    --image side=./outfit/side.jpg \
    --category top --coverage torso --silhouette fitted --material cotton \
    --sleeve short --length mid_thigh \
    --product-name "Blue Striped Top" --brand "Example" --license "licensed"

# 7. Gate the render on deterministic compatibility rules
app compatibility check --template <template_id> --garment <garment_id> \
    --exposed-view front --exposed-view back --exposed-view side

# 8. Render with the mock backend (deterministic, CPU-only, obviously synthetic)
app job create --template <template_id> --garment <garment_id> --backend mock --seed 1234
app job render <job_id> --backend mock

# 9. Join the cached intro to the reveal and encode 1080x1920 H.264
app job compose <job_id>

# 10. Run automated QC (JSON + text reports and contact sheets)
app job qc <job_id>

# 11. Confirm nothing requires the network
app offline verify
```

Steps 8–11 need no GPU, no model weights and no ComfyUI. The mock backend paints
a deliberately obvious procedural pattern inside the garment mask, which is
enough to exercise orchestration, compositing, the transition, export, resume,
logging, manifests and QC end to end.

Optional local HTTP API and minimal web page:

```bash
app serve            # http://127.0.0.1:8077  (/docs for the API reference)
```

---

## The mask contract (one screen)

Masks are 8-bit grayscale PNG, one per frame, matching the source frame size:

| Value | Meaning |
| ----: | --- |
| `0` | **immutable** — the output pixel must equal the source pixel |
| `255` | **editable** — the backend may replace this pixel |
| `1…254` | feather / blend boundary (linear blend source ↔ render) |

Four mask families per frame, combined in a fixed order:

```
editable = garment ∪ expansion        widen for a larger silhouette
editable = editable − occlusion       hair/hands/props in FRONT of the garment
editable = editable − protected       face, hair, hands, skin, background
editable = feather(editable)          softened inward only
editable[protected] = 0               final clamp — protected always wins
```

Full specification: [`docs/mask-semantics.md`](docs/mask-semantics.md).

---

## Commands

| Command | Purpose |
| --- | --- |
| `app doctor` | Environment, backends, adapters, data paths, GPU |
| `app template ingest` | Probe, extract lossless frames, hash, segment |
| `app template inspect` | Show a template and validate frames + masks |
| `app template import-masks` | Import externally authored masks |
| `app template list` | List templates |
| `app garment ingest` | Copy, hash and measure garment references |
| `app garment inspect` / `list` | Show garment assets |
| `app compatibility check` | Run the deterministic rule engine |
| `app compatibility override` | Store a reviewed, audited override |
| `app job create` | Create a render job |
| `app job render --backend mock` | Render with the CPU-only mock backend |
| `app job render --backend comfyui` | Render via a local ComfyUI workflow |
| `app job resume` | Resume without re-rendering completed frames |
| `app job inspect` / `list` | Job status, progress, checkpoint, artefacts |
| `app job compose` | Join cached intro + reveal, encode, write manifest |
| `app job qc` | Automated QC + contact sheets |
| `app job delete --yes` | Delete exactly one validated job directory |
| `app job backends` | Backend capabilities and health |
| `app motion ingest` | Register a motion reference (no frames extracted) |
| `app motion import-pose` | Attach externally computed pose JSON |
| `app motion extract-pose` | **Extract pose with DWPose ONNX** (local models, no downloads) |
| `app motion inspect` / `list` | Show and validate a motion reference |
| `app motion normalize` | Dry-run the canonical transform for one source |
| `app motion match-anchors` | Rank compatible join frames between two sources |
| `app motion compose` | Normalize, join, bridge and assemble a control sequence |
| `app motion preview` | Skeleton preview — review before animating |
| `app motion qc` | Motion QC (drift, joins, bridge endpoints, no pixels) |
| `app master register-hero` | Register an original Hero Character |
| `app master create --origin synthetic` | Create a candidate master |
| `app master animate` | Animate it in chunks (resumable) |
| `app master qc` | Master QC + manifest |
| `app master inspect` | Status, chunks, acceptance state |
| `app master accept` | Record the operator's acceptance of a candidate (audited) |
| `app master promote` | Build the immutable `HumanTemplate` the garment pipeline renders |
| `app offline verify` | Prove no cloud/network dependency |
| `app serve` | Localhost HTTP API |

Add `--json` to any command for machine-readable output on stdout (structured
logs go to stderr, so piping stays clean).

---

## Documentation

| Document | Contents |
| --- | --- |
| [`docs/architecture.md`](docs/architecture.md) | Components, data flow, invariants, design decisions |
| [`docs/motion-composition.md`](docs/motion-composition.md) | Motion references, normalization, anchors, bridges, acceptance and promotion |
| [`docs/windows-setup.md`](docs/windows-setup.md) | Windows 11 + RTX 5060 8GB setup, step by step |
| [`docs/data-layout.md`](docs/data-layout.md) | Directory layout and file format specification |
| [`docs/mask-semantics.md`](docs/mask-semantics.md) | Mask values, families, combination order, authoring |
| [`docs/compatibility-rules.md`](docs/compatibility-rules.md) | Every rule, its thresholds and how to tune them |
| [`docs/comfyui-integration.md`](docs/comfyui-integration.md) | Wiring a real workflow to logical inputs |
| [`docs/offline-security.md`](docs/offline-security.md) | Offline model, path safety, consent, data handling |
| [`docs/operator-workflow.md`](docs/operator-workflow.md) | The day-to-day production loop |
| [`docs/dwpose-setup.md`](docs/dwpose-setup.md) | DWPose ONNX pose extraction: models, config, smoke test, tuning |
| [`docs/model-selection-checklist.md`](docs/model-selection-checklist.md) | What to evaluate before choosing any model |
| [`docs/troubleshooting-8gb-vram.md`](docs/troubleshooting-8gb-vram.md) | Making this work inside 8GB |
| [`docs/implementation-status.md`](docs/implementation-status.md) | What is done, what is a stub, known limitations |

---

## Target environment

* Windows 11 (the code is OS-agnostic; paths, process handling and docs target Windows)
* Python 3.11+
* NVIDIA RTX 5060, 8GB VRAM — architecture assumes a model does **not** fit in VRAM
* 32GB system RAM
* FFmpeg + ffprobe on `PATH`
* ComfyUI running locally, as an **optional** rendering backend
* No cloud API, no telemetry, no network access during render, localhost services only

---

## Development

```bash
# Install against the tested pins. The constraints file is not optional
# bookkeeping: unconstrained resolution has installed NumPy 2.5 with OpenCV 5.0,
# a combination that aborts the interpreter on `import cv2`.
python -m pip install -e ".[dev]" -c constraints/tested-py311.txt

python -m black src tests          # format
python -m ruff check src tests     # lint
python -m mypy                     # type check
python -m pytest                   # tests (no GPU, no weights, no network)
```

The test suite runs with no GPU, no model weights, no ComfyUI and no network
access. The no-network rule is *enforced*: `tests/conftest.py` patches the
socket layer so any outbound connection attempt — or DNS lookup — fails the test
that made it. Proxy environment variables are covered too: the suite sets
`HTTP_PROXY`, `HTTPS_PROXY` and `ALL_PROXY` to deliberately broken values and
asserts that localhost behaviour is unaffected.
Fixture videos, frames, masks and product images are all generated
programmatically — the repository contains no media.

---

## Safety and consent

Every template requires a consent/provenance record: either a fully synthetic
character or a real, consenting adult performer with an on-file consent
reference. `adult_confirmed` is mandatory and validated at the schema level; a
`consented_human` template without a `consent_document_ref` cannot be
persisted. Usage rights for garment references are recorded and are part of the
compatibility rules.

## License

Proprietary. See the repository owner.
