# ComfyUI integration

The ComfyUI backend is complete as **plumbing** and deliberately incomplete as
a garment model. It submits whatever workflow you author locally, maps logical
inputs onto that workflow, waits, and collects outputs. It contains no model
names, no checkpoint paths, and nothing that triggers a download.

## Safety properties

| Property | How |
| --- | --- |
| Remote endpoints refused | `assert_local_endpoint()` runs in the client **constructor**, so a misconfigured backend fails before any request is possible. |
| Allowlist restricted to localhost | `comfyui.allowed_hosts` defaults to `127.0.0.1`, `localhost`, `::1`. Overriding needs `allow_remote: true` **and** an allowlist entry. |
| No downloads | Only `/system_stats`, `/object_info`, `/prompt`, `/history`, `/queue`, `/interrupt`, `/upload/image`, `/view` are ever called. |
| Missing nodes reported, not fetched | `prepare()` diffs the workflow's `class_type`s against `/object_info` and fails with the exact list. |
| Bounded waits | Every request has a timeout; polling has a job deadline (`job_timeout_s`). |
| Proxy variables ignored | The owned HTTP client sets `trust_env=False`, so `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY` cannot route loopback traffic off the machine. |
| Interruption-safe | `resume()` interrupts any still-running prompt before re-preparing, so a stale prompt cannot write frames behind the pipeline's back. |

```yaml
# config/app.yaml
comfyui:
  base_url: "http://127.0.0.1:8188"
  allowed_hosts: ["127.0.0.1", "localhost", "::1"]
  allow_remote: false        # keep this false
  request_timeout_s: 30.0
  poll_interval_s: 1.0
  job_timeout_s: 900.0
  workflow_id: garment_replace_placeholder
  upload_inputs: true
```

## No node ids in application code

Application code never references a ComfyUI node id. A **contract** file maps
logical input names to node *titles* (what you type into a node's header in the
UI, stored as `_meta.title` in an API export) plus the input field on that node.

Re-arranging or re-numbering a graph therefore does not break the integration.
Renaming a titled node does — loudly, at `prepare()`, with the exact list of
unresolved bindings.

```yaml
inputs:
  - logical_name: source_frame
    node_title: SOURCE_FRAME      # the node's title in the ComfyUI UI
    input_name: image             # the input field on that node
    kind: image_upload
    required: true
```

Logical inputs the pipeline can supply:

| Name | Value |
| --- | --- |
| `source_frame` | The immutable source frame (uploaded; the value is the returned filename) |
| `mask` | The effective editable mask, feathered, protection already applied |
| `expansion_mask`, `protected_mask` | The individual families, if a workflow wants them |
| `garment_reference`, `garment_reference_back`, `garment_reference_side` | Uploaded garment references |
| `pose`, `depth` | Control data, when the operator has produced it |
| `seed` | Per-frame derived seed |
| `steps`, `denoise`, `guidance_scale` | From `RenderSettings` |
| `prompt`, `negative_prompt` | From the job |
| `width`, `height`, `frame_index`, `batch_size` | Frame facts |
| `output_prefix` | `<job_id>_<frame_index>`, so concurrent jobs cannot collide |

Character animation adds sequence-level inputs, because a chunk is 8–24 frames
and a workflow that receives one still is not animating anything:

| Name | Value |
| --- | --- |
| `pose_sequence` | **Every pose in the chunk**, in order, as one asset |
| `context_sequence` | Every accepted context frame handed to the chunk, in order |
| `context_frame` | Just the last one, for a workflow that only wants that |
| `hero_reference`, `hero_reference_back`, `hero_reference_side` | Hero Character references |
| `frame_count`, `batch_size` | How many frames this submission must produce |
| `start_frame`, `fps` | Where the chunk sits in the output, and at what rate |

A contract declaring a name outside this set fails to load, rather than being
silently ignored at render time.

### Binding kinds

| `kind` | What the bound value is |
| --- | --- |
| `value` | A scalar written straight into the node input |
| `image_path` | A local path (ComfyUI reads it directly) |
| `image_upload` | Uploaded to ComfyUI's `input/`; the value is the returned name |
| `sequence_dir` | A directory of ordinal-numbered PNGs; the value is the input-relative folder |
| `sequence_video` | One visually lossless clip (`-qp 0`, `yuv444p`); the value is the uploaded file |

`pose_sequence` and `context_sequence` **must** use a `sequence_*` kind, and a
`sequence_*` kind may only be used for them. A sequence input bound as a single
still is precisely the bug this validation exists to catch.

Sequence assets are staged per chunk under
`<candidate>/comfy_inputs/chunk_NNNN/`, with a `sequence.json` manifest mapping
each ordinal position back to its absolute output frame index. Files are named
by ordinal (`pose_00000.png`, …) so a directory-loading node reads them in frame
order under plain lexicographic sort; the manifest is what keeps the mapping to
absolute indices auditable.

## Two workflows, two backends

There are now two ComfyUI integrations, sharing the same client and contract
machinery:

| Backend | Workflow id | Job |
| --- | --- | --- |
| `ComfyUIBackend` (renderer) | `garment_replace_placeholder` | Garment pixels inside a mask |
| `ComfyUIAnimatorBackend` | `character_animate_placeholder` | A whole character, from pose control |

Both refuse remote endpoints, both bind by node title, and both report
`requires_model_weights: True` with no model integrated. The animator
additionally reports `produces_photoreal: False` — the honest value until a real
workflow has been reviewed.

The animator submits **one prompt per chunk**. Before it submits anything it
checks that

* the chunk's poses cover the requested range exactly, with no gap or duplicate;
* the contract declares `pose_sequence` and `batch_size` — without the latter
  the output count is the workflow's guess, not the chunk's;
* the contract has somewhere to put the context it was handed, matching the
  backend's declared `ContextMode`;
* the staged pose sequence has one entry per requested frame, in order, and the
  staged context sequence matches the frames the pipeline gathered.

After collecting it checks that exactly `batch_size` frames came back and that
no output image was returned twice. The frame counts it records in the manifest
come from the staged assets, so the manifest cannot claim context the workflow
never received.

`ComfyUIAnimatorBackend` declares `ContextMode.SEQUENCE` and honours it: every
gathered context frame is staged and bound, not just the last one.

## The shipped placeholders

### Character animation

`workflows/comfyui/character_animate_placeholder.json` binds the full sequence
contract — `pose_sequence`, `context_sequence`, `batch_size`, `frame_count`,
`hero_reference`, `width`, `height`, `output_prefix` — and blends the pose run
over a repeated character reference.

Its directory loader (`LoadImagesFromDirectory`) is **not** a core ComfyUI node:
it comes from a directory-loading custom node pack. That is deliberate and
honest — a real multi-frame workflow needs one, and `prepare()` reports exactly
which node types your install is missing rather than pretending. Install the
pack yourself; this application never downloads anything.

The placeholder loads the context sequence and throws it away, so chunk
continuity cannot be evaluated with it. It is a wiring test, not an animator.

### Garment replacement

`workflows/comfyui/garment_replace_placeholder.json` uses **core nodes only** —
`LoadImage`, `LoadImageMask`, `ImageScale`, `ImageCompositeMasked`,
`SaveImage`. No checkpoints, no custom nodes, no weights, so it validates on a
bare ComfyUI install and the test suite can exercise the contract machinery
without any model.

It flat-composites the garment reference into the editable mask region. The
result looks nothing like clothing on a body — **that is the point.** It proves
inputs arrive, outputs come back, resume works and manifests are complete. Do
not evaluate garment quality with it.

## Connecting a real workflow

1. **Build and test it inside ComfyUI first**, by hand, on a single extracted
   frame. Confirm it produces a plausible garment before automating anything.
2. **Title every node** the application must write into. Use loud names —
   `SOURCE_FRAME`, `EDITABLE_MASK`, `GARMENT_REFERENCE_FRONT`, `SAMPLER_SEED`.
3. **Export the API format**: Workflow → Export (API). The editor format is
   rejected with a message telling you so.
4. **Write the contract** — copy the placeholder's YAML:

   ```yaml
   workflow_id: garment_replace_v1
   workflow_file: garment_replace_v1.json
   description: Garment replacement conditioned on source frame + mask + reference.
   requires_model_nodes: true

   inputs:
     - logical_name: source_frame
       node_title: SOURCE_FRAME
       input_name: image
       kind: image_upload
     - logical_name: mask
       node_title: EDITABLE_MASK
       input_name: image
       kind: image_upload
     - logical_name: garment_reference
       node_title: GARMENT_REFERENCE_FRONT
       input_name: image
       kind: image_upload
     - logical_name: seed
       node_title: SAMPLER_SEED
       input_name: seed
     - logical_name: steps
       node_title: SAMPLER_SEED
       input_name: steps
     - logical_name: denoise
       node_title: SAMPLER_SEED
       input_name: denoise
     - logical_name: prompt
       node_title: POSITIVE_PROMPT
       input_name: text
     - logical_name: negative_prompt
       node_title: NEGATIVE_PROMPT
       input_name: text
     - logical_name: output_prefix
       node_title: RENDERED_FRAME_OUT
       input_name: filename_prefix

   output_node_titles:
     - RENDERED_FRAME_OUT
   ```

5. **Point config at it**: `comfyui.workflow_id: garment_replace_v1`.
6. **Validate**: `app doctor` reports whether ComfyUI is reachable; the backend's
   `prepare()` validates that every binding resolves and every node class is
   installed.
7. **Render one frame first**:

   ```bash
   app job create --template <id> --garment <id> --backend comfyui
   app job render <job_id> --max-frames 1
   ```

   Inspect `data/jobs/<job_id>/raw_frames/` (what ComfyUI returned) and
   `composited_frames/` (what survived the mask). Then continue with
   `app job resume <job_id>`.

## What the backend does per frame

1. Writes the source frame and effective mask into `jobs/<id>/comfy_inputs/`.
2. Uploads both to ComfyUI's input folder (when `upload_inputs: true`).
3. Applies the contract to a **copy** of the graph — the on-disk workflow is
   never mutated.
4. `POST /prompt`, records the `prompt_id`, polls `/history` and `/queue`.
5. Downloads the first output image via `/view`.
6. Verifies the returned dimensions match the source, then hands the frame to
   the pipeline, which composites it through the mask.

Step 6 is why a workflow that returns a cropped or rescaled frame fails with a
clear message instead of producing a subtly misaligned render.

## Determinism

The mock backend is deterministic by construction. A ComfyUI workflow is
**not**, until you have verified it: samplers, custom nodes and non-deterministic
CUDA kernels all break reproducibility. `capabilities()` reports
`deterministic: false` for this backend, honestly.

Before trusting a workflow, render the same frame twice with the same seed and
compare hashes:

```bash
app job render <job_a> --max-frames 1
app job render <job_b> --max-frames 1   # same template, garment, seed
# then compare the frame hashes in each job's manifest
```

If they differ, resume will not be bit-exact and the reproducibility digest
becomes a record of intent rather than a guarantee. Note that in the manifest.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `backend_unavailable: Cannot reach ComfyUI` | Not running, or wrong port. Start with `--listen 127.0.0.1 --port 8188`. |
| `offline_policy_violation` | `base_url` is not loopback. Keep it local; this is intended behaviour. |
| `Workflow does not satisfy its contract` | A titled node was renamed or removed. The error lists each unresolved binding. |
| `missing node types` | Custom nodes are not installed. Install them in ComfyUI yourself; the app never fetches anything. |
| `This looks like a ComfyUI *editor* workflow` | Re-export with Workflow → Export (API). |
| `returned a frame whose dimensions do not match` | A node is rescaling or cropping. Bind `width`/`height` and remove the rescale. |
| `Timed out waiting for ComfyUI` | Raise `job_timeout_s`, or lower `backend.frame_window` so each prompt is smaller. |
| `ComfyUI reported completion but returned no images` | The workflow has no `SaveImage` node reachable from the output. |
