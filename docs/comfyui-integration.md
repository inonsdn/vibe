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

A contract declaring a name outside this set fails to load, rather than being
silently ignored at render time.

## The shipped placeholder

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
