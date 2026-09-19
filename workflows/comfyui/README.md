# ComfyUI workflows

Each workflow is a pair of files:

| File | What it is |
| --- | --- |
| `<workflow_id>.json` | A ComfyUI **API-format** graph (Workflow → Export (API)) |
| `<workflow_id>.contract.yaml` | The mapping from logical inputs to node titles + input names |

Application code never references a node id. It asks the contract where
`source_frame`, `mask`, `pose`, `seed` and friends belong, so editing a graph in
the ComfyUI UI does not require a code change.

## Shipped workflows

Both are **wiring tests**, not renderers. Neither contains a checkpoint, a
model name or anything that triggers a download.

| Workflow | Backend | What it proves |
| --- | --- | --- |
| `garment_replace_placeholder` | `ComfyUIBackend` | Garment inputs arrive, outputs come back, manifests are complete |
| `character_animate_placeholder` | `ComfyUIAnimatorBackend` | A whole pose *sequence* arrives in order, batch size is bound, context frames arrive, chunk outputs come back one per frame |

Neither produces usable output. `garment_replace_placeholder` flat-composites a
reference image into the mask; `character_animate_placeholder` blends the pose
run over a repeated character reference and drops the context sequence on the
floor, so chunk continuity cannot be judged with it.

**Node requirements.** `garment_replace_placeholder` uses core nodes only, so it
validates on a bare ComfyUI install. `character_animate_placeholder` needs a
directory-loading node (`LoadImagesFromDirectory`) from a custom node pack,
because loading an ordered image run is what a multi-frame workflow actually
does. `prepare()` names the missing node classes; install the pack yourself.

## Adding a real workflow

1. Build and test it inside ComfyUI first, by hand, on one frame or one chunk.
2. Title every node the application must write into.
3. Export the API format JSON into this directory.
4. Copy the matching placeholder contract, rename it to
   `<your_workflow_id>.contract.yaml`, and update `workflow_file`, the node
   titles and the set of `inputs`.
5. Point config at it:
   * garment renderer → `comfyui.workflow_id`
   * character animator → `animator.workflow_id`
6. `app doctor` reports whether the contract resolves and whether all node
   classes are installed.

Nothing here downloads models. If a workflow needs a checkpoint or a custom
node, install it yourself in ComfyUI; the backend reports exactly which node
classes are missing rather than fetching anything.

See `docs/comfyui-integration.md` for the full guide.
