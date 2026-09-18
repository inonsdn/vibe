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

Both are **wiring tests**, not renderers. Each uses core ComfyUI nodes only — no
checkpoints, no custom nodes, no weights — so they validate on a bare ComfyUI
install and the test suite can exercise the contract machinery without a model.

| Workflow | Backend | What it proves |
| --- | --- | --- |
| `garment_replace_placeholder` | `ComfyUIBackend` | Garment inputs arrive, outputs come back, manifests are complete |
| `character_animate_placeholder` | `ComfyUIAnimatorBackend` | Pose control and context frames arrive, chunk outputs come back |

Neither produces usable output. `garment_replace_placeholder` flat-composites a
reference image into the mask; `character_animate_placeholder` blends a pose
image over a character reference and returns a single frame, which the animator
backend correctly rejects as a chunk-size mismatch.

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
