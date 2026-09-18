# ComfyUI workflows

Each workflow is a pair of files:

| File | What it is |
| --- | --- |
| `<workflow_id>.json` | A ComfyUI **API-format** graph (Workflow → Export (API)) |
| `<workflow_id>.contract.yaml` | The mapping from logical inputs to node titles + input names |

Application code never references a node id. It asks the contract where
`source_frame`, `mask`, `seed` and friends belong, so editing a graph in the
ComfyUI UI does not require a code change.

## Shipped workflows

- **`garment_replace_placeholder`** — wiring test. Core nodes only
  (`LoadImage`, `LoadImageMask`, `ImageScale`, `ImageCompositeMasked`,
  `SaveImage`), no checkpoints, no custom nodes, no weights. Use it to confirm
  that inputs arrive, outputs come back and manifests are complete. It is
  **not** a garment renderer and its output should never be judged as one.

## Adding a real workflow

1. Build and test it inside ComfyUI first, by hand, on a single frame.
2. Title every node the application must write into.
3. Export the API format JSON into this directory.
4. Copy `garment_replace_placeholder.contract.yaml`, rename it to
   `<your_workflow_id>.contract.yaml`, and update `workflow_file`, the node
   titles, and the set of `inputs`.
5. Set `comfyui.workflow_id: <your_workflow_id>` in `config/app.yaml` (or
   `config/local.yaml`).
6. `app doctor` reports whether the contract resolves against the graph and
   whether all node classes are installed.

Nothing here downloads models. If a workflow needs a checkpoint or a custom
node, install it yourself in ComfyUI; the backend will report exactly which
node classes are missing rather than fetching anything.

See `docs/comfyui-integration.md` for the full guide.
