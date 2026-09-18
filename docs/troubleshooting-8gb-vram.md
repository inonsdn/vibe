# Troubleshooting: 8GB VRAM (RTX 5060)

The architecture assumes a model does **not** fit in 8GB. This document is the
set of levers, cheapest first.

## Nothing here needs a GPU today

The mock backend and the entire test suite run on CPU. If you are hitting VRAM
limits, it is from a ComfyUI workflow you supplied — which means the levers are
in ComfyUI and in this application's render settings.

## Two pipelines, two sets of levers

| Pipeline | Unit of work | Primary lever |
| --- | --- | --- |
| Garment replacement | a frame window | `backend.frame_window` |
| Character animation | a chunk | `animator.chunk_frames` |

The animation levers mirror the render ones:

```yaml
animator:
  chunk_frames: 24        # try 16, then 12, then 8
  overlap_frames: 16      # context frames handed to the next chunk (12-24)
  low_vram_mode: true
  cpu_offload: true
```

Lower `chunk_frames` first. `overlap_frames` is the continuity budget: below 12
chunk boundaries start to show, above 24 you are paying for context the model
cannot use. Both are validated, so a nonsensical pair fails at config load
rather than mid-animation.

A composition is animated in `ceil(frame_count / chunk_frames)` chunks, and
`app master animate --max-chunks N` renders only the next N — the animation
equivalent of `--max-frames`, and the pragmatic way to drive a long animation on
a machine you are also using.

## Levers, in order of cost

### 1. Frame window (free)

The single biggest lever this application controls. Fewer frames in flight, less
VRAM.

```yaml
# config/app.yaml
backend:
  frame_window: 4        # try 4, then 2, then 1
```

Or per job via `RenderSettings.frame_window`. `frame_window: 1` is always valid —
the pipeline is per-frame at heart, and windows are an optimisation.

### 2. Low-VRAM mode and CPU offload (free, costs time)

```yaml
# RenderSettings defaults; both on by default
low_vram_mode: true
cpu_offload: true
```

Start ComfyUI with matching flags:

```powershell
python main.py --listen 127.0.0.1 --lowvram
# or, when even that is too much:
python main.py --listen 127.0.0.1 --novram
```

`--lowvram` moves model parts between VRAM and RAM as needed. Slower per frame,
but it fits. With 32GB of system RAM you have plenty of room to offload into.

### 3. Tiling (free, watch for seams)

```yaml
tile_size_px: 512
tile_overlap_px: 64
```

Also enable tiled VAE decode in ComfyUI — VAE decode at 1080×1920 is often the
actual peak, not the sampler. Overlap below ~32px tends to show seams; if you
see banding at tile boundaries, raise the overlap before anything else.

### 4. Quantisation (free, small quality cost)

```yaml
quantization: "fp8"     # or int8, nf4 — whatever your workflow supports
```

FP8 roughly halves weight memory against FP16 with modest quality loss. Set it
in the workflow's loader nodes; record it here so the manifest carries it.

### 5. Render fewer frames at a time (free)

```bash
app job render <job_id> --max-frames 20
app job resume <job_id>
```

Each invocation is a fresh process, so VRAM is fully released between slices.
This is also the pragmatic way to run a long render on a machine you are using
for other things.

### 6. Lower the working resolution (quality cost)

Render the reveal at a lower resolution and let the final encode upscale. The
delivery contract stays 1080×1920; only the intermediate changes. Expect softer
garment detail — try everything above first.

## Diagnosing

```powershell
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1
```

Watch during a single frame. Where the peak lands tells you which lever to pull:

| Peak during | Lever |
| --- | --- |
| Model load | quantisation, `--lowvram` |
| Sampling | frame window, tiling |
| VAE decode | tiled VAE decode (very common culprit) |
| Steady growth across frames | a leak in a custom node; restart ComfyUI between slices |

```bash
app job backends       # reports ComfyUI's view of VRAM total/free
app doctor             # GPU info without requiring CUDA
```

## Symptoms

### `CUDA out of memory`

Work the list above in order. Start at `frame_window: 1` plus `--lowvram` plus
tiled VAE to establish *something* that completes, then relax one lever at a
time until it breaks. Knowing your actual ceiling is worth the twenty minutes.

### Progressive slowdown

Usually VRAM thrashing: the model is being shuttled to RAM every frame. Either
free VRAM (close the browser — a modern browser can hold 1–2GB) or accept it and
render in slices.

### ComfyUI dies mid-render

The job is resumable. Restart ComfyUI, then:

```bash
app job resume <job_id>
```

`resume()` interrupts any stale prompt before re-preparing, so a leftover prompt
cannot write frames behind the pipeline's back.

### Frames render but the garment is low quality

Check whether a VRAM-saving lever caused it: aggressive quantisation, tiles that
are too small, or a resolution reduction. Re-render a single frame with the
levers relaxed and compare before concluding the model is at fault.

### Tile seams

Raise `tile_overlap_px` (64, then 96). If seams persist, the workflow is tiling
at a stage where tiling is not safe — tile the VAE decode, not the sampler, or
render at a lower resolution instead.

## Realistic expectations

At 1080×1920 on an 8GB card, a diffusion-based garment workflow will land
roughly in the 10–40 s/frame range depending on steps and offload. For a
210-frame reveal that is 35 minutes to 2.5 hours per outfit.

Plan around it:

* Iterate with `--max-frames 1` while tuning the workflow.
* Validate with the mock backend, which is instant.
* Batch real renders overnight.
* Never re-render what is already done — `app job resume` exists for this.

Measure your actual per-frame time on a single frame before starting a batch,
and multiply. The number is usually larger than the estimate.
