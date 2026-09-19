# DWPose ONNX setup (Windows 11 + RTX 5060 8GB)

The first real local model in this system. It turns a reference clip into
canonical COCO-17 pose JSON that the Motion Composition pipeline already knows
how to normalize, join, bridge and QC.

**Nothing here is downloaded by the application.** You place two ONNX files on
the machine and point the configuration at them. A missing file produces a
refusal that names the path; it never falls back to synthetic poses.

---

## 1. What you need

| Thing | Why |
| --- | --- |
| Python 3.11 | The pinned runtime. See [`windows-setup.md`](windows-setup.md). |
| FFmpeg + ffprobe on `PATH` | Clip probing and the diagnostic previews. |
| `onnxruntime-gpu` (CUDA) or `onnxruntime` (CPU) | Not a hard dependency of the app — installed separately, by you. |
| A person detector `.onnx` | Finds people. DWPose is normally paired with a YOLOX-style COCO detector. |
| A DWPose / RTMPose whole-body `.onnx` | Produces the keypoints. |

### Model filenames this integration expects

These are the files the defaults are tuned for. Any equivalent export works —
the sizes are configuration, not assumptions baked into code.

| Role | Expected filename | Input size | Config key |
| --- | --- | --- | --- |
| Person detector | `yolox_l.onnx` | 640×640 | `pose.detector_model` |
| Whole-body pose | `dw-ll_ucoco_384.onnx` | 288×384 | `pose.pose_model` |

Smaller alternatives that also work, with their sizes set explicitly:

| File | Set |
| --- | --- |
| `yolox_m.onnx` / `yolox_s.onnx` | `pose.detector_input_size: [640, 640]` |
| `dw-ll_ucoco.onnx` | `pose.pose_input_size: [192, 256]` |
| `rtmpose-m_simcc-coco_*.onnx` (body only, 17 keypoints) | `pose.pose_input_size: [192, 256]`, `pose.emit_hands: false` |

Obtain them yourself from the DWPose project's published releases, your
organisation's model store, or by exporting them from the upstream checkpoints.
Put them somewhere stable, for example `C:\models\dwpose\`.

> If the pose model's input size does not match `pose.pose_input_size`, every
> joint is silently distorted — the crop is resized to the wrong shape and the
> inverse mapping cannot undo it. Set it deliberately; do not guess.

---

## 2. Install

```powershell
# In the repository root, with the virtual environment active.
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]" -c constraints\tested-py311.txt

# ONNX Runtime is NOT a dependency of this package: you choose GPU or CPU.
# For the RTX 5060, pick the build matching your installed CUDA runtime.
pip install onnxruntime-gpu
# CPU-only fallback, or a machine without CUDA:
# pip install onnxruntime
```

Check what the build actually offers before blaming the application:

```powershell
python -c "import onnxruntime; print(onnxruntime.__version__); print(onnxruntime.get_available_providers())"
```

You want `CUDAExecutionProvider` in that list. If it is absent, the GPU path
cannot work no matter what this repository does, and `--provider cuda` will
say so rather than crawling on CPU for an hour.

---

## 3. Configure

Copy the template and edit it. `config/local.yaml` is gitignored, so your
machine's paths never reach the repository.

```powershell
copy config\local.example.yaml config\local.yaml
notepad config\local.yaml
```

```yaml
pose:
  adapter: dwpose_onnx
  detector_model: "C:/models/dwpose/yolox_l.onnx"
  pose_model: "C:/models/dwpose/dw-ll_ucoco_384.onnx"
  provider: auto
  pose_input_size: [288, 384]
```

Forward slashes are fine on Windows and avoid backslash escaping.

Equivalent without editing a file:

```powershell
$env:APP_POSE__DETECTOR_MODEL = "C:\models\dwpose\yolox_l.onnx"
$env:APP_POSE__POSE_MODEL     = "C:\models\dwpose\dw-ll_ucoco_384.onnx"
$env:APP_POSE__PROVIDER       = "cuda"
```

Or per-invocation with `--detector-model` / `--pose-model` / `--provider`,
which beat both.

---

## 4. Smoke test, before touching the real clips

Each step is meant to fail in a legible way if something is wrong.

```powershell
# 1. The application runs at all, and reports what it can and cannot do.
app doctor
app offline verify

# 2. The pose adapter's own view. With no models configured this says
#    "missing_weights" and names the paths; with them configured, "available".
app doctor --json | python -c "import json,sys; print(json.load(sys.stdin)['adapters']['pose'])"

# 3. The full pipeline with NO model at all, using the deliberately synthetic
#    mock adapter. Proves ffmpeg, the database and the pipeline are healthy.
app motion ingest .\clip_a.mp4 --name "Smoke" --start-frame 0 --end-frame 60 `
    --motion-use-authorized --rights-holder "you" --license "internal"
app motion extract-pose <motion_id> --adapter mock
app motion inspect <motion_id>

# 4. The real adapter on a short range. Start with 60 frames, not 600.
app motion extract-pose <motion_id> `
    --adapter dwpose_onnx `
    --detector-model C:\models\dwpose\yolox_l.onnx `
    --pose-model C:\models\dwpose\dw-ll_ucoco_384.onnx `
    --provider cuda `
    --diagnostics

# 5. LOOK AT THE OVERLAY before extracting the whole clip.
start data\diagnostics\motion\<motion_id>\overlay.mp4
```

### What each smoke-test step tells you

| Symptom | Meaning | Fix |
| --- | --- | --- |
| `missing_weights`, path named | The file is not where you said | Fix the path in `config/local.yaml` |
| `not an .onnx file` | A `.pth` or a folder was configured | Export to ONNX, or point at the right file |
| `onnxruntime is not installed` | No runtime in this venv | `pip install onnxruntime-gpu` |
| `CUDA was requested but…` | CPU-only onnxruntime build | Install `onnxruntime-gpu`, or accept `--provider cpu` |
| `WARNING: ran on CPU after asking for CUDA` | The build has CUDA but bound CPU | CUDA/cuDNN version mismatch; check the onnxruntime matrix |
| `Detector output row count does not match the stride pyramid` | `detector_input_size` is wrong for this model | Set it to the size the ONNX was exported at |
| `Unrecognised keypoint layout` | Not a 17/26/133-keypoint model | Use a COCO or COCO-WholeBody export |
| Skeleton on the wrong person in the overlay | Subject selection needs tuning | See **Subject selection**, below |
| `ended before every requested frame was read` | `--end-frame` is past the clip | Check `app motion inspect` for the real frame count |

---

## 5. The two real clips, end to end

Replace the paths, the ranges and the rights metadata with yours.

```powershell
# --- clip A -------------------------------------------------------------
app motion ingest "C:\refs\clip_a.mp4" `
    --name "Motion A" --start-frame 0 --end-frame 168 `
    --motion-use-authorized --rights-holder "<who>" --license "<terms>"
# note the printed motion id, e.g. mot_20260919T101500Z_ab12cd

app motion extract-pose mot_<A> `
    --adapter dwpose_onnx `
    --detector-model C:\models\dwpose\yolox_l.onnx `
    --pose-model C:\models\dwpose\dw-ll_ucoco_384.onnx `
    --provider cuda `
    --diagnostics

app motion inspect mot_<A>

# --- clip B -------------------------------------------------------------
app motion ingest "C:\refs\clip_b.mp4" `
    --name "Motion B" --start-frame 12 --end-frame 210 `
    --motion-use-authorized --rights-holder "<who>" --license "<terms>"

app motion extract-pose mot_<B> `
    --adapter dwpose_onnx `
    --detector-model C:\models\dwpose\yolox_l.onnx `
    --pose-model C:\models\dwpose\dw-ll_ucoco_384.onnx `
    --provider cuda `
    --diagnostics

app motion inspect mot_<B>

# --- then the existing pipeline, unchanged -------------------------------
app motion normalize mot_<A>
app motion normalize mot_<B>
app motion match-anchors --prev mot_<A> --next mot_<B> --top 10
app motion compose --name "Prototype v1" --segment mot_<A> --segment mot_<B> --bridge-frames 12
app motion preview <composition_id>
app motion qc <composition_id>
```

Once `app motion qc` passes, the clips have done their job: everything
downstream consumes pose JSON and never touches the reference footage again.

---

## 6. Diagnostics

`--diagnostics` writes to:

```
data\diagnostics\motion\<motion_source_id>\
├── overlay.mp4              skeleton + candidate boxes on the reference clip
├── skeleton.mp4             the same skeleton on a flat background
├── subject_boxes.json       every candidate per frame, with its scored terms
├── confidence_summary.json  per-joint confidence and coverage
├── missing_joints.json      missing runs, and frames with no subject
├── extraction.json          provider, model hashes, ROI, settings
└── README.txt              "this is not a composition input"
```

**`overlay.mp4` contains pixels from the reference clip.** That is deliberate —
you cannot judge tracking without seeing it on the real footage. It lives in a
tree the pipeline never reads, and `app motion qc` independently verifies that
no image file exists in any composition input. Pass `--no-overlay` to skip it
entirely; the other artefacts contain no imagery.

`skeleton.mp4` is safe to share: it is drawn from pose JSON alone.

---

## 7. Subject selection

The reference clips are Instagram screen recordings, so a person detector finds
more than the dancer: header avatars, suggested-account thumbnails, a cartoon
parked in a corner, sometimes a second person in the background.

Selection is a deterministic score, not a learned tracker:

```
score = 0.35·area + 0.25·centrality + 0.25·IoU-with-previous + 0.15·continuity
```

with two hard gates applied first — `subject_min_area_fraction` (an avatar is
tiny) and `subject_max_center_distance` (a corner image is not the subject).
Detection confidence is deliberately **not** a term: a crisp cartoon in a corner
routinely scores higher than a motion-blurred dancer.

Once locked on, a rival must beat the incumbent by `subject_switch_margin` to
take over, so a momentarily larger background figure cannot steal the track.

If `subject_boxes.json` shows the wrong pick:

| Problem | Turn this | Direction |
| --- | --- | --- |
| A UI avatar is being tracked | `subject_min_area_fraction` | up (0.03–0.06) |
| A corner image is being tracked | `subject_max_center_distance` | down (0.5–0.7) |
| The track jumps between two people | `subject_switch_margin` | up (0.25–0.4) |
| The dancer is dropped when partly out of frame | `subject_max_coast_frames` | up |
| The dancer is rejected as too small | `subject_min_area_fraction` | down |

A crop is usually better than tuning. `roi_mode: normalized` with
`roi: [0.0, 0.12, 1.0, 0.76]` removes the header and footer chrome before
detection ever runs, which is cheaper and more reliable than out-scoring it
afterwards. Coordinates in the exported pose JSON are always in **original
video pixels** — the crop is internal and reversed before anything is written.

---

## 8. Temporal cleanup

Three rules, in order:

1. Gaps up to `max_interpolation_gap` frames are linearly interpolated, and the
   filled joints carry reduced confidence so downstream code can tell them from
   measured ones.
2. Longer runs stay missing. Interpolating across twenty frames invents motion;
   the existing motion QC already knows how to report and reject long runs, and
   hiding them would remove the signal that a clip is unusable.
3. Smoothing is confidence-weighted, never reaches across a gap, and **stands
   off fast joints**: above `fast_motion_px` of per-frame travel the smoothing
   weight falls to zero. A dancer's wrist crossing 40 px in a frame is signal.
   Averaging it shortens the arc, which is the most visible artefact in the
   final video.

Lower `fast_motion_px` if hands look mushy; raise `smoothing_window` if a
slow-moving hip jitters.

---

## 9. What is recorded, and why

Every extraction writes into the motion source record, and from there into the
composition manifest:

* the **execution provider actually used**, per session, plus what the build
  offered — a silent CPU fallback looks identical in the output and is roughly
  twenty times slower,
* the **SHA-256 of both ONNX files**, so a manifest identifies the weights
  rather than a filename someone could overwrite,
* the resolved ROI, the subject-tracking summary (switches, rejection counts)
  and what the temporal cleanup did.

Read it back with `app motion inspect <motion_id> --json`.

---

## 10. Known limits

* **CUDA is unverified here.** The test suite runs against injected fake ONNX
  sessions, so every decode path, the tracker and the cleanup are exercised —
  but no real weights and no GPU have been run. The first real run on the RTX
  5060 is the smoke test above, and it may need the input sizes corrected.
* **Accuracy is the model's, not ours.** This integration maps and cleans up
  what DWPose returns. Occlusion, motion blur and a dancer leaving frame are
  the model's failure modes, and the diagnostics exist to make them visible.
* **No character animation and no garment generation.** This commit does pose
  extraction only.
