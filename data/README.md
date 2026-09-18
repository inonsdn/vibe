# `data/` — runtime state

Everything in this directory is generated at runtime and is **excluded from
Git** (see `data/.gitignore`). No user media, generated frames, model weights
or secrets ever belong in version control.

## Layout

```
data/
├── templates/<template_id>/       one immutable Master Human Performance
│   ├── source_frames/             lossless PNG, frame_000000.png … (IMMUTABLE)
│   ├── masks_garment/             the garment region to replace
│   ├── masks_expansion/           extra area a larger garment may occupy
│   ├── masks_protected/           face / hair / hands / skin / background
│   ├── masks_occlusion/           things IN FRONT of the garment
│   ├── pose/                      per-frame keypoints (optional)
│   ├── depth/                     per-frame depth (optional)
│   ├── optical_flow/              frame-to-frame flow (optional)
│   ├── face_landmarks/            per-frame landmarks + bbox (optional)
│   ├── identity_refs/             identity reference stills
│   ├── background_plate/          clean background plate, if available
│   └── intro_cache/               byte-copied intro frames, reused by every job
│
├── motion_sources/<motion_source_id>/
│   └── pose/                      frame_000000.json — POSE DATA ONLY
│                                  (no reference imagery is ever copied here)
│
├── compositions/<composition_id>/
│   ├── normalized_poses/<src>/    per-source canonical poses
│   ├── bridge_poses/              generated bridge poses
│   ├── composed_poses/            the contiguous output sequence (0..N-1)
│   ├── preview.mp4                skeleton preview (drawn from poses)
│   ├── manifest.json
│   └── qc_report.{json,txt}
│
├── heroes/<hero_id>/images/       Hero Character reference images
│
├── masters/<candidate_id>/
│   ├── frames/                    candidate master frames
│   ├── manifest.json
│   └── qc_report.{json,txt}
│
├── garments/<garment_id>/
│   └── images/                    front_*.png, back_*.png, side_*.png …
│
├── jobs/<job_id>/
│   ├── raw_frames/                what the backend returned (pre-composite)
│   ├── composited_frames/         after masking — the frames that ship
│   ├── effective_masks/           the exact mask used per frame (auditable)
│   ├── assembly_frames/           cached intro + rendered reveal, joined
│   ├── comfy_inputs/              per-frame files staged for ComfyUI
│   ├── contact_sheets/            transition + periodic reveal sheets
│   ├── manifest.json              the reproducibility contract
│   ├── qc_report.json             machine-readable QC
│   └── qc_report.txt              human-readable QC
│
├── exports/                       final H.264 videos + previews
├── cache/                         reusable intermediates
├── logs/                          structured JSON logs (app.jsonl)
├── tmp/                           scratch space
└── db/app.db                      SQLite catalogue (WAL mode)
```

## Rules

1. **`source_frames/` is immutable.** Its SHA-256 is recorded at ingestion and
   re-verified before every render and compose. Editing a frame there makes the
   template unrenderable until it is re-ingested — by design.
2. **`intro_cache/` is byte-copied, never re-encoded.** That is what makes the
   intro identical across every outfit.
3. **Frame files are named by their absolute index in the master video**, so an
   index means the same thing in every directory.
4. Deleting a job directory is safe; deleting a template directory invalidates
   every job that referenced it.
5. **Motion artifacts contain pose JSON only.** A motion reference's pixels are
   never copied anywhere under `data/`, and the
   `no_source_pixels_in_motion_artifacts` QC check fails if an image or video
   file appears in a composition's pose directories.
6. An **accepted** master candidate is immutable: re-animating it is refused.

See `docs/data-layout.md` for the full specification and `docs/mask-semantics.md`
for the mask value conventions.
