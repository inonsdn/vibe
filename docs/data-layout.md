# Data formats and directory layout

## Design rules

1. **Absolute frame indices.** `frame_000090.png` is frame 90 of the master
   video in *every* directory. No per-segment renumbering anywhere.
2. **Lossless intermediates.** All frames and masks are PNG. Only the final
   delivery file is compressed.
3. **Records store relative paths.** Database rows hold data-root-relative
   paths (`templates/tpl_x/source_frames`), so a data directory can be moved or
   backed up without rewriting the database. Every read resolves through
   `DataRoot.resolve()`, which rejects traversal.
4. **Nothing generated is committed.** `data/.gitignore` excludes the whole
   tree.

## Directory layout

```
data/
├── templates/<template_id>/
│   ├── source_frames/          IMMUTABLE. frame_000000.png …
│   ├── masks_garment/          + README.txt describing the family
│   ├── masks_expansion/
│   ├── masks_protected/
│   ├── masks_occlusion/
│   ├── pose/                   frame_000000.json
│   ├── depth/                  frame_000000.png (16-bit) + scale.json
│   ├── optical_flow/           flow_000000_to_000001.npz
│   ├── face_landmarks/         frame_000000.json
│   ├── identity_refs/          operator-supplied stills
│   ├── background_plate/       clean plate, if available
│   ├── intro_cache/            byte copies of source_frames[intro range]
│   └── source_cfr.mp4          only if a VFR source was converted
│
├── garments/<garment_id>/images/
│   ├── front_<original>.png
│   ├── back_<original>.png
│   └── front_alpha_<original>.png   optional alpha mattes
│
├── jobs/<job_id>/
│   ├── raw_frames/             backend output, before compositing
│   ├── composited_frames/      after masking — these ship
│   ├── effective_masks/        the exact mask used, per frame
│   ├── assembly_frames/        intro + reveal, one contiguous sequence
│   ├── comfy_inputs/           per-frame files staged for ComfyUI
│   ├── contact_sheets/         contact_sheet_transition.png, …_reveal.png
│   ├── audio.m4a               copied from the master, trimmed
│   ├── manifest.json           the reproducibility contract
│   ├── qc_report.json
│   └── qc_report.txt
│
├── exports/<job_id>.mp4        final 1080×1920 H.264 yuv420p
├── exports/<job_id>_preview.mp4
├── cache/ · logs/ · tmp/
└── db/app.db
```

## File formats

| Artefact | Format | Notes |
| --- | --- | --- |
| Source frame | PNG, RGB24, lossless | `-compression_level 1` — speed, not size; PNG is lossless at any level |
| Mask | PNG, 8-bit grayscale | `0` immutable · `255` editable · `1–254` feather |
| Pose | JSON | `{"schema", "keypoints": [{"name","x","y","score"}]}` |
| Depth | 16-bit PNG + `scale.json` | relative depth is sufficient |
| Optical flow | `.npz` with `flow` | `(H, W, 2)` float32, frame N → N+1 |
| Face landmarks | JSON | `{"schema", "bbox": [x,y,w,h], "landmarks": [[x,y],…]}` |
| Final video | MP4 / H.264 / yuv420p | 1080×1920, CRF 17, `+faststart`, AAC 192k |
| Manifest | JSON | sorted keys, 2-space indent — diffable |
| QC report | JSON + text | same data, two audiences |

## Database schema

SQLite in WAL mode, forward-only migrations in `src/app/db/migrations/`.

| Table | Key | Purpose |
| --- | --- | --- |
| `human_templates` | `(id, version)` | Template records; a new version is a new row |
| `garment_assets` | `(id, version)` | Garment records |
| `compatibility_reports` | `id` | Rule results, state, confidence, overrides |
| `render_jobs` | `id` | Job definition + mutable execution state |
| `job_frames` | `(job_id, frame_index)` | Crash-safe per-frame completion — drives resume |
| `job_manifests` | `job_id` | Manifest payload + reproducibility digest |
| `audit_events` | autoincrement | Append-only: ingests, overrides, renders, deletions |
| `schema_migrations` | `version` | Applied migrations + their file hashes |

Each record is stored as canonical JSON in a `payload` column plus a handful of
promoted columns for indexing. The Pydantic model remains the single source of
truth for shape; reads always go back through `model_validate`, so a schema
change surfaces immediately instead of as a mysterious attribute error later.

Foreign keys are enforced (`PRAGMA foreign_keys=ON`) with `ON DELETE RESTRICT`
from jobs to templates and garments: a template cannot be deleted out from
under a job that references it.

## Identifiers

```
tpl_20260917T120000Z_9f3a1c     template
grm_20260917T120500Z_1b7e44     garment
cmp_20260917T120600Z_77aa01     compatibility report
job_20260917T120700Z_0c5d92     render job
```

Readable, lexicographically sortable by creation time, and filesystem-safe.
Every identifier used as a directory name passes `safe_identifier()`, which
rejects separators, `..`, and Windows reserved names (`CON`, `NUL`, `COM1`, …).

## Frame range conventions

All ranges are **half-open**: `[start, end)`, Python slice semantics.

```
intro  = [intro.start, anchor)      count = anchor − intro.start
reveal = [anchor, reveal.end)       count = reveal.end − anchor
```

`HumanTemplate`'s model validator requires `intro.end == anchor ==
reveal.start`. No gap and no overlap is representable, which is what makes the
"no off-by-one at the anchor" guarantee structural rather than aspirational.

## Manifest contents

```jsonc
{
  "schema_version": "1",
  "job_id": "job_…",
  "template_id": "tpl_…", "template_version": 1,
  "template_source_sha256": "…",
  "garment_id": "grm_…", "garment_version": 1,
  "compatibility_report_id": "cmp_…",
  "compatibility_state": "READY",
  "compatibility_overridden": false,
  "frame_range": {"start": 90, "end": 300},
  "transition_anchor_frame": 90,
  "intro_frame_count": 90, "reveal_frame_count": 210, "flash_frames": 0,
  "video_width": 1080, "video_height": 1920, "video_fps": 30.0,
  "pixel_format": "yuv420p", "video_codec": "libx264",
  "prompt": "", "negative_prompt": "",
  "settings": { "…": "every render setting, verbatim" },
  "input_hashes": {
    "template_source_video": "…", "template_source_frames_dir": "…",
    "garment_image_front_…": "…", "masks_garment": "…",
    "masks_protected": "…", "intro_cache_dir": "…"
  },
  "output_hashes": {"final_video": "…", "preview_video": "…"},
  "frame_checksums": [{"index": 0, "sha256": "…", "source": "intro_cache"}],
  "reproducibility": {
    "app_version": "0.1.0", "preprocessing_version": "1",
    "git": {"commit": "…", "branch": "…", "dirty": false},
    "platform": {"python": "3.11.x", "system": "Windows"},
    "dependencies": {"pydantic": "2.x", "numpy": "…"},
    "ffmpeg_version": "…", "ffprobe_version": "…",
    "config_hash": "…", "rules_file_sha256": "…",
    "workflow_id": null, "workflow_sha256": null,
    "backend_name": "mock", "backend_version": "1.0.0",
    "seed": 1234, "seed_strategy": "derived",
    "per_frame_seeds": {"90": 123456789},
    "ffmpeg_commands": [["ffmpeg", "-i", "…"]]
  },
  "qc": {"passed": true, "checks_total": 15, "checks_failed": 0}
}
```

`frame_checksums[].source` is one of `intro_cache`, `rendered` or `flash`, so
the manifest itself proves which frames came from the cached intro.

`reproducibility_digest()` folds the reproducibility-critical subset into one
hash. It deliberately excludes wall-clock timestamps, machine details and the
hashes of *compressed* outputs — encoders are not bit-identical across builds —
while including every lossless frame checksum, which must match exactly.
