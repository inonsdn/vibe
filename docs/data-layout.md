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
├── motion_sources/<motion_source_id>/
│   ├── pose/                   frame_000000.json — pose data ONLY
│   └── source_placeholder.bin  (tests) — no reference imagery is ever copied
│
├── compositions/<composition_id>/
│   ├── normalized_poses/segment_<NNN>_<motion_source_id>/  per-SEGMENT canonical poses
│   ├── bridge_poses/           generated bridge poses
│   ├── composed_poses/         the final contiguous sequence (0..N-1)
│   ├── preview.mp4             skeleton preview, drawn from poses
│   ├── manifest.json
│   └── qc_report.{json,txt}
│
├── heroes/<hero_id>/images/    Hero Character reference images
│
├── masters/<candidate_id>/
│   ├── frames/                 candidate master frames (the source of truth)
│   ├── master_archive.mp4      archival only, written at promotion; never read back
│   ├── comfy_inputs/chunk_NNNN/
│   │   ├── pose_sequence/      pose_00000.png … + sequence.json (ordinal → frame)
│   │   ├── context_sequence/   context_00000.png … + sequence.json
│   │   └── poses/              pose_NNNNNN.json, the audit trail
│   ├── manifest.json
│   └── qc_report.{json,txt}
│
├── exports/<job_id>.mp4        final 1080×1920 H.264 yuv420p
├── exports/<job_id>_preview.mp4
├── cache/ · logs/ · tmp/
└── db/app.db
```

## Promoted synthetic masters

`app master promote` builds `templates/<template_id>/` with exactly the layout
above. Its `source_frames/` are **hardlinks (or byte copies) of
`masters/<candidate_id>/frames/`** — the PNGs the animator generated, unchanged.
`source_frames_sha256` is computed over those files and re-checked before every
render.

`master_archive.mp4` is written for operators to watch and is what the
template's `source_video_path` points at, but nothing in the render path decodes
it. Encoding to H.264 and reading back would quantise and chroma-subsample
precisely the pixels the pipeline promises to restore outside the editable mask.

## File formats

| Artefact | Format | Notes |
| --- | --- | --- |
| Source frame | PNG, RGB24, lossless | `-compression_level 1` — speed, not size; PNG is lossless at any level |
| Mask | PNG, 8-bit grayscale | `0` immutable · `255` editable · `1–254` feather |
| Pose | JSON | `{"schema", "keypoints": [{"name","x","y","score"}]}` |
| Depth | 16-bit PNG + `scale.json` | relative depth is sufficient |
| Optical flow | `.npz` with `flow` | `(H, W, 2)` float32, frame N → N+1 |
| Face landmarks | JSON | `{"schema", "bbox": [x,y,w,h], "landmarks": [[x,y],…]}` |
| Pose frame | JSON | `{"schema_version", "frame_index", "timestamp_s", "space", "origin", "body": {joint: {x, y, confidence}}, "source_bbox"}` |
| Skeleton preview | MP4 / H.264 | Drawn from poses; contains no source imagery |
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
| `motion_sources` | `(id, version)` | Motion references + pose provenance |
| `skeleton_profiles` | `(id, version)` | Canonical profiles, cached from config |
| `hero_characters` | `(id, version)` | Hero Character records |
| `motion_compositions` | `(id, version)` | Segments, joins, bridges |
| `master_candidates` | `id` | Candidate masters + acceptance state |
| `master_chunks` | `(candidate_id, chunk_index)` | Crash-safe chunk completion — drives resume |
| `master_manifests` | `candidate_id` | Master manifest + digest |
| `audit_events` | autoincrement | Append-only: ingests, overrides, renders, acceptances, deletions |
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
mot_20260917T120800Z_44be10     motion source
cmp_20260917T120900Z_1d90ab     motion composition (and compatibility report)
hero_20260917T121000Z_bb2f31    Hero Character
mst_20260917T121100Z_3e77c4     master candidate
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


## Motion frame numbering

Two numbering schemes exist, and they are deliberately kept apart until the
composition assigns output indices:

* **Source numbering** — a frame's index in its own reference clip. Pose files
  under `motion_sources/<id>/pose/` and `compositions/<id>/normalized_poses/`
  use this.
* **Output numbering** — the composition's own `0..N-1`. Files under
  `composed_poses/`, `bridge_poses/` and `masters/<id>/frames/` use this.

Mixing them early is how off-by-one bugs get in, so normalization keeps each
segment in its source numbering and only `assemble_sequence` maps to output
indices. The layout it produces is documented in
[`motion-composition.md`](motion-composition.md#the-frame-layout-at-a-join).

## Composition manifest

```jsonc
{
  "schema_version": "1",
  "composition_id": "cmp_…",
  "output_fps": 30.0,
  "output_frame_count": 214,
  "skeleton_profile": { "canonical_shoulder_width": 300.0, "…": "…" },
  "segments": [
    {
      "motion_source": "mot_…@v1",
      "effective_range": [0, 168],
      "playback_speed": 1.0,
      "canonical_transform": {
        "base_scale": 3.33, "source_shoulder_width": 90.0,
        "source_torso_length": 130.0, "interpolated_frames": [], "stats": {}
      }
    }
  ],
  "joins": [{ "prev_source_frame": 150, "next_source_frame": 20, "…": "…" }],
  "bridge_settings": [{ "interpolation": "cubic_hermite", "frame_count": 12 }],
  "input_hashes": {
    "motion_source_video::mot_…@v1": "…",
    "motion_source_pose::mot_…@v1": "…"
  },
  "contains_source_pixels": false,
  "digest": "…"
}
```

The digest excludes wall-clock timestamps (including the profile record's own
`created_at`), so two identical compositions hash identically — otherwise the
digest would be a label rather than a reproducibility claim.
