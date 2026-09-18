# Mask semantics

Masks are the safety mechanism of this system. They decide, per pixel, whether
the renderer is allowed to have an opinion.

## Format

* 8-bit **grayscale PNG**, single channel
* one file per frame, named `frame_{index:06d}.png` using the **absolute frame
  index in the master video**
* dimensions must match the source frame exactly (validated on import and again
  before every render)

## Values

| Value | Name | Meaning |
| ----: | --- | --- |
| `0` | immutable | The output pixel **must** equal the source pixel, byte for byte. |
| `255` | editable | The backend's pixel replaces the source pixel entirely. |
| `1`–`254` | feather | Linear blend: `out = source·(1−α) + rendered·α`, `α = value/255`. |

Two configured thresholds describe the fully-saturated bands
(`config/app.yaml → mask`):

```yaml
immutable_threshold: 2     # ≤ 2   is treated as fully immutable
editable_threshold: 253    # ≥ 253 is treated as fully editable
```

## The four mask families

| Family | Directory | What belongs in it | 255 means |
| --- | --- | --- | --- |
| **garment** | `masks_garment/` | The base outfit region to be replaced. | replace this |
| **expansion** | `masks_expansion/` | Extra area a *larger* garment may occupy — longer sleeves, a fuller skirt, a hood. | may also be replaced |
| **protected** | `masks_protected/` | Face, hair, hands, exposed skin, legs, background. | never touch this |
| **occlusion** | `masks_occlusion/` | Things in **front** of the garment: hair strands, hands, held props. | keep the original |

Only `garment` is strictly required to render a frame. `protected` is required
for a template to pass validation, because without it identity preservation
cannot be verified numerically — and a system that cannot prove it preserved the
performer is not doing its job.

## Combination order

For each frame, `build_effective_mask()` computes one mask and the compositor
uses only that:

```
1.  editable = max(garment, dilate(expansion, expansion_dilate_px))
2.  editable = editable − occlusion                 # saturating subtract
3.  editable = editable − dilate(protected, protected_dilate_px)
4.  editable = feather(editable, feather_radius_px) # softened INWARD only
5.  editable[dilate(protected) > 0] = 0             # final hard clamp
```

Three properties follow from this order:

* **Protected always wins.** Step 3 removes protected pixels and step 5 clamps
  them again *after* the blur, so feathering can never reopen one.
* **Feathering cannot grow the mask.** `feather()` blurs an *eroded* copy and
  then takes `min(blurred, original)`, so the result is always a subset of its
  input. An edge softens inward, never outward into a neighbour.
* **Every decision is auditable.** The effective mask is written to
  `jobs/<job_id>/effective_masks/frame_NNNNNN.png`. If a render looks wrong,
  that file shows exactly what the compositor was permitted to change.

Overlaps are not silently resolved: they are recorded as `MaskConflict` entries
(`editable_over_protected`, `editable_over_occlusion`) with pixel counts, and
those counts appear in the job's metrics.

## Compositing

```python
alpha  = effective_mask / 255.0
out    = round(source·(1 − alpha) + rendered·alpha)
out[effective_mask <= 0] = source[effective_mask <= 0]   # hard restore
```

The hard restore runs **after** the blend, so floating-point rounding cannot
perturb an immutable pixel by ±1. `composite_with_stats()` then measures
`changed_outside_mask`; the render orchestrator treats any non-zero value as a
bug and fails the job rather than shipping the frame.

## Overriding protection

There is exactly one way to let an editable mask beat a protected mask:

1. a compatibility report in a blocking state,
2. a stored `ReviewerOverride` with a reviewer, a substantive reason (10+
   characters) and a timestamp,
3. that override's `acknowledged_rule_ids` explicitly containing
   `protected_mask_override`.

```bash
app compatibility override <report_id> \
    --reviewer "your name" \
    --reason "deliberate stylised edit across the shoulder line" \
    --acknowledge protected_mask_override
```

Anything short of all three and protected pixels stay protected. The override is
written to the audit log with the reviewer, reason and timestamp, and the
manifest records `compatibility_overridden: true`.

## Area sanity limits

Before rendering a frame, the effective mask's area is checked against
`config/app.yaml → mask`:

```yaml
min_editable_area_fraction: 0.0005   # below this: nothing to render → error
max_editable_area_fraction: 0.60     # above this: protected masks likely missing → error
```

The upper bound exists to catch the dangerous failure: an operator forgets the
protected masks, the editable region covers the whole performer, and the render
"succeeds" while replacing the person. The job fails instead.

## Authoring masks today

No mask-generation model is installed, so masks are authored externally and
imported:

```bash
app template import-masks <template_id> --kind garment   --from ./out/garment
app template import-masks <template_id> --kind protected --from ./out/protected
app template import-masks <template_id> --kind occlusion --from ./out/occlusion
app template import-masks <template_id> --kind expansion --from ./out/expansion
```

Import validates every file (readable, correct dimensions, parseable frame
index, within the template's frame count) *before* copying any of them, so a
bad batch cannot half-land. `app template inspect` then reports, per family,
how many reveal frames have a mask and which are missing.

Practical notes from authoring these by hand:

* Cover the **whole** reveal range. A missing mask at one frame fails the render
  at that frame, not at submission.
* Be generous with `protected`. A mask that is slightly too large costs a little
  garment coverage; one that is slightly too small costs the performer's hand.
* `protected_dilate_px` (default 2) adds an automatic safety margin, so you do
  not need to hand-pad every edge.
* Keep `garment` tight to the real garment and put the growth in `expansion`.
  That keeps the compatibility rules meaningful — they check for an expansion
  mask when a garment needs one.
* Author `occlusion` wherever hair or a hand crosses the garment. Without it,
  the renderer paints over them and QC will flag the flicker, not the cause.

Adapter interfaces exist for SAM 2-style video mask propagation, human parsing,
pose, DensePose, depth, optical flow and face landmarks
(`src/app/adapters/`). They are documented stubs that **raise** when called;
they never fabricate output. See
[`model-selection-checklist.md`](model-selection-checklist.md).
