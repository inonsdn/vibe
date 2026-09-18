# Compatibility rules

The rule engine runs **before** any backend touches a frame. It is
deterministic, config-driven and cheap: an impossible outfit is rejected in
milliseconds instead of after twenty minutes of GPU time.

```bash
app compatibility check --template <id> --garment <id> \
    --exposed-view front --exposed-view back --exposed-view side
```

Exit code `0` for `READY`, `3` for anything blocking.

## States

| State | Meaning | Render? |
| --- | --- | --- |
| `READY` | Every blocking and needs-input rule passed. | yes |
| `NEEDS_INPUT` | The operator must supply something — a missing view, an expansion mask, product metadata. | blocked |
| `INCOMPATIBLE` | A hard rule failed; this pairing cannot work on this performance. | blocked |

The aggregate is the worst outcome seen: any `fail` → `INCOMPATIBLE`, else any
`needs_input` → `NEEDS_INPUT`, else `READY`. Warnings never block.

Blocking is enforced in **two** places — `create_job` and `render_job` — so a
job created before a garment was edited cannot render afterwards.

## The rules

Defined in `config/compatibility_rules.v1.yaml`, implemented in
`src/app/pipeline/compatibility.py`. Each entry sets `enabled`, `severity`
(`warning` / `needs_input` / `blocking`), `weight` and rule-specific `params`.

| Rule | Default | What it catches |
| --- | --- | --- |
| `category_vs_template_class` | blocking | A garment category that cannot be placed over this base outfit (per `category_matrix`). |
| `required_body_coverage` | blocking | A garment covering less than its category implies, or less than the base outfit already covers — the original garment would show. |
| `sleeve_mismatch` | needs_input | Sleeves shorter than the base (blocking: the original sleeve shows), or much longer without an expansion mask. |
| `hem_length_mismatch` | needs_input | Same logic for hems. A crop top cannot hide a mid-thigh base garment. |
| `silhouette_expansion` | needs_input | A larger silhouette than the base. Beyond `blocking_steps` (4) no mask expansion can contain it on a fixed performance. |
| `transparent_garment_requirements` | needs_input | Transparency above `0.15` over a non-neutral base: the original outfit shows through. |
| `reflective_material_warning` | warning | Latex, sequin, metallic, satin, leather — specular highlights will not match the master lighting. |
| `high_flow_fabric_warning` | warning | Fabric that swings independently of the body. Above `0.95` it is blocking: a fixed performance cannot produce that motion. |
| `missing_views_for_motion` | needs_input | A required reference view is absent. If the exposed views are unknown, back and side are assumed required. |
| `hair_hand_occlusion_risk` | needs_input | No occlusion masks, so hair and hands crossing the garment would be painted over. |
| `base_garment_leakage_risk` | needs_input | Coverage below the base, or a light garment over a very dark base bleeding through at the boundary. |
| `source_image_quality` | needs_input | Reference images below `768px` min side, `0.5MP`, sharpness `0.15`, or aspect ratio above `3.0`. |
| `proportion_mismatch` | warning | Reference aspect ratio far from the body region's — usually an unusable flat-lay crop. Blocking beyond `3.5×`. |
| `incomplete_product_information` | needs_input | Metadata completeness below `0.6`, or undocumented usage rights. |

## Scales

Rules compare positions on ordered scales rather than special-casing pairs:

```
sleeve:     none < strap < cap < short < elbow < three_quarter < long
hem:        crop < waist < hip < mid_thigh < knee < midi < ankle < floor
coverage:   minimal < torso < torso_arms < torso_legs < full_body < full_body_limbs
silhouette: bodycon < fitted < straight < a_line < flared < oversized < voluminous
```

Each `TemplateClothingClass` maps to a base position on the sleeve, hem and
coverage scales, so "the garment must cover at least as much as the base" is a
single comparison. The deltas appear in each rule's `evidence`, which is what
makes a failure explicable rather than mysterious.

## Facts measured, not declared

`build_evaluation_context()` derives what the rules need from what is actually
on disk:

* `has_occlusion_masks` / `has_expansion_masks` — by listing the directories
* `base_garment_luminance` — mean luma inside the garment mask on the first
  masked reveal frame
* `body_region_aspect_ratio` — the garment mask's bounding box
* `exposed_views` — operator-supplied; `None` means unknown, and the rules
  assume a dance turns rather than assuming it does not

## Tuning

Change a threshold, not a code path:

```yaml
source_image_quality:
  enabled: true
  severity: needs_input     # warning | needs_input | blocking
  weight: 2.0
  params:
    min_image_min_side_px: 1024    # was 768
    min_megapixels: 1.0
```

Rules can be switched off entirely with `enabled: false`.

**Versioning.** Every report records `rules_version` *and*
`rules_file_sha256`, so an old report still says which rules judged it. For a
meaningful change, copy the file to `compatibility_rules.v2.yaml`, bump
`rules_version`, and point `compatibility.rules_file` at it rather than editing
a version existing reports already reference.

The loader is strict: a rule id it does not implement, or an unknown severity,
fails to load rather than being ignored.

## Confidence

```
confidence = 1 − Σ(penalty(outcome) × weight) / Σ(weight)
```

with default penalties `warn 0.35`, `needs_input 0.7`, `fail 1.0`. It is a
triage aid for ranking candidates, not a gate — the state decides.

## Overrides

A blocking state can be overridden by a human, with an audit trail:

```bash
app compatibility override <report_id> \
    --reviewer "your name" \
    --reason "the choreography never turns; the back is never visible" \
    --acknowledge missing_views_for_motion \
    --expires-in-hours 24
```

* The reason must be at least 10 characters — "ok" is not a reason.
* Overrides are additive: previous ones move to `override_audit_trail` and the
  rule results are never rewritten.
* `--expires-in-hours` makes it temporary; an expired override blocks again.
* `compatibility.allow_override: false` disables the mechanism entirely.
* Acknowledging `protected_mask_override` additionally permits editing inside
  protected regions — the only route to that, anywhere in the system.

Every override is written to `audit_events` with reviewer, reason and
timestamp, and the manifest records `compatibility_overridden: true`.
