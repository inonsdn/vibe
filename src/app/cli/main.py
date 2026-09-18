"""Command-line interface.

Every command builds a :class:`~app.pipeline.context.ServiceContext` and calls
the same pipeline functions the HTTP API calls. Nothing is implemented twice.

Output is human-readable by default and JSON with ``--json``, so the CLI is
scriptable without a second code path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer

from app.backends.registry import available_backends, create_backend
from app.core.config import AppConfig, load_config
from app.core.errors import AppError
from app.core.ids import utc_now
from app.core.logging import configure_logging
from app.domain.enums import (
    BodyCoverage,
    GarmentCategory,
    GarmentLength,
    ImageViewType,
    JobStatus,
    MaskKind,
    Material,
    Silhouette,
    SleeveLength,
    TemplateClothingClass,
)
from app.pipeline import compat_service
from app.pipeline.compose import ComposeOptions, compose_job
from app.pipeline.context import ServiceContext
from app.pipeline.garment_ingest import GarmentIngestOptions, ImageSpec, ingest_garment
from app.pipeline.render import JobCreateOptions, create_job, render_job, resume_job
from app.pipeline.template_ingest import (
    IngestOptions,
    import_masks,
    ingest_template,
    validate_template,
)
from app.qc.report import QCOptions, run_qc
from app.version import APP_NAME, APP_VERSION

app = typer.Typer(
    name="app",
    help=(
        "Local, model-agnostic video garment replacement.\n\n"
        "The human performance is immutable: only the garment region of the "
        "reveal segment is ever re-rendered. Start with `app doctor`, then "
        "`app template ingest`."
    ),
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)

template_app = typer.Typer(
    help="Ingest and inspect Master Human Performance templates.", no_args_is_help=True
)
garment_app = typer.Typer(help="Ingest and inspect garment reference assets.", no_args_is_help=True)
compat_app = typer.Typer(help="Run and override compatibility checks.", no_args_is_help=True)
job_app = typer.Typer(
    help="Create, render, resume, compose and QC render jobs.", no_args_is_help=True
)
offline_app = typer.Typer(
    help="Verify offline guarantees and the local environment.", no_args_is_help=True
)

app.add_typer(template_app, name="template")
app.add_typer(garment_app, name="garment")
app.add_typer(compat_app, name="compatibility")
app.add_typer(job_app, name="job")
app.add_typer(offline_app, name="offline")


# ---------------------------------------------------------------------------
# shared plumbing
# ---------------------------------------------------------------------------
class State:
    config_file: Path | None = None
    data_root: Path | None = None
    json_output: bool = False
    log_level: str | None = None


state = State()

JsonFlag = Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")]


@app.callback()
def main(
    config_file: Annotated[
        Path | None,
        typer.Option("--config", help="Path to an app.yaml (defaults to config/app.yaml)."),
    ] = None,
    data_root: Annotated[
        Path | None,
        typer.Option("--data-root", help="Override paths.data_root for this invocation."),
    ] = None,
    log_level: Annotated[
        str | None, typer.Option("--log-level", help="DEBUG, INFO, WARNING, ERROR.")
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable JSON from every command.")
    ] = False,
) -> None:
    """Global options."""
    state.config_file = config_file
    state.data_root = data_root
    state.json_output = json_output
    state.log_level = log_level


def _load_config() -> AppConfig:
    overrides: dict[str, Any] = {}
    if state.data_root is not None:
        overrides["paths"] = {"data_root": str(state.data_root)}
    if state.log_level is not None:
        overrides.setdefault("runtime", {})["log_level"] = state.log_level
    return load_config(state.config_file, overrides=overrides or None)


def _context() -> ServiceContext:
    config = _load_config()
    configure_logging(
        config.runtime.log_level,
        log_dir=(config.data_root().resolve("logs") if config.runtime.log_to_file else None),
    )
    return ServiceContext.create(config=config, configure_logs=False)


def _emit(payload: dict[str, Any], text: str | None = None, *, force_json: bool = False) -> None:
    if state.json_output or force_json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        typer.echo(text if text is not None else json.dumps(payload, indent=2, default=str))


def _fail(exc: AppError) -> None:
    payload = exc.to_dict()
    if state.json_output:
        typer.echo(json.dumps(payload, indent=2, default=str), err=True)
    else:
        typer.secho(f"error [{exc.code}]: {exc.message}", fg=typer.colors.RED, err=True)
        for key, value in exc.details.items():
            typer.echo(f"  {key}: {value}", err=True)
    raise typer.Exit(code=2)


def _run(function: Any, *args: Any, **kwargs: Any) -> Any:
    """Call a pipeline function, turning AppError into a clean CLI failure."""
    try:
        return function(*args, **kwargs)
    except AppError as exc:
        _fail(exc)
    except KeyboardInterrupt:
        typer.secho("interrupted; the job remains resumable", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=130) from None


# ---------------------------------------------------------------------------
# doctor / version
# ---------------------------------------------------------------------------
@app.command()
def version() -> None:
    """Print the application version."""
    _emit({"name": APP_NAME, "version": APP_VERSION}, f"{APP_NAME} {APP_VERSION}")


@app.command()
def doctor(
    json_output: JsonFlag = False,
    probe_comfyui: Annotated[
        bool, typer.Option("--probe-comfyui/--no-probe-comfyui", help="Probe localhost ComfyUI.")
    ] = True,
) -> None:
    """Check the local environment, backends, adapters and data paths."""
    from app.offline.verify import render_text_report, verify_offline

    config = _load_config()
    report = verify_offline(config, probe_comfyui=probe_comfyui)
    payload = report.as_dict()
    payload["backends_available"] = available_backends()
    payload["workflows_dir"] = str(config.workflows_dir())
    text = render_text_report(report)
    text += "\nBackends: " + ", ".join(available_backends()) + "\n"
    if state.json_output or json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        typer.echo(text)
    if not report.ok:
        raise typer.Exit(code=1)


@offline_app.command("verify")
def offline_verify(
    json_output: JsonFlag = False,
    probe_comfyui: Annotated[bool, typer.Option("--probe-comfyui/--no-probe-comfyui")] = True,
) -> None:
    """Verify that nothing requires the network and all local tools exist."""
    from app.offline.verify import render_text_report, verify_offline

    report = verify_offline(_load_config(), probe_comfyui=probe_comfyui)
    if state.json_output or json_output:
        typer.echo(json.dumps(report.as_dict(), indent=2, sort_keys=True, default=str))
    else:
        typer.echo(render_text_report(report))
    if not report.ok:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# template
# ---------------------------------------------------------------------------
@template_app.command("ingest")
def template_ingest(
    source: Annotated[Path, typer.Argument(help="Master performance video file.")],
    name: Annotated[str, typer.Option("--name", help="Display name for the template.")],
    intro_start: Annotated[
        int, typer.Option("--intro-start", help="First frame of the intro.")
    ] = 0,
    transition_anchor: Annotated[
        int | None,
        typer.Option(
            "--transition-anchor",
            help="Seam frame. The intro is [intro-start, anchor); the reveal is "
            "[anchor, reveal-end). Defaults to the midpoint.",
        ),
    ] = None,
    reveal_end: Annotated[
        int | None,
        typer.Option(
            "--reveal-end", help="Exclusive last frame of the reveal (default: all frames)."
        ),
    ] = None,
    clothing_class: Annotated[
        TemplateClothingClass,
        typer.Option("--clothing-class", help="What the performer is wearing in the source."),
    ] = TemplateClothingClass.FITTED_SHORT,
    subject_kind: Annotated[
        str, typer.Option("--subject-kind", help="synthetic | consented_human")
    ] = "synthetic",
    consent_document: Annotated[
        str | None,
        typer.Option(
            "--consent-document", help="Consent record reference (required for consented_human)."
        ),
    ] = None,
    rights_holder: Annotated[str | None, typer.Option("--rights-holder")] = None,
    license_text: Annotated[str | None, typer.Option("--license")] = None,
    identity_ref: Annotated[
        list[Path] | None,
        typer.Option("--identity-ref", help="Identity reference image (repeatable)."),
    ] = None,
    background_plate: Annotated[Path | None, typer.Option("--background-plate")] = None,
    template_id: Annotated[
        str | None, typer.Option("--template-id", help="Use a specific id.")
    ] = None,
    allow_vfr_conversion: Annotated[
        bool,
        typer.Option(
            "--allow-vfr-conversion",
            help="Convert a variable-frame-rate source to CFR (writes a new file; "
            "the original is never modified).",
        ),
    ] = False,
    json_output: JsonFlag = False,
) -> None:
    """Ingest a master performance video: probe, extract frames, hash, segment."""
    with _context() as context:
        options = IngestOptions(
            display_name=name,
            intro_start=intro_start,
            transition_anchor=transition_anchor,
            reveal_end=reveal_end,
            template_clothing_class=clothing_class.value,
            subject_kind=subject_kind,
            consent_document_ref=consent_document,
            rights_holder=rights_holder,
            license=license_text,
            identity_reference_images=list(identity_ref or []),
            background_plate=background_plate,
            template_id=template_id,
            allow_vfr_conversion=allow_vfr_conversion,
        )
        result = _run(ingest_template, context, source, options)
        template = result.template
        lines = [
            f"template   {template.id} v{template.version}  ({template.display_name})",
            f"video      {template.video.width}x{template.video.height}"
            f" @ {template.video.fps:.3f}fps",
            f"frames     {result.extracted_frames} extracted (0..{result.extracted_frames - 1})",
            f"intro      [{template.intro.start}, {template.intro.end})  "
            f"{template.intro.count} frames",
            f"reveal     [{template.reveal.start}, {template.reveal.end})  "
            f"{template.reveal.count} frames",
            f"anchor     {template.transition_anchor_frame}",
            f"source     sha256 {template.source_sha256[:16]}...",
            f"status     {template.status.value}",
            "",
            "Next: author masks and import them:",
            f"  app template import-masks {template.id} --kind garment --from <dir>",
            f"  app template import-masks {template.id} --kind protected --from <dir>",
            f"  app template inspect {template.id}",
        ]
        for warning in result.warnings:
            lines.append(f"warning: {warning}")
        _emit(result.as_dict(), "\n".join(lines), force_json=json_output)


@template_app.command("inspect")
def template_inspect(
    template_id: Annotated[str, typer.Argument()],
    version: Annotated[int | None, typer.Option("--version")] = None,
    validate: Annotated[
        bool, typer.Option("--validate/--no-validate", help="Re-validate frames and masks.")
    ] = True,
    json_output: JsonFlag = False,
) -> None:
    """Show a template and validate its frames and masks."""
    with _context() as context:
        template = _run(context.repos.templates.get, template_id, version)
        payload: dict[str, Any] = {"template": template.to_json_dict()}
        lines = [
            f"template   {template.id} v{template.version}  ({template.display_name})",
            f"status     {template.status.value}",
            f"class      {template.template_clothing_class.value}",
            f"video      {template.video.width}x{template.video.height} @ "
            f"{template.video.fps:.3f}fps  {template.video.frame_count} frames",
            f"intro      [{template.intro.start}, {template.intro.end})",
            f"reveal     [{template.reveal.start}, {template.reveal.end})",
            f"anchor     {template.transition_anchor_frame}",
            f"consent    {template.consent.subject_kind} adult={template.consent.adult_confirmed}",
        ]
        if validate:
            validation = _run(validate_template, context, template.id, version=template.version)
            payload["validation"] = validation.as_dict()
            lines.append("")
            lines.append(f"validation {'OK' if validation.ok else 'PROBLEMS'}")
            for problem in validation.problems:
                lines.append(f"  problem: {problem}")
            for warning in validation.warnings:
                lines.append(f"  warning: {warning}")
            masks = validation.details.get("masks", {})
            for kind, info in sorted(masks.items()):
                lines.append(
                    f"  masks/{kind:<10} present={info.get('present_count')} "
                    f"missing={info.get('missing_count')}"
                )
        _emit(payload, "\n".join(lines), force_json=json_output)


@template_app.command("import-masks")
def template_import_masks(
    template_id: Annotated[str, typer.Argument()],
    kind: Annotated[
        MaskKind, typer.Option("--kind", help="garment | expansion | protected | occlusion")
    ],
    source: Annotated[Path, typer.Option("--from", help="Directory of frame_NNNNNN.png masks.")],
    version: Annotated[int | None, typer.Option("--version")] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    json_output: JsonFlag = False,
) -> None:
    """Import externally authored masks for a template."""
    with _context() as context:
        result = _run(
            import_masks,
            context,
            template_id,
            kind,
            source,
            version=version,
            overwrite=overwrite,
        )
        lines = [
            f"imported   {len(result.imported)} {kind.value} masks",
            f"into       {result.destination}",
        ]
        if result.imported:
            lines.append(f"range      {min(result.imported)}..{max(result.imported)}")
        for skip in result.skipped[:10]:
            lines.append(f"skipped    {skip}")
        _emit(result.as_dict(), "\n".join(lines), force_json=json_output)


@template_app.command("list")
def template_list(json_output: JsonFlag = False) -> None:
    """List ingested templates."""
    with _context() as context:
        templates = context.repos.templates.list()
        payload = {"templates": [t.to_json_dict() for t in templates]}
        lines = [f"{len(templates)} template(s)"]
        for template in templates:
            lines.append(
                f"  {template.id} v{template.version:<3} {template.status.value:<16} "
                f"{template.display_name}"
            )
        _emit(payload, "\n".join(lines), force_json=json_output)


# ---------------------------------------------------------------------------
# garment
# ---------------------------------------------------------------------------
@garment_app.command("ingest")
def garment_ingest(
    image: Annotated[
        list[str],
        typer.Option(
            "--image",
            help="Reference image as view=path, e.g. --image front=./a.png (repeatable). "
            "Views: front, back, side, detail, flat_lay.",
        ),
    ],
    category: Annotated[GarmentCategory, typer.Option("--category")],
    coverage: Annotated[BodyCoverage, typer.Option("--coverage")],
    silhouette: Annotated[Silhouette, typer.Option("--silhouette")],
    material: Annotated[Material, typer.Option("--material")],
    sleeve: Annotated[SleeveLength, typer.Option("--sleeve")] = SleeveLength.NOT_APPLICABLE,
    length: Annotated[GarmentLength, typer.Option("--length")] = GarmentLength.NOT_APPLICABLE,
    transparency: Annotated[float, typer.Option("--transparency", min=0.0, max=1.0)] = 0.0,
    reflectivity: Annotated[float, typer.Option("--reflectivity", min=0.0, max=1.0)] = 0.0,
    fabric_flow: Annotated[float, typer.Option("--fabric-flow", min=0.0, max=1.0)] = 0.0,
    product_name: Annotated[str | None, typer.Option("--product-name")] = None,
    brand: Annotated[str | None, typer.Option("--brand")] = None,
    source_url: Annotated[
        str | None, typer.Option("--source-url", help="Recorded as metadata only; never fetched.")
    ] = None,
    license_text: Annotated[str | None, typer.Option("--license")] = None,
    rights_holder: Annotated[str | None, typer.Option("--rights-holder")] = None,
    pattern: Annotated[str | None, typer.Option("--pattern", help="Pattern description.")] = None,
    color: Annotated[
        list[str] | None, typer.Option("--color", help="Dominant colour as #RRGGBB (repeatable).")
    ] = None,
    requires_underlayer: Annotated[bool, typer.Option("--requires-underlayer")] = False,
    garment_id: Annotated[str | None, typer.Option("--garment-id")] = None,
    json_output: JsonFlag = False,
) -> None:
    """Ingest garment reference images plus their structured attributes."""
    specs: list[ImageSpec] = []
    for entry in image:
        if "=" not in entry:
            typer.secho(
                f"--image must be view=path, got {entry!r}. "
                "Views: front, back, side, detail, flat_lay.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2)
        view_text, _, path_text = entry.partition("=")
        try:
            view = ImageViewType(view_text.strip().lower())
        except ValueError:
            typer.secho(
                f"unknown view {view_text!r}; expected one of "
                + ", ".join(v.value for v in ImageViewType),
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2) from None
        specs.append(ImageSpec(path=Path(path_text.strip()), view=view))

    with _context() as context:
        options = GarmentIngestOptions(
            category=category,
            body_coverage=coverage,
            silhouette=silhouette,
            material=material,
            sleeve_length=sleeve,
            garment_length=length,
            transparency=transparency,
            reflectivity=reflectivity,
            fabric_flow=fabric_flow,
            dominant_colors=list(color or []),
            pattern_description=pattern,
            requires_underlayer=requires_underlayer,
            product_name=product_name,
            brand=brand,
            source_url=source_url,
            license=license_text,
            rights_holder=rights_holder,
            garment_id=garment_id,
        )
        result = _run(ingest_garment, context, specs, options)
        garment = result.garment
        lines = [
            f"garment    {garment.id} v{garment.version}",
            f"category   {garment.category.value}  coverage={garment.body_coverage.value}",
            f"views      {', '.join(sorted(v.value for v in garment.available_views))}",
            f"material   {garment.material.value}  transparency={garment.transparency}",
            f"colors     {', '.join(garment.dominant_colors) or '(none)'}",
            f"status     {garment.status.value}",
        ]
        for warning in result.warnings:
            lines.append(f"warning: {warning}")
        _emit(result.as_dict(), "\n".join(lines), force_json=json_output)


@garment_app.command("inspect")
def garment_inspect(
    garment_id: Annotated[str, typer.Argument()],
    version: Annotated[int | None, typer.Option("--version")] = None,
    json_output: JsonFlag = False,
) -> None:
    """Show a garment asset."""
    with _context() as context:
        garment = _run(context.repos.garments.get, garment_id, version)
        lines = [
            f"garment    {garment.id} v{garment.version}",
            f"product    {garment.product_name or '(unnamed)'}  brand={garment.brand or '-'}",
            f"category   {garment.category.value}",
            f"coverage   {garment.body_coverage.value}",
            f"sleeve     {garment.sleeve_length.value}   length={garment.garment_length.value}",
            f"silhouette {garment.silhouette.value}  material={garment.material.value}",
            f"transp.    {garment.transparency}  reflect={garment.reflectivity}  "
            f"flow={garment.fabric_flow}",
            f"rights     license={garment.usage_rights.license or '-'} "
            f"holder={garment.usage_rights.rights_holder or '-'}",
            f"complete   {garment.metadata_completeness:.2f}",
            "images:",
        ]
        for img in garment.images:
            lines.append(
                f"  {img.view.value:<9} {img.width}x{img.height} "
                f"sharp={img.sharpness_score}  {img.path}"
            )
        _emit({"garment": garment.to_json_dict()}, "\n".join(lines), force_json=json_output)


@garment_app.command("list")
def garment_list(json_output: JsonFlag = False) -> None:
    """List ingested garments."""
    with _context() as context:
        garments = context.repos.garments.list()
        lines = [f"{len(garments)} garment(s)"]
        for garment in garments:
            lines.append(
                f"  {garment.id} v{garment.version:<3} {garment.status.value:<22} "
                f"{garment.category.value:<12} {garment.product_name or ''}"
            )
        _emit(
            {"garments": [g.to_json_dict() for g in garments]},
            "\n".join(lines),
            force_json=json_output,
        )


# ---------------------------------------------------------------------------
# compatibility
# ---------------------------------------------------------------------------
@compat_app.command("check")
def compatibility_check(
    template: Annotated[str, typer.Option("--template", help="Template id.")],
    garment: Annotated[str, typer.Option("--garment", help="Garment id.")],
    template_version: Annotated[int | None, typer.Option("--template-version")] = None,
    garment_version: Annotated[int | None, typer.Option("--garment-version")] = None,
    exposed_view: Annotated[
        list[ImageViewType] | None,
        typer.Option(
            "--exposed-view",
            help="A view the performance actually exposes (repeatable). Omit if unknown.",
        ),
    ] = None,
    json_output: JsonFlag = False,
) -> None:
    """Run the deterministic compatibility rules and store the report."""
    with _context() as context:
        report = _run(
            compat_service.check_compatibility,
            context,
            template,
            garment,
            template_version=template_version,
            garment_version=garment_version,
            options=compat_service.CheckOptions(
                exposed_views=list(exposed_view) if exposed_view else None
            ),
        )
        summary = compat_service.summarize(report)
        colour = {
            "READY": typer.colors.GREEN,
            "NEEDS_INPUT": typer.colors.YELLOW,
            "INCOMPATIBLE": typer.colors.RED,
        }[report.state.value]
        lines = [
            f"report     {report.id}",
            f"state      {report.state.value}",
            f"confidence {report.confidence:.3f}   rules v{report.rules_version}",
            "",
        ]
        for result in report.results:
            marker = {
                "pass": "ok  ",
                "warn": "WARN",
                "needs_input": "NEED",
                "fail": "FAIL",
                "skipped": "skip",
            }[result.outcome.value]
            lines.append(f"[{marker}] {result.rule_id:<34} {result.message}")
            if result.remediation and result.outcome.value in {"warn", "needs_input", "fail"}:
                lines.append(f"         -> {result.remediation}")
        if report.required_missing_views:
            lines.append("")
            lines.append(
                "missing views: " + ", ".join(v.value for v in report.required_missing_views)
            )
        if report.required_mask_expansions:
            lines.append(
                "required mask work: " + ", ".join(k.value for k in report.required_mask_expansions)
            )
        lines.append("")
        lines.append(f"render allowed: {summary['render_allowed']}")
        if not summary["render_allowed"]:
            lines.append(
                f"  fix the inputs and re-check, or: app compatibility override {report.id} "
                '--reviewer <name> --reason "<why>"'
            )
        if state.json_output or json_output:
            typer.echo(json.dumps(summary, indent=2, sort_keys=True, default=str))
        else:
            typer.secho("\n".join(lines), fg=None)
            typer.secho(f"\n{report.state.value}", fg=colour, bold=True)
        if report.state.value != "READY":
            raise typer.Exit(code=3)


@compat_app.command("override")
def compatibility_override(
    report_id: Annotated[str, typer.Argument()],
    reviewer: Annotated[str, typer.Option("--reviewer", help="Who is approving this.")],
    reason: Annotated[str, typer.Option("--reason", help="Why (10+ characters, audited).")],
    expires_in_hours: Annotated[float | None, typer.Option("--expires-in-hours")] = None,
    acknowledge: Annotated[
        list[str] | None,
        typer.Option(
            "--acknowledge",
            help="Rule id being overridden (repeatable). Use protected_mask_override "
            "to also permit editing protected pixels.",
        ),
    ] = None,
    json_output: JsonFlag = False,
) -> None:
    """Store a reviewed override so a blocked pair may be rendered."""
    with _context() as context:
        report = _run(
            compat_service.store_override,
            context,
            report_id,
            reviewer=reviewer,
            reason=reason,
            acknowledged_rule_ids=list(acknowledge) if acknowledge else None,
            expires_in_hours=expires_in_hours,
        )
        lines = [
            f"override stored for report {report.id}",
            f"state      {report.state.value} (overridden)",
            f"reviewer   {reviewer}",
            "acknowledged "
            + (", ".join(report.override.acknowledged_rule_ids) if report.override else ""),
            f"render allowed: {report.render_allowed(utc_now())}",
        ]
        _emit(compat_service.summarize(report), "\n".join(lines), force_json=json_output)


# ---------------------------------------------------------------------------
# job
# ---------------------------------------------------------------------------
@job_app.command("create")
def job_create(
    template: Annotated[str, typer.Option("--template")],
    garment: Annotated[str, typer.Option("--garment")],
    backend: Annotated[str, typer.Option("--backend", help="mock | comfyui")] = "mock",
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    prompt: Annotated[str, typer.Option("--prompt")] = "",
    negative_prompt: Annotated[str, typer.Option("--negative-prompt")] = "",
    workflow_id: Annotated[str | None, typer.Option("--workflow-id")] = None,
    frame_end: Annotated[
        int | None,
        typer.Option("--frame-end", help="Render only up to this exclusive frame (debugging)."),
    ] = None,
    template_version: Annotated[int | None, typer.Option("--template-version")] = None,
    garment_version: Annotated[int | None, typer.Option("--garment-version")] = None,
    json_output: JsonFlag = False,
) -> None:
    """Create a render job for a template/garment pair."""
    with _context() as context:
        job = _run(
            create_job,
            context,
            template,
            garment,
            JobCreateOptions(
                backend_name=backend,
                seed=seed,
                prompt=prompt,
                negative_prompt=negative_prompt,
                workflow_id=workflow_id,
                frame_end=frame_end,
            ),
            template_version=template_version,
            garment_version=garment_version,
        )
        lines = [
            f"job        {job.id}",
            f"template   {job.template_key}",
            f"garment    {job.garment_key}",
            f"backend    {job.backend_name} v{job.backend_version}",
            f"frames     [{job.frame_range.start}, {job.frame_range.end})  "
            f"{job.frame_range.count} frames",
            f"anchor     {job.transition_anchor_frame}",
            f"seed       {job.seed}",
            f"status     {job.status.value}",
            "",
            f"Next: app job render {job.id} --backend {job.backend_name}",
        ]
        _emit({"job": job.to_json_dict()}, "\n".join(lines), force_json=json_output)


@job_app.command("render")
def job_render(
    job_id: Annotated[str, typer.Argument()],
    backend: Annotated[
        str | None, typer.Option("--backend", help="Override the job's backend.")
    ] = None,
    max_frames: Annotated[
        int | None, typer.Option("--max-frames", help="Render only the next N frames.")
    ] = None,
    quiet: Annotated[bool, typer.Option("--quiet", help="Suppress the progress bar.")] = False,
    json_output: JsonFlag = False,
) -> None:
    """Render a job's reveal frames (compositing through the mask)."""
    with _context() as context:
        job = _run(context.repos.jobs.get, job_id)
        total = job.frame_range.count
        show_progress = not quiet and not (state.json_output or json_output) and sys.stderr.isatty()

        if show_progress:
            with typer.progressbar(length=total, label="rendering") as bar:
                seen = {"count": 0}

                def on_progress(_index: int, completed: int, _total: int) -> None:
                    bar.update(completed - seen["count"])
                    seen["count"] = completed

                outcome = _run(
                    render_job,
                    context,
                    job_id,
                    backend_name=backend,
                    progress=on_progress,
                    max_frames=max_frames,
                )
        else:
            outcome = _run(render_job, context, job_id, backend_name=backend, max_frames=max_frames)

        lines = [
            f"job        {outcome.job.id}",
            f"status     {outcome.job.status.value}",
            f"rendered   {len(outcome.rendered_frames)} frames",
            f"skipped    {len(outcome.skipped_frames)} already-complete frames",
            f"leakage    {len(outcome.leaked_frames)} frames (must be 0)",
            "",
            f"Next: app job compose {outcome.job.id}",
        ]
        _emit(outcome.as_dict(), "\n".join(lines), force_json=json_output)


@job_app.command("resume")
def job_resume(
    job_id: Annotated[str, typer.Argument()],
    backend: Annotated[str | None, typer.Option("--backend")] = None,
    max_frames: Annotated[int | None, typer.Option("--max-frames")] = None,
    json_output: JsonFlag = False,
) -> None:
    """Resume an interrupted job without re-rendering completed frames."""
    with _context() as context:
        outcome = _run(resume_job, context, job_id, backend_name=backend, max_frames=max_frames)
        lines = [
            f"job        {outcome.job.id}",
            f"status     {outcome.job.status.value}",
            f"rendered   {len(outcome.rendered_frames)} new frames",
            f"skipped    {len(outcome.skipped_frames)} already-complete frames",
        ]
        _emit(outcome.as_dict(), "\n".join(lines), force_json=json_output)


@job_app.command("compose")
def job_compose(
    job_id: Annotated[str, typer.Argument()],
    no_audio: Annotated[
        bool, typer.Option("--no-audio", help="Skip copying master audio.")
    ] = False,
    flash_frames: Annotated[
        int | None, typer.Option("--flash-frames", help="0 (off) or 2-4 flash frames at the seam.")
    ] = None,
    preview: Annotated[
        bool, typer.Option("--preview", help="Also write a small preview encode.")
    ] = False,
    output_name: Annotated[str | None, typer.Option("--output-name")] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    json_output: JsonFlag = False,
) -> None:
    """Join the cached intro with the rendered reveal and encode the final video."""
    with _context() as context:
        result = _run(
            compose_job,
            context,
            job_id,
            ComposeOptions(
                include_audio=not no_audio,
                flash_frames=flash_frames,
                make_preview=preview,
                output_name=output_name,
                overwrite=overwrite,
            ),
        )
        lines = [
            f"job        {result.job.id}",
            f"video      {result.final_video}",
            f"frames     {result.assembly['intro_frame_count']} intro + "
            f"{result.assembly['reveal_frame_count']} reveal = "
            f"{result.assembly['total_frames']}",
            f"anchor     {result.assembly['transition_anchor_frame']}",
            f"manifest   {result.job.artifacts.manifest_path}",
            f"digest     {result.manifest.reproducibility_digest()[:16]}",
        ]
        if result.preview_video:
            lines.append(f"preview    {result.preview_video}")
        lines.append("")
        lines.append(f"Next: app job qc {result.job.id}")
        _emit(result.as_dict(), "\n".join(lines), force_json=json_output)


@job_app.command("qc")
def job_qc(
    job_id: Annotated[str, typer.Argument()],
    max_sampled_frames: Annotated[
        int | None,
        typer.Option("--max-sampled-frames", help="Cap pixel-level sampling (default: all)."),
    ] = None,
    no_contact_sheets: Annotated[bool, typer.Option("--no-contact-sheets")] = False,
    json_output: JsonFlag = False,
) -> None:
    """Run automated QC and write JSON + text reports."""
    from app.qc.report import render_text_report

    with _context() as context:
        report = _run(
            run_qc,
            context,
            job_id,
            QCOptions(
                max_sampled_frames=max_sampled_frames,
                make_contact_sheets=not no_contact_sheets,
            ),
        )
        if state.json_output or json_output:
            typer.echo(json.dumps(report.as_dict(), indent=2, sort_keys=True, default=str))
        else:
            typer.echo(render_text_report(report))
        if not report.passed:
            raise typer.Exit(code=1)


@job_app.command("inspect")
def job_inspect(
    job_id: Annotated[str, typer.Argument()],
    json_output: JsonFlag = False,
) -> None:
    """Show a job's status, progress, checkpoint and artefacts."""
    with _context() as context:
        job = _run(context.repos.jobs.get, job_id)
        completed = context.repos.jobs.completed_frames(job.id)
        remaining = [i for i in job.frame_range.indices() if i not in set(completed)]
        lines = [
            f"job        {job.id}",
            f"status     {job.status.value}",
            f"template   {job.template_key}",
            f"garment    {job.garment_key}",
            f"backend    {job.backend_name} v{job.backend_version}",
            f"workflow   {job.workflow_id or '-'}" f"  sha={(job.workflow_sha256 or '-')[:12]}",
            f"frames     [{job.frame_range.start}, {job.frame_range.end})",
            f"progress   {len(completed)}/{job.frame_range.count} "
            f"({job.progress.percent:.1f}%)  remaining={len(remaining)}",
            f"seed       {job.seed}",
            f"manifest   {job.artifacts.manifest_path or '-'}",
            f"video      {job.artifacts.final_video_path or '-'}",
            f"qc report  {job.artifacts.qc_report_path or '-'}",
        ]
        if job.error:
            lines.append(f"error      [{job.error.code}] {job.error.message}")
        payload = {
            "job": job.to_json_dict(),
            "completed_frames": len(completed),
            "remaining_frames": remaining[:64],
        }
        _emit(payload, "\n".join(lines), force_json=json_output)


@job_app.command("list")
def job_list(
    status: Annotated[JobStatus | None, typer.Option("--status")] = None,
    json_output: JsonFlag = False,
) -> None:
    """List render jobs."""
    with _context() as context:
        jobs = context.repos.jobs.list(status=status)
        lines = [f"{len(jobs)} job(s)"]
        for job in jobs:
            lines.append(
                f"  {job.id} {job.status.value:<12} {job.backend_name:<9} "
                f"{job.progress.completed_frames}/{job.progress.total_frames} "
                f"{job.template_id}/{job.garment_id}"
            )
        _emit({"jobs": [j.to_json_dict() for j in jobs]}, "\n".join(lines), force_json=json_output)


@job_app.command("delete")
def job_delete(
    job_id: Annotated[str, typer.Argument()],
    yes: Annotated[bool, typer.Option("--yes", help="Confirm deletion.")] = False,
) -> None:
    """Delete exactly one job's artefact directory (requires confirmation).

    Only the resolved directory of one existing job is ever removed, and only
    after it is confirmed to live inside the configured data root.
    """
    import shutil

    with _context() as context:
        job = _run(context.repos.jobs.get, job_id)
        target = context.absolute(job.artifacts.root)
        expected = context.data_root.job_dir(job.id)
        if target != expected:
            typer.secho(
                f"refusing to delete: {target} is not the canonical job directory " f"({expected})",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2)
        if not target.is_dir():
            typer.echo(f"nothing to delete: {target} does not exist")
            raise typer.Exit(code=0)
        if not yes:
            typer.echo(f"would delete: {target}")
            typer.echo("re-run with --yes to confirm")
            raise typer.Exit(code=1)
        shutil.rmtree(target)
        context.repos.audit.record(
            "job_directory_deleted",
            entity_type="job",
            entity_id=job.id,
            details={"path": str(target)},
        )
        typer.echo(f"deleted {target}")


@job_app.command("backends")
def job_backends(json_output: JsonFlag = False) -> None:
    """Show each backend's capabilities and health."""
    config = _load_config()
    payload: dict[str, Any] = {}
    lines: list[str] = []
    for name in available_backends():
        try:
            backend = create_backend(name, config)
        except AppError as exc:
            payload[name] = {"error": exc.to_dict()}
            lines.append(f"{name:<10} ERROR {exc.message}")
            continue
        try:
            capabilities = backend.capabilities().as_dict()
            health = backend.healthcheck().as_dict()
        finally:
            backend.close()
        payload[name] = {"capabilities": capabilities, "health": health}
        lines.append(
            f"{name:<10} {'healthy' if health['healthy'] else 'unavailable':<12} "
            f"gpu={capabilities['requires_gpu']} weights={capabilities['requires_model_weights']} "
            f"deterministic={capabilities['deterministic']}"
        )
        if not health["healthy"]:
            lines.append(f"           {health['detail']}")
    _emit(payload, "\n".join(lines), force_json=json_output)


@app.command("serve")
def serve(
    host: Annotated[str | None, typer.Option("--host")] = None,
    port: Annotated[int | None, typer.Option("--port")] = None,
    reload: Annotated[bool, typer.Option("--reload")] = False,
) -> None:
    """Run the localhost HTTP API."""
    import uvicorn

    config = _load_config()
    bind_host = host or config.api.host
    if bind_host not in {"127.0.0.1", "localhost", "::1"} and not config.api.allow_remote_bind:
        typer.secho(
            f"refusing to bind {bind_host}: set api.allow_remote_bind: true if you "
            "really mean to expose this service beyond localhost",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)
    uvicorn.run(
        "app.api.main:create_app",
        factory=True,
        host=bind_host,
        port=port or config.api.port,
        reload=reload,
        log_level=config.runtime.log_level.lower(),
    )


if __name__ == "__main__":  # pragma: no cover
    app()
