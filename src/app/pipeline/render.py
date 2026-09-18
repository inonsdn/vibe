"""Render orchestration.

This is where the core design principle is enforced in code:

* the human is **never** regenerated — every output pixel starts as a source
  pixel read from the immutable template frames,
* only the reveal range ``[anchor, reveal_end)`` is processed,
* the backend's output is composited through the effective mask, so pixels
  outside it are restored byte-for-byte,
* work is checkpointed per frame so an interrupted job resumes without
  re-rendering anything that already finished,
* every input hash, seed and setting is recorded for the manifest.

Nothing here knows what model (if any) is behind the backend.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.backends.base import (
    FrameRequest,
    FrameResult,
    RenderContext,
    RendererBackend,
    WindowRequest,
)
from app.backends.registry import create_backend
from app.core.determinism import derive_seed, frame_seed, window_seed
from app.core.errors import (
    CompatibilityBlockedError,
    ConflictError,
    MaskError,
    ValidationError,
)
from app.core.hashing import sha256_file
from app.core.ids import job_id as new_job_id
from app.core.ids import utc_now
from app.core.logging import get_logger, log_context, log_event
from app.domain.compatibility import CompatibilityReport
from app.domain.enums import JobStatus, MaskKind, ProcessingStatus
from app.domain.garment import GarmentAsset
from app.domain.human_template import FrameRange, HumanTemplate
from app.domain.render_job import (
    Checkpoint,
    JobArtifacts,
    JobError,
    JobProgress,
    RenderJob,
    RenderSettings,
)
from app.media.compositor import composite_with_stats
from app.media.frames import DEFAULT_TEMPLATE, frame_path, read_frame_at, write_frame
from app.media.masks import EffectiveMask, build_effective_mask, load_mask_set, save_mask
from app.pipeline.context import ServiceContext
from app.pipeline.template_ingest import verify_source_immutability

logger = get_logger(__name__)

ProgressCallback = Callable[[int, int, int], None]  # (frame_index, completed, total)


@dataclass
class JobCreateOptions:
    backend_name: str = "mock"
    seed: int | None = None
    prompt: str = ""
    negative_prompt: str = ""
    settings: RenderSettings | None = None
    workflow_id: str | None = None
    compatibility_report_id: str | None = None
    #: Restrict the render to a sub-range of the reveal segment (debugging).
    frame_start: int | None = None
    frame_end: int | None = None
    job_id: str | None = None
    #: Permit a blocked compatibility state because a stored override exists.
    allow_override: bool = True


@dataclass
class RenderOutcome:
    job: RenderJob
    rendered_frames: list[int]
    skipped_frames: list[int]
    backend_info: dict[str, Any]
    mask_stats: dict[int, dict[str, Any]] = field(default_factory=dict)
    composite_stats: dict[int, dict[str, Any]] = field(default_factory=dict)
    leaked_frames: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job.id,
            "status": self.job.status.value,
            "rendered_frames": len(self.rendered_frames),
            "skipped_frames": len(self.skipped_frames),
            "first_frame": self.job.frame_range.start,
            "last_frame": self.job.frame_range.last,
            "leaked_frames": self.leaked_frames,
            "backend": self.backend_info.get("backend", self.job.backend_name),
        }


# ---------------------------------------------------------------------------
# job creation
# ---------------------------------------------------------------------------
def create_job(
    context: ServiceContext,
    template_id: str,
    garment_id: str,
    options: JobCreateOptions,
    *,
    template_version: int | None = None,
    garment_version: int | None = None,
) -> RenderJob:
    """Create a render job, refusing anything a render would later reject."""
    template = context.repos.templates.get(template_id, template_version)
    garment = context.repos.garments.get(garment_id, garment_version)

    if template.status not in {ProcessingStatus.READY, ProcessingStatus.VALIDATED}:
        raise ValidationError(
            "Template is not validated; run `app template inspect` first",
            template_id=template.id,
            status=template.status.value,
        )

    report = _resolve_report(context, template, garment, options)
    _assert_render_allowed(context, report, allow_override=options.allow_override)

    start = options.frame_start if options.frame_start is not None else template.reveal.start
    end = options.frame_end if options.frame_end is not None else template.reveal.end
    if start != template.transition_anchor_frame:
        raise ValidationError(
            "A render must begin exactly at the transition anchor; the intro is "
            "reused from cache and never re-rendered",
            frame_start=start,
            transition_anchor_frame=template.transition_anchor_frame,
        )
    if end > template.reveal.end:
        raise ValidationError(
            "frame_end exceeds the template's reveal range",
            frame_end=end,
            reveal_end=template.reveal.end,
        )
    frame_range = FrameRange(start=start, end=end)

    identifier = options.job_id or new_job_id()
    job_dir = context.data_root.job_dir(identifier)
    artifacts = JobArtifacts.standard(context.relative(job_dir.parent) + f"/{identifier}")
    for directory in (
        artifacts.raw_frames_dir,
        artifacts.composited_frames_dir,
        artifacts.effective_masks_dir,
    ):
        context.absolute(directory).mkdir(parents=True, exist_ok=True)

    settings = options.settings or RenderSettings(
        frame_window=context.config.backend.frame_window,
        feather_radius_px=context.config.mask.feather_radius_px,
        expansion_dilate_px=context.config.mask.expansion_dilate_px,
        protected_dilate_px=context.config.mask.protected_dilate_px,
        protected_wins=context.config.mask.protected_wins,
        low_vram_mode=True,
        cpu_offload=True,
    )
    seed = options.seed if options.seed is not None else context.config.runtime.default_seed
    backend = create_backend(options.backend_name, context.config)
    try:
        capabilities = backend.capabilities()
    finally:
        backend.close()

    job = RenderJob(
        id=identifier,
        template_id=template.id,
        template_version=template.version,
        garment_id=garment.id,
        garment_version=garment.version,
        compatibility_report_id=report.id if report else None,
        frame_range=frame_range,
        transition_anchor_frame=template.transition_anchor_frame,
        backend_name=options.backend_name,
        backend_version=capabilities.version,
        workflow_id=options.workflow_id
        or (context.config.comfyui.workflow_id if options.backend_name == "comfyui" else None),
        seed=seed,
        settings=settings,
        prompt=options.prompt,
        negative_prompt=options.negative_prompt,
        input_hashes=collect_input_hashes(context, template, garment, frame_range),
        config_hash=context.config.config_hash(),
        status=JobStatus.CREATED,
        progress=JobProgress(total_frames=frame_range.count),
        artifacts=artifacts,
    )
    saved = context.repos.jobs.save(job)
    context.repos.audit.record(
        "job_created",
        entity_type="job",
        entity_id=saved.id,
        details={
            "template": saved.template_key,
            "garment": saved.garment_key,
            "backend": saved.backend_name,
            "frames": [frame_range.start, frame_range.end],
            "compatibility_state": report.state.value if report else None,
        },
    )
    log_event(
        logger,
        "job_created",
        job_id=saved.id,
        template_id=template.id,
        garment_id=garment.id,
        frames=frame_range.count,
    )
    return saved


def _resolve_report(
    context: ServiceContext,
    template: HumanTemplate,
    garment: GarmentAsset,
    options: JobCreateOptions,
) -> CompatibilityReport | None:
    if options.compatibility_report_id:
        report = context.repos.compatibility.get(options.compatibility_report_id)
        if (report.template_id, report.garment_id) != (template.id, garment.id):
            raise ValidationError(
                "Compatibility report does not match this template/garment pair",
                report_id=report.id,
            )
        return report
    return context.repos.compatibility.latest_for_pair(
        template.id, template.version, garment.id, garment.version
    )


def _assert_render_allowed(
    context: ServiceContext,
    report: CompatibilityReport | None,
    *,
    allow_override: bool,
) -> None:
    """Rendering is blocked unless compatibility is READY or overridden."""
    if report is None:
        raise CompatibilityBlockedError(
            "No compatibility report exists for this template/garment pair. "
            "Run `app compatibility check` first.",
            hint="app compatibility check --template <id> --garment <id>",
        )
    permitted = allow_override and context.config.compatibility.allow_override
    if report.render_allowed(utc_now(), allow_override=permitted):
        return
    raise CompatibilityBlockedError(
        f"Rendering is blocked: {report.block_explanation()}",
        report_id=report.id,
        state=report.state.value,
        blocking_reasons=report.blocking_reasons,
        required_missing_views=[v.value for v in report.required_missing_views],
        required_mask_expansions=[k.value for k in report.required_mask_expansions],
        hint=(
            "Fix the inputs and re-check, or store a reviewed override with "
            "`app compatibility override`."
        ),
    )


def collect_input_hashes(
    context: ServiceContext,
    template: HumanTemplate,
    garment: GarmentAsset,
    frame_range: FrameRange,
) -> dict[str, str]:
    """Hash everything a render consumes, for the manifest."""
    hashes: dict[str, str] = {
        "template_source_video": template.source_sha256,
        "template_source_frames_dir": template.source_frames_sha256 or "unknown",
    }
    for image in garment.images:
        hashes[f"garment_image_{image.view.value}_{Path(image.path).name}"] = image.sha256
    for relative, digest in template.identity_reference_hashes.items():
        hashes[f"identity_ref_{Path(relative).name}"] = digest

    # Masks are hashed per kind over the rendered range only: masks outside the
    # range cannot influence this job's output.
    from app.core.hashing import sha256_text

    for kind in MaskKind:
        directory = context.absolute(template.directories.mask_dir(kind))
        parts: list[str] = []
        for index in frame_range.indices():
            path = frame_path(directory, index)
            if path.is_file():
                parts.append(f"{index}:{sha256_file(path)}")
        hashes[f"masks_{kind.value}"] = sha256_text("|".join(parts)) if parts else "absent"
    return hashes


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def render_job(
    context: ServiceContext,
    job_id: str,
    *,
    backend_name: str | None = None,
    resume: bool = False,
    progress: ProgressCallback | None = None,
    max_frames: int | None = None,
) -> RenderOutcome:
    """Render (or resume) a job's reveal frames.

    ``max_frames`` renders only the next N frames and leaves the job
    resumable — used by the interruption tests and by an operator who wants to
    eyeball a few frames before committing the GPU time.
    """
    job = context.repos.jobs.get(job_id)
    if job.status.is_terminal and job.status is not JobStatus.FAILED and not resume:
        raise ConflictError(
            "Job has already finished; create a new job or pass resume",
            job_id=job.id,
            status=job.status.value,
        )

    template = context.repos.templates.get(job.template_id, job.template_version)
    garment = context.repos.garments.get(job.garment_id, job.garment_version)
    report = (
        context.repos.compatibility.try_get(job.compatibility_report_id)
        if job.compatibility_report_id
        else None
    )
    _assert_render_allowed(context, report, allow_override=True)
    verify_source_immutability(context, template)

    effective_backend = backend_name or job.backend_name
    backend = create_backend(effective_backend, context.config)

    render_context = _build_render_context(context, job, template, garment)
    outcome: RenderOutcome | None = None

    with log_context(job_id=job.id, backend=effective_backend), backend:
        try:
            backend_info = (
                backend.resume(render_context) if resume else backend.prepare(render_context)
            )
            capabilities = backend.capabilities()
            job = job.model_copy(
                update={
                    "backend_name": effective_backend,
                    "backend_version": capabilities.version,
                    "workflow_sha256": backend_info.get("workflow_sha256"),
                    "workflow_id": backend_info.get("workflow_id", job.workflow_id),
                    "status": JobStatus.RENDERING,
                    "started_at": job.started_at or utc_now(),
                    "error": None,
                }
            )
            job = context.repos.jobs.save(job)

            outcome = _render_frames(
                context,
                job,
                template,
                garment,
                backend,
                render_context,
                backend_info,
                progress=progress,
                max_frames=max_frames,
            )
        except Exception as exc:
            failed = _record_failure(context, job, exc)
            log_event(
                logger,
                "job_failed",
                job_id=failed.id,
                error_code=failed.error.code if failed.error else "unknown",
            )
            raise

    return outcome


def _build_render_context(
    context: ServiceContext,
    job: RenderJob,
    template: HumanTemplate,
    garment: GarmentAsset,
) -> RenderContext:
    from app.pipeline.garment_ingest import garment_image_paths

    workflow_path: Path | None = None
    if job.workflow_id:
        candidate = context.config.workflows_dir() / f"{job.workflow_id}.json"
        workflow_path = candidate if candidate.is_file() else None

    return RenderContext(
        job=job,
        template=template,
        garment=garment,
        config=context.config,
        source_frames_dir=context.absolute(template.directories.source_frames),
        job_dir=context.absolute(job.artifacts.root),
        raw_frames_dir=context.absolute(job.artifacts.raw_frames_dir),
        garment_image_paths=garment_image_paths(context, garment),
        workflow_path=workflow_path,
    )


def _render_frames(
    context: ServiceContext,
    job: RenderJob,
    template: HumanTemplate,
    garment: GarmentAsset,
    backend: RendererBackend,
    render_context: RenderContext,
    backend_info: dict[str, Any],
    *,
    progress: ProgressCallback | None,
    max_frames: int | None,
) -> RenderOutcome:
    config = context.config
    source_dir = context.absolute(template.directories.source_frames)
    composited_dir = context.absolute(job.artifacts.composited_frames_dir)
    raw_dir = context.absolute(job.artifacts.raw_frames_dir)
    masks_out_dir = context.absolute(job.artifacts.effective_masks_dir)
    for directory in (composited_dir, raw_dir, masks_out_dir):
        directory.mkdir(parents=True, exist_ok=True)

    mask_dirs = {kind: context.absolute(template.directories.mask_dir(kind)) for kind in MaskKind}
    frame_shape = (template.video.height, template.video.width)

    # Already-finished frames come from the crash-safe job_frames table, not
    # from the in-payload checkpoint, so a hard kill mid-write cannot lose them.
    already_done: set[int] = set(context.repos.jobs.completed_frames(job.id))
    pending = [index for index in job.frame_range.indices() if index not in already_done]
    skipped = sorted(already_done & set(job.frame_range.indices()))
    if max_frames is not None:
        pending = pending[:max_frames]

    capabilities = backend.capabilities()
    window_size = max(
        1, min(job.settings.frame_window, capabilities.max_frame_window, len(pending) or 1)
    )
    override_protected = _override_permits_protected_edit(context, job)

    rendered: list[int] = []
    mask_stats: dict[int, dict[str, Any]] = {}
    composite_stats: dict[int, dict[str, Any]] = {}
    leaked: list[int] = []
    per_frame_seeds: dict[str, int] = {}
    previous_render: np.ndarray | None = None
    window_index = job.checkpoint.window_index

    for offset in range(0, len(pending), window_size):
        chunk = pending[offset : offset + window_size]
        requests: list[FrameRequest] = []
        sources: dict[int, np.ndarray] = {}
        masks: dict[int, EffectiveMask] = {}

        for index in chunk:
            source = read_frame_at(source_dir, index, template=DEFAULT_TEMPLATE)
            if source.shape[:2] != frame_shape:
                raise ValidationError(
                    "Source frame dimensions do not match the template",
                    frame_index=index,
                    got=list(source.shape[:2]),
                    expected=list(frame_shape),
                )
            mask_set = load_mask_set(
                {kind: mask_dirs[kind] for kind in MaskKind},
                index,
                expect_shape=frame_shape,
                require=(MaskKind.GARMENT,),
            )
            effective = build_effective_mask(
                mask_set,
                feather_radius_px=job.settings.feather_radius_px,
                feather_sigma=config.mask.feather_sigma,
                expansion_dilate_px=job.settings.expansion_dilate_px,
                protected_dilate_px=job.settings.protected_dilate_px,
                protected_wins=job.settings.protected_wins,
                override_protected=override_protected,
            )
            if effective.editable_fraction < config.mask.min_editable_area_fraction:
                raise MaskError(
                    "Effective editable region is empty after applying protected "
                    "masks; nothing could be rendered for this frame",
                    frame_index=index,
                    editable_fraction=effective.editable_fraction,
                )
            if effective.editable_fraction > config.mask.max_editable_area_fraction:
                raise MaskError(
                    "Effective editable region is implausibly large; refusing to "
                    "render in case protected masks are missing",
                    frame_index=index,
                    editable_fraction=effective.editable_fraction,
                    limit=config.mask.max_editable_area_fraction,
                )

            save_mask(frame_path(masks_out_dir, index), effective.mask)
            sources[index] = source
            masks[index] = effective
            mask_stats[index] = effective.as_dict()

            seed = frame_seed(job.seed, index, job.garment_key)
            per_frame_seeds[str(index)] = seed
            requests.append(
                FrameRequest(
                    frame_index=index,
                    source=source,
                    effective_mask=effective.mask,
                    seed=seed,
                    prompt=job.prompt,
                    negative_prompt=job.negative_prompt,
                    settings=job.settings.model_dump(mode="json"),
                    previous_render=previous_render,
                    control=_control_inputs(context, template, index),
                )
            )

        results = _invoke_backend(backend, render_context, requests, job, window_index)

        for result in results:
            index = result.frame_index
            composited, stats = composite_with_stats(
                sources[index], result.image, masks[index].mask
            )
            if stats.leaked:
                # This is a hard invariant, not a warning: the compositor is
                # supposed to make leakage impossible, so a non-zero count means
                # a real bug and the job must stop.
                leaked.append(index)
                raise ValidationError(
                    "Compositing changed pixels outside the editable mask",
                    frame_index=index,
                    changed_outside_mask=stats.changed_outside_mask,
                    max_diff_outside_mask=stats.max_diff_outside_mask,
                )

            target = frame_path(composited_dir, index)
            write_frame(target, composited)
            digest = sha256_file(target)
            context.repos.jobs.record_frame(
                job.id,
                index,
                status="composited",
                frame_sha256=digest,
                frame_seed=result.seed,
                duration_ms=result.duration_ms,
            )
            composite_stats[index] = stats.as_dict()
            rendered.append(index)
            previous_render = result.image

            if progress is not None:
                progress(index, len(skipped) + len(rendered), job.frame_range.count)

        window_index += 1
        job = _save_checkpoint(context, job, rendered, skipped, window_index, backend_info)

    complete = not [
        index
        for index in job.frame_range.indices()
        if index not in set(context.repos.jobs.completed_frames(job.id))
    ]
    job = job.model_copy(
        update={
            "status": JobStatus.RENDERED if complete else JobStatus.PAUSED,
            "qc_metrics": {
                **job.qc_metrics,
                "render": {
                    "per_frame_seeds": per_frame_seeds,
                    "mask_stats_sample": _sample(mask_stats),
                    "composite_stats_sample": _sample(composite_stats),
                    "max_changed_outside_mask": max(
                        (s["changed_outside_mask"] for s in composite_stats.values()), default=0
                    ),
                },
            },
            "finished_at": utc_now() if complete else None,
        }
    )
    job = context.repos.jobs.save(job)
    log_event(
        logger,
        "job_rendered" if complete else "job_paused",
        job_id=job.id,
        rendered=len(rendered),
        skipped=len(skipped),
        complete=complete,
    )
    return RenderOutcome(
        job=job,
        rendered_frames=rendered,
        skipped_frames=skipped,
        backend_info={**backend_info, **backend.collect_artifacts(render_context)},
        mask_stats=mask_stats,
        composite_stats=composite_stats,
        leaked_frames=leaked,
    )


def _invoke_backend(
    backend: RendererBackend,
    render_context: RenderContext,
    requests: list[FrameRequest],
    job: RenderJob,
    window_index: int,
) -> list[FrameResult]:
    capabilities = backend.capabilities()
    if capabilities.supports_windows and len(requests) > 1:
        window = WindowRequest(
            frames=requests,
            window_index=window_index,
            seed=window_seed(job.seed, requests[0].frame_index, requests[-1].frame_index + 1),
        )
        result = backend.render_window(render_context, window)
        returned = {item.frame_index for item in result.results}
        expected = {item.frame_index for item in requests}
        if returned != expected:
            raise ValidationError(
                "Backend returned a different set of frames than requested",
                expected=sorted(expected),
                returned=sorted(returned),
            )
        return sorted(result.results, key=lambda item: item.frame_index)
    return [backend.render_frame(render_context, request) for request in requests]


def _control_inputs(
    context: ServiceContext, template: HumanTemplate, frame_index: int
) -> dict[str, Any]:
    """Optional auxiliary control data, when the operator has provided it."""
    control: dict[str, Any] = {}
    for key, relative in (
        ("pose", template.directories.pose),
        ("depth", template.directories.depth),
        ("face_landmarks", template.directories.face_landmarks),
    ):
        directory = context.absolute(relative)
        for suffix in (".json", ".png", ".npy", ".npz"):
            candidate = directory / f"frame_{frame_index:06d}{suffix}"
            if candidate.is_file():
                control[key] = str(candidate)
                break
    return control


def _override_permits_protected_edit(context: ServiceContext, job: RenderJob) -> bool:
    """Protected masks may only lose to an explicit, stored, reviewed override."""
    if not job.compatibility_report_id:
        return False
    report = context.repos.compatibility.try_get(job.compatibility_report_id)
    if report is None or report.override is None:
        return False
    if not report.override.is_valid_at(utc_now()):
        return False
    return "protected_mask_override" in set(report.override.acknowledged_rule_ids)


def _save_checkpoint(
    context: ServiceContext,
    job: RenderJob,
    rendered: list[int],
    skipped: list[int],
    window_index: int,
    backend_info: dict[str, Any],
) -> RenderJob:
    completed: list[int] = sorted(set(context.repos.jobs.completed_frames(job.id)))
    remaining = [index for index in job.frame_range.indices() if index not in set(completed)]
    checkpoint = Checkpoint(
        completed_frames=completed,
        last_completed_frame=completed[-1] if completed else None,
        next_frame=remaining[0] if remaining else None,
        window_index=window_index,
        backend_state={
            key: value
            for key, value in backend_info.items()
            if key in {"workflow_id", "workflow_sha256", "comfyui_version", "prompt_ids"}
        },
        updated_at=utc_now(),
    )
    updated = job.model_copy(
        update={
            "checkpoint": checkpoint,
            "progress": job.progress.model_copy(
                update={
                    "completed_frames": len(completed),
                    "current_frame": checkpoint.next_frame,
                    "started_at": job.progress.started_at or utc_now(),
                    "last_update_at": utc_now(),
                }
            ),
        }
    )
    return context.repos.jobs.save(updated)


def _record_failure(context: ServiceContext, job: RenderJob, exc: Exception) -> RenderJob:
    from app.core.errors import AppError

    details = exc.details if isinstance(exc, AppError) else {}
    code = exc.code if isinstance(exc, AppError) else type(exc).__name__
    failed = job.model_copy(
        update={
            "status": JobStatus.FAILED,
            "error": JobError(
                code=code,
                message=str(exc),
                frame_index=details.get("frame_index"),
                stage="render",
                details={k: v for k, v in details.items() if k != "frame_index"},
                occurred_at=utc_now(),
                retryable=code in {"backend_unavailable", "media_tool_error"},
            ),
        }
    )
    return context.repos.jobs.save(failed)


def _sample(values: dict[int, dict[str, Any]], limit: int = 5) -> dict[str, Any]:
    keys = sorted(values)[:limit]
    return {str(key): values[key] for key in keys}


def resume_job(
    context: ServiceContext,
    job_id: str,
    *,
    backend_name: str | None = None,
    progress: ProgressCallback | None = None,
    max_frames: int | None = None,
) -> RenderOutcome:
    """Resume an interrupted job, skipping frames already on disk."""
    job = context.repos.jobs.get(job_id)
    if not job.status.is_resumable:
        raise ConflictError("Job is not resumable", job_id=job.id, status=job.status.value)
    remaining = [
        index
        for index in job.frame_range.indices()
        if index not in set(context.repos.jobs.completed_frames(job.id))
    ]
    log_event(logger, "job_resume", job_id=job.id, remaining=len(remaining))
    if not remaining:
        updated = context.repos.jobs.save(job.model_copy(update={"status": JobStatus.RENDERED}))
        return RenderOutcome(
            job=updated,
            rendered_frames=[],
            skipped_frames=sorted(job.frame_range.indices()),
            backend_info={"resumed": True, "nothing_to_do": True},
        )
    return render_job(
        context,
        job_id,
        backend_name=backend_name,
        resume=True,
        progress=progress,
        max_frames=max_frames,
    )


def job_seeds(job: RenderJob) -> dict[str, int]:
    """Re-derive every per-frame seed from the job's immutable inputs."""
    return {
        str(index): frame_seed(job.seed, index, job.garment_key)
        for index in job.frame_range.indices()
    }


def job_signature(job: RenderJob) -> str:
    """Stable identity of a job's *inputs* (used to detect accidental drift)."""
    return derive_seed(
        job.template_key,
        job.garment_key,
        job.frame_range.start,
        job.frame_range.end,
        job.seed,
        job.backend_name,
        job.workflow_sha256 or "",
        job.prompt,
        job.negative_prompt,
        bits=64,
    ).__format__("016x")


__all__ = [
    "JobCreateOptions",
    "ProgressCallback",
    "RenderOutcome",
    "collect_input_hashes",
    "create_job",
    "job_seeds",
    "job_signature",
    "render_job",
    "resume_job",
]
