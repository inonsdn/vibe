"""Render job endpoints.

Rendering is synchronous by design: one job at a time on an 8GB GPU, driven by
an operator who is watching. ``max_frames`` keeps a request bounded, and a
partially rendered job is always resumable, so a long render can be driven in
slices without any background-worker machinery.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from app.api.deps import context_dependency
from app.api.schemas import JobComposeRequest, JobCreateRequest, JobQCRequest, JobRenderRequest
from app.backends.registry import available_backends, create_backend
from app.domain.enums import JobStatus
from app.pipeline.compose import ComposeOptions, compose_job
from app.pipeline.context import ServiceContext
from app.pipeline.render import JobCreateOptions, create_job, render_job, resume_job
from app.qc.report import QCOptions, run_qc

router = APIRouter(prefix="/jobs", tags=["jobs"])

Ctx = Annotated[ServiceContext, Depends(context_dependency)]


@router.get("/backends")
def backends(context: Ctx) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in available_backends():
        backend = create_backend(name, context.config)
        try:
            out[name] = {
                "capabilities": backend.capabilities().as_dict(),
                "health": backend.healthcheck().as_dict(),
            }
        finally:
            backend.close()
    return {"backends": out}


@router.get("")
def list_jobs(
    context: Ctx,
    status: JobStatus | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    jobs = context.repos.jobs.list(status=status, limit=limit, offset=offset)
    return {"count": len(jobs), "jobs": [job.to_json_dict() for job in jobs]}


@router.post("", status_code=201)
def post_job(payload: JobCreateRequest, context: Ctx) -> dict[str, Any]:
    job = create_job(
        context,
        payload.template_id,
        payload.garment_id,
        JobCreateOptions(
            backend_name=payload.backend,
            seed=payload.seed,
            prompt=payload.prompt,
            negative_prompt=payload.negative_prompt,
            workflow_id=payload.workflow_id,
            frame_end=payload.frame_end,
        ),
        template_version=payload.template_version,
        garment_version=payload.garment_version,
    )
    return {"job": job.to_json_dict()}


@router.get("/{job_id}")
def get_job(job_id: str, context: Ctx) -> dict[str, Any]:
    job = context.repos.jobs.get(job_id)
    completed = context.repos.jobs.completed_frames(job_id)
    remaining = [i for i in job.frame_range.indices() if i not in set(completed)]
    return {
        "job": job.to_json_dict(),
        "completed_frames": len(completed),
        "remaining_frames": len(remaining),
        "next_frame": remaining[0] if remaining else None,
    }


@router.post("/{job_id}/render")
def post_render(job_id: str, payload: JobRenderRequest, context: Ctx) -> dict[str, Any]:
    function = resume_job if payload.resume else render_job
    kwargs: dict[str, Any] = {
        "backend_name": payload.backend,
        "max_frames": payload.max_frames,
    }
    outcome = function(context, job_id, **kwargs)
    return outcome.as_dict()


@router.post("/{job_id}/compose")
def post_compose(job_id: str, payload: JobComposeRequest, context: Ctx) -> dict[str, Any]:
    result = compose_job(
        context,
        job_id,
        ComposeOptions(
            include_audio=payload.include_audio,
            flash_frames=payload.flash_frames,
            make_preview=payload.make_preview,
            output_name=payload.output_name,
            overwrite=payload.overwrite,
        ),
    )
    return result.as_dict()


@router.post("/{job_id}/qc")
def post_qc(job_id: str, payload: JobQCRequest, context: Ctx) -> dict[str, Any]:
    report = run_qc(
        context,
        job_id,
        QCOptions(
            max_sampled_frames=payload.max_sampled_frames,
            make_contact_sheets=payload.make_contact_sheets,
        ),
    )
    return report.as_dict()


@router.get("/{job_id}/manifest")
def get_manifest(job_id: str, context: Ctx) -> dict[str, Any]:
    manifest = context.repos.jobs.get_manifest(job_id)
    if manifest is None:
        return {"manifest": None, "digest": None}
    return {
        "manifest": manifest.model_dump(mode="json"),
        "digest": manifest.reproducibility_digest(),
        "missing_fields": manifest.required_fields_present(),
    }


__all__ = ["router"]
