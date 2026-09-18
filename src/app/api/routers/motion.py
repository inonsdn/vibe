"""Motion Composition endpoints.

Thin adapters over ``app.pipeline.motion_ingest`` and
``app.pipeline.motion_compose`` — the same functions the CLI calls.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from app.api.deps import context_dependency
from app.api.schemas import (
    MotionComposeRequest,
    MotionIngestRequest,
    PoseImportRequest,
)
from app.pipeline.context import ServiceContext
from app.pipeline.motion_compose import (
    ComposeOptions,
    JoinSpec,
    SegmentSpec,
    compose_motion,
    load_profile,
    match_anchors,
    normalize_segment,
)
from app.pipeline.motion_ingest import (
    MotionIngestOptions,
    import_pose,
    ingest_motion_source,
    validate_motion_source,
)
from app.qc.motion_checks import run_motion_qc

router = APIRouter(prefix="/motion", tags=["motion"])

Ctx = Annotated[ServiceContext, Depends(context_dependency)]


@router.get("/profile")
def get_profile(context: Ctx) -> dict[str, Any]:
    """The canonical skeleton profile every motion is normalized into."""
    profile = load_profile(context)
    return {"profile": profile.model_dump(mode="json")}


@router.get("/sources")
def list_sources(context: Ctx, limit: int = 100, offset: int = 0) -> dict[str, Any]:
    sources = context.repos.motion_sources.list(limit=limit, offset=offset)
    return {"count": len(sources), "motion_sources": [s.to_json_dict() for s in sources]}


@router.post("/sources", status_code=201)
def create_source(payload: MotionIngestRequest, context: Ctx) -> dict[str, Any]:
    result = ingest_motion_source(
        context,
        payload.source_video,
        MotionIngestOptions(
            display_name=payload.display_name,
            start_frame=payload.start_frame,
            end_frame=payload.end_frame,
            motion_source_id=payload.motion_source_id,
            motion_use_authorized=payload.motion_use_authorized,
            rights_holder=payload.rights_holder,
            license=payload.license,
            acquired_from=payload.acquired_from,
            depicted_person_consent_ref=payload.depicted_person_consent_ref,
            allow_vfr=payload.allow_vfr,
        ),
    )
    return {"motion_source": result.source.to_json_dict(), "result": result.as_dict()}


@router.get("/sources/{motion_source_id}")
def get_source(motion_source_id: str, context: Ctx, version: int | None = None) -> dict[str, Any]:
    return {
        "motion_source": context.repos.motion_sources.get(motion_source_id, version).to_json_dict()
    }


@router.post("/sources/{motion_source_id}/pose")
def post_pose(
    motion_source_id: str,
    payload: PoseImportRequest,
    context: Ctx,
    version: int | None = None,
) -> dict[str, Any]:
    result = import_pose(
        context,
        motion_source_id,
        payload.source_dir,
        version=version,
        overwrite=payload.overwrite,
    )
    return result.as_dict()


@router.post("/sources/{motion_source_id}/validate")
def validate_source(
    motion_source_id: str, context: Ctx, version: int | None = None
) -> dict[str, Any]:
    return validate_motion_source(context, motion_source_id, version=version).as_dict()


@router.post("/sources/{motion_source_id}/normalize")
def normalize(
    motion_source_id: str,
    context: Ctx,
    version: int | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> dict[str, Any]:
    """Dry-run normalization: report the transform without composing."""
    profile = load_profile(context)
    record = context.repos.motion_sources.get(motion_source_id, version)
    segment = normalize_segment(
        context,
        SegmentSpec(
            motion_source_id=record.id,
            motion_source_version=record.version,
            start_frame=start_frame,
            end_frame=end_frame,
        ),
        profile,
        output_fps=record.video.fps,
    )
    return {
        "motion_source_id": record.id,
        "frames": len(segment.poses),
        "canonical_transform": segment.transform.model_dump(mode="json"),
        "warnings": segment.warnings,
    }


@router.get("/anchors")
def anchors(
    context: Ctx,
    prev_motion_id: str,
    next_motion_id: str,
    prev_start: int | None = None,
    prev_end: int | None = None,
    next_start: int | None = None,
    next_end: int | None = None,
    top: int = 5,
) -> dict[str, Any]:
    """Rank compatible join frames between two motion references."""
    profile = load_profile(context)
    prev_segment = normalize_segment(
        context,
        SegmentSpec(motion_source_id=prev_motion_id, start_frame=prev_start, end_frame=prev_end),
        profile,
        output_fps=context.config.video.fps,
    )
    next_segment = normalize_segment(
        context,
        SegmentSpec(motion_source_id=next_motion_id, start_frame=next_start, end_frame=next_end),
        profile,
        output_fps=context.config.video.fps,
    )
    candidates = match_anchors(context, prev_segment, next_segment, profile)
    return {
        "total": len(candidates),
        "candidates": [c.as_dict() for c in candidates[:top]],
    }


@router.get("/compositions")
def list_compositions(context: Ctx, limit: int = 100) -> dict[str, Any]:
    compositions = context.repos.compositions.list(limit=limit)
    return {
        "count": len(compositions),
        "compositions": [c.to_json_dict() for c in compositions],
    }


@router.post("/compositions", status_code=201)
def create_composition(payload: MotionComposeRequest, context: Ctx) -> dict[str, Any]:
    result = compose_motion(
        context,
        ComposeOptions(
            display_name=payload.display_name,
            segments=[
                SegmentSpec(
                    motion_source_id=segment.motion_source_id,
                    motion_source_version=segment.motion_source_version,
                    start_frame=segment.start_frame,
                    end_frame=segment.end_frame,
                    playback_speed=segment.playback_speed,
                    trim_start=segment.trim_start,
                    trim_end=segment.trim_end,
                    exposed_views=[v.value for v in segment.exposed_views],
                )
                for segment in payload.segments
            ],
            joins=[
                JoinSpec(
                    prev_frame=join.prev_frame,
                    next_frame=join.next_frame,
                    bridge_frames=join.bridge_frames,
                )
                for join in payload.joins
            ],
            output_fps=payload.output_fps,
            composition_id=payload.composition_id,
            make_preview=payload.make_preview,
        ),
    )
    return {"composition": result.composition.to_json_dict(), "result": result.as_dict()}


@router.get("/compositions/{composition_id}")
def get_composition(
    composition_id: str, context: Ctx, version: int | None = None
) -> dict[str, Any]:
    return {"composition": context.repos.compositions.get(composition_id, version).to_json_dict()}


@router.post("/compositions/{composition_id}/qc")
def post_composition_qc(
    composition_id: str, context: Ctx, version: int | None = None
) -> dict[str, Any]:
    return run_motion_qc(context, composition_id, version=version).as_dict()


__all__ = ["router"]
