"""Master Human Performance creation endpoints.

A synthetic master is a **candidate** until an operator accepts it. The
acceptance endpoint is the only route to that, and it is audited.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends

from app.api.deps import context_dependency
from app.api.schemas import (
    HeroRegisterRequest,
    MasterAcceptRequest,
    MasterAnimateRequest,
    MasterCreateRequest,
    MasterRejectRequest,
)
from app.backends.animator.registry import available_animators, create_animator
from app.pipeline.context import ServiceContext
from app.pipeline.master_create import (
    HeroOptions,
    MasterCreateOptions,
    accept_master,
    animate_master,
    create_master_candidate,
    register_hero,
    reject_master,
    resume_master,
    write_master_manifest,
)
from app.qc.motion_checks import run_master_qc

router = APIRouter(prefix="/master", tags=["master"])

Ctx = Annotated[ServiceContext, Depends(context_dependency)]


@router.get("/animators")
def animators(context: Ctx) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in available_animators():
        backend = create_animator(name, context.config)
        try:
            out[name] = {
                "capabilities": backend.capabilities().as_dict(),
                "health": backend.healthcheck().as_dict(),
            }
        finally:
            backend.close()
    return {"animators": out}


@router.get("/heroes")
def list_heroes(context: Ctx, limit: int = 100) -> dict[str, Any]:
    heroes = context.repos.heroes.list(limit=limit)
    return {"count": len(heroes), "heroes": [h.to_json_dict() for h in heroes]}


@router.post("/heroes", status_code=201)
def create_hero(payload: HeroRegisterRequest, context: Ctx) -> dict[str, Any]:
    hero = register_hero(
        context,
        HeroOptions(
            display_name=payload.display_name,
            reference_images=[Path(p) for p in payload.reference_images],
            subject_kind=payload.subject_kind,
            consent_document_ref=payload.consent_document_ref,
            rights_holder=payload.rights_holder,
            license=payload.license,
            hero_id=payload.hero_id,
        ),
    )
    return {"hero": hero.to_json_dict()}


@router.get("/candidates")
def list_candidates(context: Ctx, limit: int = 100) -> dict[str, Any]:
    candidates = context.repos.masters.list(limit=limit)
    return {
        "count": len(candidates),
        "candidates": [c.to_json_dict() for c in candidates],
        "note": "A candidate is not a master until it is accepted.",
    }


@router.post("/candidates", status_code=201)
def create_candidate(payload: MasterCreateRequest, context: Ctx) -> dict[str, Any]:
    candidate = create_master_candidate(
        context,
        MasterCreateOptions(
            display_name=payload.display_name,
            composition_id=payload.composition_id,
            hero_character_id=payload.hero_character_id,
            composition_version=payload.composition_version,
            hero_version=payload.hero_version,
            backend_name=payload.backend,
            seed=payload.seed,
            chunk_frames=payload.chunk_frames,
            overlap_frames=payload.overlap_frames,
            candidate_id=payload.candidate_id,
        ),
    )
    return {"candidate": candidate.to_json_dict()}


@router.get("/candidates/{candidate_id}")
def get_candidate(candidate_id: str, context: Ctx) -> dict[str, Any]:
    candidate = context.repos.masters.get(candidate_id)
    return {
        "candidate": candidate.to_json_dict(),
        "frames_done": len(candidate.completed_chunk_frames()),
        "remaining_frames": len(candidate.remaining_frames()),
        "accepted": candidate.is_accepted,
    }


@router.post("/candidates/{candidate_id}/animate")
def post_animate(candidate_id: str, payload: MasterAnimateRequest, context: Ctx) -> dict[str, Any]:
    function = resume_master if payload.resume else animate_master
    result = function(
        context, candidate_id, backend_name=payload.backend, max_chunks=payload.max_chunks
    )
    return result.as_dict()


@router.post("/candidates/{candidate_id}/qc")
def post_qc(candidate_id: str, context: Ctx) -> dict[str, Any]:
    candidate = context.repos.masters.get(candidate_id)
    write_master_manifest(context, candidate)
    return run_master_qc(context, candidate_id).as_dict()


@router.get("/candidates/{candidate_id}/manifest")
def get_manifest(candidate_id: str, context: Ctx) -> dict[str, Any]:
    manifest = context.repos.masters.get_manifest(candidate_id)
    return {"manifest": manifest, "digest": (manifest or {}).get("digest")}


@router.post("/candidates/{candidate_id}/accept")
def post_accept(candidate_id: str, payload: MasterAcceptRequest, context: Ctx) -> dict[str, Any]:
    candidate = accept_master(
        context,
        candidate_id,
        accepted_by=payload.accepted_by,
        reason=payload.reason,
        acknowledged_warnings=payload.acknowledged_warnings,
        require_qc_pass=not payload.allow_qc_failure,
    )
    return {"candidate": candidate.to_json_dict(), "accepted": candidate.is_accepted}


@router.post("/candidates/{candidate_id}/reject")
def post_reject(candidate_id: str, payload: MasterRejectRequest, context: Ctx) -> dict[str, Any]:
    candidate = reject_master(
        context, candidate_id, rejected_by=payload.rejected_by, reason=payload.reason
    )
    return {"candidate": candidate.to_json_dict()}


__all__ = ["router"]
