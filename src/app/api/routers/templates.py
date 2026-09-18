"""Template endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends

from app.api.deps import context_dependency
from app.api.schemas import MaskImportRequest, TemplateIngestRequest
from app.domain.enums import MaskKind
from app.pipeline.context import ServiceContext
from app.pipeline.template_ingest import (
    IngestOptions,
    import_masks,
    ingest_template,
    validate_template,
)

router = APIRouter(prefix="/templates", tags=["templates"])

Ctx = Annotated[ServiceContext, Depends(context_dependency)]


@router.get("")
def list_templates(context: Ctx, limit: int = 100, offset: int = 0) -> dict[str, Any]:
    templates = context.repos.templates.list(limit=limit, offset=offset)
    return {"count": len(templates), "templates": [t.to_json_dict() for t in templates]}


@router.post("", status_code=201)
def create_template(payload: TemplateIngestRequest, context: Ctx) -> dict[str, Any]:
    options = IngestOptions(
        display_name=payload.display_name,
        intro_start=payload.intro_start,
        transition_anchor=payload.transition_anchor,
        reveal_end=payload.reveal_end,
        template_clothing_class=payload.template_clothing_class.value,
        subject_kind=payload.subject_kind,
        consent_document_ref=payload.consent_document_ref,
        rights_holder=payload.rights_holder,
        license=payload.license,
        identity_reference_images=[Path(p) for p in payload.identity_reference_images],
        template_id=payload.template_id,
        allow_vfr_conversion=payload.allow_vfr_conversion,
    )
    result = ingest_template(context, payload.source_video, options)
    return {"template": result.template.to_json_dict(), "result": result.as_dict()}


@router.get("/{template_id}")
def get_template(template_id: str, context: Ctx, version: int | None = None) -> dict[str, Any]:
    return {"template": context.repos.templates.get(template_id, version).to_json_dict()}


@router.post("/{template_id}/validate")
def validate(template_id: str, context: Ctx, version: int | None = None) -> dict[str, Any]:
    return validate_template(context, template_id, version=version).as_dict()


@router.post("/{template_id}/masks")
def post_masks(
    template_id: str,
    payload: MaskImportRequest,
    context: Ctx,
    version: int | None = None,
) -> dict[str, Any]:
    result = import_masks(
        context,
        template_id,
        MaskKind(payload.kind),
        payload.source_dir,
        version=version,
        overwrite=payload.overwrite,
    )
    return result.as_dict()


@router.get("/{template_id}/audit")
def audit(template_id: str, context: Ctx) -> dict[str, Any]:
    return {"events": context.repos.audit.for_entity("template", template_id)}


__all__ = ["router"]
