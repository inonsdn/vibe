"""Garment endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends

from app.api.deps import context_dependency
from app.api.schemas import GarmentIngestRequest
from app.pipeline.context import ServiceContext
from app.pipeline.garment_ingest import GarmentIngestOptions, ImageSpec, ingest_garment

router = APIRouter(prefix="/garments", tags=["garments"])

Ctx = Annotated[ServiceContext, Depends(context_dependency)]


@router.get("")
def list_garments(context: Ctx, limit: int = 100, offset: int = 0) -> dict[str, Any]:
    garments = context.repos.garments.list(limit=limit, offset=offset)
    return {"count": len(garments), "garments": [g.to_json_dict() for g in garments]}


@router.post("", status_code=201)
def create_garment(payload: GarmentIngestRequest, context: Ctx) -> dict[str, Any]:
    specs = [
        ImageSpec(
            path=Path(image.path),
            view=image.view,
            alpha_mask=Path(image.alpha_mask) if image.alpha_mask else None,
            notes=image.notes,
        )
        for image in payload.images
    ]
    options = GarmentIngestOptions(
        category=payload.category,
        body_coverage=payload.body_coverage,
        silhouette=payload.silhouette,
        material=payload.material,
        sleeve_length=payload.sleeve_length,
        garment_length=payload.garment_length,
        transparency=payload.transparency,
        reflectivity=payload.reflectivity,
        fabric_flow=payload.fabric_flow,
        dominant_colors=payload.dominant_colors,
        pattern_description=payload.pattern_description,
        requires_underlayer=payload.requires_underlayer,
        product_name=payload.product_name,
        brand=payload.brand,
        source_url=payload.source_url,
        license=payload.license,
        rights_holder=payload.rights_holder,
        garment_id=payload.garment_id,
    )
    result = ingest_garment(context, specs, options)
    return {"garment": result.garment.to_json_dict(), "warnings": result.warnings}


@router.get("/{garment_id}")
def get_garment(garment_id: str, context: Ctx, version: int | None = None) -> dict[str, Any]:
    return {"garment": context.repos.garments.get(garment_id, version).to_json_dict()}


__all__ = ["router"]
