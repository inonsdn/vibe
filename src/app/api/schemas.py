"""HTTP request/response schemas.

Kept separate from the domain models: the domain models are the storage
contract, these are the wire contract. Where a response is just a domain model,
it is serialised directly rather than re-declared.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.domain.enums import (
    BodyCoverage,
    GarmentCategory,
    GarmentLength,
    ImageViewType,
    MaskKind,
    Material,
    Silhouette,
    SleeveLength,
    TemplateClothingClass,
)


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ErrorResponse(ApiModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(ApiModel):
    status: str
    app: str
    version: str
    schema_version: int
    offline_ok: bool
    backends: list[str]


class TemplateIngestRequest(ApiModel):
    source_video: str = Field(description="Absolute path to the master performance video.")
    display_name: str
    intro_start: int = 0
    transition_anchor: int | None = None
    reveal_end: int | None = None
    template_clothing_class: TemplateClothingClass = TemplateClothingClass.FITTED_SHORT
    subject_kind: str = "synthetic"
    consent_document_ref: str | None = None
    rights_holder: str | None = None
    license: str | None = None
    identity_reference_images: list[str] = Field(default_factory=list)
    template_id: str | None = None
    allow_vfr_conversion: bool = False


class MaskImportRequest(ApiModel):
    kind: MaskKind
    source_dir: str
    overwrite: bool = False


class GarmentImageRequest(ApiModel):
    path: str
    view: ImageViewType
    alpha_mask: str | None = None
    notes: str | None = None


class GarmentIngestRequest(ApiModel):
    images: list[GarmentImageRequest] = Field(min_length=1)
    category: GarmentCategory
    body_coverage: BodyCoverage
    silhouette: Silhouette
    material: Material
    sleeve_length: SleeveLength = SleeveLength.NOT_APPLICABLE
    garment_length: GarmentLength = GarmentLength.NOT_APPLICABLE
    transparency: float = Field(default=0.0, ge=0.0, le=1.0)
    reflectivity: float = Field(default=0.0, ge=0.0, le=1.0)
    fabric_flow: float = Field(default=0.0, ge=0.0, le=1.0)
    dominant_colors: list[str] = Field(default_factory=list)
    pattern_description: str | None = None
    requires_underlayer: bool = False
    product_name: str | None = None
    brand: str | None = None
    source_url: str | None = None
    license: str | None = None
    rights_holder: str | None = None
    garment_id: str | None = None


class CompatibilityCheckRequest(ApiModel):
    template_id: str
    garment_id: str
    template_version: int | None = None
    garment_version: int | None = None
    exposed_views: list[ImageViewType] | None = None


class OverrideRequest(ApiModel):
    reviewer: str = Field(min_length=1)
    reason: str = Field(min_length=10)
    acknowledged_rule_ids: list[str] = Field(default_factory=list)
    expires_in_hours: float | None = None


class JobCreateRequest(ApiModel):
    template_id: str
    garment_id: str
    backend: str = "mock"
    seed: int | None = None
    prompt: str = ""
    negative_prompt: str = ""
    workflow_id: str | None = None
    frame_end: int | None = None
    template_version: int | None = None
    garment_version: int | None = None


class JobRenderRequest(ApiModel):
    backend: str | None = None
    max_frames: int | None = Field(default=None, ge=1)
    resume: bool = False


class JobComposeRequest(ApiModel):
    include_audio: bool = True
    flash_frames: int | None = None
    make_preview: bool = False
    output_name: str | None = None
    overwrite: bool = False


class JobQCRequest(ApiModel):
    max_sampled_frames: int | None = Field(default=None, ge=1)
    make_contact_sheets: bool = True


__all__ = [
    "ApiModel",
    "CompatibilityCheckRequest",
    "ErrorResponse",
    "GarmentImageRequest",
    "GarmentIngestRequest",
    "HealthResponse",
    "JobComposeRequest",
    "JobCreateRequest",
    "JobQCRequest",
    "JobRenderRequest",
    "MaskImportRequest",
    "OverrideRequest",
    "TemplateIngestRequest",
]
