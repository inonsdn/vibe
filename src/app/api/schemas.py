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


class MotionIngestRequest(ApiModel):
    source_video: str = Field(description="Absolute path to the motion reference video.")
    display_name: str
    start_frame: int = 0
    end_frame: int | None = None
    motion_source_id: str | None = None
    motion_use_authorized: bool = False
    rights_holder: str | None = None
    license: str | None = None
    acquired_from: str | None = None
    depicted_person_consent_ref: str | None = None
    allow_vfr: bool = False


class PoseImportRequest(ApiModel):
    source_dir: str
    overwrite: bool = False


class MotionSegmentRequest(ApiModel):
    motion_source_id: str
    motion_source_version: int | None = None
    start_frame: int | None = None
    end_frame: int | None = None
    playback_speed: float = Field(default=1.0, gt=0.0, le=4.0)
    trim_start: int = Field(default=0, ge=0)
    trim_end: int = Field(default=0, ge=0)
    exposed_views: list[ImageViewType] = Field(default_factory=list)


class MotionJoinRequest(ApiModel):
    prev_frame: int | None = None
    next_frame: int | None = None
    bridge_frames: int | None = Field(default=None, ge=2)


class MotionComposeRequest(ApiModel):
    display_name: str
    segments: list[MotionSegmentRequest] = Field(min_length=1)
    joins: list[MotionJoinRequest] = Field(default_factory=list)
    output_fps: float | None = Field(default=None, gt=0, le=240)
    composition_id: str | None = None
    make_preview: bool = True


class HeroRegisterRequest(ApiModel):
    display_name: str
    reference_images: list[str] = Field(min_length=1)
    subject_kind: str = "synthetic"
    consent_document_ref: str | None = None
    rights_holder: str | None = None
    license: str | None = None
    hero_id: str | None = None


class MasterCreateRequest(ApiModel):
    display_name: str
    composition_id: str
    hero_character_id: str
    composition_version: int | None = None
    hero_version: int | None = None
    backend: str = "mock"
    seed: int | None = None
    chunk_frames: int | None = Field(default=None, ge=1)
    overlap_frames: int | None = Field(default=None, ge=0)
    candidate_id: str | None = None


class MasterAnimateRequest(ApiModel):
    backend: str | None = None
    max_chunks: int | None = Field(default=None, ge=1)
    resume: bool = False


class MasterAcceptRequest(ApiModel):
    accepted_by: str = Field(min_length=1)
    reason: str = Field(min_length=10)
    acknowledged_warnings: list[str] = Field(default_factory=list)
    allow_qc_failure: bool = False


class MasterRejectRequest(ApiModel):
    rejected_by: str = Field(min_length=1)
    reason: str = Field(min_length=10)


__all__ = [
    "ApiModel",
    "CompatibilityCheckRequest",
    "ErrorResponse",
    "GarmentImageRequest",
    "GarmentIngestRequest",
    "HealthResponse",
    "HeroRegisterRequest",
    "JobComposeRequest",
    "JobCreateRequest",
    "JobQCRequest",
    "JobRenderRequest",
    "MaskImportRequest",
    "MasterAcceptRequest",
    "MasterAnimateRequest",
    "MasterCreateRequest",
    "MasterRejectRequest",
    "MotionComposeRequest",
    "MotionIngestRequest",
    "MotionJoinRequest",
    "MotionSegmentRequest",
    "OverrideRequest",
    "PoseImportRequest",
    "TemplateIngestRequest",
]
