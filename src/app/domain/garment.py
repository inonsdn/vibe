"""Garment reference assets.

A garment is described by reference images plus structured attributes. The
attributes — not the pixels — drive the deterministic compatibility engine, so
they are mandatory and strictly typed. ``source_url`` is metadata only: nothing
in the system ever fetches it.
"""

from __future__ import annotations

from typing import Self

from pydantic import Field, field_validator, model_validator

from app.domain.base import DomainModel, Identifier, Sha256, TimestampedModel
from app.domain.enums import (
    SHEER_MATERIALS,
    BodyCoverage,
    GarmentCategory,
    GarmentLength,
    ImageViewType,
    IngestionStatus,
    Material,
    Silhouette,
    SleeveLength,
)


class UsageRights(DomainModel):
    """Licensing / usage-rights metadata for the garment references."""

    license: str | None = None
    rights_holder: str | None = None
    commercial_use_allowed: bool | None = None
    attribution_required: bool = False
    attribution_text: str | None = None
    restrictions: list[str] = Field(default_factory=list)
    acquired_from: str | None = None
    acquired_at: str | None = None

    @property
    def is_documented(self) -> bool:
        return bool(self.license or self.rights_holder or self.acquired_from)


class GarmentReferenceImage(DomainModel):
    """One reference image with its view type, hash and measured quality."""

    path: str
    view: ImageViewType
    sha256: Sha256
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    alpha_mask_path: str | None = None
    sharpness_score: float | None = Field(default=None, ge=0)
    #: Fraction of the image occupied by the garment, when known.
    subject_area_fraction: float | None = Field(default=None, ge=0, le=1)
    notes: str | None = None

    @property
    def megapixels(self) -> float:
        return (self.width * self.height) / 1_000_000

    @property
    def min_side(self) -> int:
        return min(self.width, self.height)

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height


class GarmentAsset(TimestampedModel):
    """A garment the operator wants to place on the master performance."""

    id: Identifier
    version: int = Field(default=1, ge=1)

    # -- commerce metadata (never fetched, only recorded) -----------------
    product_id: str | None = None
    source_id: str | None = None
    brand: str | None = None
    product_name: str | None = None
    source_url: str | None = Field(default=None, max_length=2048)
    affiliate_url: str | None = Field(default=None, max_length=2048)
    affiliate_network: str | None = None
    affiliate_tag: str | None = None
    price_text: str | None = None
    currency: str | None = Field(default=None, max_length=8)
    usage_rights: UsageRights = Field(default_factory=UsageRights)

    # -- references -------------------------------------------------------
    images: list[GarmentReferenceImage] = Field(default_factory=list)

    # -- structured garment attributes -----------------------------------
    category: GarmentCategory
    sleeve_length: SleeveLength = SleeveLength.NOT_APPLICABLE
    garment_length: GarmentLength = GarmentLength.NOT_APPLICABLE
    body_coverage: BodyCoverage
    silhouette: Silhouette
    material: Material
    transparency: float = Field(default=0.0, ge=0.0, le=1.0)
    reflectivity: float = Field(default=0.0, ge=0.0, le=1.0)
    fabric_flow: float = Field(default=0.0, ge=0.0, le=1.0)
    dominant_colors: list[str] = Field(default_factory=list)
    pattern_description: str | None = None
    requires_underlayer: bool = False
    structured_notes: dict[str, str] = Field(default_factory=dict)

    status: IngestionStatus = IngestionStatus.CREATED

    @field_validator("dominant_colors")
    @classmethod
    def _hex_colors(cls, value: list[str]) -> list[str]:
        for color in value:
            if not color.startswith("#") or len(color) not in (4, 7):
                raise ValueError(f"dominant color must be a hex string like #RRGGBB, got {color!r}")
        return value

    @field_validator("source_url", "affiliate_url")
    @classmethod
    def _url_is_metadata_only(cls, value: str | None) -> str | None:
        # Recorded verbatim for attribution; deliberately never dereferenced.
        if value is not None and not value.startswith(("http://", "https://")):
            raise ValueError("URL metadata must be an http(s) URL")
        return value

    @model_validator(mode="after")
    def _sheer_material_implies_transparency(self) -> Self:
        if self.material in SHEER_MATERIALS and self.transparency == 0.0:
            raise ValueError(
                f"material {self.material.value!r} is inherently sheer; "
                "set transparency > 0 (or choose another material)"
            )
        return self

    # -- convenience ------------------------------------------------------
    @property
    def available_views(self) -> set[ImageViewType]:
        return {image.view for image in self.images}

    def images_for(self, view: ImageViewType) -> list[GarmentReferenceImage]:
        return [image for image in self.images if image.view == view]

    def has_view(self, view: ImageViewType) -> bool:
        return view in self.available_views

    @property
    def is_transparent(self) -> bool:
        return self.transparency > 0.0

    @property
    def metadata_completeness(self) -> float:
        """Fraction of the optional-but-wanted metadata fields present."""
        fields = (
            self.product_name,
            self.brand,
            self.pattern_description,
            self.usage_rights.license or self.usage_rights.rights_holder,
            self.dominant_colors or None,
        )
        return sum(1 for value in fields if value) / len(fields)

    def version_key(self) -> str:
        return f"{self.id}@v{self.version}"


__all__ = ["GarmentAsset", "GarmentReferenceImage", "UsageRights"]
