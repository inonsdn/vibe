"""Garment ingestion.

Reference images are copied into the data root (never referenced in place, so a
job cannot break because the operator moved a download), hashed, and measured:
dimensions, sharpness and subject area feed the quality rules. Metadata is
recorded verbatim, including ``source_url`` — which is never fetched.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.errors import NotFoundError, ValidationError
from app.core.hashing import sha256_file
from app.core.ids import garment_id as new_garment_id
from app.core.logging import get_logger, log_event
from app.core.paths import safe_identifier
from app.domain.enums import (
    BodyCoverage,
    GarmentCategory,
    GarmentLength,
    ImageViewType,
    IngestionStatus,
    Material,
    Silhouette,
    SleeveLength,
)
from app.domain.garment import GarmentAsset, GarmentReferenceImage, UsageRights
from app.pipeline.context import ServiceContext

logger = get_logger(__name__)

SUPPORTED_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})


@dataclass
class ImageSpec:
    """One image the operator wants to attach, with its declared view."""

    path: Path
    view: ImageViewType
    alpha_mask: Path | None = None
    notes: str | None = None


@dataclass
class GarmentIngestOptions:
    category: GarmentCategory
    body_coverage: BodyCoverage
    silhouette: Silhouette
    material: Material
    sleeve_length: SleeveLength = SleeveLength.NOT_APPLICABLE
    garment_length: GarmentLength = GarmentLength.NOT_APPLICABLE
    transparency: float = 0.0
    reflectivity: float = 0.0
    fabric_flow: float = 0.0
    dominant_colors: list[str] = field(default_factory=list)
    pattern_description: str | None = None
    requires_underlayer: bool = False
    product_name: str | None = None
    brand: str | None = None
    product_id: str | None = None
    source_id: str | None = None
    source_url: str | None = None
    affiliate_url: str | None = None
    affiliate_network: str | None = None
    affiliate_tag: str | None = None
    price_text: str | None = None
    currency: str | None = None
    license: str | None = None
    rights_holder: str | None = None
    commercial_use_allowed: bool | None = None
    acquired_from: str | None = None
    structured_notes: dict[str, str] = field(default_factory=dict)
    garment_id: str | None = None
    version: int = 1
    #: Derive dominant colours from the front image when none are declared.
    auto_dominant_colors: bool = True


@dataclass
class GarmentIngestResult:
    garment: GarmentAsset
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "garment_id": self.garment.id,
            "version": self.garment.version,
            "status": self.garment.status.value,
            "images": [
                {"view": image.view.value, "path": image.path, "sha256": image.sha256[:12]}
                for image in self.garment.images
            ],
            "views": sorted(v.value for v in self.garment.available_views),
            "metadata_completeness": round(self.garment.metadata_completeness, 3),
            "warnings": self.warnings,
        }


def measure_image(path: Path) -> tuple[int, int, float, float]:
    """Return ``(width, height, sharpness, subject_area_fraction)``.

    Sharpness is a normalised variance-of-Laplacian; subject area is the
    fraction of pixels that differ from the (assumed flat) background corners,
    which is a cheap proxy for "is the product actually in frame".
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValidationError("Reference image could not be read", path=str(path))
    height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # Normalise into a 0..1-ish band; 500 is a typical crisp-photo variance.
    sharpness = float(min(1.0, variance / 500.0))

    corners = np.concatenate(
        [
            gray[:16, :16].reshape(-1),
            gray[:16, -16:].reshape(-1),
            gray[-16:, :16].reshape(-1),
            gray[-16:, -16:].reshape(-1),
        ]
    )
    background = float(np.median(corners)) if corners.size else 0.0
    subject = float(np.count_nonzero(np.abs(gray.astype(np.int16) - background) > 18) / gray.size)
    return width, height, round(sharpness, 6), round(subject, 6)


def dominant_colors(path: Path, count: int = 3) -> list[str]:
    """Deterministic dominant colours via fixed-seed k-means."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return []
    small = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    samples = small.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    # PP_CENTERS with a fixed attempt count keeps this reproducible enough for
    # metadata; the values are advisory, not part of any hash-critical path.
    # cv2's type stubs do not model this overload; the runtime call is correct.
    _, labels, centers = cv2.kmeans(  # type: ignore[call-overload]
        samples, count, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    order = np.argsort(-np.bincount(labels.flatten(), minlength=count))
    out: list[str] = []
    for index in order:
        b, g, r = (round(v) for v in centers[index])
        out.append(f"#{r:02x}{g:02x}{b:02x}")
    return out


def ingest_garment(
    context: ServiceContext,
    images: list[ImageSpec],
    options: GarmentIngestOptions,
) -> GarmentIngestResult:
    """Copy, hash and measure garment references; persist the asset."""
    if not images:
        raise ValidationError("At least one reference image is required")

    identifier = safe_identifier(options.garment_id or new_garment_id())
    garment_dir = context.data_root.garment_dir(identifier)
    images_dir = garment_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    records: list[GarmentReferenceImage] = []
    seen_hashes: dict[str, str] = {}

    for spec in images:
        source = Path(spec.path).expanduser().resolve()
        if not source.is_file():
            raise NotFoundError("Reference image not found", path=str(source))
        if source.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValidationError(
                "Unsupported image format",
                path=str(source),
                supported=sorted(SUPPORTED_SUFFIXES),
            )
        destination = images_dir / f"{spec.view.value}_{source.name}"
        shutil.copy2(source, destination)

        digest = sha256_file(destination)
        if digest in seen_hashes:
            warnings.append(
                f"{destination.name} is byte-identical to {seen_hashes[digest]}; "
                "duplicate views weaken the compatibility checks"
            )
        seen_hashes[digest] = destination.name

        width, height, sharpness, subject = measure_image(destination)
        alpha_relative: str | None = None
        if spec.alpha_mask is not None:
            alpha_source = Path(spec.alpha_mask).expanduser().resolve()
            if not alpha_source.is_file():
                raise NotFoundError("Alpha mask not found", path=str(alpha_source))
            alpha_destination = images_dir / f"{spec.view.value}_alpha_{alpha_source.name}"
            shutil.copy2(alpha_source, alpha_destination)
            alpha_relative = context.relative(alpha_destination)

        records.append(
            GarmentReferenceImage(
                path=context.relative(destination),
                view=spec.view,
                sha256=digest,
                width=width,
                height=height,
                alpha_mask_path=alpha_relative,
                sharpness_score=sharpness,
                subject_area_fraction=subject,
                notes=spec.notes,
            )
        )

    colors = list(options.dominant_colors)
    if not colors and options.auto_dominant_colors:
        front = next((r for r in records if r.view is ImageViewType.FRONT), records[0])
        colors = dominant_colors(context.absolute(front.path))
        if colors:
            warnings.append("Dominant colours were derived from the reference image.")

    garment = GarmentAsset(
        id=identifier,
        version=options.version,
        product_id=options.product_id,
        source_id=options.source_id,
        brand=options.brand,
        product_name=options.product_name,
        source_url=options.source_url,
        affiliate_url=options.affiliate_url,
        affiliate_network=options.affiliate_network,
        affiliate_tag=options.affiliate_tag,
        price_text=options.price_text,
        currency=options.currency,
        usage_rights=UsageRights(
            license=options.license,
            rights_holder=options.rights_holder,
            commercial_use_allowed=options.commercial_use_allowed,
            acquired_from=options.acquired_from,
        ),
        images=records,
        category=options.category,
        sleeve_length=options.sleeve_length,
        garment_length=options.garment_length,
        body_coverage=options.body_coverage,
        silhouette=options.silhouette,
        material=options.material,
        transparency=options.transparency,
        reflectivity=options.reflectivity,
        fabric_flow=options.fabric_flow,
        dominant_colors=colors,
        pattern_description=options.pattern_description,
        requires_underlayer=options.requires_underlayer,
        structured_notes=options.structured_notes,
        status=IngestionStatus.IMAGES_IMPORTED,
    )

    if garment.metadata_completeness < 0.6:
        garment = garment.model_copy(update={"status": IngestionStatus.METADATA_INCOMPLETE})
        warnings.append(
            "Metadata is incomplete; the compatibility check will report "
            "NEEDS_INPUT until product information is filled in."
        )
    else:
        garment = garment.model_copy(update={"status": IngestionStatus.READY})

    saved = context.repos.garments.save(garment, allow_update=False)
    context.repos.audit.record(
        "garment_ingested",
        entity_type="garment",
        entity_id=saved.id,
        details={
            "version": saved.version,
            "views": sorted(v.value for v in saved.available_views),
            "category": saved.category.value,
        },
    )
    log_event(
        logger,
        "garment_ingested",
        garment_id=saved.id,
        version=saved.version,
        images=len(saved.images),
    )
    return GarmentIngestResult(garment=saved, warnings=warnings)


def garment_image_paths(context: ServiceContext, garment: GarmentAsset) -> list[Path]:
    """Absolute, traversal-checked paths to a garment's reference images."""
    return [context.absolute(image.path) for image in garment.images]


__all__ = [
    "SUPPORTED_SUFFIXES",
    "GarmentIngestOptions",
    "GarmentIngestResult",
    "ImageSpec",
    "dominant_colors",
    "garment_image_paths",
    "ingest_garment",
    "measure_image",
]
