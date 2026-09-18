"""The Master Human Performance template.

A ``HumanTemplate`` is the immutable source of truth for a character's
performance: one video, one set of extracted frames, one set of masks. Every
outfit render reads pixels from it and never regenerates the human.

Frame ranges are **inclusive of start, exclusive of end** (Python slice
semantics) and are validated against the extracted frame count so that the
transition anchor can never be off by one.
"""

from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import Field, field_validator, model_validator

from app.domain.base import DomainModel, Identifier, Sha256, TimestampedModel
from app.domain.enums import MASK_DIR_NAMES, MaskKind, ProcessingStatus, TemplateClothingClass
from app.version import PREPROCESSING_VERSION


class VideoSpec(DomainModel):
    """Measured properties of the master video (from ffprobe, not assumed)."""

    width: int = Field(gt=0, le=16384)
    height: int = Field(gt=0, le=16384)
    fps: float = Field(gt=0, le=240)
    duration_s: float = Field(gt=0)
    frame_count: int = Field(gt=0)
    codec: str | None = None
    pixel_format: str | None = None
    constant_frame_rate: bool = True
    avg_frame_rate: str | None = None
    r_frame_rate: str | None = None
    has_audio: bool = False
    audio_codec: str | None = None
    audio_duration_s: float | None = None

    @property
    def frame_duration_s(self) -> float:
        return 1.0 / self.fps


class FrameRange(DomainModel):
    """Half-open frame interval ``[start, end)``."""

    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end <= self.start:
            raise ValueError(f"frame range end ({self.end}) must exceed start ({self.start})")
        return self

    @property
    def count(self) -> int:
        return self.end - self.start

    @property
    def last(self) -> int:
        """Last frame index actually inside the range."""
        return self.end - 1

    def contains(self, index: int) -> bool:
        return self.start <= index < self.end

    def indices(self) -> range:
        return range(self.start, self.end)


class ConsentRecord(DomainModel):
    """Consent, provenance and licensing for the depicted character.

    ``subject_kind`` distinguishes a fully synthetic character from a real,
    consenting adult performer; both require an on-file record before any
    render is permitted.
    """

    subject_kind: str = Field(pattern=r"^(synthetic|consented_human)$")
    adult_confirmed: bool
    consent_document_ref: str | None = None
    consent_granted_at: datetime | None = None
    consent_expires_at: datetime | None = None
    rights_holder: str | None = None
    license: str | None = None
    provenance_notes: str | None = None
    usage_restrictions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _human_requires_document(self) -> Self:
        if self.subject_kind == "consented_human" and not self.consent_document_ref:
            raise ValueError("consented_human subjects require consent_document_ref")
        if not self.adult_confirmed:
            raise ValueError("adult_confirmed must be true for every template")
        return self

    def is_expired(self, now: datetime) -> bool:
        return self.consent_expires_at is not None and self.consent_expires_at <= now


class TemplateDirectories(DomainModel):
    """Relative (data-root-anchored) directory layout for one template.

    Paths are stored relative so a data directory can be moved or backed up
    without rewriting the database.
    """

    root: str
    source_frames: str
    masks_garment: str
    masks_expansion: str
    masks_protected: str
    masks_occlusion: str
    pose: str
    depth: str
    optical_flow: str
    face_landmarks: str
    identity_refs: str
    intro_cache: str
    background_plate: str | None = None

    @classmethod
    def standard(cls, root: str) -> TemplateDirectories:
        """Default layout under ``<data_root>/templates/<id>``."""
        base = root.rstrip("/")
        return cls(
            root=base,
            source_frames=f"{base}/source_frames",
            masks_garment=f"{base}/{MASK_DIR_NAMES[MaskKind.GARMENT]}",
            masks_expansion=f"{base}/{MASK_DIR_NAMES[MaskKind.EXPANSION]}",
            masks_protected=f"{base}/{MASK_DIR_NAMES[MaskKind.PROTECTED]}",
            masks_occlusion=f"{base}/{MASK_DIR_NAMES[MaskKind.OCCLUSION]}",
            pose=f"{base}/pose",
            depth=f"{base}/depth",
            optical_flow=f"{base}/optical_flow",
            face_landmarks=f"{base}/face_landmarks",
            identity_refs=f"{base}/identity_refs",
            intro_cache=f"{base}/intro_cache",
            background_plate=f"{base}/background_plate",
        )

    def mask_dir(self, kind: MaskKind) -> str:
        return getattr(self, MASK_DIR_NAMES[kind])

    def all_dirs(self) -> list[str]:
        values = [
            self.root,
            self.source_frames,
            self.masks_garment,
            self.masks_expansion,
            self.masks_protected,
            self.masks_occlusion,
            self.pose,
            self.depth,
            self.optical_flow,
            self.face_landmarks,
            self.identity_refs,
            self.intro_cache,
        ]
        if self.background_plate:
            values.append(self.background_plate)
        return values


class HumanTemplate(TimestampedModel):
    """Immutable master performance record."""

    id: Identifier
    version: int = Field(default=1, ge=1)
    display_name: str = Field(min_length=1, max_length=200)

    # -- source -----------------------------------------------------------
    source_video_path: str
    source_sha256: Sha256
    video: VideoSpec

    # -- segmentation -----------------------------------------------------
    intro: FrameRange
    reveal: FrameRange
    transition_anchor_frame: int = Field(ge=0)

    # -- identity & consent ----------------------------------------------
    identity_reference_images: list[str] = Field(default_factory=list)
    identity_reference_hashes: dict[str, Sha256] = Field(default_factory=dict)
    consent: ConsentRecord
    template_clothing_class: TemplateClothingClass

    # -- derived artefacts ------------------------------------------------
    directories: TemplateDirectories
    background_plate_path: str | None = None
    source_frames_sha256: str | None = None
    extracted_frame_count: int = Field(default=0, ge=0)

    # -- bookkeeping ------------------------------------------------------
    status: ProcessingStatus = ProcessingStatus.CREATED
    preprocessing_version: str = PREPROCESSING_VERSION
    tool_versions: dict[str, str] = Field(default_factory=dict)
    notes: str | None = None

    @field_validator("identity_reference_images")
    @classmethod
    def _unique_refs(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("identity reference image paths must be unique")
        return value

    @model_validator(mode="after")
    def _validate_segmentation(self) -> Self:
        """The transition anchor must be the exact seam between the segments.

        ``intro`` is ``[intro.start, anchor)`` and ``reveal`` is
        ``[anchor, reveal.end)``. Requiring equality here is what makes the
        "no off-by-one at the anchor" guarantee structural rather than
        aspirational: no other combination can be persisted.
        """
        if self.intro.end != self.transition_anchor_frame:
            raise ValueError(
                "transition_anchor_frame must equal intro.end "
                f"(anchor={self.transition_anchor_frame}, intro.end={self.intro.end})"
            )
        if self.reveal.start != self.transition_anchor_frame:
            raise ValueError(
                "reveal.start must equal transition_anchor_frame "
                f"(reveal.start={self.reveal.start}, anchor={self.transition_anchor_frame})"
            )
        if self.video.frame_count and self.reveal.end > self.video.frame_count:
            raise ValueError(
                f"reveal.end ({self.reveal.end}) exceeds video frame_count "
                f"({self.video.frame_count})"
            )
        return self

    # -- convenience ------------------------------------------------------
    @property
    def total_output_frames(self) -> int:
        return self.intro.count + self.reveal.count

    @property
    def is_renderable(self) -> bool:
        return self.status in {ProcessingStatus.VALIDATED, ProcessingStatus.READY}

    def version_key(self) -> str:
        return f"{self.id}@v{self.version}"


__all__ = [
    "ConsentRecord",
    "FrameRange",
    "HumanTemplate",
    "TemplateDirectories",
    "VideoSpec",
]
