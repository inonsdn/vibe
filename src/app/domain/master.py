"""Master origin and the synthetic-master acceptance gate.

A Master Human Performance can now arise two ways:

``captured_master``
    An authorized performer video already exists. This is the original path and
    is unchanged.
``synthetic_master``
    An original Hero Character is animated from a :class:`MotionComposition`.

The important asymmetry: a synthetic master is **not** production-ready when the
animator finishes. It is a *candidate* until an operator reviews QC and
explicitly accepts it. Only acceptance freezes it into the immutable
``HumanTemplate`` the garment pipeline consumes.

That gate exists because the candidate is the one artifact in the system a model
invented wholesale. Everything downstream treats the master as ground truth, so
a human says "yes, this is our character, performing correctly" before it earns
that status.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import Field, model_validator

from app.domain.base import DomainModel, Identifier, Sha256, TimestampedModel


class MasterOrigin(StrEnum):
    """Where a Master Human Performance came from."""

    CAPTURED = "captured_master"
    SYNTHETIC = "synthetic_master"


class MasterCandidateStatus(StrEnum):
    CREATED = "created"
    PREPARING = "preparing"
    ANIMATING = "animating"
    PAUSED = "paused"
    ANIMATED = "animated"
    QC_RUNNING = "qc_running"
    AWAITING_ACCEPTANCE = "awaiting_acceptance"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in {
            MasterCandidateStatus.ACCEPTED,
            MasterCandidateStatus.REJECTED,
            MasterCandidateStatus.FAILED,
        }

    @property
    def is_resumable(self) -> bool:
        return self in {
            MasterCandidateStatus.CREATED,
            MasterCandidateStatus.PREPARING,
            MasterCandidateStatus.ANIMATING,
            MasterCandidateStatus.PAUSED,
            MasterCandidateStatus.FAILED,
        }


class HeroCharacter(TimestampedModel):
    """The original character a synthetic master depicts.

    Carries the same consent discipline as a captured performer: a Hero
    Character is either fully synthetic, or derived from a consenting adult.
    """

    id: Identifier
    version: int = Field(default=1, ge=1)
    display_name: str = Field(min_length=1, max_length=200)

    reference_images: list[str] = Field(default_factory=list)
    reference_hashes: dict[str, Sha256] = Field(default_factory=dict)
    identity_notes: str | None = None

    subject_kind: str = Field(default="synthetic", pattern=r"^(synthetic|consented_human)$")
    adult_confirmed: bool = True
    consent_document_ref: str | None = None
    rights_holder: str | None = None
    license: str | None = None

    @model_validator(mode="after")
    def _consent_discipline(self) -> Self:
        if not self.adult_confirmed:
            raise ValueError("adult_confirmed must be true for every Hero Character")
        if self.subject_kind == "consented_human" and not self.consent_document_ref:
            raise ValueError("consented_human Hero Characters require consent_document_ref")
        if not self.reference_images:
            raise ValueError("a Hero Character needs at least one reference image")
        return self

    def version_key(self) -> str:
        return f"{self.id}@v{self.version}"


class ChunkRecord(DomainModel):
    """One animated chunk, with the overlap that tied it to its predecessor."""

    index: int = Field(ge=0)
    start_frame: int = Field(ge=0)
    end_frame: int = Field(gt=0)
    context_frames: list[int] = Field(default_factory=list)
    overlap_frames: int = Field(default=0, ge=0)
    #: What the backend declared it consumes: ``none``, ``last_frame`` or
    #: ``sequence``. Recorded per chunk so QC can tell "this backend conditions
    #: on nothing by design" apart from "this chunk lost its context".
    context_mode: str = Field(default="sequence", pattern=r"^(none|last_frame|sequence)$")
    seed: int = Field(ge=0)
    frame_hashes: dict[str, str] = Field(default_factory=dict)
    duration_ms: int | None = None

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end_frame <= self.start_frame:
            raise ValueError("chunk end_frame must exceed start_frame")
        return self

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame


class MasterAcceptance(DomainModel):
    """The operator's explicit decision to promote a candidate."""

    accepted_by: str = Field(min_length=1, max_length=200)
    accepted_at: datetime
    reason: str = Field(min_length=10, max_length=2000)
    qc_report_id: str | None = None
    qc_passed: bool = False
    acknowledged_warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _tz_aware(self) -> Self:
        if self.accepted_at.tzinfo is None:
            raise ValueError("accepted_at must be timezone-aware (UTC)")
        return self


class MasterCandidate(TimestampedModel):
    """A candidate Master Human Performance awaiting operator acceptance."""

    id: Identifier
    version: int = Field(default=1, ge=1)
    display_name: str = Field(min_length=1, max_length=200)
    origin: MasterOrigin = MasterOrigin.SYNTHETIC

    composition_id: Identifier
    composition_version: int = Field(ge=1)
    hero_character_id: Identifier
    hero_character_version: int = Field(ge=1)

    backend_name: str = Field(min_length=1, max_length=64)
    backend_version: str = Field(default="unknown", max_length=64)
    workflow_id: str | None = None
    workflow_sha256: str | None = None

    seed: int = Field(ge=0, le=2**63 - 1)
    settings: dict[str, Any] = Field(default_factory=dict)

    frame_count: int = Field(default=0, ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0, le=240)

    frames_dir: str
    preview_path: str | None = None
    video_path: str | None = None
    manifest_path: str | None = None
    qc_report_path: str | None = None

    chunks: list[ChunkRecord] = Field(default_factory=list)
    input_hashes: dict[str, str] = Field(default_factory=dict)
    output_hashes: dict[str, str] = Field(default_factory=dict)
    qc_metrics: dict[str, Any] = Field(default_factory=dict)

    status: MasterCandidateStatus = MasterCandidateStatus.CREATED
    acceptance: MasterAcceptance | None = None
    #: Set once accepted and promoted into an immutable HumanTemplate.
    promoted_template_id: Identifier | None = None
    error: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _acceptance_consistency(self) -> Self:
        if self.status is MasterCandidateStatus.ACCEPTED and self.acceptance is None:
            raise ValueError("an accepted master must carry its acceptance record")
        if self.acceptance is not None and self.status not in {
            MasterCandidateStatus.ACCEPTED,
            MasterCandidateStatus.REJECTED,
        }:
            raise ValueError(f"an acceptance record is meaningless in status {self.status.value}")
        # Promotion is downstream of acceptance and cannot be undone by
        # rewinding the status: a HumanTemplate now exists that points back at
        # this candidate's frames.
        if self.promoted_template_id is not None and self.status is not (
            MasterCandidateStatus.ACCEPTED
        ):
            raise ValueError(
                "a promoted candidate must stay accepted "
                f"(status={self.status.value}, template={self.promoted_template_id})"
            )
        return self

    @property
    def is_promoted(self) -> bool:
        return self.promoted_template_id is not None

    @property
    def is_accepted(self) -> bool:
        return self.status is MasterCandidateStatus.ACCEPTED and self.acceptance is not None

    def completed_chunk_frames(self) -> set[int]:
        frames: set[int] = set()
        for chunk in self.chunks:
            frames.update(range(chunk.start_frame, chunk.end_frame))
        return frames

    def remaining_frames(self) -> list[int]:
        done = self.completed_chunk_frames()
        return [index for index in range(self.frame_count) if index not in done]

    def version_key(self) -> str:
        return f"{self.id}@v{self.version}"


__all__ = [
    "ChunkRecord",
    "HeroCharacter",
    "MasterAcceptance",
    "MasterCandidate",
    "MasterCandidateStatus",
    "MasterOrigin",
]
