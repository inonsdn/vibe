"""Render job: an immutable request plus mutable execution state.

The fields that define *what* gets rendered (template version, garment version,
frame range, seeds, settings, workflow hash, input hashes) never change after
creation. Only ``status``, ``progress``, ``checkpoint``, ``artifacts``,
``qc_metrics``, ``error`` and the timestamps move.

This split is what makes resume safe: a resumed job re-derives every frame seed
from the same immutable inputs, so completed frames can simply be skipped.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Self

from pydantic import Field, model_validator

from app.domain.base import DomainModel, Identifier, TimestampedModel
from app.domain.enums import JobStatus
from app.domain.human_template import FrameRange


class RenderSettings(DomainModel):
    """Everything a backend may consult, recorded verbatim in the manifest."""

    frame_window: int = Field(default=8, ge=1, le=256)
    feather_radius_px: int = Field(default=9, ge=0, le=128)
    expansion_dilate_px: int = Field(default=0, ge=0, le=256)
    protected_dilate_px: int = Field(default=2, ge=0, le=256)
    protected_wins: bool = True
    denoise_strength: float = Field(default=0.75, ge=0.0, le=1.0)
    guidance_scale: float = Field(default=5.0, ge=0.0, le=50.0)
    steps: int = Field(default=20, ge=1, le=200)
    #: Tiling / offload knobs for 8GB-class GPUs.
    tile_size_px: int | None = Field(default=None, ge=64, le=4096)
    tile_overlap_px: int = Field(default=32, ge=0, le=512)
    low_vram_mode: bool = True
    cpu_offload: bool = True
    quantization: str | None = Field(default=None, max_length=32)
    extra: dict[str, Any] = Field(default_factory=dict)


class JobProgress(DomainModel):
    total_frames: int = Field(ge=0)
    completed_frames: int = Field(default=0, ge=0)
    failed_frames: int = Field(default=0, ge=0)
    current_frame: int | None = None
    started_at: datetime | None = None
    last_update_at: datetime | None = None

    @model_validator(mode="after")
    def _bounds(self) -> Self:
        if self.completed_frames > self.total_frames:
            raise ValueError("completed_frames cannot exceed total_frames")
        return self

    @property
    def fraction(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return self.completed_frames / self.total_frames

    @property
    def percent(self) -> float:
        return round(self.fraction * 100, 2)


class Checkpoint(DomainModel):
    """Resumable state written after each completed window."""

    #: Frame indices already rendered *and* composited on disk.
    completed_frames: list[int] = Field(default_factory=list)
    last_completed_frame: int | None = None
    next_frame: int | None = None
    window_index: int = Field(default=0, ge=0)
    backend_state: dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime | None = None

    def is_done(self, frame_index: int) -> bool:
        return frame_index in set(self.completed_frames)

    def remaining(self, frame_range: FrameRange) -> list[int]:
        done = set(self.completed_frames)
        return [index for index in frame_range.indices() if index not in done]


class JobArtifacts(DomainModel):
    """Data-root-relative paths produced by the job."""

    root: str
    raw_frames_dir: str
    composited_frames_dir: str
    effective_masks_dir: str
    intro_segment_path: str | None = None
    reveal_segment_path: str | None = None
    final_video_path: str | None = None
    preview_video_path: str | None = None
    manifest_path: str | None = None
    qc_report_path: str | None = None
    qc_report_text_path: str | None = None
    contact_sheet_paths: list[str] = Field(default_factory=list)
    log_path: str | None = None

    @classmethod
    def standard(cls, root: str) -> JobArtifacts:
        base = root.rstrip("/")
        return cls(
            root=base,
            raw_frames_dir=f"{base}/raw_frames",
            composited_frames_dir=f"{base}/composited_frames",
            effective_masks_dir=f"{base}/effective_masks",
            manifest_path=f"{base}/manifest.json",
            qc_report_path=f"{base}/qc_report.json",
            qc_report_text_path=f"{base}/qc_report.txt",
            log_path=f"{base}/job.jsonl",
        )


class JobError(DomainModel):
    code: str
    message: str
    frame_index: int | None = None
    stage: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    occurred_at: datetime
    retryable: bool = False


class RenderJob(TimestampedModel):
    """One outfit render against one master performance."""

    id: Identifier

    # -- immutable inputs -------------------------------------------------
    template_id: Identifier
    template_version: int = Field(ge=1)
    garment_id: Identifier
    garment_version: int = Field(ge=1)
    compatibility_report_id: Identifier | None = None

    frame_range: FrameRange
    transition_anchor_frame: int = Field(ge=0)

    backend_name: str = Field(min_length=1, max_length=64)
    backend_version: str = Field(default="unknown", max_length=64)
    workflow_id: str | None = None
    workflow_sha256: str | None = None

    seed: int = Field(ge=0, le=2**63 - 1)
    seed_strategy: str = Field(default="derived", max_length=32)
    settings: RenderSettings = Field(default_factory=RenderSettings)
    prompt: str = ""
    negative_prompt: str = ""

    input_hashes: dict[str, str] = Field(default_factory=dict)
    config_hash: str | None = None

    # -- mutable execution state -----------------------------------------
    status: JobStatus = JobStatus.CREATED
    progress: JobProgress
    checkpoint: Checkpoint = Field(default_factory=Checkpoint)
    artifacts: JobArtifacts
    qc_metrics: dict[str, Any] = Field(default_factory=dict)
    error: JobError | None = None

    started_at: datetime | None = None
    finished_at: datetime | None = None

    @model_validator(mode="after")
    def _anchor_inside_or_at_range_start(self) -> Self:
        """The reveal range must start exactly at the transition anchor."""
        if self.frame_range.start != self.transition_anchor_frame:
            raise ValueError(
                "frame_range.start must equal transition_anchor_frame "
                f"({self.frame_range.start} != {self.transition_anchor_frame})"
            )
        if self.progress.total_frames not in (0, self.frame_range.count):
            raise ValueError(
                "progress.total_frames must match frame_range.count "
                f"({self.progress.total_frames} != {self.frame_range.count})"
            )
        return self

    # -- convenience ------------------------------------------------------
    @property
    def template_key(self) -> str:
        return f"{self.template_id}@v{self.template_version}"

    @property
    def garment_key(self) -> str:
        return f"{self.garment_id}@v{self.garment_version}"

    def remaining_frames(self) -> list[int]:
        return self.checkpoint.remaining(self.frame_range)

    def is_complete(self) -> bool:
        return not self.remaining_frames()


__all__ = [
    "Checkpoint",
    "JobArtifacts",
    "JobError",
    "JobProgress",
    "RenderJob",
    "RenderSettings",
]
