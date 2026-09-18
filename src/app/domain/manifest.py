"""The render manifest: the reproducibility contract of a finished job.

A manifest is sufficient, on its own, to answer "what exactly produced this
file?" — code version, config hash, workflow hash, backend version, seeds,
dependency versions, the literal ffmpeg command, every input hash and every
output hash.

:meth:`RenderManifest.reproducibility_digest` folds the reproducibility-critical
subset into a single digest, which is what the deterministic-manifest QC check
compares between two runs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from app.core.hashing import sha256_json
from app.domain.base import DomainModel, Identifier
from app.domain.compatibility import CompatibilityState
from app.domain.human_template import FrameRange
from app.domain.render_job import RenderSettings
from app.version import MANIFEST_SCHEMA_VERSION


class FrameChecksum(DomainModel):
    index: int = Field(ge=0)
    sha256: str
    source: str = Field(default="rendered", pattern=r"^(rendered|intro_cache|flash)$")


class ReproducibilityBlock(DomainModel):
    """Everything needed to re-run a job byte-identically."""

    app_version: str
    preprocessing_version: str
    git: dict[str, Any] = Field(default_factory=dict)
    platform: dict[str, Any] = Field(default_factory=dict)
    dependencies: dict[str, str] = Field(default_factory=dict)
    ffmpeg_version: str | None = None
    ffprobe_version: str | None = None
    config_hash: str
    rules_file_sha256: str | None = None
    workflow_id: str | None = None
    workflow_sha256: str | None = None
    backend_name: str
    backend_version: str
    seed: int
    seed_strategy: str
    per_frame_seeds: dict[str, int] = Field(default_factory=dict)
    ffmpeg_commands: list[list[str]] = Field(default_factory=list)


class QCSummary(DomainModel):
    passed: bool
    checks_total: int = Field(ge=0)
    checks_failed: int = Field(ge=0)
    failed_check_ids: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class RenderManifest(DomainModel):
    """Sidecar JSON written next to every rendered output."""

    schema_version: str = MANIFEST_SCHEMA_VERSION
    job_id: Identifier
    created_at: datetime

    template_id: Identifier
    template_version: int = Field(ge=1)
    template_source_sha256: str
    garment_id: Identifier
    garment_version: int = Field(ge=1)

    compatibility_report_id: Identifier | None = None
    compatibility_state: CompatibilityState | None = None
    compatibility_overridden: bool = False

    frame_range: FrameRange
    transition_anchor_frame: int = Field(ge=0)
    intro_frame_count: int = Field(ge=0)
    reveal_frame_count: int = Field(ge=0)
    flash_frames: int = Field(default=0, ge=0)

    video_width: int = Field(gt=0)
    video_height: int = Field(gt=0)
    video_fps: float = Field(gt=0)
    pixel_format: str
    video_codec: str

    prompt: str = ""
    negative_prompt: str = ""
    settings: RenderSettings

    input_hashes: dict[str, str] = Field(default_factory=dict)
    output_hashes: dict[str, str] = Field(default_factory=dict)
    frame_checksums: list[FrameChecksum] = Field(default_factory=list)

    reproducibility: ReproducibilityBlock
    qc: QCSummary | None = None
    notes: str | None = None

    def reproducibility_digest(self) -> str:
        """Digest over the fields that must match between two identical runs.

        Deliberately excludes wall-clock timestamps, machine details and
        output-file hashes of *compressed* artefacts (encoders are not always
        bit-identical across builds); frame checksums of the lossless
        intermediates are included, because those must match exactly.
        """
        payload = {
            "schema_version": self.schema_version,
            "template": [self.template_id, self.template_version, self.template_source_sha256],
            "garment": [self.garment_id, self.garment_version],
            "frame_range": [self.frame_range.start, self.frame_range.end],
            "anchor": self.transition_anchor_frame,
            "flash_frames": self.flash_frames,
            "video": [
                self.video_width,
                self.video_height,
                self.video_fps,
                self.pixel_format,
                self.video_codec,
            ],
            "prompt": [self.prompt, self.negative_prompt],
            "settings": self.settings.model_dump(mode="json"),
            "input_hashes": dict(sorted(self.input_hashes.items())),
            "backend": [
                self.reproducibility.backend_name,
                self.reproducibility.backend_version,
            ],
            "workflow": [
                self.reproducibility.workflow_id,
                self.reproducibility.workflow_sha256,
            ],
            "seed": [self.reproducibility.seed, self.reproducibility.seed_strategy],
            "per_frame_seeds": dict(sorted(self.reproducibility.per_frame_seeds.items())),
            "config_hash": self.reproducibility.config_hash,
            "frame_checksums": [
                [c.index, c.sha256, c.source]
                for c in sorted(self.frame_checksums, key=lambda c: c.index)
            ],
        }
        return sha256_json(payload)

    def required_fields_present(self) -> list[str]:
        """Return the names of reproducibility fields that are still missing."""
        missing: list[str] = []
        if not self.input_hashes:
            missing.append("input_hashes")
        if not self.output_hashes:
            missing.append("output_hashes")
        if not self.reproducibility.dependencies:
            missing.append("reproducibility.dependencies")
        if not self.reproducibility.config_hash:
            missing.append("reproducibility.config_hash")
        if not self.reproducibility.ffmpeg_commands:
            missing.append("reproducibility.ffmpeg_commands")
        if not self.reproducibility.per_frame_seeds:
            missing.append("reproducibility.per_frame_seeds")
        if not self.frame_checksums:
            missing.append("frame_checksums")
        return missing


__all__ = ["FrameChecksum", "QCSummary", "RenderManifest", "ReproducibilityBlock"]
