"""The model-agnostic character animator interface.

Contract, in call order — deliberately parallel to ``RendererBackend`` so the
two are learned once:

``healthcheck()`` / ``capabilities()``
    Cheap probes. Neither may download anything.
``prepare(context)``
    Per-candidate setup. Returns a dict recorded in the manifest.
``animate_chunk(context, request)``
    Render one contiguous chunk of frames from pose control data, conditioned on
    previously accepted frames where the backend supports it.
``resume(context)``
    Re-attach to a partially animated candidate.
``collect_artifacts(context)``
    Backend-side artefacts worth recording.

**Chunk continuity.** An 8GB card cannot hold a long sequence, so output is
produced in chunks. Chunks are never hard-concatenated: each request carries
``context_frames`` — already-accepted frames immediately preceding it — and the
backend is expected to condition on them. The pipeline then keeps only the
chunk's *new* frames, so the overlap region is generated once and reused, not
generated twice and crossfaded.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.backends.base import HealthStatus
from app.core.config import AppConfig
from app.domain.master import HeroCharacter, MasterCandidate
from app.domain.motion import CanonicalSkeletonProfile, MotionComposition
from app.motion.pose_format import PoseFrame


@dataclass
class AnimatorContext:
    """Everything a backend may read about a candidate. Treated as read-only."""

    candidate: MasterCandidate
    composition: MotionComposition
    hero: HeroCharacter
    profile: CanonicalSkeletonProfile
    config: AppConfig
    candidate_dir: Path
    frames_dir: Path
    pose_dir: Path
    hero_image_paths: list[Path] = field(default_factory=list)
    workflow_path: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def frame_shape(self) -> tuple[int, int]:
        return (self.candidate.height, self.candidate.width)


@dataclass
class AnimationChunkRequest:
    """One contiguous chunk of frames to animate.

    ``poses`` covers ``[start_frame, end_frame)``. ``context_frames`` and
    ``context_poses`` describe already-accepted frames immediately before it —
    the continuity signal. A backend that cannot condition on them must say so
    via ``supports_context_frames``; the pipeline then records that the chunk
    boundary is unconditioned, rather than pretending otherwise.
    """

    chunk_index: int
    start_frame: int
    end_frame: int
    poses: list[PoseFrame]
    seed: int
    context_frames: list[np.ndarray] = field(default_factory=list)
    context_poses: list[PoseFrame] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def frame_indices(self) -> list[int]:
        return list(range(self.start_frame, self.end_frame))


@dataclass
class AnimationChunkResult:
    """Frames produced for a chunk, keyed by absolute output frame index."""

    chunk_index: int
    frames: dict[int, np.ndarray]
    seed: int
    backend_metadata: dict[str, Any] = field(default_factory=dict)
    duration_ms: int | None = None


@dataclass
class AnimatorCapabilities:
    name: str
    version: str
    requires_gpu: bool
    requires_model_weights: bool
    deterministic: bool
    supports_context_frames: bool
    max_chunk_frames: int
    recommended_chunk_frames: int
    recommended_overlap_frames: int
    expected_vram_mb: int | None = None
    supports_low_vram: bool = True
    produces_photoreal: bool = False
    notes: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "requires_gpu": self.requires_gpu,
            "requires_model_weights": self.requires_model_weights,
            "deterministic": self.deterministic,
            "supports_context_frames": self.supports_context_frames,
            "max_chunk_frames": self.max_chunk_frames,
            "recommended_chunk_frames": self.recommended_chunk_frames,
            "recommended_overlap_frames": self.recommended_overlap_frames,
            "expected_vram_mb": self.expected_vram_mb,
            "supports_low_vram": self.supports_low_vram,
            "produces_photoreal": self.produces_photoreal,
            "notes": self.notes,
        }


class CharacterAnimatorBackend(abc.ABC):
    """Base class every character animator implements."""

    name: str = "unnamed"
    version: str = "0.0.0"

    @abc.abstractmethod
    def healthcheck(self) -> HealthStatus: ...

    @abc.abstractmethod
    def capabilities(self) -> AnimatorCapabilities: ...

    @abc.abstractmethod
    def prepare(self, context: AnimatorContext) -> dict[str, Any]: ...

    @abc.abstractmethod
    def animate_chunk(
        self, context: AnimatorContext, request: AnimationChunkRequest
    ) -> AnimationChunkResult: ...

    def resume(self, context: AnimatorContext) -> dict[str, Any]:
        """Re-attach to a partially animated candidate. Idempotent by default."""
        return {"resumed": True, "backend": self.name}

    def collect_artifacts(self, context: AnimatorContext) -> dict[str, Any]:
        return {}

    def close(self) -> None:
        return None

    def __enter__(self) -> CharacterAnimatorBackend:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = [
    "AnimationChunkRequest",
    "AnimationChunkResult",
    "AnimatorCapabilities",
    "AnimatorContext",
    "CharacterAnimatorBackend",
]
