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

**Context honesty.** :class:`ContextMode` states what a backend actually
consumes, and the pipeline gathers exactly that much. Reporting "16 context
frames" while handing a workflow a single image is the specific failure this
enum exists to prevent: the number in the manifest has to be the number the
model saw.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from app.backends.base import HealthStatus
from app.core.config import AppConfig
from app.domain.master import HeroCharacter, MasterCandidate
from app.domain.motion import CanonicalSkeletonProfile, MotionComposition
from app.motion.pose_format import PoseFrame


class ContextMode(StrEnum):
    """How much continuity context a backend consumes between chunks."""

    #: Conditions on nothing. Chunk boundaries are unconditioned, and the
    #: pipeline does not waste IO gathering frames the backend will ignore.
    NONE = "none"
    #: Consumes the single frame immediately preceding the chunk.
    LAST_FRAME = "last_frame"
    #: Consumes the whole configured tail, in order.
    SEQUENCE = "sequence"

    @property
    def uses_context(self) -> bool:
        return self is not ContextMode.NONE


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

    ``poses`` covers ``[start_frame, end_frame)``, in order, with no gaps.
    ``context_frames`` and ``context_poses`` describe already-accepted frames
    immediately before it — the continuity signal — and are sized by the
    backend's :class:`ContextMode`.
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

    @property
    def context_frame_indices(self) -> list[int]:
        """Output indices of the context frames, immediately preceding the chunk."""
        return list(range(self.start_frame - len(self.context_frames), self.start_frame))

    def validate(self) -> None:
        """Structural checks a backend may rely on. Raises on violation."""
        from app.core.errors import ValidationError

        if self.end_frame <= self.start_frame:
            raise ValidationError(
                "Chunk end_frame must exceed start_frame",
                start_frame=self.start_frame,
                end_frame=self.end_frame,
            )
        indices = [pose.frame_index for pose in self.poses]
        if indices != self.frame_indices:
            raise ValidationError(
                "Chunk pose sequence does not cover the requested frames exactly",
                expected_first=self.start_frame,
                expected_last=self.end_frame - 1,
                expected_count=self.frame_count,
                got_count=len(indices),
                got_first=indices[0] if indices else None,
                got_last=indices[-1] if indices else None,
            )
        if self.context_poses and len(self.context_poses) != len(self.context_frames):
            raise ValidationError(
                "Context frames and context poses must come in matching counts",
                frames=len(self.context_frames),
                poses=len(self.context_poses),
            )
        context_indices = [pose.frame_index for pose in self.context_poses]
        if context_indices and context_indices != self.context_frame_indices:
            raise ValidationError(
                "Context poses are not the frames immediately preceding the chunk",
                expected=self.context_frame_indices,
                got=context_indices,
            )


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
    context_mode: ContextMode
    max_chunk_frames: int
    recommended_chunk_frames: int
    recommended_overlap_frames: int
    expected_vram_mb: int | None = None
    supports_low_vram: bool = True
    produces_photoreal: bool = False
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def supports_context_frames(self) -> bool:
        """Back-compatible view of :attr:`context_mode`."""
        return self.context_mode.uses_context

    def context_frames_for(self, configured_overlap: int) -> int:
        """How many context frames this backend will actually consume."""
        if self.context_mode is ContextMode.NONE:
            return 0
        if self.context_mode is ContextMode.LAST_FRAME:
            return min(1, configured_overlap)
        return configured_overlap

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "requires_gpu": self.requires_gpu,
            "requires_model_weights": self.requires_model_weights,
            "deterministic": self.deterministic,
            "context_mode": self.context_mode.value,
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
    "ContextMode",
]
