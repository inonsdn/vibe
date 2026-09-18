"""The model-agnostic renderer interface.

Contract, in the order the pipeline calls it:

``healthcheck()``
    Cheap liveness/readiness probe. Must not download anything.
``capabilities()``
    Static self-description: does it need a GPU, can it do windows, is it
    deterministic, what VRAM does it expect.
``prepare(context)``
    Per-job setup: validate inputs, stage files, resolve a workflow. Returns a
    dict recorded in the manifest.
``render_frame(request)`` / ``render_window(request)``
    Produce **full-frame** images. The returned image is composited through the
    effective mask by the pipeline; the backend does not own the output pixels.
``resume(context)``
    Re-attach to a partially completed job. Must be safe to call when nothing
    was in flight.
``collect_artifacts(context)``
    Report any backend-side artefacts worth recording (logs, prompt ids).

Determinism requirement: for a fixed ``FrameRequest`` (same seed, same inputs,
same settings), ``render_frame`` must return identical pixels. The pipeline's
resume logic and the deterministic-output test both depend on it.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import AppConfig
from app.domain.garment import GarmentAsset
from app.domain.human_template import HumanTemplate
from app.domain.render_job import RenderJob


@dataclass
class RenderContext:
    """Everything a backend may read about a job. Treated as read-only."""

    job: RenderJob
    template: HumanTemplate
    garment: GarmentAsset
    config: AppConfig
    #: Absolute, already-validated directories.
    source_frames_dir: Path
    job_dir: Path
    raw_frames_dir: Path
    garment_image_paths: list[Path] = field(default_factory=list)
    workflow_path: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def frame_shape(self) -> tuple[int, int]:
        return (self.template.video.height, self.template.video.width)


@dataclass
class FrameRequest:
    """One frame to render.

    ``source`` and ``effective_mask`` are provided so a backend can condition on
    the real pixels and know which region it is allowed to influence. A backend
    must still return a full frame; the pipeline enforces the mask.
    """

    frame_index: int
    source: np.ndarray
    effective_mask: np.ndarray
    seed: int
    prompt: str = ""
    negative_prompt: str = ""
    settings: dict[str, Any] = field(default_factory=dict)
    #: Previously rendered neighbour frame, when the backend supports temporal
    #: conditioning. ``None`` for the first frame of a window.
    previous_render: np.ndarray | None = None
    control: dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameResult:
    frame_index: int
    image: np.ndarray
    seed: int
    backend_metadata: dict[str, Any] = field(default_factory=dict)
    duration_ms: int | None = None


@dataclass
class WindowRequest:
    """A contiguous window of frames, for backends with temporal models."""

    frames: list[FrameRequest]
    window_index: int
    seed: int

    @property
    def start_frame(self) -> int:
        return self.frames[0].frame_index

    @property
    def end_frame(self) -> int:
        """Exclusive end, matching the half-open convention used everywhere."""
        return self.frames[-1].frame_index + 1


@dataclass
class WindowResult:
    results: list[FrameResult]
    window_index: int
    backend_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthStatus:
    healthy: bool
    detail: str
    version: str = "unknown"
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "healthy": self.healthy,
            "detail": self.detail,
            "version": self.version,
            **self.extra,
        }


@dataclass
class BackendCapabilities:
    name: str
    version: str
    requires_gpu: bool
    requires_model_weights: bool
    deterministic: bool
    supports_windows: bool
    supports_resume: bool
    max_frame_window: int
    expected_vram_mb: int | None = None
    supports_low_vram: bool = True
    supports_tiling: bool = False
    notes: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "requires_gpu": self.requires_gpu,
            "requires_model_weights": self.requires_model_weights,
            "deterministic": self.deterministic,
            "supports_windows": self.supports_windows,
            "supports_resume": self.supports_resume,
            "max_frame_window": self.max_frame_window,
            "expected_vram_mb": self.expected_vram_mb,
            "supports_low_vram": self.supports_low_vram,
            "supports_tiling": self.supports_tiling,
            "notes": self.notes,
        }


class RendererBackend(abc.ABC):
    """Base class every backend implements."""

    name: str = "unnamed"
    version: str = "0.0.0"

    @abc.abstractmethod
    def healthcheck(self) -> HealthStatus: ...

    @abc.abstractmethod
    def capabilities(self) -> BackendCapabilities: ...

    @abc.abstractmethod
    def prepare(self, context: RenderContext) -> dict[str, Any]:
        """Per-job setup; the returned dict is stored in the manifest."""

    @abc.abstractmethod
    def render_frame(self, context: RenderContext, request: FrameRequest) -> FrameResult: ...

    def render_window(self, context: RenderContext, request: WindowRequest) -> WindowResult:
        """Default: render a window frame by frame.

        Backends with genuine temporal models override this. The default keeps
        the pipeline's windowed loop valid for simple per-frame backends.
        """
        results = [self.render_frame(context, frame) for frame in request.frames]
        return WindowResult(results=results, window_index=request.window_index)

    def resume(self, context: RenderContext) -> dict[str, Any]:
        """Re-attach to a partially completed job. Idempotent by default."""
        return {"resumed": True, "backend": self.name}

    def collect_artifacts(self, context: RenderContext) -> dict[str, Any]:
        """Backend-side artefacts worth recording in the manifest."""
        return {}

    def close(self) -> None:
        """Release resources (GPU memory, HTTP sessions). Safe to call twice."""
        return None

    def __enter__(self) -> RendererBackend:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = [
    "BackendCapabilities",
    "FrameRequest",
    "FrameResult",
    "HealthStatus",
    "RenderContext",
    "RendererBackend",
    "WindowRequest",
    "WindowResult",
]
