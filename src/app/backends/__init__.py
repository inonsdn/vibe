"""Renderer backends.

A backend is the *only* component allowed to synthesise new pixels, and it never
writes an output frame directly: it returns a full-frame image, and the pipeline
composites it through the effective mask. A misbehaving backend can therefore
produce a bad garment, but it cannot alter the performer's face.
"""

from app.backends.base import (
    BackendCapabilities,
    FrameRequest,
    FrameResult,
    HealthStatus,
    RenderContext,
    RendererBackend,
    WindowRequest,
    WindowResult,
)
from app.backends.registry import (
    BackendFactory,
    available_backends,
    create_backend,
    register_backend,
)

__all__ = [
    "BackendCapabilities",
    "BackendFactory",
    "FrameRequest",
    "FrameResult",
    "HealthStatus",
    "RenderContext",
    "RendererBackend",
    "WindowRequest",
    "WindowResult",
    "available_backends",
    "create_backend",
    "register_backend",
]
