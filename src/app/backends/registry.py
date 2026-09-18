"""Backend registry: name -> factory.

Backends are constructed from config, so the CLI and API both just pass
``--backend mock`` / ``--backend comfyui`` and get a configured instance.
"""

from __future__ import annotations

from collections.abc import Callable

from app.backends.base import RendererBackend
from app.core.config import AppConfig
from app.core.errors import ValidationError

BackendFactory = Callable[[AppConfig], RendererBackend]

_REGISTRY: dict[str, BackendFactory] = {}


def register_backend(name: str, factory: BackendFactory, *, replace: bool = False) -> None:
    key = name.strip().lower()
    if not key:
        raise ValidationError("Backend name must not be empty")
    if key in _REGISTRY and not replace:
        raise ValidationError("Backend already registered", name=key)
    _REGISTRY[key] = factory


def available_backends() -> list[str]:
    _ensure_bootstrapped()
    return sorted(_REGISTRY)


def create_backend(name: str, config: AppConfig) -> RendererBackend:
    _ensure_bootstrapped()
    key = name.strip().lower()
    factory = _REGISTRY.get(key)
    if factory is None:
        raise ValidationError(
            "Unknown backend",
            name=name,
            available=available_backends(),
        )
    return factory(config)


_bootstrapped = False


def _ensure_bootstrapped() -> None:
    """Register the built-in backends on first use.

    Done lazily rather than at import time because the backend modules import
    this one; a module-level call would be a cycle.
    """
    global _bootstrapped
    if _bootstrapped:
        return
    _bootstrapped = True
    from app.backends.comfyui.backend import ComfyUIBackend
    from app.backends.mock.backend import MockBackend

    register_backend("mock", lambda config: MockBackend(config), replace=True)
    register_backend("comfyui", lambda config: ComfyUIBackend(config), replace=True)


__all__ = [
    "BackendFactory",
    "available_backends",
    "create_backend",
    "register_backend",
]
