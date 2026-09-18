"""Animator registry: name -> factory."""

from __future__ import annotations

from collections.abc import Callable

from app.backends.animator.base import CharacterAnimatorBackend
from app.core.config import AppConfig
from app.core.errors import ValidationError

AnimatorFactory = Callable[[AppConfig], CharacterAnimatorBackend]

_REGISTRY: dict[str, AnimatorFactory] = {}
_bootstrapped = False


def register_animator(name: str, factory: AnimatorFactory, *, replace: bool = False) -> None:
    key = name.strip().lower()
    if not key:
        raise ValidationError("Animator name must not be empty")
    if key in _REGISTRY and not replace:
        raise ValidationError("Animator already registered", name=key)
    _REGISTRY[key] = factory


def _ensure_bootstrapped() -> None:
    """Register built-ins on first use (module-level would be an import cycle)."""
    global _bootstrapped
    if _bootstrapped:
        return
    _bootstrapped = True
    from app.backends.animator.comfyui import ComfyUIAnimatorBackend
    from app.backends.animator.mock import MockAnimatorBackend

    register_animator("mock", lambda config: MockAnimatorBackend(config), replace=True)
    register_animator("comfyui", lambda config: ComfyUIAnimatorBackend(config), replace=True)


def available_animators() -> list[str]:
    _ensure_bootstrapped()
    return sorted(_REGISTRY)


def create_animator(name: str, config: AppConfig) -> CharacterAnimatorBackend:
    _ensure_bootstrapped()
    key = name.strip().lower()
    factory = _REGISTRY.get(key)
    if factory is None:
        raise ValidationError(
            "Unknown character animator", name=name, available=available_animators()
        )
    return factory(config)


__all__ = ["AnimatorFactory", "available_animators", "create_animator", "register_animator"]
