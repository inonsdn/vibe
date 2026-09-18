"""API dependencies: one shared ServiceContext per process."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from app.core.config import AppConfig, load_config
from app.pipeline.context import ServiceContext

_state: dict[str, Any] = {}


def configure(config: AppConfig | None = None, **create_kwargs: Any) -> ServiceContext:
    """Build (or rebuild) the process-wide service context."""
    close()
    context = ServiceContext.create(config=config or load_config(), **create_kwargs)
    _state["context"] = context
    return context


def get_context() -> ServiceContext:
    context = _state.get("context")
    if context is None:
        context = configure()
    return context


def context_dependency() -> Iterator[ServiceContext]:
    yield get_context()


def close() -> None:
    context = _state.pop("context", None)
    if context is not None:
        context.close()


__all__ = ["close", "configure", "context_dependency", "get_context"]
