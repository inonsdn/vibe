"""Constructing a pose adapter by name, from config plus per-run overrides.

The registry in :mod:`app.adapters.base` holds one adapter per kind and is
populated with honest stubs at import time. That is right for "what can this
installation do?", and wrong for "run this particular extraction with these
model files on CUDA" — which needs construction, not lookup. This is that
construction, kept in one place so the CLI, the API and tests all build
adapters the same way.

``mock`` is deliberately reachable only by naming it explicitly. It is never a
fallback for a misconfigured real adapter: producing synthetic poses because a
model file was missing is precisely the failure mode this system is built to
refuse.
"""

from __future__ import annotations

from typing import Any

from app.core.config import AppConfig
from app.core.errors import ConfigError

#: Adapter names `app motion extract-pose --adapter` accepts.
POSE_ADAPTERS: tuple[str, ...] = ("dwpose_onnx", "mock", "registry")


def create_pose_adapter(
    name: str,
    config: AppConfig,
    *,
    session_factory: Any = None,
    frame_reader: Any = None,
    **overrides: Any,
) -> Any:
    """Build the named pose adapter. Never downloads, never silently substitutes."""
    resolved = (name or config.pose.adapter or "registry").strip().lower()
    if resolved in {"", "none", "registry"}:
        from app.adapters.base import AdapterKind
        from app.adapters.base import registry as adapter_registry

        return adapter_registry.require(AdapterKind.POSE)

    if resolved == "dwpose_onnx":
        from app.adapters.dwpose import build_dwpose_adapter

        return build_dwpose_adapter(
            config,
            session_factory=session_factory,
            frame_reader=frame_reader,
            **overrides,
        )

    if resolved == "mock":
        from app.adapters.pose import MockPoseAdapter

        if overrides:
            raise ConfigError(
                "The mock pose adapter takes no model or provider settings",
                supplied=sorted(overrides),
            )
        return MockPoseAdapter()

    raise ConfigError(
        "Unknown pose adapter",
        adapter=resolved,
        known=list(POSE_ADAPTERS),
    )


__all__ = ["POSE_ADAPTERS", "create_pose_adapter"]
