"""Health, environment and offline-verification endpoints."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from app.adapters.base import AdapterKind
from app.api.deps import context_dependency
from app.api.schemas import HealthResponse
from app.backends.registry import available_backends
from app.offline.verify import adapter_capability, verify_offline
from app.pipeline.context import ServiceContext
from app.version import APP_NAME, APP_VERSION

router = APIRouter(tags=["system"])

Ctx = Annotated[ServiceContext, Depends(context_dependency)]


@router.get("/health", response_model=HealthResponse)
def health(context: Ctx) -> HealthResponse:
    report = verify_offline(context.config, probe_comfyui=False, include_backends=False)
    return HealthResponse(
        status="ok" if report.ok else "degraded",
        app=APP_NAME,
        version=APP_VERSION,
        schema_version=context.database.schema_version,
        offline_ok=report.ok,
        backends=available_backends(),
    )


@router.get("/offline/verify")
def offline(context: Ctx, probe_comfyui: bool = True) -> dict[str, Any]:
    return verify_offline(context.config, probe_comfyui=probe_comfyui).as_dict()


@router.get("/adapters")
def adapters() -> dict[str, Any]:
    return {
        "adapters": {kind.value: adapter_capability(kind) for kind in AdapterKind},
        "note": (
            "No neural model is implemented or downloaded. Each entry is an "
            "adapter interface with a capability check."
        ),
    }


@router.get("/config")
def config(context: Ctx) -> dict[str, Any]:
    """The effective configuration and its hash (as recorded in manifests)."""
    return {
        "config": context.config.model_dump(mode="json"),
        "config_hash": context.config.config_hash(),
        "data_root": str(context.data_root.path),
        "workflows_dir": str(context.config.workflows_dir()),
    }


__all__ = ["router"]
