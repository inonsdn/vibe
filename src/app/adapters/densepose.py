"""Dense surface correspondence (DensePose class of models).

Purpose in this system: per-pixel UV body coordinates let a future garment
renderer map a flat garment reference onto the moving body consistently across
frames, which is what keeps a printed pattern from swimming.

Output contract: per frame, an ``IUV`` array saved as ``.npz`` plus an optional
visualisation PNG.
"""

from __future__ import annotations

from app.adapters.base import (
    AdapterKind,
    AnalysisAdapter,
    NotImplementedAdapter,
    registry,
)


class DensePoseAdapter(AnalysisAdapter):
    """Interface a future implementation must satisfy."""

    kind = AdapterKind.DENSEPOSE


class _DensePoseAdapterStub(NotImplementedAdapter, DensePoseAdapter):
    """Documented placeholder; :meth:`run` raises rather than faking output."""


densepose_stub = _DensePoseAdapterStub(
    AdapterKind.DENSEPOSE,
    "densepose",
    reason="No dense-correspondence model has been selected or installed yet.",
    expected_outputs=(
        "frame_{index:06d}.npz (I, U, V arrays)",
        "frame_{index:06d}.png (optional viz)",
    ),
    requires_gpu=True,
    estimated_vram_mb=3072,
    candidate_models=("DensePose (Detectron2)", "DensePose-CSE", "Sapiens-dense"),
    integration_notes=(
        "Enables temporally consistent garment texture mapping; optional for " "the mock pipeline."
    ),
)

registry.register(densepose_stub)

__all__ = ["DensePoseAdapter", "densepose_stub"]
