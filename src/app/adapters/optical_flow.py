"""Dense optical flow between consecutive frames.

Purpose in this system: flow is used to warp a rendered garment from frame N to
frame N+1 as a temporal-consistency prior, and by QC to distinguish real motion
from flicker.

Output contract: per frame pair, a ``.npz`` holding a ``(H, W, 2)`` float32
array named ``flow``, written as ``flow_{index:06d}_to_{next:06d}.npz``.
"""

from __future__ import annotations

from app.adapters.base import (
    AdapterKind,
    AnalysisAdapter,
    NotImplementedAdapter,
    registry,
)


class OpticalFlowAdapter(AnalysisAdapter):
    """Interface a future implementation must satisfy."""

    kind = AdapterKind.OPTICAL_FLOW


class _OpticalFlowAdapterStub(NotImplementedAdapter, OpticalFlowAdapter):
    """Documented placeholder; :meth:`run` raises rather than faking output."""


optical_flow_stub = _OpticalFlowAdapterStub(
    AdapterKind.OPTICAL_FLOW,
    "optical-flow",
    reason="No optical-flow model has been selected or installed yet.",
    expected_outputs=("flow_{index:06d}_to_{next:06d}.npz (H,W,2 float32)",),
    requires_gpu=True,
    estimated_vram_mb=2048,
    candidate_models=("RAFT", "GMFlow", "SEA-RAFT", "OpenCV DIS (classical fallback)"),
    integration_notes=(
        "A classical CPU fallback (OpenCV DIS) is acceptable here and needs "
        "no weights, but is still not wired up until it is actually tested."
    ),
)

registry.register(optical_flow_stub)

__all__ = ["OpticalFlowAdapter", "optical_flow_stub"]
