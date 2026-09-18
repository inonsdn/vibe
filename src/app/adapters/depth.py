"""Monocular depth estimation.

Purpose in this system: depth disambiguates occlusion (is the hand in front of
the garment?) and gives a future renderer a shading cue. Relative depth is
sufficient; metric depth is not required.

Output contract: per frame, a 16-bit single-channel PNG or ``.npy`` in
``output_dir``, plus ``scale.json`` describing the normalisation used.
"""

from __future__ import annotations

from app.adapters.base import (
    AdapterKind,
    AnalysisAdapter,
    NotImplementedAdapter,
    registry,
)


class DepthAdapter(AnalysisAdapter):
    """Interface a future implementation must satisfy."""

    kind = AdapterKind.DEPTH


class _DepthAdapterStub(NotImplementedAdapter, DepthAdapter):
    """Documented placeholder; :meth:`run` raises rather than faking output."""


depth_stub = _DepthAdapterStub(
    AdapterKind.DEPTH,
    "depth-estimation",
    reason="No depth model has been selected or installed yet.",
    expected_outputs=("frame_{index:06d}.png (16-bit depth)", "scale.json"),
    requires_gpu=True,
    estimated_vram_mb=2048,
    candidate_models=("Depth Anything V2", "Marigold", "MiDaS", "DPT"),
    integration_notes=(
        "Relative depth is enough; temporal stability matters more than " "absolute accuracy."
    ),
)

registry.register(depth_stub)

__all__ = ["DepthAdapter", "depth_stub"]
