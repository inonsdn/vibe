"""2D/3D body pose estimation.

Purpose in this system: pose is **control and QA metadata only**. It is used to
decide which garment views a motion requires (a turn exposes the back), to sanity
check that motion is preserved, and to drive future pose-conditioned renderers.
It is never used to regenerate the human.

Output contract: one JSON per frame with a stable keypoint schema
(``{"keypoints": [{"name", "x", "y", "score"}], "schema": "<name>"}``).
"""

from __future__ import annotations

from app.adapters.base import (
    AdapterKind,
    AnalysisAdapter,
    NotImplementedAdapter,
    registry,
)


class PoseAdapter(AnalysisAdapter):
    """Interface a future implementation must satisfy."""

    kind = AdapterKind.POSE


class _PoseAdapterStub(NotImplementedAdapter, PoseAdapter):
    """Documented placeholder; :meth:`run` raises rather than faking output."""


pose_stub = _PoseAdapterStub(
    AdapterKind.POSE,
    "pose-estimation",
    reason="No pose model has been selected or installed yet.",
    expected_outputs=("frame_{index:06d}.json (keypoints)",),
    requires_gpu=True,
    estimated_vram_mb=1024,
    candidate_models=("RTMPose", "ViTPose", "OpenPose", "MediaPipe Pose", "DWPose"),
    integration_notes=(
        "Auxiliary control/QA metadata. The master video pixels remain the "
        "source of truth for the performance."
    ),
)

registry.register(pose_stub)

__all__ = ["PoseAdapter", "pose_stub"]
