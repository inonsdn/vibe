"""Face landmark extraction.

Purpose in this system: landmarks define the face region used by the
face-difference QC check, which is the numeric proof that the performer's
identity was not altered. They also feed the protected mask.

Output contract: per frame, a JSON with a stable landmark schema and a
bounding box: ``{"schema", "bbox": [x, y, w, h], "landmarks": [[x, y], ...]}``.
"""

from __future__ import annotations

from app.adapters.base import (
    AdapterKind,
    AnalysisAdapter,
    NotImplementedAdapter,
    registry,
)


class FaceLandmarkAdapter(AnalysisAdapter):
    """Interface a future implementation must satisfy."""

    kind = AdapterKind.FACE_LANDMARKS


class _FaceLandmarkAdapterStub(NotImplementedAdapter, FaceLandmarkAdapter):
    """Documented placeholder; :meth:`run` raises rather than faking output."""


face_landmarks_stub = _FaceLandmarkAdapterStub(
    AdapterKind.FACE_LANDMARKS,
    "face-landmarks",
    reason="No face-landmark model has been selected or installed yet.",
    expected_outputs=("frame_{index:06d}.json (landmarks + bbox)",),
    requires_gpu=True,
    estimated_vram_mb=512,
    candidate_models=("MediaPipe FaceMesh", "InsightFace", "3DDFA_V2", "SynergyNet"),
    integration_notes=(
        "The face QC check falls back to the protected mask when landmarks "
        "are unavailable, so this adapter is optional but recommended."
    ),
)

registry.register(face_landmarks_stub)

__all__ = ["FaceLandmarkAdapter", "face_landmarks_stub"]
