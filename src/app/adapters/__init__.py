"""Adapter interfaces for future neural preprocessing components.

**Nothing in this package implements a neural model, and nothing downloads
weights.** Each adapter is a documented interface plus a capability check that
reports honestly that it is unavailable. The pipeline treats every one of them
as optional: masks and pose data can be produced manually and imported, which
is how the system runs end to end today.

When a model is selected later, the only work is a new subclass in the matching
module plus registering it — no pipeline, API or CLI changes.
"""

from app.adapters.base import (
    AdapterCapability,
    AdapterKind,
    AdapterRegistry,
    AdapterStatus,
    AnalysisAdapter,
    NotImplementedAdapter,
    registry,
)
from app.adapters.densepose import DensePoseAdapter, densepose_stub
from app.adapters.depth import DepthAdapter, depth_stub
from app.adapters.face_landmarks import FaceLandmarkAdapter, face_landmarks_stub
from app.adapters.human_parsing import HumanParsingAdapter, human_parsing_stub
from app.adapters.optical_flow import OpticalFlowAdapter, optical_flow_stub
from app.adapters.pose import PoseAdapter, pose_stub
from app.adapters.sam2 import VideoMaskPropagationAdapter, sam2_stub

__all__ = [
    "AdapterCapability",
    "AdapterKind",
    "AdapterRegistry",
    "AdapterStatus",
    "AnalysisAdapter",
    "DensePoseAdapter",
    "DepthAdapter",
    "FaceLandmarkAdapter",
    "HumanParsingAdapter",
    "NotImplementedAdapter",
    "OpticalFlowAdapter",
    "PoseAdapter",
    "VideoMaskPropagationAdapter",
    "densepose_stub",
    "depth_stub",
    "face_landmarks_stub",
    "human_parsing_stub",
    "optical_flow_stub",
    "pose_stub",
    "registry",
    "sam2_stub",
]
