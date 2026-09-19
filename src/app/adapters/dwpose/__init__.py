"""DWPose ONNX pose extraction — the first real local model integration.

This package turns a reference video into canonical ``PoseFrame`` JSON using
two ONNX models the operator supplies **from local disk**: a person detector
and an RTMPose/DWPose whole-body keypoint model. Nothing here contains a model
name to download, a URL, or a fallback that fetches anything. Missing weights
produce an actionable refusal, never a silent stub.

Structure, so each piece can be tested without onnxruntime, CUDA or a video
file:

``session``     provider resolution (CUDA preferred, CPU fallback, recorded)
``detector``    person boxes from the detector session
``pose_model``  keypoints from the pose session (SimCC / heatmap / direct)
``keypoints``   COCO-WholeBody index -> canonical COCO-17 joint names
``roi``         crop rectangles and the coordinate restoration that undoes them
``tracking``    deterministic main-subject selection across frames
``temporal``    short-gap interpolation and confidence-aware smoothing
``video``       frame reading by absolute source index
``diagnostics`` overlay/skeleton previews and reports, written outside inputs
``adapter``     the ``PoseAdapter`` implementation that composes all of it
"""

from app.adapters.dwpose.adapter import ADAPTER_NAME, DWPoseOnnxAdapter, build_dwpose_adapter
from app.adapters.dwpose.session import (
    OnnxRuntimeSessionFactory,
    OnnxSessionFactory,
    ProviderResolution,
    resolve_providers,
)

__all__ = [
    "ADAPTER_NAME",
    "DWPoseOnnxAdapter",
    "OnnxRuntimeSessionFactory",
    "OnnxSessionFactory",
    "ProviderResolution",
    "build_dwpose_adapter",
    "resolve_providers",
]
