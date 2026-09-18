"""Motion Composition: build a motion-control sequence from motion references.

This package extracts **motion only**. No source pixel, face, garment or
background from a motion reference video ever reaches a Master Human
Performance — the artifacts produced here are pose JSON and skeleton previews.

The stages are:

1. :mod:`app.motion.skeleton` — the canonical joint vocabulary
2. :mod:`app.motion.pose_format` — the stable internal pose JSON schema and IO
3. :mod:`app.motion.normalize` — map each source motion into one canonical
   body coordinate system, independently of the others
4. :mod:`app.motion.anchors` — deterministic search for compatible join frames
5. :mod:`app.motion.bridge` — cubic Hermite pose bridges between segments
6. :mod:`app.motion.preview` — skeleton preview video for operator review
"""

from app.motion.anchors import AnchorCandidate, AnchorSearchSettings, rank_anchor_candidates
from app.motion.bridge import BridgeResult, BridgeSettings, generate_bridge
from app.motion.normalize import NormalizationResult, normalize_sequence
from app.motion.pose_format import (
    POSE_SCHEMA_VERSION,
    Joint2D,
    PoseFrame,
    load_pose_frame,
    load_pose_sequence,
    save_pose_frame,
    save_pose_sequence,
)
from app.motion.preview import render_skeleton_preview
from app.motion.skeleton import (
    BODY_JOINTS,
    HIGH_PRIORITY_JOINTS,
    SKELETON_EDGES,
    SkeletonFormat,
)

__all__ = [
    "BODY_JOINTS",
    "HIGH_PRIORITY_JOINTS",
    "POSE_SCHEMA_VERSION",
    "SKELETON_EDGES",
    "AnchorCandidate",
    "AnchorSearchSettings",
    "BridgeResult",
    "BridgeSettings",
    "Joint2D",
    "NormalizationResult",
    "PoseFrame",
    "SkeletonFormat",
    "generate_bridge",
    "load_pose_frame",
    "load_pose_sequence",
    "normalize_sequence",
    "rank_anchor_candidates",
    "render_skeleton_preview",
    "save_pose_frame",
    "save_pose_sequence",
]
