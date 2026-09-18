"""The canonical joint vocabulary.

COCO-17 is used as the internal body skeleton because every realistic future
pose adapter (DWPose, RTMPose, ViTPose, MediaPipe via a mapping) can emit it,
and because it contains exactly the joints the normalizer and QC checks need:
both shoulders and both hips define the torso frame, and the wrists and nose
carry most of the perceptual weight at a join.

Adding a joint here is a schema change: bump ``POSE_SCHEMA_VERSION``.
"""

from __future__ import annotations

from enum import StrEnum


class SkeletonFormat(StrEnum):
    """Which joint vocabulary a pose file uses."""

    COCO_17 = "coco_17"


#: Canonical body joints, in a fixed order. The order is part of the contract:
#: distance metrics and preview drawing both rely on it.
BODY_JOINTS: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

BODY_JOINT_INDEX: dict[str, int] = {name: index for index, name in enumerate(BODY_JOINTS)}

#: Joints whose absence makes a frame unusable. A long run missing any of these
#: is rejected rather than interpolated: the torso frame and the hands carry the
#: information a garment render depends on.
HIGH_PRIORITY_JOINTS: tuple[str, ...] = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
)

#: Joints that define the torso coordinate frame used for normalization.
TORSO_JOINTS: tuple[str, ...] = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
)

#: Joints used by the head-orientation term of the anchor score.
HEAD_JOINTS: tuple[str, ...] = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")

#: Hand joints, weighted heavily at a join because a hand jump is very visible.
HAND_JOINTS: tuple[str, ...] = ("left_wrist", "right_wrist")

#: Bones, for preview drawing and limb-length continuity checks.
SKELETON_EDGES: tuple[tuple[str, str], ...] = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
)

#: Limbs whose length must stay continuous across a join or a bridge. Left and
#: right are checked separately so an asymmetric glitch is not averaged away.
LIMB_EDGES: tuple[tuple[str, str], ...] = (
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
)


def is_body_joint(name: str) -> bool:
    return name in BODY_JOINT_INDEX


__all__ = [
    "BODY_JOINTS",
    "BODY_JOINT_INDEX",
    "HAND_JOINTS",
    "HEAD_JOINTS",
    "HIGH_PRIORITY_JOINTS",
    "LIMB_EDGES",
    "SKELETON_EDGES",
    "TORSO_JOINTS",
    "SkeletonFormat",
    "is_body_joint",
]
