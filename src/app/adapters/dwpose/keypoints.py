"""COCO-WholeBody keypoint indices -> the canonical COCO-17 joint vocabulary.

DWPose emits 133 whole-body keypoints. Its first 17 are the COCO-17 body in the
standard order, which makes the body mapping look like an identity — so it is
written out explicitly here rather than assumed. An off-by-one in this table
would swap a left ear for a right eye everywhere downstream and nothing else in
the system could detect it.

A plain 17-keypoint model (RTMPose body-only, ViTPose, HRNet) uses the same
first 17 indices, so the same table serves both; ``expects`` reports which
keypoint counts are understood.
"""

from __future__ import annotations

from app.motion.skeleton import BODY_JOINTS

#: Canonical joint name -> index in a COCO / COCO-WholeBody keypoint array.
COCO_BODY_INDEX: dict[str, int] = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

#: Whole-body hand ranges, half-open. 21 points each, wrist-rooted.
LEFT_HAND_RANGE: tuple[int, int] = (91, 112)
RIGHT_HAND_RANGE: tuple[int, int] = (112, 133)

#: Keypoint counts this mapping understands.
COCO_17 = 17
COCO_WHOLEBODY_133 = 133
SUPPORTED_KEYPOINT_COUNTS: tuple[int, ...] = (COCO_17, 26, 133)


def expects() -> tuple[int, ...]:
    return SUPPORTED_KEYPOINT_COUNTS


def validate_keypoint_count(count: int) -> None:
    """Refuse a keypoint layout whose body indices we cannot vouch for."""
    from app.core.errors import ValidationError

    if count < COCO_17:
        raise ValidationError(
            "The pose model returned fewer keypoints than COCO-17 needs",
            returned=count,
            required=COCO_17,
            supported=list(SUPPORTED_KEYPOINT_COUNTS),
        )
    if count not in SUPPORTED_KEYPOINT_COUNTS:
        raise ValidationError(
            "Unrecognised keypoint layout. The first 17 keypoints must be the "
            "COCO body in standard order; confirm your model before proceeding.",
            returned=count,
            supported=list(SUPPORTED_KEYPOINT_COUNTS),
        )


def has_hands(count: int) -> bool:
    return count >= COCO_WHOLEBODY_133


def body_indices() -> list[int]:
    """Keypoint indices for :data:`~app.motion.skeleton.BODY_JOINTS`, in order."""
    return [COCO_BODY_INDEX[name] for name in BODY_JOINTS]


def hand_point_names(side: str) -> list[str]:
    """Stable key names for one hand's 21 points."""
    return [f"{side}_hand_{i:02d}" for i in range(21)]


__all__ = [
    "COCO_17",
    "COCO_BODY_INDEX",
    "COCO_WHOLEBODY_133",
    "LEFT_HAND_RANGE",
    "RIGHT_HAND_RANGE",
    "SUPPORTED_KEYPOINT_COUNTS",
    "body_indices",
    "expects",
    "hand_point_names",
    "has_hands",
    "validate_keypoint_count",
]
