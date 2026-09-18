"""The stable internal pose JSON format.

Every pose in this system — imported, extracted, normalized or bridged — uses
this one shape. A future DWPose/RTMPose/ViTPose/MediaPipe adapter converts *to*
this format; nothing downstream knows which model produced a pose.

On-disk shape (``frame_{index:06d}.json``)::

    {
      "schema_version": "1",
      "skeleton_format": "coco_17",
      "frame_index": 42,
      "timestamp_s": 1.4,
      "body": {"nose": {"x": 512.0, "y": 300.0, "confidence": 0.98}, …},
      "hands": {"left_hand_0": {…}},        // optional
      "face": {"yaw_deg": 3.1, "pitch_deg": -1.4, "roll_deg": 0.2},  // optional
      "source_bbox": [x, y, w, h],          // optional
      "source": "imported|extracted|normalized|bridge"
    }

Coordinates are pixels in the frame the pose was measured in — the *source*
frame before normalization, the *canonical target* frame afterwards. Which one
is recorded in ``space``, so a normalized pose can never be mistaken for a raw
one.

Deliberately absent: any pixel data. A pose file describes where joints are, and
that is the whole point of this phase — a motion reference contributes geometry,
never imagery.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

from pydantic import Field, model_validator

from app.core.errors import ValidationError
from app.domain.base import DomainModel
from app.motion.skeleton import BODY_JOINTS, SkeletonFormat

#: Bump when the on-disk pose shape changes incompatibly.
POSE_SCHEMA_VERSION = "1"

POSE_FILENAME_TEMPLATE = "frame_{index:06d}.json"


class PoseSpace(StrEnum):
    """Which coordinate frame a pose's pixel values live in."""

    SOURCE = "source"
    CANONICAL = "canonical"


class PoseOrigin(StrEnum):
    """How a pose came to exist. Recorded so provenance survives the pipeline."""

    IMPORTED = "imported"
    EXTRACTED = "extracted"
    NORMALIZED = "normalized"
    BRIDGE = "bridge"


class Joint2D(DomainModel):
    """One joint: a 2D position plus the detector's confidence in it."""

    x: float
    y: float
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @property
    def is_confident(self) -> bool:
        return self.confidence > 0.0

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    def distance_to(self, other: Joint2D) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


class FaceOrientation(DomainModel):
    """Head orientation in degrees, when a detector can supply it."""

    yaw_deg: float | None = None
    pitch_deg: float | None = None
    roll_deg: float | None = None

    @property
    def is_known(self) -> bool:
        return any(v is not None for v in (self.yaw_deg, self.pitch_deg, self.roll_deg))


class PoseFrame(DomainModel):
    """One frame of pose data."""

    schema_version: str = POSE_SCHEMA_VERSION
    skeleton_format: SkeletonFormat = SkeletonFormat.COCO_17
    frame_index: int = Field(ge=0)
    timestamp_s: float = Field(ge=0.0)
    space: PoseSpace = PoseSpace.SOURCE
    origin: PoseOrigin = PoseOrigin.IMPORTED

    body: dict[str, Joint2D] = Field(default_factory=dict)
    hands: dict[str, Joint2D] = Field(default_factory=dict)
    face: FaceOrientation | None = None
    #: The person's bounding box in the source frame, ``[x, y, w, h]``.
    source_bbox: tuple[float, float, float, float] | None = None

    @model_validator(mode="after")
    def _known_joints_only(self) -> Self:
        unknown = sorted(set(self.body) - set(BODY_JOINTS))
        if unknown:
            raise ValueError(
                f"pose contains joints outside the {self.skeleton_format.value} "
                f"vocabulary: {unknown}"
            )
        return self

    # -- accessors --------------------------------------------------------
    def joint(self, name: str) -> Joint2D | None:
        """The joint, or ``None`` when it is absent."""
        return self.body.get(name)

    def confident_joint(self, name: str, threshold: float) -> Joint2D | None:
        """The joint only if its confidence clears ``threshold``."""
        joint = self.body.get(name)
        if joint is None or joint.confidence < threshold:
            return None
        return joint

    def has_all(self, names: Iterable[str], threshold: float = 0.0) -> bool:
        return all(self.confident_joint(name, threshold) is not None for name in names)

    def midpoint(self, a: str, b: str, threshold: float = 0.0) -> tuple[float, float] | None:
        """Midpoint of two joints, or ``None`` if either is missing."""
        first = self.confident_joint(a, threshold)
        second = self.confident_joint(b, threshold)
        if first is None or second is None:
            return None
        return ((first.x + second.x) / 2.0, (first.y + second.y) / 2.0)

    def shoulder_width(self, threshold: float = 0.0) -> float | None:
        left = self.confident_joint("left_shoulder", threshold)
        right = self.confident_joint("right_shoulder", threshold)
        if left is None or right is None:
            return None
        return left.distance_to(right)

    def hip_width(self, threshold: float = 0.0) -> float | None:
        left = self.confident_joint("left_hip", threshold)
        right = self.confident_joint("right_hip", threshold)
        if left is None or right is None:
            return None
        return left.distance_to(right)

    def torso_center(self, threshold: float = 0.0) -> tuple[float, float] | None:
        """Midpoint between the shoulder centre and the hip centre.

        Preferred over the hip centre alone as a root: it is far less sensitive
        to a single mis-detected hip, which is the common failure mode.
        """
        shoulders = self.midpoint("left_shoulder", "right_shoulder", threshold)
        hips = self.midpoint("left_hip", "right_hip", threshold)
        if shoulders is None or hips is None:
            return None
        return ((shoulders[0] + hips[0]) / 2.0, (shoulders[1] + hips[1]) / 2.0)

    def torso_length(self, threshold: float = 0.0) -> float | None:
        shoulders = self.midpoint("left_shoulder", "right_shoulder", threshold)
        hips = self.midpoint("left_hip", "right_hip", threshold)
        if shoulders is None or hips is None:
            return None
        return math.hypot(shoulders[0] - hips[0], shoulders[1] - hips[1])

    def torso_angle_deg(self, threshold: float = 0.0) -> float | None:
        """Lean of the shoulder line, in degrees. 0 is level."""
        left = self.confident_joint("left_shoulder", threshold)
        right = self.confident_joint("right_shoulder", threshold)
        if left is None or right is None:
            return None
        return math.degrees(math.atan2(right.y - left.y, right.x - left.x))

    def head_angle_deg(self, threshold: float = 0.0) -> float | None:
        """Head tilt from the eye line, falling back to face orientation."""
        left = self.confident_joint("left_eye", threshold)
        right = self.confident_joint("right_eye", threshold)
        if left is not None and right is not None:
            return math.degrees(math.atan2(right.y - left.y, right.x - left.x))
        if self.face is not None and self.face.roll_deg is not None:
            return self.face.roll_deg
        return None

    def missing_joints(self, names: Iterable[str], threshold: float) -> list[str]:
        return [name for name in names if self.confident_joint(name, threshold) is None]

    # -- transforms -------------------------------------------------------
    def transformed(
        self,
        *,
        scale: float,
        offset: tuple[float, float],
        space: PoseSpace | None = None,
        origin: PoseOrigin | None = None,
    ) -> PoseFrame:
        """Return a copy with ``p -> p * scale + offset`` applied.

        Confidences and joint membership are preserved exactly: normalization
        changes where a joint is, never whether it was detected.
        """

        def move(joint: Joint2D) -> Joint2D:
            return Joint2D(
                x=joint.x * scale + offset[0],
                y=joint.y * scale + offset[1],
                confidence=joint.confidence,
            )

        return self.model_copy(
            update={
                "body": {name: move(joint) for name, joint in self.body.items()},
                "hands": {name: move(joint) for name, joint in self.hands.items()},
                "space": space or self.space,
                "origin": origin or self.origin,
            }
        )

    def with_index(self, frame_index: int, timestamp_s: float) -> PoseFrame:
        return self.model_copy(update={"frame_index": frame_index, "timestamp_s": timestamp_s})

    def joint_array(self, names: Sequence[str] = BODY_JOINTS) -> list[tuple[float, float, float]]:
        """``(x, y, confidence)`` per requested joint; missing joints are 0-confidence."""
        out: list[tuple[float, float, float]] = []
        for name in names:
            joint = self.body.get(name)
            out.append((joint.x, joint.y, joint.confidence) if joint else (0.0, 0.0, 0.0))
        return out


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def pose_path(directory: str | os.PathLike[str], frame_index: int) -> Path:
    return Path(directory) / POSE_FILENAME_TEMPLATE.format(index=frame_index)


def save_pose_frame(path: str | os.PathLike[str], pose: PoseFrame) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(pose.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target


def load_pose_frame(path: str | os.PathLike[str]) -> PoseFrame:
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValidationError("Pose file could not be read", path=str(target)) from exc
    except json.JSONDecodeError as exc:
        raise ValidationError("Pose file is not valid JSON", path=str(target)) from exc
    if not isinstance(payload, dict):
        raise ValidationError("Pose file must contain an object", path=str(target))
    version = str(payload.get("schema_version", ""))
    if version and version != POSE_SCHEMA_VERSION:
        raise ValidationError(
            "Unsupported pose schema version",
            path=str(target),
            found=version,
            supported=POSE_SCHEMA_VERSION,
        )
    try:
        return PoseFrame.model_validate(payload)
    except Exception as exc:
        raise ValidationError(
            "Pose file does not match the schema", path=str(target), error=str(exc)
        ) from exc


def save_pose_sequence(directory: str | os.PathLike[str], poses: Iterable[PoseFrame]) -> list[int]:
    """Write a sequence keyed by each pose's own ``frame_index``."""
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    written: list[int] = []
    for pose in poses:
        save_pose_frame(pose_path(folder, pose.frame_index), pose)
        written.append(pose.frame_index)
    return sorted(written)


def list_pose_indices(directory: str | os.PathLike[str]) -> list[int]:
    folder = Path(directory)
    if not folder.is_dir():
        return []
    indices: list[int] = []
    for entry in folder.iterdir():
        if not entry.is_file() or entry.suffix != ".json":
            continue
        stem = entry.stem
        if stem.startswith("frame_") and stem[6:].isdigit():
            indices.append(int(stem[6:]))
    return sorted(indices)


def load_pose_sequence(
    directory: str | os.PathLike[str],
    indices: Iterable[int] | None = None,
) -> list[PoseFrame]:
    """Load poses in ascending frame order, raising on a missing frame."""
    folder = Path(directory)
    wanted = list(indices) if indices is not None else list_pose_indices(folder)
    out: list[PoseFrame] = []
    for index in wanted:
        path = pose_path(folder, index)
        if not path.is_file():
            raise ValidationError(
                "Pose frame missing from sequence", frame_index=index, path=str(path)
            )
        out.append(load_pose_frame(path))
    return out


def poses_equal(a: PoseFrame, b: PoseFrame, *, tolerance: float = 0.0) -> bool:
    """Geometric equality: same joints at the same places.

    Frame index, timestamp and origin are ignored on purpose — a bridge endpoint
    is "the same pose as the anchor" even though it sits at a different output
    index.
    """
    if set(a.body) != set(b.body):
        return False
    for name, joint in a.body.items():
        other = b.body[name]
        if abs(joint.x - other.x) > tolerance or abs(joint.y - other.y) > tolerance:
            return False
        if abs(joint.confidence - other.confidence) > max(tolerance, 1e-9):
            return False
    return True


def summarize_bbox(poses: Sequence[PoseFrame]) -> dict[str, Any]:
    """Bounding-box statistics for the tracked person across a sequence."""
    boxes = [pose.source_bbox for pose in poses if pose.source_bbox is not None]
    if not boxes:
        return {"frames_with_bbox": 0}
    widths = [box[2] for box in boxes]
    heights = [box[3] for box in boxes]
    centers_x = [box[0] + box[2] / 2.0 for box in boxes]
    centers_y = [box[1] + box[3] / 2.0 for box in boxes]

    def stats(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        middle = len(ordered) // 2
        median = (
            ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0
        )
        return {
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "mean": round(sum(values) / len(values), 4),
            "median": round(median, 4),
        }

    return {
        "frames_with_bbox": len(boxes),
        "width": stats(widths),
        "height": stats(heights),
        "center_x": stats(centers_x),
        "center_y": stats(centers_y),
    }


__all__ = [
    "POSE_FILENAME_TEMPLATE",
    "POSE_SCHEMA_VERSION",
    "FaceOrientation",
    "Joint2D",
    "PoseFrame",
    "PoseOrigin",
    "PoseSpace",
    "list_pose_indices",
    "load_pose_frame",
    "load_pose_sequence",
    "pose_path",
    "poses_equal",
    "save_pose_frame",
    "save_pose_sequence",
    "summarize_bbox",
]
