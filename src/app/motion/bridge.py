"""Pose bridges between motion segments.

A bridge is **generated pose data**, never blended source imagery. Two clips of
two different people cannot be cross-faded without both people appearing; a
bridge sidesteps that entirely by interpolating skeletons and letting the
character animator render the result later.

Interpolation is cubic Hermite, because linear interpolation matches positions
but not velocities: the body would arrive at the join, stop dead, and set off
again in a new direction. Hermite takes the outgoing velocity of the previous
segment and the incoming velocity of the next, so the motion flows through.

Interpolating each joint independently is not enough on its own: two joints
sharing a bone are pulled along different curves, so the bone stretches in the
middle of the bridge. Measured on the test fixtures that reached 11% — visibly
wrong. Every interpolated frame therefore goes through a **bone-length
correction** pass that walks the skeleton outward from the shoulders and puts
each child joint at the interpolated bone length along its raw direction. Limb
lengths then follow the interpolation of the two anchors' lengths by
construction, not by luck.

The endpoint guarantee: ``bridge[0]`` is *exactly* the previous anchor pose and
``bridge[-1]`` is *exactly* the next anchor pose — byte-for-byte, not merely
close. The composition layout relies on it, and a QC check asserts it.

Frame layout at a join (no duplication, no gap)::

    … A.start … (anchor_prev-1) │ bridge[0..B-1] │ (anchor_next+1) … B.end …
                                 ↑                ↑
                    bridge[0] == pose(anchor_prev)
                                  bridge[B-1] == pose(anchor_next)

The anchor frames themselves are contributed by the bridge, not by the
segments, so every output frame has exactly one source.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

from app.core.errors import ValidationError
from app.motion.pose_format import (
    FaceOrientation,
    Joint2D,
    PoseFrame,
    PoseOrigin,
    PoseSpace,
)
from app.motion.skeleton import LIMB_EDGES

#: Kinematic chain used by the bone-length correction, parent -> child, in the
#: order it must be applied. Shoulders anchor the structure; everything else is
#: placed relative to an already-corrected parent.
CORRECTION_CHAIN: tuple[tuple[str, str], ...] = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)

#: The prototype's supported bridge length.
MIN_BRIDGE_FRAMES = 10
MAX_BRIDGE_FRAMES = 12


@dataclass
class BridgeSettings:
    frame_count: int = 12
    #: Scales the estimated endpoint velocities. 0 degenerates to a smoothstep
    #: (positions matched, velocities ignored); 1 is full Hermite.
    tangent_strength: float = 1.0
    #: "linear" | "smoothstep" | "ease_in_out" -- reparameterises t.
    #: Default is linear: with Hermite tangents, linear t is what makes the
    #: bridge's endpoint velocities match the segments it joins. smoothstep and
    #: ease_in_out give a gentler middle at the cost of zeroing velocity at the
    #: endpoints, which is a visible stall unless the anchors are already still.
    easing: str = "linear"
    #: Reject a bridge whose limb lengths wander further than this fraction.
    max_limb_length_drift: float = 0.08
    #: Reject a bridge whose shoulder scale wanders further than this fraction.
    max_scale_drift: float = 0.02
    #: Reject a bridge whose frame-to-frame root velocity jumps by more than this.
    max_velocity_step_px: float = 24.0
    enforce_prototype_range: bool = True

    def validate(self) -> None:
        if self.enforce_prototype_range and not (
            MIN_BRIDGE_FRAMES <= self.frame_count <= MAX_BRIDGE_FRAMES
        ):
            raise ValidationError(
                "Bridge length is outside the supported prototype range",
                frame_count=self.frame_count,
                supported=[MIN_BRIDGE_FRAMES, MAX_BRIDGE_FRAMES],
                hint="Set enforce_prototype_range=false to experiment with other lengths.",
            )
        if self.frame_count < 2:
            raise ValidationError(
                "A bridge needs at least 2 frames to carry both endpoints",
                frame_count=self.frame_count,
            )
        if self.easing not in {"linear", "smoothstep", "ease_in_out"}:
            raise ValidationError("Unknown easing", easing=self.easing)


@dataclass
class BridgeResult:
    poses: list[PoseFrame]
    settings: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def frame_count(self) -> int:
        return len(self.poses)

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_count": self.frame_count,
            "first_frame": self.poses[0].frame_index if self.poses else None,
            "last_frame": self.poses[-1].frame_index if self.poses else None,
            "settings": self.settings,
            "metrics": self.metrics,
            "warnings": self.warnings,
        }


def _ease(t: float, easing: str) -> float:
    if easing == "linear":
        return t
    if easing == "smoothstep":
        return t * t * (3.0 - 2.0 * t)
    return 0.5 - 0.5 * math.cos(math.pi * t)  # ease_in_out


def hermite(p0: float, p1: float, m0: float, m1: float, t: float) -> float:
    """Cubic Hermite basis: position and tangent at both ends."""
    t2 = t * t
    t3 = t2 * t
    return (
        (2 * t3 - 3 * t2 + 1) * p0
        + (t3 - 2 * t2 + t) * m0
        + (-2 * t3 + 3 * t2) * p1
        + (t3 - t2) * m1
    )


def _joint_velocity(
    poses: list[PoseFrame], position: int, name: str, *, forward: bool
) -> tuple[float, float]:
    """One joint's velocity in px/frame at ``position``.

    ``forward`` selects the outgoing side (previous segment) or the incoming
    side (next segment), so each endpoint's tangent comes from the motion that
    actually surrounds it.
    """
    if not poses:
        return (0.0, 0.0)
    here = poses[position].joint(name)
    if here is None:
        return (0.0, 0.0)
    neighbour_index = position - 1 if forward else position + 1
    if not 0 <= neighbour_index < len(poses):
        return (0.0, 0.0)
    neighbour = poses[neighbour_index].joint(name)
    if neighbour is None:
        return (0.0, 0.0)
    if forward:
        return (here.x - neighbour.x, here.y - neighbour.y)
    return (neighbour.x - here.x, neighbour.y - here.y)


def generate_bridge(
    prev_poses: list[PoseFrame],
    prev_position: int,
    next_poses: list[PoseFrame],
    next_position: int,
    settings: BridgeSettings | None = None,
    *,
    start_output_index: int = 0,
    output_fps: float = 30.0,
) -> BridgeResult:
    """Generate the bridge poses between two anchors.

    ``prev_position`` and ``next_position`` are *positions* within their
    sequences (not frame indices), so velocity can be read from the neighbours.
    """
    options = settings or BridgeSettings()
    options.validate()

    if not 0 <= prev_position < len(prev_poses):
        raise ValidationError("prev_position is out of range", prev_position=prev_position)
    if not 0 <= next_position < len(next_poses):
        raise ValidationError("next_position is out of range", next_position=next_position)

    start_pose = prev_poses[prev_position]
    end_pose = next_poses[next_position]
    warnings: list[str] = []

    shared = sorted(set(start_pose.body) & set(end_pose.body))
    if not shared:
        raise ValidationError(
            "Anchor poses share no joints; cannot build a bridge",
            prev_frame=start_pose.frame_index,
            next_frame=end_pose.frame_index,
        )
    dropped = sorted((set(start_pose.body) | set(end_pose.body)) - set(shared))
    if dropped:
        warnings.append(
            f"{len(dropped)} joint(s) present in only one anchor were omitted: {dropped[:6]}"
        )

    count = options.frame_count
    poses: list[PoseFrame] = []

    for step in range(count):
        raw_t = step / (count - 1)
        t = _ease(raw_t, options.easing)

        # The endpoints are copied, never interpolated. Floating-point evaluation
        # at t=0 and t=1 is *almost* exact; "almost" is not what the contract
        # says, so the exact poses are substituted.
        if step == 0:
            body = {name: start_pose.body[name].model_copy() for name in shared}
            face = start_pose.face
        elif step == count - 1:
            body = {name: end_pose.body[name].model_copy() for name in shared}
            face = end_pose.face
        else:
            body = {}
            for name in shared:
                a = start_pose.body[name]
                b = end_pose.body[name]
                m0 = _joint_velocity(prev_poses, prev_position, name, forward=True)
                m1 = _joint_velocity(next_poses, next_position, name, forward=False)
                # Velocities are px/frame; the Hermite basis is parameterised on
                # t in [0, 1], so they scale by the number of frame steps.
                strength = options.tangent_strength * (count - 1)
                body[name] = Joint2D(
                    x=hermite(a.x, b.x, m0[0] * strength, m1[0] * strength, t),
                    y=hermite(a.y, b.y, m0[1] * strength, m1[1] * strength, t),
                    confidence=min(a.confidence, b.confidence),
                )
            body = correct_bone_lengths(body, start_pose, end_pose, t)
            face = _interpolate_face(start_pose.face, end_pose.face, t)

        output_index = start_output_index + step
        poses.append(
            PoseFrame(
                skeleton_format=start_pose.skeleton_format,
                frame_index=output_index,
                timestamp_s=output_index / output_fps,
                space=PoseSpace.CANONICAL,
                origin=PoseOrigin.BRIDGE,
                body=body,
                face=face,
                source_bbox=None,
            )
        )

    metrics = bridge_metrics(poses)
    _enforce_limits(metrics, options, warnings)
    return BridgeResult(
        poses=poses,
        settings={
            "frame_count": count,
            "tangent_strength": options.tangent_strength,
            "easing": options.easing,
            "anchor_prev_frame": start_pose.frame_index,
            "anchor_next_frame": end_pose.frame_index,
            "interpolation": "cubic_hermite",
            "joints": shared,
        },
        metrics=metrics,
        warnings=warnings,
    )


def correct_bone_lengths(
    body: dict[str, Joint2D],
    start_pose: PoseFrame,
    end_pose: PoseFrame,
    t: float,
) -> dict[str, Joint2D]:
    """Put every bone at the interpolated length of the two anchors' bones.

    Walks :data:`CORRECTION_CHAIN` outward from the shoulders. Each child keeps
    the *direction* the raw interpolation gave it — so the pose still looks
    interpolated — but is moved to the exact interpolated bone length, so the
    skeleton stays rigid.

    A bone whose length is unknown at either anchor is left alone: inventing a
    length would be worse than leaving the raw interpolation in place.
    """
    corrected = dict(body)

    def bone_length(pose: PoseFrame, parent: str, child: str) -> float | None:
        a = pose.joint(parent)
        b = pose.joint(child)
        return a.distance_to(b) if a and b else None

    for parent, child in CORRECTION_CHAIN:
        if parent not in corrected or child not in corrected:
            continue
        start_length = bone_length(start_pose, parent, child)
        end_length = bone_length(end_pose, parent, child)
        if start_length is None or end_length is None:
            continue
        target = start_length + (end_length - start_length) * t

        anchor = corrected[parent]
        raw = corrected[child]
        dx, dy = raw.x - anchor.x, raw.y - anchor.y
        distance = math.hypot(dx, dy)
        if distance <= 1e-9:
            # Degenerate direction: fall back to the start anchor's direction so
            # the joint still lands somewhere physically sensible.
            start_parent = start_pose.joint(parent)
            start_child = start_pose.joint(child)
            if start_parent is None or start_child is None:
                continue
            dx, dy = start_child.x - start_parent.x, start_child.y - start_parent.y
            distance = math.hypot(dx, dy)
            if distance <= 1e-9:
                continue
        scale = target / distance
        corrected[child] = Joint2D(
            x=anchor.x + dx * scale,
            y=anchor.y + dy * scale,
            confidence=raw.confidence,
        )
    return corrected


def _interpolate_face(
    a: FaceOrientation | None, b: FaceOrientation | None, t: float
) -> FaceOrientation | None:
    if a is None or b is None:
        return None

    def blend(first: float | None, second: float | None) -> float | None:
        if first is None or second is None:
            return None
        return first + (second - first) * t

    return FaceOrientation(
        yaw_deg=blend(a.yaw_deg, b.yaw_deg),
        pitch_deg=blend(a.pitch_deg, b.pitch_deg),
        roll_deg=blend(a.roll_deg, b.roll_deg),
    )


def bridge_metrics(poses: list[PoseFrame]) -> dict[str, Any]:
    """Continuity measurements over a generated bridge."""
    if len(poses) < 2:
        return {}

    # Same reasoning as the limb metric: measure the deviation from the
    # interpolated expectation, not from the first frame.
    shoulder_widths = [p.shoulder_width(0.0) for p in poses]
    scale_drift = 0.0
    if shoulder_widths[0] and shoulder_widths[-1]:
        start_width, end_width = shoulder_widths[0], shoulder_widths[-1]
        for index, width in enumerate(shoulder_widths):
            if width is None:
                continue
            t = index / (len(poses) - 1)
            expected = start_width + (end_width - start_width) * t
            if expected > 1e-6:
                scale_drift = max(scale_drift, abs(width - expected) / expected)

    # Each bone should follow the linear interpolation between the two anchors'
    # lengths. Deviation from that expectation is distortion the bridge itself
    # introduced -- which is the thing worth failing on. Comparing against the
    # first frame instead would penalise anchors that legitimately differ.
    limb_drift: dict[str, float] = {}
    count = len(poses)
    for a_name, b_name in LIMB_EDGES:
        lengths: list[float | None] = []
        for pose in poses:
            first_joint = pose.joint(a_name)
            second_joint = pose.joint(b_name)
            lengths.append(
                first_joint.distance_to(second_joint) if first_joint and second_joint else None
            )
        if lengths[0] is None or lengths[-1] is None or lengths[0] <= 1e-6:
            continue
        start_length, end_length = lengths[0], lengths[-1]
        worst = 0.0
        for index, length in enumerate(lengths):
            if length is None:
                continue
            t = index / (count - 1)
            expected = start_length + (end_length - start_length) * t
            if expected <= 1e-6:
                continue
            worst = max(worst, abs(length - expected) / expected)
        limb_drift[f"{a_name}->{b_name}"] = worst

    centers = [p.torso_center(0.0) for p in poses]
    velocities: list[float] = []
    for a, b in pairwise(centers):
        if a is None or b is None:
            continue
        velocities.append(math.hypot(b[0] - a[0], b[1] - a[1]))
    velocity_steps = [abs(b - a) for a, b in pairwise(velocities)]

    return {
        "max_scale_drift": round(scale_drift, 8),
        "max_limb_length_drift": round(max(limb_drift.values()), 8) if limb_drift else 0.0,
        "limb_length_drift": {k: round(v, 8) for k, v in sorted(limb_drift.items())},
        "max_velocity_step_px": round(max(velocity_steps), 6) if velocity_steps else 0.0,
        "mean_velocity_px": round(sum(velocities) / len(velocities), 6) if velocities else 0.0,
    }


def _enforce_limits(metrics: dict[str, Any], settings: BridgeSettings, warnings: list[str]) -> None:
    """Fail loudly on a bridge that would be visibly wrong."""
    scale_drift = float(metrics.get("max_scale_drift", 0.0))
    if scale_drift > settings.max_scale_drift:
        raise ValidationError(
            "Generated bridge changes body scale beyond the configured limit",
            max_scale_drift=settings.max_scale_drift,
            observed=scale_drift,
            hint="Choose a closer anchor pair, or lengthen the bridge.",
        )
    limb_drift = float(metrics.get("max_limb_length_drift", 0.0))
    if limb_drift > settings.max_limb_length_drift:
        raise ValidationError(
            "Generated bridge produces impossible limb lengths",
            max_limb_length_drift=settings.max_limb_length_drift,
            observed=limb_drift,
            worst=metrics.get("limb_length_drift"),
            hint="The anchors are too far apart; pick a more similar pair.",
        )
    velocity_step = float(metrics.get("max_velocity_step_px", 0.0))
    if velocity_step > settings.max_velocity_step_px:
        warnings.append(
            f"bridge root velocity steps by {velocity_step:.2f}px "
            f"(limit {settings.max_velocity_step_px}); review the preview"
        )


__all__ = [
    "MAX_BRIDGE_FRAMES",
    "MIN_BRIDGE_FRAMES",
    "BridgeResult",
    "BridgeSettings",
    "bridge_metrics",
    "generate_bridge",
    "hermite",
]
