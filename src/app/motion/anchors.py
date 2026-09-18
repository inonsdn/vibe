"""Deterministic anchor search: where two motion segments can be joined.

Given the tail of one normalized segment and the head of the next, this scores
every candidate pair and returns them ranked. The score is a weighted sum of
differences that a viewer would actually notice at a cut:

============================  =========================================
Term                          Why it matters
============================  =========================================
body joint distance           the overall shape must match
wrist / hand position         a hand jump is the most visible artefact
torso rotation                a shoulder-line flip reads as a stumble
head rotation                 a head snap breaks the illusion instantly
root position                 the body must not teleport
shoulder scale                a size change reads as a camera cut
incoming / outgoing velocity  matching poses moving in opposite
                              directions still cut badly
joint confidence              never anchor on a guessed pose
============================  =========================================

Everything is computed in canonical space, so the numbers are comparable across
sources. The search is exhaustive over the configured windows and fully
deterministic: same inputs, same ranking, every time. Ties break on the earliest
frame pair, so ordering never depends on dict iteration or float noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from app.core.errors import ValidationError
from app.motion.pose_format import PoseFrame
from app.motion.skeleton import BODY_JOINTS, HAND_JOINTS


@dataclass
class AnchorWeights:
    """Relative importance of each term. Configurable, not hard-coded."""

    body_distance: float = 1.0
    hand_distance: float = 1.5
    torso_angle: float = 0.8
    head_angle: float = 0.6
    root_position: float = 1.2
    shoulder_scale: float = 1.0
    velocity: float = 0.9

    def total(self) -> float:
        return (
            self.body_distance
            + self.hand_distance
            + self.torso_angle
            + self.head_angle
            + self.root_position
            + self.shoulder_scale
            + self.velocity
        )


@dataclass
class AnchorSearchSettings:
    """Where to look, and what counts as acceptable."""

    #: How many frames at the end of the previous segment to consider.
    prev_window: int = 24
    #: How many frames at the start of the next segment to consider.
    next_window: int = 24
    confidence_threshold: float = 0.35
    weights: AnchorWeights = field(default_factory=AnchorWeights)
    #: Normalisers turning a raw difference into a 0..1-ish penalty.
    position_scale_px: float = 60.0
    angle_scale_deg: float = 25.0
    velocity_scale_px: float = 18.0
    scale_tolerance: float = 0.02
    #: Candidates scoring above this are reported but marked unacceptable.
    max_acceptable_score: float = 0.35
    max_candidates: int = 10


@dataclass
class AnchorCandidate:
    """One scored (prev frame, next frame) pairing."""

    prev_frame: int
    next_frame: int
    score: float
    body_distance: float
    hand_distance: float
    torso_angle_delta: float
    head_angle_delta: float
    root_delta: float
    shoulder_scale_delta: float
    velocity_delta: float
    mean_confidence: float
    acceptable: bool
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "prev_frame": self.prev_frame,
            "next_frame": self.next_frame,
            "score": round(self.score, 6),
            "body_distance": round(self.body_distance, 4),
            "hand_distance": round(self.hand_distance, 4),
            "torso_angle_delta": round(self.torso_angle_delta, 4),
            "head_angle_delta": round(self.head_angle_delta, 4),
            "root_delta": round(self.root_delta, 4),
            "shoulder_scale_delta": round(self.shoulder_scale_delta, 6),
            "velocity_delta": round(self.velocity_delta, 4),
            "mean_confidence": round(self.mean_confidence, 4),
            "acceptable": self.acceptable,
            "warnings": self.warnings,
        }


def _angle_delta(a: float | None, b: float | None) -> float | None:
    """Smallest signed difference between two angles, in degrees."""
    if a is None or b is None:
        return None
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _velocity(
    poses: list[PoseFrame], position: int, threshold: float
) -> tuple[float, float] | None:
    """Root velocity at ``position``, from its neighbours where possible."""
    if not poses:
        return None
    before = poses[max(0, position - 1)].torso_center(threshold)
    after = poses[min(len(poses) - 1, position + 1)].torso_center(threshold)
    if before is None or after is None:
        return None
    span = min(len(poses) - 1, position + 1) - max(0, position - 1)
    if span <= 0:
        return (0.0, 0.0)
    return ((after[0] - before[0]) / span, (after[1] - before[1]) / span)


def _mean_joint_distance(a: PoseFrame, b: PoseFrame, threshold: float) -> tuple[float, float]:
    """Mean distance over jointly-confident body joints, and mean confidence."""
    distances: list[float] = []
    confidences: list[float] = []
    for name in BODY_JOINTS:
        first = a.confident_joint(name, threshold)
        second = b.confident_joint(name, threshold)
        if first is None or second is None:
            continue
        distances.append(first.distance_to(second))
        confidences.append(min(first.confidence, second.confidence))
    if not distances:
        return (float("inf"), 0.0)
    return (sum(distances) / len(distances), sum(confidences) / len(confidences))


def score_pair(
    prev_poses: list[PoseFrame],
    prev_position: int,
    next_poses: list[PoseFrame],
    next_position: int,
    settings: AnchorSearchSettings,
) -> AnchorCandidate:
    """Score one candidate pairing. Lower is better; 0 is a perfect match."""
    threshold = settings.confidence_threshold
    weights = settings.weights
    a = prev_poses[prev_position]
    b = next_poses[next_position]
    warnings: list[str] = []

    body_distance, mean_confidence = _mean_joint_distance(a, b, threshold)
    if not math.isfinite(body_distance):
        warnings.append("no jointly-confident body joints")

    hand_distances: list[float] = []
    for name in HAND_JOINTS:
        first = a.confident_joint(name, threshold)
        second = b.confident_joint(name, threshold)
        if first is not None and second is not None:
            hand_distances.append(first.distance_to(second))
    hand_distance = sum(hand_distances) / len(hand_distances) if hand_distances else body_distance
    if not hand_distances:
        warnings.append("wrists not confidently detected in both frames")

    torso_delta = _angle_delta(a.torso_angle_deg(threshold), b.torso_angle_deg(threshold))
    if torso_delta is None:
        torso_delta = settings.angle_scale_deg
        warnings.append("torso angle unavailable; penalised at the nominal scale")

    head_delta = _angle_delta(a.head_angle_deg(threshold), b.head_angle_deg(threshold))
    if head_delta is None:
        head_delta = 0.0
        warnings.append("head angle unavailable; term skipped")

    root_a = a.torso_center(threshold)
    root_b = b.torso_center(threshold)
    root_delta = (
        math.hypot(root_a[0] - root_b[0], root_a[1] - root_b[1])
        if root_a and root_b
        else settings.position_scale_px
    )

    shoulder_a = a.shoulder_width(threshold)
    shoulder_b = b.shoulder_width(threshold)
    if shoulder_a and shoulder_b and shoulder_a > 1e-6:
        shoulder_scale_delta = abs(shoulder_a - shoulder_b) / shoulder_a
    else:
        shoulder_scale_delta = settings.scale_tolerance
        warnings.append("shoulder width unavailable in one frame")

    velocity_a = _velocity(prev_poses, prev_position, threshold)
    velocity_b = _velocity(next_poses, next_position, threshold)
    if velocity_a is None or velocity_b is None:
        velocity_delta = settings.velocity_scale_px
        warnings.append("root velocity unavailable; penalised at the nominal scale")
    else:
        velocity_delta = math.hypot(velocity_a[0] - velocity_b[0], velocity_a[1] - velocity_b[1])

    def normalise(value: float, scale: float) -> float:
        return value / scale if scale > 0 else value

    raw = (
        weights.body_distance * normalise(body_distance, settings.position_scale_px)
        + weights.hand_distance * normalise(hand_distance, settings.position_scale_px)
        + weights.torso_angle * normalise(torso_delta, settings.angle_scale_deg)
        + weights.head_angle * normalise(head_delta, settings.angle_scale_deg)
        + weights.root_position * normalise(root_delta, settings.position_scale_px)
        + weights.shoulder_scale * normalise(shoulder_scale_delta, settings.scale_tolerance)
        + weights.velocity * normalise(velocity_delta, settings.velocity_scale_px)
    )
    score = raw / weights.total()

    # A low-confidence pose can score well by accident; discount it explicitly
    # rather than letting a guessed skeleton win the ranking.
    if mean_confidence < threshold:
        score += 1.0
        warnings.append("mean joint confidence is below the threshold")

    return AnchorCandidate(
        prev_frame=a.frame_index,
        next_frame=b.frame_index,
        score=score,
        body_distance=body_distance if math.isfinite(body_distance) else -1.0,
        hand_distance=hand_distance if math.isfinite(hand_distance) else -1.0,
        torso_angle_delta=torso_delta,
        head_angle_delta=head_delta,
        root_delta=root_delta,
        shoulder_scale_delta=shoulder_scale_delta,
        velocity_delta=velocity_delta,
        mean_confidence=mean_confidence,
        acceptable=score <= settings.max_acceptable_score and math.isfinite(body_distance),
        warnings=warnings,
    )


def rank_anchor_candidates(
    prev_poses: list[PoseFrame],
    next_poses: list[PoseFrame],
    settings: AnchorSearchSettings | None = None,
) -> list[AnchorCandidate]:
    """Score every pairing in the search windows and rank them, best first.

    The previous segment is searched from its **end** and the next from its
    **start**, because that is where a join can physically happen.
    """
    options = settings or AnchorSearchSettings()
    if not prev_poses or not next_poses:
        raise ValidationError(
            "Both segments need poses to search for an anchor",
            prev_frames=len(prev_poses),
            next_frames=len(next_poses),
        )

    prev_start = max(0, len(prev_poses) - max(1, options.prev_window))
    next_end = min(len(next_poses), max(1, options.next_window))

    candidates = [
        score_pair(prev_poses, prev_position, next_poses, next_position, options)
        for prev_position in range(prev_start, len(prev_poses))
        for next_position in range(next_end)
    ]
    # Deterministic ordering: score first, then the earliest frame pair.
    candidates.sort(key=lambda c: (round(c.score, 9), c.prev_frame, c.next_frame))
    return candidates[: options.max_candidates]


def select_anchor(
    prev_poses: list[PoseFrame],
    next_poses: list[PoseFrame],
    settings: AnchorSearchSettings | None = None,
    *,
    override_prev_frame: int | None = None,
    override_next_frame: int | None = None,
) -> AnchorCandidate:
    """Pick the anchor, honouring an operator override when given.

    An override is scored exactly like an automatic choice, so a human decision
    is recorded with the same evidence — it is not exempt from measurement.
    """
    options = settings or AnchorSearchSettings()
    if override_prev_frame is not None or override_next_frame is not None:
        if override_prev_frame is None or override_next_frame is None:
            raise ValidationError(
                "An anchor override must specify both frames",
                override_prev_frame=override_prev_frame,
                override_next_frame=override_next_frame,
            )
        prev_position = _position_of(prev_poses, override_prev_frame, "previous")
        next_position = _position_of(next_poses, override_next_frame, "next")
        candidate = score_pair(prev_poses, prev_position, next_poses, next_position, options)
        candidate.warnings.append("operator override")
        return candidate

    ranked = rank_anchor_candidates(prev_poses, next_poses, options)
    if not ranked:  # pragma: no cover - guarded by rank_anchor_candidates
        raise ValidationError("No anchor candidates were produced")
    return ranked[0]


def _position_of(poses: list[PoseFrame], frame_index: int, label: str) -> int:
    for position, pose in enumerate(poses):
        if pose.frame_index == frame_index:
            return position
    raise ValidationError(
        f"Frame is not part of the {label} segment",
        frame_index=frame_index,
        available=[poses[0].frame_index, poses[-1].frame_index] if poses else [],
    )


__all__ = [
    "AnchorCandidate",
    "AnchorSearchSettings",
    "AnchorWeights",
    "rank_anchor_candidates",
    "score_pair",
    "select_anchor",
]
