"""Canonical normalization: map one source motion into the shared body frame.

Two reference clips can show different people, at different distances, framed
differently. Before their motions can be joined, each must be expressed in one
canonical body coordinate system so that "the same pose" means the same numbers.

Each source is normalized **independently** — never against the other — so
adding a third reference later cannot change how the first two were normalized.

The transform per frame is a uniform scale plus a translation:

    canonical = source * scale + offset

Uniform, because anisotropic scaling would distort limb proportions and make
limb-length continuity meaningless. The scale comes from robust statistics over
the *whole* segment (median shoulder width and torso length), not from the
current frame, which is what stops the body pumping in and out as the detector
jitters. A per-frame residual correction is then smoothed over time.

What this module never does: touch a pixel. It reads pose JSON and writes pose
JSON. Source imagery does not travel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

from app.core.errors import ValidationError
from app.motion.pose_format import Joint2D, PoseFrame, PoseOrigin, PoseSpace
from app.motion.skeleton import HIGH_PRIORITY_JOINTS


@dataclass
class NormalizationSettings:
    """Tunables for :func:`normalize_sequence`. Mirrors the canonical profile."""

    target_width: int
    target_height: int
    target_center: tuple[float, float]
    canonical_shoulder_width: float
    canonical_torso_length: float
    confidence_threshold: float = 0.35
    #: Weight of shoulder width vs torso length when deriving the scale.
    shoulder_weight: float = 0.6
    #: Exponential smoothing factor for per-frame scale/offset residuals.
    #: 0 disables smoothing; 1 freezes at the segment-wide value.
    smoothing: float = 0.8
    #: Longest run of low-confidence frames that may be interpolated.
    max_interpolation_gap: int = 5
    #: Longest tolerated run missing a high-priority joint before rejection.
    max_missing_joint_run: int = 8
    #: Reject the segment if frame-to-frame scale changes by more than this.
    max_scale_step: float = 0.02


@dataclass
class FrameTransform:
    """The transform applied to one frame, recorded for the manifest."""

    frame_index: int
    scale: float
    offset_x: float
    offset_y: float
    interpolated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "scale": round(self.scale, 8),
            "offset_x": round(self.offset_x, 6),
            "offset_y": round(self.offset_y, 6),
            "interpolated": self.interpolated,
        }


@dataclass
class NormalizationResult:
    poses: list[PoseFrame]
    transforms: list[FrameTransform]
    base_scale: float
    source_shoulder_width: float
    source_torso_length: float
    interpolated_frames: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_count": len(self.poses),
            "base_scale": round(self.base_scale, 8),
            "source_shoulder_width": round(self.source_shoulder_width, 4),
            "source_torso_length": round(self.source_torso_length, 4),
            "interpolated_frames": self.interpolated_frames,
            "warnings": self.warnings,
            "stats": self.stats,
            "transforms": [t.as_dict() for t in self.transforms],
        }


def _median(values: list[float]) -> float:
    if not values:
        raise ValidationError("Cannot take the median of an empty sample")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _longest_run(flags: list[bool]) -> int:
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def robust_scale(
    poses: list[PoseFrame], settings: NormalizationSettings
) -> tuple[float, float, float]:
    """Return ``(scale, median_shoulder_width, median_torso_length)``.

    Both measurements are used because either alone is fragile: shoulder width
    collapses when the performer turns sideways, and torso length shortens when
    they bend. Combining them with a configurable weight is materially steadier
    than trusting one.
    """
    threshold = settings.confidence_threshold
    shoulders = [w for w in (p.shoulder_width(threshold) for p in poses) if w and w > 1e-6]
    torsos = [t for t in (p.torso_length(threshold) for p in poses) if t and t > 1e-6]

    if not shoulders and not torsos:
        raise ValidationError(
            "Cannot derive a canonical scale: no frame has a confident torso",
            frames=len(poses),
            confidence_threshold=threshold,
        )

    shoulder_median = _median(shoulders) if shoulders else 0.0
    torso_median = _median(torsos) if torsos else 0.0

    scales: list[tuple[float, float]] = []
    if shoulder_median > 0:
        scales.append(
            (settings.canonical_shoulder_width / shoulder_median, settings.shoulder_weight)
        )
    if torso_median > 0:
        scales.append(
            (settings.canonical_torso_length / torso_median, 1.0 - settings.shoulder_weight)
        )

    total_weight = sum(weight for _, weight in scales) or 1.0
    scale = sum(value * weight for value, weight in scales) / total_weight
    if not math.isfinite(scale) or scale <= 0:
        raise ValidationError("Derived a non-positive canonical scale", scale=scale)
    return scale, shoulder_median, torso_median


def _interpolate_centers(
    centers: list[tuple[float, float] | None], max_gap: int
) -> tuple[list[tuple[float, float]], list[int], list[str]]:
    """Fill short holes in the per-frame root centre by linear interpolation.

    Only *interior* gaps no longer than ``max_gap`` are filled. Leading and
    trailing holes are extended from the nearest known value, because there is
    nothing to interpolate between — and that is recorded as a warning rather
    than hidden.
    """
    warnings: list[str] = []
    interpolated: list[int] = []
    known = [i for i, c in enumerate(centers) if c is not None]
    if not known:
        raise ValidationError("No frame has a resolvable root centre")

    filled: list[tuple[float, float]] = [(0.0, 0.0)] * len(centers)
    for index in known:
        value = centers[index]
        assert value is not None
        filled[index] = value

    # Leading / trailing extension.
    first, last = known[0], known[-1]
    if first > 0:
        warnings.append(
            f"root centre missing for the first {first} frame(s); held from frame {first}"
        )
        for i in range(first):
            filled[i] = filled[first]
            interpolated.append(i)
    if last < len(centers) - 1:
        tail = len(centers) - 1 - last
        warnings.append(f"root centre missing for the last {tail} frame(s); held from frame {last}")
        for i in range(last + 1, len(centers)):
            filled[i] = filled[last]
            interpolated.append(i)

    # Interior gaps.
    for a, b in pairwise(known):
        gap = b - a - 1
        if gap <= 0:
            continue
        if gap > max_gap:
            raise ValidationError(
                "Missing-pose gap is too long to interpolate",
                gap_frames=gap,
                between_frames=[a, b],
                max_interpolation_gap=max_gap,
                hint="Trim the segment, or supply pose data for the missing frames.",
            )
        start, end = filled[a], filled[b]
        for step in range(1, gap + 1):
            t = step / (gap + 1)
            filled[a + step] = (
                start[0] + (end[0] - start[0]) * t,
                start[1] + (end[1] - start[1]) * t,
            )
            interpolated.append(a + step)
    return filled, sorted(set(interpolated)), warnings


def _smooth(values: list[float], factor: float) -> list[float]:
    """Symmetric exponential smoothing (forward then backward).

    A single forward pass introduces a lag, which at a join reads as the body
    drifting into place. Running it in both directions and averaging keeps the
    result phase-aligned.
    """
    if factor <= 0.0 or len(values) < 2:
        return list(values)
    alpha = 1.0 - min(factor, 0.99)

    forward: list[float] = [values[0]]
    for value in values[1:]:
        forward.append(alpha * value + (1 - alpha) * forward[-1])

    backward: list[float] = [values[-1]]
    for value in reversed(values[:-1]):
        backward.append(alpha * value + (1 - alpha) * backward[-1])
    backward.reverse()

    return [(f + b) / 2.0 for f, b in zip(forward, backward, strict=True)]


def check_missing_joint_runs(
    poses: list[PoseFrame], settings: NormalizationSettings
) -> dict[str, int]:
    """Longest run per high-priority joint, rejecting anything over tolerance."""
    threshold = settings.confidence_threshold
    runs: dict[str, int] = {}
    for name in HIGH_PRIORITY_JOINTS:
        flags = [pose.confident_joint(name, threshold) is None for pose in poses]
        runs[name] = _longest_run(flags)
    worst = max(runs.items(), key=lambda item: item[1], default=("", 0))
    if worst[1] > settings.max_missing_joint_run:
        raise ValidationError(
            "A high-priority joint is missing for too many consecutive frames",
            joint=worst[0],
            longest_run=worst[1],
            max_missing_joint_run=settings.max_missing_joint_run,
            hint="Trim the segment to a section where the body is fully visible.",
        )
    return runs


def normalize_sequence(
    poses: list[PoseFrame],
    settings: NormalizationSettings,
    *,
    playback_speed: float = 1.0,
    output_fps: float = 30.0,
    start_output_index: int = 0,
) -> NormalizationResult:
    """Normalize one source segment into the canonical body coordinate system.

    Source timing is preserved: output frame *k* corresponds to source frame *k*
    unless ``playback_speed`` is explicitly changed, in which case only the
    recorded timestamps change. Resampling the motion itself is deliberately out
    of scope — it is a separate, lossy decision that should be explicit.
    """
    if not poses:
        raise ValidationError("Cannot normalize an empty pose sequence")
    if playback_speed <= 0:
        raise ValidationError("playback_speed must be positive", playback_speed=playback_speed)

    joint_runs = check_missing_joint_runs(poses, settings)
    scale, shoulder_median, torso_median = robust_scale(poses, settings)
    threshold = settings.confidence_threshold

    centers = [pose.torso_center(threshold) for pose in poses]
    filled_centers, interpolated, warnings = _interpolate_centers(
        centers, settings.max_interpolation_gap
    )

    # Per-frame offset that would place this frame's root exactly on target.
    target_x, target_y = settings.target_center
    raw_offset_x = [target_x - center[0] * scale for center in filled_centers]
    raw_offset_y = [target_y - center[1] * scale for center in filled_centers]

    # Smooth the translation so detector jitter does not shake the body, while
    # leaving genuine travel intact.
    offset_x = _smooth(raw_offset_x, settings.smoothing)
    offset_y = _smooth(raw_offset_y, settings.smoothing)

    # A single segment-wide scale is used for every frame. This is what makes
    # "no frame-to-frame scale pumping" true by construction rather than by
    # tuning: there is nothing per-frame left to pump.
    scales = [scale] * len(poses)

    transforms: list[FrameTransform] = []
    normalized: list[PoseFrame] = []
    interpolated_set = set(interpolated)

    for position, pose in enumerate(poses):
        output_index = start_output_index + position
        timestamp = output_index / output_fps / playback_speed
        moved = pose.transformed(
            scale=scales[position],
            offset=(offset_x[position], offset_y[position]),
            space=PoseSpace.CANONICAL,
            origin=PoseOrigin.NORMALIZED,
        ).with_index(output_index, timestamp)
        # The source bounding box describes the source frame and would be
        # meaningless (and misleading) in canonical space.
        normalized.append(moved.model_copy(update={"source_bbox": None}))
        transforms.append(
            FrameTransform(
                frame_index=output_index,
                scale=scales[position],
                offset_x=offset_x[position],
                offset_y=offset_y[position],
                interpolated=position in interpolated_set,
            )
        )

    scale_steps = [abs(b - a) / max(a, 1e-9) for a, b in pairwise(scales)]
    max_step = max(scale_steps, default=0.0)
    if max_step > settings.max_scale_step:  # pragma: no cover - constant by construction
        raise ValidationError(
            "Frame-to-frame scale changes exceed the configured limit",
            max_scale_step=settings.max_scale_step,
            observed=max_step,
        )

    observed = [p.shoulder_width(threshold) for p in normalized]
    confident = [w for w in observed if w]
    stats = {
        "canonical_shoulder_width_median": round(_median(confident), 4) if confident else None,
        "canonical_shoulder_width_min": round(min(confident), 4) if confident else None,
        "canonical_shoulder_width_max": round(max(confident), 4) if confident else None,
        "max_frame_to_frame_scale_step": round(max_step, 8),
        "missing_joint_runs": joint_runs,
        "playback_speed": playback_speed,
    }

    return NormalizationResult(
        poses=normalized,
        transforms=transforms,
        base_scale=scale,
        source_shoulder_width=shoulder_median,
        source_torso_length=torso_median,
        interpolated_frames=interpolated,
        warnings=warnings,
        stats=stats,
    )


def relative_motion_signature(poses: list[PoseFrame], joint: str) -> list[tuple[float, float]]:
    """A joint's position relative to the torso centre, per frame.

    Normalization must move the body into a shared frame **without** altering
    how limbs move relative to the body. This signature is invariant to the
    translation part of the transform and scales linearly with the scale part,
    which is exactly what the normalization tests assert.
    """
    signature: list[tuple[float, float]] = []
    for pose in poses:
        center = pose.torso_center(0.0)
        target = pose.joint(joint)
        if center is None or target is None:
            signature.append((float("nan"), float("nan")))
            continue
        signature.append((target.x - center[0], target.y - center[1]))
    return signature


def interpolate_joint(a: Joint2D, b: Joint2D, t: float) -> Joint2D:
    """Linear joint interpolation, carrying the lower confidence forward."""
    return Joint2D(
        x=a.x + (b.x - a.x) * t,
        y=a.y + (b.y - a.y) * t,
        confidence=min(a.confidence, b.confidence),
    )


__all__ = [
    "FrameTransform",
    "NormalizationResult",
    "NormalizationSettings",
    "check_missing_joint_runs",
    "interpolate_joint",
    "normalize_sequence",
    "relative_motion_signature",
    "robust_scale",
]
