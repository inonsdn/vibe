"""Temporal cleanup: fill short gaps, smooth gently, and leave fast hands alone.

Three rules, in this order:

1. **Short gaps are interpolated.** A joint missing for one to ``max_gap``
   frames between two confident observations is linearly interpolated, and its
   confidence is scaled down so downstream code can tell a filled joint from a
   measured one.
2. **Long runs stay missing.** Interpolating across twenty frames invents
   motion. The pipeline already knows how to report and reject long missing
   runs; hiding them here would remove the signal that the clip is unusable.
3. **Smoothing is confidence-aware and speed-aware.** Jitter on a low-confidence
   joint is noise worth averaging away. Displacement on a fast-moving wrist is
   *signal*: a dancer's hand can cross 40 pixels in a frame, and averaging that
   shortens the arc, which is the most visible artefact in the final video. So
   the smoothing weight falls to zero as the per-frame displacement approaches
   ``fast_motion_px``, and smoothing never reaches across a gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.motion.pose_format import Joint2D, PoseFrame
from app.motion.skeleton import BODY_JOINTS


@dataclass
class CleanupSettings:
    max_interpolation_gap: int = 3
    interpolation_confidence_scale: float = 0.6
    smoothing_window: int = 2
    smoothing_strength: float = 0.6
    fast_motion_px: float = 18.0


@dataclass
class CleanupReport:
    """What the cleanup actually did, per joint. Written into diagnostics."""

    interpolated: dict[str, int] = field(default_factory=dict)
    smoothed: dict[str, int] = field(default_factory=dict)
    smoothing_skipped_fast: dict[str, int] = field(default_factory=dict)
    long_runs: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "interpolated": dict(sorted(self.interpolated.items())),
            "smoothed": dict(sorted(self.smoothed.items())),
            "smoothing_skipped_fast_motion": dict(sorted(self.smoothing_skipped_fast.items())),
            "longest_missing_run": dict(sorted(self.long_runs.items())),
            "total_interpolated": sum(self.interpolated.values()),
        }


Track = list[Joint2D | None]


def _tracks(poses: list[PoseFrame]) -> dict[str, Track]:
    return {name: [pose.body.get(name) for pose in poses] for name in BODY_JOINTS}


def interpolate_short_gaps(
    track: Track, *, max_gap: int, confidence_scale: float
) -> tuple[Track, int, int]:
    """Fill runs of ``None`` up to ``max_gap`` long. Returns (track, filled, longest)."""
    filled = list(track)
    count = 0
    longest = 0
    index = 0
    size = len(filled)
    while index < size:
        if filled[index] is not None:
            index += 1
            continue
        start = index
        while index < size and filled[index] is None:
            index += 1
        gap = index - start
        longest = max(longest, gap)
        before = filled[start - 1] if start > 0 else None
        after = filled[index] if index < size else None
        if before is None or after is None or gap > max_gap:
            # Unbounded on one side, or too long to invent: leave it missing.
            continue
        for step in range(1, gap + 1):
            t = step / (gap + 1)
            filled[start + step - 1] = Joint2D(
                x=before.x + (after.x - before.x) * t,
                y=before.y + (after.y - before.y) * t,
                confidence=min(before.confidence, after.confidence) * confidence_scale,
            )
            count += 1
    return filled, count, longest


def smooth_track(
    track: Track, *, window: int, strength: float, fast_motion_px: float
) -> tuple[Track, int, int]:
    """Confidence-weighted moving average within contiguous runs only.

    Returns ``(track, smoothed_count, skipped_for_speed)``.
    """
    if window <= 0 or strength <= 0.0:
        return list(track), 0, 0

    out: Track = list(track)
    smoothed = 0
    skipped = 0
    size = len(track)
    index = 0
    while index < size:
        if track[index] is None:
            index += 1
            continue
        start = index
        while index < size and track[index] is not None:
            index += 1
        run = range(start, index)

        for position in run:
            current = track[position]
            if current is None:  # pragma: no cover - guarded by the run scan
                continue
            # Local speed: how far this joint moved since the previous frame in
            # the same run. A fast hand is preserved verbatim.
            speed = 0.0
            if position > start:
                previous = track[position - 1]
                if previous is not None:
                    speed = current.distance_to(previous)
            if position + 1 < index:
                nxt = track[position + 1]
                if nxt is not None:
                    speed = max(speed, current.distance_to(nxt))

            alpha = strength * max(0.0, 1.0 - speed / fast_motion_px)
            if alpha <= 0.0:
                skipped += 1
                continue

            low = max(start, position - window)
            high = min(index, position + window + 1)
            weight_sum = 0.0
            x_sum = 0.0
            y_sum = 0.0
            for neighbour in range(low, high):
                joint = track[neighbour]
                if joint is None:  # pragma: no cover - contiguous by construction
                    continue
                weight = max(joint.confidence, 1e-3)
                weight_sum += weight
                x_sum += joint.x * weight
                y_sum += joint.y * weight
            if weight_sum <= 0.0:  # pragma: no cover - weights have a floor
                continue
            mean_x, mean_y = x_sum / weight_sum, y_sum / weight_sum
            out[position] = Joint2D(
                x=current.x + (mean_x - current.x) * alpha,
                y=current.y + (mean_y - current.y) * alpha,
                confidence=current.confidence,
            )
            smoothed += 1
    return out, smoothed, skipped


def clean_sequence(
    poses: list[PoseFrame], settings: CleanupSettings
) -> tuple[list[PoseFrame], CleanupReport]:
    """Apply gap filling then smoothing to a whole sequence, in frame order."""
    report = CleanupReport()
    if not poses:
        return [], report

    ordered = sorted(poses, key=lambda p: p.frame_index)
    tracks = _tracks(ordered)
    cleaned: dict[str, Track] = {}

    for name, track in tracks.items():
        filled, count, longest = interpolate_short_gaps(
            track,
            max_gap=settings.max_interpolation_gap,
            confidence_scale=settings.interpolation_confidence_scale,
        )
        if count:
            report.interpolated[name] = count
        if longest:
            report.long_runs[name] = longest

        smoothed_track, smoothed, skipped = smooth_track(
            filled,
            window=settings.smoothing_window,
            strength=settings.smoothing_strength,
            fast_motion_px=settings.fast_motion_px,
        )
        if smoothed:
            report.smoothed[name] = smoothed
        if skipped:
            report.smoothing_skipped_fast[name] = skipped
        cleaned[name] = smoothed_track

    out: list[PoseFrame] = []
    for position, pose in enumerate(ordered):
        body = {
            name: joint for name, track in cleaned.items() if (joint := track[position]) is not None
        }
        out.append(pose.model_copy(update={"body": body}))
    return out, report


__all__ = [
    "CleanupReport",
    "CleanupSettings",
    "clean_sequence",
    "interpolate_short_gaps",
    "smooth_track",
]
