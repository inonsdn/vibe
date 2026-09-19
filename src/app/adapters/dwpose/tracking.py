"""Deterministic main-subject selection across a clip.

The reference clips are Instagram screen recordings. A person detector run on
them finds, besides the dancer: profile-picture avatars in the header, a
cartoon or reference image parked in a corner, suggested-account thumbnails,
and sometimes a second person in the background. Picking "the highest-scoring
detection" switches between these from frame to frame, and a pose sequence that
switches subject is worse than one with gaps — normalization will happily fit a
canonical body to the wrong person.

Selection is a **deterministic score**, not a learned tracker:

    score = w_area·area + w_center·centrality + w_iou·IoU + w_cont·continuity

with two hard gates applied first — a minimum box area (an avatar is tiny) and
a maximum distance from frame centre (a corner image is not the subject). On
the first frame only area and centrality can contribute, which is exactly what
picks the large central dancer. From then on IoU and continuity dominate, and a
challenger must beat the incumbent by ``switch_margin`` to take over, so a
momentarily larger background figure cannot steal the track.

Every term is a pure function of the inputs, so the same video always produces
the same subject choices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from app.adapters.dwpose.detector import Detection


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection over union of two ``(x1, y1, x2, y2)`` boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class SubjectScore:
    """One candidate's terms, kept separately so a bad choice is explainable."""

    detection: Detection
    area_term: float
    center_term: float
    iou_term: float
    continuity_term: float
    total: float
    rejected: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "box": self.detection.as_dict(),
            "area": round(self.area_term, 4),
            "centrality": round(self.center_term, 4),
            "iou": round(self.iou_term, 4),
            "continuity": round(self.continuity_term, 4),
            "total": round(self.total, 4),
            "rejected": self.rejected,
        }


@dataclass
class TrackerSettings:
    area_weight: float = 0.35
    center_weight: float = 0.25
    iou_weight: float = 0.25
    continuity_weight: float = 0.15
    min_area_fraction: float = 0.02
    max_center_distance: float = 0.85
    switch_margin: float = 0.15
    max_coast_frames: int = 12


@dataclass
class TrackerState:
    previous: Detection | None = None
    frames_since_seen: int = 0
    switches: int = 0
    rejected_counts: dict[str, int] = field(default_factory=dict)


class SubjectTracker:
    """Chooses the main dancer, frame by frame, and remembers its choice."""

    def __init__(self, settings: TrackerSettings, *, frame_width: int, frame_height: int) -> None:
        self.settings = settings
        self.frame_width = max(1, frame_width)
        self.frame_height = max(1, frame_height)
        self.state = TrackerState()
        self._frame_area = float(self.frame_width * self.frame_height)
        self._half_diagonal = math.hypot(self.frame_width, self.frame_height) / 2.0
        self._center = (self.frame_width / 2.0, self.frame_height / 2.0)

    # -- terms ------------------------------------------------------------
    def _area_term(self, detection: Detection) -> float:
        """Square-rooted area fraction: linear in *size*, not in pixel count.

        Without the root, a dancer twice the linear size of a background figure
        scores four times higher and the term swamps everything else.
        """
        return min(1.0, math.sqrt(max(0.0, detection.area) / self._frame_area) * 2.0)

    def _center_term(self, detection: Detection) -> float:
        cx, cy = detection.center
        distance = math.hypot(cx - self._center[0], cy - self._center[1])
        return max(0.0, 1.0 - distance / self._half_diagonal)

    def _continuity_term(self, detection: Detection, previous: Detection | None) -> float:
        if previous is None:
            return 0.0
        cx, cy = detection.center
        px, py = previous.center
        # Scaled by the subject's own size: a big dancer moves more pixels per
        # frame than a small one without being a different person.
        reference = max(previous.width, previous.height, 1.0)
        return max(0.0, 1.0 - math.hypot(cx - px, cy - py) / reference)

    def score(self, detection: Detection, previous: Detection | None) -> SubjectScore:
        settings = self.settings
        area_fraction = max(0.0, detection.area) / self._frame_area
        cx, cy = detection.center
        center_distance = math.hypot(cx - self._center[0], cy - self._center[1])

        rejected: str | None = None
        if area_fraction < settings.min_area_fraction:
            rejected = "too_small"
        elif center_distance / self._half_diagonal > settings.max_center_distance:
            rejected = "too_far_from_centre"

        area_term = self._area_term(detection)
        center_term = self._center_term(detection)
        iou_term = iou(detection.box, previous.box) if previous is not None else 0.0
        continuity_term = self._continuity_term(detection, previous)
        total = (
            settings.area_weight * area_term
            + settings.center_weight * center_term
            + settings.iou_weight * iou_term
            + settings.continuity_weight * continuity_term
        )
        return SubjectScore(
            detection=detection,
            area_term=area_term,
            center_term=center_term,
            iou_term=iou_term,
            continuity_term=continuity_term,
            total=0.0 if rejected else total,
            rejected=rejected,
        )

    # -- selection --------------------------------------------------------
    def select(self, detections: list[Detection]) -> tuple[Detection | None, list[SubjectScore]]:
        """Pick this frame's subject and update the track.

        Returns the chosen detection (``None`` when nothing survives the gates)
        and every candidate's scored terms, which the diagnostics report writes
        out so a wrong choice can be inspected rather than guessed at.
        """
        previous = self.state.previous
        if previous is not None and self.state.frames_since_seen > self.settings.max_coast_frames:
            # The subject has been gone long enough that continuing to match
            # against a stale box would be worse than starting over.
            previous = None

        scored = [self.score(detection, previous) for detection in detections]
        for candidate in scored:
            if candidate.rejected:
                self.state.rejected_counts[candidate.rejected] = (
                    self.state.rejected_counts.get(candidate.rejected, 0) + 1
                )

        eligible = [candidate for candidate in scored if candidate.rejected is None]
        if not eligible:
            self.state.frames_since_seen += 1
            return None, scored

        # Stable ordering: score, then area, then left-to-right. Two identical
        # boxes must not be resolved by list order alone.
        best = max(
            eligible,
            key=lambda c: (round(c.total, 6), round(c.detection.area, 3), -c.detection.x1),
        )

        if previous is not None:
            incumbent = max(
                eligible,
                key=lambda c: (round(c.iou_term, 6), round(c.continuity_term, 6)),
            )
            keeps_track = incumbent.iou_term > 0.0 or incumbent.continuity_term > 0.0
            if (
                keeps_track
                and incumbent is not best
                and best.total < incumbent.total + self.settings.switch_margin
            ):
                best = incumbent
            elif keeps_track and incumbent is not best:
                self.state.switches += 1

        self.state.previous = best.detection
        self.state.frames_since_seen = 0
        return best.detection, scored

    def summary(self) -> dict[str, Any]:
        return {
            "switches": self.state.switches,
            "rejected": dict(sorted(self.state.rejected_counts.items())),
            "settings": {
                "area_weight": self.settings.area_weight,
                "center_weight": self.settings.center_weight,
                "iou_weight": self.settings.iou_weight,
                "continuity_weight": self.settings.continuity_weight,
                "min_area_fraction": self.settings.min_area_fraction,
                "max_center_distance": self.settings.max_center_distance,
                "switch_margin": self.settings.switch_margin,
            },
        }


__all__ = ["SubjectScore", "SubjectTracker", "TrackerSettings", "TrackerState", "iou"]
