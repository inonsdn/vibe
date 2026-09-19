"""Region of interest: crop before detection, restore coordinates after.

A phone screen recording puts the dancer in the middle and the interface
everywhere else. Cropping to the dancer before detection removes the UI
avatars and the corner reference image from consideration entirely, which is
cheaper and more reliable than out-scoring them later.

The invariant that matters: **every coordinate leaving this adapter is in
original video pixels.** The ROI is an internal optimisation, so a pose
extracted with a crop and the same pose extracted without one describe the same
point in the same clip. :func:`restore_point` is the single place that undoes
the crop, and it is exercised directly by tests.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.errors import ValidationError


@dataclass(frozen=True)
class Roi:
    """A crop rectangle in original video pixels, already clipped to the frame."""

    x: int
    y: int
    width: int
    height: int
    mode: str = "none"

    @property
    def is_identity(self) -> bool:
        return self.mode == "none" or (self.x == 0 and self.y == 0)

    @property
    def offset(self) -> tuple[int, int]:
        return (self.x, self.y)

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


def resolve_roi(
    mode: str,
    values: tuple[float, float, float, float],
    *,
    frame_width: int,
    frame_height: int,
) -> Roi:
    """Turn configured ROI values into a clipped pixel rectangle.

    ``normalized`` values are fractions of the frame; ``pixels`` are absolute.
    Either way the result is clipped to the frame, because a crop that runs off
    the edge silently shortens the image and shifts every coordinate by an
    amount nothing downstream could recover.
    """
    if mode == "none":
        return Roi(x=0, y=0, width=frame_width, height=frame_height, mode="none")

    x, y, w, h = values
    if mode == "normalized":
        x, y, w, h = (x * frame_width, y * frame_height, w * frame_width, h * frame_height)
    elif mode != "pixels":
        raise ValidationError("Unknown roi_mode", roi_mode=mode)

    left = max(0, round(x))
    top = max(0, round(y))
    right = min(frame_width, round(x + w))
    bottom = min(frame_height, round(y + h))
    if right <= left or bottom <= top:
        raise ValidationError(
            "The configured ROI does not overlap the frame",
            roi=[x, y, w, h],
            mode=mode,
            frame=[frame_width, frame_height],
        )
    return Roi(x=left, y=top, width=right - left, height=bottom - top, mode=mode)


def restore_point(point: tuple[float, float], roi: Roi) -> tuple[float, float]:
    """Map a point measured inside the crop back to original video pixels."""
    return (point[0] + roi.x, point[1] + roi.y)


def restore_box(box: tuple[float, float, float, float], roi: Roi) -> tuple[float, ...]:
    """Map an ``(x1, y1, x2, y2)`` box back to original video pixels."""
    x1, y1, x2, y2 = box
    return (x1 + roi.x, y1 + roi.y, x2 + roi.x, y2 + roi.y)


__all__ = ["Roi", "resolve_roi", "restore_box", "restore_point"]
