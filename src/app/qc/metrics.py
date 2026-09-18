"""Pure numeric measurements used by the QC checks.

Every function here takes arrays and returns numbers. Keeping them free of
config and filesystem access makes them trivially testable and reusable from
the API.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def per_pixel_abs_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Max absolute channel difference per pixel, as ``int16``."""
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return diff.max(axis=2) if diff.ndim == 3 else diff


@dataclass
class RegionDiff:
    pixels: int
    max_diff: int
    mean_diff: float
    changed_pixels: int
    changed_fraction: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "pixels": self.pixels,
            "max_diff": self.max_diff,
            "mean_diff": round(self.mean_diff, 6),
            "changed_pixels": self.changed_pixels,
            "changed_fraction": round(self.changed_fraction, 8),
        }


EMPTY_REGION = RegionDiff(0, 0, 0.0, 0, 0.0)


def region_diff(a: np.ndarray, b: np.ndarray, selection: np.ndarray) -> RegionDiff:
    """Difference statistics restricted to ``selection`` (bool or mask)."""
    per_pixel = per_pixel_abs_diff(a, b)
    mask = selection.astype(bool) if selection.dtype != bool else selection
    if mask.shape[:2] != per_pixel.shape[:2]:
        raise ValueError("selection shape does not match frame shape")
    values = per_pixel[mask]
    if values.size == 0:
        return EMPTY_REGION
    changed = int(np.count_nonzero(values))
    return RegionDiff(
        pixels=int(values.size),
        max_diff=int(values.max()),
        mean_diff=float(values.mean()),
        changed_pixels=changed,
        changed_fraction=float(changed / values.size),
    )


def luma(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame


def mean_abs_frame_delta(
    a: np.ndarray, b: np.ndarray, selection: np.ndarray | None = None
) -> float:
    """Mean absolute luma delta between two frames (optionally masked)."""
    diff = np.abs(luma(a).astype(np.int16) - luma(b).astype(np.int16))
    if selection is not None:
        mask = selection.astype(bool) if selection.dtype != bool else selection
        values = diff[mask]
    else:
        values = diff.reshape(-1)
    return float(values.mean()) if values.size else 0.0


def black_frame_fraction(frame: np.ndarray, threshold: int = 12) -> float:
    """Fraction of pixels whose luma is at or below ``threshold``."""
    values = luma(frame)
    return float(np.count_nonzero(values <= threshold) / values.size) if values.size else 0.0


def boundary_band(mask: np.ndarray, width_px: int = 3) -> np.ndarray:
    """A band of pixels just *outside* the editable region.

    Leakage shows up here first: a renderer that bleeds past the feather edge
    changes these pixels even though the mask says they are immutable.
    """
    if width_px <= 0:
        return np.zeros(mask.shape[:2], dtype=bool)
    size = 2 * width_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    editable = (mask > 0).astype(np.uint8)
    dilated = cv2.dilate(editable, kernel)
    return (dilated > 0) & (editable == 0)


def temporal_deltas(frames: list[np.ndarray], selection: np.ndarray | None = None) -> list[float]:
    """Mean absolute luma delta between consecutive frames."""
    return [
        mean_abs_frame_delta(frames[i], frames[i + 1], selection) for i in range(len(frames) - 1)
    ]


def longest_run(values: list[float], *, at_or_below: float) -> int:
    """Longest consecutive run of values at or below a threshold.

    Used for frozen-frame detection: a long run of near-zero inter-frame delta
    means the video stopped moving.
    """
    best = current = 0
    for value in values:
        if value <= at_or_below:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def flicker_score(deltas: list[float]) -> dict[str, float]:
    """Summarise temporal change inside a region.

    ``max_delta`` catches a single popping frame; ``std`` catches sustained
    jitter that a mean would hide.
    """
    if not deltas:
        return {"mean_delta": 0.0, "max_delta": 0.0, "std_delta": 0.0}
    array = np.asarray(deltas, dtype=np.float64)
    return {
        "mean_delta": float(array.mean()),
        "max_delta": float(array.max()),
        "std_delta": float(array.std()),
    }


def duplicate_frame_indices(hashes: dict[int, str]) -> list[list[int]]:
    """Groups of frame indices whose bytes are identical."""
    buckets: dict[str, list[int]] = {}
    for index, digest in sorted(hashes.items()):
        buckets.setdefault(digest, []).append(index)
    return [indices for indices in buckets.values() if len(indices) > 1]


def bbox_to_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Rasterise ``(x, y, w, h)`` into a boolean mask, clipped to the frame."""
    height, width = shape
    x, y, w, h = bbox
    x0 = max(0, min(width, int(x)))
    y0 = max(0, min(height, int(y)))
    x1 = max(0, min(width, int(x) + int(w)))
    y1 = max(0, min(height, int(y) + int(h)))
    mask = np.zeros(shape, dtype=bool)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return mask


__all__ = [
    "EMPTY_REGION",
    "RegionDiff",
    "bbox_to_mask",
    "black_frame_fraction",
    "boundary_band",
    "duplicate_frame_indices",
    "flicker_score",
    "longest_run",
    "luma",
    "mean_abs_frame_delta",
    "per_pixel_abs_diff",
    "region_diff",
    "temporal_deltas",
]
