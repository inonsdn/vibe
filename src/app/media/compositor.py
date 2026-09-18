"""Compositing: source pixels win everywhere the mask does not allow an edit.

``composite`` is the single place in the system where rendered pixels enter an
output frame. It is written so that the identity-preservation guarantee is a
property of the arithmetic rather than of backend good behaviour:

* where ``mask == 0``   -> output is the **exact** source byte,
* where ``mask == 255`` -> output is the rendered pixel,
* in between            -> a linear blend, rounded half-up.

The hard-assignment of the ``mask == 0`` region happens *after* the blend, so
floating-point rounding can never perturb a protected pixel by ±1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.core.errors import ValidationError
from app.media.masks import EDITABLE, IMMUTABLE


@dataclass
class CompositeStats:
    """Per-frame evidence that the edit stayed inside the mask."""

    changed_pixels: int
    changed_outside_mask: int
    max_diff_outside_mask: int
    max_diff_inside_mask: int
    mean_diff_inside_mask: float
    editable_pixels: int
    extras: dict[str, float] = field(default_factory=dict)

    @property
    def leaked(self) -> bool:
        return self.changed_outside_mask > 0

    def as_dict(self) -> dict[str, object]:
        return {
            "changed_pixels": self.changed_pixels,
            "changed_outside_mask": self.changed_outside_mask,
            "max_diff_outside_mask": self.max_diff_outside_mask,
            "max_diff_inside_mask": self.max_diff_inside_mask,
            "mean_diff_inside_mask": round(self.mean_diff_inside_mask, 6),
            "editable_pixels": self.editable_pixels,
            "leaked": self.leaked,
            **self.extras,
        }


def _validate_inputs(source: np.ndarray, rendered: np.ndarray, mask: np.ndarray) -> None:
    if source.shape != rendered.shape:
        raise ValidationError(
            "Source and rendered frames differ in shape",
            source_shape=list(source.shape),
            rendered_shape=list(rendered.shape),
        )
    if mask.shape[:2] != source.shape[:2]:
        raise ValidationError(
            "Mask shape does not match frame shape",
            mask_shape=list(mask.shape[:2]),
            frame_shape=list(source.shape[:2]),
        )
    if source.dtype != np.uint8 or rendered.dtype != np.uint8:
        raise ValidationError(
            "Frames must be uint8",
            source_dtype=str(source.dtype),
            rendered_dtype=str(rendered.dtype),
        )


def composite(
    source: np.ndarray,
    rendered: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Blend ``rendered`` over ``source`` using ``mask`` as per-pixel opacity."""
    _validate_inputs(source, rendered, mask)

    alpha = mask.astype(np.float32) / float(EDITABLE)
    if source.ndim == 3:
        alpha = alpha[:, :, None]

    blended = source.astype(np.float32) * (1.0 - alpha) + rendered.astype(np.float32) * alpha
    out = np.floor(blended + 0.5).clip(0, 255).astype(np.uint8)

    # Hard restore: every pixel the mask calls immutable is copied byte-for-byte
    # from the source. This is the identity-preservation guarantee.
    immutable = mask <= IMMUTABLE
    if immutable.any():
        if source.ndim == 3:
            out[immutable, :] = source[immutable, :]
        else:
            out[immutable] = source[immutable]
    return out


def composite_with_stats(
    source: np.ndarray,
    rendered: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, CompositeStats]:
    """Composite and measure where pixels actually changed."""
    out = composite(source, rendered, mask)
    diff = np.abs(out.astype(np.int16) - source.astype(np.int16))
    per_pixel = diff.max(axis=2) if diff.ndim == 3 else diff

    allowed = mask > IMMUTABLE
    outside = ~allowed
    changed = per_pixel > 0

    inside_diff = per_pixel[allowed]
    return out, CompositeStats(
        changed_pixels=int(np.count_nonzero(changed)),
        changed_outside_mask=int(np.count_nonzero(changed & outside)),
        max_diff_outside_mask=int(per_pixel[outside].max()) if outside.any() else 0,
        max_diff_inside_mask=int(inside_diff.max()) if inside_diff.size else 0,
        mean_diff_inside_mask=float(inside_diff.mean()) if inside_diff.size else 0.0,
        editable_pixels=int(np.count_nonzero(allowed)),
    )


def restore_protected(
    frame: np.ndarray,
    source: np.ndarray,
    protected_mask: np.ndarray,
) -> np.ndarray:
    """Force every protected pixel back to its source value.

    Used as a belt-and-braces pass after compositing, and by QC to construct
    the "what it should have been" reference frame.
    """
    _validate_inputs(source, frame, protected_mask)
    out = frame.copy()
    selection = protected_mask > IMMUTABLE
    if selection.any():
        if source.ndim == 3:
            out[selection, :] = source[selection, :]
        else:
            out[selection] = source[selection]
    return out


def apply_flash(
    frame: np.ndarray,
    color: tuple[int, int, int],
    opacity: float,
) -> np.ndarray:
    """Blend a flat colour over a whole frame (optional transition flash)."""
    if not 0.0 <= opacity <= 1.0:
        raise ValidationError("Flash opacity must be within [0, 1]", opacity=opacity)
    # Frames are BGR in OpenCV; the configured colour is RGB.
    bgr = np.array(color[::-1], dtype=np.float32)
    overlay = np.empty_like(frame, dtype=np.float32)
    overlay[:, :] = bgr
    blended = frame.astype(np.float32) * (1.0 - opacity) + overlay * opacity
    return np.floor(blended + 0.5).clip(0, 255).astype(np.uint8)


def diff_stats(
    a: np.ndarray,
    b: np.ndarray,
    selection: np.ndarray | None = None,
) -> dict[str, float]:
    """Absolute-difference statistics, optionally restricted to a region."""
    if a.shape != b.shape:
        raise ValidationError(
            "Cannot diff frames of different shape",
            a_shape=list(a.shape),
            b_shape=list(b.shape),
        )
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    per_pixel = diff.max(axis=2) if diff.ndim == 3 else diff
    if selection is not None:
        if selection.shape[:2] != per_pixel.shape[:2]:
            raise ValidationError("Selection shape does not match frame shape")
        values = per_pixel[selection > 0] if selection.dtype != bool else per_pixel[selection]
    else:
        values = per_pixel.reshape(-1)
    if values.size == 0:
        return {"count": 0.0, "max": 0.0, "mean": 0.0, "changed_fraction": 0.0}
    return {
        "count": float(values.size),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "changed_fraction": float(np.count_nonzero(values) / values.size),
    }


__all__ = [
    "CompositeStats",
    "apply_flash",
    "composite",
    "composite_with_stats",
    "diff_stats",
    "restore_protected",
]
