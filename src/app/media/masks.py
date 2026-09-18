"""Mask semantics, combination and validation.

Grayscale conventions (single channel, uint8) — see ``docs/mask-semantics.md``:

============  =========================================================
Value         Meaning
============  =========================================================
``0``         immutable: output pixel **must** equal the source pixel
``255``       editable: the backend may replace this pixel entirely
``1..254``    feather / blend boundary: linear blend source <-> render
============  =========================================================

The *effective* editable mask for a frame is built in a fixed order:

1. ``editable = max(garment, expansion)`` (expansion widens the garment region
   for silhouettes larger than the base outfit),
2. ``occlusion`` is subtracted — hair, hands or props in front of the garment
   must keep their original pixels,
3. ``protected`` is subtracted — face, hair, hands, exposed skin, background,
4. feathering is applied **inside** the allowed region only, so a feathered
   edge can never reach into protected pixels,
5. the result is clamped: anything protected stays exactly ``0``.

Step 5 is applied last and unconditionally (unless an explicit reviewed
override is passed), which is what makes "protected always wins" true even for
overlapping or sloppily authored masks.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.core.errors import MaskError
from app.core.logging import get_logger
from app.domain.enums import MaskKind

logger = get_logger(__name__)

IMMUTABLE = 0
EDITABLE = 255


# ---------------------------------------------------------------------------
# basic IO
# ---------------------------------------------------------------------------
def load_mask(
    path: str | os.PathLike[str], *, expect_shape: tuple[int, int] | None = None
) -> np.ndarray:
    """Load a single-channel uint8 mask."""
    target = Path(path)
    mask = cv2.imread(str(target), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise MaskError("Mask could not be read", path=str(target))
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if expect_shape is not None and mask.shape[:2] != expect_shape:
        raise MaskError(
            "Mask dimensions do not match the frame",
            path=str(target),
            mask_shape=list(mask.shape[:2]),
            expected_shape=list(expect_shape),
        )
    return mask


def save_mask(path: str | os.PathLike[str], mask: np.ndarray) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = as_mask(mask)
    if not cv2.imwrite(str(target), data, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
        raise MaskError("Failed to write mask", path=str(target))
    return target


def as_mask(array: np.ndarray) -> np.ndarray:
    """Coerce an array to a single-channel uint8 mask."""
    data = array
    if data.ndim == 3:
        data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    if data.dtype != np.uint8:
        data = np.clip(data, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(data)


def empty_mask(shape: tuple[int, int]) -> np.ndarray:
    """A fully immutable mask: nothing may be edited."""
    return np.zeros(shape, dtype=np.uint8)


def full_mask(shape: tuple[int, int]) -> np.ndarray:
    return np.full(shape, EDITABLE, dtype=np.uint8)


# ---------------------------------------------------------------------------
# morphology helpers
# ---------------------------------------------------------------------------
def dilate(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return mask
    size = 2 * radius_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask, kernel)


def erode(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return mask
    size = 2 * radius_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.erode(mask, kernel)


def feather(mask: np.ndarray, radius_px: int, sigma: float = 0.0) -> np.ndarray:
    """Soften a mask's edges *inward*.

    The blur is applied to an eroded copy and then clamped by the original
    mask, so the feathered result is always a subset of the input. A feather
    can therefore never grow into a neighbouring protected region.
    """
    if radius_px <= 0:
        return mask
    kernel = 2 * radius_px + 1
    shrunk = erode(mask, max(1, radius_px // 2))
    blurred = cv2.GaussianBlur(shrunk, (kernel, kernel), sigma if sigma > 0 else 0)
    return np.minimum(blurred, mask)


def binarize(mask: np.ndarray, threshold: int = 127) -> np.ndarray:
    return ((mask > threshold) * EDITABLE).astype(np.uint8)


def invert(mask: np.ndarray) -> np.ndarray:
    return (EDITABLE - mask.astype(np.int16)).clip(0, 255).astype(np.uint8)


def subtract(base: np.ndarray, removal: np.ndarray) -> np.ndarray:
    """``base`` minus ``removal``, saturating at 0 (per-pixel opacity math)."""
    return np.clip(base.astype(np.int16) - removal.astype(np.int16), 0, 255).astype(np.uint8)


def union(*masks: np.ndarray) -> np.ndarray:
    if not masks:
        raise MaskError("union() requires at least one mask")
    out = masks[0].copy()
    for mask in masks[1:]:
        out = np.maximum(out, mask)
    return out


def intersect(*masks: np.ndarray) -> np.ndarray:
    if not masks:
        raise MaskError("intersect() requires at least one mask")
    out = masks[0].copy()
    for mask in masks[1:]:
        out = np.minimum(out, mask)
    return out


def area_fraction(mask: np.ndarray, *, threshold: int = 0) -> float:
    """Fraction of the frame whose mask value exceeds ``threshold``."""
    total = mask.size
    return float(np.count_nonzero(mask > threshold) / total) if total else 0.0


def weighted_area_fraction(mask: np.ndarray) -> float:
    """Opacity-weighted editable area (feather counts partially)."""
    return float(mask.astype(np.float64).sum() / (mask.size * EDITABLE)) if mask.size else 0.0


# ---------------------------------------------------------------------------
# effective mask construction
# ---------------------------------------------------------------------------
@dataclass
class MaskSet:
    """The four masks for one frame. Only ``garment`` is required."""

    garment: np.ndarray
    expansion: np.ndarray | None = None
    protected: np.ndarray | None = None
    occlusion: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.garment.shape[:2]

    def validate_shapes(self) -> None:
        for name in ("expansion", "protected", "occlusion"):
            mask = getattr(self, name)
            if mask is not None and mask.shape[:2] != self.shape:
                raise MaskError(
                    f"{name} mask shape does not match garment mask",
                    mask=name,
                    shape=list(mask.shape[:2]),
                    expected=list(self.shape),
                )

    def get(self, kind: MaskKind) -> np.ndarray | None:
        return {
            MaskKind.GARMENT: self.garment,
            MaskKind.EXPANSION: self.expansion,
            MaskKind.PROTECTED: self.protected,
            MaskKind.OCCLUSION: self.occlusion,
        }[kind]


@dataclass
class MaskConflict:
    """A region where an editable mask overlaps a protected/occluded region."""

    kind: str
    overlap_pixels: int
    overlap_fraction: float
    resolved_by: str

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "overlap_pixels": self.overlap_pixels,
            "overlap_fraction": round(self.overlap_fraction, 8),
            "resolved_by": self.resolved_by,
        }


@dataclass
class EffectiveMask:
    """The single mask the compositor actually uses, plus its audit trail."""

    mask: np.ndarray
    editable_fraction: float
    weighted_fraction: float
    conflicts: list[MaskConflict] = field(default_factory=list)
    protected_applied: bool = True
    feather_radius_px: int = 0

    @property
    def shape(self) -> tuple[int, int]:
        return self.mask.shape[:2]

    def as_dict(self) -> dict[str, object]:
        return {
            "editable_fraction": round(self.editable_fraction, 8),
            "weighted_fraction": round(self.weighted_fraction, 8),
            "protected_applied": self.protected_applied,
            "feather_radius_px": self.feather_radius_px,
            "conflicts": [c.as_dict() for c in self.conflicts],
        }


def build_effective_mask(
    masks: MaskSet,
    *,
    feather_radius_px: int = 9,
    feather_sigma: float = 0.0,
    expansion_dilate_px: int = 0,
    protected_dilate_px: int = 2,
    protected_wins: bool = True,
    override_protected: bool = False,
) -> EffectiveMask:
    """Combine the mask set into the one mask the compositor trusts.

    ``override_protected`` is the *only* way to let an editable mask win over a
    protected one, and callers must have a stored reviewer override to pass it.
    """
    masks.validate_shapes()
    conflicts: list[MaskConflict] = []
    total = float(masks.garment.size) or 1.0

    editable = as_mask(masks.garment).copy()
    if masks.expansion is not None:
        expansion = as_mask(masks.expansion)
        if expansion_dilate_px:
            expansion = dilate(expansion, expansion_dilate_px)
        editable = union(editable, expansion)

    if masks.occlusion is not None:
        occlusion = as_mask(masks.occlusion)
        overlap = int(np.count_nonzero((editable > 0) & (occlusion > 0)))
        if overlap:
            conflicts.append(
                MaskConflict(
                    kind="editable_over_occlusion",
                    overlap_pixels=overlap,
                    overlap_fraction=overlap / total,
                    resolved_by="occlusion",
                )
            )
        editable = subtract(editable, occlusion)

    protected_applied = False
    protected_hard: np.ndarray | None = None
    if masks.protected is not None:
        protected = as_mask(masks.protected)
        if protected_dilate_px:
            protected = dilate(protected, protected_dilate_px)
        overlap = int(np.count_nonzero((editable > 0) & (protected > 0)))
        if overlap:
            conflicts.append(
                MaskConflict(
                    kind="editable_over_protected",
                    overlap_pixels=overlap,
                    overlap_fraction=overlap / total,
                    resolved_by="override" if override_protected else "protected",
                )
            )
        if protected_wins and not override_protected:
            editable = subtract(editable, protected)
            protected_hard = protected
            protected_applied = True

    if feather_radius_px:
        editable = feather(editable, feather_radius_px, feather_sigma)

    # Final clamp: feathering and blurring cannot resurrect protected pixels.
    if protected_hard is not None:
        editable[protected_hard > 0] = IMMUTABLE

    return EffectiveMask(
        mask=editable,
        editable_fraction=area_fraction(editable),
        weighted_fraction=weighted_area_fraction(editable),
        conflicts=conflicts,
        protected_applied=protected_applied,
        feather_radius_px=feather_radius_px,
    )


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
@dataclass
class MaskValidation:
    ok: bool
    problems: list[str] = field(default_factory=list)
    editable_fraction: float = 0.0
    conflicts: list[MaskConflict] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "problems": self.problems,
            "editable_fraction": round(self.editable_fraction, 8),
            "conflicts": [c.as_dict() for c in self.conflicts],
        }


def validate_mask_set(
    masks: MaskSet,
    *,
    frame_shape: tuple[int, int] | None = None,
    min_editable_fraction: float = 0.0005,
    max_editable_fraction: float = 0.60,
    feather_radius_px: int = 9,
    protected_dilate_px: int = 2,
) -> MaskValidation:
    """Structural checks that must pass before a frame is rendered."""
    problems: list[str] = []
    if frame_shape is not None and masks.shape != frame_shape:
        problems.append(
            f"garment mask shape {masks.shape} does not match frame shape {frame_shape}"
        )
    try:
        masks.validate_shapes()
    except MaskError as exc:
        problems.append(str(exc))
        return MaskValidation(ok=False, problems=problems)

    effective = build_effective_mask(
        masks,
        feather_radius_px=feather_radius_px,
        protected_dilate_px=protected_dilate_px,
    )
    fraction = effective.editable_fraction
    if fraction < min_editable_fraction:
        problems.append(
            f"editable region is too small after protection ({fraction:.6f} "
            f"< {min_editable_fraction}); the garment mask may be empty or "
            "entirely covered by protected pixels"
        )
    if fraction > max_editable_fraction:
        problems.append(
            f"editable region is implausibly large ({fraction:.6f} > "
            f"{max_editable_fraction}); check that protected masks are present"
        )
    if masks.protected is None:
        problems.append("no protected mask supplied; face/hair/skin cannot be guaranteed")

    return MaskValidation(
        ok=not problems,
        problems=problems,
        editable_fraction=fraction,
        conflicts=effective.conflicts,
    )


def load_mask_set(
    directories: dict[MaskKind, str | os.PathLike[str] | None],
    frame_index: int,
    *,
    template: str = "frame_{index:06d}.png",
    expect_shape: tuple[int, int] | None = None,
    require: Iterable[MaskKind] = (MaskKind.GARMENT,),
) -> MaskSet:
    """Load one frame's mask set from the template's mask directories."""
    loaded: dict[MaskKind, np.ndarray | None] = {}
    filename = template.format(index=frame_index)
    required = set(require)
    for kind in MaskKind:
        directory = directories.get(kind)
        if directory is None:
            loaded[kind] = None
            continue
        path = Path(directory) / filename
        if not path.is_file():
            if kind in required:
                raise MaskError(
                    f"Required {kind.value} mask is missing",
                    frame_index=frame_index,
                    path=str(path),
                )
            loaded[kind] = None
            continue
        loaded[kind] = load_mask(path, expect_shape=expect_shape)

    garment = loaded[MaskKind.GARMENT]
    if garment is None:
        raise MaskError("Garment mask directory not configured", frame_index=frame_index)
    return MaskSet(
        garment=garment,
        expansion=loaded[MaskKind.EXPANSION],
        protected=loaded[MaskKind.PROTECTED],
        occlusion=loaded[MaskKind.OCCLUSION],
    )


__all__ = [
    "EDITABLE",
    "IMMUTABLE",
    "EffectiveMask",
    "MaskConflict",
    "MaskSet",
    "MaskValidation",
    "area_fraction",
    "as_mask",
    "binarize",
    "build_effective_mask",
    "dilate",
    "empty_mask",
    "erode",
    "feather",
    "full_mask",
    "intersect",
    "invert",
    "load_mask",
    "load_mask_set",
    "save_mask",
    "subtract",
    "union",
    "validate_mask_set",
    "weighted_area_fraction",
]
