"""Mask semantics: requirement 5 — protected masks override garment masks."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.errors import MaskError
from app.domain.enums import MaskKind
from app.media.frames import frame_path
from app.media.masks import (
    EDITABLE,
    IMMUTABLE,
    MaskSet,
    area_fraction,
    build_effective_mask,
    dilate,
    empty_mask,
    feather,
    load_mask_set,
    save_mask,
    subtract,
    union,
    validate_mask_set,
)

SHAPE = (64, 48)


def rect(
    shape: tuple[int, int], y0: int, y1: int, x0: int, x1: int, value: int = 255
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[y0:y1, x0:x1] = value
    return mask


def test_protected_beats_garment_everywhere_they_overlap() -> None:
    garment = rect(SHAPE, 10, 50, 5, 40)
    protected = rect(SHAPE, 20, 30, 10, 20)
    effective = build_effective_mask(
        MaskSet(garment=garment, protected=protected),
        feather_radius_px=0,
        protected_dilate_px=0,
    )
    assert (effective.mask[20:30, 10:20] == IMMUTABLE).all()
    assert effective.protected_applied
    assert any(c.kind == "editable_over_protected" for c in effective.conflicts)


def test_protected_wins_even_after_feathering() -> None:
    """Feathering must never re-open a protected pixel."""
    garment = rect(SHAPE, 0, 64, 0, 48)
    protected = rect(SHAPE, 30, 40, 20, 30)
    effective = build_effective_mask(
        MaskSet(garment=garment, protected=protected),
        feather_radius_px=9,
        protected_dilate_px=2,
    )
    assert (effective.mask[30:40, 20:30] == IMMUTABLE).all()


def test_protected_dilation_adds_a_safety_margin() -> None:
    garment = rect(SHAPE, 0, 64, 0, 48)
    protected = rect(SHAPE, 30, 34, 20, 24)
    effective = build_effective_mask(
        MaskSet(garment=garment, protected=protected),
        feather_radius_px=0,
        protected_dilate_px=3,
    )
    # The margin around the protected rectangle is also immutable.
    assert effective.mask[29, 20] == IMMUTABLE
    assert effective.mask[34, 23] == IMMUTABLE


def test_override_is_required_to_edit_protected_pixels() -> None:
    garment = rect(SHAPE, 10, 50, 5, 40)
    protected = rect(SHAPE, 20, 30, 10, 20)
    overridden = build_effective_mask(
        MaskSet(garment=garment, protected=protected),
        feather_radius_px=0,
        protected_dilate_px=0,
        override_protected=True,
    )
    assert (overridden.mask[20:30, 10:20] == EDITABLE).all()
    assert not overridden.protected_applied
    assert any(c.resolved_by == "override" for c in overridden.conflicts)


def test_occlusion_is_subtracted_from_the_editable_region() -> None:
    garment = rect(SHAPE, 10, 50, 5, 40)
    occlusion = rect(SHAPE, 15, 20, 8, 12)
    effective = build_effective_mask(
        MaskSet(garment=garment, occlusion=occlusion),
        feather_radius_px=0,
        protected_dilate_px=0,
    )
    assert (effective.mask[15:20, 8:12] == IMMUTABLE).all()
    assert any(c.kind == "editable_over_occlusion" for c in effective.conflicts)


def test_expansion_widens_the_editable_region() -> None:
    garment = rect(SHAPE, 20, 30, 20, 30)
    expansion = rect(SHAPE, 15, 35, 15, 35)
    effective = build_effective_mask(
        MaskSet(garment=garment, expansion=expansion),
        feather_radius_px=0,
        protected_dilate_px=0,
    )
    assert effective.mask[16, 16] == EDITABLE
    assert effective.editable_fraction > area_fraction(garment)


def test_feather_never_grows_the_mask() -> None:
    mask = rect(SHAPE, 20, 40, 10, 30)
    feathered = feather(mask, 5)
    assert (feathered <= mask).all()


def test_feather_produces_intermediate_values() -> None:
    mask = rect(SHAPE, 15, 50, 8, 40)
    feathered = feather(mask, 5)
    intermediate = feathered[(feathered > 0) & (feathered < 255)]
    assert intermediate.size > 0, "a feathered edge must contain blend values"


def test_shape_mismatch_is_rejected() -> None:
    garment = np.zeros((10, 10), dtype=np.uint8)
    protected = np.zeros((12, 10), dtype=np.uint8)
    with pytest.raises(MaskError):
        build_effective_mask(MaskSet(garment=garment, protected=protected))


def test_validate_mask_set_flags_missing_protected_mask() -> None:
    garment = rect(SHAPE, 10, 50, 5, 40)
    validation = validate_mask_set(MaskSet(garment=garment), feather_radius_px=0)
    assert not validation.ok
    assert any("protected mask" in problem for problem in validation.problems)


def test_validate_mask_set_flags_fully_protected_garment() -> None:
    garment = rect(SHAPE, 20, 30, 10, 20)
    protected = rect(SHAPE, 0, 64, 0, 48)
    validation = validate_mask_set(
        MaskSet(garment=garment, protected=protected), feather_radius_px=0
    )
    assert not validation.ok
    assert any("too small" in problem for problem in validation.problems)


def test_validate_mask_set_flags_implausibly_large_region() -> None:
    validation = validate_mask_set(
        MaskSet(garment=np.full(SHAPE, 255, dtype=np.uint8), protected=empty_mask(SHAPE)),
        feather_radius_px=0,
        max_editable_fraction=0.5,
    )
    assert not validation.ok
    assert any("implausibly large" in problem for problem in validation.problems)


def test_mask_set_loads_from_directories(tmp_path) -> None:
    directories = {}
    for kind in MaskKind:
        directory = tmp_path / kind.value
        directory.mkdir()
        directories[kind] = directory
        save_mask(frame_path(directory, 7), rect(SHAPE, 5, 15, 5, 15))
    loaded = load_mask_set(directories, 7, expect_shape=SHAPE)
    assert loaded.garment.shape == SHAPE
    assert loaded.protected is not None and loaded.occlusion is not None


def test_missing_required_mask_raises(tmp_path) -> None:
    garment_dir = tmp_path / "garment"
    garment_dir.mkdir()
    with pytest.raises(MaskError):
        load_mask_set({MaskKind.GARMENT: garment_dir}, 3, expect_shape=SHAPE)


def test_mask_dimension_mismatch_on_load_raises(tmp_path) -> None:
    garment_dir = tmp_path / "garment"
    garment_dir.mkdir()
    save_mask(frame_path(garment_dir, 0), np.zeros((8, 8), dtype=np.uint8))
    with pytest.raises(MaskError):
        load_mask_set({MaskKind.GARMENT: garment_dir}, 0, expect_shape=SHAPE)


def test_set_algebra_helpers() -> None:
    a = rect(SHAPE, 0, 10, 0, 10)
    b = rect(SHAPE, 5, 15, 5, 15)
    assert union(a, b)[12, 12] == EDITABLE
    assert subtract(a, b)[7, 7] == IMMUTABLE
    assert dilate(rect(SHAPE, 20, 21, 20, 21), 2)[19, 20] == EDITABLE
