"""Compositing: requirements 3 and 4.

3. A render changes only the allowed garment pixels.
4. Protected pixels remain exactly equal to the source.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.errors import ValidationError
from app.media.compositor import (
    apply_flash,
    composite,
    composite_with_stats,
    diff_stats,
    restore_protected,
)

SHAPE = (48, 32)


def rng_frame(seed: int, shape: tuple[int, int] = SHAPE) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (*shape, 3), dtype=np.uint8)


def mask_rect(y0: int, y1: int, x0: int, x1: int, value: int = 255) -> np.ndarray:
    mask = np.zeros(SHAPE, dtype=np.uint8)
    mask[y0:y1, x0:x1] = value
    return mask


def test_zero_mask_output_is_byte_identical_to_source() -> None:
    source, rendered = rng_frame(1), rng_frame(2)
    out = composite(source, rendered, np.zeros(SHAPE, dtype=np.uint8))
    assert np.array_equal(out, source)


def test_full_mask_output_equals_rendered() -> None:
    source, rendered = rng_frame(1), rng_frame(2)
    out = composite(source, rendered, np.full(SHAPE, 255, dtype=np.uint8))
    assert np.array_equal(out, rendered)


def test_only_masked_pixels_change() -> None:
    source, rendered = rng_frame(3), rng_frame(4)
    mask = mask_rect(10, 20, 5, 15)
    out, stats = composite_with_stats(source, rendered, mask)

    outside = mask == 0
    assert np.array_equal(out[outside], source[outside])
    assert stats.changed_outside_mask == 0
    assert stats.max_diff_outside_mask == 0
    assert not stats.leaked
    assert stats.editable_pixels == int((mask > 0).sum())


def test_feathered_boundary_blends_but_never_leaks() -> None:
    source, rendered = rng_frame(5), rng_frame(6)
    mask = mask_rect(10, 20, 5, 15)
    mask[20:23, 5:15] = 128  # feather band
    out, stats = composite_with_stats(source, rendered, mask)

    assert stats.changed_outside_mask == 0
    band = out[20:23, 5:15].astype(np.int16)
    src_band = source[20:23, 5:15].astype(np.int16)
    ren_band = rendered[20:23, 5:15].astype(np.int16)
    # Each blended pixel lies between its source and rendered value.
    lower = np.minimum(src_band, ren_band)
    upper = np.maximum(src_band, ren_band)
    assert (band >= lower).all() and (band <= upper).all()


def test_protected_pixels_survive_a_hostile_backend() -> None:
    """Even a backend that returns garbage cannot touch protected pixels."""
    source = rng_frame(7)
    hostile = np.full_like(source, 255)  # a backend painting the entire frame
    protected = mask_rect(0, 15, 0, 32)
    editable = mask_rect(20, 40, 5, 25)
    # The effective mask already excludes protected pixels.
    out, stats = composite_with_stats(source, hostile, editable)
    assert np.array_equal(out[protected > 0], source[protected > 0])
    assert stats.changed_outside_mask == 0


def test_restore_protected_forces_source_pixels_back() -> None:
    source = rng_frame(8)
    tampered = np.full_like(source, 7)
    protected = mask_rect(5, 25, 5, 25)
    restored = restore_protected(tampered, source, protected)
    assert np.array_equal(restored[protected > 0], source[protected > 0])
    assert np.array_equal(restored[protected == 0], tampered[protected == 0])


def test_rounding_is_half_up_and_stable() -> None:
    source = np.zeros((2, 2, 3), dtype=np.uint8)
    rendered = np.full((2, 2, 3), 255, dtype=np.uint8)
    mask = np.full((2, 2), 128, dtype=np.uint8)
    out = composite(source, rendered, mask)
    # 255 * (128/255) = 128 exactly.
    assert (out == 128).all()
    assert np.array_equal(out, composite(source, rendered, mask))


def test_grayscale_frames_are_supported() -> None:
    source = np.zeros(SHAPE, dtype=np.uint8)
    rendered = np.full(SHAPE, 200, dtype=np.uint8)
    mask = mask_rect(5, 10, 5, 10)
    out = composite(source, rendered, mask)
    assert out[7, 7] == 200
    assert out[0, 0] == 0


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValidationError):
        composite(rng_frame(1), rng_frame(1, (16, 16)), np.zeros(SHAPE, dtype=np.uint8))


def test_mask_shape_mismatch_raises() -> None:
    with pytest.raises(ValidationError):
        composite(rng_frame(1), rng_frame(2), np.zeros((8, 8), dtype=np.uint8))


def test_non_uint8_raises() -> None:
    source = rng_frame(1).astype(np.float32)
    with pytest.raises(ValidationError):
        composite(source, rng_frame(2), np.zeros(SHAPE, dtype=np.uint8))


def test_apply_flash_blends_toward_the_configured_colour() -> None:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    flashed = apply_flash(frame, (255, 255, 255), 1.0)
    assert (flashed == 255).all()
    half = apply_flash(frame, (255, 255, 255), 0.5)
    assert (half == 128).all()


def test_apply_flash_rejects_out_of_range_opacity() -> None:
    with pytest.raises(ValidationError):
        apply_flash(np.zeros((2, 2, 3), dtype=np.uint8), (255, 255, 255), 1.5)


def test_diff_stats_restricted_to_a_selection() -> None:
    a = np.zeros(SHAPE, dtype=np.uint8)
    b = a.copy()
    b[5, 5] = 40
    selection = mask_rect(0, 10, 0, 10)
    stats = diff_stats(a, b, selection)
    assert stats["max"] == 40
    outside = diff_stats(a, b, mask_rect(20, 30, 20, 30))
    assert outside["max"] == 0
