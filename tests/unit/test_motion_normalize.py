"""Canonical normalization: requirements 1 and 2.

1. Normalization makes two different source scales share one canonical scale.
2. Root centres align without destroying relative motion.
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from app.core.errors import ValidationError
from app.motion.normalize import (
    NormalizationSettings,
    check_missing_joint_runs,
    normalize_sequence,
    relative_motion_signature,
    robust_scale,
)
from app.motion.pose_format import PoseSpace
from tests import motion_fixtures as mf

PROFILE = NormalizationSettings(
    target_width=1080,
    target_height=1920,
    target_center=(540.0, 1010.0),
    canonical_shoulder_width=300.0,
    canonical_torso_length=420.0,
)


def normalize(spec, start=0, end=48, **kwargs):
    poses = mf.make_poses(spec, start, end)
    return poses, normalize_sequence(poses, PROFILE, **kwargs)


# ---------------------------------------------------------------------------
# requirement 1: one canonical scale
# ---------------------------------------------------------------------------
def test_two_very_different_source_scales_become_one_canonical_scale() -> None:
    _, a = normalize(mf.MOTION_A)
    _, b = normalize(mf.MOTION_B)

    a_widths = mf.canonical_shoulder_widths(a.poses)
    b_widths = mf.canonical_shoulder_widths(b.poses)

    # Precondition: the sources really are very different.
    assert mf.MOTION_B["shoulder_width"] / mf.MOTION_A["shoulder_width"] > 2.0

    for width in a_widths + b_widths:
        assert width == pytest.approx(PROFILE.canonical_shoulder_width, rel=0.02)

    # And they agree with each other, which is what makes a join possible.
    assert max(a_widths) == pytest.approx(max(b_widths), rel=0.02)


def test_canonical_torso_length_also_converges() -> None:
    _, a = normalize(mf.MOTION_A)
    _, b = normalize(mf.MOTION_B)
    a_torsos = [t for t in (p.torso_length(0.0) for p in a.poses) if t]
    b_torsos = [t for t in (p.torso_length(0.0) for p in b.poses) if t]
    assert sum(a_torsos) / len(a_torsos) == pytest.approx(sum(b_torsos) / len(b_torsos), rel=0.05)


def test_scale_is_derived_from_robust_statistics() -> None:
    poses = mf.make_poses(mf.MOTION_A, 0, 40)
    scale, shoulder, torso = robust_scale(poses, PROFILE)
    assert shoulder == pytest.approx(mf.MOTION_A["shoulder_width"], rel=0.01)
    assert torso == pytest.approx(mf.MOTION_A["torso_length"], rel=0.01)
    assert scale > 1.0  # the small source must be scaled up


def test_a_single_outlier_frame_does_not_move_the_scale() -> None:
    """Median-based statistics, so one bad detection cannot rescale the body."""
    poses = mf.make_poses(mf.MOTION_A, 0, 40)
    clean, _, _ = robust_scale(poses, PROFILE)

    from app.motion.pose_format import Joint2D

    broken = list(poses)
    outlier = broken[20]
    broken[20] = outlier.model_copy(
        update={
            "body": {
                **outlier.body,
                "left_shoulder": Joint2D(x=-900.0, y=0.0, confidence=0.9),
            }
        }
    )
    polluted, _, _ = robust_scale(broken, PROFILE)
    assert polluted == pytest.approx(clean, rel=0.02)


# ---------------------------------------------------------------------------
# requirement 2: aligned roots, preserved relative motion
# ---------------------------------------------------------------------------
def test_root_centers_land_on_the_canonical_target() -> None:
    for spec in (mf.MOTION_A, mf.MOTION_B):
        _, result = normalize(spec)
        centers = [p.torso_center(0.0) for p in result.poses]
        assert all(c is not None for c in centers)
        xs = [c[0] for c in centers if c]
        ys = [c[1] for c in centers if c]
        # Translation is smoothed, so the body tracks the target closely rather
        # than being pinned to it exactly -- travel is motion, not error.
        assert sum(xs) / len(xs) == pytest.approx(PROFILE.target_center[0], abs=25.0)
        assert sum(ys) / len(ys) == pytest.approx(PROFILE.target_center[1], abs=25.0)


def test_both_sources_share_a_root_position_after_normalization() -> None:
    _, a = normalize(mf.MOTION_A)
    _, b = normalize(mf.MOTION_B)
    a_center = a.poses[0].torso_center(0.0)
    b_center = b.poses[0].torso_center(0.0)
    assert a_center and b_center
    assert math.hypot(a_center[0] - b_center[0], a_center[1] - b_center[1]) < 20.0


def test_relative_joint_motion_is_preserved_up_to_the_scale_factor() -> None:
    """Normalization may move and resize the body; it must not change the dance."""
    poses, result = normalize(mf.MOTION_A)
    scale = result.base_scale

    before = relative_motion_signature(poses, "left_wrist")
    after = relative_motion_signature(result.poses, "left_wrist")
    assert len(before) == len(after)

    for (bx, by), (ax, ay) in zip(before, after, strict=True):
        assert ax == pytest.approx(bx * scale, abs=1e-6)
        assert ay == pytest.approx(by * scale, abs=1e-6)


def test_relative_motion_matches_between_differently_scaled_sources() -> None:
    """The same motion at two sizes normalizes to the same relative motion."""
    _, a = normalize(mf.MOTION_A)
    _, b = normalize(mf.MOTION_B)
    a_sig = relative_motion_signature(a.poses, "right_wrist")
    b_sig = relative_motion_signature(b.poses, "right_wrist")
    for (ax, ay), (bx, by) in zip(a_sig, b_sig, strict=True):
        assert ax == pytest.approx(bx, abs=6.0)
        assert ay == pytest.approx(by, abs=6.0)


def test_motion_is_not_flattened_into_a_static_pose() -> None:
    """A normalizer that over-smoothed would pass the tests above and be useless."""
    _, result = normalize(mf.MOTION_A)
    signature = relative_motion_signature(result.poses, "left_wrist")
    xs = [x for x, _ in signature]
    assert max(xs) - min(xs) > 5.0, "the wrist must still move relative to the body"


# ---------------------------------------------------------------------------
# stability and rejection
# ---------------------------------------------------------------------------
def test_scale_does_not_pump_between_frames() -> None:
    _, result = normalize(mf.MOTION_A)
    scales = [t.scale for t in result.transforms]
    steps = [abs(b - a) / a for a, b in pairwise(scales)]
    assert max(steps, default=0.0) == 0.0
    assert result.stats["max_frame_to_frame_scale_step"] == 0.0


def test_output_space_and_origin_are_marked() -> None:
    _, result = normalize(mf.MOTION_A)
    assert all(p.space is PoseSpace.CANONICAL for p in result.poses)
    assert all(p.origin.value == "normalized" for p in result.poses)


def test_source_bbox_is_dropped_in_canonical_space() -> None:
    """A source bounding box describes the source frame and would mislead."""
    poses, result = normalize(mf.MOTION_A)
    assert poses[0].source_bbox is not None
    assert all(p.source_bbox is None for p in result.poses)


def test_short_gaps_are_interpolated() -> None:
    poses = mf.make_poses(mf.MOTION_A, 0, 40)
    poses = mf.drop_joint(poses, "left_hip", range(10, 13))
    poses = mf.drop_joint(poses, "right_hip", range(10, 13))
    result = normalize_sequence(poses, PROFILE)
    assert set(result.interpolated_frames) >= {10, 11, 12}
    assert len(result.poses) == len(poses)


def test_long_gaps_are_rejected_rather_than_invented() -> None:
    poses = mf.make_poses(mf.MOTION_A, 0, 40)
    poses = mf.drop_joint(poses, "left_hip", range(5, 25))
    poses = mf.drop_joint(poses, "right_hip", range(5, 25))
    with pytest.raises(ValidationError, match=r"too many consecutive frames|too long"):
        normalize_sequence(poses, PROFILE)


def test_missing_high_priority_joint_run_is_measured() -> None:
    poses = mf.make_poses(mf.MOTION_A, 0, 30)
    poses = mf.drop_joint(poses, "left_shoulder", range(4, 8))
    runs = check_missing_joint_runs(poses, PROFILE)
    assert runs["left_shoulder"] == 4
    assert runs["right_shoulder"] == 0


def test_empty_sequence_is_rejected() -> None:
    with pytest.raises(ValidationError, match="empty"):
        normalize_sequence([], PROFILE)


def test_a_sequence_with_no_confident_torso_is_rejected() -> None:
    poses = mf.make_poses(mf.MOTION_A, 0, 10)
    poses = mf.blur_confidence(poses, range(0, 10), 0.05)
    with pytest.raises(ValidationError):
        normalize_sequence(poses, PROFILE)


def test_playback_speed_changes_timestamps_not_frame_count() -> None:
    _, normal = normalize(mf.MOTION_A, end=30)
    _, fast = normalize(mf.MOTION_A, end=30, playback_speed=2.0)
    assert len(normal.poses) == len(fast.poses)
    assert fast.poses[10].timestamp_s == pytest.approx(normal.poses[10].timestamp_s / 2.0)
    assert fast.stats["playback_speed"] == 2.0


def test_negative_playback_speed_is_rejected() -> None:
    poses = mf.make_poses(mf.MOTION_A, 0, 10)
    with pytest.raises(ValidationError, match="playback_speed"):
        normalize_sequence(poses, PROFILE, playback_speed=0.0)


def test_transforms_are_recorded_per_frame() -> None:
    _, result = normalize(mf.MOTION_A, end=25)
    assert len(result.transforms) == len(result.poses)
    first = result.transforms[0].as_dict()
    assert {"frame_index", "scale", "offset_x", "offset_y", "interpolated"} <= set(first)
