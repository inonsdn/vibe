"""Anchor matching and bridge generation: requirements 3, 4 and 5.

3. The anchor matcher ranks the known compatible pair first.
4. Bridge endpoints exactly match both anchors.
5. The bridge has no scale or limb-length discontinuity.
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from app.core.errors import ValidationError
from app.motion.anchors import (
    AnchorSearchSettings,
    rank_anchor_candidates,
    score_pair,
    select_anchor,
)
from app.motion.bridge import (
    MAX_BRIDGE_FRAMES,
    MIN_BRIDGE_FRAMES,
    BridgeSettings,
    bridge_metrics,
    generate_bridge,
    hermite,
)
from app.motion.normalize import NormalizationSettings, normalize_sequence
from app.motion.pose_format import PoseOrigin, poses_equal
from app.motion.skeleton import LIMB_EDGES
from tests import motion_fixtures as mf

PROFILE = NormalizationSettings(
    target_width=1080,
    target_height=1920,
    target_center=(540.0, 1010.0),
    canonical_shoulder_width=300.0,
    canonical_torso_length=420.0,
)


def normalized(spec, start, end, **kwargs):
    return normalize_sequence(mf.make_poses(spec, start, end, **kwargs), PROFILE).poses


# ---------------------------------------------------------------------------
# requirement 3: the matcher finds the known compatible pair
# ---------------------------------------------------------------------------
def test_matcher_ranks_the_known_compatible_pair_first() -> None:
    """A genuinely identical pose exists in the windows; the matcher must find it.

    Motion B is time-shifted by KNOWN_ANCHOR_OFFSET, so B[f] is the same body
    pose as A[f + offset]. Both are normalized to the same canonical scale, so a
    perfect match is available and every other pairing is worse.
    """
    offset = mf.KNOWN_ANCHOR_OFFSET
    prev = normalized(mf.MOTION_A, 0, 60)
    nxt = normalized(mf.MOTION_B, 0, 60, frame_offset=offset)

    ranked = rank_anchor_candidates(
        prev, nxt, AnchorSearchSettings(prev_window=16, next_window=16, max_candidates=50)
    )
    best = ranked[0]

    # The correct pairing: a previous frame p matches next frame p - offset.
    assert best.prev_frame - best.next_frame == offset, best.as_dict()
    assert best.acceptable
    assert all(best.score <= candidate.score for candidate in ranked)

    # The residual is small but not zero: translation smoothing is applied to
    # each source independently, so two identical body poses at different
    # absolute positions land a couple of pixels apart out of a 300px shoulder
    # width. That is the normalizer working, not the matcher failing.
    assert best.body_distance < 0.02 * 300.0, best.as_dict()


def test_the_whole_top_of_the_ranking_is_near_the_correct_offset() -> None:
    """Not just the winner: the ordering as a whole must be meaningful.

    Exact separation of "correct" from "incorrect" is not the right claim for
    continuous motion — a frame one step either side of the true match really is
    almost the same pose, and a matcher that pretended otherwise would be
    overfitting. What must hold is that the top of the ranking clusters tightly
    around the true offset and that distant pairings score worse.
    """
    offset = mf.KNOWN_ANCHOR_OFFSET
    prev = normalized(mf.MOTION_A, 0, 60)
    nxt = normalized(mf.MOTION_B, 0, 60, frame_offset=offset)
    ranked = rank_anchor_candidates(
        prev, nxt, AnchorSearchSettings(prev_window=16, next_window=16, max_candidates=400)
    )
    top = ranked[:5]
    assert all(abs((c.prev_frame - c.next_frame) - offset) <= 1 for c in top), [
        (c.prev_frame, c.next_frame, round(c.score, 5)) for c in top
    ]

    far = [c for c in ranked if abs((c.prev_frame - c.next_frame) - offset) >= 8]
    assert far, "the search window should contain clearly-wrong pairings"
    assert max(c.score for c in top) < min(c.score for c in far)


def test_matching_phases_score_far_better_than_mismatched_ones() -> None:
    prev = normalized(mf.MOTION_A, 40, 60)
    same = normalized(mf.MOTION_B, 40, 60, frame_offset=0)
    shifted = normalized(mf.MOTION_B, 40, 60, phase_offset=math.pi)

    settings = AnchorSearchSettings(prev_window=20, next_window=20)
    good = rank_anchor_candidates(prev, same, settings)[0]
    bad = rank_anchor_candidates(prev, shifted, settings)[0]
    assert good.score < bad.score


def test_ranking_is_deterministic() -> None:
    prev = normalized(mf.MOTION_A, 0, 40)
    nxt = normalized(mf.MOTION_B, 0, 40)
    settings = AnchorSearchSettings(prev_window=12, next_window=12)
    first = [
        (c.prev_frame, c.next_frame, round(c.score, 9))
        for c in rank_anchor_candidates(prev, nxt, settings)
    ]
    second = [
        (c.prev_frame, c.next_frame, round(c.score, 9))
        for c in rank_anchor_candidates(prev, nxt, settings)
    ]
    assert first == second


def test_search_windows_are_respected() -> None:
    prev = normalized(mf.MOTION_A, 0, 60)
    nxt = normalized(mf.MOTION_B, 0, 60)
    settings = AnchorSearchSettings(prev_window=5, next_window=4, max_candidates=100)
    ranked = rank_anchor_candidates(prev, nxt, settings)
    # The previous segment is searched from its END, the next from its START.
    assert all(c.prev_frame >= prev[-5].frame_index for c in ranked)
    assert all(c.next_frame <= nxt[3].frame_index for c in ranked)
    assert len(ranked) == 20


def test_candidate_records_every_score_term() -> None:
    prev = normalized(mf.MOTION_A, 0, 30)
    nxt = normalized(mf.MOTION_B, 0, 30)
    candidate = rank_anchor_candidates(prev, nxt)[0].as_dict()
    for key in (
        "body_distance",
        "hand_distance",
        "torso_angle_delta",
        "head_angle_delta",
        "root_delta",
        "shoulder_scale_delta",
        "velocity_delta",
        "mean_confidence",
    ):
        assert key in candidate


def test_low_confidence_poses_are_penalised() -> None:
    """A guessed skeleton must not win on accident.

    Confidence is lowered AFTER normalization: normalization legitimately
    refuses a sequence this uncertain, and the point here is the anchor score.
    """
    prev = normalized(mf.MOTION_A, 0, 30)
    nxt = mf.blur_confidence(normalized(mf.MOTION_B, 0, 30), range(0, 30), 0.1)
    candidate = score_pair(prev, 29, nxt, 0, AnchorSearchSettings())
    assert candidate.score > 1.0
    assert not candidate.acceptable
    assert any("confidence" in w for w in candidate.warnings)


def test_operator_override_is_scored_like_any_other_choice() -> None:
    prev = normalized(mf.MOTION_A, 0, 40)
    nxt = normalized(mf.MOTION_B, 0, 40)
    chosen = select_anchor(
        prev, nxt, override_prev_frame=prev[10].frame_index, override_next_frame=nxt[3].frame_index
    )
    assert chosen.prev_frame == prev[10].frame_index
    assert chosen.next_frame == nxt[3].frame_index
    assert "operator override" in chosen.warnings
    assert chosen.score >= 0.0


def test_partial_override_is_rejected() -> None:
    prev = normalized(mf.MOTION_A, 0, 20)
    nxt = normalized(mf.MOTION_B, 0, 20)
    with pytest.raises(ValidationError, match="both frames"):
        select_anchor(prev, nxt, override_prev_frame=5)


def test_override_of_a_frame_outside_the_segment_is_rejected() -> None:
    prev = normalized(mf.MOTION_A, 0, 20)
    nxt = normalized(mf.MOTION_B, 0, 20)
    with pytest.raises(ValidationError, match="not part of"):
        select_anchor(prev, nxt, override_prev_frame=999, override_next_frame=3)


def test_empty_segments_are_rejected() -> None:
    with pytest.raises(ValidationError, match="need poses"):
        rank_anchor_candidates([], normalized(mf.MOTION_B, 0, 5))


# ---------------------------------------------------------------------------
# requirement 4: exact endpoints
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("frame_count", [MIN_BRIDGE_FRAMES, 11, MAX_BRIDGE_FRAMES])
def test_bridge_endpoints_exactly_match_both_anchors(frame_count: int) -> None:
    prev = normalized(mf.MOTION_A, 0, 40)
    nxt = normalized(mf.MOTION_B, 0, 40)
    result = generate_bridge(
        prev, 30, nxt, 6, BridgeSettings(frame_count=frame_count), start_output_index=100
    )

    assert result.frame_count == frame_count
    # Exact, not approximate: tolerance 0.
    assert poses_equal(result.poses[0], prev[30], tolerance=0.0)
    assert poses_equal(result.poses[-1], nxt[6], tolerance=0.0)

    for name, joint in prev[30].body.items():
        assert result.poses[0].body[name].x == joint.x
        assert result.poses[0].body[name].y == joint.y
        assert result.poses[0].body[name].confidence == joint.confidence


def test_bridge_frames_are_marked_as_bridge_origin() -> None:
    prev = normalized(mf.MOTION_A, 0, 30)
    nxt = normalized(mf.MOTION_B, 0, 30)
    result = generate_bridge(prev, 25, nxt, 4, BridgeSettings(frame_count=10))
    assert all(p.origin is PoseOrigin.BRIDGE for p in result.poses)


def test_bridge_output_indices_are_contiguous() -> None:
    prev = normalized(mf.MOTION_A, 0, 30)
    nxt = normalized(mf.MOTION_B, 0, 30)
    result = generate_bridge(
        prev, 25, nxt, 4, BridgeSettings(frame_count=12), start_output_index=57
    )
    assert [p.frame_index for p in result.poses] == list(range(57, 69))


def test_interior_frames_actually_interpolate() -> None:
    """A bridge that just held the first pose would pass the endpoint test."""
    prev = normalized(mf.MOTION_A, 0, 40)
    nxt = normalized(mf.MOTION_B, 0, 40)
    result = generate_bridge(prev, 30, nxt, 6, BridgeSettings(frame_count=12))
    middle = result.poses[6]
    assert not poses_equal(middle, result.poses[0], tolerance=1e-6)
    assert not poses_equal(middle, result.poses[-1], tolerance=1e-6)


# ---------------------------------------------------------------------------
# requirement 5: no discontinuity
# ---------------------------------------------------------------------------
def test_bridge_has_no_scale_discontinuity() -> None:
    prev = normalized(mf.MOTION_A, 0, 40)
    nxt = normalized(mf.MOTION_B, 0, 40)
    result = generate_bridge(prev, 30, nxt, 6, BridgeSettings(frame_count=12))
    assert result.metrics["max_scale_drift"] <= 0.02


def test_bridge_has_no_limb_length_discontinuity() -> None:
    prev = normalized(mf.MOTION_A, 0, 40)
    nxt = normalized(mf.MOTION_B, 0, 40)
    result = generate_bridge(prev, 30, nxt, 6, BridgeSettings(frame_count=12))
    assert result.metrics["max_limb_length_drift"] <= 0.08

    # Measured independently of the implementation's own metrics.
    for a_name, b_name in LIMB_EDGES:
        lengths = [
            p.joint(a_name).distance_to(p.joint(b_name))
            for p in result.poses
            if p.joint(a_name) and p.joint(b_name)
        ]
        if len(lengths) < 2 or lengths[0] <= 1e-6:
            continue
        drift = max(abs(length - lengths[0]) / lengths[0] for length in lengths)
        assert drift <= 0.08, f"{a_name}->{b_name} drifted {drift:.3%}"


def test_bridge_keeps_body_center_stable() -> None:
    prev = normalized(mf.MOTION_A, 0, 40)
    nxt = normalized(mf.MOTION_B, 0, 40)
    result = generate_bridge(prev, 30, nxt, 6, BridgeSettings(frame_count=12))
    centers = [p.torso_center(0.0) for p in result.poses]
    xs = [c[0] for c in centers if c]
    ys = [c[1] for c in centers if c]
    assert max(xs) - min(xs) < 40.0
    assert max(ys) - min(ys) < 40.0


def test_bridge_joint_velocity_is_continuous() -> None:
    prev = normalized(mf.MOTION_A, 0, 40)
    nxt = normalized(mf.MOTION_B, 0, 40)
    result = generate_bridge(prev, 30, nxt, 6, BridgeSettings(frame_count=12))
    centers = [p.torso_center(0.0) for p in result.poses]
    velocities = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in pairwise(centers) if a and b]
    steps = [abs(b - a) for a, b in pairwise(velocities)]
    assert max(steps, default=0.0) < 24.0


def test_bones_follow_the_interpolated_length_even_for_mismatched_anchors() -> None:
    """Mismatched anchors are absorbed by the correction, not emitted as stretch.

    Two anchors can legitimately have different apparent bone lengths (a real
    detector foreshortens). The bridge's job is to move smoothly between them
    without inventing intermediate lengths of its own, and that is what is
    asserted here: every emitted bone sits on the interpolation of the two
    anchors' lengths.
    """
    from app.motion.pose_format import Joint2D

    prev = normalized(mf.MOTION_A, 0, 30)
    nxt_poses = mf.make_poses(mf.MOTION_B, 0, 30)

    # Give the next anchor a visibly shorter forearm, as foreshortening would.
    broken = nxt_poses[0]
    body = dict(broken.body)
    elbow = body["left_elbow"]
    body["left_wrist"] = Joint2D(x=elbow.x + 12.0, y=elbow.y + 12.0, confidence=0.9)
    nxt_poses[0] = broken.model_copy(update={"body": body})
    nxt = normalize_sequence(nxt_poses, PROFILE).poses

    result = generate_bridge(prev, 25, nxt, 0, BridgeSettings(frame_count=10))
    count = result.frame_count
    for a_name, b_name in LIMB_EDGES:
        lengths = [
            p.joint(a_name).distance_to(p.joint(b_name))
            for p in result.poses
            if p.joint(a_name) and p.joint(b_name)
        ]
        if len(lengths) != count or lengths[0] <= 1e-6:
            continue
        for index, length in enumerate(lengths):
            expected = lengths[0] + (lengths[-1] - lengths[0]) * (index / (count - 1))
            assert length == pytest.approx(expected, rel=0.02), f"{a_name}->{b_name}"


def test_a_bridge_that_deformed_the_body_would_be_refused() -> None:
    """The guard is live: bypassing the correction must fail the bridge."""
    import app.motion.bridge as bridge_module

    prev = normalized(mf.MOTION_A, 0, 30)
    nxt = normalized(mf.MOTION_B, 0, 30, frame_offset=mf.KNOWN_ANCHOR_OFFSET)

    original = bridge_module.correct_bone_lengths
    bridge_module.correct_bone_lengths = lambda body, a, b, t: body
    try:
        # With the correction disabled, the raw Hermite curves pull bones apart
        # and the limb-length guard must catch it.
        with pytest.raises(ValidationError, match=r"limb lengths|body scale"):
            generate_bridge(
                prev,
                25,
                nxt,
                4,
                BridgeSettings(frame_count=10, max_limb_length_drift=0.001),
            )
    finally:
        bridge_module.correct_bone_lengths = original


# ---------------------------------------------------------------------------
# settings and validation
# ---------------------------------------------------------------------------
def test_prototype_bridge_range_is_enforced_by_default() -> None:
    prev = normalized(mf.MOTION_A, 0, 20)
    nxt = normalized(mf.MOTION_B, 0, 20)
    for count in (2, 5, 9, 13, 30):
        with pytest.raises(ValidationError, match="prototype range"):
            generate_bridge(prev, 15, nxt, 3, BridgeSettings(frame_count=count))


def test_prototype_range_can_be_disabled_explicitly() -> None:
    prev = normalized(mf.MOTION_A, 0, 20)
    nxt = normalized(mf.MOTION_B, 0, 20)
    result = generate_bridge(
        prev,
        15,
        nxt,
        3,
        BridgeSettings(frame_count=6, enforce_prototype_range=False),
    )
    assert result.frame_count == 6


def test_unknown_easing_is_rejected() -> None:
    with pytest.raises(ValidationError, match="easing"):
        BridgeSettings(frame_count=12, easing="bouncy").validate()


@pytest.mark.parametrize("easing", ["linear", "smoothstep", "ease_in_out"])
def test_every_easing_keeps_the_endpoints_exact(easing: str) -> None:
    prev = normalized(mf.MOTION_A, 0, 30)
    nxt = normalized(mf.MOTION_B, 0, 30)
    result = generate_bridge(prev, 25, nxt, 4, BridgeSettings(frame_count=12, easing=easing))
    assert poses_equal(result.poses[0], prev[25], tolerance=0.0)
    assert poses_equal(result.poses[-1], nxt[4], tolerance=0.0)


def test_hermite_basis_matches_its_endpoints() -> None:
    assert hermite(0.0, 10.0, 1.0, 1.0, 0.0) == pytest.approx(0.0)
    assert hermite(0.0, 10.0, 1.0, 1.0, 1.0) == pytest.approx(10.0)


def test_bridge_settings_are_recorded_for_the_manifest() -> None:
    prev = normalized(mf.MOTION_A, 0, 30)
    nxt = normalized(mf.MOTION_B, 0, 30)
    result = generate_bridge(prev, 25, nxt, 4, BridgeSettings(frame_count=11))
    assert result.settings["interpolation"] == "cubic_hermite"
    assert result.settings["frame_count"] == 11
    assert result.settings["anchor_prev_frame"] == prev[25].frame_index
    assert result.settings["anchor_next_frame"] == nxt[4].frame_index
    assert result.settings["joints"]


def test_bridge_metrics_of_a_static_bridge_are_zero() -> None:
    prev = normalized(mf.MOTION_A, 0, 20)
    assert bridge_metrics([prev[0], prev[0], prev[0]])["max_scale_drift"] == 0.0
