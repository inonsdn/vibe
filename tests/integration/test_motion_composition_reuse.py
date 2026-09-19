"""Reusing one motion source, and the output-space metadata a join carries.

**P1-3.** Normalized poses were written to ``normalized_poses/<source_id>/``.
Using the same reference clip twice — two different ranges of the same
performance, which is an ordinary thing to want — made the second segment
overwrite the first, silently. Segments are now keyed by position as well as
source.

**P1-4.** A join knew where its bridge sat in *source* numbering but not in the
composed output, so nothing downstream could say "start the garment reveal
here". Each join now records its bridge range and a recommended transition
anchor in output frame numbering, which is what ``master promote`` consumes.
"""

from __future__ import annotations

import json

import pytest

from app.core.errors import ValidationError
from app.domain.motion import segment_dirname
from app.motion.pose_format import list_pose_indices, poses_equal
from app.pipeline.master_promote import resolve_transition_anchor
from app.pipeline.motion_compose import ComposeOptions, JoinSpec, SegmentSpec, compose_motion
from app.qc.motion_checks import run_motion_qc
from tests import motion_fixtures as mf


@pytest.fixture
def single_source(context):
    return mf.register_motion_source(
        context,
        motion_id="mot_reused",
        spec=mf.MOTION_A,
        start=0,
        end=90,
        display_name="One long reference clip",
    )


# ---------------------------------------------------------------------------
# P1-3: the same source, twice
# ---------------------------------------------------------------------------
@pytest.fixture
def reused(context, single_source):
    return compose_motion(
        context,
        ComposeOptions(
            display_name="Same clip, two ranges",
            segments=[
                SegmentSpec(
                    motion_source_id=single_source.source.id,
                    start_frame=0,
                    end_frame=30,
                    exposed_views=["front"],
                ),
                SegmentSpec(
                    motion_source_id=single_source.source.id,
                    start_frame=50,
                    end_frame=90,
                    exposed_views=["front"],
                ),
            ],
            joins=[JoinSpec(bridge_frames=12)],
            composition_id="cmp_reused",
            make_preview=False,
        ),
    )


def test_two_segments_from_one_source_keep_separate_pose_directories(
    context, reused, single_source
) -> None:
    composition = reused.composition
    assert len(composition.segments) == 2
    first, second = composition.segments

    assert first.motion_source_id == second.motion_source_id == single_source.source.id
    assert first.segment_index == 0 and second.segment_index == 1
    assert first.pose_dirname != second.pose_dirname
    assert first.pose_dirname == segment_dirname(0, single_source.source.id)
    assert second.pose_dirname == segment_dirname(1, single_source.source.id)

    root = context.absolute(composition.normalized_pose_dir)
    first_dir = root / first.pose_dirname
    second_dir = root / second.pose_dirname
    assert first_dir.is_dir() and second_dir.is_dir()

    # Each directory holds its own range, in the source clip's numbering.
    assert list_pose_indices(first_dir) == list(range(0, 30))
    assert list_pose_indices(second_dir) == list(range(50, 90))


def test_the_second_use_does_not_overwrite_the_first(context, reused) -> None:
    """The regression itself: distinct frames, not one range written twice."""
    from app.motion.pose_format import load_pose_sequence

    composition = reused.composition
    root = context.absolute(composition.normalized_pose_dir)
    first = load_pose_sequence(root / composition.segments[0].pose_dirname)
    second = load_pose_sequence(root / composition.segments[1].pose_dirname)

    assert len(first) == 30
    assert len(second) == 40
    assert {p.frame_index for p in first}.isdisjoint({p.frame_index for p in second})


def test_the_manifest_records_both_segments_separately(context, reused) -> None:
    composition = reused.composition
    manifest = json.loads(context.absolute(composition.manifest_path).read_text())

    segments = manifest["segments"]
    assert len(segments) == 2
    assert [s["segment_index"] for s in segments] == [0, 1]
    assert len({s["pose_dir"] for s in segments}) == 2
    assert [s["effective_range"] for s in segments] == [[0, 30], [50, 90]]
    # Each segment keeps its own canonical transform in the manifest ...
    assert all(s["canonical_transform"] for s in segments)

    # ... and in the normalization summary, which used to be keyed by source id
    # and so collapsed to a single entry when one clip was used twice.
    summary = reused.normalization
    assert len(summary) == 2, "each segment's transform must survive on its own"
    assert set(summary) == {s["pose_dir"] for s in segments}


def test_a_reused_source_still_passes_motion_qc(context, reused) -> None:
    report = run_motion_qc(context, reused.composition.id)
    assert report.passed, [c.check_id for c in report.failed_checks]


# ---------------------------------------------------------------------------
# P1-4: output-space join metadata
# ---------------------------------------------------------------------------
def test_joins_record_their_bridge_in_output_numbering(reused) -> None:
    composition = reused.composition
    assert len(composition.joins) == 1
    join = composition.joins[0]

    assert join.output_bridge_start is not None
    assert join.output_bridge_end is not None
    assert join.output_bridge_end - join.output_bridge_start == join.bridge_frame_count
    assert join.output_bridge_range == (join.output_bridge_start, join.output_bridge_end)
    # The recommended anchor is where the borrowed opening ends.
    assert join.recommended_transition_anchor == join.output_bridge_start
    assert 0 < join.recommended_transition_anchor < composition.output_frame_count
    # The source-side anchors are kept too: the operator can trace both.
    assert join.prev_source_frame is not None and join.next_source_frame is not None


def test_the_manifest_carries_the_recommended_anchor(context, reused) -> None:
    manifest = json.loads(context.absolute(reused.composition.manifest_path).read_text())
    stored = reused.composition.joins[0]
    join = manifest["joins"][0]
    assert join["recommended_transition_anchor"] == stored.recommended_transition_anchor
    assert join["output_bridge_start"] == stored.output_bridge_start
    assert join["output_bridge_end"] == stored.output_bridge_end
    # The digest covers them, so a changed anchor changes the digest.
    assert manifest["digest"] != ""


def test_one_join_needs_no_operator_confirmation(reused) -> None:
    anchor, source = resolve_transition_anchor(
        reused.composition, explicit=None, confirm_multiple_joins=False
    )
    assert source == "composition_join"
    assert anchor == reused.composition.joins[0].recommended_transition_anchor


def test_several_joins_require_an_explicit_choice(context, single_source) -> None:
    """Two seams, two defensible reveals — the operator picks, not the code."""
    composition = compose_motion(
        context,
        ComposeOptions(
            display_name="Three segments",
            segments=[
                SegmentSpec(
                    motion_source_id=single_source.source.id,
                    start_frame=start,
                    end_frame=end,
                    exposed_views=["front"],
                )
                for start, end in ((0, 30), (30, 60), (60, 90))
            ],
            joins=[JoinSpec(bridge_frames=12), JoinSpec(bridge_frames=12)],
            composition_id="cmp_three",
            make_preview=False,
        ),
    ).composition
    assert len(composition.joins) == 2

    with pytest.raises(ValidationError, match="more than one join"):
        resolve_transition_anchor(composition, explicit=None, confirm_multiple_joins=False)

    anchor, source = resolve_transition_anchor(
        composition, explicit=None, confirm_multiple_joins=True
    )
    assert source == "composition_join"
    assert anchor == composition.joins[0].recommended_transition_anchor

    explicit, source = resolve_transition_anchor(
        composition, explicit=7, confirm_multiple_joins=False
    )
    assert (explicit, source) == (7, "explicit")


def test_poses_equal_helper_is_used_for_the_disjointness_claim(context, reused) -> None:
    """Sanity: the two segments really do hold different body poses, not just
    different file names."""
    from app.motion.pose_format import load_pose_sequence

    root = context.absolute(reused.composition.normalized_pose_dir)
    first = load_pose_sequence(root / reused.composition.segments[0].pose_dirname)
    second = load_pose_sequence(root / reused.composition.segments[1].pose_dirname)
    assert not poses_equal(first[0], second[0])


# ---------------------------------------------------------------------------
# the invariant behind the recorded geometry
# ---------------------------------------------------------------------------
def _join(**overrides):
    from app.domain.motion import MotionJoin

    fields = {
        "prev_segment_index": 0,
        "next_segment_index": 1,
        "prev_source_frame": 20,
        "next_source_frame": 5,
        "pose_distance_score": 0.1,
        "root_position_delta": 0.0,
        "shoulder_scale_delta": 0.0,
        "torso_angle_delta": 0.0,
        "head_angle_delta": 0.0,
        "hand_position_delta": 0.0,
        "incoming_velocity_delta": 0.0,
        "outgoing_velocity_delta": 0.0,
        "bridge_frame_count": 12,
    }
    fields.update(overrides)
    return MotionJoin(**fields)


def test_a_coherent_output_bridge_is_accepted() -> None:
    join = _join(output_bridge_start=100, output_bridge_end=112)
    assert join.output_bridge_range == (100, 112)


@pytest.mark.parametrize(
    "overrides",
    [
        {"output_bridge_start": 100},  # end missing
        {"output_bridge_end": 112},  # start missing
        {"output_bridge_start": 112, "output_bridge_end": 100},  # inverted
        {"output_bridge_start": 100, "output_bridge_end": 105},  # wrong span
    ],
)
def test_an_incoherent_output_bridge_is_rejected(overrides) -> None:
    """The span must equal bridge_frame_count, or the recommended anchor is a
    number with no frames behind it."""
    from pydantic import ValidationError as PydanticValidationError

    with pytest.raises(PydanticValidationError):
        _join(**overrides)
