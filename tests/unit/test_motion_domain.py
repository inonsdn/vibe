"""Motion and master domain schemas, and the pose format contract."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError as PydanticValidationError

from app.core.errors import ValidationError
from app.domain.human_template import FrameRange, VideoSpec
from app.domain.master import (
    ChunkRecord,
    HeroCharacter,
    MasterAcceptance,
    MasterCandidate,
    MasterCandidateStatus,
    MasterOrigin,
)
from app.domain.motion import (
    CanonicalSkeletonProfile,
    CanonicalTransform,
    MotionComposition,
    MotionJoin,
    MotionSegment,
    MotionSource,
    TransitionType,
)
from app.motion.pose_format import (
    POSE_SCHEMA_VERSION,
    Joint2D,
    PoseFrame,
    load_pose_frame,
    load_pose_sequence,
    poses_equal,
    save_pose_frame,
    save_pose_sequence,
    summarize_bbox,
)
from tests import motion_fixtures as mf


def video() -> VideoSpec:
    return VideoSpec(width=720, height=1280, fps=30.0, duration_s=2.0, frame_count=60)


def transform() -> CanonicalTransform:
    return CanonicalTransform(
        base_scale=2.0, source_shoulder_width=100.0, source_torso_length=150.0
    )


def segment(**overrides) -> MotionSegment:
    payload = {
        "motion_source_id": "mot_a",
        "motion_source_version": 1,
        "source_range": FrameRange(start=0, end=40),
        "canonical_transform": transform(),
    }
    payload.update(overrides)
    return MotionSegment(**payload)


def join(**overrides) -> MotionJoin:
    payload = {
        "prev_segment_index": 0,
        "next_segment_index": 1,
        "prev_source_frame": 30,
        "next_source_frame": 5,
        "pose_distance_score": 0.02,
        "root_position_delta": 1.0,
        "shoulder_scale_delta": 0.001,
        "torso_angle_delta": 0.5,
        "head_angle_delta": 0.3,
        "hand_position_delta": 2.0,
        "incoming_velocity_delta": 0.1,
        "outgoing_velocity_delta": 0.1,
        "bridge_frame_count": 12,
    }
    payload.update(overrides)
    return MotionJoin(**payload)


# ---------------------------------------------------------------------------
# pose format
# ---------------------------------------------------------------------------
def test_pose_rejects_joints_outside_the_vocabulary() -> None:
    with pytest.raises(PydanticValidationError, match="vocabulary"):
        PoseFrame(frame_index=0, timestamp_s=0.0, body={"tail": Joint2D(x=1, y=1)})


def test_pose_geometry_helpers() -> None:
    pose = mf.make_poses(mf.MOTION_A, 0, 1)[0]
    assert pose.shoulder_width() == pytest.approx(mf.MOTION_A["shoulder_width"])
    assert pose.torso_length() == pytest.approx(mf.MOTION_A["torso_length"], rel=0.01)
    assert pose.torso_center() == pytest.approx(mf.MOTION_A["center"], rel=0.01)
    assert pose.torso_angle_deg() == pytest.approx(0.0, abs=0.01)


def test_confidence_threshold_hides_a_joint() -> None:
    pose = PoseFrame(
        frame_index=0,
        timestamp_s=0.0,
        body={"nose": Joint2D(x=1.0, y=2.0, confidence=0.2)},
    )
    assert pose.joint("nose") is not None
    assert pose.confident_joint("nose", 0.35) is None
    assert pose.missing_joints(["nose"], 0.35) == ["nose"]


def test_transform_preserves_confidence_and_membership() -> None:
    pose = mf.make_poses(mf.MOTION_A, 0, 1)[0]
    moved = pose.transformed(scale=2.0, offset=(10.0, 20.0))
    assert set(moved.body) == set(pose.body)
    for name, joint_ in pose.body.items():
        assert moved.body[name].x == pytest.approx(joint_.x * 2.0 + 10.0)
        assert moved.body[name].confidence == joint_.confidence


def test_pose_round_trips_through_disk(tmp_path) -> None:
    pose = mf.make_poses(mf.MOTION_A, 7, 8)[0]
    path = tmp_path / "frame_000007.json"
    save_pose_frame(path, pose)
    loaded = load_pose_frame(path)
    assert poses_equal(loaded, pose, tolerance=0.0)
    assert loaded.frame_index == 7


def test_unknown_pose_schema_version_is_rejected(tmp_path) -> None:
    path = tmp_path / "frame_000000.json"
    path.write_text(
        '{"schema_version": "99", "frame_index": 0, "timestamp_s": 0}', encoding="utf-8"
    )
    with pytest.raises(ValidationError, match="schema version"):
        load_pose_frame(path)


def test_malformed_pose_file_is_rejected(tmp_path) -> None:
    path = tmp_path / "frame_000000.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(ValidationError, match="valid JSON"):
        load_pose_frame(path)


def test_missing_frame_in_a_sequence_is_reported(tmp_path) -> None:
    save_pose_sequence(tmp_path, mf.make_poses(mf.MOTION_A, 0, 3))
    with pytest.raises(ValidationError, match="missing from sequence"):
        load_pose_sequence(tmp_path, [0, 1, 2, 3])


def test_poses_equal_ignores_index_and_timestamp() -> None:
    pose = mf.make_poses(mf.MOTION_A, 0, 1)[0]
    moved = pose.with_index(99, 3.3)
    assert poses_equal(pose, moved, tolerance=0.0)


def test_schema_version_is_recorded() -> None:
    pose = mf.make_poses(mf.MOTION_A, 0, 1)[0]
    assert pose.schema_version == POSE_SCHEMA_VERSION
    assert pose.model_dump(mode="json")["schema_version"] == POSE_SCHEMA_VERSION


def test_bbox_summary() -> None:
    poses = mf.make_poses(mf.MOTION_A, 0, 10)
    summary = summarize_bbox(poses)
    assert summary["frames_with_bbox"] == 10
    assert summary["width"]["median"] > 0


# ---------------------------------------------------------------------------
# canonical profile
# ---------------------------------------------------------------------------
def test_profile_rejects_a_body_center_outside_the_frame() -> None:
    with pytest.raises(PydanticValidationError, match="target_body_center"):
        CanonicalSkeletonProfile(
            canonical_shoulder_width=300.0,
            canonical_torso_length=420.0,
            target_width=1080,
            target_height=1920,
            target_body_center=(5000.0, 100.0),
            target_head_position=(540.0, 430.0),
            target_head_scale=96.0,
        )


def test_profile_rejects_inverted_confidence_thresholds() -> None:
    with pytest.raises(PydanticValidationError, match="confidence_threshold"):
        CanonicalSkeletonProfile(
            canonical_shoulder_width=300.0,
            canonical_torso_length=420.0,
            target_width=1080,
            target_height=1920,
            target_body_center=(540.0, 1010.0),
            target_head_position=(540.0, 430.0),
            target_head_scale=96.0,
            confidence_threshold=0.9,
            high_confidence_threshold=0.3,
        )


def test_shipped_profile_loads() -> None:
    import yaml

    from app.core.config import REPO_ROOT

    payload = yaml.safe_load(
        (REPO_ROOT / "config" / "canonical_skeleton.v1.yaml").read_text(encoding="utf-8")
    )
    profile = CanonicalSkeletonProfile.model_validate(payload)
    assert profile.target_width == 1080
    assert profile.target_height == 1920
    assert profile.id == "canonical_v1"


# ---------------------------------------------------------------------------
# motion source / segment / join
# ---------------------------------------------------------------------------
def test_motion_source_range_must_fit_the_video() -> None:
    with pytest.raises(PydanticValidationError, match="exceeds"):
        MotionSource(
            id="mot_a",
            display_name="A",
            source_video_path="a.mp4",
            source_sha256="0" * 64,
            video=video(),
            selected_range=FrameRange(start=0, end=61),
            pose_dir="motion_sources/mot_a/pose",
        )


def test_segment_trim_cannot_remove_everything() -> None:
    with pytest.raises(PydanticValidationError, match="whole segment"):
        segment(trim_start=20, trim_end=20)


def test_segment_effective_range_applies_the_trim() -> None:
    trimmed = segment(trim_start=3, trim_end=5)
    assert trimmed.effective_range.start == 3
    assert trimmed.effective_range.end == 35
    assert trimmed.frame_count == 32


def test_join_must_connect_adjacent_segments() -> None:
    with pytest.raises(PydanticValidationError, match="adjacent"):
        join(prev_segment_index=0, next_segment_index=2)


def test_pose_bridge_needs_both_endpoints() -> None:
    with pytest.raises(PydanticValidationError, match="at least 2 frames"):
        join(bridge_frame_count=1)


def test_direct_cut_must_not_declare_bridge_frames() -> None:
    with pytest.raises(PydanticValidationError, match="must not declare"):
        join(transition_type=TransitionType.DIRECT_CUT, bridge_frame_count=12)


def test_direct_cut_with_zero_frames_is_valid() -> None:
    cut = join(transition_type=TransitionType.DIRECT_CUT, bridge_frame_count=0)
    assert cut.transition_type is TransitionType.DIRECT_CUT


# ---------------------------------------------------------------------------
# composition arithmetic
# ---------------------------------------------------------------------------
def composition(**overrides) -> MotionComposition:
    payload = {
        "id": "cmp_x",
        "display_name": "X",
        "segments": [
            segment(source_range=FrameRange(start=0, end=40)),
            segment(motion_source_id="mot_b", source_range=FrameRange(start=0, end=40)),
        ],
        "joins": [join()],
        "output_fps": 30.0,
        "normalized_pose_dir": "compositions/cmp_x/normalized_poses",
        "bridge_pose_dir": "compositions/cmp_x/bridge_poses",
        "composed_pose_dir": "compositions/cmp_x/composed_poses",
    }
    payload.update(overrides)
    return MotionComposition(**payload)


def test_composition_requires_one_join_per_gap() -> None:
    with pytest.raises(PydanticValidationError, match="exactly 1 join"):
        composition(joins=[])


def test_composition_frame_count_arithmetic() -> None:
    """Segment A stops before its anchor; the bridge carries both anchors."""
    comp = composition()
    # A contributes [0, 30) = 30; bridge = 12; B contributes [6, 40) = 34.
    assert comp.expected_frame_count() == 30 + 12 + 34


def test_single_segment_composition_needs_no_joins() -> None:
    comp = composition(segments=[segment()], joins=[])
    assert comp.expected_frame_count() == 40


def test_direct_cut_contributes_no_bridge_frames() -> None:
    comp = composition(
        joins=[join(transition_type=TransitionType.DIRECT_CUT, bridge_frame_count=0)]
    )
    assert comp.expected_frame_count() == 30 + 0 + 34


# ---------------------------------------------------------------------------
# hero and master candidate
# ---------------------------------------------------------------------------
def test_hero_requires_a_reference_image() -> None:
    with pytest.raises(PydanticValidationError, match="reference image"):
        HeroCharacter(id="hero_a", display_name="A", reference_images=[])


def test_consented_hero_requires_a_document() -> None:
    with pytest.raises(PydanticValidationError, match="consent_document_ref"):
        HeroCharacter(
            id="hero_a",
            display_name="A",
            reference_images=["heroes/hero_a/images/front.png"],
            subject_kind="consented_human",
        )


def test_hero_adult_confirmation_is_mandatory() -> None:
    with pytest.raises(PydanticValidationError, match="adult_confirmed"):
        HeroCharacter(
            id="hero_a",
            display_name="A",
            reference_images=["x.png"],
            adult_confirmed=False,
        )


def candidate(**overrides) -> MasterCandidate:
    payload = {
        "id": "mst_a",
        "display_name": "A",
        "composition_id": "cmp_x",
        "composition_version": 1,
        "hero_character_id": "hero_a",
        "hero_character_version": 1,
        "backend_name": "mock",
        "seed": 1,
        "width": 1080,
        "height": 1920,
        "fps": 30.0,
        "frames_dir": "masters/mst_a/frames",
        "frame_count": 40,
    }
    payload.update(overrides)
    return MasterCandidate(**payload)


def test_accepted_status_requires_an_acceptance_record() -> None:
    with pytest.raises(PydanticValidationError, match="acceptance record"):
        candidate(status=MasterCandidateStatus.ACCEPTED)


def test_acceptance_record_is_meaningless_without_a_decision() -> None:
    acceptance = MasterAcceptance(
        accepted_by="op",
        accepted_at=datetime.now(UTC),
        reason="reviewed and approved for production",
    )
    with pytest.raises(PydanticValidationError, match="meaningless"):
        candidate(status=MasterCandidateStatus.ANIMATED, acceptance=acceptance)


def test_acceptance_timestamp_must_be_tz_aware() -> None:
    with pytest.raises(PydanticValidationError, match="timezone-aware"):
        MasterAcceptance(
            accepted_by="op",
            accepted_at=datetime(2026, 1, 1),
            reason="reviewed and approved for production",
        )


def test_candidate_tracks_remaining_frames() -> None:
    record = candidate(
        chunks=[
            ChunkRecord(index=0, start_frame=0, end_frame=24, seed=1),
            ChunkRecord(index=1, start_frame=24, end_frame=40, seed=2),
        ]
    )
    assert record.remaining_frames() == []
    partial = candidate(chunks=[ChunkRecord(index=0, start_frame=0, end_frame=24, seed=1)])
    assert partial.remaining_frames() == list(range(24, 40))


def test_chunk_record_rejects_an_inverted_range() -> None:
    with pytest.raises(PydanticValidationError, match="end_frame"):
        ChunkRecord(index=0, start_frame=10, end_frame=10, seed=1)


def test_master_origin_values() -> None:
    assert MasterOrigin.CAPTURED.value == "captured_master"
    assert MasterOrigin.SYNTHETIC.value == "synthetic_master"


def test_candidate_status_classification() -> None:
    assert MasterCandidateStatus.ACCEPTED.is_terminal
    assert MasterCandidateStatus.FAILED.is_resumable
    assert not MasterCandidateStatus.ACCEPTED.is_resumable
