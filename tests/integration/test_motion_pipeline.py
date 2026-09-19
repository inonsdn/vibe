"""Motion composition and synthetic master, end to end.

Covers requirements:

6.  composition frame count has no off-by-one error
7.  original source pixels are never copied into motion artifacts
8.  resume skips completed motion/master chunks
9.  identical inputs and settings give identical mock output
10. manifests contain all source hashes, transforms, joins and bridge settings
11. operator acceptance is required before a synthetic master is usable
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.errors import ConflictError, ValidationError
from app.domain.master import MasterCandidateStatus
from app.domain.motion import MotionCompositionStatus, TransitionType
from app.media.frames import frame_path, frames_equal, list_frame_indices
from app.motion.pose_format import list_pose_indices, load_pose_sequence, poses_equal
from app.pipeline.master_create import (
    MasterCreateOptions,
    accept_master,
    animate_master,
    create_master_candidate,
    plan_chunks,
    reject_master,
    resume_master,
    write_master_manifest,
)
from app.pipeline.motion_compose import ComposeOptions, JoinSpec, SegmentSpec, compose_motion
from app.qc.motion_checks import run_master_qc, run_motion_qc
from tests import motion_fixtures as mf
from tests.conftest import requires_ffmpeg


@pytest.fixture
def motion_pair(context):
    return mf.make_motion_pair(context, a_range=(0, 60), b_range=(0, 60))


@pytest.fixture
def composition(context, motion_pair):
    a, b = motion_pair
    return compose_motion(
        context,
        ComposeOptions(
            display_name="Test composition",
            segments=[
                SegmentSpec(motion_source_id=a.source.id, exposed_views=["front"]),
                SegmentSpec(motion_source_id=b.source.id, exposed_views=["front"]),
            ],
            joins=[JoinSpec(bridge_frames=12)],
            composition_id="cmp_test",
            make_preview=False,
        ),
    )


# ---------------------------------------------------------------------------
# requirement 6: no off-by-one in the frame count
# ---------------------------------------------------------------------------
def test_composition_frame_count_has_no_off_by_one(context, composition) -> None:
    result = composition
    comp = result.composition

    # Three independent sources of truth must agree.
    assert comp.output_frame_count == len(result.poses)
    assert comp.output_frame_count == comp.expected_frame_count()

    composed_dir = context.absolute(comp.composed_pose_dir)
    on_disk = list_pose_indices(composed_dir)
    assert on_disk == list(range(comp.output_frame_count))

    # And the arithmetic is the one documented: each segment stops before its
    # anchor, the bridge carries both anchors, the next segment starts after.
    join = comp.joins[0]
    seg_a, seg_b = comp.segments
    expected = (
        (join.prev_source_frame - seg_a.effective_range.start)
        + join.bridge_frame_count
        + (seg_b.effective_range.end - join.next_source_frame - 1)
    )
    assert comp.output_frame_count == expected


def test_every_output_frame_has_exactly_one_origin(context, composition) -> None:
    """No frame is contributed twice, and none is missing."""
    poses = composition.poses
    indices = [p.frame_index for p in poses]
    assert indices == sorted(indices)
    assert len(set(indices)) == len(indices)
    assert indices == list(range(len(indices)))

    origins = [p.origin.value for p in poses]
    assert set(origins) == {"normalized", "bridge"}
    bridge_count = origins.count("bridge")
    assert bridge_count == composition.composition.joins[0].bridge_frame_count


def test_bridge_sits_between_the_segments_without_gap_or_overlap(context, composition) -> None:
    poses = composition.poses
    bridge_indices = [p.frame_index for p in poses if p.origin.value == "bridge"]
    assert bridge_indices == list(range(bridge_indices[0], bridge_indices[-1] + 1))

    before = poses[bridge_indices[0] - 1]
    after = poses[bridge_indices[-1] + 1]
    assert before.origin.value == "normalized"
    assert after.origin.value == "normalized"


def test_bridge_endpoints_equal_the_selected_anchors(context, composition) -> None:
    """Requirement 4, verified through the whole composition path."""
    comp = composition.composition
    join = comp.joins[0]
    poses = composition.poses
    bridge = [p for p in poses if p.origin.value == "bridge"]

    normalized_root = context.absolute(comp.normalized_pose_dir)
    prev_anchor = load_pose_sequence(
        normalized_root / comp.segments[0].pose_dirname, [join.prev_source_frame]
    )[0]
    next_anchor = load_pose_sequence(
        normalized_root / comp.segments[1].pose_dirname, [join.next_source_frame]
    )[0]

    assert poses_equal(bridge[0], prev_anchor, tolerance=0.0)
    assert poses_equal(bridge[-1], next_anchor, tolerance=0.0)


def test_composition_status_and_directories(context, composition) -> None:
    comp = composition.composition
    assert comp.status is MotionCompositionStatus.READY
    for relative in (comp.normalized_pose_dir, comp.bridge_pose_dir, comp.composed_pose_dir):
        assert context.absolute(relative).is_dir()
    assert comp.joins[0].transition_type is TransitionType.POSE_BRIDGE


# ---------------------------------------------------------------------------
# requirement 7: no source pixels in motion artifacts
# ---------------------------------------------------------------------------
def test_no_source_pixels_are_copied_into_motion_artifacts(context, composition) -> None:
    comp = composition.composition
    image_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".mp4", ".mov", ".avi"}

    for relative in (comp.normalized_pose_dir, comp.bridge_pose_dir, comp.composed_pose_dir):
        directory = context.absolute(relative)
        files = [p for p in directory.rglob("*") if p.is_file()]
        assert files, f"{relative} should not be empty"
        for path in files:
            assert path.suffix == ".json", f"non-pose file in motion artifacts: {path}"
            assert path.suffix.lower() not in image_suffixes


def test_motion_ingestion_extracts_no_frames(context, motion_pair) -> None:
    """The cheapest guarantee: the pixels are never copied in the first place."""
    for fixture in motion_pair:
        root = context.absolute(fixture.source.pose_dir).parent
        files = [p for p in root.rglob("*") if p.is_file()]
        suffixes = {p.suffix.lower() for p in files}
        assert suffixes <= {".json", ".bin"}, suffixes
        assert not any(p.suffix.lower() in {".png", ".jpg", ".mp4"} for p in files)


def test_pose_files_contain_no_pixel_data(context, composition) -> None:
    composed = context.absolute(composition.composition.composed_pose_dir)
    sample = json.loads((composed / "frame_000000.json").read_text(encoding="utf-8"))
    assert set(sample) <= {
        "schema_version",
        "skeleton_format",
        "frame_index",
        "timestamp_s",
        "space",
        "origin",
        "body",
        "hands",
        "face",
        "source_bbox",
    }
    assert sample["source_bbox"] is None  # meaningless in canonical space


def test_qc_check_catches_a_planted_image_file(context, composition) -> None:
    """The guarantee is checked, not merely asserted."""
    import numpy as np

    from app.media.frames import write_frame

    report = run_motion_qc(context, composition.composition.id)
    assert report.passed, [c.as_dict() for c in report.failed_checks]

    planted = context.absolute(composition.composition.composed_pose_dir) / "leaked.png"
    write_frame(planted, np.zeros((4, 4, 3), dtype=np.uint8))

    after = run_motion_qc(context, composition.composition.id, options=None)
    failed = {c.check_id for c in after.failed_checks}
    assert "no_source_pixels_in_motion_artifacts" in failed


# ---------------------------------------------------------------------------
# motion QC
# ---------------------------------------------------------------------------
def test_motion_qc_passes_for_a_clean_composition(context, composition) -> None:
    report = run_motion_qc(context, composition.composition.id)
    assert report.passed, [c.as_dict() for c in report.failed_checks]

    ids = {c.check_id for c in report.checks}
    for expected in (
        "composition_frame_count",
        "shoulder_scale_drift",
        "torso_length_drift",
        "head_scale_drift",
        "limb_length_continuity",
        "velocity_continuity",
        "missing_joint_runs",
        "body_center_jump_at_joins",
        "bridge_endpoints_exact",
        "exposed_views",
        "no_source_pixels_in_motion_artifacts",
    ):
        assert expected in ids, f"missing motion QC check: {expected}"

    by_id = {c.check_id: c for c in report.checks}
    assert by_id["shoulder_scale_drift"].metrics["max_drift"] <= 0.02
    assert by_id["body_center_jump_at_joins"].metrics["max_jump_px"] <= 8.0
    assert by_id["bridge_endpoints_exact"].metrics["joins"][0]["start_matches_anchor"]
    assert by_id["bridge_endpoints_exact"].metrics["joins"][0]["end_matches_anchor"]


def test_motion_qc_thresholds_come_from_config(context, composition, config) -> None:
    """Thresholds are configurable, not hard-coded."""
    from app.pipeline.context import ServiceContext

    strict = config.with_overrides(motion_qc={"max_shoulder_scale_drift": 0.0})
    with ServiceContext.create(config=strict, configure_logs=False) as ctx:
        report = run_motion_qc(ctx, composition.composition.id)
        by_id = {c.check_id: c for c in report.checks}
        assert by_id["shoulder_scale_drift"].threshold == {"max_drift": 0.0}


def test_motion_qc_writes_reports(context, composition) -> None:
    run_motion_qc(context, composition.composition.id)
    root = context.absolute(composition.composition.composed_pose_dir).parent
    assert json.loads((root / "qc_report.json").read_text(encoding="utf-8"))["passed"]
    assert "QC REPORT" in (root / "qc_report.txt").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# requirement 10: manifests
# ---------------------------------------------------------------------------
def test_composition_manifest_records_everything(context, composition, motion_pair) -> None:
    comp = composition.composition
    manifest = json.loads(context.absolute(comp.manifest_path).read_text(encoding="utf-8"))

    # Source hashes: both the video and the pose data, per source.
    a, b = motion_pair
    for fixture in (a, b):
        key = fixture.source.version_key()
        assert manifest["input_hashes"][f"motion_source_video::{key}"] == (
            fixture.source.source_sha256
        )
        assert manifest["input_hashes"][f"motion_source_pose::{key}"]

    # Canonical transforms, per segment.
    assert len(manifest["segments"]) == 2
    for segment in manifest["segments"]:
        transform = segment["canonical_transform"]
        assert transform["base_scale"] > 0
        assert transform["source_shoulder_width"] > 0
        assert "stats" in transform

    # Joins, with every scored term.
    join = manifest["joins"][0]
    for key in (
        "prev_source_frame",
        "next_source_frame",
        "pose_distance_score",
        "root_position_delta",
        "shoulder_scale_delta",
        "torso_angle_delta",
        "head_angle_delta",
        "hand_position_delta",
        "bridge_frame_count",
    ):
        assert key in join

    # Bridge settings.
    bridge = manifest["bridge_settings"][0]
    assert bridge["interpolation"] == "cubic_hermite"
    assert bridge["frame_count"] == 12
    assert bridge["anchor_prev_frame"] == join["prev_source_frame"]

    # Reproducibility and the standing guarantee.
    assert manifest["reproducibility"]["config_hash"]
    assert manifest["reproducibility"]["dependencies"]
    assert manifest["contains_source_pixels"] is False
    assert len(manifest["digest"]) == 64


def test_composition_records_ranked_candidates_for_review(context, composition) -> None:
    join = composition.composition.joins[0]
    assert join.candidates, "the operator needs the alternatives that were considered"
    assert all("score" in candidate for candidate in join.candidates)


# ---------------------------------------------------------------------------
# master creation
# ---------------------------------------------------------------------------
@pytest.fixture
def hero(context):
    return mf.make_hero(context)


def make_candidate(context, composition, hero, **kwargs):
    return create_master_candidate(
        context,
        MasterCreateOptions(
            display_name=kwargs.pop("display_name", "Test master"),
            composition_id=composition.composition.id,
            hero_character_id=hero.id,
            backend_name="mock",
            seed=kwargs.pop("seed", 4242),
            chunk_frames=kwargs.pop("chunk_frames", 24),
            overlap_frames=kwargs.pop("overlap_frames", 16),
            **kwargs,
        ),
    )


def test_chunk_plan_tiles_the_sequence_exactly() -> None:
    for frame_count, chunk, overlap in ((70, 24, 16), (24, 24, 12), (100, 16, 12), (1, 24, 16)):
        plan = plan_chunks(frame_count, chunk, overlap)
        covered = [f for _, start, end in plan for f in range(start, end)]
        assert covered == list(range(frame_count))


def test_master_animation_produces_every_frame(context, composition, hero) -> None:
    candidate = make_candidate(context, composition, hero)
    result = animate_master(context, candidate.id)

    assert result.candidate.status is MasterCandidateStatus.ANIMATED
    frames_dir = context.absolute(result.candidate.frames_dir)
    assert list_frame_indices(frames_dir) == list(range(candidate.frame_count))
    assert not result.candidate.remaining_frames()


def test_the_mock_animator_records_the_context_it_really_consumed(
    context, composition, hero
) -> None:
    """None. The mock paints each frame independently, so it says so.

    The failure this guards against is the opposite one: a backend that reports
    16 context frames while the renderer looked at zero. What is recorded per
    chunk must be what the backend declared and was handed.
    """
    from app.backends.animator.base import ContextMode
    from app.backends.animator.mock import MockAnimatorBackend

    assert MockAnimatorBackend(context.config).capabilities().context_mode is ContextMode.NONE

    candidate = make_candidate(context, composition, hero)
    result = animate_master(context, candidate.id)
    chunks = sorted(result.candidate.chunks, key=lambda c: c.index)
    assert len(chunks) > 1, "the fixture must exercise more than one chunk"
    for chunk in chunks:
        assert chunk.context_mode == "none"
        assert chunk.overlap_frames == 0
        assert chunk.context_frames == []


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("last_frame", 1), ("sequence", 16)],
)
def test_the_pipeline_gathers_exactly_the_declared_context(
    context, composition, hero, monkeypatch, mode, expected
) -> None:
    """A declaring backend gets the tail it declared -- contiguous, and ending
    at the frame immediately before the chunk."""
    from app.backends.animator.base import ContextMode
    from app.backends.animator.mock import MockAnimatorBackend

    declared = ContextMode(mode)
    original_capabilities = MockAnimatorBackend.capabilities
    original_animate = MockAnimatorBackend.animate_chunk
    seen: dict[int, list[int]] = {}

    def capabilities(self):  # type: ignore[no-untyped-def]
        caps = original_capabilities(self)
        caps.context_mode = declared
        return caps

    def animate(self, animator_context, request):  # type: ignore[no-untyped-def]
        seen[request.chunk_index] = list(request.context_frame_indices)
        assert len(request.context_frames) == len(request.context_poses)
        request.validate()
        return original_animate(self, animator_context, request)

    monkeypatch.setattr(MockAnimatorBackend, "capabilities", capabilities)
    monkeypatch.setattr(MockAnimatorBackend, "animate_chunk", animate)

    candidate = make_candidate(context, composition, hero)
    result = animate_master(context, candidate.id)
    chunks = sorted(result.candidate.chunks, key=lambda c: c.index)
    assert len(chunks) > 1

    assert seen[0] == [], "nothing precedes the first chunk"
    for chunk in chunks[1:]:
        assert chunk.context_mode == mode
        assert chunk.overlap_frames == expected
        assert chunk.context_frames == list(range(chunk.start_frame - expected, chunk.start_frame))
        assert seen[chunk.index] == chunk.context_frames


def test_master_frames_match_the_canonical_profile(context, composition, hero) -> None:
    from app.media.frames import read_frame

    candidate = make_candidate(context, composition, hero)
    animate_master(context, candidate.id)
    frames_dir = context.absolute(candidate.frames_dir)
    frame = read_frame(frame_path(frames_dir, 0))
    assert frame.shape[:2] == (candidate.height, candidate.width)


# ---------------------------------------------------------------------------
# requirement 8: resume
# ---------------------------------------------------------------------------
def test_resume_skips_completed_chunks(context, composition, hero) -> None:
    candidate = make_candidate(context, composition, hero)
    first = animate_master(context, candidate.id, max_chunks=1)
    assert first.candidate.status is MasterCandidateStatus.PAUSED
    assert len(first.animated_chunks) == 1

    second = resume_master(context, candidate.id)
    assert second.skipped_chunks == first.animated_chunks
    assert not set(second.animated_chunks) & set(first.animated_chunks)
    assert second.candidate.status is MasterCandidateStatus.ANIMATED
    assert not second.candidate.remaining_frames()


def test_resumed_output_is_identical_to_a_single_pass(context, composition, hero) -> None:
    whole = make_candidate(context, composition, hero, candidate_id="mst_whole", seed=99)
    animate_master(context, whole.id)

    sliced = make_candidate(context, composition, hero, candidate_id="mst_sliced", seed=99)
    animate_master(context, sliced.id, max_chunks=1)
    resume_master(context, sliced.id, max_chunks=1)
    resume_master(context, sliced.id)

    whole_dir = context.absolute(context.repos.masters.get("mst_whole").frames_dir)
    sliced_dir = context.absolute(context.repos.masters.get("mst_sliced").frames_dir)
    indices = list_frame_indices(whole_dir)
    assert indices == list_frame_indices(sliced_dir)
    for index in indices:
        assert frames_equal(frame_path(whole_dir, index), frame_path(sliced_dir, index)), index


def test_chunks_are_not_handed_to_the_backend_twice(context, composition, hero) -> None:
    from app.backends.animator.mock import MockAnimatorBackend

    candidate = make_candidate(context, composition, hero)
    animate_master(context, candidate.id, max_chunks=1)

    seen: list[int] = []
    original = MockAnimatorBackend.animate_chunk

    def spy(self, animator_context, request):  # type: ignore[no-untyped-def]
        seen.append(request.chunk_index)
        return original(self, animator_context, request)

    MockAnimatorBackend.animate_chunk = spy  # type: ignore[method-assign]
    try:
        resume_master(context, candidate.id)
    finally:
        MockAnimatorBackend.animate_chunk = original  # type: ignore[method-assign]

    assert 0 not in seen
    assert seen == sorted(seen)


def test_checkpoints_survive_a_new_service_context(context, composition, hero, config) -> None:
    from app.pipeline.context import ServiceContext

    candidate = make_candidate(context, composition, hero)
    animate_master(context, candidate.id, max_chunks=1)
    context.close()

    with ServiceContext.create(config=config, configure_logs=False) as reopened:
        done = reopened.repos.masters.completed_chunks(candidate.id)
        assert len(done) == 1
        result = resume_master(reopened, candidate.id)
        assert result.skipped_chunks == [0]
        assert not result.candidate.remaining_frames()


# ---------------------------------------------------------------------------
# requirement 9: determinism
# ---------------------------------------------------------------------------
def test_identical_inputs_and_settings_give_identical_output(context, composition, hero) -> None:
    first = make_candidate(context, composition, hero, candidate_id="mst_a", seed=31337)
    second = make_candidate(context, composition, hero, candidate_id="mst_b", seed=31337)
    animate_master(context, first.id)
    animate_master(context, second.id)

    dir_a = context.absolute(context.repos.masters.get("mst_a").frames_dir)
    dir_b = context.absolute(context.repos.masters.get("mst_b").frames_dir)
    indices = list_frame_indices(dir_a)
    assert indices == list_frame_indices(dir_b)
    for index in indices:
        assert frames_equal(frame_path(dir_a, index), frame_path(dir_b, index)), index


def test_a_different_seed_changes_the_output(context, composition, hero) -> None:
    first = make_candidate(context, composition, hero, candidate_id="mst_s1", seed=1)
    second = make_candidate(context, composition, hero, candidate_id="mst_s2", seed=2)
    animate_master(context, first.id)
    animate_master(context, second.id)

    dir_a = context.absolute(context.repos.masters.get("mst_s1").frames_dir)
    dir_b = context.absolute(context.repos.masters.get("mst_s2").frames_dir)
    differing = [
        index
        for index in list_frame_indices(dir_a)
        if not frames_equal(frame_path(dir_a, index), frame_path(dir_b, index))
    ]
    assert differing


def test_composition_is_deterministic(context, motion_pair) -> None:
    """The same sources and settings must produce the same poses."""
    a, b = motion_pair
    digests = []
    for identifier in ("cmp_d1", "cmp_d2"):
        result = compose_motion(
            context,
            ComposeOptions(
                display_name="determinism",
                segments=[
                    SegmentSpec(motion_source_id=a.source.id),
                    SegmentSpec(motion_source_id=b.source.id),
                ],
                joins=[JoinSpec(bridge_frames=12)],
                composition_id=identifier,
                make_preview=False,
            ),
        )
        manifest = json.loads(
            context.absolute(result.composition.manifest_path).read_text(encoding="utf-8")
        )
        digests.append(manifest["digest"])
        assert result.composition.joins[0].prev_source_frame == (
            context.repos.compositions.get("cmp_d1").joins[0].prev_source_frame
        )
    assert digests[0] == digests[1]


# ---------------------------------------------------------------------------
# requirement 10 (master) + 11 (acceptance gate)
# ---------------------------------------------------------------------------
def test_master_manifest_records_all_inputs(context, composition, hero) -> None:
    candidate = make_candidate(context, composition, hero)
    animate_master(context, candidate.id)
    manifest = write_master_manifest(context, context.repos.masters.get(candidate.id))

    assert manifest["composition"] == composition.composition.version_key()
    assert manifest["hero_character"] == hero.version_key()
    assert manifest["input_hashes"]
    assert any(k.startswith("motion_source_video::") for k in manifest["input_hashes"])
    assert any(k.startswith("hero_reference::") for k in manifest["input_hashes"])
    assert any(k.startswith("composition_poses::") for k in manifest["input_hashes"])
    assert len(manifest["frame_hashes"]) == candidate.frame_count
    assert manifest["chunks"]
    assert manifest["settings"]["chunk_frames"] == 24
    assert manifest["settings"]["overlap_frames"] == 16
    assert manifest["reproducibility"]["config_hash"]
    assert manifest["contains_source_pixels"] is False
    assert len(manifest["digest"]) == 64


def test_identical_candidates_share_a_manifest_digest(context, composition, hero) -> None:
    digests = []
    for identifier in ("mst_m1", "mst_m2"):
        candidate = make_candidate(context, composition, hero, candidate_id=identifier, seed=7)
        animate_master(context, candidate.id)
        manifest = write_master_manifest(context, context.repos.masters.get(identifier))
        digests.append(manifest["digest"])
    assert digests[0] == digests[1]


def test_a_candidate_is_not_usable_until_accepted(context, composition, hero) -> None:
    """Requirement 11."""
    candidate = make_candidate(context, composition, hero)
    animate_master(context, candidate.id)
    write_master_manifest(context, context.repos.masters.get(candidate.id))

    current = context.repos.masters.get(candidate.id)
    assert not current.is_accepted
    assert current.status is not MasterCandidateStatus.ACCEPTED

    # QC alone does not promote it.
    report = run_master_qc(context, candidate.id)
    assert report.passed, [c.as_dict() for c in report.failed_checks]
    after_qc = context.repos.masters.get(candidate.id)
    assert not after_qc.is_accepted
    assert after_qc.status is MasterCandidateStatus.AWAITING_ACCEPTANCE

    accepted = accept_master(
        context,
        candidate.id,
        accepted_by="operator",
        reason="reviewed the preview and the QC report; framing is correct",
    )
    assert accepted.is_accepted
    assert accepted.status is MasterCandidateStatus.ACCEPTED
    assert accepted.acceptance is not None
    assert accepted.acceptance.accepted_by == "operator"
    assert accepted.acceptance.qc_passed


def test_acceptance_is_refused_before_qc(context, composition, hero) -> None:
    candidate = make_candidate(context, composition, hero)
    animate_master(context, candidate.id)
    with pytest.raises(ConflictError, match="has not been QC"):
        accept_master(context, candidate.id, accepted_by="op", reason="looks fine to me honestly")


def test_acceptance_is_refused_while_frames_are_missing(context, composition, hero) -> None:
    candidate = make_candidate(context, composition, hero)
    animate_master(context, candidate.id, max_chunks=1)
    with pytest.raises(ConflictError, match="incomplete"):
        accept_master(context, candidate.id, accepted_by="op", reason="impatient but incorrect")


def test_acceptance_requires_a_substantive_reason(context, composition, hero) -> None:
    candidate = make_candidate(context, composition, hero)
    animate_master(context, candidate.id)
    write_master_manifest(context, context.repos.masters.get(candidate.id))
    run_master_qc(context, candidate.id)
    with pytest.raises(ValidationError, match="real explanation"):
        accept_master(context, candidate.id, accepted_by="op", reason="ok")


def test_accepting_twice_is_refused(context, composition, hero) -> None:
    candidate = make_candidate(context, composition, hero)
    animate_master(context, candidate.id)
    write_master_manifest(context, context.repos.masters.get(candidate.id))
    run_master_qc(context, candidate.id)
    accept_master(
        context, candidate.id, accepted_by="op", reason="reviewed and accepted for production"
    )
    with pytest.raises(ConflictError, match="already accepted"):
        accept_master(context, candidate.id, accepted_by="op", reason="reviewed and accepted again")


def test_an_accepted_master_is_immutable(context, composition, hero) -> None:
    candidate = make_candidate(context, composition, hero)
    animate_master(context, candidate.id)
    write_master_manifest(context, context.repos.masters.get(candidate.id))
    run_master_qc(context, candidate.id)
    accept_master(
        context, candidate.id, accepted_by="op", reason="reviewed and accepted for production"
    )
    with pytest.raises(ConflictError, match="immutable"):
        animate_master(context, candidate.id, resume=True)


def test_acceptance_is_audited(context, composition, hero) -> None:
    candidate = make_candidate(context, composition, hero)
    animate_master(context, candidate.id)
    write_master_manifest(context, context.repos.masters.get(candidate.id))
    run_master_qc(context, candidate.id)
    accept_master(
        context,
        candidate.id,
        accepted_by="alex",
        reason="reviewed the transition and the framing; approved",
    )
    events = context.repos.audit.for_entity("master_candidate", candidate.id)
    accepted = next(e for e in events if e["event"] == "master_accepted")
    assert accepted["actor"] == "alex"
    assert "approved" in accepted["details"]["reason"]


def test_rejection_is_recorded(context, composition, hero) -> None:
    candidate = make_candidate(context, composition, hero)
    animate_master(context, candidate.id)
    rejected = reject_master(
        context, candidate.id, rejected_by="op", reason="the bridge reads as a stumble"
    )
    assert rejected.status is MasterCandidateStatus.REJECTED
    assert not rejected.is_accepted


def test_master_qc_covers_framing_background_and_chunks(context, composition, hero) -> None:
    candidate = make_candidate(context, composition, hero)
    animate_master(context, candidate.id)
    write_master_manifest(context, context.repos.masters.get(candidate.id))
    report = run_master_qc(context, candidate.id)

    ids = {c.check_id for c in report.checks}
    for expected in (
        "master_frame_sequence",
        "master_framing_consistency",
        "master_background_consistency",
        "master_chunk_boundaries",
        "master_manifest_complete",
        "master_acceptance_recorded",
    ):
        assert expected in ids
    assert report.passed, [c.as_dict() for c in report.failed_checks]


def test_unauthorized_motion_cannot_be_composed(context) -> None:
    a = mf.register_motion_source(
        context, motion_id="mot_unauth", spec=mf.MOTION_A, start=0, end=40, authorized=False
    )
    with pytest.raises(ConflictError, match="not authorized"):
        compose_motion(
            context,
            ComposeOptions(
                display_name="unauthorized",
                segments=[SegmentSpec(motion_source_id=a.source.id)],
                composition_id="cmp_unauth",
                make_preview=False,
            ),
        )


@requires_ffmpeg
def test_preview_is_rendered_and_checked(context, motion_pair) -> None:
    a, b = motion_pair
    result = compose_motion(
        context,
        ComposeOptions(
            display_name="with preview",
            segments=[
                SegmentSpec(motion_source_id=a.source.id),
                SegmentSpec(motion_source_id=b.source.id),
            ],
            joins=[JoinSpec(bridge_frames=10)],
            composition_id="cmp_preview",
            make_preview=True,
        ),
    )
    preview = context.absolute(result.composition.preview_path)
    assert preview.is_file()

    report = run_motion_qc(context, result.composition.id)
    by_id = {c.check_id: c for c in report.checks}
    check = by_id["preview_frame_count_and_fps"]
    assert check.passed, check.as_dict()
    assert check.metrics["frame_count"] == result.composition.output_frame_count


@requires_ffmpeg
def test_preview_contains_no_source_imagery(context, motion_pair) -> None:
    """The preview is drawn from poses; the reference videos are never opened."""
    a, _ = motion_pair
    result = compose_motion(
        context,
        ComposeOptions(
            display_name="preview only",
            segments=[SegmentSpec(motion_source_id=a.source.id)],
            composition_id="cmp_prev_only",
            make_preview=True,
        ),
    )
    # The sources have no decodable video at all -- only a placeholder byte file.
    # A preview could not have used them even if it wanted to.
    assert Path(a.source.source_video_path).suffix == ".bin"
    assert context.absolute(result.composition.preview_path).is_file()
