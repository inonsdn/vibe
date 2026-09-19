"""Motion input semantics: what the pose adapter is given, and what a
confidence threshold actually means.

Two round-3 defects are pinned here.

**P1-1.** ``extract_pose`` used to pass ``Path(source_video).parent`` as
``frames_dir``. That directory is neither the video nor a controlled frame
sequence — it is whatever else happens to sit beside the clip. The adapter now
receives the video file and the exact source frame indices, and the poses it
writes must carry those same indices.

**P1-2.** ``measure_pose_quality`` assigned the configured threshold to an
unused variable and then treated only ``confidence <= 0`` as missing, so a
sequence of joints that were placed but never confidently detected was reported
as high-quality usable motion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.adapters.pose import MockPoseAdapter
from app.core.errors import ValidationError
from app.pipeline.motion_ingest import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    extract_pose,
    measure_pose_quality,
    validate_motion_source,
)
from tests import motion_fixtures as mf

SELECTED = (12, 30)


# ---------------------------------------------------------------------------
# P1-1: adapter input semantics
# ---------------------------------------------------------------------------
class RecordingAdapter(MockPoseAdapter):
    """A mock adapter that remembers exactly how it was called."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.calls: list[dict[str, Any]] = []
        self.frames_dir_calls = 0

    def estimate_sequence(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return super().estimate_sequence(**kwargs)

    def run(self, **kwargs: Any) -> dict[str, Any]:  # pragma: no cover - must not happen
        self.frames_dir_calls += 1
        return super().run(**kwargs)


@pytest.fixture
def source(context):
    fixture = mf.register_motion_source(
        context,
        motion_id="mot_inputs",
        spec=mf.MOTION_A,
        start=SELECTED[0],
        end=SELECTED[1],
    )
    # Clear the fixture's poses so extraction has to produce them.
    for path in fixture.pose_dir.glob("*.json"):
        path.unlink()
    return fixture


def test_the_adapter_is_given_the_video_and_the_selected_range(context, source) -> None:
    adapter = RecordingAdapter()
    extract_pose(context, source.source.id, adapter=adapter)

    assert adapter.frames_dir_calls == 0, "the frames-directory entry point is the wrong one here"
    assert len(adapter.calls) == 1
    call = adapter.calls[0]
    assert Path(call["video_path"]) == Path(source.source.source_video_path)
    assert call["video_path"].is_file()
    assert call["frame_indices"] == list(range(*SELECTED))
    assert call["fps"] == source.source.video.fps
    assert call["options"]["selected_range"] == list(SELECTED)


def test_extracted_poses_keep_the_source_clips_frame_numbering(context, source) -> None:
    """A range starting at 12 writes frame_000012.json, not frame_000000.json."""
    from app.motion.pose_format import list_pose_indices

    result = extract_pose(context, source.source.id, adapter=RecordingAdapter())
    assert result.imported == list(range(*SELECTED))
    indices = list_pose_indices(context.absolute(source.source.pose_dir))
    assert indices == list(range(*SELECTED))
    assert indices[0] == SELECTED[0] != 0


def test_a_misnumbering_adapter_is_caught(context, source) -> None:
    """The alignment is checked, not trusted."""

    class ZeroBasedAdapter(MockPoseAdapter):
        def estimate_sequence(self, **kwargs: Any) -> dict[str, Any]:
            kwargs["frame_indices"] = list(range(len(kwargs["frame_indices"])))
            return super().estimate_sequence(**kwargs)

    with pytest.raises(ValidationError, match="source clip's own numbering"):
        extract_pose(context, source.source.id, adapter=ZeroBasedAdapter())


# ---------------------------------------------------------------------------
# P1-2: the confidence threshold is the confidence threshold
# ---------------------------------------------------------------------------
def test_placed_but_unconfident_joints_are_not_usable_motion() -> None:
    """Every joint exists, every confidence is below threshold.

    Before the fix this reported 100% frames_with_pose, a healthy shoulder
    width and no missing-joint runs — a sequence normalization would then
    refuse, described as high-quality motion.
    """
    threshold = DEFAULT_CONFIDENCE_THRESHOLD
    unconfident = mf.make_poses(mf.MOTION_A, 0, 30, confidence=threshold - 0.05)

    metrics = measure_pose_quality(unconfident, threshold)

    assert metrics.confidence_threshold == threshold
    assert metrics.frames_with_pose == 0
    assert metrics.in_frame_fraction == 0.0
    assert metrics.median_shoulder_width_px is None
    assert metrics.median_torso_length_px is None
    assert metrics.longest_missing_run == 30
    assert all(run == 30 for run in metrics.missing_joint_runs.values())
    # The raw confidences are still reported honestly -- the joints are there,
    # they are just not trusted.
    assert 0 < metrics.mean_joint_confidence < threshold


def test_the_same_poses_above_threshold_are_usable() -> None:
    threshold = DEFAULT_CONFIDENCE_THRESHOLD
    confident = mf.make_poses(mf.MOTION_A, 0, 30, confidence=threshold + 0.05)
    metrics = measure_pose_quality(confident, threshold)

    assert metrics.frames_with_pose == 30
    assert metrics.in_frame_fraction == 1.0
    assert metrics.median_shoulder_width_px
    assert metrics.longest_missing_run == 0


def test_validation_refuses_a_source_whose_joints_are_all_unconfident(context) -> None:
    threshold = mf.TEST_PROFILE["confidence_threshold"]
    unconfident = mf.make_poses(mf.MOTION_A, 0, 30, confidence=threshold - 0.05)
    fixture = mf.register_motion_source(
        context,
        motion_id="mot_unconfident",
        spec=mf.MOTION_A,
        start=0,
        end=30,
        poses=unconfident,
    )
    result = validate_motion_source(context, fixture.source.id)
    assert not result.ok
    assert any("consecutive frames" in problem for problem in result.problems)


@pytest.mark.parametrize("threshold", [0.1, 0.5, 0.9])
def test_the_threshold_is_applied_consistently(threshold: float) -> None:
    """One number decides 'missing' everywhere: measurements, runs and counts."""
    poses = mf.make_poses(mf.MOTION_A, 0, 20, confidence=0.5)
    metrics = measure_pose_quality(poses, threshold)
    confident = threshold <= 0.5

    assert metrics.confidence_threshold == threshold
    assert (metrics.frames_with_pose == 20) is confident
    assert (metrics.median_shoulder_width_px is not None) is confident
    assert (metrics.longest_missing_run == 0) is confident
