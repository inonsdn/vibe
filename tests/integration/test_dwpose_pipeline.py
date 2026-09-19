"""DWPose through the Motion Composition pipeline, with fake ONNX sessions.

The unit tests prove the adapter. These prove it is *wired in*: that
``app motion extract-pose`` produces pose JSON the existing pipeline accepts,
that the execution provider survives into the motion source record, that
diagnostics land outside every composition input, and that a poseable clip goes
on to normalize, compose and pass motion QC exactly as imported poses do.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.adapters.dwpose import DWPoseOnnxAdapter
from app.adapters.dwpose.diagnostics import (
    BOXES_NAME,
    CONFIDENCE_NAME,
    MISSING_NAME,
    OVERLAY_NAME,
    README_NAME,
    SKELETON_NAME,
    diagnostics_dir,
)
from app.adapters.dwpose.settings import DWPoseSettings
from app.adapters.dwpose.temporal import CleanupSettings
from app.core.errors import ValidationError
from app.motion.pose_format import list_pose_indices, load_pose_sequence
from tests import dwpose_fixtures as dw
from tests import motion_fixtures as mf
from tests.conftest import requires_ffmpeg

FRAME_W, FRAME_H = 360, 640
DETECTOR_SCALE = min(dw.DETECTOR_INPUT[0] / FRAME_W, dw.DETECTOR_INPUT[1] / FRAME_H)
SELECTED = (12, 42)

DANCER = (110.0, 150.0, 250.0, 550.0, 0.92)
AVATAR = (8.0, 12.0, 46.0, 50.0, 0.88)


def make_adapter(
    tmp_path: Path,
    *,
    boxes: list[tuple[float, float, float, float, float]] | None = None,
    providers: tuple[str, ...] = (dw.CUDA_PROVIDER, dw.CPU_PROVIDER),
    **settings_kwargs: Any,
) -> DWPoseOnnxAdapter:
    detector_path, pose_path = dw.touch_models(tmp_path / "models")
    detections = boxes if boxes is not None else [AVATAR, DANCER]
    points = dw.canonical_body_points()

    factory = dw.FakeSessionFactory(
        detector=dw.FakeSession(lambda _b: dw.yolox_output(detections, scale=DETECTOR_SCALE)),
        pose=dw.FakeSession(lambda _b: dw.simcc_output(points, [0.85] * len(points))),
        available=providers,
    )
    defaults: dict[str, Any] = {
        "detector_model": str(detector_path),
        "pose_model": str(pose_path),
        "detector_input_size": dw.DETECTOR_INPUT,
        "detector_strides": dw.DETECTOR_STRIDES,
        "pose_input_size": dw.POSE_INPUT,
        "simcc_split_ratio": dw.SPLIT_RATIO,
        "emit_hands": False,
        "cleanup": CleanupSettings(smoothing_window=1, max_interpolation_gap=3),
    }
    defaults.update(settings_kwargs)
    return DWPoseOnnxAdapter(
        DWPoseSettings(**defaults),
        session_factory=factory,
        frame_reader=dw.FakeFrameReader(width=FRAME_W, height=FRAME_H),
    )


@pytest.fixture
def source(context):
    """A registered motion reference with its fixture poses removed."""
    fixture = mf.register_motion_source(
        context,
        motion_id="mot_dwpose",
        spec=mf.MOTION_A,
        start=SELECTED[0],
        end=SELECTED[1],
    )
    for path in fixture.pose_dir.glob("*.json"):
        path.unlink()
    return fixture


# ---------------------------------------------------------------------------
# extraction through the pipeline
# ---------------------------------------------------------------------------
def test_extraction_produces_pipeline_ready_poses(context, source, tmp_path) -> None:
    from app.pipeline.motion_ingest import extract_pose, validate_motion_source

    result = extract_pose(context, source.source.id, adapter=make_adapter(tmp_path))

    assert result.imported == list(range(*SELECTED))
    assert list_pose_indices(source.pose_dir) == list(range(*SELECTED))

    record = context.repos.motion_sources.get(source.source.id)
    assert record.pose_origin == "extracted"
    assert record.pose_adapter == "dwpose_onnx"
    assert record.status.value == "pose_extracted"
    assert record.quality.frames_with_pose == 30

    validation = validate_motion_source(context, source.source.id)
    assert validation.ok, validation.problems


def test_the_execution_provider_reaches_the_motion_source_record(context, source, tmp_path) -> None:
    """A silent CPU fallback must be visible after the fact, not just in a log."""
    from app.pipeline.motion_ingest import extract_pose

    extract_pose(context, source.source.id, adapter=make_adapter(tmp_path))
    record = context.repos.motion_sources.get(source.source.id)

    extraction = record.pose_extraction
    assert extraction["adapter"] == "dwpose_onnx"
    assert extraction["provider"]["detector"]["active"] == dw.CUDA_PROVIDER
    assert extraction["provider"]["pose"]["using_cuda"] is True
    assert extraction["provider"]["detector"]["fell_back_to_cpu"] is False
    assert set(extraction["model_sha256"]) == {"detector", "pose"}
    assert extraction["settings"]["roi_mode"] == "none"
    assert extraction["contains_source_pixels"] is False


def test_a_cpu_fallback_is_recorded_on_the_record(context, source, tmp_path) -> None:
    from app.pipeline.motion_ingest import extract_pose

    adapter = make_adapter(tmp_path, providers=(dw.CPU_PROVIDER,))
    extract_pose(context, source.source.id, adapter=adapter)
    record = context.repos.motion_sources.get(source.source.id)

    provider = record.pose_extraction["provider"]["detector"]
    assert provider["active"] == dw.CPU_PROVIDER
    assert provider["fell_back_to_cpu"] is True
    assert provider["available"] == [dw.CPU_PROVIDER]


def test_the_audit_log_records_the_provider_and_the_weights(context, source, tmp_path) -> None:
    from app.pipeline.motion_ingest import extract_pose

    extract_pose(context, source.source.id, adapter=make_adapter(tmp_path))
    events = context.repos.audit.for_entity("motion_source", source.source.id)
    extracted = [e for e in events if e["event"] == "motion_pose_extracted"]
    assert extracted, [e["event"] for e in events]
    details = extracted[-1]["details"]
    assert details["adapter"] == "dwpose_onnx"
    assert details["synthetic"] is False
    assert details["provider"]["detector"]["active"] == dw.CUDA_PROVIDER
    assert len(details["model_sha256"]["pose"]) == 64


def test_an_unavailable_adapter_refuses_before_touching_the_record(
    context, source, tmp_path
) -> None:
    from app.adapters.base import AdapterNotAvailableError
    from app.pipeline.motion_ingest import extract_pose

    unconfigured = DWPoseOnnxAdapter(DWPoseSettings())
    with pytest.raises(AdapterNotAvailableError, match="never downloads"):
        extract_pose(context, source.source.id, adapter=unconfigured)

    record = context.repos.motion_sources.get(source.source.id)
    assert record.pose_origin == "imported"
    assert record.pose_extraction == {}


def test_frames_with_no_subject_leave_gaps_the_pipeline_notices(context, source, tmp_path) -> None:
    """Only the UI avatar is detectable: no dancer, so no poses, and the
    pipeline's own frame-range check is what reports it."""
    from app.pipeline.motion_ingest import extract_pose

    adapter = make_adapter(tmp_path, boxes=[AVATAR])
    with pytest.raises(ValidationError, match="source clip's own numbering"):
        extract_pose(context, source.source.id, adapter=adapter)


# ---------------------------------------------------------------------------
# diagnostics live outside every composition input (requirement 13)
# ---------------------------------------------------------------------------
def test_diagnostics_are_written_outside_the_pose_directory(context, source, tmp_path) -> None:
    from app.pipeline.motion_ingest import extract_pose

    result = extract_pose(
        context,
        source.source.id,
        adapter=make_adapter(tmp_path),
        diagnostics=True,
        diagnostics_overlay=False,
    )

    destination = Path(result.diagnostics_dir or "")
    expected = diagnostics_dir(context.data_root.path, source.source.id)
    assert destination == expected
    assert destination.is_dir()

    # Not inside the pose directory, and not inside anything a composition reads.
    assert source.pose_dir not in destination.parents
    assert "motion_sources" not in destination.parts
    assert "compositions" not in destination.parts

    # The pose directory itself is still JSON only.
    suffixes = {p.suffix for p in source.pose_dir.rglob("*") if p.is_file()}
    assert suffixes == {".json"}


def test_the_diagnostic_reports_are_complete(context, source, tmp_path) -> None:
    from app.pipeline.motion_ingest import extract_pose

    result = extract_pose(
        context,
        source.source.id,
        adapter=make_adapter(tmp_path),
        diagnostics=True,
        diagnostics_overlay=False,
    )
    destination = Path(result.diagnostics_dir or "")

    boxes = json.loads((destination / BOXES_NAME).read_text())
    assert len(boxes["frames"]) == 30
    first = boxes["frames"][0]
    assert first["selected"]["x1"] == pytest.approx(DANCER[0], abs=0.5)
    # Both candidates are recorded with their scored terms, so a wrong pick can
    # be inspected rather than guessed at.
    assert len(first["candidates"]) == 2
    assert {c["rejected"] for c in first["candidates"]} == {None, "too_small"}

    confidence = json.loads((destination / CONFIDENCE_NAME).read_text())
    assert confidence["frames"] == 30
    assert confidence["joints"]["nose"]["coverage"] == 1.0
    assert confidence["overall"]["mean"] == pytest.approx(0.85, abs=0.01)

    missing = json.loads((destination / MISSING_NAME).read_text())
    assert missing["requested_frames"] == 30
    assert missing["frames_without_subject_count"] == 0

    extraction = json.loads((destination / "extraction.json").read_text())
    assert extraction["adapter"] == "dwpose_onnx"

    readme = (destination / README_NAME).read_text()
    assert "NOT a composition\ninput" in readme or "NOT a composition" in readme


def test_the_missing_joint_report_names_the_runs(context, source, tmp_path) -> None:
    from app.adapters.dwpose.diagnostics import build_missing_report
    from app.motion.pose_format import Joint2D, PoseFrame
    from app.motion.skeleton import BODY_JOINTS

    poses = []
    for index in range(6):
        body = {name: Joint2D(x=1.0, y=2.0, confidence=0.9) for name in BODY_JOINTS}
        if index in (2, 3, 4):
            body.pop("right_wrist")
        poses.append(PoseFrame(frame_index=index, timestamp_s=index / 30.0, body=body))

    report = build_missing_report(poses, [], list(range(6)))
    assert report["missing_runs"]["right_wrist"]["runs"] == [[2, 4]]
    assert report["missing_runs"]["right_wrist"]["longest"] == 3


@requires_ffmpeg
def test_diagnostic_videos_are_rendered(context, source, tmp_path) -> None:
    from app.pipeline.motion_ingest import extract_pose

    result = extract_pose(
        context, source.source.id, adapter=make_adapter(tmp_path), diagnostics=True
    )
    destination = Path(result.diagnostics_dir or "")

    skeleton = destination / SKELETON_NAME
    overlay = destination / OVERLAY_NAME
    assert skeleton.is_file() and skeleton.stat().st_size > 0
    assert overlay.is_file() and overlay.stat().st_size > 0
    # Staging frames are cleaned up; only the finished artefacts remain.
    assert not (destination / ".overlay_frames").exists()


@requires_ffmpeg
def test_the_overlay_can_be_suppressed_entirely(context, source, tmp_path) -> None:
    """For an operator who does not want source pixels written anywhere."""
    from app.pipeline.motion_ingest import extract_pose

    result = extract_pose(
        context,
        source.source.id,
        adapter=make_adapter(tmp_path),
        diagnostics=True,
        diagnostics_overlay=False,
    )
    destination = Path(result.diagnostics_dir or "")
    assert not (destination / OVERLAY_NAME).exists()
    assert (destination / SKELETON_NAME).is_file()


# ---------------------------------------------------------------------------
# extracted poses behave exactly like imported ones downstream
# ---------------------------------------------------------------------------
def test_extracted_poses_normalize_and_compose(context, source, tmp_path) -> None:
    from app.pipeline.motion_compose import (
        ComposeOptions,
        SegmentSpec,
        compose_motion,
        load_profile,
        normalize_segment,
    )
    from app.pipeline.motion_ingest import extract_pose

    extract_pose(context, source.source.id, adapter=make_adapter(tmp_path))
    record = context.repos.motion_sources.get(source.source.id)
    normalized = normalize_segment(
        context,
        SegmentSpec(motion_source_id=record.id, motion_source_version=record.version),
        load_profile(context),
        output_fps=record.video.fps,
    )
    assert normalized.transform.base_scale > 0

    composition = compose_motion(
        context,
        ComposeOptions(
            display_name="From a real extraction",
            segments=[SegmentSpec(motion_source_id=source.source.id, exposed_views=["front"])],
            joins=[],
            composition_id="cmp_dwpose",
            make_preview=False,
        ),
    ).composition
    assert composition.output_frame_count == 30

    poses = load_pose_sequence(context.absolute(composition.composed_pose_dir))
    assert [p.frame_index for p in poses] == list(range(30))


def test_motion_qc_passes_on_an_extracted_composition(context, source, tmp_path) -> None:
    from app.pipeline.motion_compose import ComposeOptions, SegmentSpec, compose_motion
    from app.pipeline.motion_ingest import extract_pose
    from app.qc.motion_checks import run_motion_qc

    extract_pose(context, source.source.id, adapter=make_adapter(tmp_path))
    composition = compose_motion(
        context,
        ComposeOptions(
            display_name="QC me",
            segments=[SegmentSpec(motion_source_id=source.source.id, exposed_views=["front"])],
            joins=[],
            composition_id="cmp_dwpose_qc",
            make_preview=False,
        ),
    ).composition

    report = run_motion_qc(context, composition.id)
    assert report.passed, [c.check_id for c in report.failed_checks]
    assert report.metrics["no_source_pixels"]["count"] == 0


def test_a_stray_image_in_the_pose_directory_is_refused(context, source, tmp_path) -> None:
    """The structural guarantee, checked rather than asserted: an adapter that
    wrote an overlay next to the poses would be caught here."""
    from app.pipeline.motion_ingest import extract_pose

    class LeakyAdapter(DWPoseOnnxAdapter):
        def estimate_sequence(self, **kwargs: Any) -> dict[str, Any]:
            report = super().estimate_sequence(**kwargs)
            (Path(kwargs["output_dir"]) / "overlay.mp4").write_bytes(b"source pixels")
            return report

    adapter = make_adapter(tmp_path)
    leaky = LeakyAdapter(
        adapter.settings,
        session_factory=adapter._factory,
        frame_reader=adapter._reader,
    )
    with pytest.raises(ValidationError, match="geometry, never imagery"):
        extract_pose(context, source.source.id, adapter=leaky)
