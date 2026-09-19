"""DWPose ONNX adapter: the real pipeline, driven by fake ONNX sessions.

Nothing here needs onnxruntime, CUDA, model weights, a video file or a network.
The sessions and the frame reader are injected; everything between them — the
letterbox, the YOLOX decode, NMS, subject scoring, the crop inversion, the SimCC
decode, the ROI restoration, the COCO index mapping and the temporal cleanup —
is the production code path.

Frame geometry used throughout: a 360x640 portrait frame (a phone screen
recording), letterboxed into a 64x64 detector input, so the detector scale is
exactly 0.1 and expected coordinates can be worked out by hand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.adapters.base import AdapterStatus
from app.adapters.dwpose import DWPoseOnnxAdapter
from app.adapters.dwpose.keypoints import COCO_BODY_INDEX
from app.adapters.dwpose.session import (
    CPU_PROVIDER,
    CUDA_PROVIDER,
    ModelFileMissingError,
    ProviderUnavailableError,
)
from app.adapters.dwpose.settings import DWPoseSettings
from app.adapters.dwpose.temporal import CleanupSettings
from app.adapters.dwpose.tracking import TrackerSettings
from app.core.errors import ValidationError
from app.motion.pose_format import list_pose_indices, load_pose_sequence
from app.motion.skeleton import BODY_JOINTS
from tests import dwpose_fixtures as dw

FRAME_W, FRAME_H = 360, 640
DETECTOR_SCALE = min(dw.DETECTOR_INPUT[0] / FRAME_W, dw.DETECTOR_INPUT[1] / FRAME_H)

#: A large, central dancer. Roughly half the frame height, centred.
DANCER = (110.0, 150.0, 250.0, 550.0, 0.92)
#: A tiny profile avatar in the top-left: the UI, not a person to track.
AVATAR = (8.0, 12.0, 46.0, 50.0, 0.88)
#: A cartoon reference parked in the bottom-right corner.
CORNER_IMAGE = (250.0, 470.0, 350.0, 630.0, 0.95)


def _unique_scores(count: int = 17) -> list[float]:
    """Score i == (i + 1) / 200 — a fingerprint for keypoint index i.

    This is what makes the joint-mapping assertions independent of any
    coordinate arithmetic: if ``left_ear`` carries score 4/200, it came from
    keypoint index 3 and nothing else.
    """
    return [(index + 1) / 200.0 for index in range(count)]


def build_adapter(
    tmp_path: Path,
    *,
    boxes_per_frame: list[list[tuple[float, float, float, float, float]]] | None = None,
    boxes: list[tuple[float, float, float, float, float]] | None = None,
    keypoint_scores: list[float] | None = None,
    keypoints: list[tuple[float, float]] | None = None,
    keypoint_count: int = 17,
    detector_scale: float = DETECTOR_SCALE,
    providers: tuple[str, ...] = (CPU_PROVIDER,),
    reader: dw.FakeFrameReader | None = None,
    **settings_kwargs: Any,
) -> tuple[DWPoseOnnxAdapter, dw.FakeSessionFactory, dw.FakeFrameReader]:
    """An adapter wired to scripted detector and pose sessions."""
    detector_path, pose_path = dw.touch_models(tmp_path)

    scripted = boxes_per_frame
    frame_counter = {"n": 0}

    def detector_handler(_batch: np.ndarray) -> np.ndarray:
        if scripted is not None:
            index = min(frame_counter["n"], len(scripted) - 1)
            this_frame = scripted[index]
            frame_counter["n"] += 1
        else:
            this_frame = boxes if boxes is not None else [DANCER]
        return dw.yolox_output(this_frame, scale=detector_scale)

    points = keypoints or dw.canonical_body_points(count=keypoint_count)
    scores = keypoint_scores or _unique_scores(len(points))

    def pose_handler(_batch: np.ndarray) -> list[np.ndarray]:
        return dw.simcc_output(points, scores)

    factory = dw.FakeSessionFactory(
        detector=dw.FakeSession(detector_handler),
        pose=dw.FakeSession(pose_handler),
        available=providers,
    )
    defaults: dict[str, Any] = {
        "detector_model": str(detector_path),
        "pose_model": str(pose_path),
        "detector_input_size": dw.DETECTOR_INPUT,
        "detector_strides": dw.DETECTOR_STRIDES,
        "pose_input_size": dw.POSE_INPUT,
        "simcc_split_ratio": dw.SPLIT_RATIO,
        "keypoint_score_threshold": 0.0,
        "emit_hands": False,
        "tracker": TrackerSettings(),
        # Cleanup off by default here so a test sees raw detections; the
        # temporal tests turn it on deliberately.
        "cleanup": CleanupSettings(smoothing_window=0, max_interpolation_gap=0),
    }
    defaults.update(settings_kwargs)
    settings = DWPoseSettings(**defaults)
    frame_reader = reader or dw.FakeFrameReader(width=FRAME_W, height=FRAME_H)
    adapter = DWPoseOnnxAdapter(settings, session_factory=factory, frame_reader=frame_reader)
    return adapter, factory, frame_reader


def extract(adapter: DWPoseOnnxAdapter, tmp_path: Path, indices: list[int], **kwargs: Any):
    return adapter.estimate_sequence(
        video_path=tmp_path / "clip.mp4",
        output_dir=tmp_path / "pose",
        frame_indices=indices,
        fps=30.0,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# no downloads, no network, actionable failures
# ---------------------------------------------------------------------------
def test_an_unconfigured_adapter_reports_missing_weights_and_refuses() -> None:
    adapter = DWPoseOnnxAdapter(DWPoseSettings())
    capability = adapter.capability()

    assert capability.status is AdapterStatus.MISSING_WEIGHTS
    assert capability.available is False
    assert capability.notes["downloads"] == "none - model files are supplied by the operator"
    assert capability.notes["synthetic"] is False
    # The reason has to tell the operator what to do, not just that it failed.
    assert "never downloads" in capability.reason
    assert "config/local.yaml" in capability.reason


def test_a_missing_model_file_names_the_path(tmp_path: Path) -> None:
    settings = DWPoseSettings(
        detector_model=str(tmp_path / "absent.onnx"), pose_model=str(tmp_path / "also_absent.onnx")
    )
    adapter = DWPoseOnnxAdapter(settings)
    with pytest.raises(ModelFileMissingError) as exc:
        adapter.ensure_sessions()
    assert str(tmp_path / "absent.onnx") in str(exc.value.details["path"])
    assert "hint" in exc.value.details


def test_a_non_onnx_model_path_is_refused(tmp_path: Path) -> None:
    """A .pth or a folder is a configuration mistake, not something to load."""
    wrong = tmp_path / "dwpose.pth"
    wrong.write_bytes(b"x")
    adapter = DWPoseOnnxAdapter(DWPoseSettings(detector_model=str(wrong), pose_model=str(wrong)))
    with pytest.raises(ModelFileMissingError, match=r"not an \.onnx file"):
        adapter.ensure_sessions()


def test_the_package_contains_no_download_machinery() -> None:
    """Grep the shipped source: no URLs, no fetchers, no hub clients."""
    import app.adapters.dwpose as package

    root = Path(package.__file__).parent
    banned = (
        "http://",
        "https://",
        "urllib",
        "requests.",
        "huggingface",
        "hf_hub",
        "wget",
        "curl ",
    )
    offenders: list[str] = []
    for path in sorted(root.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for needle in banned:
            if needle in text:
                offenders.append(f"{path.name}: {needle}")
    assert offenders == [], offenders


def test_extraction_opens_no_sockets(tmp_path: Path) -> None:
    """The conftest guard fails any non-loopback connection; this asserts the
    whole extraction path runs inside it without tripping it."""
    adapter, _factory, _reader = build_adapter(tmp_path)
    report = extract(adapter, tmp_path, [0, 1, 2])
    assert report["frames"] == 3
    assert report["contains_source_pixels"] is False


# ---------------------------------------------------------------------------
# provider reporting (requirement 4)
# ---------------------------------------------------------------------------
def test_cpu_only_build_reports_cpu(tmp_path: Path) -> None:
    adapter, factory, _ = build_adapter(tmp_path, providers=(CPU_PROVIDER,))
    detector, pose = adapter.ensure_sessions()

    assert detector.active == CPU_PROVIDER
    assert pose.active == CPU_PROVIDER
    assert detector.using_cuda is False
    # `auto` on a CPU-only build IS a fallback, and says so.
    assert detector.fell_back is True
    assert detector.as_dict()["available"] == [CPU_PROVIDER]
    assert factory.requests[0][1] == [CPU_PROVIDER]


def test_cuda_build_reports_cuda(tmp_path: Path) -> None:
    adapter, factory, _ = build_adapter(tmp_path, providers=(CUDA_PROVIDER, CPU_PROVIDER))
    detector, pose = adapter.ensure_sessions()

    assert detector.active == CUDA_PROVIDER
    assert pose.using_cuda is True
    assert detector.fell_back is False
    # CPU stays in the offered list as the fallback onnxruntime may drop to.
    assert factory.requests[0][1] == [CUDA_PROVIDER, CPU_PROVIDER]


def test_a_silent_cuda_to_cpu_fallback_is_recorded_not_hidden(tmp_path: Path) -> None:
    """onnxruntime can accept CUDA and still bind CPU. The session is believed."""
    adapter, factory, _ = build_adapter(tmp_path, providers=(CUDA_PROVIDER, CPU_PROVIDER))
    factory.reports = {
        "yolox_l_fake.onnx": [CPU_PROVIDER],
        "dw_ll_ucoco_fake.onnx": [CPU_PROVIDER],
    }
    detector, _pose = adapter.ensure_sessions()

    assert detector.active == CPU_PROVIDER
    assert detector.fell_back is True
    assert detector.as_dict()["fell_back_to_cpu"] is True


def test_requesting_cuda_on_a_cpu_build_fails_loudly(tmp_path: Path) -> None:
    adapter, _factory, _ = build_adapter(
        tmp_path, providers=(CPU_PROVIDER,), provider="cuda", require_requested_provider=True
    )
    with pytest.raises(ProviderUnavailableError, match="onnxruntime-gpu"):
        adapter.ensure_sessions()


def test_cuda_can_be_downgraded_explicitly(tmp_path: Path) -> None:
    adapter, _factory, _ = build_adapter(
        tmp_path, providers=(CPU_PROVIDER,), provider="cuda", require_requested_provider=False
    )
    detector, _pose = adapter.ensure_sessions()
    assert detector.active == CPU_PROVIDER
    assert detector.fell_back is True


def test_the_provider_reaches_the_extraction_report(tmp_path: Path) -> None:
    adapter, _factory, _ = build_adapter(tmp_path, providers=(CUDA_PROVIDER, CPU_PROVIDER))
    report = extract(adapter, tmp_path, [0, 1])

    assert report["provider"]["detector"]["active"] == CUDA_PROVIDER
    assert report["provider"]["pose"]["using_cuda"] is True
    assert report["provider"]["detector"]["onnxruntime_version"] == "1.99.0-fake"
    # The weights are identified by hash, so a manifest says which files ran.
    assert set(report["model_sha256"]) == {"detector", "pose"}
    assert len(report["model_sha256"]["pose"]) == 64


# ---------------------------------------------------------------------------
# frame indices (requirement 7)
# ---------------------------------------------------------------------------
def test_frame_indices_are_preserved_exactly(tmp_path: Path) -> None:
    """A range starting at 137 writes frame_000137.json, not frame_000000.json."""
    wanted = list(range(137, 149))
    adapter, _factory, reader = build_adapter(tmp_path)
    report = extract(adapter, tmp_path, wanted)

    assert list_pose_indices(tmp_path / "pose") == wanted
    assert report["first_frame"] == 137
    assert report["last_frame"] == 148
    assert reader.reads == [wanted], "the reader is asked for exactly those frames"

    poses = load_pose_sequence(tmp_path / "pose")
    assert [p.frame_index for p in poses] == wanted
    # Timestamps follow the source clip's own numbering, not a rebased one.
    assert poses[0].timestamp_s == pytest.approx(137 / 30.0)


def test_a_non_contiguous_request_is_honoured_frame_for_frame(tmp_path: Path) -> None:
    wanted = [4, 9, 10, 40]
    adapter, _factory, _reader = build_adapter(tmp_path)
    extract(adapter, tmp_path, wanted)
    assert list_pose_indices(tmp_path / "pose") == wanted


def test_no_frames_requested_is_refused(tmp_path: Path) -> None:
    adapter, _factory, _reader = build_adapter(tmp_path)
    with pytest.raises(ValidationError, match="No frames requested"):
        extract(adapter, tmp_path, [])


def test_a_clip_shorter_than_the_range_fails_rather_than_shifting(tmp_path: Path) -> None:
    """Reading past the end must not renumber what was read."""
    reader = dw.FakeFrameReader(width=FRAME_W, height=FRAME_H, truncate_after=5)
    adapter, _factory, _ = build_adapter(tmp_path, reader=reader)
    extract(adapter, tmp_path, [0, 1, 2, 3, 4, 5, 6, 7])
    # Frames that could not be read simply do not appear; the ones that could
    # keep their own indices.
    assert list_pose_indices(tmp_path / "pose") == [0, 1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# COCO joint mapping (requirement 6)
# ---------------------------------------------------------------------------
def test_every_coco_joint_maps_to_its_own_keypoint_index(tmp_path: Path) -> None:
    """Score i == (i+1)/200 fingerprints keypoint index i, independently of
    any coordinate arithmetic. A swapped ear and eye fails here."""
    adapter, _factory, _ = build_adapter(tmp_path)
    extract(adapter, tmp_path, [0])
    pose = load_pose_sequence(tmp_path / "pose")[0]

    assert set(pose.body) == set(BODY_JOINTS)
    for name in BODY_JOINTS:
        index = COCO_BODY_INDEX[name]
        assert pose.body[name].confidence == pytest.approx((index + 1) / 200.0, abs=1e-6), name


@pytest.mark.parametrize(
    "name",
    [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ],
)
def test_each_required_joint_is_present(tmp_path: Path, name: str) -> None:
    adapter, _factory, _ = build_adapter(tmp_path)
    extract(adapter, tmp_path, [0])
    pose = load_pose_sequence(tmp_path / "pose")[0]
    assert name in pose.body


def test_the_body_keeps_its_anatomical_layout_after_mapping(tmp_path: Path) -> None:
    """Coordinates survive the crop inversion with the body the right way up."""
    adapter, _factory, _ = build_adapter(tmp_path)
    extract(adapter, tmp_path, [0])
    body = load_pose_sequence(tmp_path / "pose")[0].body

    assert body["left_shoulder"].x < body["right_shoulder"].x
    assert body["left_hip"].x < body["right_hip"].x
    assert body["nose"].y < body["left_shoulder"].y < body["left_hip"].y
    assert body["left_hip"].y < body["left_knee"].y < body["left_ankle"].y
    assert body["left_elbow"].y < body["left_wrist"].y
    # And it landed on the dancer, not somewhere else in the frame.
    x1, y1, w, h = load_pose_sequence(tmp_path / "pose")[0].source_bbox
    assert (x1, y1) == pytest.approx((DANCER[0], DANCER[1]), abs=0.5)
    assert (w, h) == pytest.approx((DANCER[2] - DANCER[0], DANCER[3] - DANCER[1]), abs=0.5)


def test_hands_are_emitted_for_a_wholebody_model(tmp_path: Path) -> None:
    adapter, _factory, _ = build_adapter(tmp_path, keypoint_count=133, emit_hands=True)
    extract(adapter, tmp_path, [0])
    pose = load_pose_sequence(tmp_path / "pose")[0]

    assert len(pose.hands) == 42
    assert "left_hand_00" in pose.hands and "right_hand_20" in pose.hands
    # Body joints are unaffected by the extra keypoints.
    assert set(pose.body) == set(BODY_JOINTS)


def test_an_unknown_keypoint_layout_is_refused(tmp_path: Path) -> None:
    """A 21-keypoint model is not something whose first 17 we can vouch for."""
    adapter, _factory, _ = build_adapter(tmp_path, keypoint_count=21)
    with pytest.raises(ValidationError, match="Unrecognised keypoint layout"):
        extract(adapter, tmp_path, [0])


def test_low_confidence_joints_are_dropped_not_invented(tmp_path: Path) -> None:
    scores = [0.8] * 17
    scores[COCO_BODY_INDEX["left_wrist"]] = 0.05
    adapter, _factory, _ = build_adapter(
        tmp_path, keypoint_scores=scores, keypoint_score_threshold=0.2
    )
    extract(adapter, tmp_path, [0])
    body = load_pose_sequence(tmp_path / "pose")[0].body

    assert "left_wrist" not in body
    assert "right_wrist" in body


# ---------------------------------------------------------------------------
# ROI (requirement 9)
# ---------------------------------------------------------------------------
def test_pixel_roi_coordinates_come_back_in_original_video_space(tmp_path: Path) -> None:
    """The crop is internal. What is written is original-video pixels."""
    without, _f, _r = build_adapter(tmp_path / "plain")
    extract(without, tmp_path / "plain", [0])
    plain = load_pose_sequence(tmp_path / "plain" / "pose")[0]

    # Same scripted detection, but expressed inside a crop that starts at
    # (40, 90) -- so the adapter must add that offset back on.
    offset = (40.0, 90.0)
    shifted = [
        (
            DANCER[0] - offset[0],
            DANCER[1] - offset[1],
            DANCER[2] - offset[0],
            DANCER[3] - offset[1],
            DANCER[4],
        )
    ]
    # The crop is a different size, so the detector's letterbox scale differs.
    crop_w, crop_h = 280.0, 500.0
    cropped, _f, _r = build_adapter(
        tmp_path / "roi",
        boxes=shifted,
        detector_scale=min(dw.DETECTOR_INPUT[0] / crop_w, dw.DETECTOR_INPUT[1] / crop_h),
        roi_mode="pixels",
        roi=(offset[0], offset[1], crop_w, crop_h),
    )
    extract(cropped, tmp_path / "roi", [0])
    restored = load_pose_sequence(tmp_path / "roi" / "pose")[0]

    assert restored.source_bbox[0] == pytest.approx(DANCER[0], abs=1.0)
    assert restored.source_bbox[1] == pytest.approx(DANCER[1], abs=1.0)
    for name in ("nose", "left_wrist", "right_ankle"):
        assert restored.body[name].x == pytest.approx(plain.body[name].x, abs=1.5), name
        assert restored.body[name].y == pytest.approx(plain.body[name].y, abs=1.5), name


def test_a_normalized_roi_resolves_to_the_same_pixels(tmp_path: Path) -> None:
    adapter, _factory, _ = build_adapter(
        tmp_path,
        roi_mode="normalized",
        roi=(0.1, 0.2, 0.5, 0.6),
    )
    report = extract(adapter, tmp_path, [0])
    roi = report["roi"]
    assert roi["mode"] == "normalized"
    assert (roi["x"], roi["y"]) == (36, 128)
    assert (roi["width"], roi["height"]) == (180, 384)


def test_the_roi_is_reported_even_when_it_is_the_whole_frame(tmp_path: Path) -> None:
    adapter, _factory, _ = build_adapter(tmp_path)
    report = extract(adapter, tmp_path, [0])
    assert report["roi"] == {
        "mode": "none",
        "x": 0,
        "y": 0,
        "width": FRAME_W,
        "height": FRAME_H,
    }
    assert report["frame_size"] == [FRAME_W, FRAME_H]


# ---------------------------------------------------------------------------
# subject selection (requirement 8)
# ---------------------------------------------------------------------------
def test_the_central_dancer_wins_over_a_corner_image_and_a_ui_avatar(
    tmp_path: Path,
) -> None:
    """The corner image scores HIGHER as a detection (0.95 vs 0.92). Detection
    score is deliberately not part of the subject decision."""
    adapter, _factory, _ = build_adapter(tmp_path, boxes=[AVATAR, CORNER_IMAGE, DANCER])
    extract(adapter, tmp_path, [0])
    bbox = load_pose_sequence(tmp_path / "pose")[0].source_bbox

    assert bbox[0] == pytest.approx(DANCER[0], abs=0.5)
    assert bbox[1] == pytest.approx(DANCER[1], abs=0.5)


def test_a_tiny_avatar_is_rejected_outright(tmp_path: Path) -> None:
    adapter, _factory, _ = build_adapter(tmp_path, boxes=[AVATAR])
    report = extract(adapter, tmp_path, [0, 1])

    assert report["frames_with_pose"] == 0
    assert report["subject_tracking"]["rejected"]["too_small"] == 2
    assert report["frames_without_pose"] == 2


def test_a_corner_figure_is_rejected_by_the_distance_gate(tmp_path: Path) -> None:
    """A big cartoon in the corner is not the subject, however big it is."""
    far_corner = (300.0, 560.0, 360.0, 640.0, 0.99)
    adapter, _factory, _ = build_adapter(
        tmp_path,
        boxes=[far_corner],
        tracker=TrackerSettings(min_area_fraction=0.0, max_center_distance=0.55),
    )
    report = extract(adapter, tmp_path, [0])
    assert report["subject_tracking"]["rejected"]["too_far_from_centre"] == 1
    assert report["frames_with_pose"] == 0


def test_tracking_does_not_switch_to_a_momentarily_larger_rival(tmp_path: Path) -> None:
    """A background figure briefly outgrows the dancer. The track must hold."""
    dancer_frames = [(30.0 + step, 150.0, 170.0 + step, 550.0, 0.9) for step in range(6)]
    # Bigger than the dancer (76800 px vs 56000) and scored higher by the
    # detector, but off to one side. No overlap, so NMS keeps both and the
    # tracker has to make the call.
    rival = (190.0, 120.0, 350.0, 600.0, 0.99)
    script = [[dancer_frames[0]], [dancer_frames[1]]]
    script += [[dancer_frames[i], rival] for i in range(2, 5)]
    script += [[dancer_frames[5]]]

    adapter, _factory, _ = build_adapter(tmp_path, boxes_per_frame=script)
    report = extract(adapter, tmp_path, [0, 1, 2, 3, 4, 5])

    boxes = [p.source_bbox for p in load_pose_sequence(tmp_path / "pose")]
    for index, bbox in enumerate(boxes):
        assert bbox[0] == pytest.approx(dancer_frames[index][0], abs=0.5), index
    assert report["subject_tracking"]["switches"] == 0


def test_the_lock_holds_while_the_subject_is_briefly_lost() -> None:
    """A few frames with no detection must not release the track."""
    from app.adapters.dwpose.detector import Detection
    from app.adapters.dwpose.tracking import SubjectTracker

    tracker = SubjectTracker(
        TrackerSettings(max_coast_frames=4), frame_width=FRAME_W, frame_height=FRAME_H
    )
    incumbent = Detection(60.0, 200.0, 200.0, 560.0, 0.9)
    tracker.select([incumbent])
    for _ in range(3):
        assert tracker.select([])[0] is None

    big_central = Detection(100.0, 120.0, 280.0, 600.0, 0.9)
    chosen, _scores = tracker.select([incumbent, big_central])
    assert chosen == incumbent, "a 3-frame gap is not a reason to change subject"


def test_the_track_is_released_after_a_long_absence() -> None:
    """Past the coast window, holding on to a stale box is worse than starting
    again -- the dancer may genuinely have left and come back elsewhere."""
    from app.adapters.dwpose.detector import Detection
    from app.adapters.dwpose.tracking import SubjectTracker

    tracker = SubjectTracker(
        TrackerSettings(max_coast_frames=2), frame_width=FRAME_W, frame_height=FRAME_H
    )
    incumbent = Detection(60.0, 200.0, 200.0, 560.0, 0.9)
    tracker.select([incumbent])
    for _ in range(3):
        tracker.select([])

    big_central = Detection(100.0, 120.0, 280.0, 600.0, 0.9)
    chosen, scored = tracker.select([incumbent, big_central])
    assert chosen == big_central
    # History was discarded, so size and centrality decided it again.
    assert all(candidate.iou_term == 0.0 for candidate in scored)


def test_the_first_frame_is_decided_by_size_and_centrality_alone(tmp_path: Path) -> None:
    from app.adapters.dwpose.detector import Detection
    from app.adapters.dwpose.tracking import SubjectTracker

    tracker = SubjectTracker(TrackerSettings(), frame_width=FRAME_W, frame_height=FRAME_H)
    dancer = Detection(*DANCER[:4], score=DANCER[4])
    corner = Detection(*CORNER_IMAGE[:4], score=CORNER_IMAGE[4])
    chosen, scored = tracker.select([corner, dancer])

    assert chosen == dancer
    by_box = {s.detection.box: s for s in scored}
    # With no history both IoU and continuity are zero for everyone, so the
    # decision rests entirely on the two terms that describe "main subject".
    assert by_box[dancer.box].iou_term == 0.0
    assert by_box[dancer.box].continuity_term == 0.0
    assert by_box[dancer.box].area_term > by_box[corner.box].area_term
    assert by_box[dancer.box].center_term > by_box[corner.box].center_term


def test_subject_selection_is_deterministic(tmp_path: Path) -> None:
    """Same detections, same order out — twice, and independent of input order."""
    from app.adapters.dwpose.detector import Detection
    from app.adapters.dwpose.tracking import SubjectTracker

    detections = [
        Detection(*DANCER[:4], score=DANCER[4]),
        Detection(*CORNER_IMAGE[:4], score=CORNER_IMAGE[4]),
        Detection(*AVATAR[:4], score=AVATAR[4]),
    ]
    first = SubjectTracker(TrackerSettings(), frame_width=FRAME_W, frame_height=FRAME_H).select(
        detections
    )[0]
    second = SubjectTracker(TrackerSettings(), frame_width=FRAME_W, frame_height=FRAME_H).select(
        list(reversed(detections))
    )[0]
    assert first == second


# ---------------------------------------------------------------------------
# missing poses (requirement 10 / robustness)
# ---------------------------------------------------------------------------
def test_frames_with_no_detection_are_reported_not_faked(tmp_path: Path) -> None:
    script = [[DANCER], [], [], [DANCER]]
    adapter, _factory, _ = build_adapter(tmp_path, boxes_per_frame=script)
    report = extract(adapter, tmp_path, [0, 1, 2, 3])

    assert list_pose_indices(tmp_path / "pose") == [0, 3]
    assert report["missing_frames_sample"] == [1, 2]
    assert report["frames_without_pose"] == 2
    assert report["subject_tracking"]["frames_without_subject"] == 2


def test_a_whole_clip_with_no_subject_produces_no_poses(tmp_path: Path) -> None:
    adapter, _factory, _ = build_adapter(tmp_path, boxes_per_frame=[[], [], []])
    report = extract(adapter, tmp_path, [0, 1, 2])

    assert report["frames"] == 0
    assert report["frames_with_pose"] == 0
    assert report["confidence"]["mean"] == 0.0


def test_an_unreadable_clip_is_an_error_not_an_empty_success(tmp_path: Path) -> None:
    reader = dw.FakeFrameReader(truncate_after=-1)
    adapter, _factory, _ = build_adapter(tmp_path, reader=reader)
    with pytest.raises(ValidationError, match="No frames could be read"):
        extract(adapter, tmp_path, [0, 1])


def test_the_frames_directory_entry_point_is_refused(tmp_path: Path) -> None:
    """DWPose reads the clip. Extracting a motion reference's frames to disk is
    exactly what this phase is designed never to do."""
    adapter, _factory, _ = build_adapter(tmp_path)
    with pytest.raises(ValidationError, match="reads a video"):
        adapter.run(frames_dir=tmp_path, output_dir=tmp_path / "out", frame_indices=[0])


# ---------------------------------------------------------------------------
# only geometry is written (requirement 13)
# ---------------------------------------------------------------------------
def test_the_pose_directory_contains_json_and_nothing_else(tmp_path: Path) -> None:
    adapter, _factory, _ = build_adapter(tmp_path)
    extract(adapter, tmp_path, [0, 1, 2])
    written = sorted(p.suffix for p in (tmp_path / "pose").rglob("*") if p.is_file())
    assert set(written) == {".json"}
