"""The DWPose building blocks, tested on their own.

Decoders, geometry and temporal cleanup are pure functions over arrays, so they
can be pinned precisely — which matters because a silent error in any of them
produces pose files that look entirely plausible and are wrong.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.adapters.dwpose import detector as det
from app.adapters.dwpose import keypoints as kp
from app.adapters.dwpose import pose_model as pm
from app.adapters.dwpose import roi as roi_module
from app.adapters.dwpose.session import (
    CPU_PROVIDER,
    CUDA_PROVIDER,
    ModelFileMissingError,
    assert_model_file,
    resolve_providers,
)
from app.adapters.dwpose.temporal import (
    CleanupSettings,
    clean_sequence,
    interpolate_short_gaps,
    smooth_track,
)
from app.core.errors import ValidationError
from app.motion.pose_format import Joint2D, PoseFrame
from app.motion.skeleton import BODY_JOINTS
from tests import dwpose_fixtures as dw


# ---------------------------------------------------------------------------
# keypoint vocabulary
# ---------------------------------------------------------------------------
def test_the_coco_table_covers_every_canonical_joint_exactly_once() -> None:
    assert set(kp.COCO_BODY_INDEX) == set(BODY_JOINTS)
    indices = kp.body_indices()
    assert indices == list(range(17)), "COCO-17 is the first 17 whole-body points"
    assert len(set(indices)) == 17, "no two joints may share a keypoint index"


def test_the_named_joints_sit_at_their_documented_indices() -> None:
    """Spot-check the pairs that are easiest to transpose."""
    assert kp.COCO_BODY_INDEX["nose"] == 0
    assert kp.COCO_BODY_INDEX["left_eye"] < kp.COCO_BODY_INDEX["right_eye"]
    assert kp.COCO_BODY_INDEX["left_ear"] == 3 and kp.COCO_BODY_INDEX["right_ear"] == 4
    assert kp.COCO_BODY_INDEX["left_shoulder"] == 5
    assert kp.COCO_BODY_INDEX["left_wrist"] == 9 and kp.COCO_BODY_INDEX["right_wrist"] == 10
    assert kp.COCO_BODY_INDEX["left_ankle"] == 15 and kp.COCO_BODY_INDEX["right_ankle"] == 16


@pytest.mark.parametrize(("count", "ok"), [(17, True), (26, True), (133, True), (21, False)])
def test_keypoint_counts_are_validated(count: int, ok: bool) -> None:
    if ok:
        kp.validate_keypoint_count(count)
    else:
        with pytest.raises(ValidationError):
            kp.validate_keypoint_count(count)


def test_hand_ranges_are_21_points_each() -> None:
    assert kp.LEFT_HAND_RANGE[1] - kp.LEFT_HAND_RANGE[0] == 21
    assert kp.RIGHT_HAND_RANGE[1] - kp.RIGHT_HAND_RANGE[0] == 21
    assert kp.RIGHT_HAND_RANGE[1] == kp.COCO_WHOLEBODY_133
    assert kp.has_hands(133) and not kp.has_hands(17)


# ---------------------------------------------------------------------------
# detector geometry
# ---------------------------------------------------------------------------
def test_letterbox_preserves_aspect_ratio_and_anchors_top_left() -> None:
    image = np.zeros((640, 360, 3), dtype=np.uint8)
    canvas, info = det.letterbox(image, (64, 64))

    assert canvas.shape == (64, 64, 3)
    assert info.scale == pytest.approx(0.1)
    assert (info.pad_x, info.pad_y) == (0, 0)
    # Padding fills the unused strip, so the decode has something inert there.
    assert canvas[:, 40:].max() == 114


def test_yolox_decode_recovers_the_box_it_was_given() -> None:
    """Encode a known box backwards through YOLOX's convention, decode it."""
    box = (100.0, 200.0, 260.0, 600.0, 0.9)
    raw = dw.yolox_output([box], scale=0.1)
    table = det.decode_yolox(raw, input_size=dw.DETECTOR_INPUT, strides=dw.DETECTOR_STRIDES)

    best = int(np.argmax(table[:, 4] * table[:, 5]))
    assert table[best, 0] == pytest.approx(10.0, abs=0.01)  # x1 * 0.1
    assert table[best, 2] == pytest.approx(26.0, abs=0.01)


def test_decode_rejects_an_output_that_does_not_match_the_stride_pyramid() -> None:
    raw = np.zeros((1, 7, 85), dtype=np.float32)
    with pytest.raises(ValidationError, match="stride pyramid"):
        det.decode_yolox(raw, input_size=dw.DETECTOR_INPUT, strides=dw.DETECTOR_STRIDES)


def test_nms_keeps_the_best_of_an_overlapping_pair() -> None:
    boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105], [500, 500, 600, 600]], dtype=np.float32)
    scores = np.array([0.8, 0.9, 0.7], dtype=np.float32)
    assert det.nms(boxes, scores, 0.45) == [1, 2]


def test_nms_is_deterministic_for_tied_scores() -> None:
    boxes = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=np.float32)
    scores = np.array([0.5, 0.5], dtype=np.float32)
    assert det.nms(boxes, scores, 0.45) == det.nms(boxes, scores, 0.45) == [0, 1]


def test_only_the_person_class_survives() -> None:
    raw = dw.yolox_output([(10.0, 10.0, 60.0, 120.0, 0.95)], person_class=15, scale=1.0)
    people = det.detections_from_output(
        raw,
        layout="yolox",
        input_size=dw.DETECTOR_INPUT,
        strides=dw.DETECTOR_STRIDES,
        person_class=0,
        score_threshold=0.3,
        iou_threshold=0.45,
        max_detections=10,
        letterbox_info=det.LetterboxInfo(scale=1.0, pad_x=0, pad_y=0),
    )
    assert people == [], "a cat scored 0.95 is still not a person"


def test_a_person_class_outside_the_model_range_is_refused() -> None:
    raw = dw.yolox_output([(10.0, 10.0, 60.0, 120.0, 0.9)], num_classes=1, scale=1.0)
    with pytest.raises(ValidationError, match="class range"):
        det.detections_from_output(
            raw,
            layout="yolox",
            input_size=dw.DETECTOR_INPUT,
            strides=dw.DETECTOR_STRIDES,
            person_class=5,
            score_threshold=0.3,
            iou_threshold=0.45,
            max_detections=10,
            letterbox_info=det.LetterboxInfo(scale=1.0, pad_x=0, pad_y=0),
        )


# ---------------------------------------------------------------------------
# pose decoding and crop inversion
# ---------------------------------------------------------------------------
def test_simcc_decode_finds_the_peak_bins() -> None:
    points = [(10.0, 20.0), (30.5, 5.0)]
    outputs = dw.simcc_output(points, [0.9, 0.4])
    coords, scores = pm.decode_simcc(outputs[0], outputs[1], split_ratio=dw.SPLIT_RATIO)

    assert coords[0] == pytest.approx([10.0, 20.0], abs=0.5)
    assert coords[1] == pytest.approx([30.5, 5.0], abs=0.5)
    assert scores == pytest.approx([0.9, 0.4])


def test_simcc_confidence_is_the_weaker_of_the_two_axes() -> None:
    """A keypoint is only as certain as its least certain coordinate."""
    simcc_x = np.full((1, 1, 32), -1.0, dtype=np.float32)
    simcc_y = np.full((1, 1, 32), -1.0, dtype=np.float32)
    simcc_x[0, 0, 8] = 0.95
    simcc_y[0, 0, 4] = 0.30
    _coords, scores = pm.decode_simcc(simcc_x, simcc_y, split_ratio=2.0)
    assert scores[0] == pytest.approx(0.30)


def test_heatmap_and_direct_keypoint_outputs_are_both_understood() -> None:
    heatmaps = np.zeros((1, 2, 8, 6), dtype=np.float32)
    heatmaps[0, 0, 4, 3] = 0.8
    heatmaps[0, 1, 1, 5] = 0.6
    coords, scores = pm.decode_outputs([heatmaps], input_size=(48, 64), split_ratio=2.0)
    assert coords[0] == pytest.approx([24.0, 32.0])
    assert scores[1] == pytest.approx(0.6)

    direct = np.array([[[11.0, 22.0, 0.7], [33.0, 44.0, 0.5]]], dtype=np.float32)
    coords, scores = pm.decode_outputs([direct], input_size=(48, 64), split_ratio=2.0)
    assert coords[1] == pytest.approx([33.0, 44.0])
    assert scores[0] == pytest.approx(0.7)


def test_an_unrecognised_output_shape_is_refused_not_guessed() -> None:
    with pytest.raises(ValidationError, match="Unrecognised pose model output"):
        pm.decode_outputs(
            [np.zeros((3, 4, 5, 6, 7), dtype=np.float32)], input_size=(48, 64), split_ratio=2.0
        )


def test_the_crop_keeps_the_models_aspect_ratio() -> None:
    crop = pm.expand_box((100.0, 100.0, 200.0, 500.0), input_size=(48, 64), padding=1.0)
    assert crop.width / crop.height == pytest.approx(48 / 64)
    # Padding the *narrow* axis keeps the subject inside the frame.
    assert crop.height == pytest.approx(400.0)
    assert crop.width == pytest.approx(300.0)


def test_crop_inversion_round_trips() -> None:
    crop = pm.expand_box((100.0, 150.0, 260.0, 550.0), input_size=(48, 64), padding=1.25)
    model_points = np.array([[0.0, 0.0], [48.0, 64.0], [24.0, 32.0]], dtype=np.float32)
    image_points = pm.keypoints_to_image(model_points, crop)

    assert image_points[0] == pytest.approx([crop.x, crop.y])
    assert image_points[1] == pytest.approx([crop.x + crop.width, crop.y + crop.height])
    # The centre of the crop is the centre of the original box.
    assert image_points[2] == pytest.approx([180.0, 350.0], abs=0.01)


def test_a_crop_running_off_the_frame_is_padded_not_clipped() -> None:
    """Clipping would move the rectangle and break the inverse mapping."""
    image = np.full((100, 100, 3), 200, dtype=np.uint8)
    crop = pm.CropInfo(x=-40.0, y=-30.0, width=80.0, height=60.0, input_size=(16, 12))
    patch = pm.crop_and_resize(image, crop)
    assert patch.shape == (12, 16, 3)
    assert patch[0, 0].tolist() == [0, 0, 0], "outside the frame is padding"
    assert patch[-1, -1].tolist() == [200, 200, 200]


# ---------------------------------------------------------------------------
# ROI
# ---------------------------------------------------------------------------
def test_roi_none_is_the_whole_frame() -> None:
    region = roi_module.resolve_roi("none", (0, 0, 0, 0), frame_width=360, frame_height=640)
    assert region.is_identity
    assert (region.width, region.height) == (360, 640)
    assert roi_module.restore_point((12.0, 34.0), region) == (12.0, 34.0)


def test_pixel_and_normalized_rois_agree() -> None:
    pixels = roi_module.resolve_roi(
        "pixels", (36, 128, 180, 384), frame_width=360, frame_height=640
    )
    fractions = roi_module.resolve_roi(
        "normalized", (0.1, 0.2, 0.5, 0.6), frame_width=360, frame_height=640
    )
    assert (pixels.x, pixels.y, pixels.width, pixels.height) == (
        fractions.x,
        fractions.y,
        fractions.width,
        fractions.height,
    )


def test_an_roi_running_off_the_frame_is_clipped() -> None:
    region = roi_module.resolve_roi(
        "pixels", (300, 600, 500, 500), frame_width=360, frame_height=640
    )
    assert (region.x, region.y) == (300, 600)
    assert (region.width, region.height) == (60, 40)


def test_an_roi_outside_the_frame_is_refused() -> None:
    with pytest.raises(ValidationError, match="does not overlap"):
        roi_module.resolve_roi("pixels", (900, 900, 100, 100), frame_width=360, frame_height=640)


def test_restoration_undoes_the_crop_for_points_and_boxes() -> None:
    region = roi_module.resolve_roi("pixels", (40, 90, 280, 500), frame_width=360, frame_height=640)
    assert roi_module.restore_point((10.0, 20.0), region) == (50.0, 110.0)
    assert roi_module.restore_box((1.0, 2.0, 3.0, 4.0), region) == (41.0, 92.0, 43.0, 94.0)


# ---------------------------------------------------------------------------
# temporal cleanup
# ---------------------------------------------------------------------------
def joint(x: float, y: float, confidence: float = 0.9) -> Joint2D:
    return Joint2D(x=x, y=y, confidence=confidence)


def test_a_short_gap_is_interpolated_linearly() -> None:
    track = [joint(0.0, 0.0), None, None, joint(30.0, 60.0)]
    filled, count, longest = interpolate_short_gaps(track, max_gap=3, confidence_scale=0.5)

    assert count == 2 and longest == 2
    assert filled[1].x == pytest.approx(10.0) and filled[1].y == pytest.approx(20.0)
    assert filled[2].x == pytest.approx(20.0)
    # Filled joints are marked as less certain than measured ones.
    assert filled[1].confidence == pytest.approx(0.45)


def test_a_long_gap_is_left_missing() -> None:
    """Interpolating across a long run invents motion and hides a bad clip."""
    track = [joint(0.0, 0.0)] + [None] * 8 + [joint(80.0, 0.0)]
    filled, count, longest = interpolate_short_gaps(track, max_gap=3, confidence_scale=0.6)
    assert count == 0
    assert longest == 8
    assert filled[1:9] == [None] * 8


def test_gaps_at_the_ends_are_not_extrapolated() -> None:
    track = [None, joint(5.0, 5.0), None]
    filled, count, _ = interpolate_short_gaps(track, max_gap=3, confidence_scale=0.6)
    assert count == 0
    assert filled[0] is None and filled[2] is None


def test_smoothing_reduces_jitter_on_a_slow_joint() -> None:
    track = [joint(10.0, 0.0), joint(12.0, 0.0), joint(10.0, 0.0), joint(12.0, 0.0)]
    smoothed, count, skipped = smooth_track(track, window=2, strength=1.0, fast_motion_px=50.0)
    assert count == 4 and skipped == 0
    spread_before = max(j.x for j in track) - min(j.x for j in track)
    spread_after = max(j.x for j in smoothed) - min(j.x for j in smoothed)
    assert spread_after < spread_before


def test_a_fast_hand_is_not_averaged_into_a_stump() -> None:
    """40px per frame is a dancer's wrist, not noise. Smoothing must stand off."""
    track = [joint(0.0, 0.0), joint(40.0, 0.0), joint(80.0, 0.0), joint(120.0, 0.0)]
    smoothed, count, skipped = smooth_track(track, window=2, strength=1.0, fast_motion_px=18.0)

    assert count == 0 and skipped == 4
    assert [j.x for j in smoothed] == [0.0, 40.0, 80.0, 120.0]


def test_smoothing_never_reaches_across_a_gap() -> None:
    track = [joint(0.0, 0.0), joint(1.0, 0.0), None, joint(500.0, 0.0), joint(501.0, 0.0)]
    smoothed, _count, _skipped = smooth_track(track, window=3, strength=1.0, fast_motion_px=1000.0)
    assert smoothed[2] is None
    # The far-away run did not pull the first run toward it.
    assert smoothed[0].x < 5.0
    assert smoothed[3].x > 495.0


def test_smoothing_is_confidence_weighted() -> None:
    """A confident neighbour pulls harder than an uncertain one."""
    track = [joint(0.0, 0.0, 0.05), joint(10.0, 0.0, 0.99), joint(0.0, 0.0, 0.05)]
    smoothed, _count, _skipped = smooth_track(track, window=1, strength=1.0, fast_motion_px=1000.0)
    assert smoothed[0].x > 5.0, "pulled toward the confident observation"


def test_cleanup_reports_what_it_did_per_joint() -> None:
    poses = []
    for index in range(6):
        body = {name: joint(float(index), float(index)) for name in BODY_JOINTS}
        if index in (2, 3):
            body.pop("left_wrist")
        poses.append(PoseFrame(frame_index=index, timestamp_s=index / 30.0, body=body))

    cleaned, report = clean_sequence(
        poses, CleanupSettings(max_interpolation_gap=3, smoothing_window=0)
    )
    assert report.interpolated["left_wrist"] == 2
    assert report.long_runs["left_wrist"] == 2
    assert all("left_wrist" in pose.body for pose in cleaned)
    assert report.as_dict()["total_interpolated"] == 2


def test_cleanup_preserves_frame_indices_and_order() -> None:
    poses = [
        PoseFrame(
            frame_index=index,
            timestamp_s=index / 30.0,
            body={name: joint(1.0, 2.0) for name in BODY_JOINTS},
        )
        for index in (40, 41, 42)
    ]
    cleaned, _report = clean_sequence(poses, CleanupSettings())
    assert [p.frame_index for p in cleaned] == [40, 41, 42]


def test_cleanup_of_an_empty_sequence_is_a_no_op() -> None:
    cleaned, report = clean_sequence([], CleanupSettings())
    assert cleaned == []
    assert report.as_dict()["total_interpolated"] == 0


# ---------------------------------------------------------------------------
# provider resolution and model paths
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("requested", "available", "expected"),
    [
        ("auto", [CPU_PROVIDER], [CPU_PROVIDER]),
        ("auto", [CUDA_PROVIDER, CPU_PROVIDER], [CUDA_PROVIDER, CPU_PROVIDER]),
        ("cuda", [CUDA_PROVIDER, CPU_PROVIDER], [CUDA_PROVIDER, CPU_PROVIDER]),
        ("cuda", [CPU_PROVIDER], [CPU_PROVIDER]),
        ("cpu", [CUDA_PROVIDER, CPU_PROVIDER], [CPU_PROVIDER]),
    ],
)
def test_provider_priority(requested: str, available: list[str], expected: list[str]) -> None:
    assert resolve_providers(requested, available) == expected


def test_an_empty_model_path_says_what_to_do(tmp_path: Path) -> None:
    with pytest.raises(ModelFileMissingError) as exc:
        assert_model_file("", role="pose", hint="put it somewhere")
    assert "never downloads" in exc.value.message
    assert exc.value.details["hint"] == "put it somewhere"


def test_a_present_model_path_resolves(tmp_path: Path) -> None:
    detector, _pose = dw.touch_models(tmp_path)
    assert assert_model_file(detector, role="detector", hint="") == detector
