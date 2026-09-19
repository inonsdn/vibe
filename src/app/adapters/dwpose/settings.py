"""Resolved DWPose settings: config, then environment, then CLI flags.

Kept apart from :class:`~app.core.config.PoseConfig` so the adapter can be
constructed directly in a test with a couple of overrides, without building a
whole ``AppConfig``. :func:`from_app_config` is the one place the two are
reconciled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.adapters.dwpose.temporal import CleanupSettings
from app.adapters.dwpose.tracking import TrackerSettings
from app.core.config import AppConfig, PoseConfig


@dataclass
class DWPoseSettings:
    """Everything the adapter needs, already resolved."""

    detector_model: str = ""
    pose_model: str = ""
    provider: str = "auto"
    require_requested_provider: bool = True
    intra_op_threads: int = 0

    detector_input_size: tuple[int, int] = (640, 640)
    detector_layout: str = "yolox"
    detector_strides: tuple[int, ...] = (8, 16, 32)
    detector_person_class: int = 0
    detection_score_threshold: float = 0.3
    nms_iou_threshold: float = 0.45
    max_detections: int = 20

    pose_input_size: tuple[int, int] = (288, 384)
    simcc_split_ratio: float = 2.0
    bbox_padding: float = 1.25
    keypoint_score_threshold: float = 0.3
    emit_hands: bool = True

    roi_mode: str = "none"
    roi: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    tracker: TrackerSettings = field(default_factory=TrackerSettings)
    cleanup: CleanupSettings = field(default_factory=CleanupSettings)

    diagnostics_crf: int = 26
    diagnostics_scale: float = 0.5

    def as_dict(self) -> dict[str, Any]:
        """Manifest-friendly view. Model *paths* are deliberately excluded —
        they are machine-specific; the file hashes identify the weights."""
        return {
            "provider_requested": self.provider,
            "detector_input_size": list(self.detector_input_size),
            "detector_layout": self.detector_layout,
            "detection_score_threshold": self.detection_score_threshold,
            "nms_iou_threshold": self.nms_iou_threshold,
            "pose_input_size": list(self.pose_input_size),
            "simcc_split_ratio": self.simcc_split_ratio,
            "bbox_padding": self.bbox_padding,
            "keypoint_score_threshold": self.keypoint_score_threshold,
            "emit_hands": self.emit_hands,
            "roi_mode": self.roi_mode,
            "roi": list(self.roi),
            "subject_selection": {
                "area_weight": self.tracker.area_weight,
                "center_weight": self.tracker.center_weight,
                "iou_weight": self.tracker.iou_weight,
                "continuity_weight": self.tracker.continuity_weight,
                "min_area_fraction": self.tracker.min_area_fraction,
                "max_center_distance": self.tracker.max_center_distance,
                "switch_margin": self.tracker.switch_margin,
                "max_coast_frames": self.tracker.max_coast_frames,
            },
            "temporal_cleanup": {
                "max_interpolation_gap": self.cleanup.max_interpolation_gap,
                "interpolation_confidence_scale": self.cleanup.interpolation_confidence_scale,
                "smoothing_window": self.cleanup.smoothing_window,
                "smoothing_strength": self.cleanup.smoothing_strength,
                "fast_motion_px": self.cleanup.fast_motion_px,
            },
        }


def from_pose_config(pose: PoseConfig, **overrides: Any) -> DWPoseSettings:
    """Build settings from a :class:`PoseConfig`, with CLI-level overrides.

    ``overrides`` with a ``None`` value are ignored, so a CLI flag that was not
    passed cannot blank out a configured value.
    """
    settings = DWPoseSettings(
        detector_model=pose.detector_model,
        pose_model=pose.pose_model,
        provider=pose.provider,
        require_requested_provider=pose.require_requested_provider,
        intra_op_threads=pose.intra_op_threads,
        detector_input_size=tuple(pose.detector_input_size),  # type: ignore[arg-type]
        detector_layout=pose.detector_layout,
        detector_strides=tuple(pose.detector_strides),
        detector_person_class=pose.detector_person_class,
        detection_score_threshold=pose.detection_score_threshold,
        nms_iou_threshold=pose.nms_iou_threshold,
        max_detections=pose.max_detections,
        pose_input_size=tuple(pose.pose_input_size),  # type: ignore[arg-type]
        simcc_split_ratio=pose.simcc_split_ratio,
        bbox_padding=pose.bbox_padding,
        keypoint_score_threshold=pose.keypoint_score_threshold,
        emit_hands=pose.emit_hands,
        roi_mode=pose.roi_mode,
        roi=tuple(pose.roi),  # type: ignore[arg-type]
        tracker=TrackerSettings(
            area_weight=pose.subject_area_weight,
            center_weight=pose.subject_center_weight,
            iou_weight=pose.subject_iou_weight,
            continuity_weight=pose.subject_continuity_weight,
            min_area_fraction=pose.subject_min_area_fraction,
            max_center_distance=pose.subject_max_center_distance,
            switch_margin=pose.subject_switch_margin,
            max_coast_frames=pose.subject_max_coast_frames,
        ),
        cleanup=CleanupSettings(
            max_interpolation_gap=pose.max_interpolation_gap,
            interpolation_confidence_scale=pose.interpolation_confidence_scale,
            smoothing_window=pose.smoothing_window,
            smoothing_strength=pose.smoothing_strength,
            fast_motion_px=pose.fast_motion_px,
        ),
        diagnostics_crf=pose.diagnostics_crf,
        diagnostics_scale=pose.diagnostics_scale,
    )
    for key, value in overrides.items():
        if value is None:
            continue
        if not hasattr(settings, key):
            from app.core.errors import ValidationError

            raise ValidationError("Unknown DWPose setting override", setting=key)
        setattr(settings, key, value)
    return settings


def from_app_config(config: AppConfig, **overrides: Any) -> DWPoseSettings:
    return from_pose_config(config.pose, **overrides)


__all__ = ["DWPoseSettings", "from_app_config", "from_pose_config"]
