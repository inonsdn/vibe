"""The DWPose ONNX pose adapter.

Per requested source frame:

1. read the frame (absolute index preserved end to end),
2. crop to the configured ROI, if any,
3. detect people,
4. choose the **main dancer** with the deterministic tracker,
5. run the keypoint model on that one box,
6. map keypoints back through the crop and the ROI into original video pixels,
7. convert COCO-WholeBody indices into canonical COCO-17 joint names.

Afterwards the whole sequence is cleaned temporally and written as pose JSON.

**Only geometry is written.** The pose directory receives JSON and nothing else;
the diagnostic overlay — which does contain source pixels — is written to a
separate diagnostics tree that is never a composition input. See
:mod:`app.adapters.dwpose.diagnostics`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.adapters.base import (
    AdapterCapability,
    AdapterKind,
    AdapterStatus,
    AnalysisAdapter,
)
from app.adapters.dwpose import detector as det
from app.adapters.dwpose import keypoints as kp
from app.adapters.dwpose import pose_model as pm
from app.adapters.dwpose import roi as roi_module
from app.adapters.dwpose.session import (
    CPU_PROVIDER,
    ModelFileMissingError,
    OnnxSessionFactory,
    ProviderResolution,
    assert_model_file,
    create_session,
)
from app.adapters.dwpose.settings import DWPoseSettings, from_app_config
from app.adapters.dwpose.temporal import CleanupReport, clean_sequence
from app.adapters.dwpose.tracking import SubjectScore, SubjectTracker
from app.adapters.dwpose.video import FrameReader, OpenCVFrameReader
from app.core.errors import ValidationError
from app.core.hashing import sha256_file
from app.core.logging import get_logger, log_event
from app.motion.pose_format import Joint2D, PoseFrame, save_pose_sequence
from app.motion.skeleton import BODY_JOINTS

logger = get_logger(__name__)

ADAPTER_NAME = "dwpose_onnx"
ADAPTER_VERSION = "1.0.0"

MODEL_HINT = (
    "Place the ONNX files on this machine and point at them with "
    "pose.detector_model / pose.pose_model in config/local.yaml, the "
    "APP_POSE__DETECTOR_MODEL / APP_POSE__POSE_MODEL environment variables, or "
    "--detector-model / --pose-model. See docs/dwpose-setup.md. This "
    "application never downloads weights."
)


@dataclass
class FrameOutcome:
    """Everything one frame produced, including why it produced nothing."""

    frame_index: int
    pose: PoseFrame | None
    subject: det.Detection | None
    candidates: list[SubjectScore] = field(default_factory=list)
    reason: str | None = None


@dataclass
class ExtractionResult:
    """The adapter's own report. Persisted into the motion source record."""

    adapter: str
    version: str
    frames_requested: int
    frames_with_pose: int
    detector_provider: ProviderResolution | None
    pose_provider: ProviderResolution | None
    settings: dict[str, Any]
    model_hashes: dict[str, str]
    tracker: dict[str, Any]
    cleanup: dict[str, Any]
    missing_frames: list[int]
    confidence: dict[str, float]
    roi: dict[str, Any]
    duration_s: float
    frame_size: tuple[int, int] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "version": self.version,
            "frames_requested": self.frames_requested,
            "frames_with_pose": self.frames_with_pose,
            "frames_without_pose": len(self.missing_frames),
            "missing_frames_sample": self.missing_frames[:32],
            # The provider is recorded because a silent CPU fallback looks
            # identical in the output and is ~20x slower.
            "provider": {
                "detector": self.detector_provider.as_dict() if self.detector_provider else None,
                "pose": self.pose_provider.as_dict() if self.pose_provider else None,
            },
            "model_sha256": self.model_hashes,
            "settings": self.settings,
            "subject_tracking": self.tracker,
            "temporal_cleanup": self.cleanup,
            "confidence": self.confidence,
            "roi": self.roi,
            "frame_size": list(self.frame_size) if self.frame_size else None,
            "duration_s": round(self.duration_s, 3),
            "contains_source_pixels": False,
        }


class DWPoseOnnxAdapter(AnalysisAdapter):
    """Real pose extraction from local ONNX models. No downloads, ever."""

    kind = AdapterKind.POSE
    name = ADAPTER_NAME
    version = ADAPTER_VERSION

    def __init__(
        self,
        settings: DWPoseSettings,
        *,
        session_factory: OnnxSessionFactory | None = None,
        frame_reader: FrameReader | None = None,
    ) -> None:
        self.settings = settings
        self._factory = session_factory
        self._reader = frame_reader or OpenCVFrameReader()
        self._detector_session: Any = None
        self._pose_session: Any = None
        self._detector_provider: ProviderResolution | None = None
        self._pose_provider: ProviderResolution | None = None
        self.last_result: ExtractionResult | None = None
        self.last_outcomes: list[FrameOutcome] = []

    # -- capability -------------------------------------------------------
    def capability(self) -> AdapterCapability:
        problems: list[str] = []
        for role, path in (
            ("detector", self.settings.detector_model),
            ("pose", self.settings.pose_model),
        ):
            try:
                assert_model_file(path, role=role, hint=MODEL_HINT)
            except ModelFileMissingError as exc:
                problems.append(exc.message)

        status = AdapterStatus.MISSING_WEIGHTS if problems else AdapterStatus.AVAILABLE
        reason = (
            "DWPose ONNX models are configured and present."
            if not problems
            else " ".join(problems) + " " + MODEL_HINT
        )
        return AdapterCapability(
            kind=self.kind,
            name=self.name,
            status=status,
            reason=reason,
            requires_gpu=False,
            estimated_vram_mb=1200,
            expected_outputs=("frame_{index:06d}.json (internal pose format)",),
            notes={
                "synthetic": False,
                "version": ADAPTER_VERSION,
                "provider_requested": self.settings.provider,
                "downloads": "none - model files are supplied by the operator",
                "detector_model_configured": bool(self.settings.detector_model),
                "pose_model_configured": bool(self.settings.pose_model),
                "skeleton": "coco_17 (from COCO-WholeBody indices 0..16)",
            },
        )

    # -- sessions ---------------------------------------------------------
    def _factory_or_default(self) -> OnnxSessionFactory:
        if self._factory is None:
            from app.adapters.dwpose.session import OnnxRuntimeSessionFactory

            self._factory = OnnxRuntimeSessionFactory(
                intra_op_threads=self.settings.intra_op_threads
            )
        return self._factory

    def ensure_sessions(self) -> tuple[ProviderResolution, ProviderResolution]:
        """Create both sessions, recording the provider each really uses."""
        detector_path = assert_model_file(
            self.settings.detector_model, role="detector", hint=MODEL_HINT
        )
        pose_path = assert_model_file(self.settings.pose_model, role="pose", hint=MODEL_HINT)
        factory = self._factory_or_default()

        if self._detector_session is None:
            self._detector_session, self._detector_provider = create_session(
                factory,
                detector_path,
                requested=self.settings.provider,
                require_requested=self.settings.require_requested_provider,
                role="detector",
            )
        if self._pose_session is None:
            self._pose_session, self._pose_provider = create_session(
                factory,
                pose_path,
                requested=self.settings.provider,
                require_requested=self.settings.require_requested_provider,
                role="pose",
            )
        assert self._detector_provider is not None and self._pose_provider is not None
        return self._detector_provider, self._pose_provider

    @staticmethod
    def _feed(session: Any, batch: np.ndarray) -> list[Any]:
        inputs = session.get_inputs()
        if not inputs:
            raise ValidationError("ONNX session reports no inputs")
        return session.run(None, {inputs[0].name: batch})

    # -- per-frame work ---------------------------------------------------
    def detect(self, image: np.ndarray) -> list[det.Detection]:
        batch, info = det.preprocess(image, self.settings.detector_input_size)
        outputs = self._feed(self._detector_session, batch)
        return det.detections_from_output(
            outputs[0],
            layout=self.settings.detector_layout,
            input_size=self.settings.detector_input_size,
            strides=self.settings.detector_strides,
            person_class=self.settings.detector_person_class,
            score_threshold=self.settings.detection_score_threshold,
            iou_threshold=self.settings.nms_iou_threshold,
            max_detections=self.settings.max_detections,
            letterbox_info=info,
        )

    def keypoints_for(
        self, image: np.ndarray, box: tuple[float, float, float, float]
    ) -> tuple[np.ndarray, np.ndarray]:
        crop = pm.expand_box(
            box, input_size=self.settings.pose_input_size, padding=self.settings.bbox_padding
        )
        batch = pm.preprocess(image, crop)
        outputs = self._feed(self._pose_session, batch)
        coords, scores = pm.decode_outputs(
            [np.asarray(o) for o in outputs],
            input_size=self.settings.pose_input_size,
            split_ratio=self.settings.simcc_split_ratio,
        )
        kp.validate_keypoint_count(int(coords.shape[0]))
        return pm.keypoints_to_image(coords, crop), scores

    def build_pose(
        self,
        frame_index: int,
        fps: float,
        coords: np.ndarray,
        scores: np.ndarray,
        region: roi_module.Roi,
        subject: det.Detection,
    ) -> PoseFrame:
        """Assemble a canonical PoseFrame in ORIGINAL video coordinates."""
        threshold = self.settings.keypoint_score_threshold
        body: dict[str, Joint2D] = {}
        for name, index in zip(BODY_JOINTS, kp.body_indices(), strict=True):
            score = float(scores[index])
            if score < threshold:
                continue
            x, y = roi_module.restore_point(
                (float(coords[index][0]), float(coords[index][1])), region
            )
            body[name] = Joint2D(x=x, y=y, confidence=min(1.0, max(0.0, score)))

        hands: dict[str, Joint2D] = {}
        if self.settings.emit_hands and kp.has_hands(int(coords.shape[0])):
            for side, (start, end) in (
                ("left", kp.LEFT_HAND_RANGE),
                ("right", kp.RIGHT_HAND_RANGE),
            ):
                for offset, key in enumerate(kp.hand_point_names(side)):
                    index = start + offset
                    if index >= end:  # pragma: no cover - ranges are 21 wide
                        break
                    score = float(scores[index])
                    if score < threshold:
                        continue
                    x, y = roi_module.restore_point(
                        (float(coords[index][0]), float(coords[index][1])), region
                    )
                    hands[key] = Joint2D(x=x, y=y, confidence=min(1.0, max(0.0, score)))

        x1, y1, x2, y2 = roi_module.restore_box(subject.box, region)
        return PoseFrame(
            frame_index=frame_index,
            timestamp_s=frame_index / max(fps, 1e-6),
            body=body,
            hands=hands,
            source_bbox=(x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)),
        )

    # -- entry points -----------------------------------------------------
    def estimate_sequence(
        self,
        *,
        video_path: Path,
        output_dir: Path,
        frame_indices: list[int],
        fps: float,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Extract poses for exactly ``frame_indices`` of ``video_path``."""
        started = time.perf_counter()
        opts = options or {}
        wanted = [int(i) for i in frame_indices]
        if not wanted:
            raise ValidationError("No frames requested for pose extraction")

        detector_provider, pose_provider = self.ensure_sessions()
        region: roi_module.Roi | None = None
        tracker: SubjectTracker | None = None
        outcomes: list[FrameOutcome] = []
        poses: list[PoseFrame] = []
        frame_size: tuple[int, int] | None = None

        for frame_index, frame in self._reader.read(Path(video_path), wanted):
            if frame_size is None:
                height, width = frame.shape[:2]
                frame_size = (width, height)
                region = roi_module.resolve_roi(
                    self.settings.roi_mode,
                    self.settings.roi,
                    frame_width=width,
                    frame_height=height,
                )
                tracker = SubjectTracker(
                    self.settings.tracker, frame_width=region.width, frame_height=region.height
                )
            assert region is not None and tracker is not None

            cropped = (
                frame
                if region.mode == "none"
                else frame[region.y : region.y + region.height, region.x : region.x + region.width]
            )
            detections = self.detect(cropped)
            subject, candidates = tracker.select(detections)
            if subject is None:
                outcomes.append(
                    FrameOutcome(
                        frame_index=frame_index,
                        pose=None,
                        subject=None,
                        candidates=candidates,
                        reason="no_subject" if detections else "no_detections",
                    )
                )
                continue

            coords, scores = self.keypoints_for(cropped, subject.box)
            pose = self.build_pose(frame_index, fps, coords, scores, region, subject)
            poses.append(pose)
            outcomes.append(
                FrameOutcome(
                    frame_index=frame_index, pose=pose, subject=subject, candidates=candidates
                )
            )

        if region is None or tracker is None:
            raise ValidationError(
                "No frames could be read from the reference video",
                video=str(video_path),
                requested=len(wanted),
            )

        cleaned, cleanup = clean_sequence(poses, self.settings.cleanup)
        written = save_pose_sequence(output_dir, cleaned)

        result = self._build_result(
            wanted=wanted,
            cleaned=cleaned,
            outcomes=outcomes,
            cleanup=cleanup,
            tracker=tracker,
            region=region,
            frame_size=frame_size,
            detector_provider=detector_provider,
            pose_provider=pose_provider,
            duration_s=time.perf_counter() - started,
        )
        self.last_result = result
        self.last_outcomes = outcomes

        payload = result.as_dict()
        payload.update(
            {
                "frames": len(written),
                "first_frame": written[0] if written else None,
                "last_frame": written[-1] if written else None,
                "output_dir": str(output_dir),
                "input": str(video_path),
            }
        )
        log_event(
            logger,
            "dwpose_extraction_completed",
            frames=len(written),
            requested=len(wanted),
            detector_provider=detector_provider.active,
            pose_provider=pose_provider.active,
            fell_back=detector_provider.fell_back or pose_provider.fell_back,
        )
        if opts.get("diagnostics_dir"):
            from app.adapters.dwpose.diagnostics import write_diagnostics

            payload["diagnostics"] = write_diagnostics(
                Path(opts["diagnostics_dir"]),
                video_path=Path(video_path),
                outcomes=outcomes,
                poses=cleaned,
                result=result,
                region=region,
                fps=fps,
                settings=self.settings,
                frame_reader=self._reader,
                overlay=bool(opts.get("diagnostics_overlay", True)),
                ffmpeg_binary=str(opts.get("ffmpeg_binary", "ffmpeg")),
            )
        return payload

    def _build_result(
        self,
        *,
        wanted: list[int],
        cleaned: list[PoseFrame],
        outcomes: list[FrameOutcome],
        cleanup: CleanupReport,
        tracker: SubjectTracker,
        region: roi_module.Roi,
        frame_size: tuple[int, int] | None,
        detector_provider: ProviderResolution,
        pose_provider: ProviderResolution,
        duration_s: float,
    ) -> ExtractionResult:
        found = {pose.frame_index for pose in cleaned}
        confidences = [joint.confidence for pose in cleaned for joint in pose.body.values()]
        return ExtractionResult(
            adapter=self.name,
            version=ADAPTER_VERSION,
            frames_requested=len(wanted),
            frames_with_pose=len(cleaned),
            detector_provider=detector_provider,
            pose_provider=pose_provider,
            settings=self.settings.as_dict(),
            model_hashes=self.model_hashes(),
            tracker={
                **tracker.summary(),
                "frames_without_subject": sum(1 for o in outcomes if o.subject is None),
            },
            cleanup=cleanup.as_dict(),
            missing_frames=sorted(set(wanted) - found),
            confidence={
                "mean": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
                "min": round(min(confidences), 4) if confidences else 0.0,
                "max": round(max(confidences), 4) if confidences else 0.0,
            },
            roi=region.as_dict(),
            duration_s=duration_s,
            frame_size=frame_size,
        )

    def model_hashes(self) -> dict[str, str]:
        """Hash the weights actually used, so a manifest identifies them."""
        out: dict[str, str] = {}
        for role, path in (
            ("detector", self.settings.detector_model),
            ("pose", self.settings.pose_model),
        ):
            candidate = Path(str(path)).expanduser()
            if candidate.is_file():
                out[role] = sha256_file(candidate)
        return out

    def run(
        self,
        *,
        frames_dir: Path,
        output_dir: Path,
        frame_indices: list[int],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """The generic frame-directory entry point is not supported here.

        DWPose reads the clip directly, which is what keeps a motion reference
        from ever having its frames extracted to disk.
        """
        raise ValidationError(
            "The DWPose adapter reads a video, not an extracted frame directory. "
            "Call estimate_sequence(video_path=...).",
            adapter=self.name,
            frames_dir=str(frames_dir),
        )


def build_dwpose_adapter(
    config: Any,
    *,
    session_factory: OnnxSessionFactory | None = None,
    frame_reader: FrameReader | None = None,
    **overrides: Any,
) -> DWPoseOnnxAdapter:
    """Construct the adapter from an ``AppConfig`` plus CLI-level overrides."""
    settings = from_app_config(config, **overrides)
    return DWPoseOnnxAdapter(settings, session_factory=session_factory, frame_reader=frame_reader)


__all__ = [
    "ADAPTER_NAME",
    "ADAPTER_VERSION",
    "CPU_PROVIDER",
    "MODEL_HINT",
    "DWPoseOnnxAdapter",
    "ExtractionResult",
    "FrameOutcome",
    "build_dwpose_adapter",
]
