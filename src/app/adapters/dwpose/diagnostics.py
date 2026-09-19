"""Diagnostic exports for a DWPose extraction.

Five artefacts, all written under ``<data_root>/diagnostics/motion/<id>/``:

``overlay.mp4``          skeleton + subject box drawn on the reference clip
``skeleton.mp4``         the same skeleton on a flat background, no source pixels
``subject_boxes.json``   every candidate box per frame, with its scored terms
``confidence_summary.json``  per-joint confidence distribution
``missing_joints.json``  runs of missing joints, and frames with no subject

**The overlay contains source pixels.** That is the whole point of it — an
operator has to see the skeleton sitting on the real dancer to trust the
extraction. It is therefore written outside every directory the composition
reads, and :func:`diagnostics_dir` is the only function that decides where. The
pose directory receives JSON and nothing else, which the motion QC check
verifies independently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from app.core.logging import get_logger
from app.motion.pose_format import PoseFrame
from app.motion.preview import PreviewSettings, draw_pose
from app.motion.skeleton import BODY_JOINTS

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.adapters.dwpose.adapter import ExtractionResult, FrameOutcome
    from app.adapters.dwpose.roi import Roi
    from app.adapters.dwpose.settings import DWPoseSettings
    from app.adapters.dwpose.video import FrameReader

logger = get_logger(__name__)

#: Subdirectory of the data root. Deliberately a sibling of, never inside,
#: ``motion_sources/`` or ``compositions/``.
DIAGNOSTICS_ROOT = "diagnostics"

OVERLAY_NAME = "overlay.mp4"
SKELETON_NAME = "skeleton.mp4"
BOXES_NAME = "subject_boxes.json"
CONFIDENCE_NAME = "confidence_summary.json"
MISSING_NAME = "missing_joints.json"
README_NAME = "README.txt"

README_TEXT = """\
DWPose extraction diagnostics -- REVIEW MATERIAL ONLY.

overlay.mp4 contains pixels from the reference clip. It exists so an operator
can confirm the skeleton tracks the right person. It is NOT a composition
input, must never be copied into motion_sources/<id>/pose/ or into a
composition directory, and nothing in the pipeline reads it.

Everything the pipeline consumes is pose JSON under the motion source's pose
directory. A motion reference contributes geometry and nothing else.
"""


def diagnostics_dir(
    data_root: Path, motion_source_id: str, *, dirname: str = DIAGNOSTICS_ROOT
) -> Path:
    """Where diagnostics for one motion source go. Outside every input tree."""
    return data_root / dirname / "motion" / motion_source_id


def _bbox_of(poses: list[PoseFrame]) -> tuple[float, float, float, float]:
    xs = [j.x for p in poses for j in p.body.values()]
    ys = [j.y for p in poses for j in p.body.values()]
    if not xs or not ys:
        return (0.0, 0.0, 1.0, 1.0)
    return (min(xs), min(ys), max(xs), max(ys))


def build_confidence_summary(poses: list[PoseFrame]) -> dict[str, Any]:
    """Per-joint confidence distribution across the whole sequence."""
    per_joint: dict[str, list[float]] = {name: [] for name in BODY_JOINTS}
    for pose in poses:
        for name in BODY_JOINTS:
            joint = pose.body.get(name)
            if joint is not None:
                per_joint[name].append(joint.confidence)

    summary: dict[str, Any] = {"frames": len(poses), "joints": {}}
    for name, values in per_joint.items():
        if not values:
            summary["joints"][name] = {"present": 0, "coverage": 0.0}
            continue
        ordered = sorted(values)
        summary["joints"][name] = {
            "present": len(values),
            "coverage": round(len(values) / max(len(poses), 1), 4),
            "mean": round(sum(values) / len(values), 4),
            "min": round(ordered[0], 4),
            "median": round(ordered[len(ordered) // 2], 4),
            "max": round(ordered[-1], 4),
        }
    all_values = [v for values in per_joint.values() for v in values]
    summary["overall"] = {
        "mean": round(sum(all_values) / len(all_values), 4) if all_values else 0.0,
        "min": round(min(all_values), 4) if all_values else 0.0,
    }
    return summary


def build_missing_report(
    poses: list[PoseFrame], outcomes: list[FrameOutcome], requested: list[int]
) -> dict[str, Any]:
    """Missing-joint runs plus the frames that produced no subject at all."""
    by_index = {pose.frame_index: pose for pose in poses}
    ordered = sorted(requested)

    runs: dict[str, list[list[int]]] = {}
    for name in BODY_JOINTS:
        current: list[int] = []
        for index in ordered:
            pose = by_index.get(index)
            if pose is None or name not in pose.body:
                current.append(index)
                continue
            if current:
                runs.setdefault(name, []).append([current[0], current[-1]])
                current = []
        if current:
            runs.setdefault(name, []).append([current[0], current[-1]])

    no_subject = [o.frame_index for o in outcomes if o.subject is None]
    return {
        "requested_frames": len(ordered),
        "frames_with_pose": len(poses),
        "frames_without_subject": no_subject[:64],
        "frames_without_subject_count": len(no_subject),
        "missing_runs": {
            name: {
                "runs": spans[:32],
                "longest": max((b - a + 1) for a, b in spans),
                "total_frames": sum((b - a + 1) for a, b in spans),
            }
            for name, spans in sorted(runs.items())
        },
    }


def build_boxes_report(outcomes: list[FrameOutcome]) -> dict[str, Any]:
    """Every candidate box per frame, with the terms that decided the winner."""
    return {
        "frames": [
            {
                "frame_index": outcome.frame_index,
                "selected": outcome.subject.as_dict() if outcome.subject else None,
                "reason": outcome.reason,
                "candidates": [candidate.as_dict() for candidate in outcome.candidates],
            }
            for outcome in outcomes
        ]
    }


def _draw_overlay_frame(
    frame: np.ndarray,
    pose: PoseFrame | None,
    outcome: FrameOutcome | None,
    settings: PreviewSettings,
) -> np.ndarray:
    import cv2

    canvas = frame.copy()
    if outcome is not None:
        for candidate in outcome.candidates:
            box = candidate.detection
            chosen = outcome.subject is not None and box.box == outcome.subject.box
            colour = (90, 220, 90) if chosen else (70, 70, 200)
            cv2.rectangle(
                canvas,
                (int(box.x1), int(box.y1)),
                (int(box.x2), int(box.y2)),
                colour,
                3 if chosen else 1,
            )
            label = f"{candidate.total:.2f}" + (
                f" {candidate.rejected}" if candidate.rejected else ""
            )
            cv2.putText(
                canvas,
                label,
                (int(box.x1), max(14, int(box.y1) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                colour,
                1,
                cv2.LINE_AA,
            )
    if pose is not None:
        draw_pose(
            canvas,
            pose,
            color=(90, 220, 240),
            joint_radius=settings.joint_radius,
            bone_thickness=settings.bone_thickness,
        )
    return canvas


def write_diagnostics(
    destination: Path,
    *,
    video_path: Path,
    outcomes: list[FrameOutcome],
    poses: list[PoseFrame],
    result: ExtractionResult,
    region: Roi,
    fps: float,
    settings: DWPoseSettings,
    frame_reader: FrameReader,
    overlay: bool = True,
    ffmpeg_binary: str = "ffmpeg",
) -> dict[str, Any]:
    """Write every diagnostic artefact. Returns what was produced."""
    destination.mkdir(parents=True, exist_ok=True)
    (destination / README_NAME).write_text(README_TEXT, encoding="utf-8")

    requested = [outcome.frame_index for outcome in outcomes]
    written: dict[str, Any] = {"directory": str(destination), "contains_source_pixels": overlay}

    (destination / BOXES_NAME).write_text(
        json.dumps(build_boxes_report(outcomes), indent=2, sort_keys=True), encoding="utf-8"
    )
    (destination / CONFIDENCE_NAME).write_text(
        json.dumps(build_confidence_summary(poses), indent=2, sort_keys=True), encoding="utf-8"
    )
    (destination / MISSING_NAME).write_text(
        json.dumps(build_missing_report(poses, outcomes, requested), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (destination / "extraction.json").write_text(
        json.dumps(result.as_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    written["reports"] = [BOXES_NAME, CONFIDENCE_NAME, MISSING_NAME, "extraction.json"]

    if not poses:
        written["videos"] = []
        return written

    width = result.frame_size[0] if result.frame_size else int(_bbox_of(poses)[2] + 40)
    height = result.frame_size[1] if result.frame_size else int(_bbox_of(poses)[3] + 40)
    preview = PreviewSettings(
        width=width, height=height, fps=fps, draw_labels=False, scale=settings.diagnostics_scale
    )

    videos: list[str] = []
    try:
        from app.motion.preview import render_skeleton_preview

        render_skeleton_preview(
            poses,
            destination / SKELETON_NAME,
            preview,
            ffmpeg_binary=ffmpeg_binary,
        )
        videos.append(SKELETON_NAME)
    except Exception as exc:  # pragma: no cover - ffmpeg absence is not fatal
        logger.warning(
            "dwpose_skeleton_preview_failed",
            extra={"event": "dwpose_skeleton_preview_failed", "error": str(exc)},
        )
        written["skeleton_error"] = str(exc)

    if overlay:
        try:
            videos.append(
                _render_overlay(
                    destination,
                    video_path=video_path,
                    outcomes=outcomes,
                    poses=poses,
                    region=region,
                    preview=preview,
                    frame_reader=frame_reader,
                    ffmpeg_binary=ffmpeg_binary,
                    crf=settings.diagnostics_crf,
                )
            )
        except Exception as exc:  # pragma: no cover - ffmpeg absence is not fatal
            logger.warning(
                "dwpose_overlay_failed",
                extra={"event": "dwpose_overlay_failed", "error": str(exc)},
            )
            written["overlay_error"] = str(exc)

    written["videos"] = videos
    return written


def _render_overlay(
    destination: Path,
    *,
    video_path: Path,
    outcomes: list[FrameOutcome],
    poses: list[PoseFrame],
    region: Roi,
    preview: PreviewSettings,
    frame_reader: FrameReader,
    ffmpeg_binary: str,
    crf: int,
) -> str:
    from app.media import ffmpeg
    from app.media.frames import DEFAULT_TEMPLATE, ffmpeg_pattern, frame_path, write_frame

    staging = destination / ".overlay_frames"
    staging.mkdir(parents=True, exist_ok=True)
    pose_by_index = {pose.frame_index: pose for pose in poses}
    outcome_by_index = {outcome.frame_index: outcome for outcome in outcomes}
    indices = sorted(outcome_by_index)

    first = indices[0]
    count = 0
    try:
        for frame_index, frame in frame_reader.read(video_path, indices):
            # Boxes and joints are in original video coordinates, so the overlay
            # is drawn on the whole frame -- the ROI is only an internal crop.
            canvas = _draw_overlay_frame(
                frame,
                pose_by_index.get(frame_index),
                outcome_by_index.get(frame_index),
                preview,
            )
            if region.mode != "none":
                import cv2

                cv2.rectangle(
                    canvas,
                    (region.x, region.y),
                    (region.x + region.width, region.y + region.height),
                    (200, 200, 60),
                    2,
                )
            write_frame(frame_path(staging, frame_index, DEFAULT_TEMPLATE), canvas)
            count += 1

        if count == 0:  # pragma: no cover - guarded by the caller
            raise ValueError("no frames to overlay")

        argv = ffmpeg.build_encode_from_frames_command(
            staging / ffmpeg_pattern(DEFAULT_TEMPLATE),
            destination / OVERLAY_NAME,
            ffmpeg=ffmpeg_binary,
            fps=preview.fps,
            width=int(preview.width * preview.scale) // 2 * 2,
            height=int(preview.height * preview.scale) // 2 * 2,
            crf=crf,
            preset="veryfast",
            start_number=first,
        )
        ffmpeg.run_command(argv)
    finally:
        for entry in staging.glob("*"):
            entry.unlink(missing_ok=True)
        staging.rmdir()
    return OVERLAY_NAME


__all__ = [
    "BOXES_NAME",
    "CONFIDENCE_NAME",
    "DIAGNOSTICS_ROOT",
    "MISSING_NAME",
    "OVERLAY_NAME",
    "README_NAME",
    "SKELETON_NAME",
    "build_boxes_report",
    "build_confidence_summary",
    "build_missing_report",
    "diagnostics_dir",
    "write_diagnostics",
]
