"""QC for motion compositions and candidate masters.

Motion QC answers: *is this motion-control sequence physically coherent, and is
the join invisible?* Master QC adds: *is the framing consistent, and does the
background hold still?*

Every threshold comes from ``config/app.yaml -> motion_qc``. None is hard-coded.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from app.core.ids import utc_now
from app.core.logging import get_logger, log_event
from app.domain.master import MasterCandidate, MasterCandidateStatus
from app.domain.motion import MotionComposition, TransitionType
from app.media.frames import frame_path, list_frame_indices, read_frame
from app.motion.pose_format import PoseFrame, load_pose_sequence, poses_equal
from app.motion.skeleton import HIGH_PRIORITY_JOINTS, LIMB_EDGES
from app.pipeline.context import ServiceContext
from app.qc.checks import CheckResult, CheckSeverity, QCReport, failed, passed, skipped
from app.qc.metrics import region_diff
from app.qc.report import render_text_report

logger = get_logger(__name__)


@dataclass
class MotionQCOptions:
    write_reports: bool = True
    check_master_frames: bool = True
    #: Cap the frames inspected pixel-by-pixel (all of them when None).
    max_sampled_frames: int | None = 120


def _longest_run(flags: list[bool]) -> int:
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def _series(poses: list[PoseFrame], fn: Any) -> list[float | None]:
    return [fn(pose) for pose in poses]


# ---------------------------------------------------------------------------
# motion composition QC
# ---------------------------------------------------------------------------
def run_motion_qc(
    context: ServiceContext,
    composition_id: str,
    *,
    version: int | None = None,
    options: MotionQCOptions | None = None,
) -> QCReport:
    """Check a composed motion-control sequence."""
    opts = options or MotionQCOptions()
    thresholds = context.config.motion_qc
    composition = context.repos.compositions.get(composition_id, version)
    poses = load_pose_sequence(
        context.absolute(composition.composed_pose_dir),
        range(composition.output_frame_count),
    )

    checks: list[CheckResult] = []
    metrics: dict[str, Any] = {"frame_count": len(poses)}

    checks.append(_check_frame_count(composition, poses, metrics))
    checks.append(_check_shoulder_scale_drift(poses, thresholds, metrics))
    checks.append(_check_torso_length_drift(poses, thresholds, metrics))
    checks.append(_check_head_drift(poses, thresholds, metrics))
    checks.append(_check_limb_length_continuity(poses, thresholds, metrics))
    checks.append(_check_velocity_continuity(poses, thresholds, metrics))
    checks.append(_check_missing_joint_runs(poses, thresholds, metrics))
    checks.append(_check_body_center_at_joins(composition, poses, thresholds, metrics))
    checks.append(_check_bridge_endpoints(context, composition, poses, thresholds, metrics))
    checks.append(_check_exposed_views(composition, metrics))
    checks.append(_check_preview(context, composition, metrics))
    checks.append(_check_no_source_pixels(context, composition, metrics))

    report = QCReport(
        job_id=composition.id,
        checks=checks,
        metrics=metrics,
        generated_at=utc_now().isoformat(),
    )
    if opts.write_reports:
        _persist_motion_report(context, composition, report)
    log_event(
        logger,
        "motion_qc_completed",
        composition_id=composition.id,
        passed=report.passed,
        failed=len(report.failed_checks),
    )
    return report


def _check_frame_count(
    composition: MotionComposition, poses: list[PoseFrame], metrics: dict[str, Any]
) -> CheckResult:
    """The declared count, the layout arithmetic and the files must agree."""
    expected = composition.expected_frame_count()
    indices = [pose.frame_index for pose in poses]
    contiguous = indices == list(range(len(indices)))
    evidence = {
        "declared": composition.output_frame_count,
        "layout_arithmetic": expected,
        "poses_on_disk": len(poses),
        "contiguous_from_zero": contiguous,
    }
    metrics["frame_count_check"] = evidence
    if composition.output_frame_count == expected == len(poses) and contiguous:
        return passed(
            "composition_frame_count",
            f"Frame count is consistent at {len(poses)} with no gaps.",
            metrics=evidence,
        )
    return failed(
        "composition_frame_count",
        "Frame count is inconsistent between the declaration, the layout and the files.",
        metrics=evidence,
    )


def _drift_check(
    check_id: str,
    values: list[float | None],
    limit: float,
    metrics: dict[str, Any],
    label: str,
) -> CheckResult:
    known = [v for v in values if v is not None and v > 0]
    if len(known) < 2:
        return skipped(check_id, f"not enough frames with a measurable {label}")
    reference = known[0]
    drift = max(abs(v - reference) / reference for v in known)
    evidence = {
        "reference": round(reference, 4),
        "min": round(min(known), 4),
        "max": round(max(known), 4),
        "max_drift": round(drift, 6),
    }
    metrics[check_id] = evidence
    if drift <= limit:
        return passed(
            check_id,
            f"{label} stays within {limit:.1%} (observed {drift:.2%}).",
            metrics=evidence,
            threshold={"max_drift": limit},
        )
    return failed(
        check_id,
        f"{label} drifts by {drift:.2%}, beyond the {limit:.1%} limit.",
        metrics=evidence,
        threshold={"max_drift": limit},
    )


def _check_shoulder_scale_drift(
    poses: list[PoseFrame], thresholds: Any, metrics: dict[str, Any]
) -> CheckResult:
    return _drift_check(
        "shoulder_scale_drift",
        _series(poses, lambda p: p.shoulder_width(0.0)),
        thresholds.max_shoulder_scale_drift,
        metrics,
        "shoulder scale",
    )


def _check_torso_length_drift(
    poses: list[PoseFrame], thresholds: Any, metrics: dict[str, Any]
) -> CheckResult:
    return _drift_check(
        "torso_length_drift",
        _series(poses, lambda p: p.torso_length(0.0)),
        thresholds.max_torso_length_drift,
        metrics,
        "torso length",
    )


def _check_head_drift(
    poses: list[PoseFrame], thresholds: Any, metrics: dict[str, Any]
) -> CheckResult:
    """Head size and position, measured from the nose-to-shoulder geometry."""
    scales: list[float | None] = []
    positions: list[tuple[float, float] | None] = []
    for pose in poses:
        nose = pose.joint("nose")
        shoulders = pose.midpoint("left_shoulder", "right_shoulder", 0.0)
        if nose is None or shoulders is None:
            scales.append(None)
            positions.append(None)
            continue
        scales.append(math.hypot(nose.x - shoulders[0], nose.y - shoulders[1]))
        positions.append((nose.x, nose.y))

    scale_result = _drift_check(
        "head_scale_drift", scales, thresholds.max_head_scale_drift, metrics, "head scale"
    )
    known = [p for p in positions if p is not None]
    if len(known) >= 2:
        jumps = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in pairwise(known)]
        worst = max(jumps)
        metrics["head_position_jump"] = {"max_jump_px": round(worst, 4)}
        if worst > thresholds.max_head_position_jump_px:
            return failed(
                "head_scale_drift",
                f"Head position jumps by {worst:.2f}px, beyond the "
                f"{thresholds.max_head_position_jump_px}px limit.",
                metrics={**metrics.get("head_scale_drift", {}), "max_jump_px": round(worst, 4)},
                threshold={"max_jump_px": thresholds.max_head_position_jump_px},
            )
    return scale_result


def _check_limb_length_continuity(
    poses: list[PoseFrame], thresholds: Any, metrics: dict[str, Any]
) -> CheckResult:
    """Bones must not stretch or snap between consecutive frames."""
    worst_name = ""
    worst_value = 0.0
    worst_frame = -1
    for a_name, b_name in LIMB_EDGES:
        lengths: list[tuple[int, float]] = []
        for pose in poses:
            first = pose.joint(a_name)
            second = pose.joint(b_name)
            if first and second:
                lengths.append((pose.frame_index, first.distance_to(second)))
        for (_, previous), (index, current) in pairwise(lengths):
            if previous <= 1e-6:
                continue
            change = abs(current - previous) / previous
            if change > worst_value:
                worst_value, worst_name, worst_frame = change, f"{a_name}->{b_name}", index

    evidence = {
        "worst_limb": worst_name,
        "worst_change": round(worst_value, 6),
        "worst_frame": worst_frame,
    }
    metrics["limb_length_continuity"] = evidence
    limit = thresholds.max_limb_length_discontinuity
    if worst_value <= limit:
        return passed(
            "limb_length_continuity",
            f"Limb lengths change smoothly (worst {worst_value:.2%}).",
            metrics=evidence,
            threshold={"max_change": limit},
        )
    return failed(
        "limb_length_continuity",
        f"Limb {worst_name!r} changes length by {worst_value:.2%} at frame "
        f"{worst_frame}, beyond the {limit:.1%} limit.",
        metrics=evidence,
        threshold={"max_change": limit},
        offending_frames=[worst_frame],
    )


def _check_velocity_continuity(
    poses: list[PoseFrame], thresholds: Any, metrics: dict[str, Any]
) -> CheckResult:
    centers = [p.torso_center(0.0) for p in poses]
    velocities: list[tuple[int, float]] = []
    for (index, a), b in zip(
        [(p.frame_index, c) for p, c in zip(poses, centers, strict=True)],
        centers[1:],
        strict=False,
    ):
        if a is None or b is None:
            continue
        velocities.append((index, math.hypot(b[0] - a[0], b[1] - a[1])))

    steps = [
        (index, abs(current - previous)) for (_, previous), (index, current) in pairwise(velocities)
    ]
    worst_frame, worst = max(steps, key=lambda item: item[1], default=(-1, 0.0))
    evidence = {"max_velocity_step_px": round(worst, 4), "worst_frame": worst_frame}
    metrics["velocity_continuity"] = evidence
    limit = thresholds.max_velocity_discontinuity_px
    if worst <= limit:
        return passed(
            "velocity_continuity",
            f"Root velocity is continuous (worst step {worst:.2f}px).",
            metrics=evidence,
            threshold={"max_step_px": limit},
        )
    return failed(
        "velocity_continuity",
        f"Root velocity jumps by {worst:.2f}px at frame {worst_frame}, beyond "
        f"the {limit}px limit.",
        severity=CheckSeverity.WARNING,
        metrics=evidence,
        threshold={"max_step_px": limit},
        offending_frames=[worst_frame],
    )


def _check_missing_joint_runs(
    poses: list[PoseFrame], thresholds: Any, metrics: dict[str, Any]
) -> CheckResult:
    runs = {
        name: _longest_run([pose.joint(name) is None for pose in poses])
        for name in HIGH_PRIORITY_JOINTS
    }
    worst_name, worst = max(runs.items(), key=lambda item: item[1], default=("", 0))
    metrics["missing_joint_runs"] = runs
    limit = thresholds.max_missing_joint_run
    if worst <= limit:
        return passed(
            "missing_joint_runs",
            f"No high-priority joint is missing for more than {worst} frame(s).",
            metrics=runs,
            threshold={"max_run": limit},
        )
    return failed(
        "missing_joint_runs",
        f"Joint {worst_name!r} is missing for {worst} consecutive frames (limit {limit}).",
        metrics=runs,
        threshold={"max_run": limit},
    )


def _check_body_center_at_joins(
    composition: MotionComposition,
    poses: list[PoseFrame],
    thresholds: Any,
    metrics: dict[str, Any],
) -> CheckResult:
    """The body must not teleport where a bridge meets a segment."""
    if not composition.joins:
        return skipped("body_center_jump_at_joins", "composition has a single segment")

    by_index = {pose.frame_index: pose for pose in poses}
    bridge_frames = sorted(
        index for index, pose in by_index.items() if pose.origin.value == "bridge"
    )
    if not bridge_frames:
        return skipped("body_center_jump_at_joins", "composition contains no bridge frames")

    # Boundaries are where a bridge starts and where it ends.
    boundaries: list[int] = []
    for index in bridge_frames:
        if index - 1 not in bridge_frames and index - 1 >= 0:
            boundaries.append(index)
        if index + 1 not in bridge_frames and index + 1 < len(poses):
            boundaries.append(index + 1)

    worst = 0.0
    worst_frame = -1
    for index in boundaries:
        before = by_index.get(index - 1)
        after = by_index.get(index)
        if before is None or after is None:
            continue
        a = before.torso_center(0.0)
        b = after.torso_center(0.0)
        if a is None or b is None:
            continue
        jump = math.hypot(b[0] - a[0], b[1] - a[1])
        if jump > worst:
            worst, worst_frame = jump, index

    evidence = {
        "max_jump_px": round(worst, 4),
        "worst_frame": worst_frame,
        "boundaries": sorted(set(boundaries)),
    }
    metrics["body_center_jump"] = evidence
    limit = thresholds.max_body_center_jump_px
    if worst <= limit:
        return passed(
            "body_center_jump_at_joins",
            f"Body centre moves at most {worst:.2f}px across a join boundary.",
            metrics=evidence,
            threshold={"max_jump_px": limit},
        )
    return failed(
        "body_center_jump_at_joins",
        f"Body centre jumps {worst:.2f}px at frame {worst_frame}, beyond the {limit}px limit.",
        metrics=evidence,
        threshold={"max_jump_px": limit},
        offending_frames=[worst_frame],
    )


def _check_bridge_endpoints(
    context: ServiceContext,
    composition: MotionComposition,
    poses: list[PoseFrame],
    thresholds: Any,
    metrics: dict[str, Any],
) -> CheckResult:
    """Each bridge's first and last pose must equal its anchors exactly."""
    bridging = [j for j in composition.joins if j.transition_type is TransitionType.POSE_BRIDGE]
    if not bridging:
        return skipped("bridge_endpoints_exact", "composition has no pose bridges")

    normalized_root = context.absolute(composition.normalized_pose_dir)
    by_index = {pose.frame_index: pose for pose in poses}
    bridge_frames = sorted(i for i, p in by_index.items() if p.origin.value == "bridge")

    groups: list[list[int]] = []
    for index in bridge_frames:
        if groups and index == groups[-1][-1] + 1:
            groups[-1].append(index)
        else:
            groups.append([index])

    problems: list[str] = []
    evidence: list[dict[str, Any]] = []
    tolerance = thresholds.bridge_endpoint_tolerance_px

    for position, join in enumerate(bridging):
        if position >= len(groups):
            problems.append(f"join {position}: no bridge frames were emitted")
            continue
        group = groups[position]
        if len(group) != join.bridge_frame_count:
            problems.append(
                f"join {position}: emitted {len(group)} bridge frames, declared "
                f"{join.bridge_frame_count}"
            )
            continue

        prev_source = composition.segments[join.prev_segment_index]
        next_source = composition.segments[join.next_segment_index]
        prev_anchor = _load_normalized_pose(
            normalized_root, prev_source.pose_dirname, join.prev_source_frame
        )
        next_anchor = _load_normalized_pose(
            normalized_root, next_source.pose_dirname, join.next_source_frame
        )
        start_ok = prev_anchor is not None and poses_equal(
            by_index[group[0]], prev_anchor, tolerance=tolerance
        )
        end_ok = next_anchor is not None and poses_equal(
            by_index[group[-1]], next_anchor, tolerance=tolerance
        )
        # The recorded output geometry must match the frames actually emitted:
        # promotion picks the garment seam from it, so a stale value would put
        # the reveal in the wrong place.
        recorded = join.output_bridge_range
        geometry_ok = recorded == (group[0], group[-1] + 1)
        evidence.append(
            {
                "join": position,
                "bridge_frames": [group[0], group[-1]],
                "anchor_prev_frame": join.prev_source_frame,
                "anchor_next_frame": join.next_source_frame,
                "recorded_output_bridge": list(recorded) if recorded else None,
                "recommended_transition_anchor": join.recommended_transition_anchor,
                "start_matches_anchor": start_ok,
                "end_matches_anchor": end_ok,
                "output_geometry_matches": geometry_ok,
            }
        )
        if not start_ok:
            problems.append(f"join {position}: bridge start does not equal the previous anchor")
        if not end_ok:
            problems.append(f"join {position}: bridge end does not equal the next anchor")
        if not geometry_ok:
            problems.append(
                f"join {position}: recorded output bridge {recorded} does not match "
                f"the emitted frames [{group[0]}, {group[-1] + 1})"
            )

    metrics["bridge_endpoints"] = evidence
    if not problems:
        return passed(
            "bridge_endpoints_exact",
            f"All {len(bridging)} bridge(s) start and end exactly on their anchors.",
            metrics={"joins": evidence},
            threshold={"tolerance_px": tolerance},
        )
    return failed(
        "bridge_endpoints_exact",
        "Bridge endpoints do not match their anchors: " + "; ".join(problems),
        metrics={"joins": evidence},
        threshold={"tolerance_px": tolerance},
    )


def _load_normalized_pose(root: Path, segment_dir: str, frame_index: int) -> PoseFrame | None:
    from app.motion.pose_format import load_pose_frame, pose_path

    path = pose_path(root / segment_dir, frame_index)
    return load_pose_frame(path) if path.is_file() else None


def _check_exposed_views(composition: MotionComposition, metrics: dict[str, Any]) -> CheckResult:
    """Flag back/side exposure the operator did not declare."""
    declared: dict[str, list[str]] = {}
    unexpected: list[str] = []
    for index, segment in enumerate(composition.segments):
        views = [v.value for v in segment.quality.exposed_views]
        declared[str(index)] = views
        for view in views:
            if view in {"back", "side"}:
                unexpected.append(f"segment {index} exposes {view}")
    metrics["exposed_views"] = declared
    if not unexpected:
        return passed(
            "exposed_views",
            "No segment declares back or side exposure; the prototype assumes frontal motion.",
            metrics=declared,
        )
    return failed(
        "exposed_views",
        "Segments expose non-frontal views: "
        + "; ".join(unexpected)
        + ". Garment references will need those views.",
        severity=CheckSeverity.WARNING,
        metrics=declared,
    )


def _check_preview(
    context: ServiceContext, composition: MotionComposition, metrics: dict[str, Any]
) -> CheckResult:
    if not composition.preview_path:
        return skipped("preview_frame_count_and_fps", "no preview was rendered")
    path = context.absolute(composition.preview_path)
    if not path.is_file():
        return failed(
            "preview_frame_count_and_fps", "preview file is missing", metrics={"path": str(path)}
        )

    from app.media import ffmpeg

    probe = ffmpeg.probe(path, ffprobe=context.config.runtime.ffprobe_binary, count_frames=True)
    stream = probe.video
    if stream is None:
        return failed("preview_frame_count_and_fps", "preview has no video stream")
    fps = ffmpeg.parse_frame_rate(stream.avg_frame_rate) or 0.0
    evidence = {
        "frame_count": stream.nb_frames,
        "expected_frame_count": composition.output_frame_count,
        "fps": round(fps, 4),
        "expected_fps": composition.output_fps,
    }
    metrics["preview"] = evidence
    tolerance = context.config.motion_qc.fps_tolerance
    if (
        stream.nb_frames == composition.output_frame_count
        and abs(fps - composition.output_fps) <= tolerance
    ):
        return passed(
            "preview_frame_count_and_fps",
            f"Preview has {stream.nb_frames} frames at {fps:.3f}fps, as composed.",
            metrics=evidence,
        )
    return failed(
        "preview_frame_count_and_fps",
        "Preview frame count or fps does not match the composition.",
        metrics=evidence,
    )


def _check_no_source_pixels(
    context: ServiceContext, composition: MotionComposition, metrics: dict[str, Any]
) -> CheckResult:
    """No image file may exist in the composition's pose directories.

    This is the structural guarantee that a motion reference contributes
    geometry and never imagery. It is checked rather than asserted.
    """
    image_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".mp4", ".mov"}
    offenders: list[str] = []
    for relative in (
        composition.normalized_pose_dir,
        composition.bridge_pose_dir,
        composition.composed_pose_dir,
    ):
        directory = context.absolute(relative)
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in image_suffixes:
                offenders.append(str(path.relative_to(directory)))

    evidence = {"offending_files": offenders[:16], "count": len(offenders)}
    metrics["no_source_pixels"] = evidence
    if not offenders:
        return passed(
            "no_source_pixels_in_motion_artifacts",
            "Motion artifacts contain pose data only; no imagery was copied.",
            metrics=evidence,
        )
    return failed(
        "no_source_pixels_in_motion_artifacts",
        f"{len(offenders)} image file(s) found in the pose directories; source "
        "pixels must never enter motion artifacts.",
        metrics=evidence,
    )


def _persist_motion_report(
    context: ServiceContext, composition: MotionComposition, report: QCReport
) -> None:
    root = context.absolute(composition.composed_pose_dir).parent
    (root / "qc_report.json").write_text(
        json.dumps(report.as_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    (root / "qc_report.txt").write_text(render_text_report(report), encoding="utf-8")


# ---------------------------------------------------------------------------
# master candidate QC
# ---------------------------------------------------------------------------
def run_master_qc(
    context: ServiceContext,
    candidate_id: str,
    *,
    options: MotionQCOptions | None = None,
) -> QCReport:
    """Check a candidate master's frames for framing and background consistency."""
    opts = options or MotionQCOptions()
    thresholds = context.config.motion_qc
    candidate = context.repos.masters.get(candidate_id)
    frames_dir = context.absolute(candidate.frames_dir)

    checks: list[CheckResult] = []
    metrics: dict[str, Any] = {}

    present = list_frame_indices(frames_dir)
    expected = list(range(candidate.frame_count))
    missing = sorted(set(expected) - set(present))
    evidence = {
        "expected": len(expected),
        "present": len(present),
        "missing_count": len(missing),
        "missing_sample": missing[:16],
    }
    metrics["frames"] = evidence
    checks.append(
        passed(
            "master_frame_sequence", f"All {len(expected)} frames are present.", metrics=evidence
        )
        if not missing
        else failed(
            "master_frame_sequence",
            f"{len(missing)} frame(s) are missing.",
            metrics=evidence,
            offending_frames=missing[:32],
        )
    )

    if opts.check_master_frames and present and not missing:
        sampled = _sample(expected, opts.max_sampled_frames)
        checks.append(_check_framing(frames_dir, sampled, candidate, metrics))
        checks.append(_check_background_consistency(frames_dir, sampled, thresholds, metrics))
        checks.append(_check_chunk_boundaries(context, candidate, frames_dir, metrics))
    else:
        for check_id in (
            "master_framing_consistency",
            "master_background_consistency",
            "master_chunk_boundaries",
        ):
            checks.append(skipped(check_id, "frames are incomplete"))

    checks.append(_check_master_manifest(context, candidate, metrics))
    checks.append(_check_acceptance_gate(candidate, metrics))

    report = QCReport(
        job_id=candidate.id,
        checks=checks,
        metrics=metrics,
        generated_at=utc_now().isoformat(),
    )

    if opts.write_reports:
        root = frames_dir.parent
        (root / "qc_report.json").write_text(
            json.dumps(report.as_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        (root / "qc_report.txt").write_text(render_text_report(report), encoding="utf-8")
        status = (
            MasterCandidateStatus.AWAITING_ACCEPTANCE
            if report.passed
            else MasterCandidateStatus.ANIMATED
        )
        context.repos.masters.save(
            candidate.model_copy(
                update={
                    "qc_report_path": context.relative(root / "qc_report.json"),
                    "qc_metrics": {**candidate.qc_metrics, "qc": report.summary()},
                    "status": status if not candidate.is_accepted else candidate.status,
                }
            )
        )
    log_event(
        logger,
        "master_qc_completed",
        candidate_id=candidate.id,
        passed=report.passed,
        failed=len(report.failed_checks),
    )
    return report


def _sample(indices: list[int], limit: int | None) -> list[int]:
    if limit is None or len(indices) <= limit:
        return list(indices)
    step = (len(indices) - 1) / (limit - 1)
    return sorted({indices[round(i * step)] for i in range(limit)})


def _check_framing(
    frames_dir: Path, sampled: list[int], candidate: MasterCandidate, metrics: dict[str, Any]
) -> CheckResult:
    """Every frame must match the canonical profile's output size exactly."""
    wrong: list[int] = []
    shape: tuple[int, int] | None = None
    for index in sampled:
        frame = read_frame(frame_path(frames_dir, index))
        current = frame.shape[:2]
        shape = shape or current
        if current != (candidate.height, candidate.width):
            wrong.append(index)
    evidence = {
        "expected": [candidate.height, candidate.width],
        "observed": list(shape) if shape else None,
        "wrong_count": len(wrong),
    }
    metrics["framing"] = evidence
    if not wrong:
        return passed(
            "master_framing_consistency",
            f"All sampled frames are {candidate.width}x{candidate.height}, as the "
            "canonical profile fixes them.",
            metrics=evidence,
        )
    return failed(
        "master_framing_consistency",
        f"{len(wrong)} frame(s) do not match the canonical output size.",
        metrics=evidence,
        offending_frames=wrong[:32],
    )


def _check_background_consistency(
    frames_dir: Path, sampled: list[int], thresholds: Any, metrics: dict[str, Any]
) -> CheckResult:
    """Corner regions should hold still; the character moves, the set does not."""
    if len(sampled) < 2:
        return skipped("master_background_consistency", "fewer than 2 frames sampled")

    reference = read_frame(frame_path(frames_dir, sampled[0]))
    height, width = reference.shape[:2]
    band = max(8, min(height, width) // 12)
    selection = np.zeros((height, width), dtype=bool)
    selection[:band, :] = True
    selection[-band:, :] = True
    selection[:, :band] = True
    selection[:, -band:] = True
    # The marker row the mock animator writes is metadata, not background.
    selection[0:1, :] = False

    worst = 0.0
    worst_frame = -1
    for index in sampled[1:]:
        frame = read_frame(frame_path(frames_dir, index))
        if frame.shape != reference.shape:
            continue
        stats = region_diff(reference, frame, selection)
        if stats.mean_diff > worst:
            worst, worst_frame = stats.mean_diff, index

    evidence = {"max_mean_abs_diff": round(worst, 6), "worst_frame": worst_frame}
    metrics["background_consistency"] = evidence
    limit = thresholds.max_background_mean_abs_diff
    if worst <= limit:
        return passed(
            "master_background_consistency",
            f"Background is stable (worst mean difference {worst:.3f}).",
            metrics=evidence,
            threshold={"max_mean_abs_diff": limit},
        )
    return failed(
        "master_background_consistency",
        f"Background drifts (mean difference {worst:.3f} at frame {worst_frame}).",
        severity=CheckSeverity.WARNING,
        metrics=evidence,
        threshold={"max_mean_abs_diff": limit},
        offending_frames=[worst_frame],
    )


def _check_chunk_boundaries(
    context: ServiceContext,
    candidate: MasterCandidate,
    frames_dir: Path,
    metrics: dict[str, Any],
) -> CheckResult:
    """Chunks must tile the range exactly: no gap, no double-generated frame."""
    chunks = sorted(candidate.chunks, key=lambda c: c.index)
    if not chunks:
        return skipped("master_chunk_boundaries", "no chunk records")

    problems: list[str] = []
    covered: list[int] = []
    for chunk in chunks:
        covered.extend(range(chunk.start_frame, chunk.end_frame))
    duplicates = sorted({f for f in covered if covered.count(f) > 1})
    if duplicates:
        problems.append(f"{len(duplicates)} frame(s) covered by more than one chunk")
    missing = sorted(set(range(candidate.frame_count)) - set(covered))
    if missing:
        problems.append(f"{len(missing)} frame(s) covered by no chunk")

    # A backend that declares ContextMode.NONE consumes nothing by design, so
    # its zero-context chunks are not a defect -- reporting them as one would
    # train operators to ignore the warning that matters. A backend that
    # declares it *does* condition and then gets no frames is the real fault.
    modes = sorted({chunk.context_mode for chunk in chunks})
    conditioning = [chunk for chunk in chunks if chunk.context_mode != "none"]
    unconditioned = [
        chunk.index for chunk in conditioning if chunk.index > 0 and chunk.overlap_frames == 0
    ]
    evidence = {
        "chunks": len(chunks),
        "duplicate_frames": duplicates[:16],
        "uncovered_frames": missing[:16],
        "context_modes": modes,
        "chunks_without_context": unconditioned,
        "context_frames_total": sum(chunk.overlap_frames for chunk in chunks),
    }
    metrics["chunk_boundaries"] = evidence

    if problems:
        return failed(
            "master_chunk_boundaries",
            "Chunk layout is wrong: " + "; ".join(problems),
            metrics=evidence,
        )
    if unconditioned:
        return failed(
            "master_chunk_boundaries",
            f"{len(unconditioned)} chunk(s) were generated without context frames, "
            "so their boundaries are unconditioned.",
            severity=CheckSeverity.WARNING,
            metrics=evidence,
        )
    if not conditioning:
        return passed(
            "master_chunk_boundaries",
            f"{len(chunks)} chunk(s) tile the sequence exactly. The animator "
            "declares context_mode=none, so boundaries are unconditioned by "
            "design; judge continuity from the frames, not from this check.",
            metrics=evidence,
        )
    return passed(
        "master_chunk_boundaries",
        f"{len(chunks)} chunk(s) tile the sequence exactly, each conditioned on its predecessor.",
        metrics=evidence,
    )


def _check_master_manifest(
    context: ServiceContext, candidate: MasterCandidate, metrics: dict[str, Any]
) -> CheckResult:
    manifest = context.repos.masters.get_manifest(candidate.id)
    if manifest is None:
        return failed("master_manifest_complete", "No manifest has been written.")
    required = ("input_hashes", "frame_hashes", "chunks", "settings", "reproducibility")
    missing = [key for key in required if not manifest.get(key)]
    evidence = {
        "digest": manifest.get("digest"),
        "missing_fields": missing,
        "frame_hash_count": len(manifest.get("frame_hashes", {})),
        "contains_source_pixels": manifest.get("contains_source_pixels"),
    }
    metrics["manifest"] = evidence
    if missing:
        return failed(
            "master_manifest_complete",
            "Manifest is incomplete: " + ", ".join(missing),
            metrics=evidence,
        )
    return passed(
        "master_manifest_complete",
        f"Manifest is complete (digest {str(manifest.get('digest'))[:12]}).",
        metrics=evidence,
    )


def _check_acceptance_gate(candidate: MasterCandidate, metrics: dict[str, Any]) -> CheckResult:
    """A candidate is not a master until a human says so — reported, not enforced."""
    evidence = {
        "status": candidate.status.value,
        "accepted": candidate.is_accepted,
        "accepted_by": candidate.acceptance.accepted_by if candidate.acceptance else None,
    }
    metrics["acceptance"] = evidence
    if candidate.is_accepted:
        return passed(
            "master_acceptance_recorded",
            f"Accepted by {candidate.acceptance.accepted_by if candidate.acceptance else '?'}.",
            metrics=evidence,
        )
    return passed(
        "master_acceptance_recorded",
        "Candidate is not yet accepted; it cannot be used as a master until it is.",
        metrics=evidence,
    )


__all__ = ["MotionQCOptions", "run_master_qc", "run_motion_qc"]
