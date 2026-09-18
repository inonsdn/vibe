"""QC orchestration: run every check for a job and emit JSON + text reports.

Check inventory (each one is required by the product spec):

==========================================  ======================================
``dimensions_fps_frame_count``              output matches the configured contract
``frame_sequence_complete``                 no missing frames in the assembly
``duplicate_frames``                        no unintended byte-identical frames
``frozen_frames``                           motion never stalls
``protected_region_preserved``              protected pixels == source (lossless)
``background_preserved``                    background pixels == source
``face_region_preserved``                   face pixels == source
``garment_temporal_stability``              no flicker inside the garment region
``mask_boundary_leakage``                   no bleed just outside the mask
``black_frames``                            no blank output frames
``transition_correctness``                  the seam is exactly at the anchor
``audio_video_duration``                    audio and video durations agree
``encoding_valid``                          the encode is decodable and conformant
``manifest_deterministic``                  manifest is complete and reproducible
``intro_reuse_integrity``                   intro frames == cached source bytes
==========================================  ======================================
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.core.hashing import sha256_file
from app.core.ids import utc_now
from app.core.logging import get_logger, log_event
from app.domain.enums import JobStatus, MaskKind
from app.domain.manifest import QCSummary
from app.media import ffmpeg
from app.media.frames import frame_path, list_frame_indices, read_frame
from app.media.masks import as_mask, load_mask
from app.pipeline.context import ServiceContext
from app.qc.checks import CheckResult, CheckSeverity, QCReport, failed, passed, skipped
from app.qc.contact_sheet import build_contact_sheets
from app.qc.metrics import (
    bbox_to_mask,
    black_frame_fraction,
    boundary_band,
    duplicate_frame_indices,
    flicker_score,
    longest_run,
    region_diff,
    temporal_deltas,
)

logger = get_logger(__name__)


@dataclass
class _WorstDiff:
    """Running worst-case difference for one region across sampled frames.

    Tracked as a small object rather than a dict so the numbers keep their
    types all the way into the report (and so the comparisons below are
    checkable).
    """

    max_diff: int = 0
    mean_diff: float = 0.0
    frames: list[int] = field(default_factory=list)

    def observe(self, frame_index: int, max_diff: int, mean_diff: float) -> None:
        if max_diff > self.max_diff:
            self.max_diff = max_diff
            self.mean_diff = mean_diff
            self.frames = [frame_index]
        elif max_diff > 0:
            self.frames.append(frame_index)

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_diff": self.max_diff,
            "mean_diff": round(self.mean_diff, 6),
            "frames": self.frames[:32],
            "frame_count": len(self.frames),
        }


@dataclass
class _WorstFraction:
    """Running worst-case changed-pixel fraction for one region."""

    max_fraction: float = 0.0
    frames: list[int] = field(default_factory=list)

    def observe(self, frame_index: int, fraction: float) -> None:
        if fraction > self.max_fraction:
            self.max_fraction = fraction
            self.frames = [frame_index]
        elif fraction > 0:
            self.frames.append(frame_index)

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_fraction": round(self.max_fraction, 8),
            "frames": self.frames[:32],
            "frame_count": len(self.frames),
        }


@dataclass
class QCOptions:
    #: Cap the number of frames examined pixel-by-pixel (whole reveal if None).
    max_sampled_frames: int | None = None
    make_contact_sheets: bool = True
    check_final_video: bool = True
    write_reports: bool = True


def run_qc(
    context: ServiceContext,
    job_id: str,
    options: QCOptions | None = None,
) -> QCReport:
    """Run every QC check for a job and persist the reports."""
    opts = options or QCOptions()
    config = context.config
    job = context.repos.jobs.get(job_id)
    template = context.repos.templates.get(job.template_id, job.template_version)
    manifest = context.repos.jobs.get_manifest(job.id)

    source_dir = context.absolute(template.directories.source_frames)
    composited_dir = context.absolute(job.artifacts.composited_frames_dir)
    masks_dir = context.absolute(job.artifacts.effective_masks_dir)
    assembly_dir = context.absolute(job.artifacts.root) / "assembly_frames"
    intro_cache_dir = context.absolute(template.directories.intro_cache)
    protected_dir = context.absolute(template.directories.mask_dir(MaskKind.PROTECTED))
    garment_mask_dir = context.absolute(template.directories.mask_dir(MaskKind.GARMENT))

    checks: list[CheckResult] = []
    metrics: dict[str, Any] = {}

    reveal_indices = list(job.frame_range.indices())
    sampled = _sample_indices(reveal_indices, opts.max_sampled_frames)
    metrics["sampled_reveal_frames"] = len(sampled)

    # ---- per-frame pixel checks on the lossless intermediates -----------
    checks.extend(
        _pixel_checks(
            context,
            job,
            template,
            sampled,
            source_dir=source_dir,
            composited_dir=composited_dir,
            masks_dir=masks_dir,
            protected_dir=protected_dir,
            metrics=metrics,
        )
    )

    # ---- temporal checks -------------------------------------------------
    checks.append(
        _garment_temporal_stability(
            composited_dir,
            garment_mask_dir,
            sampled,
            config.qc.garment_flicker_max_mean_delta,
            metrics,
        )
    )
    checks.append(_frozen_frames(composited_dir, sampled, config, metrics))
    checks.append(_black_frames(composited_dir, sampled, config, metrics))

    # ---- sequence / assembly checks -------------------------------------
    checks.append(_frame_sequence_complete(assembly_dir, template, job, metrics))
    checks.append(_duplicate_frames(context, job, manifest, metrics))
    checks.append(_intro_reuse_integrity(assembly_dir, intro_cache_dir, template, metrics))
    checks.append(
        _transition_correctness(
            assembly_dir, intro_cache_dir, composited_dir, template, job, manifest, metrics
        )
    )

    # ---- final-video checks ---------------------------------------------
    if opts.check_final_video and job.artifacts.final_video_path:
        video_checks, video_metrics = _final_video_checks(context, job, template)
        checks.extend(video_checks)
        metrics["final_video"] = video_metrics
    else:
        reason = "no final video composed yet"
        checks.extend(
            [
                skipped("dimensions_fps_frame_count", reason),
                skipped("audio_video_duration", reason),
                skipped("encoding_valid", reason),
            ]
        )

    # ---- manifest --------------------------------------------------------
    checks.append(_manifest_deterministic(context, job, manifest, metrics))

    # ---- contact sheets --------------------------------------------------
    sheets: list[str] = []
    if opts.make_contact_sheets and assembly_dir.is_dir():
        produced = build_contact_sheets(
            assembly_dir,
            context.absolute(job.artifacts.root) / "contact_sheets",
            intro_start=template.intro.start,
            transition_anchor=template.transition_anchor_frame,
            reveal_end=job.frame_range.end,
            flash_frames=[
                c.index
                for c in (manifest.frame_checksums if manifest else [])
                if c.source == "flash"
            ],
            columns=config.qc.contact_sheet_columns,
            tile_width=config.qc.contact_sheet_tile_width,
            transition_span=config.qc.contact_sheet_transition_span,
            period_frames=config.qc.contact_sheet_period_frames,
        )
        sheets = [context.relative(path) for path in produced]

    report = QCReport(
        job_id=job.id,
        checks=checks,
        metrics=metrics,
        contact_sheets=sheets,
        generated_at=utc_now().isoformat(),
    )

    if opts.write_reports:
        _persist(context, job, report, manifest)

    log_event(
        logger,
        "qc_completed",
        job_id=job.id,
        passed=report.passed,
        checks=len(report.checks),
        failed=len(report.failed_checks),
    )
    return report


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------
def _sample_indices(indices: list[int], limit: int | None) -> list[int]:
    """Evenly sample frames, always keeping the first and last."""
    if limit is None or len(indices) <= limit:
        return list(indices)
    if limit <= 2:
        return [indices[0], indices[-1]][:limit]
    step = (len(indices) - 1) / (limit - 1)
    picked = sorted({indices[round(i * step)] for i in range(limit)})
    return picked


def _pixel_checks(
    context: ServiceContext,
    job: Any,
    template: Any,
    sampled: list[int],
    *,
    source_dir: Path,
    composited_dir: Path,
    masks_dir: Path,
    protected_dir: Path,
    metrics: dict[str, Any],
) -> list[CheckResult]:
    """Protected / background / face / boundary checks on lossless frames."""
    config = context.config
    if not sampled:
        reason = "no reveal frames to inspect"
        return [
            skipped("protected_region_preserved", reason),
            skipped("background_preserved", reason),
            skipped("face_region_preserved", reason),
            skipped("mask_boundary_leakage", reason),
        ]

    protected_worst = _WorstDiff()
    background_worst = _WorstDiff()
    face_worst = _WorstDiff()
    boundary_worst = _WorstFraction()
    face_available = False
    protected_available = False

    face_bboxes = _load_face_bboxes(context, template, sampled)

    for index in sampled:
        composited_path = frame_path(composited_dir, index)
        if not composited_path.is_file():
            continue
        source = read_frame(frame_path(source_dir, index))
        output = read_frame(composited_path)

        effective_path = frame_path(masks_dir, index)
        effective = (
            load_mask(effective_path, expect_shape=source.shape[:2])
            if effective_path.is_file()
            else np.zeros(source.shape[:2], dtype=np.uint8)
        )
        outside = effective == 0

        # "Background" here means every pixel the mask forbids editing that is
        # not already counted as protected, so the two checks do not overlap.
        protected_path = frame_path(protected_dir, index)
        if protected_path.is_file():
            protected_available = True
            protected = as_mask(load_mask(protected_path, expect_shape=source.shape[:2]))
            stats = region_diff(source, output, (protected > 0) & outside)
            protected_worst.observe(index, stats.max_diff, stats.mean_diff)
            background_selection = outside & (protected == 0)
        else:
            background_selection = outside

        bg = region_diff(source, output, background_selection)
        background_worst.observe(index, bg.max_diff, bg.mean_diff)

        bbox = face_bboxes.get(index)
        if bbox is not None:
            face_available = True
            face = region_diff(source, output, bbox_to_mask(source.shape[:2], bbox))
            face_worst.observe(index, face.max_diff, face.mean_diff)

        band = boundary_band(effective, width_px=3)
        if band.any():
            leak = region_diff(source, output, band)
            boundary_worst.observe(index, leak.changed_fraction)

    metrics["protected_region"] = protected_worst.as_dict()
    metrics["background_region"] = background_worst.as_dict()
    metrics["face_region"] = face_worst.as_dict()
    metrics["mask_boundary"] = boundary_worst.as_dict()

    limit = config.qc.lossless_protected_max_diff
    results: list[CheckResult] = []

    if not protected_available:
        results.append(
            failed(
                "protected_region_preserved",
                "No protected masks were found for the sampled frames; identity "
                "preservation cannot be verified numerically.",
                severity=CheckSeverity.ERROR,
                metrics=protected_worst.as_dict(),
            )
        )
    elif protected_worst.max_diff <= limit:
        results.append(
            passed(
                "protected_region_preserved",
                "Protected pixels are byte-identical to the source in every sampled frame.",
                metrics=protected_worst.as_dict(),
                threshold={"max_diff": limit},
            )
        )
    else:
        results.append(
            failed(
                "protected_region_preserved",
                f"Protected pixels changed (max diff {protected_worst.max_diff} > {limit}). "
                "The performer's identity was altered.",
                metrics=protected_worst.as_dict(),
                threshold={"max_diff": limit},
                offending_frames=protected_worst.frames,
            )
        )

    bg_limit = config.qc.lossless_background_max_diff
    if background_worst.max_diff <= bg_limit:
        results.append(
            passed(
                "background_preserved",
                "Background and all non-editable pixels are byte-identical to the source.",
                metrics=background_worst.as_dict(),
                threshold={"max_diff": bg_limit},
            )
        )
    else:
        results.append(
            failed(
                "background_preserved",
                f"Pixels outside the editable mask changed (max diff "
                f"{background_worst.max_diff} > {bg_limit}).",
                metrics=background_worst.as_dict(),
                threshold={"max_diff": bg_limit},
                offending_frames=background_worst.frames,
            )
        )

    if not face_available:
        results.append(
            skipped(
                "face_region_preserved",
                "no face landmark data available; the protected-mask check covers "
                "the face region instead",
            )
        )
    elif face_worst.mean_diff <= config.qc.face_region_mean_abs_diff:
        results.append(
            passed(
                "face_region_preserved",
                "Face region is unchanged.",
                metrics=face_worst.as_dict(),
                threshold={"mean_diff": config.qc.face_region_mean_abs_diff},
            )
        )
    else:
        results.append(
            failed(
                "face_region_preserved",
                f"Face region changed (mean diff {face_worst.mean_diff:.3f}).",
                metrics=face_worst.as_dict(),
                threshold={"mean_diff": config.qc.face_region_mean_abs_diff},
                offending_frames=face_worst.frames,
            )
        )

    leak_limit = config.qc.mask_boundary_leak_max_fraction
    if boundary_worst.max_fraction <= leak_limit:
        results.append(
            passed(
                "mask_boundary_leakage",
                "No pixel changes detected just outside the editable region.",
                metrics=boundary_worst.as_dict(),
                threshold={"max_fraction": leak_limit},
            )
        )
    else:
        results.append(
            failed(
                "mask_boundary_leakage",
                f"Changes leaked outside the mask boundary "
                f"({boundary_worst.max_fraction:.6f} > {leak_limit}).",
                metrics=boundary_worst.as_dict(),
                threshold={"max_fraction": leak_limit},
                offending_frames=boundary_worst.frames,
            )
        )
    return results


def _load_face_bboxes(
    context: ServiceContext, template: Any, indices: list[int]
) -> dict[int, tuple[int, int, int, int]]:
    """Read face bounding boxes if a landmark adapter has produced any."""
    directory = context.absolute(template.directories.face_landmarks)
    out: dict[int, tuple[int, int, int, int]] = {}
    if not directory.is_dir():
        return out
    for index in indices:
        path = directory / f"frame_{index:06d}.json"
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        bbox = payload.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            out[index] = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    return out


def _garment_temporal_stability(
    composited_dir: Path,
    garment_mask_dir: Path,
    sampled: list[int],
    threshold: float,
    metrics: dict[str, Any],
) -> CheckResult:
    contiguous = [i for i in sampled if frame_path(composited_dir, i).is_file()]
    if len(contiguous) < 3:
        return skipped("garment_temporal_stability", "fewer than 3 frames available")

    frames = [read_frame(frame_path(composited_dir, i)) for i in contiguous]
    first_mask_path = frame_path(garment_mask_dir, contiguous[0])
    selection = None
    if first_mask_path.is_file():
        mask = load_mask(first_mask_path, expect_shape=frames[0].shape[:2])
        selection = mask > 0

    deltas = temporal_deltas(frames, selection)
    score = flicker_score(deltas)
    metrics["garment_flicker"] = {**score, "frames_compared": len(deltas)}

    if score["mean_delta"] <= threshold:
        return passed(
            "garment_temporal_stability",
            f"Garment region changes smoothly (mean inter-frame delta "
            f"{score['mean_delta']:.3f}).",
            metrics=metrics["garment_flicker"],
            threshold={"mean_delta": threshold},
        )
    return failed(
        "garment_temporal_stability",
        f"Garment region flickers (mean inter-frame delta {score['mean_delta']:.3f} "
        f"> {threshold}).",
        severity=CheckSeverity.WARNING,
        metrics=metrics["garment_flicker"],
        threshold={"mean_delta": threshold},
    )


def _frozen_frames(
    composited_dir: Path, sampled: list[int], config: Any, metrics: dict[str, Any]
) -> CheckResult:
    contiguous = [i for i in sampled if frame_path(composited_dir, i).is_file()]
    if len(contiguous) < 3:
        return skipped("frozen_frames", "fewer than 3 frames available")
    frames = [read_frame(frame_path(composited_dir, i)) for i in contiguous]
    deltas = temporal_deltas(frames)
    run = longest_run(deltas, at_or_below=config.qc.frozen_frame_mean_abs_diff)
    metrics["frozen_frames"] = {
        "longest_static_run": run,
        "frames_compared": len(deltas),
        "min_delta": min(deltas) if deltas else 0.0,
    }
    limit = config.qc.frozen_frame_max_run
    if run <= limit:
        return passed(
            "frozen_frames",
            f"No frozen segment longer than {limit} frames (longest run: {run}).",
            metrics=metrics["frozen_frames"],
            threshold={"max_run": limit},
        )
    return failed(
        "frozen_frames",
        f"Video appears frozen for {run} consecutive frames (limit {limit}).",
        metrics=metrics["frozen_frames"],
        threshold={"max_run": limit},
    )


def _black_frames(
    composited_dir: Path, sampled: list[int], config: Any, metrics: dict[str, Any]
) -> CheckResult:
    offenders: list[int] = []
    worst = 0.0
    for index in sampled:
        path = frame_path(composited_dir, index)
        if not path.is_file():
            continue
        fraction = black_frame_fraction(
            read_frame(path), threshold=config.qc.black_frame_luma_threshold
        )
        worst = max(worst, fraction)
        if fraction >= config.qc.black_frame_max_fraction:
            offenders.append(index)
    metrics["black_frames"] = {"worst_dark_fraction": round(worst, 6), "count": len(offenders)}
    if not offenders:
        return passed(
            "black_frames",
            "No black output frames.",
            metrics=metrics["black_frames"],
            threshold={"max_dark_fraction": config.qc.black_frame_max_fraction},
        )
    return failed(
        "black_frames",
        f"{len(offenders)} output frames are effectively black.",
        metrics=metrics["black_frames"],
        offending_frames=offenders,
    )


def _frame_sequence_complete(
    assembly_dir: Path, template: Any, job: Any, metrics: dict[str, Any]
) -> CheckResult:
    if not assembly_dir.is_dir():
        return skipped("frame_sequence_complete", "job has not been composed yet")
    expected = list(range(template.intro.start, job.frame_range.end))
    present = list_frame_indices(assembly_dir)
    missing = sorted(set(expected) - set(present))
    extra = sorted(set(present) - set(expected))
    metrics["assembly_sequence"] = {
        "expected": len(expected),
        "present": len(present),
        "missing_count": len(missing),
        "unexpected_count": len(extra),
    }
    if not missing and not extra:
        return passed(
            "frame_sequence_complete",
            f"Assembled sequence is complete: {len(present)} frames, "
            f"{expected[0]}..{expected[-1]}.",
            metrics=metrics["assembly_sequence"],
        )
    return failed(
        "frame_sequence_complete",
        f"Assembled sequence is wrong: {len(missing)} missing, {len(extra)} unexpected.",
        metrics=metrics["assembly_sequence"],
        offending_frames=missing[:32] or extra[:32],
    )


def _duplicate_frames(
    context: ServiceContext, job: Any, manifest: Any, metrics: dict[str, Any]
) -> CheckResult:
    if manifest is None or not manifest.frame_checksums:
        return skipped("duplicate_frames", "no manifest frame checksums available")
    reveal = {
        c.index: c.sha256
        for c in manifest.frame_checksums
        if c.source == "rendered" and job.frame_range.contains(c.index)
    }
    groups = duplicate_frame_indices(reveal)
    metrics["duplicate_frames"] = {
        "group_count": len(groups),
        "groups": [g[:8] for g in groups[:8]],
    }
    if not groups:
        return passed(
            "duplicate_frames",
            "No byte-identical rendered frames.",
            metrics=metrics["duplicate_frames"],
        )
    return failed(
        "duplicate_frames",
        f"{len(groups)} groups of identical rendered frames found; the render may "
        "have stalled or repeated a frame.",
        severity=CheckSeverity.WARNING,
        metrics=metrics["duplicate_frames"],
        offending_frames=[index for group in groups for index in group][:32],
    )


def _intro_reuse_integrity(
    assembly_dir: Path, intro_cache_dir: Path, template: Any, metrics: dict[str, Any]
) -> CheckResult:
    if not assembly_dir.is_dir():
        return skipped("intro_reuse_integrity", "job has not been composed yet")
    mismatched: list[int] = []
    compared = 0
    for index in range(template.intro.start, template.intro.end):
        cached = frame_path(intro_cache_dir, index)
        assembled = frame_path(assembly_dir, index)
        if not (cached.is_file() and assembled.is_file()):
            mismatched.append(index)
            continue
        compared += 1
        if sha256_file(cached) != sha256_file(assembled):
            mismatched.append(index)
    metrics["intro_reuse"] = {"compared": compared, "mismatched": len(mismatched)}
    if not mismatched:
        return passed(
            "intro_reuse_integrity",
            f"All {compared} intro frames are byte-identical to the cached source frames.",
            metrics=metrics["intro_reuse"],
        )
    return failed(
        "intro_reuse_integrity",
        f"{len(mismatched)} intro frames differ from the cache; the intro was not reused.",
        metrics=metrics["intro_reuse"],
        offending_frames=mismatched,
    )


def _transition_correctness(
    assembly_dir: Path,
    intro_cache_dir: Path,
    composited_dir: Path,
    template: Any,
    job: Any,
    manifest: Any,
    metrics: dict[str, Any],
) -> CheckResult:
    if not assembly_dir.is_dir():
        return skipped("transition_correctness", "job has not been composed yet")
    anchor = template.transition_anchor_frame
    flash = {c.index for c in (manifest.frame_checksums if manifest else []) if c.source == "flash"}

    evidence: dict[str, Any] = {
        "anchor": anchor,
        "intro_end": template.intro.end,
        "reveal_start": template.reveal.start,
        "flash_frames": sorted(flash),
    }
    problems: list[str] = []

    if template.intro.end != anchor or template.reveal.start != anchor:
        problems.append("template ranges do not meet exactly at the anchor")

    last_intro = anchor - 1
    if last_intro >= template.intro.start:
        cached = frame_path(intro_cache_dir, last_intro)
        assembled = frame_path(assembly_dir, last_intro)
        matches = (
            cached.is_file()
            and assembled.is_file()
            and sha256_file(cached) == sha256_file(assembled)
        )
        evidence["last_intro_frame"] = last_intro
        evidence["last_intro_matches_cache"] = matches
        if not matches:
            problems.append(f"frame {last_intro} (last intro) is not the cached source frame")

    rendered = frame_path(composited_dir, anchor)
    assembled_anchor = frame_path(assembly_dir, anchor)
    if anchor in flash:
        evidence["anchor_is_flash"] = True
    else:
        matches = (
            rendered.is_file()
            and assembled_anchor.is_file()
            and sha256_file(rendered) == sha256_file(assembled_anchor)
        )
        evidence["anchor_matches_render"] = matches
        if not matches:
            problems.append(f"frame {anchor} (first reveal) is not the rendered frame")

    metrics["transition"] = evidence
    if not problems:
        return passed(
            "transition_correctness",
            f"Transition is exact at frame {anchor}: intro ends at {anchor - 1}, "
            f"reveal starts at {anchor}.",
            metrics=evidence,
        )
    return failed(
        "transition_correctness",
        "Transition is incorrect: " + "; ".join(problems),
        metrics=evidence,
    )


def _final_video_checks(
    context: ServiceContext, job: Any, template: Any
) -> tuple[list[CheckResult], dict[str, Any]]:
    config = context.config
    path = context.absolute(job.artifacts.final_video_path)
    if not path.is_file():
        reason = "final video file is missing"
        return (
            [
                failed("encoding_valid", reason),
                skipped("dimensions_fps_frame_count", reason),
                skipped("audio_video_duration", reason),
            ],
            {},
        )

    probe = ffmpeg.probe(path, ffprobe=config.runtime.ffprobe_binary, count_frames=True)
    stream = probe.video
    if stream is None:
        return (
            [
                failed("encoding_valid", "output has no video stream"),
                skipped("dimensions_fps_frame_count", "no video stream"),
                skipped("audio_video_duration", "no video stream"),
            ],
            {},
        )

    fps = ffmpeg.parse_frame_rate(stream.avg_frame_rate) or 0.0
    expected_frames = (template.intro.count) + job.frame_range.count
    video_metrics = {
        "width": stream.width,
        "height": stream.height,
        "fps": round(fps, 6),
        "frame_count": stream.nb_frames,
        "expected_frame_count": expected_frames,
        "pixel_format": stream.pix_fmt,
        "codec": stream.codec_name,
        "duration_s": stream.duration_s or probe.duration_s,
        "has_audio": probe.has_audio,
        "sha256": sha256_file(path),
    }

    problems: list[str] = []
    if stream.width != config.video.width or stream.height != config.video.height:
        problems.append(
            f"dimensions {stream.width}x{stream.height} != configured "
            f"{config.video.width}x{config.video.height}"
        )
    if abs(fps - template.video.fps) > config.qc.fps_tolerance:
        problems.append(f"fps {fps:.4f} != source {template.video.fps:.4f}")
    if stream.nb_frames is not None and stream.nb_frames != expected_frames:
        problems.append(f"frame count {stream.nb_frames} != expected {expected_frames}")

    checks: list[CheckResult] = []
    if not problems:
        checks.append(
            passed(
                "dimensions_fps_frame_count",
                f"Output is {stream.width}x{stream.height} @ {fps:.3f}fps with "
                f"{stream.nb_frames} frames, as configured.",
                metrics=video_metrics,
            )
        )
    else:
        checks.append(
            failed(
                "dimensions_fps_frame_count",
                "Output metadata does not match the contract: " + "; ".join(problems),
                metrics=video_metrics,
            )
        )

    encoding_problems: list[str] = []
    if stream.pix_fmt != config.video.pixel_format:
        encoding_problems.append(f"pixel format {stream.pix_fmt} != {config.video.pixel_format}")
    if stream.codec_name not in {"h264", "libx264"}:
        encoding_problems.append(f"unexpected codec {stream.codec_name}")
    if not encoding_problems:
        checks.append(
            passed(
                "encoding_valid",
                f"Encode is valid: {stream.codec_name} / {stream.pix_fmt}.",
                metrics={"codec": stream.codec_name, "pixel_format": stream.pix_fmt},
            )
        )
    else:
        checks.append(
            failed(
                "encoding_valid",
                "Encoding is not conformant: " + "; ".join(encoding_problems),
                metrics=video_metrics,
            )
        )

    audio = probe.audio
    if audio is None:
        checks.append(
            skipped("audio_video_duration", "output has no audio stream")
            if not template.video.has_audio
            else failed(
                "audio_video_duration",
                "Source has audio but the output does not.",
                severity=CheckSeverity.WARNING,
                metrics=video_metrics,
            )
        )
    else:
        video_duration = stream.duration_s or probe.duration_s or 0.0
        audio_duration = audio.duration_s or 0.0
        delta = abs(video_duration - audio_duration)
        audio_metrics = {
            "video_duration_s": round(video_duration, 6),
            "audio_duration_s": round(audio_duration, 6),
            "delta_s": round(delta, 6),
        }
        video_metrics["audio"] = audio_metrics
        if delta <= config.qc.duration_tolerance_s:
            checks.append(
                passed(
                    "audio_video_duration",
                    f"Audio and video durations agree within {delta:.4f}s.",
                    metrics=audio_metrics,
                    threshold={"tolerance_s": config.qc.duration_tolerance_s},
                )
            )
        else:
            checks.append(
                failed(
                    "audio_video_duration",
                    f"Audio/video duration mismatch of {delta:.4f}s "
                    f"(tolerance {config.qc.duration_tolerance_s}s).",
                    severity=CheckSeverity.WARNING,
                    metrics=audio_metrics,
                    threshold={"tolerance_s": config.qc.duration_tolerance_s},
                )
            )
    return checks, video_metrics


def _manifest_deterministic(
    context: ServiceContext, job: Any, manifest: Any, metrics: dict[str, Any]
) -> CheckResult:
    if manifest is None:
        return failed("manifest_deterministic", "No manifest has been written for this job.")
    missing = manifest.required_fields_present()
    digest = manifest.reproducibility_digest()
    stored = context.repos.jobs.manifest_digest(job.id)
    metrics["manifest"] = {
        "digest": digest,
        "stored_digest": stored,
        "missing_fields": missing,
        "frame_checksum_count": len(manifest.frame_checksums),
        "input_hash_count": len(manifest.input_hashes),
    }
    problems: list[str] = []
    if missing:
        problems.append("incomplete fields: " + ", ".join(missing))
    if stored is not None and stored != digest:
        problems.append(f"stored digest {stored[:12]} != recomputed {digest[:12]}")
    if not problems:
        return passed(
            "manifest_deterministic",
            f"Manifest is complete and its reproducibility digest is stable " f"({digest[:12]}).",
            metrics=metrics["manifest"],
        )
    return failed(
        "manifest_deterministic",
        "Manifest is not reproducible: " + "; ".join(problems),
        metrics=metrics["manifest"],
    )


# ---------------------------------------------------------------------------
# persistence & text rendering
# ---------------------------------------------------------------------------
def _persist(context: ServiceContext, job: Any, report: QCReport, manifest: Any) -> None:
    json_path = context.absolute(
        job.artifacts.qc_report_path or f"{job.artifacts.root}/qc_report.json"
    )
    text_path = context.absolute(
        job.artifacts.qc_report_text_path or f"{job.artifacts.root}/qc_report.txt"
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    text_path.write_text(render_text_report(report), encoding="utf-8")

    artifacts = job.artifacts.model_copy(
        update={
            "qc_report_path": context.relative(json_path),
            "qc_report_text_path": context.relative(text_path),
            "contact_sheet_paths": report.contact_sheets,
        }
    )
    status = JobStatus.COMPLETED if report.passed else JobStatus.FAILED
    context.repos.jobs.save(
        job.model_copy(
            update={
                "artifacts": artifacts,
                "qc_metrics": {**job.qc_metrics, "qc": report.summary()},
                "status": status if job.status is JobStatus.COMPOSED else job.status,
            }
        )
    )

    if manifest is not None:
        summary = QCSummary(
            passed=report.passed,
            checks_total=len(report.checks),
            checks_failed=len(report.failed_checks),
            failed_check_ids=[c.check_id for c in report.failed_checks],
            metrics=report.metrics,
        )
        updated = manifest.model_copy(update={"qc": summary})
        context.repos.jobs.save_manifest(updated)
        manifest_path = context.absolute(
            job.artifacts.manifest_path or f"{job.artifacts.root}/manifest.json"
        )
        manifest_path.write_text(
            json.dumps(updated.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )


def render_text_report(report: QCReport) -> str:
    """Human-readable QC report."""
    lines: list[str] = []
    verdict = "PASS" if report.passed else "FAIL"
    lines.append("=" * 72)
    lines.append(f"QC REPORT  job={report.job_id}  verdict={verdict}")
    lines.append(f"generated={report.generated_at}")
    lines.append("=" * 72)
    lines.append("")

    width = max((len(c.check_id) for c in report.checks), default=10)
    for check in report.checks:
        if check.skipped:
            status = "SKIP"
        elif check.passed:
            status = "ok"
        elif check.severity is CheckSeverity.WARNING:
            status = "WARN"
        else:
            status = "FAIL"
        lines.append(f"[{status:>4}] {check.check_id.ljust(width)}  {check.message}")
        if not check.passed and check.metrics:
            for key, value in sorted(check.metrics.items()):
                lines.append(f"         {key} = {value}")
            if check.threshold:
                lines.append(f"         threshold = {check.threshold}")
            if check.offending_frames:
                shown = ", ".join(str(i) for i in check.offending_frames[:16])
                lines.append(f"         frames = {shown}")

    lines.append("")
    lines.append("-" * 72)
    lines.append(
        f"{len(report.checks)} checks, {len(report.failed_checks)} failed, "
        f"{len(report.warnings)} warnings"
    )
    if report.contact_sheets:
        lines.append("contact sheets:")
        lines.extend(f"  {sheet}" for sheet in report.contact_sheets)
    lines.append("-" * 72)
    return "\n".join(lines) + "\n"


__all__ = ["QCOptions", "render_text_report", "run_qc"]
