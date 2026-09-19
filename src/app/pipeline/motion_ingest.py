"""Motion reference ingestion: register a clip, attach its pose data.

Two entry points, matching the two honest ways to get pose today:

``ingest_motion_source``
    Register the reference video: probe it, hash it, record the selected frame
    range and the rights paperwork. **No frames are extracted.** The pixels are
    never needed, so they are never copied — which is the cheapest possible way
    to guarantee they cannot leak into a master.
``import_pose``
    Attach externally computed pose JSON, validated frame by frame before
    anything is copied.

``extract_pose`` exists for when a real adapter is installed; it refuses clearly
today rather than fabricating data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.adapters.base import AdapterKind
from app.adapters.base import registry as adapter_registry
from app.core.errors import NotFoundError, ValidationError
from app.core.hashing import sha256_dir, sha256_file
from app.core.ids import new_id
from app.core.logging import get_logger, log_event
from app.core.paths import safe_identifier
from app.domain.human_template import FrameRange
from app.domain.motion import (
    BoundingBoxStats,
    MotionQualityMetrics,
    MotionSource,
    MotionSourceStatus,
    MotionUsageRights,
)
from app.media import ffmpeg
from app.motion.pose_format import (
    POSE_SCHEMA_VERSION,
    PoseFrame,
    list_pose_indices,
    load_pose_frame,
    load_pose_sequence,
    pose_path,
    save_pose_frame,
    summarize_bbox,
)
from app.motion.skeleton import HIGH_PRIORITY_JOINTS
from app.pipeline.context import ServiceContext
from app.pipeline.template_ingest import build_video_spec

logger = get_logger(__name__)

#: File types that must never appear in a motion source's pose directory. A
#: motion reference contributes geometry; imagery there would be a source-pixel
#: leak into a composition input.
NON_POSE_SUFFIXES: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".mp4", ".mov", ".mkv", ".webm"}
)

#: Used when no canonical profile can be loaded. Matches the shipped profile.
DEFAULT_CONFIDENCE_THRESHOLD = 0.35


def motion_source_id() -> str:
    return new_id("mot")


@dataclass
class MotionIngestOptions:
    display_name: str
    start_frame: int = 0
    end_frame: int | None = None
    motion_source_id: str | None = None
    version: int = 1
    # -- rights paperwork -------------------------------------------------
    source_description: str | None = None
    rights_holder: str | None = None
    license: str | None = None
    acquired_from: str | None = None
    motion_use_authorized: bool = False
    depicted_person_consent_ref: str | None = None
    notes: str | None = None
    allow_vfr: bool = False


@dataclass
class MotionIngestResult:
    source: MotionSource
    probe: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "motion_source_id": self.source.id,
            "version": self.source.version,
            "status": self.source.status.value,
            "source_sha256": self.source.source_sha256,
            "selected_range": [
                self.source.selected_range.start,
                self.source.selected_range.end,
            ],
            "frame_count": self.source.video.frame_count,
            "fps": self.source.video.fps,
            "pose_dir": self.source.pose_dir,
            "warnings": self.warnings,
        }


def ingest_motion_source(
    context: ServiceContext,
    source_video: str | Path,
    options: MotionIngestOptions,
) -> MotionIngestResult:
    """Register a motion reference. Probes and hashes; copies no imagery."""
    source = Path(source_video).expanduser().resolve()
    if not source.is_file():
        raise NotFoundError("Motion source video not found", path=str(source))

    identifier = safe_identifier(options.motion_source_id or motion_source_id())
    root = context.data_root.resolve("motion_sources", identifier)
    if root.exists() and any(root.iterdir()):
        raise ValidationError(
            "Motion source directory already exists and is not empty",
            motion_source_id=identifier,
            path=str(root),
        )
    pose_dir = root / "pose"
    pose_dir.mkdir(parents=True, exist_ok=True)

    probe = ffmpeg.probe(source, ffprobe=context.config.runtime.ffprobe_binary)
    spec, warnings = build_video_spec(probe, strict_cfr=not options.allow_vfr)

    end = options.end_frame if options.end_frame is not None else spec.frame_count
    if end > spec.frame_count:
        raise ValidationError(
            "Selected range extends past the end of the source video",
            end_frame=end,
            frame_count=spec.frame_count,
        )
    selected = FrameRange(start=options.start_frame, end=end)

    record = MotionSource(
        id=identifier,
        version=options.version,
        display_name=options.display_name,
        source_video_path=str(source),
        source_sha256=sha256_file(source),
        video=spec,
        selected_range=selected,
        pose_format=context.config.motion.pose_format,
        pose_schema_version=POSE_SCHEMA_VERSION,
        pose_dir=context.relative(pose_dir),
        usage_rights=MotionUsageRights(
            source_description=options.source_description,
            rights_holder=options.rights_holder,
            license=options.license,
            acquired_from=options.acquired_from,
            motion_use_authorized=options.motion_use_authorized,
            depicted_person_consent_ref=options.depicted_person_consent_ref,
        ),
        status=MotionSourceStatus.AWAITING_POSE,
        notes=options.notes,
    )
    if not options.motion_use_authorized:
        warnings.append(
            "motion_use_authorized is false: record the authorization before "
            "building a composition from this source."
        )

    saved = context.repos.motion_sources.save(record, allow_update=False)
    context.repos.audit.record(
        "motion_source_ingested",
        entity_type="motion_source",
        entity_id=saved.id,
        details={
            "version": saved.version,
            "source_sha256": saved.source_sha256,
            "range": [selected.start, selected.end],
            "motion_use_authorized": options.motion_use_authorized,
        },
    )
    log_event(
        logger,
        "motion_source_ingested",
        motion_source_id=saved.id,
        frames=selected.count,
        fps=spec.fps,
    )
    return MotionIngestResult(source=saved, probe=probe.raw, warnings=warnings)


@dataclass
class PoseImportResult:
    motion_source_id: str
    imported: list[int]
    skipped: list[str]
    quality: MotionQualityMetrics
    bbox_stats: BoundingBoxStats
    pose_sha256: str
    #: Present for extraction, empty for import: provider, model hashes, ROI,
    #: subject tracking and temporal cleanup.
    extraction: dict[str, Any] = field(default_factory=dict)
    diagnostics_dir: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "motion_source_id": self.motion_source_id,
            "imported_count": len(self.imported),
            "imported_range": ([min(self.imported), max(self.imported)] if self.imported else None),
            "skipped": self.skipped[:16],
            "pose_sha256": self.pose_sha256,
            "extraction": self.extraction,
            "diagnostics_dir": self.diagnostics_dir,
            "quality": self.quality.model_dump(mode="json"),
            "bbox_stats": self.bbox_stats.model_dump(mode="json"),
        }


def import_pose(
    context: ServiceContext,
    motion_source_id_: str,
    pose_source_dir: str | Path,
    *,
    version: int | None = None,
    overwrite: bool = False,
) -> PoseImportResult:
    """Import precomputed pose JSON for a motion source.

    Every file is parsed and validated before any of them is copied, so a bad
    batch cannot leave the source half-populated.
    """
    record = context.repos.motion_sources.get(motion_source_id_, version)
    source_dir = Path(pose_source_dir).expanduser().resolve()
    if not source_dir.is_dir():
        raise NotFoundError("Pose source directory not found", path=str(source_dir))

    destination = context.absolute(record.pose_dir)
    destination.mkdir(parents=True, exist_ok=True)

    selected = record.selected_range
    staged: list[tuple[int, PoseFrame]] = []
    skipped: list[str] = []

    for index in list_pose_indices(source_dir):
        if not selected.contains(index):
            skipped.append(f"frame {index}: outside the selected range")
            continue
        pose = load_pose_frame(pose_path(source_dir, index))
        if pose.frame_index != index:
            raise ValidationError(
                "Pose file's frame_index disagrees with its filename",
                filename_index=index,
                payload_index=pose.frame_index,
            )
        staged.append((index, pose))

    if not staged:
        raise ValidationError(
            "No importable pose frames after validation",
            source=str(source_dir),
            selected_range=[selected.start, selected.end],
            skipped=skipped[:16],
        )

    imported: list[int] = []
    for index, pose in staged:
        target = pose_path(destination, index)
        if target.exists() and not overwrite:
            skipped.append(f"frame {index}: already imported (use --overwrite)")
            continue
        save_pose_frame(target, pose)
        imported.append(index)

    poses = load_pose_sequence(destination)
    threshold = profile_confidence_threshold(context)
    quality = measure_pose_quality(poses, threshold)
    bbox = BoundingBoxStats.model_validate(summarize_bbox(poses))
    digest = sha256_dir(destination, patterns=("*.json",))

    updated = record.model_copy(
        update={
            "status": MotionSourceStatus.POSE_IMPORTED,
            "pose_origin": "imported",
            "pose_sha256": digest,
            "quality": quality,
            "bbox_stats": bbox,
        }
    )
    context.repos.motion_sources.save(updated)
    context.repos.audit.record(
        "motion_pose_imported",
        entity_type="motion_source",
        entity_id=record.id,
        details={"count": len(imported), "source": str(source_dir), "pose_sha256": digest},
    )
    log_event(logger, "motion_pose_imported", motion_source_id=record.id, count=len(imported))
    return PoseImportResult(
        motion_source_id=record.id,
        imported=sorted(imported),
        skipped=skipped,
        quality=quality,
        bbox_stats=bbox,
        pose_sha256=digest,
    )


def profile_confidence_threshold(context: ServiceContext) -> float:
    """The canonical profile's confidence threshold, with a safe fallback.

    Quality measurement must use the *same* threshold the normalizer will,
    otherwise a source is reported as high quality and then refused.
    """
    try:
        from app.pipeline.motion_compose import load_profile

        return load_profile(context).confidence_threshold
    except Exception:
        logger.warning(
            "profile_threshold_unavailable",
            extra={"event": "profile_threshold_unavailable"},
        )
        return DEFAULT_CONFIDENCE_THRESHOLD


def measure_pose_quality(
    poses: list[PoseFrame],
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> MotionQualityMetrics:
    """Summarise how usable a pose sequence is.

    ``confidence_threshold`` is applied consistently: a joint below it counts as
    *missing* for every purpose — body measurements, missing-joint runs, and the
    count of frames that carry a usable pose. Treating only ``confidence <= 0``
    as missing made a sequence of placed-but-uncertain joints look like
    high-quality motion, which normalization would then refuse.
    """
    if not poses:
        return MotionQualityMetrics(confidence_threshold=confidence_threshold)

    confidences = [joint.confidence for pose in poses for joint in pose.body.values()]

    # Measurements use confident joints only, so an uncertain frame cannot pull
    # the median body size around.
    shoulders = [w for w in (p.shoulder_width(confidence_threshold) for p in poses) if w]
    torsos = [t for t in (p.torso_length(confidence_threshold) for p in poses) if t]

    def median(values: list[float]) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / 2.0

    runs: dict[str, int] = {}
    for name in HIGH_PRIORITY_JOINTS:
        best = current = 0
        for pose in poses:
            missing = pose.confident_joint(name, confidence_threshold) is None
            current = current + 1 if missing else 0
            best = max(best, current)
        runs[name] = best

    # A frame carries a pose only if its torso is confidently resolvable --
    # that is what every downstream stage actually needs from it.
    usable_frames = sum(
        1 for pose in poses if pose.has_all(HIGH_PRIORITY_JOINTS, confidence_threshold)
    )

    shoulder_median = median(shoulders)
    variation = None
    if shoulders and shoulder_median:
        variation = (max(shoulders) - min(shoulders)) / shoulder_median

    return MotionQualityMetrics(
        frames_with_pose=usable_frames,
        mean_joint_confidence=(sum(confidences) / len(confidences)) if confidences else 0.0,
        min_joint_confidence=min(confidences) if confidences else 0.0,
        longest_missing_run=max(runs.values(), default=0),
        missing_joint_runs=runs,
        median_shoulder_width_px=shoulder_median,
        median_torso_length_px=median(torsos),
        shoulder_width_variation=variation,
        in_frame_fraction=usable_frames / len(poses),
        confidence_threshold=confidence_threshold,
    )


def extract_pose(
    context: ServiceContext,
    motion_source_id_: str,
    *,
    version: int | None = None,
    adapter: Any = None,
    diagnostics: bool = False,
    diagnostics_overlay: bool = True,
) -> PoseImportResult:
    """Extract pose with an adapter, keeping the clip's own frame numbering.

    The adapter may be injected; otherwise the one registered for
    :attr:`AdapterKind.POSE` is used. An unavailable adapter refuses with its
    own reason rather than inventing data.

    ``diagnostics`` writes review material — including an overlay video that
    contains source pixels — into the data root's ``diagnostics/`` tree, which
    is a sibling of every composition input and is never read by the pipeline."""
    record = context.repos.motion_sources.get(motion_source_id_, version)
    pose_adapter = adapter or adapter_registry.require(AdapterKind.POSE)
    capability = pose_adapter.capability()
    if not capability.available:
        from app.adapters.base import AdapterNotAvailableError

        raise AdapterNotAvailableError(
            f"Pose extraction is unavailable: {capability.reason} "
            "Compute poses externally and import them with `app motion import-pose`.",
            kind=capability.kind.value,
            status=capability.status.value,
            motion_source_id=record.id,
        )

    destination = context.absolute(record.pose_dir)
    indices = list(record.selected_range.indices())
    source_video = Path(record.source_video_path)
    if not source_video.is_file():
        raise NotFoundError(
            "Motion source video is no longer readable; cannot extract pose",
            motion_source_id=record.id,
            path=str(source_video),
        )

    # The adapter is given the video and the exact frame range. It is NOT given
    # a directory: a motion reference has no extracted frames by design, and
    # handing over the video's parent folder would expose unrelated files.
    estimate = getattr(pose_adapter, "estimate_sequence", None)
    if estimate is None:
        raise ValidationError(
            "Pose adapter does not implement estimate_sequence(video_path=…)",
            adapter=capability.name,
        )
    options: dict[str, Any] = {
        "selected_range": [record.selected_range.start, record.selected_range.end],
        "ffmpeg_binary": context.config.runtime.ffmpeg_binary,
    }
    diagnostics_target: Path | None = None
    if diagnostics:
        from app.adapters.dwpose.diagnostics import diagnostics_dir

        diagnostics_target = diagnostics_dir(
            context.data_root.path,
            safe_identifier(record.id),
            dirname=context.config.pose.diagnostics_dirname,
        )
        options["diagnostics_dir"] = str(diagnostics_target)
        options["diagnostics_overlay"] = diagnostics_overlay

    report = estimate(
        video_path=source_video,
        output_dir=destination,
        frame_indices=indices,
        fps=record.video.fps,
        options=options,
    )
    extraction = dict(report) if isinstance(report, dict) else {}
    # Structural guarantee, checked rather than asserted: the pose directory a
    # composition reads must contain geometry and nothing else. A diagnostics
    # overlay landing here would be a silent source-pixel leak.
    strays = sorted(
        path.name
        for path in destination.rglob("*")
        if path.is_file() and path.suffix.lower() in NON_POSE_SUFFIXES
    )
    if strays:
        raise ValidationError(
            "The pose adapter wrote non-pose files into the motion source's pose "
            "directory. A motion reference contributes geometry, never imagery.",
            motion_source_id=record.id,
            offending_files=strays[:16],
        )

    poses = load_pose_sequence(destination)
    written = sorted(p.frame_index for p in poses)
    if written != indices:
        raise ValidationError(
            "Pose adapter wrote frame indices that do not match the selected range; "
            "poses must keep the source clip's own numbering",
            expected_first=indices[0] if indices else None,
            expected_last=indices[-1] if indices else None,
            written_first=written[0] if written else None,
            written_last=written[-1] if written else None,
            motion_source_id=record.id,
        )
    quality = measure_pose_quality(poses, profile_confidence_threshold(context))
    bbox = BoundingBoxStats.model_validate(summarize_bbox(poses))
    digest = sha256_dir(destination, patterns=("*.json",))

    updated = record.model_copy(
        update={
            "status": MotionSourceStatus.POSE_EXTRACTED,
            "pose_origin": "extracted",
            "pose_adapter": capability.name,
            "pose_sha256": digest,
            "quality": quality,
            "bbox_stats": bbox,
            "pose_extraction": extraction,
        }
    )
    context.repos.motion_sources.save(updated)
    context.repos.audit.record(
        "motion_pose_extracted",
        entity_type="motion_source",
        entity_id=record.id,
        details={
            "adapter": capability.name,
            "count": len(poses),
            "synthetic": capability.notes.get("synthetic", False),
            "provider": (extraction.get("provider") or {}),
            "model_sha256": extraction.get("model_sha256", {}),
        },
    )
    return PoseImportResult(
        motion_source_id=record.id,
        imported=[p.frame_index for p in poses],
        skipped=[],
        quality=quality,
        bbox_stats=bbox,
        pose_sha256=digest,
        extraction=extraction,
        diagnostics_dir=str(diagnostics_target) if diagnostics_target else None,
    )


@dataclass
class MotionValidation:
    motion_source_id: str
    version: int
    ok: bool
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "motion_source_id": self.motion_source_id,
            "version": self.version,
            "ok": self.ok,
            "problems": self.problems,
            "warnings": self.warnings,
            "details": self.details,
        }


def validate_motion_source(
    context: ServiceContext,
    motion_source_id_: str,
    *,
    version: int | None = None,
    mark_ready: bool = True,
) -> MotionValidation:
    """Check a motion source has complete, usable pose data for its range."""
    record = context.repos.motion_sources.get(motion_source_id_, version)
    problems: list[str] = []
    warnings: list[str] = []

    pose_dir = context.absolute(record.pose_dir)
    present = set(list_pose_indices(pose_dir))
    expected = set(record.selected_range.indices())
    missing = sorted(expected - present)

    details: dict[str, Any] = {
        "expected_frames": len(expected),
        "present_frames": len(present & expected),
        "missing_count": len(missing),
        "missing_sample": missing[:16],
        "pose_dir": record.pose_dir,
    }

    if missing:
        problems.append(
            f"pose data missing for {len(missing)} frame(s) in the selected range "
            f"(first: {missing[0]})"
        )
    if not present:
        problems.append("no pose data has been imported or extracted yet")

    if record.pose_sha256:
        current = sha256_dir(pose_dir, patterns=("*.json",))
        details["pose_sha256_matches"] = current == record.pose_sha256
        if current != record.pose_sha256:
            warnings.append("pose data has changed since it was imported")

    limit = context.config.motion_qc.max_missing_joint_run
    for joint, run in record.quality.missing_joint_runs.items():
        if run > limit:
            problems.append(
                f"joint {joint!r} is missing for {run} consecutive frames " f"(limit {limit})"
            )
    details["quality"] = record.quality.model_dump(mode="json")

    if not record.usage_rights.motion_use_authorized:
        problems.append(
            "motion use is not marked authorized for this source; record the "
            "authorization before composing"
        )
    if not record.usage_rights.is_documented:
        warnings.append("usage rights are undocumented (rights holder, license or source)")

    ok = not problems
    if ok and mark_ready and record.status is not MotionSourceStatus.READY:
        context.repos.motion_sources.save(
            record.model_copy(update={"status": MotionSourceStatus.READY})
        )
    return MotionValidation(
        motion_source_id=record.id,
        version=record.version,
        ok=ok,
        problems=problems,
        warnings=warnings,
        details=details,
    )


__all__ = [
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "MotionIngestOptions",
    "MotionIngestResult",
    "MotionValidation",
    "PoseImportResult",
    "extract_pose",
    "import_pose",
    "ingest_motion_source",
    "measure_pose_quality",
    "profile_confidence_threshold",
    "validate_motion_source",
]
