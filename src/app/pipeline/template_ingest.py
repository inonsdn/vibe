"""Master Human Performance ingestion.

Steps, in order:

1. probe the video with ffprobe (nothing is assumed about it),
2. require a constant frame rate, or convert to one *explicitly* into a new file
   — the original is never modified,
3. extract lossless PNG frames named by absolute frame index,
4. hash the source video and every extracted frame,
5. record the operator's intro / reveal / transition-anchor configuration,
6. create the mask and analysis-data directories,
7. optionally import externally authored masks,
8. validate that every reveal frame has correctly sized masks.

After step 3, ``source_frames/`` is treated as immutable: it is hashed into the
template record and :func:`verify_source_immutability` re-checks that hash
before every render.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.adapters.base import AdapterKind
from app.adapters.base import registry as adapter_registry
from app.core.errors import ImmutabilityError, MediaToolError, NotFoundError, ValidationError
from app.core.hashing import sha256_dir, sha256_file
from app.core.ids import template_id as new_template_id
from app.core.logging import get_logger, log_event
from app.core.paths import safe_identifier
from app.domain.enums import MASK_DIR_NAMES, MaskKind, ProcessingStatus
from app.domain.human_template import (
    ConsentRecord,
    FrameRange,
    HumanTemplate,
    TemplateDirectories,
    VideoSpec,
)
from app.media import ffmpeg
from app.media.frames import (
    DEFAULT_TEMPLATE,
    ffmpeg_pattern,
    frame_path,
    list_frame_indices,
    validate_sequence,
)
from app.media.masks import load_mask
from app.pipeline.context import ServiceContext
from app.version import PREPROCESSING_VERSION

logger = get_logger(__name__)


def _adapter_status(kind: AdapterKind) -> str:
    """Report an adapter's status, or that no adapter is registered for it."""
    adapter = adapter_registry.get(kind)
    return adapter.capability().status.value if adapter is not None else "unregistered"


@dataclass
class IngestOptions:
    """Operator-supplied ingestion parameters."""

    display_name: str
    intro_start: int = 0
    transition_anchor: int | None = None
    reveal_end: int | None = None
    template_clothing_class: str = "fitted_short"
    subject_kind: str = "synthetic"
    adult_confirmed: bool = True
    consent_document_ref: str | None = None
    rights_holder: str | None = None
    license: str | None = None
    provenance_notes: str | None = None
    identity_reference_images: list[Path] = field(default_factory=list)
    background_plate: Path | None = None
    template_id: str | None = None
    version: int = 1
    #: Convert a variable-frame-rate source to CFR instead of refusing it.
    allow_vfr_conversion: bool = False
    #: Skip frame extraction (used when re-registering an existing directory).
    extract_frames: bool = True
    notes: str | None = None


@dataclass
class IngestResult:
    template: HumanTemplate
    probe: dict[str, Any]
    commands: list[list[str]]
    extracted_frames: int
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template.id,
            "version": self.template.version,
            "status": self.template.status.value,
            "extracted_frames": self.extracted_frames,
            "intro": [self.template.intro.start, self.template.intro.end],
            "reveal": [self.template.reveal.start, self.template.reveal.end],
            "transition_anchor_frame": self.template.transition_anchor_frame,
            "source_sha256": self.template.source_sha256,
            "source_frames_sha256": self.template.source_frames_sha256,
            "commands": self.commands,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# probing
# ---------------------------------------------------------------------------
def build_video_spec(
    probe: ffmpeg.ProbeResult,
    *,
    strict_cfr: bool = True,
) -> tuple[VideoSpec, list[str]]:
    """Turn an ffprobe result into a validated :class:`VideoSpec`."""
    stream = probe.video
    if stream is None:
        raise ValidationError("Input file has no video stream")
    if not stream.width or not stream.height:
        raise ValidationError("Video stream has no dimensions")

    avg_fps = ffmpeg.parse_frame_rate(stream.avg_frame_rate)
    r_fps = ffmpeg.parse_frame_rate(stream.r_frame_rate)
    fps = avg_fps or r_fps
    if fps is None or fps <= 0:
        raise ValidationError(
            "Could not determine the video frame rate",
            avg_frame_rate=stream.avg_frame_rate,
            r_frame_rate=stream.r_frame_rate,
        )

    warnings: list[str] = []
    constant = True
    if avg_fps and r_fps and abs(avg_fps - r_fps) > 0.01:
        constant = False
        message = (
            f"Variable frame rate detected (avg={avg_fps:.4f}, r={r_fps:.4f}). "
            "Frame indices would not map to stable timestamps."
        )
        if strict_cfr:
            raise ValidationError(
                message,
                avg_frame_rate=stream.avg_frame_rate,
                r_frame_rate=stream.r_frame_rate,
                hint="Re-ingest with --allow-vfr-conversion to convert to CFR first.",
            )
        warnings.append(message)

    duration = stream.duration_s or probe.duration_s
    frame_count = stream.nb_frames
    if frame_count is None and duration is not None:
        frame_count = round(duration * fps)
        warnings.append(
            "Frame count was not present in the container; it was derived from "
            "duration x fps and will be replaced by the real extracted count."
        )
    if not frame_count or frame_count <= 0:
        raise ValidationError("Could not determine the video frame count")
    if duration is None:
        duration = frame_count / fps

    audio = probe.audio
    return (
        VideoSpec(
            width=stream.width,
            height=stream.height,
            fps=fps,
            duration_s=duration,
            frame_count=frame_count,
            codec=stream.codec_name,
            pixel_format=stream.pix_fmt,
            constant_frame_rate=constant,
            avg_frame_rate=stream.avg_frame_rate,
            r_frame_rate=stream.r_frame_rate,
            has_audio=probe.has_audio,
            audio_codec=audio.codec_name if audio else None,
            audio_duration_s=audio.duration_s if audio else None,
        ),
        warnings,
    )


def convert_to_cfr(
    source: Path,
    destination: Path,
    *,
    fps: float,
    ffmpeg_binary: str = "ffmpeg",
) -> list[str]:
    """Write a constant-frame-rate copy. The original is never touched."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(ffmpeg.assert_local_path(source, what="input")),
        "-fps_mode",
        "cfr",
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-crf",
        "14",
        "-preset",
        "slow",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        str(ffmpeg.assert_local_path(destination, what="output")),
    ]
    ffmpeg.run_command(argv)
    return argv


# ---------------------------------------------------------------------------
# ingestion
# ---------------------------------------------------------------------------
def ingest_template(
    context: ServiceContext,
    source_video: str | Path,
    options: IngestOptions,
) -> IngestResult:
    """Ingest a master performance video into an immutable template."""
    config = context.config
    source = Path(source_video).expanduser().resolve()
    if not source.is_file():
        raise NotFoundError("Source video not found", path=str(source))

    template_identifier = safe_identifier(options.template_id or new_template_id())
    template_dir = context.data_root.template_dir(template_identifier)
    if template_dir.exists() and any(template_dir.iterdir()) and options.extract_frames:
        raise ValidationError(
            "Template directory already exists and is not empty",
            template_id=template_identifier,
            path=str(template_dir),
            hint="Use a new --template-id, or a new --version for a re-ingest.",
        )

    commands: list[list[str]] = []
    warnings: list[str] = []

    probe = ffmpeg.probe(source, ffprobe=config.runtime.ffprobe_binary)
    spec, probe_warnings = build_video_spec(probe, strict_cfr=not options.allow_vfr_conversion)
    warnings.extend(probe_warnings)

    working_source = source
    if not spec.constant_frame_rate:
        if not options.allow_vfr_conversion:  # pragma: no cover - guarded above
            raise ValidationError("Refusing a variable-frame-rate source")
        converted = template_dir / "source_cfr.mp4"
        commands.append(
            convert_to_cfr(
                source,
                converted,
                fps=spec.fps,
                ffmpeg_binary=config.runtime.ffmpeg_binary,
            )
        )
        working_source = converted
        probe = ffmpeg.probe(converted, ffprobe=config.runtime.ffprobe_binary)
        spec, _ = build_video_spec(probe, strict_cfr=True)
        warnings.append(f"Source was converted to CFR at {spec.fps:.4f} fps: {converted}")

    directories = TemplateDirectories.standard(
        context.relative(template_dir)
        if template_dir.exists()
        else f"templates/{template_identifier}"
    )
    _create_directories(context, directories)

    frames_dir = context.absolute(directories.source_frames)
    extracted = 0
    if options.extract_frames:
        argv = ffmpeg.build_extract_frames_command(
            working_source,
            frames_dir / ffmpeg_pattern(DEFAULT_TEMPLATE),
            ffmpeg=config.runtime.ffmpeg_binary,
            start_frame=0,
        )
        ffmpeg.run_command(argv)
        commands.append(argv)
    indices = list_frame_indices(frames_dir)
    extracted = len(indices)
    if extracted == 0:
        raise MediaToolError(
            "Frame extraction produced no frames",
            frames_dir=str(frames_dir),
            source=str(working_source),
        )
    if indices != list(range(extracted)):
        raise ValidationError(
            "Extracted frames are not a contiguous range starting at 0",
            first=indices[0],
            last=indices[-1],
            count=extracted,
        )

    # The real extracted count replaces any container-reported estimate.
    if extracted != spec.frame_count:
        warnings.append(
            f"Container reported {spec.frame_count} frames; {extracted} were "
            "extracted. Using the extracted count."
        )
        spec = spec.model_copy(update={"frame_count": extracted})

    anchor = options.transition_anchor if options.transition_anchor is not None else extracted // 2
    reveal_end = options.reveal_end if options.reveal_end is not None else extracted
    _validate_segmentation(options.intro_start, anchor, reveal_end, extracted)

    identity_paths, identity_hashes = _copy_identity_refs(
        context, directories, options.identity_reference_images
    )
    background = _copy_background_plate(context, directories, options.background_plate)

    template = HumanTemplate(
        id=template_identifier,
        version=options.version,
        display_name=options.display_name,
        source_video_path=str(working_source),
        source_sha256=sha256_file(working_source),
        video=spec,
        intro=FrameRange(start=options.intro_start, end=anchor),
        reveal=FrameRange(start=anchor, end=reveal_end),
        transition_anchor_frame=anchor,
        identity_reference_images=identity_paths,
        identity_reference_hashes=identity_hashes,
        consent=ConsentRecord(
            subject_kind=options.subject_kind,
            adult_confirmed=options.adult_confirmed,
            consent_document_ref=options.consent_document_ref,
            rights_holder=options.rights_holder,
            license=options.license,
            provenance_notes=options.provenance_notes,
        ),
        template_clothing_class=options.template_clothing_class,  # type: ignore[arg-type]
        directories=directories,
        background_plate_path=background,
        source_frames_sha256=sha256_dir(frames_dir),
        extracted_frame_count=extracted,
        status=ProcessingStatus.AWAITING_MASKS,
        preprocessing_version=PREPROCESSING_VERSION,
        tool_versions=_tool_versions(config),
        notes=options.notes,
    )

    saved = context.repos.templates.save(template, allow_update=False)
    context.repos.audit.record(
        "template_ingested",
        entity_type="template",
        entity_id=saved.id,
        details={
            "version": saved.version,
            "frames": extracted,
            "source_sha256": saved.source_sha256,
            "anchor": anchor,
        },
    )
    log_event(
        logger,
        "template_ingested",
        template_id=saved.id,
        version=saved.version,
        frames=extracted,
        anchor=anchor,
    )
    return IngestResult(
        template=saved,
        probe=probe.raw,
        commands=commands,
        extracted_frames=extracted,
        warnings=warnings,
    )


def _validate_segmentation(
    intro_start: int, anchor: int, reveal_end: int, frame_count: int
) -> None:
    """Reject every off-by-one before it can reach the database."""
    if intro_start < 0:
        raise ValidationError("intro_start must be >= 0", intro_start=intro_start)
    if anchor <= intro_start:
        raise ValidationError(
            "The transition anchor must be after the intro start",
            intro_start=intro_start,
            anchor=anchor,
        )
    if reveal_end <= anchor:
        raise ValidationError(
            "reveal_end must be after the transition anchor",
            anchor=anchor,
            reveal_end=reveal_end,
        )
    if reveal_end > frame_count:
        raise ValidationError(
            "reveal_end exceeds the number of extracted frames",
            reveal_end=reveal_end,
            frame_count=frame_count,
            hint=f"Frames are 0..{frame_count - 1}; reveal_end is exclusive.",
        )


def _create_directories(context: ServiceContext, directories: TemplateDirectories) -> None:
    for relative in directories.all_dirs():
        context.absolute(relative).mkdir(parents=True, exist_ok=True)
    # A README in each mask directory tells the operator exactly what belongs
    # there, which matters because these are authored by hand today.
    for kind, attribute in MASK_DIR_NAMES.items():
        target = context.absolute(getattr(directories, attribute))
        (target / "README.txt").write_text(_mask_dir_readme(kind), encoding="utf-8")


def _mask_dir_readme(kind: MaskKind) -> str:
    purpose = {
        MaskKind.GARMENT: ("The garment region to be replaced. 255 = replace, 0 = keep source."),
        MaskKind.EXPANSION: (
            "Extra area the new garment may occupy beyond the base outfit "
            "(longer sleeves, fuller skirt). Unioned with the garment mask."
        ),
        MaskKind.PROTECTED: (
            "Face, hair, hands, exposed skin and background. ALWAYS wins over "
            "the garment mask. 255 = protect."
        ),
        MaskKind.OCCLUSION: (
            "Things in FRONT of the garment (hair strands, hands, props). "
            "Subtracted from the editable region. 255 = occluded."
        ),
    }[kind]
    return (
        f"{kind.value.upper()} MASKS\n"
        f"{'=' * (len(kind.value) + 6)}\n\n"
        f"{purpose}\n\n"
        "Format: 8-bit grayscale PNG, one per frame, named frame_000000.png,\n"
        "matching the source frame dimensions exactly.\n\n"
        "Values: 0 = immutable, 255 = editable/selected, 1..254 = feather.\n"
        "See docs/mask-semantics.md.\n"
    )


def _copy_identity_refs(
    context: ServiceContext,
    directories: TemplateDirectories,
    sources: list[Path],
) -> tuple[list[str], dict[str, str]]:
    target_dir = context.absolute(directories.identity_refs)
    paths: list[str] = []
    hashes: dict[str, str] = {}
    for source in sources:
        resolved = Path(source).expanduser().resolve()
        if not resolved.is_file():
            raise NotFoundError("Identity reference image not found", path=str(resolved))
        destination = target_dir / resolved.name
        shutil.copy2(resolved, destination)
        relative = context.relative(destination)
        paths.append(relative)
        hashes[relative] = sha256_file(destination)
    return paths, hashes


def _copy_background_plate(
    context: ServiceContext,
    directories: TemplateDirectories,
    source: Path | None,
) -> str | None:
    if source is None:
        return None
    resolved = Path(source).expanduser().resolve()
    if not resolved.is_file():
        raise NotFoundError("Background plate not found", path=str(resolved))
    target_dir = context.absolute(
        directories.background_plate or f"{directories.root}/background_plate"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / resolved.name
    shutil.copy2(resolved, destination)
    return context.relative(destination)


def _tool_versions(config: Any) -> dict[str, str]:
    from app.core.provenance import dependency_versions, ffmpeg_version

    versions = {"preprocessing": PREPROCESSING_VERSION}
    banner = ffmpeg_version(config.runtime.ffmpeg_binary)
    if banner:
        versions["ffmpeg"] = banner
    versions.update(dependency_versions())
    return versions


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------
@dataclass
class MaskImportResult:
    kind: MaskKind
    imported: list[int]
    skipped: list[str]
    destination: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "imported_count": len(self.imported),
            "imported_range": ([min(self.imported), max(self.imported)] if self.imported else None),
            "skipped": self.skipped[:32],
            "destination": self.destination,
        }


def import_masks(
    context: ServiceContext,
    template_id: str,
    kind: MaskKind,
    source_dir: str | Path,
    *,
    version: int | None = None,
    overwrite: bool = False,
    template_filename: str = DEFAULT_TEMPLATE,
) -> MaskImportResult:
    """Copy externally authored masks into a template's mask directory.

    Each file is validated for readability and dimensions before it is copied,
    so a half-imported directory cannot leave the template in a state where a
    render would fail mid-way.
    """
    template = context.repos.templates.get(template_id, version)
    source = Path(source_dir).expanduser().resolve()
    if not source.is_dir():
        raise NotFoundError("Mask source directory not found", path=str(source))

    destination = context.absolute(template.directories.mask_dir(kind))
    destination.mkdir(parents=True, exist_ok=True)
    expected_shape = (template.video.height, template.video.width)

    candidates = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() == ".png")
    if not candidates:
        raise ValidationError("No PNG masks found in the source directory", path=str(source))

    staged: list[tuple[int, Path]] = []
    skipped: list[str] = []
    prefix, _, suffix = template_filename.partition("{index:06d}")
    for path in candidates:
        name = path.name
        if not (name.startswith(prefix) and name.endswith(suffix)):
            skipped.append(f"{name}: filename does not match {template_filename}")
            continue
        digits = name[len(prefix) : len(name) - len(suffix)]
        if not digits.isdigit():
            skipped.append(f"{name}: no frame index in filename")
            continue
        index = int(digits)
        if index >= template.extracted_frame_count:
            skipped.append(f"{name}: frame {index} is beyond the template's frames")
            continue
        load_mask(path, expect_shape=expected_shape)  # raises on mismatch
        staged.append((index, path))

    if not staged:
        raise ValidationError(
            "No importable masks after validation",
            source=str(source),
            skipped=skipped[:32],
        )

    imported: list[int] = []
    for index, path in staged:
        target = frame_path(destination, index, template_filename)
        if target.exists() and not overwrite:
            skipped.append(f"{target.name}: already exists (use --overwrite)")
            continue
        target.write_bytes(path.read_bytes())
        imported.append(index)

    updated = template.model_copy(
        update={
            "status": (
                ProcessingStatus.MASKS_IMPORTED
                if template.status is ProcessingStatus.AWAITING_MASKS
                else template.status
            )
        }
    )
    context.repos.templates.save(updated)
    context.repos.audit.record(
        "masks_imported",
        entity_type="template",
        entity_id=template.id,
        details={"kind": kind.value, "count": len(imported), "source": str(source)},
    )
    log_event(
        logger,
        "masks_imported",
        template_id=template.id,
        kind=kind.value,
        count=len(imported),
    )
    return MaskImportResult(
        kind=kind,
        imported=sorted(imported),
        skipped=skipped,
        destination=context.relative(destination),
    )


@dataclass
class TemplateValidation:
    template_id: str
    version: int
    ok: bool
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "version": self.version,
            "ok": self.ok,
            "problems": self.problems,
            "warnings": self.warnings,
            "details": self.details,
        }


def validate_template(
    context: ServiceContext,
    template_id: str,
    *,
    version: int | None = None,
    require_masks: tuple[MaskKind, ...] = (MaskKind.GARMENT, MaskKind.PROTECTED),
    mark_ready: bool = True,
) -> TemplateValidation:
    """Verify that a template is complete enough to render.

    Checks the frame sequence, the immutability hash and the presence and size
    of every required mask across the whole reveal range.
    """
    template = context.repos.templates.get(template_id, version)
    problems: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {}

    frames_dir = context.absolute(template.directories.source_frames)
    sequence = validate_sequence(
        frames_dir,
        0,
        template.extracted_frame_count,
        expect_width=template.video.width,
        expect_height=template.video.height,
    )
    details["source_frames"] = sequence.as_dict()
    if not sequence.ok:
        problems.append(
            f"source frames incomplete: {len(sequence.missing)} missing, "
            f"{len(sequence.wrong_size)} wrong size"
        )

    if template.source_frames_sha256:
        current = sha256_dir(frames_dir)
        details["source_frames_sha256_matches"] = current == template.source_frames_sha256
        if current != template.source_frames_sha256:
            problems.append(
                "source frames have changed since ingestion "
                f"(expected {template.source_frames_sha256[:12]}, got {current[:12]})"
            )

    reveal = template.reveal
    mask_details: dict[str, Any] = {}
    for kind in MaskKind:
        directory = context.absolute(template.directories.mask_dir(kind))
        report = validate_sequence(
            directory,
            reveal.start,
            reveal.end,
            expect_width=template.video.width,
            expect_height=template.video.height,
        )
        mask_details[kind.value] = report.as_dict()
        if kind in require_masks:
            if not report.present:
                problems.append(
                    f"{kind.value} masks are missing entirely for reveal frames "
                    f"{reveal.start}..{reveal.last}"
                )
            elif report.missing:
                problems.append(
                    f"{kind.value} masks missing for {len(report.missing)} reveal "
                    f"frames (first: {report.missing[0]})"
                )
            if report.wrong_size:
                problems.append(
                    f"{kind.value} masks have wrong dimensions for "
                    f"{len(report.wrong_size)} frames"
                )
        elif not report.present:
            warnings.append(f"no {kind.value} masks present (optional)")
    details["masks"] = mask_details

    adapter_states = {kind.value: _adapter_status(kind) for kind in AdapterKind}
    details["adapters"] = adapter_states

    ok = not problems
    if ok and mark_ready and template.status is not ProcessingStatus.READY:
        context.repos.templates.save(template.model_copy(update={"status": ProcessingStatus.READY}))
    elif not ok and template.status is ProcessingStatus.READY:
        context.repos.templates.save(
            template.model_copy(update={"status": ProcessingStatus.VALIDATED})
        )

    return TemplateValidation(
        template_id=template.id,
        version=template.version,
        ok=ok,
        problems=problems,
        warnings=warnings,
        details=details,
    )


def verify_source_immutability(context: ServiceContext, template: HumanTemplate) -> None:
    """Raise if the immutable source frames or video have changed."""
    frames_dir = context.absolute(template.directories.source_frames)
    if template.source_frames_sha256:
        current = sha256_dir(frames_dir)
        if current != template.source_frames_sha256:
            raise ImmutabilityError(
                "The template's source frames have been modified since ingestion. "
                "Renders must read the original pixels; refusing to continue.",
                template_id=template.id,
                expected=template.source_frames_sha256,
                actual=current,
            )
    source = Path(template.source_video_path)
    if source.is_file():
        digest = sha256_file(source)
        if digest != template.source_sha256:
            raise ImmutabilityError(
                "The template's source video has changed since ingestion.",
                template_id=template.id,
                expected=template.source_sha256,
                actual=digest,
            )


__all__ = [
    "IngestOptions",
    "IngestResult",
    "MaskImportResult",
    "TemplateValidation",
    "build_video_spec",
    "convert_to_cfr",
    "import_masks",
    "ingest_template",
    "validate_template",
    "verify_source_immutability",
]
