"""Promoting an accepted synthetic master into the garment pipeline.

Acceptance and promotion are two different acts and this module exists because
conflating them was a bug:

``accept_master``
    An operator says "yes, this is our character, performing correctly." It
    changes status and writes an audit record. Nothing else.
``promote_master``
    The system turns that accepted candidate into a :class:`HumanTemplate` — the
    immutable record the garment pipeline actually consumes. New directories,
    frozen pixels, an intro/reveal split around a transition anchor, hashes.

Keeping them apart means an operator can accept without choosing an anchor, and
a failed promotion leaves an accepted candidate rather than a half-built
template.

**The generated PNGs are the source of truth.** They are hardlinked (or copied)
into the template's ``source_frames/`` and hashed there. They are never encoded
to H.264 and decoded back first: that would quantise, subsample chroma and
shift every pixel the garment pipeline later promises to preserve byte for
byte. An archival MP4 may be written alongside for operators to watch, and the
template's ``source_video_path`` points at it, but no part of rendering reads
it — :func:`~app.pipeline.template_ingest.verify_source_immutability` checks
``source_frames_sha256`` against the directory, and the compositor reads PNGs.

Promotion is idempotent: running it twice returns the same template rather than
building a second one. If it fails part way, the template record and directory
it created are removed, so there is no such thing as a half-promoted candidate.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.errors import ConflictError, MediaToolError, NotFoundError, ValidationError
from app.core.hashing import sha256_dir, sha256_file
from app.core.ids import template_id as new_template_id
from app.core.logging import get_logger, log_event
from app.core.paths import safe_identifier
from app.domain.enums import ProcessingStatus, TemplateClothingClass
from app.domain.human_template import (
    ConsentRecord,
    FrameRange,
    HumanTemplate,
    TemplateDirectories,
    VideoSpec,
)
from app.domain.master import MasterCandidate
from app.domain.motion import MotionComposition
from app.media import ffmpeg
from app.media.frames import DEFAULT_TEMPLATE, ffmpeg_pattern, frame_path, list_frame_indices
from app.pipeline.context import ServiceContext

# Reused from ingestion on purpose: a promoted template must have exactly the
# same on-disk shape and tool provenance as an ingested one, or the garment
# pipeline would be able to tell the two origins apart.
from app.pipeline.template_ingest import _create_directories, _tool_versions
from app.version import PREPROCESSING_VERSION

logger = get_logger(__name__)

#: Where the archival video lands inside the candidate's directory.
ARCHIVE_VIDEO_NAME = "master_archive.mp4"


@dataclass
class PromoteOptions:
    """Everything promotion needs that the candidate cannot tell us itself."""

    #: Frame where the intro ends and the reveal begins. ``None`` means "use the
    #: anchor the composition recommended", which is only unambiguous when the
    #: composition has exactly one join.
    transition_anchor: int | None = None
    template_id: str | None = None
    display_name: str | None = None
    template_clothing_class: TemplateClothingClass = TemplateClothingClass.FITTED_SHORT
    promoted_by: str = "operator"
    #: Required when the composition has more than one join and no explicit
    #: anchor was given: the operator must say which seam is the reveal.
    confirm_multiple_joins: bool = False
    encode_archive_video: bool = True
    notes: str | None = None


@dataclass
class PromoteResult:
    template: HumanTemplate
    candidate: MasterCandidate
    transition_anchor: int
    #: ``explicit`` | ``composition_join`` | ``already_promoted``
    anchor_source: str
    frames_linked: int = 0
    hardlinked: bool = False
    archive_video: str | None = None
    created: bool = True
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template.id,
            "template_version": self.template.version,
            "candidate_id": self.candidate.id,
            "transition_anchor": self.transition_anchor,
            "anchor_source": self.anchor_source,
            "intro": [self.template.intro.start, self.template.intro.end],
            "reveal": [self.template.reveal.start, self.template.reveal.end],
            "frames": self.frames_linked,
            "hardlinked": self.hardlinked,
            "source_frames_sha256": self.template.source_frames_sha256,
            "archive_video": self.archive_video,
            "created": self.created,
            "warnings": self.warnings,
        }


def resolve_transition_anchor(
    composition: MotionComposition,
    *,
    explicit: int | None,
    confirm_multiple_joins: bool,
) -> tuple[int, str]:
    """Pick the reveal anchor, or refuse to guess.

    A composition join already knows where the borrowed opening motion ends and
    the new performance begins — that is what ``recommended_transition_anchor``
    records, in output frame numbering. With more than one join there is more
    than one defensible seam, and silently taking the first would bake an
    arbitrary choice into an immutable template.
    """
    if explicit is not None:
        return explicit, "explicit"

    recommended = [
        join.recommended_transition_anchor
        for join in composition.joins
        if join.recommended_transition_anchor is not None
    ]
    if not recommended:
        raise ValidationError(
            "The composition records no recommended transition anchor, so one "
            "must be given explicitly",
            composition_id=composition.id,
            joins=len(composition.joins),
            hint="Pass --transition-anchor <frame>.",
        )
    if len(recommended) > 1 and not confirm_multiple_joins:
        raise ValidationError(
            "This composition has more than one join, so its reveal anchor is "
            "an operator decision, not a default",
            composition_id=composition.id,
            recommended_anchors=recommended,
            hint=(
                "Pass --transition-anchor <frame> to choose, or "
                "--confirm-multiple-joins to accept the first recommendation."
            ),
        )
    return int(recommended[0]), "composition_join"


def promote_master(
    context: ServiceContext,
    candidate_id: str,
    options: PromoteOptions | None = None,
) -> PromoteResult:
    """Turn an accepted master candidate into an immutable ``HumanTemplate``."""
    opts = options or PromoteOptions()
    candidate = context.repos.masters.get(candidate_id)

    if not candidate.is_accepted:
        raise ConflictError(
            "Only an accepted master candidate can be promoted",
            candidate_id=candidate.id,
            status=candidate.status.value,
            hint="Run `app master accept` first; acceptance is a human decision.",
        )

    # -- idempotence ------------------------------------------------------
    if candidate.promoted_template_id:
        existing = context.repos.templates.try_get(candidate.promoted_template_id)
        if existing is None:
            raise ConflictError(
                "The candidate records a promoted template that no longer exists",
                candidate_id=candidate.id,
                template_id=candidate.promoted_template_id,
                hint="Restore the template, or clear promoted_template_id deliberately.",
            )
        if opts.transition_anchor is not None and opts.transition_anchor != (
            existing.transition_anchor_frame
        ):
            raise ConflictError(
                "This candidate is already promoted with a different transition anchor. "
                "A promoted master is immutable; promote a new candidate instead.",
                candidate_id=candidate.id,
                template_id=existing.id,
                existing_anchor=existing.transition_anchor_frame,
                requested_anchor=opts.transition_anchor,
            )
        return PromoteResult(
            template=existing,
            candidate=candidate,
            transition_anchor=existing.transition_anchor_frame,
            anchor_source="already_promoted",
            frames_linked=existing.extracted_frame_count,
            created=False,
        )

    # -- the candidate must actually be finished --------------------------
    remaining = candidate.remaining_frames()
    if remaining:
        raise ConflictError(
            "Cannot promote a master whose animation is incomplete",
            candidate_id=candidate.id,
            missing_frames=len(remaining),
            first_missing=remaining[0],
        )
    qc = (candidate.qc_metrics or {}).get("qc")
    if qc is None:
        raise ConflictError(
            "Cannot promote a master that has not been QC'd",
            candidate_id=candidate.id,
            hint="Run `app master qc` first.",
        )

    frames_dir = context.absolute(candidate.frames_dir)
    indices = list_frame_indices(frames_dir)
    if indices != list(range(candidate.frame_count)):
        raise ValidationError(
            "The candidate's frames are not a complete contiguous range",
            candidate_id=candidate.id,
            expected=candidate.frame_count,
            found=len(indices),
            first=indices[0] if indices else None,
            last=indices[-1] if indices else None,
        )

    composition = context.repos.compositions.get(
        candidate.composition_id, candidate.composition_version
    )
    anchor, anchor_source = resolve_transition_anchor(
        composition,
        explicit=opts.transition_anchor,
        confirm_multiple_joins=opts.confirm_multiple_joins,
    )
    _validate_anchor(anchor, candidate.frame_count)

    hero = context.repos.heroes.get(candidate.hero_character_id, candidate.hero_character_version)

    identifier = safe_identifier(opts.template_id or new_template_id())
    template_dir = context.data_root.template_dir(identifier)
    if template_dir.exists() and any(template_dir.iterdir()):
        raise ConflictError(
            "Template directory already exists and is not empty",
            template_id=identifier,
            path=str(template_dir),
        )
    if context.repos.templates.try_get(identifier) is not None:
        raise ConflictError("Template id is already in use", template_id=identifier)

    warnings: list[str] = []
    created_dir = False
    saved_template: HumanTemplate | None = None
    try:
        directories = TemplateDirectories.standard(f"templates/{identifier}")
        _create_directories(context, directories)
        created_dir = True

        source_frames = context.absolute(directories.source_frames)
        linked, hardlinked = _freeze_frames(frames_dir, source_frames, indices)

        identity_paths, identity_hashes = _copy_hero_references(context, directories, hero)

        archive: Path | None = None
        if opts.encode_archive_video:
            archive = _encode_archive(context, candidate, source_frames)
            if archive is None:
                warnings.append(
                    "No archival video was encoded (ffmpeg unavailable). The "
                    "template's pixels are unaffected: rendering reads the PNGs."
                )

        spec = _video_spec(context, candidate, archive)
        template = HumanTemplate(
            id=identifier,
            version=1,
            display_name=opts.display_name or candidate.display_name,
            # Archival only. The immutable pixels are source_frames/, hashed
            # below; nothing in the render path decodes this file.
            source_video_path=str(archive) if archive else str(source_frames),
            source_sha256=sha256_file(archive) if archive else sha256_dir(source_frames),
            video=spec,
            intro=FrameRange(start=0, end=anchor),
            reveal=FrameRange(start=anchor, end=candidate.frame_count),
            transition_anchor_frame=anchor,
            identity_reference_images=identity_paths,
            identity_reference_hashes=identity_hashes,
            consent=ConsentRecord(
                subject_kind=hero.subject_kind,
                adult_confirmed=hero.adult_confirmed,
                consent_document_ref=hero.consent_document_ref,
                rights_holder=hero.rights_holder,
                license=hero.license,
                provenance_notes=(
                    "Synthetic master: Hero Character "
                    f"{hero.version_key()} animated from composition "
                    f"{composition.version_key()} as candidate {candidate.id}. "
                    "Contains no source-performer pixels."
                ),
            ),
            template_clothing_class=opts.template_clothing_class,
            directories=directories,
            source_frames_sha256=sha256_dir(source_frames),
            extracted_frame_count=len(indices),
            status=ProcessingStatus.AWAITING_MASKS,
            preprocessing_version=PREPROCESSING_VERSION,
            tool_versions=_tool_versions(context.config),
            notes=opts.notes,
        )
        saved_template = context.repos.templates.save(template, allow_update=False)

        output_hashes = {
            "source_frames_sha256": saved_template.source_frames_sha256 or "",
            "promoted_template_id": saved_template.id,
        }
        if archive is not None:
            output_hashes["archive_video_sha256"] = sha256_file(archive)
        updated = context.repos.masters.save(
            candidate.model_copy(
                update={
                    "promoted_template_id": saved_template.id,
                    "video_path": context.relative(archive) if archive else candidate.video_path,
                    "output_hashes": {**candidate.output_hashes, **output_hashes},
                }
            )
        )
    except Exception:
        # No half-promoted candidates: undo the record and the directory before
        # re-raising, so a retry starts from the same state as the first try.
        if saved_template is not None:
            context.repos.templates.delete(saved_template.id, saved_template.version)
        if created_dir and template_dir.exists():
            shutil.rmtree(template_dir, ignore_errors=True)
        raise

    from app.pipeline.master_create import write_master_manifest

    write_master_manifest(context, updated)
    context.repos.audit.record(
        "master_promoted",
        actor=opts.promoted_by,
        entity_type="master_candidate",
        entity_id=updated.id,
        details={
            "template_id": saved_template.id,
            "transition_anchor": anchor,
            "anchor_source": anchor_source,
            "frames": len(indices),
            "hardlinked": hardlinked,
            "source_frames_sha256": saved_template.source_frames_sha256,
        },
    )
    log_event(
        logger,
        "master_promoted",
        candidate_id=updated.id,
        template_id=saved_template.id,
        anchor=anchor,
        frames=len(indices),
    )
    return PromoteResult(
        template=saved_template,
        candidate=updated,
        transition_anchor=anchor,
        anchor_source=anchor_source,
        frames_linked=linked,
        hardlinked=hardlinked,
        archive_video=context.relative(archive) if archive else None,
        created=True,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _validate_anchor(anchor: int, frame_count: int) -> None:
    if anchor <= 0:
        raise ValidationError(
            "The transition anchor must leave at least one intro frame",
            anchor=anchor,
        )
    if anchor >= frame_count:
        raise ValidationError(
            "The transition anchor must leave at least one reveal frame",
            anchor=anchor,
            frame_count=frame_count,
            hint=f"Frames are 0..{frame_count - 1}; the anchor is the first reveal frame.",
        )


def _freeze_frames(source: Path, destination: Path, indices: list[int]) -> tuple[int, bool]:
    """Hardlink (or copy) the generated PNGs into the template, unchanged.

    Returns ``(count, hardlinked)``. Hardlinking is preferred because a
    1080x1920 PNG sequence is large and a link is byte-identical by
    construction; a copy is the fallback where the filesystem refuses.
    """
    destination.mkdir(parents=True, exist_ok=True)
    hardlinked = True
    count = 0
    for index in indices:
        origin = frame_path(source, index)
        target = frame_path(destination, index)
        if target.exists():
            target.unlink()
        try:
            os.link(origin, target)
        except (OSError, NotImplementedError):
            shutil.copy2(origin, target)
            hardlinked = False
        count += 1
    return count, hardlinked


def _copy_hero_references(
    context: ServiceContext, directories: TemplateDirectories, hero: Any
) -> tuple[list[str], dict[str, str]]:
    """Copy the Hero Character's reference images into the template."""
    target_dir = context.absolute(directories.identity_refs)
    target_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    hashes: dict[str, str] = {}
    for relative in hero.reference_images:
        source = context.absolute(relative)
        if not source.is_file():
            raise NotFoundError(
                "Hero Character reference image is missing",
                hero_id=hero.id,
                path=str(source),
            )
        target = target_dir / source.name
        shutil.copy2(source, target)
        stored = context.relative(target)
        paths.append(stored)
        hashes[stored] = sha256_file(target)
    return paths, hashes


def _encode_archive(
    context: ServiceContext, candidate: MasterCandidate, source_frames: Path
) -> Path | None:
    """Encode an operator-facing MP4. Never read back as template pixels."""
    binary = context.config.runtime.ffmpeg_binary
    if not ffmpeg.tool_available(binary):
        return None
    destination = context.absolute(candidate.frames_dir).parent / ARCHIVE_VIDEO_NAME
    argv = ffmpeg.build_encode_from_frames_command(
        source_frames / ffmpeg_pattern(DEFAULT_TEMPLATE),
        destination,
        ffmpeg=binary,
        fps=candidate.fps,
        width=candidate.width,
        height=candidate.height,
    )
    result = ffmpeg.run_command(argv)
    if not result.ok or not destination.is_file():
        raise MediaToolError(
            "Failed to encode the archival master video",
            candidate_id=candidate.id,
            stderr=result.stderr[-2000:],
        )
    return destination


def _video_spec(
    context: ServiceContext, candidate: MasterCandidate, archive: Path | None
) -> VideoSpec:
    """Describe the master. The frame count is the PNG count, always.

    The container is asked only for codec and pixel format. Where the two
    disagree about frame count, the files on disk win — they are what renders.
    """
    codec: str | None = None
    pixel_format: str | None = None
    if archive is not None:
        try:
            probe = ffmpeg.probe(archive, ffprobe=context.config.runtime.ffprobe_binary)
        except MediaToolError:  # pragma: no cover - archival metadata is optional
            probe = None
        video = probe.video if probe is not None else None
        if video is not None:
            codec = video.codec_name
            pixel_format = video.pix_fmt
    return VideoSpec(
        width=candidate.width,
        height=candidate.height,
        fps=candidate.fps,
        duration_s=max(candidate.frame_count / candidate.fps, 1e-6),
        frame_count=candidate.frame_count,
        codec=codec,
        pixel_format=pixel_format,
        constant_frame_rate=True,
        has_audio=False,
    )


__all__ = [
    "ARCHIVE_VIDEO_NAME",
    "PromoteOptions",
    "PromoteResult",
    "promote_master",
    "resolve_transition_anchor",
]
