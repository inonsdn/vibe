"""Synthetic master creation: animate a Hero Character from a composition.

The pipeline is chunked for an 8GB card, and the chunks are stitched by
*conditioning*, not concatenation: each chunk receives the last N accepted
frames as context, and only the chunk's own new frames are kept. The overlap
region is therefore generated once, never twice-and-crossfaded.

The output is a **candidate**. It becomes a Master Human Performance only when
an operator reviews QC and accepts it — see :func:`accept_master`. That gate
exists because the candidate is the one artifact a model invented wholesale;
everything downstream treats the master as ground truth.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.backends.animator.base import (
    AnimationChunkRequest,
    AnimatorContext,
    CharacterAnimatorBackend,
)
from app.backends.animator.registry import create_animator
from app.core.determinism import derive_seed
from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.core.hashing import sha256_file, sha256_json
from app.core.ids import new_id, utc_now
from app.core.logging import get_logger, log_context, log_event
from app.core.paths import safe_identifier
from app.core.provenance import collect as collect_provenance
from app.domain.master import (
    ChunkRecord,
    HeroCharacter,
    MasterAcceptance,
    MasterCandidate,
    MasterCandidateStatus,
    MasterOrigin,
)
from app.domain.motion import MotionComposition
from app.media.frames import frame_path, list_frame_indices, read_frame, write_frame
from app.motion.pose_format import load_pose_sequence
from app.pipeline.context import ServiceContext

logger = get_logger(__name__)

ProgressCallback = Callable[[int, int, int], None]  # (chunk_index, done, total)


def master_candidate_id() -> str:
    return new_id("mst")


def hero_character_id() -> str:
    return new_id("hero")


@dataclass
class HeroOptions:
    display_name: str
    reference_images: list[Path]
    subject_kind: str = "synthetic"
    consent_document_ref: str | None = None
    rights_holder: str | None = None
    license: str | None = None
    identity_notes: str | None = None
    hero_id: str | None = None
    version: int = 1


def register_hero(context: ServiceContext, options: HeroOptions) -> HeroCharacter:
    """Copy and hash a Hero Character's reference images, then record it."""
    import shutil

    if not options.reference_images:
        raise ValidationError("A Hero Character needs at least one reference image")

    identifier = safe_identifier(options.hero_id or hero_character_id())
    images_dir = context.data_root.resolve("heroes", identifier, "images")
    images_dir.mkdir(parents=True, exist_ok=True)

    paths: list[str] = []
    hashes: dict[str, str] = {}
    for source in options.reference_images:
        resolved = Path(source).expanduser().resolve()
        if not resolved.is_file():
            raise NotFoundError("Hero reference image not found", path=str(resolved))
        destination = images_dir / resolved.name
        shutil.copy2(resolved, destination)
        relative = context.relative(destination)
        paths.append(relative)
        hashes[relative] = sha256_file(destination)

    hero = HeroCharacter(
        id=identifier,
        version=options.version,
        display_name=options.display_name,
        reference_images=paths,
        reference_hashes=hashes,
        identity_notes=options.identity_notes,
        subject_kind=options.subject_kind,
        consent_document_ref=options.consent_document_ref,
        rights_holder=options.rights_holder,
        license=options.license,
    )
    saved = context.repos.heroes.save(hero)
    context.repos.audit.record(
        "hero_registered",
        entity_type="hero_character",
        entity_id=saved.id,
        details={"version": saved.version, "images": len(paths)},
    )
    return saved


@dataclass
class MasterCreateOptions:
    display_name: str
    composition_id: str
    hero_character_id: str
    composition_version: int | None = None
    hero_version: int | None = None
    backend_name: str = "mock"
    seed: int | None = None
    chunk_frames: int | None = None
    overlap_frames: int | None = None
    workflow_id: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    candidate_id: str | None = None
    version: int = 1


@dataclass
class MasterAnimateResult:
    candidate: MasterCandidate
    animated_chunks: list[int]
    skipped_chunks: list[int]
    backend_info: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate.id,
            "status": self.candidate.status.value,
            "animated_chunks": len(self.animated_chunks),
            "skipped_chunks": len(self.skipped_chunks),
            "frame_count": self.candidate.frame_count,
            "frames_done": len(self.candidate.completed_chunk_frames()),
            "backend": self.backend_info.get("backend", self.candidate.backend_name),
        }


def create_master_candidate(
    context: ServiceContext,
    options: MasterCreateOptions,
) -> MasterCandidate:
    """Create a candidate master from a composition and a Hero Character."""
    composition = context.repos.compositions.get(
        options.composition_id, options.composition_version
    )
    hero = context.repos.heroes.get(options.hero_character_id, options.hero_version)
    profile = context.repos.skeleton_profiles.get(
        composition.skeleton_profile_id, composition.skeleton_profile_version
    )

    if composition.output_frame_count <= 0:
        raise ValidationError("Composition has no frames", composition_id=composition.id)

    identifier = safe_identifier(options.candidate_id or master_candidate_id())
    root = context.data_root.resolve("masters", identifier)
    frames_dir = root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    animator_config = context.config.animator
    chunk_frames = options.chunk_frames or animator_config.chunk_frames
    overlap = (
        options.overlap_frames
        if options.overlap_frames is not None
        else animator_config.overlap_frames
    )
    if overlap >= chunk_frames:
        raise ValidationError(
            "overlap_frames must be smaller than chunk_frames",
            chunk_frames=chunk_frames,
            overlap_frames=overlap,
        )

    backend = create_animator(options.backend_name, context.config)
    try:
        capabilities = backend.capabilities()
    finally:
        backend.close()

    candidate = MasterCandidate(
        id=identifier,
        version=options.version,
        display_name=options.display_name,
        origin=MasterOrigin.SYNTHETIC,
        composition_id=composition.id,
        composition_version=composition.version,
        hero_character_id=hero.id,
        hero_character_version=hero.version,
        backend_name=options.backend_name,
        backend_version=capabilities.version,
        workflow_id=options.workflow_id
        or (animator_config.workflow_id if options.backend_name == "comfyui" else None),
        seed=options.seed if options.seed is not None else context.config.runtime.default_seed,
        settings={
            "chunk_frames": chunk_frames,
            "overlap_frames": overlap,
            "low_vram_mode": animator_config.low_vram_mode,
            "cpu_offload": animator_config.cpu_offload,
            **options.settings,
        },
        frame_count=composition.output_frame_count,
        width=profile.target_width,
        height=profile.target_height,
        fps=composition.output_fps,
        frames_dir=context.relative(frames_dir),
        input_hashes=_input_hashes(context, composition, hero),
        status=MasterCandidateStatus.CREATED,
    )
    saved = context.repos.masters.save(candidate)
    context.repos.audit.record(
        "master_candidate_created",
        entity_type="master_candidate",
        entity_id=saved.id,
        details={
            "composition": composition.version_key(),
            "hero": hero.version_key(),
            "backend": saved.backend_name,
            "frames": saved.frame_count,
        },
    )
    log_event(
        logger,
        "master_candidate_created",
        candidate_id=saved.id,
        frames=saved.frame_count,
        chunk_frames=chunk_frames,
    )
    return saved


def plan_chunks(frame_count: int, chunk_frames: int, overlap: int) -> list[tuple[int, int, int]]:
    """Return ``(chunk_index, start, end)`` covering ``[0, frame_count)`` exactly.

    Chunks tile the range without overlapping in *output*: the overlap is the
    number of already-accepted frames handed to the backend as context, not a
    region generated twice. That distinction is what removes the need for a
    crossfade at a chunk boundary.
    """
    if frame_count <= 0:
        return []
    if chunk_frames <= 0:
        raise ValidationError("chunk_frames must be positive", chunk_frames=chunk_frames)
    if overlap >= chunk_frames:
        raise ValidationError(
            "overlap must be smaller than chunk_frames",
            chunk_frames=chunk_frames,
            overlap=overlap,
        )
    chunks: list[tuple[int, int, int]] = []
    start = 0
    index = 0
    while start < frame_count:
        end = min(start + chunk_frames, frame_count)
        chunks.append((index, start, end))
        start = end
        index += 1
    return chunks


def animate_master(
    context: ServiceContext,
    candidate_id: str,
    *,
    backend_name: str | None = None,
    resume: bool = False,
    max_chunks: int | None = None,
    progress: ProgressCallback | None = None,
) -> MasterAnimateResult:
    """Animate (or resume) a candidate master, chunk by chunk."""
    candidate = context.repos.masters.get(candidate_id)
    if candidate.status.is_terminal and not resume:
        raise ConflictError(
            "Master candidate has already finished",
            candidate_id=candidate.id,
            status=candidate.status.value,
        )
    if candidate.is_accepted:
        raise ConflictError(
            "An accepted master is immutable and cannot be re-animated",
            candidate_id=candidate.id,
        )

    composition = context.repos.compositions.get(
        candidate.composition_id, candidate.composition_version
    )
    hero = context.repos.heroes.get(candidate.hero_character_id, candidate.hero_character_version)
    profile = context.repos.skeleton_profiles.get(
        composition.skeleton_profile_id, composition.skeleton_profile_version
    )

    pose_dir = context.absolute(composition.composed_pose_dir)
    poses = load_pose_sequence(pose_dir, range(candidate.frame_count))
    pose_by_index = {pose.frame_index: pose for pose in poses}

    frames_dir = context.absolute(candidate.frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    hero_paths = [context.absolute(p) for p in hero.reference_images]
    workflow_path: Path | None = None
    if candidate.workflow_id:
        maybe = context.config.workflows_dir() / f"{candidate.workflow_id}.json"
        workflow_path = maybe if maybe.is_file() else None

    animator_context = AnimatorContext(
        candidate=candidate,
        composition=composition,
        hero=hero,
        profile=profile,
        config=context.config,
        candidate_dir=context.absolute(candidate.frames_dir).parent,
        frames_dir=frames_dir,
        pose_dir=pose_dir,
        hero_image_paths=hero_paths,
        workflow_path=workflow_path,
    )

    effective_backend = backend_name or candidate.backend_name
    backend = create_animator(effective_backend, context.config)

    with log_context(candidate_id=candidate.id, animator=effective_backend), backend:
        try:
            info = backend.resume(animator_context) if resume else backend.prepare(animator_context)
            capabilities = backend.capabilities()
            candidate = context.repos.masters.save(
                candidate.model_copy(
                    update={
                        "backend_name": effective_backend,
                        "backend_version": capabilities.version,
                        "workflow_sha256": info.get("workflow_sha256"),
                        "status": MasterCandidateStatus.ANIMATING,
                        "error": None,
                    }
                )
            )
            result = _animate_chunks(
                context,
                candidate,
                backend,
                animator_context,
                info,
                pose_by_index=pose_by_index,
                frames_dir=frames_dir,
                max_chunks=max_chunks,
                progress=progress,
            )
        except Exception as exc:
            failed = context.repos.masters.save(
                candidate.model_copy(
                    update={
                        "status": MasterCandidateStatus.FAILED,
                        "error": {
                            "code": getattr(exc, "code", type(exc).__name__),
                            "message": str(exc),
                            "occurred_at": utc_now().isoformat(),
                        },
                    }
                )
            )
            log_event(logger, "master_animation_failed", candidate_id=failed.id)
            raise
    return result


def _animate_chunks(
    context: ServiceContext,
    candidate: MasterCandidate,
    backend: CharacterAnimatorBackend,
    animator_context: AnimatorContext,
    backend_info: dict[str, Any],
    *,
    pose_by_index: dict[int, Any],
    frames_dir: Path,
    max_chunks: int | None,
    progress: ProgressCallback | None,
) -> MasterAnimateResult:
    capabilities = backend.capabilities()
    chunk_frames = min(
        int(candidate.settings.get("chunk_frames", 24)), capabilities.max_chunk_frames
    )
    configured_overlap = int(candidate.settings.get("overlap_frames", 16))
    # The plan tiles the sequence using the *configured* overlap, so chunk
    # boundaries do not move when a backend is swapped. How much of that tail is
    # actually read and handed over is the backend's declared business.
    plan = plan_chunks(candidate.frame_count, chunk_frames, configured_overlap)
    context_mode = capabilities.context_mode
    effective_overlap = capabilities.context_frames_for(configured_overlap)
    done = {c["chunk_index"] for c in context.repos.masters.completed_chunks(candidate.id)}
    pending = [chunk for chunk in plan if chunk[0] not in done]
    if max_chunks is not None:
        pending = pending[:max_chunks]

    animated: list[int] = []
    chunk_records: list[ChunkRecord] = list(candidate.chunks)

    for chunk_index, start, end in pending:
        # Context frames: the accepted tail immediately before this chunk. They
        # are read from disk rather than kept in memory, so a resumed run gets
        # exactly the same conditioning a single-pass run would.
        context_indices = [
            index
            for index in range(max(0, start - effective_overlap), start)
            if frame_path(frames_dir, index).is_file()
        ]
        # Contiguity matters: the request contract says context frames are the
        # run immediately preceding the chunk, so a hole in the middle means
        # this is not a valid tail and nothing may be claimed about it.
        if context_indices and context_indices != list(range(context_indices[0], start)):
            raise ValidationError(
                "Context frames preceding the chunk are not contiguous",
                chunk_index=chunk_index,
                start_frame=start,
                available=context_indices,
            )
        context_frames: list[np.ndarray] = [
            read_frame(frame_path(frames_dir, index)) for index in context_indices
        ]

        request = AnimationChunkRequest(
            chunk_index=chunk_index,
            start_frame=start,
            end_frame=end,
            poses=[pose_by_index[i] for i in range(start, end) if i in pose_by_index],
            # Derived from the INPUTS, deliberately not from the candidate id:
            # two candidates built from the same composition, hero, seed and
            # settings must produce identical frames, which is what makes the
            # manifest digest a reproducibility claim rather than a label.
            seed=derive_seed(
                "master-chunk",
                candidate.seed,
                chunk_index,
                f"{candidate.composition_id}@v{candidate.composition_version}",
                f"{candidate.hero_character_id}@v{candidate.hero_character_version}",
            ),
            context_frames=context_frames,
            context_poses=[pose_by_index[i] for i in context_indices if i in pose_by_index],
            settings=dict(candidate.settings),
        )
        if len(request.poses) != request.frame_count:
            raise ValidationError(
                "Composition is missing pose frames for this chunk",
                chunk_index=chunk_index,
                expected=request.frame_count,
                available=len(request.poses),
            )

        result = backend.animate_chunk(animator_context, request)
        returned = set(result.frames)
        expected = set(request.frame_indices)
        if returned != expected:
            raise ValidationError(
                "Animator returned a different set of frames than the chunk requested",
                chunk_index=chunk_index,
                expected=sorted(expected),
                returned=sorted(returned),
            )

        hashes: dict[str, str] = {}
        for index in sorted(result.frames):
            target = frame_path(frames_dir, index)
            write_frame(target, result.frames[index])
            hashes[str(index)] = sha256_file(target)

        context.repos.masters.record_chunk(
            candidate.id,
            chunk_index,
            start_frame=start,
            end_frame=end,
            status="completed",
            seed=result.seed,
            overlap_frames=len(context_indices),
            duration_ms=result.duration_ms,
        )
        chunk_records = [c for c in chunk_records if c.index != chunk_index]
        chunk_records.append(
            ChunkRecord(
                index=chunk_index,
                start_frame=start,
                end_frame=end,
                context_frames=context_indices,
                overlap_frames=len(context_indices),
                context_mode=context_mode.value,
                seed=result.seed,
                frame_hashes=hashes,
                duration_ms=result.duration_ms,
            )
        )
        animated.append(chunk_index)
        if progress is not None:
            progress(chunk_index, len(done) + len(animated), len(plan))

        candidate = context.repos.masters.save(
            candidate.model_copy(update={"chunks": sorted(chunk_records, key=lambda c: c.index)})
        )

    remaining = [
        c for c in plan if c[0] not in context.repos.masters.completed_chunks_indices(candidate.id)
    ]
    complete = not remaining
    candidate = context.repos.masters.save(
        candidate.model_copy(
            update={
                "status": (
                    MasterCandidateStatus.ANIMATED if complete else MasterCandidateStatus.PAUSED
                ),
            }
        )
    )
    log_event(
        logger,
        "master_animated" if complete else "master_paused",
        candidate_id=candidate.id,
        animated_chunks=len(animated),
        complete=complete,
    )
    return MasterAnimateResult(
        candidate=candidate,
        animated_chunks=animated,
        skipped_chunks=sorted(done),
        backend_info={**backend_info, **backend.collect_artifacts(animator_context)},
    )


def resume_master(
    context: ServiceContext,
    candidate_id: str,
    *,
    backend_name: str | None = None,
    max_chunks: int | None = None,
    progress: ProgressCallback | None = None,
) -> MasterAnimateResult:
    """Resume an interrupted candidate, skipping completed chunks."""
    candidate = context.repos.masters.get(candidate_id)
    if not candidate.status.is_resumable:
        raise ConflictError(
            "Master candidate is not resumable",
            candidate_id=candidate.id,
            status=candidate.status.value,
        )
    return animate_master(
        context,
        candidate_id,
        backend_name=backend_name,
        resume=True,
        max_chunks=max_chunks,
        progress=progress,
    )


# ---------------------------------------------------------------------------
# manifest and acceptance
# ---------------------------------------------------------------------------
def build_master_manifest(context: ServiceContext, candidate: MasterCandidate) -> dict[str, Any]:
    composition = context.repos.compositions.get(
        candidate.composition_id, candidate.composition_version
    )
    hero = context.repos.heroes.get(candidate.hero_character_id, candidate.hero_character_version)
    provenance = collect_provenance(context.config)
    frames_dir = context.absolute(candidate.frames_dir)

    frame_hashes: dict[str, str] = {}
    for chunk in candidate.chunks:
        frame_hashes.update(chunk.frame_hashes)

    payload: dict[str, Any] = {
        "schema_version": "1",
        "candidate_id": candidate.id,
        "candidate_version": candidate.version,
        "origin": candidate.origin.value,
        "created_at": utc_now().isoformat(),
        "composition": composition.version_key(),
        "composition_manifest": composition.manifest_path,
        "hero_character": hero.version_key(),
        "frame_count": candidate.frame_count,
        "width": candidate.width,
        "height": candidate.height,
        "fps": candidate.fps,
        "backend": {
            "name": candidate.backend_name,
            "version": candidate.backend_version,
            "workflow_id": candidate.workflow_id,
            "workflow_sha256": candidate.workflow_sha256,
        },
        "seed": candidate.seed,
        "settings": candidate.settings,
        "chunks": [chunk.model_dump(mode="json") for chunk in candidate.chunks],
        "input_hashes": candidate.input_hashes,
        "frame_hashes": frame_hashes,
        "frames_present": len(list_frame_indices(frames_dir)),
        "reproducibility": {
            "app_version": provenance["app"]["version"],
            "git": provenance["git"],
            "platform": provenance["platform"],
            "dependencies": provenance["dependencies"],
            "config_hash": context.config.config_hash(),
            "ffmpeg": provenance["ffmpeg"],
        },
        "contains_source_pixels": False,
        "acceptance": (
            candidate.acceptance.model_dump(mode="json") if candidate.acceptance else None
        ),
    }
    payload["digest"] = sha256_json(
        {
            "composition": payload["composition"],
            "hero": payload["hero_character"],
            "backend": payload["backend"],
            "seed": payload["seed"],
            "settings": payload["settings"],
            "input_hashes": payload["input_hashes"],
            "frame_hashes": dict(sorted(frame_hashes.items())),
            "frame_count": payload["frame_count"],
        }
    )
    return payload


def write_master_manifest(context: ServiceContext, candidate: MasterCandidate) -> dict[str, Any]:
    payload = build_master_manifest(context, candidate)
    root = context.absolute(candidate.frames_dir).parent
    path = root / "manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    context.repos.masters.save_manifest(candidate.id, payload["digest"], payload)
    context.repos.masters.save(
        candidate.model_copy(update={"manifest_path": context.relative(path)})
    )
    return payload


def accept_master(
    context: ServiceContext,
    candidate_id: str,
    *,
    accepted_by: str,
    reason: str,
    acknowledged_warnings: list[str] | None = None,
    require_qc_pass: bool = True,
) -> MasterCandidate:
    """Promote a candidate to an accepted, immutable master.

    Acceptance is explicit and audited. It is refused unless the candidate is
    fully animated and its QC has run — a synthetic master must never slide into
    production because nobody looked.
    """
    candidate = context.repos.masters.get(candidate_id)
    if candidate.is_accepted:
        raise ConflictError("Master candidate is already accepted", candidate_id=candidate.id)
    if len(reason.strip()) < 10:
        raise ValidationError(
            "An acceptance reason must be a real explanation (10+ characters)",
            reason=reason,
        )

    remaining = candidate.remaining_frames()
    if remaining:
        raise ConflictError(
            "Cannot accept a master whose animation is incomplete",
            candidate_id=candidate.id,
            missing_frames=len(remaining),
            first_missing=remaining[0],
            hint="Run `app master animate --resume` first.",
        )

    qc = candidate.qc_metrics.get("qc") if candidate.qc_metrics else None
    if qc is None:
        raise ConflictError(
            "Cannot accept a master that has not been QC'd",
            candidate_id=candidate.id,
            hint="Run `app master qc` first.",
        )
    qc_passed = bool(qc.get("passed"))
    if require_qc_pass and not qc_passed:
        raise ConflictError(
            "Master QC did not pass; acceptance refused",
            candidate_id=candidate.id,
            failed_checks=qc.get("failed_check_ids", []),
            hint="Fix the composition or re-animate, or accept with --allow-qc-failure.",
        )

    acceptance = MasterAcceptance(
        accepted_by=accepted_by,
        accepted_at=utc_now(),
        reason=reason,
        qc_passed=qc_passed,
        acknowledged_warnings=acknowledged_warnings or [],
    )
    accepted = context.repos.masters.save(
        candidate.model_copy(
            update={"status": MasterCandidateStatus.ACCEPTED, "acceptance": acceptance}
        )
    )
    write_master_manifest(context, accepted)
    context.repos.audit.record(
        "master_accepted",
        actor=accepted_by,
        entity_type="master_candidate",
        entity_id=accepted.id,
        details={
            "reason": reason,
            "qc_passed": qc_passed,
            "frames": accepted.frame_count,
        },
    )
    log_event(logger, "master_accepted", candidate_id=accepted.id, accepted_by=accepted_by)
    return accepted


def reject_master(
    context: ServiceContext, candidate_id: str, *, rejected_by: str, reason: str
) -> MasterCandidate:
    """Record an explicit rejection, so a bad candidate is not silently reused."""
    candidate = context.repos.masters.get(candidate_id)
    if candidate.is_accepted:
        raise ConflictError("Cannot reject an accepted master", candidate_id=candidate.id)
    if len(reason.strip()) < 10:
        raise ValidationError("A rejection reason must be a real explanation", reason=reason)
    rejected = context.repos.masters.save(
        candidate.model_copy(
            update={
                "status": MasterCandidateStatus.REJECTED,
                "acceptance": MasterAcceptance(
                    accepted_by=rejected_by,
                    accepted_at=utc_now(),
                    reason=reason,
                    qc_passed=False,
                ),
            }
        )
    )
    context.repos.audit.record(
        "master_rejected",
        actor=rejected_by,
        entity_type="master_candidate",
        entity_id=rejected.id,
        details={"reason": reason},
    )
    return rejected


def _input_hashes(
    context: ServiceContext, composition: MotionComposition, hero: HeroCharacter
) -> dict[str, str]:
    from app.core.hashing import sha256_dir

    hashes: dict[str, str] = dict(composition.input_hashes)
    hashes[f"composition_poses::{composition.version_key()}"] = sha256_dir(
        context.absolute(composition.composed_pose_dir), patterns=("*.json",)
    )
    for relative, digest in hero.reference_hashes.items():
        hashes[f"hero_reference::{Path(relative).name}"] = digest
    return hashes


__all__ = [
    "HeroOptions",
    "MasterAnimateResult",
    "MasterCreateOptions",
    "accept_master",
    "animate_master",
    "build_master_manifest",
    "create_master_candidate",
    "plan_chunks",
    "register_hero",
    "reject_master",
    "resume_master",
    "write_master_manifest",
]
