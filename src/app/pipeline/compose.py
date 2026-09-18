"""Composition: cached intro + rendered reveal -> final video + manifest.

Order of operations, and why:

1. **Intro cache** — byte-copied from the immutable source frames once per
   template. Reused by every job, so intro pixels cannot drift between outfits.
2. **Assembly** — one contiguous directory keyed by absolute frame index, with
   the seam asserted exactly at the transition anchor.
3. **Audio** — copied (not re-encoded) from the master video, trimmed to the
   assembled frame range so audio and video durations agree.
4. **Encode** — a single explicit ffmpeg invocation whose argv is recorded
   verbatim in the manifest.
5. **Manifest** — input hashes, output hashes, per-frame checksums, seeds,
   config/workflow hashes, dependency versions and the ffmpeg commands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.errors import ValidationError
from app.core.hashing import sha256_file
from app.core.ids import utc_now
from app.core.logging import get_logger, log_event
from app.core.provenance import collect as collect_provenance
from app.domain.enums import JobStatus
from app.domain.manifest import (
    FrameChecksum,
    RenderManifest,
    ReproducibilityBlock,
)
from app.domain.render_job import RenderJob
from app.media import ffmpeg
from app.media.assembly import assemble_frames, ensure_intro_cache, verify_transition
from app.media.frames import DEFAULT_TEMPLATE, ffmpeg_pattern
from app.pipeline.context import ServiceContext
from app.pipeline.render import job_seeds
from app.pipeline.template_ingest import verify_source_immutability
from app.version import APP_VERSION, PREPROCESSING_VERSION

logger = get_logger(__name__)


@dataclass
class ComposeOptions:
    include_audio: bool = True
    flash_frames: int | None = None
    make_preview: bool = False
    output_name: str | None = None
    overwrite: bool = False


@dataclass
class ComposeResult:
    job: RenderJob
    manifest: RenderManifest
    final_video: Path
    preview_video: Path | None
    assembly: dict[str, Any]
    transition: dict[str, Any]
    commands: list[list[str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job.id,
            "final_video": str(self.final_video),
            "preview_video": str(self.preview_video) if self.preview_video else None,
            "manifest_path": self.job.artifacts.manifest_path,
            "manifest_digest": self.manifest.reproducibility_digest(),
            "assembly": self.assembly,
            "transition": self.transition,
            "total_frames": self.manifest.intro_frame_count + self.manifest.reveal_frame_count,
        }


def compose_job(
    context: ServiceContext,
    job_id: str,
    options: ComposeOptions | None = None,
) -> ComposeResult:
    """Join the cached intro with the rendered reveal and encode the result."""
    opts = options or ComposeOptions()
    config = context.config
    job = context.repos.jobs.get(job_id)
    template = context.repos.templates.get(job.template_id, job.template_version)
    garment = context.repos.garments.get(job.garment_id, job.garment_version)
    verify_source_immutability(context, template)

    missing = [
        index
        for index in job.frame_range.indices()
        if index not in set(context.repos.jobs.completed_frames(job.id))
    ]
    if missing:
        raise ValidationError(
            "Cannot compose a job with unrendered frames",
            job_id=job.id,
            missing_count=len(missing),
            first_missing=missing[0],
            hint="Run `app job resume` first.",
        )

    job = context.repos.jobs.save(job.model_copy(update={"status": JobStatus.COMPOSING}))
    commands: list[list[str]] = []

    # 1. intro cache (reused, never regenerated)
    intro_cache = ensure_intro_cache(
        context.absolute(template.directories.source_frames),
        context.absolute(template.directories.intro_cache),
        template.intro.start,
        template.intro.end,
    )

    # 2. assembly
    flash_frames = (
        config.transition.flash_frames if opts.flash_frames is None else opts.flash_frames
    )
    assembly_dir = context.absolute(job.artifacts.root) / "assembly_frames"
    report = assemble_frames(
        assembly_dir,
        intro_cache_dir=intro_cache.directory,
        intro_start=template.intro.start,
        transition_anchor=template.transition_anchor_frame,
        reveal_frames_dir=context.absolute(job.artifacts.composited_frames_dir),
        reveal_end=job.frame_range.end,
        flash_frames=flash_frames,
        flash_color=(
            config.transition.flash_color[0],
            config.transition.flash_color[1],
            config.transition.flash_color[2],
        ),
        flash_opacity=config.transition.flash_opacity,
    )
    transition = verify_transition(
        assembly_dir,
        intro_cache.directory,
        context.absolute(job.artifacts.composited_frames_dir),
        template.transition_anchor_frame,
        flash_frames=flash_frames,
    )
    if transition.get("last_intro_matches_cache") is False:
        raise ValidationError(
            "The assembled intro does not match the cached intro frames",
            job_id=job.id,
            transition=transition,
        )
    if transition.get("first_reveal_matches_render") is False:
        raise ValidationError(
            "The assembled first reveal frame does not match the rendered frame",
            job_id=job.id,
            transition=transition,
        )

    # 3. audio
    exports = context.data_root.export_dir()
    exports.mkdir(parents=True, exist_ok=True)
    audio_path: Path | None = None
    if opts.include_audio and template.video.has_audio:
        audio_path = context.absolute(job.artifacts.root) / "audio.m4a"
        start_s = template.intro.start / template.video.fps
        duration_s = report.total_frames / template.video.fps
        argv = ffmpeg.build_extract_audio_command(
            template.source_video_path,
            audio_path,
            ffmpeg=config.runtime.ffmpeg_binary,
            start_s=start_s,
            duration_s=duration_s,
        )
        result = ffmpeg.run_command(argv, check=False)
        commands.append(argv)
        if not result.ok:
            logger.warning(
                "audio_copy_failed",
                extra={"event": "audio_copy_failed", "stderr": result.stderr[-500:]},
            )
            audio_path = None

    # 4. encode
    name = opts.output_name or f"{job.id}.mp4"
    final_video = exports / name
    if final_video.exists() and not opts.overwrite:
        raise ValidationError(
            "Output file already exists", path=str(final_video), hint="Pass --overwrite."
        )
    encode_argv = ffmpeg.build_encode_from_frames_command(
        assembly_dir / ffmpeg_pattern(DEFAULT_TEMPLATE),
        final_video,
        ffmpeg=config.runtime.ffmpeg_binary,
        fps=template.video.fps,
        width=config.video.width,
        height=config.video.height,
        pixel_format=config.video.pixel_format,
        codec=config.video.video_codec,
        crf=config.video.crf,
        preset=config.video.preset,
        start_number=report.first_frame,
        audio_source=audio_path,
        audio_codec=config.video.audio_codec,
        audio_bitrate=config.video.audio_bitrate,
        faststart=config.video.faststart,
    )
    ffmpeg.run_command(encode_argv)
    commands.append(encode_argv)

    preview_video: Path | None = None
    if opts.make_preview:
        preview_video = exports / f"{Path(name).stem}_preview.mp4"
        preview_argv = ffmpeg.build_preview_command(
            final_video,
            preview_video,
            ffmpeg=config.runtime.ffmpeg_binary,
            scale=config.video.preview_scale,
            crf=config.video.preview_crf,
            width=config.video.width,
            height=config.video.height,
        )
        ffmpeg.run_command(preview_argv)
        commands.append(preview_argv)

    # 5. manifest
    manifest = build_manifest(
        context,
        job,
        template=template,
        garment=garment,
        assembly=report,
        intro_cache_digest=intro_cache.digest,
        final_video=final_video,
        preview_video=preview_video,
        flash_frames=flash_frames,
        commands=commands,
    )
    manifest_path = context.absolute(
        job.artifacts.manifest_path or f"{job.artifacts.root}/manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        __import__("json").dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    digest = context.repos.jobs.save_manifest(manifest)

    artifacts = job.artifacts.model_copy(
        update={
            "final_video_path": context.relative(final_video),
            "preview_video_path": context.relative(preview_video) if preview_video else None,
            "manifest_path": context.relative(manifest_path),
        }
    )
    job = context.repos.jobs.save(
        job.model_copy(
            update={
                "status": JobStatus.COMPOSED,
                "artifacts": artifacts,
                "finished_at": utc_now(),
            }
        )
    )
    context.repos.audit.record(
        "job_composed",
        entity_type="job",
        entity_id=job.id,
        details={"final_video": context.relative(final_video), "manifest_digest": digest},
    )
    log_event(
        logger,
        "job_composed",
        job_id=job.id,
        frames=report.total_frames,
        manifest_digest=digest,
    )
    return ComposeResult(
        job=job,
        manifest=manifest,
        final_video=final_video,
        preview_video=preview_video,
        assembly=report.as_dict(),
        transition=transition,
        commands=commands,
    )


def build_manifest(
    context: ServiceContext,
    job: RenderJob,
    *,
    template: Any,
    garment: Any,
    assembly: Any,
    intro_cache_digest: str,
    final_video: Path,
    preview_video: Path | None,
    flash_frames: int,
    commands: list[list[str]],
) -> RenderManifest:
    """Assemble the full reproducibility manifest for a composed job."""
    provenance = collect_provenance(context.config)
    report = (
        context.repos.compatibility.try_get(job.compatibility_report_id)
        if job.compatibility_report_id
        else None
    )

    flash_set = set(assembly.flash_frames)
    checksums: list[FrameChecksum] = []
    for index, digest in sorted(assembly.frame_hashes.items()):
        if index in flash_set:
            source = "flash"
        elif index < job.transition_anchor_frame:
            source = "intro_cache"
        else:
            source = "rendered"
        checksums.append(FrameChecksum(index=index, sha256=digest, source=source))

    output_hashes = {"final_video": sha256_file(final_video)}
    if preview_video is not None and preview_video.is_file():
        output_hashes["preview_video"] = sha256_file(preview_video)

    return RenderManifest(
        job_id=job.id,
        created_at=utc_now(),
        template_id=template.id,
        template_version=template.version,
        template_source_sha256=template.source_sha256,
        garment_id=garment.id,
        garment_version=garment.version,
        compatibility_report_id=report.id if report else None,
        compatibility_state=report.state if report else None,
        compatibility_overridden=bool(report and report.override),
        frame_range=job.frame_range,
        transition_anchor_frame=job.transition_anchor_frame,
        intro_frame_count=len(assembly.intro_frames),
        reveal_frame_count=len(assembly.reveal_frames),
        flash_frames=len(assembly.flash_frames),
        video_width=context.config.video.width,
        video_height=context.config.video.height,
        video_fps=template.video.fps,
        pixel_format=context.config.video.pixel_format,
        video_codec=context.config.video.video_codec,
        prompt=job.prompt,
        negative_prompt=job.negative_prompt,
        settings=job.settings,
        input_hashes={**job.input_hashes, "intro_cache_dir": intro_cache_digest},
        output_hashes=output_hashes,
        frame_checksums=checksums,
        reproducibility=ReproducibilityBlock(
            app_version=APP_VERSION,
            preprocessing_version=PREPROCESSING_VERSION,
            git=provenance["git"],
            platform=provenance["platform"],
            dependencies=provenance["dependencies"],
            ffmpeg_version=provenance["ffmpeg"],
            ffprobe_version=provenance["ffprobe"],
            config_hash=job.config_hash or provenance["config_hash"],
            rules_file_sha256=report.rules_file_sha256 if report else None,
            workflow_id=job.workflow_id,
            workflow_sha256=job.workflow_sha256,
            backend_name=job.backend_name,
            backend_version=job.backend_version,
            seed=job.seed,
            seed_strategy=job.seed_strategy,
            per_frame_seeds=job_seeds(job),
            ffmpeg_commands=commands,
        ),
    )


__all__ = ["ComposeOptions", "ComposeResult", "build_manifest", "compose_job"]
