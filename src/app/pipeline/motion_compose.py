"""Motion composition: normalize, match anchors, bridge, assemble.

Produces one contiguous motion-control sequence from two or more motion
references. Output is pose JSON plus a skeleton preview — never source pixels.

Output frame layout, stated once so the arithmetic is checkable::

    segment[0]  ->  [effective.start, prev_anchor)
    join[0]     ->  B bridge frames, bridge[0] == pose(prev_anchor)
                                     bridge[B-1] == pose(next_anchor)
    segment[1]  ->  [next_anchor + 1, effective.end)

The anchor frames belong to the bridge and to nothing else, so every output
frame has exactly one origin: no duplication, no gap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.errors import ConflictError, ValidationError
from app.core.hashing import sha256_dir, sha256_json
from app.core.ids import new_id, utc_now
from app.core.logging import get_logger, log_event
from app.core.paths import safe_identifier
from app.core.provenance import collect as collect_provenance
from app.domain.human_template import FrameRange
from app.domain.motion import (
    AnchorDescriptor,
    CanonicalSkeletonProfile,
    CanonicalTransform,
    MotionComposition,
    MotionCompositionStatus,
    MotionJoin,
    MotionSegment,
    SegmentQualityFlags,
    TransitionType,
    segment_dirname,
)
from app.motion.anchors import (
    AnchorCandidate,
    AnchorSearchSettings,
    rank_anchor_candidates,
    select_anchor,
)
from app.motion.bridge import BridgeSettings, generate_bridge
from app.motion.normalize import NormalizationSettings, normalize_sequence
from app.motion.pose_format import (
    PoseFrame,
    load_pose_sequence,
    save_pose_sequence,
)
from app.pipeline.context import ServiceContext

logger = get_logger(__name__)


def composition_id() -> str:
    return new_id("cmp")


@dataclass
class SegmentSpec:
    """Operator's request for one segment of the composition."""

    motion_source_id: str
    motion_source_version: int | None = None
    start_frame: int | None = None
    end_frame: int | None = None
    playback_speed: float = 1.0
    trim_start: int = 0
    trim_end: int = 0
    exposed_views: list[str] = field(default_factory=list)


@dataclass
class JoinSpec:
    """Operator's override for one join. Omit to search automatically."""

    prev_frame: int | None = None
    next_frame: int | None = None
    bridge_frames: int | None = None
    transition_type: TransitionType = TransitionType.POSE_BRIDGE


@dataclass
class ComposeOptions:
    display_name: str
    segments: list[SegmentSpec]
    joins: list[JoinSpec] = field(default_factory=list)
    output_fps: float | None = None
    composition_id: str | None = None
    version: int = 1
    profile_id: str = "canonical_v1"
    profile_version: int | None = None
    make_preview: bool = True


@dataclass
class ComposeResult:
    composition: MotionComposition
    poses: list[PoseFrame]
    anchor_frames: set[int]
    normalization: dict[str, Any] = field(default_factory=dict)
    preview: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "composition_id": self.composition.id,
            "version": self.composition.version,
            "status": self.composition.status.value,
            "segments": len(self.composition.segments),
            "joins": len(self.composition.joins),
            "output_frame_count": self.composition.output_frame_count,
            "output_fps": self.composition.output_fps,
            "anchor_frames": sorted(self.anchor_frames),
            "preview_path": self.composition.preview_path,
            "manifest_path": self.composition.manifest_path,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------
def load_profile(context: ServiceContext) -> CanonicalSkeletonProfile:
    """Load the canonical profile from config, caching it in the database.

    ``motion.skeleton_profile_file`` may be a bare filename resolved against the
    config directory, or an absolute path — so an operator can keep a profile
    outside the repository, and tests can use a small canonical frame without
    rewriting the shipped one.
    """
    import yaml

    configured = Path(context.config.motion.skeleton_profile_file)
    path = configured if configured.is_absolute() else context.config.config_dir() / configured
    if not path.is_file():
        raise ValidationError("Canonical skeleton profile not found", path=str(path))
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    profile = CanonicalSkeletonProfile.model_validate(payload)
    context.repos.skeleton_profiles.save(profile)
    return profile


def normalization_settings(profile: CanonicalSkeletonProfile) -> NormalizationSettings:
    return NormalizationSettings(
        target_width=profile.target_width,
        target_height=profile.target_height,
        target_center=profile.target_body_center,
        canonical_shoulder_width=profile.canonical_shoulder_width,
        canonical_torso_length=profile.canonical_torso_length,
        confidence_threshold=profile.confidence_threshold,
        shoulder_weight=profile.shoulder_weight,
        smoothing=profile.smoothing,
        max_interpolation_gap=profile.max_interpolation_gap,
        max_missing_joint_run=profile.max_missing_joint_run,
        max_scale_step=profile.max_scale_step,
    )


# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------
@dataclass
class NormalizedSegment:
    """One segment's normalized poses, keyed by their own source frame index."""

    spec: SegmentSpec
    index: int
    source_id: str
    source_version: int
    effective_range: FrameRange
    poses: list[PoseFrame]
    transform: CanonicalTransform
    warnings: list[str] = field(default_factory=list)

    @property
    def pose_dirname(self) -> str:
        """Unique per position, so one source used twice cannot collide."""
        return segment_dirname(self.index, self.source_id)

    def position_of(self, source_frame: int) -> int:
        for position, pose in enumerate(self.poses):
            if pose.frame_index == source_frame:
                return position
        raise ValidationError(
            "Frame is not part of this segment",
            source_frame=source_frame,
            range=[self.effective_range.start, self.effective_range.end],
        )


def normalize_segment(
    context: ServiceContext,
    spec: SegmentSpec,
    profile: CanonicalSkeletonProfile,
    *,
    output_fps: float,
    index: int = 0,
) -> NormalizedSegment:
    """Normalize one segment independently of every other segment."""
    record = context.repos.motion_sources.get(spec.motion_source_id, spec.motion_source_version)
    if not record.usage_rights.motion_use_authorized:
        raise ConflictError(
            "Motion use is not authorized for this source",
            motion_source_id=record.id,
            hint="Record the authorization with `app motion ingest --motion-use-authorized`.",
        )

    start = spec.start_frame if spec.start_frame is not None else record.selected_range.start
    end = spec.end_frame if spec.end_frame is not None else record.selected_range.end
    if start < record.selected_range.start or end > record.selected_range.end:
        raise ValidationError(
            "Segment range lies outside the motion source's selected range",
            requested=[start, end],
            available=[record.selected_range.start, record.selected_range.end],
        )
    requested = FrameRange(start=start, end=end)
    effective = FrameRange(
        start=requested.start + spec.trim_start, end=requested.end - spec.trim_end
    )

    pose_dir = context.absolute(record.pose_dir)
    poses = load_pose_sequence(pose_dir, effective.indices())

    result = normalize_sequence(
        poses,
        normalization_settings(profile),
        playback_speed=spec.playback_speed,
        output_fps=output_fps,
        # Keep each segment keyed by its own source frame numbering until the
        # composition assigns output indices. Mixing the two numbering schemes
        # early is exactly how off-by-one bugs get in.
        start_output_index=effective.start,
    )

    transform = CanonicalTransform(
        base_scale=result.base_scale,
        source_shoulder_width=result.source_shoulder_width,
        source_torso_length=result.source_torso_length,
        mean_offset_x=(
            sum(t.offset_x for t in result.transforms) / len(result.transforms)
            if result.transforms
            else 0.0
        ),
        mean_offset_y=(
            sum(t.offset_y for t in result.transforms) / len(result.transforms)
            if result.transforms
            else 0.0
        ),
        smoothing=profile.smoothing,
        interpolated_frames=result.interpolated_frames,
        stats=result.stats,
    )
    return NormalizedSegment(
        spec=spec,
        index=index,
        source_id=record.id,
        source_version=record.version,
        effective_range=effective,
        poses=result.poses,
        transform=transform,
        warnings=result.warnings,
    )


# ---------------------------------------------------------------------------
# anchors
# ---------------------------------------------------------------------------
def anchor_settings(
    context: ServiceContext, profile: CanonicalSkeletonProfile
) -> AnchorSearchSettings:
    motion = context.config.motion
    return AnchorSearchSettings(
        prev_window=motion.anchor_prev_window,
        next_window=motion.anchor_next_window,
        confidence_threshold=profile.confidence_threshold,
        max_acceptable_score=motion.anchor_max_acceptable_score,
        max_candidates=motion.anchor_max_candidates,
    )


def match_anchors(
    context: ServiceContext,
    prev_segment: NormalizedSegment,
    next_segment: NormalizedSegment,
    profile: CanonicalSkeletonProfile,
) -> list[AnchorCandidate]:
    """Rank the join candidates between two normalized segments."""
    return rank_anchor_candidates(
        prev_segment.poses, next_segment.poses, anchor_settings(context, profile)
    )


def _describe_anchor(pose: PoseFrame, threshold: float) -> AnchorDescriptor:
    root = pose.torso_center(threshold) or (0.0, 0.0)
    confidences = [j.confidence for j in pose.body.values()]
    return AnchorDescriptor(
        frame_index=pose.frame_index,
        root_x=root[0],
        root_y=root[1],
        shoulder_width=pose.shoulder_width(threshold) or 0.0,
        torso_angle_deg=pose.torso_angle_deg(threshold),
        head_angle_deg=pose.head_angle_deg(threshold),
        mean_confidence=(sum(confidences) / len(confidences)) if confidences else 0.0,
    )


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------
def compose_motion(
    context: ServiceContext,
    options: ComposeOptions,
) -> ComposeResult:
    """Normalize, join, bridge and assemble one motion-control sequence."""
    if len(options.segments) < 1:
        raise ValidationError("A composition needs at least one segment")
    if options.joins and len(options.joins) != len(options.segments) - 1:
        raise ValidationError(
            "Supply one join spec per gap between segments",
            segments=len(options.segments),
            joins=len(options.joins),
        )

    profile = load_profile(context)
    identifier = safe_identifier(options.composition_id or composition_id())
    root = context.data_root.resolve("compositions", identifier)
    normalized_dir = root / "normalized_poses"
    bridge_dir = root / "bridge_poses"
    composed_dir = root / "composed_poses"
    for directory in (normalized_dir, bridge_dir, composed_dir):
        directory.mkdir(parents=True, exist_ok=True)

    first_source = context.repos.motion_sources.get(
        options.segments[0].motion_source_id, options.segments[0].motion_source_version
    )
    output_fps = options.output_fps or first_source.video.fps

    warnings: list[str] = []
    normalized: list[NormalizedSegment] = []
    for index, spec in enumerate(options.segments):
        segment = normalize_segment(context, spec, profile, output_fps=output_fps, index=index)
        normalized.append(segment)
        warnings.extend(f"segment {index} ({spec.motion_source_id}): {w}" for w in segment.warnings)
        # Keyed by segment position: the same source may appear twice.
        save_pose_sequence(normalized_dir / segment.pose_dirname, segment.poses)

    # -- choose anchors ---------------------------------------------------
    joins: list[MotionJoin] = []
    chosen: list[AnchorCandidate] = []
    for index in range(len(normalized) - 1):
        # Named join_spec, not spec: `spec` is the segment loop variable above,
        # and shadowing it here silently mixed the two types.
        join_spec = options.joins[index] if index < len(options.joins) else JoinSpec()
        prev_segment, next_segment = normalized[index], normalized[index + 1]
        candidates = match_anchors(context, prev_segment, next_segment, profile)
        candidate = select_anchor(
            prev_segment.poses,
            next_segment.poses,
            anchor_settings(context, profile),
            override_prev_frame=join_spec.prev_frame,
            override_next_frame=join_spec.next_frame,
        )
        chosen.append(candidate)
        if not candidate.acceptable:
            warnings.append(
                f"join {index}: anchor score {candidate.score:.3f} exceeds the "
                f"acceptable threshold; review the preview before animating"
            )
        joins.append(
            _build_join(
                index=index,
                candidate=candidate,
                candidates=candidates,
                spec=join_spec,
                context=context,
            )
        )

    # -- assemble ---------------------------------------------------------
    composed, anchor_frames, bridge_poses = assemble_sequence(
        normalized, joins, chosen, context=context, output_fps=output_fps
    )
    save_pose_sequence(composed_dir, composed)
    if bridge_poses:
        save_pose_sequence(bridge_dir, bridge_poses)

    segments = [
        MotionSegment(
            motion_source_id=segment.source_id,
            motion_source_version=segment.source_version,
            segment_index=segment.index,
            source_range=FrameRange(
                start=segment.effective_range.start - segment.spec.trim_start,
                end=segment.effective_range.end + segment.spec.trim_end,
            ),
            playback_speed=segment.spec.playback_speed,
            trim_start=segment.spec.trim_start,
            trim_end=segment.spec.trim_end,
            canonical_transform=segment.transform,
            quality=SegmentQualityFlags(
                exposed_views=segment.spec.exposed_views,  # type: ignore[arg-type]
                hands_reliable=_joints_reliable(
                    segment.poses, ("left_wrist", "right_wrist"), profile
                ),
                face_reliable=_joints_reliable(segment.poses, ("nose",), profile),
                torso_reliable=_joints_reliable(
                    segment.poses, ("left_shoulder", "right_shoulder"), profile
                ),
            ),
            start_anchor=_describe_anchor(segment.poses[0], profile.confidence_threshold),
            end_anchor=_describe_anchor(segment.poses[-1], profile.confidence_threshold),
        )
        for segment in normalized
    ]

    composition = MotionComposition(
        id=identifier,
        version=options.version,
        display_name=options.display_name,
        segments=segments,
        joins=joins,
        output_fps=output_fps,
        output_frame_count=len(composed),
        skeleton_profile_id=profile.id,
        skeleton_profile_version=profile.version,
        normalized_pose_dir=context.relative(normalized_dir),
        bridge_pose_dir=context.relative(bridge_dir),
        composed_pose_dir=context.relative(composed_dir),
        input_hashes=_input_hashes(context, normalized),
        settings={
            "anchor": {
                "prev_window": context.config.motion.anchor_prev_window,
                "next_window": context.config.motion.anchor_next_window,
                "max_acceptable_score": context.config.motion.anchor_max_acceptable_score,
            },
            "bridge": {
                "frames": context.config.motion.bridge_frames,
                "easing": context.config.motion.bridge_easing,
                "tangent_strength": context.config.motion.bridge_tangent_strength,
            },
            "profile": profile.model_dump(mode="json"),
            "config_hash": context.config.config_hash(),
        },
        status=MotionCompositionStatus.BRIDGED,
    )

    # The declared frame count and the emitted sequence must agree exactly.
    expected = composition.expected_frame_count()
    if expected != len(composed):
        raise ValidationError(
            "Composed frame count disagrees with the layout arithmetic",
            expected=expected,
            emitted=len(composed),
            hint="This is an off-by-one in the segment/bridge layout.",
        )

    preview: dict[str, Any] | None = None
    if options.make_preview:
        preview = _render_preview(context, composition, composed, anchor_frames, root)
        composition = composition.model_copy(
            update={
                "preview_path": context.relative(preview["path"]),
                "status": MotionCompositionStatus.PREVIEWED,
            }
        )

    manifest_path = root / "manifest.json"
    manifest = _build_manifest(context, composition, normalized, joins, composed, preview)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    composition = composition.model_copy(
        update={
            "manifest_path": context.relative(manifest_path),
            "status": MotionCompositionStatus.READY,
        }
    )

    saved = context.repos.compositions.save(composition)
    context.repos.audit.record(
        "motion_composed",
        entity_type="motion_composition",
        entity_id=saved.id,
        details={
            "segments": len(segments),
            "joins": len(joins),
            "frames": saved.output_frame_count,
            "sources": [s.source_key() for s in segments],
        },
    )
    log_event(
        logger,
        "motion_composed",
        composition_id=saved.id,
        frames=saved.output_frame_count,
        segments=len(segments),
    )
    return ComposeResult(
        composition=saved,
        poses=composed,
        anchor_frames=anchor_frames,
        # Keyed by the segment's directory, not its source id: one composition
        # may legitimately use the same clip twice, and keying on the source
        # alone dropped the first segment's transform on the floor.
        normalization={
            segment.pose_dirname: segment.transform.model_dump(mode="json")
            for segment in normalized
        },
        preview=preview,
        warnings=warnings,
    )


def _joints_reliable(
    poses: list[PoseFrame], names: tuple[str, ...], profile: CanonicalSkeletonProfile
) -> bool:
    if not poses:
        return False
    threshold = profile.high_confidence_threshold
    good = sum(1 for pose in poses if pose.has_all(names, threshold))
    return good / len(poses) >= 0.9


def _build_join(
    *,
    index: int,
    candidate: AnchorCandidate,
    candidates: list[AnchorCandidate],
    spec: JoinSpec,
    context: ServiceContext,
) -> MotionJoin:
    bridge_frames = (
        0
        if spec.transition_type is TransitionType.DIRECT_CUT
        else (spec.bridge_frames or context.config.motion.bridge_frames)
    )
    return MotionJoin(
        prev_segment_index=index,
        next_segment_index=index + 1,
        prev_source_frame=candidate.prev_frame,
        next_source_frame=candidate.next_frame,
        pose_distance_score=candidate.score,
        root_position_delta=candidate.root_delta,
        shoulder_scale_delta=candidate.shoulder_scale_delta,
        torso_angle_delta=candidate.torso_angle_delta,
        head_angle_delta=candidate.head_angle_delta,
        hand_position_delta=candidate.hand_distance if candidate.hand_distance >= 0 else 0.0,
        incoming_velocity_delta=candidate.velocity_delta,
        outgoing_velocity_delta=candidate.velocity_delta,
        bridge_frame_count=bridge_frames,
        transition_type=spec.transition_type,
        operator_override=spec.prev_frame is not None or spec.next_frame is not None,
        accepted=candidate.acceptable,
        warnings=list(candidate.warnings),
        candidates=[c.as_dict() for c in candidates],
    )


def assemble_sequence(
    normalized: list[NormalizedSegment],
    joins: list[MotionJoin],
    chosen: list[AnchorCandidate],
    *,
    context: ServiceContext,
    output_fps: float,
) -> tuple[list[PoseFrame], set[int], list[PoseFrame]]:
    """Lay segments and bridges out onto contiguous output frame indices."""
    composed: list[PoseFrame] = []
    bridges: list[PoseFrame] = []
    anchor_frames: set[int] = set()
    output_index = 0

    for index, segment in enumerate(normalized):
        start = segment.effective_range.start
        end = segment.effective_range.end
        if index > 0:
            # The previous bridge already carried its next-anchor frame.
            start = joins[index - 1].next_source_frame + 1
        if index < len(joins):
            # This segment stops before its anchor; the bridge carries it.
            end = joins[index].prev_source_frame

        if end < start:
            raise ValidationError(
                "Anchor selection leaves a segment with no frames",
                segment_index=index,
                start=start,
                end=end,
                hint="Choose anchors further apart, or widen the segment range.",
            )

        for source_frame in range(start, end):
            pose = segment.poses[segment.position_of(source_frame)]
            composed.append(pose.with_index(output_index, output_index / output_fps))
            output_index += 1

        if index < len(joins):
            join = joins[index]
            next_segment = normalized[index + 1]
            if join.transition_type is TransitionType.DIRECT_CUT:
                # No bridge frames, but the seam is still a real output frame:
                # the next segment's first contributed frame.
                join.recommended_transition_anchor = output_index
                continue

            settings = BridgeSettings(
                frame_count=join.bridge_frame_count,
                easing=context.config.motion.bridge_easing,
                tangent_strength=context.config.motion.bridge_tangent_strength,
                max_limb_length_drift=context.config.motion_qc.max_limb_length_discontinuity,
                max_scale_drift=context.config.motion_qc.max_shoulder_scale_drift,
                max_velocity_step_px=context.config.motion_qc.max_velocity_discontinuity_px,
                enforce_prototype_range=context.config.motion.enforce_prototype_bridge_range,
            )
            result = generate_bridge(
                segment.poses,
                segment.position_of(join.prev_source_frame),
                next_segment.poses,
                next_segment.position_of(join.next_source_frame),
                settings,
                start_output_index=output_index,
                output_fps=output_fps,
            )
            anchor_frames.add(output_index)
            anchor_frames.add(output_index + len(result.poses) - 1)

            # Output-space geometry: the source anchors above cannot locate
            # anything in the composed sequence, so record where this join
            # actually landed. The recommended garment seam is the bridge start.
            join.output_bridge_start = output_index
            join.output_bridge_end = output_index + len(result.poses)
            join.recommended_transition_anchor = output_index

            composed.extend(result.poses)
            bridges.extend(result.poses)
            output_index += len(result.poses)

            join.bridge_settings = result.settings
            join.bridge_metrics = result.metrics
            join.warnings.extend(result.warnings)

    # Pydantic does not re-validate on assignment, and the joins above were
    # mutated in place. Re-validate them explicitly so the output-space
    # coherence rule (span == bridge_frame_count) is actually enforced rather
    # than merely declared.
    joins[:] = [MotionJoin.model_validate(join.model_dump()) for join in joins]

    return composed, anchor_frames, bridges


def _render_preview(
    context: ServiceContext,
    composition: MotionComposition,
    poses: list[PoseFrame],
    anchor_frames: set[int],
    root: Path,
) -> dict[str, Any]:
    from app.motion.preview import PreviewSettings, render_skeleton_preview

    profile_settings = composition.settings.get("profile", {})
    settings = PreviewSettings(
        width=int(profile_settings.get("target_width", 1080)),
        height=int(profile_settings.get("target_height", 1920)),
        fps=composition.output_fps,
        scale=context.config.motion.preview_scale,
    )
    return render_skeleton_preview(
        poses,
        root / "preview.mp4",
        settings,
        anchor_frames=anchor_frames,
        ffmpeg_binary=context.config.runtime.ffmpeg_binary,
    )


def _input_hashes(context: ServiceContext, normalized: list[NormalizedSegment]) -> dict[str, str]:
    """Hash every input a composition consumed, for the manifest."""
    hashes: dict[str, str] = {}
    for segment in normalized:
        record = context.repos.motion_sources.get(segment.source_id, segment.source_version)
        key = record.version_key()
        hashes[f"motion_source_video::{key}"] = record.source_sha256
        pose_dir = context.absolute(record.pose_dir)
        hashes[f"motion_source_pose::{key}"] = record.pose_sha256 or sha256_dir(
            pose_dir, patterns=("*.json",)
        )
    return hashes


def _build_manifest(
    context: ServiceContext,
    composition: MotionComposition,
    normalized: list[NormalizedSegment],
    joins: list[MotionJoin],
    poses: list[PoseFrame],
    preview: dict[str, Any] | None,
) -> dict[str, Any]:
    """The composition's reproducibility record."""
    provenance = collect_provenance(context.config)
    payload: dict[str, Any] = {
        "schema_version": "1",
        "composition_id": composition.id,
        "composition_version": composition.version,
        "created_at": utc_now().isoformat(),
        "output_fps": composition.output_fps,
        "output_frame_count": len(poses),
        "skeleton_profile": composition.settings.get("profile", {}),
        "segments": [
            {
                "segment_index": segment.index,
                "motion_source": f"{segment.source_id}@v{segment.source_version}",
                "pose_dir": segment.pose_dirname,
                "effective_range": [
                    segment.effective_range.start,
                    segment.effective_range.end,
                ],
                "playback_speed": segment.spec.playback_speed,
                "canonical_transform": segment.transform.model_dump(mode="json"),
            }
            for segment in normalized
        ],
        "joins": [join.model_dump(mode="json") for join in joins],
        "bridge_settings": [join.bridge_settings for join in joins],
        "input_hashes": composition.input_hashes,
        "settings": composition.settings,
        "preview": preview,
        "reproducibility": {
            "app_version": provenance["app"]["version"],
            "git": provenance["git"],
            "platform": provenance["platform"],
            "dependencies": provenance["dependencies"],
            "config_hash": provenance["config_hash"],
            "ffmpeg": provenance["ffmpeg"],
        },
        "contains_source_pixels": False,
    }
    # The digest covers what determines the output, and deliberately excludes
    # wall-clock timestamps. The profile record carries created_at/updated_at
    # from its database row; including those would make two identical
    # compositions hash differently and turn the digest into a label.
    profile_geometry = {
        key: value
        for key, value in payload["skeleton_profile"].items()
        if key not in {"created_at", "updated_at"}
    }
    payload["digest"] = sha256_json(
        {
            "segments": payload["segments"],
            "joins": [
                {
                    "prev": join.prev_source_frame,
                    "next": join.next_source_frame,
                    "bridge": join.bridge_frame_count,
                    "output_bridge": join.output_bridge_range,
                    "recommended_anchor": join.recommended_transition_anchor,
                    "settings": join.bridge_settings,
                }
                for join in joins
            ],
            "input_hashes": payload["input_hashes"],
            "frame_count": payload["output_frame_count"],
            "fps": payload["output_fps"],
            "profile": profile_geometry,
        }
    )
    return payload


__all__ = [
    "ComposeOptions",
    "ComposeResult",
    "JoinSpec",
    "NormalizedSegment",
    "SegmentSpec",
    "assemble_sequence",
    "compose_motion",
    "load_profile",
    "match_anchors",
    "normalization_settings",
    "normalize_segment",
]
