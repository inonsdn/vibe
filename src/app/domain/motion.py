"""Motion Composition domain schemas.

The vocabulary this phase introduces, in the order artifacts are produced:

``MotionSource``
    A reference video whose **motion** is being borrowed. Its pixels are never
    used; only pose data is extracted or imported from it.
``CanonicalSkeletonProfile``
    The one body coordinate system every motion is normalized into. Two clips
    of two differently-sized people at different distances become comparable
    only because they are both mapped through this.
``MotionSegment``
    A trimmed, normalized stretch of one motion source, with its canonical
    transform recorded.
``MotionJoin``
    The measured decision to cut from one segment to the next, with every
    difference that fed the score.
``MotionComposition``
    The ordered segments, joins and bridges that together form one
    motion-control sequence.

Nothing here references imagery. That separation is the product requirement:
the original people, faces, clothes and backgrounds must never appear in a
Master Human Performance.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

from pydantic import Field, model_validator

from app.domain.base import DomainModel, Identifier, Sha256, TimestampedModel
from app.domain.enums import ImageViewType
from app.domain.human_template import FrameRange, VideoSpec


class MotionSourceStatus(StrEnum):
    """Lifecycle of a motion reference."""

    CREATED = "created"
    PROBING = "probing"
    AWAITING_POSE = "awaiting_pose"
    POSE_IMPORTED = "pose_imported"
    POSE_EXTRACTED = "pose_extracted"
    VALIDATED = "validated"
    READY = "ready"
    REJECTED = "rejected"
    FAILED = "failed"


class MotionCompositionStatus(StrEnum):
    CREATED = "created"
    NORMALIZED = "normalized"
    ANCHORS_MATCHED = "anchors_matched"
    BRIDGED = "bridged"
    PREVIEWED = "previewed"
    READY = "ready"
    FAILED = "failed"


class TransitionType(StrEnum):
    """How two segments are joined."""

    #: A generated pose bridge. The only type that produces new poses.
    POSE_BRIDGE = "pose_bridge"
    #: A direct cut, valid only when the anchors already match closely.
    DIRECT_CUT = "direct_cut"


class MotionUsageRights(DomainModel):
    """Provenance and rights for a motion reference.

    Motion is being borrowed from a real recording, so the paperwork is part of
    the record. ``motion_use_authorized`` is a separate, explicit assertion:
    holding a clip is not the same as being allowed to derive motion from it.
    """

    source_description: str | None = None
    rights_holder: str | None = None
    license: str | None = None
    acquired_from: str | None = None
    acquired_at: str | None = None
    motion_use_authorized: bool = False
    depicted_person_consent_ref: str | None = None
    restrictions: list[str] = Field(default_factory=list)
    notes: str | None = None

    @property
    def is_documented(self) -> bool:
        return bool(self.rights_holder or self.license or self.acquired_from)


class MotionQualityMetrics(DomainModel):
    """Measured usability of a motion source's pose data."""

    frames_with_pose: int = Field(default=0, ge=0)
    mean_joint_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    min_joint_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    longest_missing_run: int = Field(default=0, ge=0)
    missing_joint_runs: dict[str, int] = Field(default_factory=dict)
    median_shoulder_width_px: float | None = None
    median_torso_length_px: float | None = None
    shoulder_width_variation: float | None = Field(default=None, ge=0.0)
    #: Fraction of frames whose high-priority joints are all confident.
    in_frame_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    #: The confidence threshold these numbers were measured at. Recorded so a
    #: stored metric can never be compared against a different threshold.
    confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)

    @property
    def is_usable(self) -> bool:
        return self.frames_with_pose > 0 and self.mean_joint_confidence > 0.0


class BoundingBoxStats(DomainModel):
    """Where the source person sits in their own frame, and how big they are."""

    frames_with_bbox: int = Field(default=0, ge=0)
    width: dict[str, float] = Field(default_factory=dict)
    height: dict[str, float] = Field(default_factory=dict)
    center_x: dict[str, float] = Field(default_factory=dict)
    center_y: dict[str, float] = Field(default_factory=dict)


class MotionSource(TimestampedModel):
    """A reference video contributing motion — and nothing else."""

    id: Identifier
    version: int = Field(default=1, ge=1)
    display_name: str = Field(min_length=1, max_length=200)

    source_video_path: str
    source_sha256: Sha256
    video: VideoSpec

    #: The stretch of the source actually used.
    selected_range: FrameRange

    pose_format: str = Field(default="coco_17", max_length=32)
    pose_schema_version: str = Field(default="1", max_length=16)
    pose_dir: str
    pose_sha256: str | None = None
    pose_origin: str = Field(default="imported", pattern=r"^(imported|extracted)$")
    pose_adapter: str | None = None
    #: What the extraction actually did: execution provider, model hashes, ROI,
    #: subject-tracking summary and temporal cleanup. Recorded because a run
    #: that silently fell back to CPU, or tracked the wrong person, looks
    #: identical in the pose files themselves.
    pose_extraction: dict[str, Any] = Field(default_factory=dict)

    bbox_stats: BoundingBoxStats = Field(default_factory=BoundingBoxStats)
    quality: MotionQualityMetrics = Field(default_factory=MotionQualityMetrics)
    usage_rights: MotionUsageRights = Field(default_factory=MotionUsageRights)

    status: MotionSourceStatus = MotionSourceStatus.CREATED
    notes: str | None = None

    @model_validator(mode="after")
    def _range_within_video(self) -> Self:
        if self.video.frame_count and self.selected_range.end > self.video.frame_count:
            raise ValueError(
                f"selected_range.end ({self.selected_range.end}) exceeds the source "
                f"frame count ({self.video.frame_count})"
            )
        return self

    def version_key(self) -> str:
        return f"{self.id}@v{self.version}"


class CanonicalSkeletonProfile(TimestampedModel):
    """The single body coordinate system all motion is normalized into."""

    id: Identifier = "canonical_v1"
    version: int = Field(default=1, ge=1)
    skeleton_format: str = Field(default="coco_17", max_length=32)

    #: Maps this profile's joint names onto the internal vocabulary. Empty means
    #: the source already speaks the canonical names.
    joint_mapping: dict[str, str] = Field(default_factory=dict)
    root_joint: str = "torso_center"

    canonical_shoulder_width: float = Field(gt=0)
    canonical_torso_length: float = Field(gt=0)

    target_width: int = Field(gt=0)
    target_height: int = Field(gt=0)
    target_body_center: tuple[float, float]
    target_head_position: tuple[float, float]
    target_head_scale: float = Field(gt=0)

    confidence_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    high_confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    smoothing: float = Field(default=0.8, ge=0.0, le=1.0)
    max_interpolation_gap: int = Field(default=5, ge=0)
    max_missing_joint_run: int = Field(default=8, ge=0)
    max_scale_step: float = Field(default=0.02, ge=0.0)
    shoulder_weight: float = Field(default=0.6, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _center_inside_frame(self) -> Self:
        x, y = self.target_body_center
        if not (0 <= x <= self.target_width and 0 <= y <= self.target_height):
            raise ValueError("target_body_center must lie inside the target frame")
        head_x, head_y = self.target_head_position
        if not (0 <= head_x <= self.target_width and 0 <= head_y <= self.target_height):
            raise ValueError("target_head_position must lie inside the target frame")
        if self.confidence_threshold > self.high_confidence_threshold:
            raise ValueError("confidence_threshold must not exceed high_confidence_threshold")
        return self

    def version_key(self) -> str:
        return f"{self.id}@v{self.version}"


class CanonicalTransform(DomainModel):
    """The scale/offset actually applied to one segment, for the manifest."""

    base_scale: float = Field(gt=0)
    source_shoulder_width: float = Field(ge=0)
    source_torso_length: float = Field(ge=0)
    mean_offset_x: float = 0.0
    mean_offset_y: float = 0.0
    smoothing: float = Field(default=0.0, ge=0.0, le=1.0)
    interpolated_frames: list[int] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)


class SegmentQualityFlags(DomainModel):
    """Per-segment usability flags that feed downstream decisions."""

    hands_reliable: bool = True
    face_reliable: bool = True
    torso_reliable: bool = True
    #: Views the motion exposes; drives the garment compatibility rules later.
    exposed_views: list[ImageViewType] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class AnchorDescriptor(DomainModel):
    """A segment's start or end anchor, in canonical space."""

    frame_index: int = Field(ge=0)
    root_x: float
    root_y: float
    shoulder_width: float = Field(ge=0)
    torso_angle_deg: float | None = None
    head_angle_deg: float | None = None
    mean_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


def segment_dirname(index: int, motion_source_id: str) -> str:
    """Per-segment directory name for normalized poses.

    Keyed by the segment's **position**, not just its source: one composition
    may legitimately use the same motion source twice (two different ranges of
    the same clip), and keying on the source id alone made the second segment
    overwrite the first.
    """
    return f"segment_{index:03d}_{motion_source_id}"


class MotionSegment(DomainModel):
    """One trimmed, normalized stretch of a motion source."""

    motion_source_id: Identifier
    motion_source_version: int = Field(ge=1)
    #: Position of this segment within its composition; also names its pose
    #: directory, so two segments from one source cannot collide.
    segment_index: int = Field(default=0, ge=0)

    source_range: FrameRange
    playback_speed: float = Field(default=1.0, gt=0.0, le=4.0)
    trim_start: int = Field(default=0, ge=0)
    trim_end: int = Field(default=0, ge=0)

    canonical_transform: CanonicalTransform
    quality: SegmentQualityFlags = Field(default_factory=SegmentQualityFlags)
    start_anchor: AnchorDescriptor | None = None
    end_anchor: AnchorDescriptor | None = None

    @model_validator(mode="after")
    def _trim_fits(self) -> Self:
        if self.trim_start + self.trim_end >= self.source_range.count:
            raise ValueError(
                f"trim_start + trim_end ({self.trim_start + self.trim_end}) removes the "
                f"whole segment ({self.source_range.count} frames)"
            )
        return self

    @property
    def effective_range(self) -> FrameRange:
        """The source range after trimming."""
        return FrameRange(
            start=self.source_range.start + self.trim_start,
            end=self.source_range.end - self.trim_end,
        )

    @property
    def frame_count(self) -> int:
        return self.effective_range.count

    def source_key(self) -> str:
        return f"{self.motion_source_id}@v{self.motion_source_version}"

    @property
    def pose_dirname(self) -> str:
        """Directory holding this segment's normalized poses."""
        return segment_dirname(self.segment_index, self.motion_source_id)


class MotionJoin(DomainModel):
    """The measured decision to cut from one segment to the next.

    Every difference that fed the score is recorded, not just the total: an
    operator reviewing a bad join needs to know *which* term was large.
    """

    prev_segment_index: int = Field(ge=0)
    next_segment_index: int = Field(ge=0)

    #: Frames chosen in each segment's own source numbering.
    prev_source_frame: int = Field(ge=0)
    next_source_frame: int = Field(ge=0)

    pose_distance_score: float = Field(ge=0.0)
    root_position_delta: float = Field(ge=0.0)
    shoulder_scale_delta: float = Field(ge=0.0)
    torso_angle_delta: float = Field(ge=0.0)
    head_angle_delta: float = Field(ge=0.0)
    hand_position_delta: float = Field(ge=0.0)
    incoming_velocity_delta: float = Field(ge=0.0)
    outgoing_velocity_delta: float = Field(ge=0.0)

    bridge_frame_count: int = Field(ge=0)
    transition_type: TransitionType = TransitionType.POSE_BRIDGE
    bridge_settings: dict[str, Any] = Field(default_factory=dict)
    bridge_metrics: dict[str, Any] = Field(default_factory=dict)

    # -- output-space geometry --------------------------------------------
    # The source-frame anchors above identify frames in each segment's OWN clip
    # numbering. After normalization and bridge insertion those numbers no
    # longer locate anything in the composed sequence, so they cannot be used to
    # choose a garment reveal seam. These fields record where the join actually
    # landed in output numbering, and are filled by the assembler.
    #: First bridge frame, in output numbering (inclusive).
    output_bridge_start: int | None = Field(default=None, ge=0)
    #: One past the last bridge frame, in output numbering (exclusive).
    output_bridge_end: int | None = Field(default=None, ge=0)
    #: Where a garment reveal should begin if this join is the seam. Defaults to
    #: the bridge start: everything before it is the borrowed opening motion,
    #: everything from it on is the new performance.
    recommended_transition_anchor: int | None = Field(default=None, ge=0)

    operator_override: bool = False
    accepted: bool = False
    warnings: list[str] = Field(default_factory=list)
    candidates: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def output_bridge_range(self) -> tuple[int, int] | None:
        """``(start, end)`` of the bridge in output numbering, when known."""
        if self.output_bridge_start is None or self.output_bridge_end is None:
            return None
        return (self.output_bridge_start, self.output_bridge_end)

    @model_validator(mode="after")
    def _output_range_is_coherent(self) -> Self:
        start, end = self.output_bridge_start, self.output_bridge_end
        if (start is None) != (end is None):
            raise ValueError("output_bridge_start and output_bridge_end must be set together")
        if start is not None and end is not None:
            if end <= start:
                raise ValueError(
                    f"output_bridge_end ({end}) must exceed output_bridge_start ({start})"
                )
            if end - start != self.bridge_frame_count:
                raise ValueError(
                    f"output bridge span ({end - start}) disagrees with "
                    f"bridge_frame_count ({self.bridge_frame_count})"
                )
        return self

    @model_validator(mode="after")
    def _segments_are_adjacent(self) -> Self:
        if self.next_segment_index != self.prev_segment_index + 1:
            raise ValueError(
                "A join must connect adjacent segments "
                f"({self.prev_segment_index} -> {self.next_segment_index})"
            )
        if self.transition_type is TransitionType.POSE_BRIDGE and self.bridge_frame_count < 2:
            raise ValueError("A pose bridge needs at least 2 frames to carry both endpoints")
        if self.transition_type is TransitionType.DIRECT_CUT and self.bridge_frame_count:
            raise ValueError("A direct cut must not declare bridge frames")
        return self


class MotionComposition(TimestampedModel):
    """An ordered set of segments and joins forming one motion-control sequence.

    Output frame layout — the arithmetic that must not be off by one::

        segment[0] contributes  [start, prev_anchor)
        join[0] bridge          B frames, endpoints == the two anchor poses
        segment[1] contributes  (next_anchor, end)   i.e. [next_anchor+1, end)

    so the anchor frames themselves are contributed by the bridge and by nothing
    else. ``output_frame_count`` is the sum of those contributions.
    """

    id: Identifier
    version: int = Field(default=1, ge=1)
    display_name: str = Field(min_length=1, max_length=200)

    segments: list[MotionSegment] = Field(min_length=1)
    joins: list[MotionJoin] = Field(default_factory=list)

    output_fps: float = Field(gt=0, le=240)
    output_frame_count: int = Field(default=0, ge=0)

    skeleton_profile_id: Identifier = "canonical_v1"
    skeleton_profile_version: int = Field(default=1, ge=1)

    normalized_pose_dir: str
    bridge_pose_dir: str
    composed_pose_dir: str
    preview_path: str | None = None
    manifest_path: str | None = None

    input_hashes: dict[str, str] = Field(default_factory=dict)
    settings: dict[str, Any] = Field(default_factory=dict)
    status: MotionCompositionStatus = MotionCompositionStatus.CREATED
    notes: str | None = None

    @model_validator(mode="after")
    def _joins_match_segments(self) -> Self:
        expected = len(self.segments) - 1
        if len(self.joins) != expected:
            raise ValueError(
                f"{len(self.segments)} segments need exactly {expected} join(s), "
                f"got {len(self.joins)}"
            )
        for position, join in enumerate(self.joins):
            if join.prev_segment_index != position or join.next_segment_index != position + 1:
                raise ValueError(
                    f"join {position} does not connect segments {position} and {position + 1}"
                )
        return self

    def expected_frame_count(self) -> int:
        """Frame count implied by the segments, anchors and bridges.

        Contribution per segment is bounded by the anchors on either side: the
        anchor frame itself belongs to the bridge, never to the segment.
        """
        total = 0
        for index, segment in enumerate(self.segments):
            effective = segment.effective_range
            start = effective.start
            end = effective.end
            if index > 0:
                # The previous join's next-anchor frame is carried by that bridge.
                start = self.joins[index - 1].next_source_frame + 1
            if index < len(self.joins):
                end = self.joins[index].prev_source_frame
            total += max(0, end - start)
        total += sum(join.bridge_frame_count for join in self.joins)
        return total

    def version_key(self) -> str:
        return f"{self.id}@v{self.version}"


__all__ = [
    "AnchorDescriptor",
    "BoundingBoxStats",
    "CanonicalSkeletonProfile",
    "CanonicalTransform",
    "MotionComposition",
    "MotionCompositionStatus",
    "MotionJoin",
    "MotionQualityMetrics",
    "MotionSegment",
    "MotionSource",
    "MotionSourceStatus",
    "MotionUsageRights",
    "SegmentQualityFlags",
    "TransitionType",
    "segment_dirname",
]
