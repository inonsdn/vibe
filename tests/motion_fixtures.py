"""Synthetic motion fixtures.

Two motion "sources" that deliberately differ the way the product requirement
describes: **different apparent body scale**, **different framing**, and a
different position in frame. Their motions are related, so a genuinely
compatible anchor pair exists and the matcher has something correct to find.

Everything is generated in code. No media is committed, and — the point of the
whole phase — no pixels are involved at all: these fixtures produce pose JSON.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.adapters.pose import MockPoseAdapter
from app.core.hashing import sha256_file
from app.domain.human_template import FrameRange, VideoSpec
from app.domain.motion import (
    MotionSource,
    MotionSourceStatus,
    MotionUsageRights,
)
from app.motion.pose_format import Joint2D, PoseFrame, save_pose_sequence
from app.pipeline.context import ServiceContext
from app.pipeline.motion_ingest import measure_pose_quality

FPS = 30.0

#: A small canonical profile for tests. Same shape and proportions as the
#: shipped 1080x1920 profile, scaled down so the suite renders quickly; the
#: pipeline is resolution-agnostic, so this exercises identical code paths.
TEST_PROFILE = {
    "id": "canonical_test",
    "version": 1,
    "skeleton_format": "coco_17",
    "joint_mapping": {},
    "root_joint": "torso_center",
    "canonical_shoulder_width": 75.0,
    "canonical_torso_length": 105.0,
    "target_width": 270,
    "target_height": 480,
    "target_body_center": [135.0, 252.0],
    "target_head_position": [135.0, 107.0],
    "target_head_scale": 24.0,
    "confidence_threshold": 0.35,
    "high_confidence_threshold": 0.6,
    "smoothing": 0.8,
    "max_interpolation_gap": 5,
    "max_missing_joint_run": 8,
    "max_scale_step": 0.02,
    "shoulder_weight": 0.6,
}


def write_test_profile(directory: Path) -> Path:
    """Write the small canonical profile and return its absolute path."""
    import yaml

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "canonical_test.yaml"
    path.write_text(yaml.safe_dump(TEST_PROFILE, sort_keys=True), encoding="utf-8")
    return path


#: Motion A: a small figure, high in a wide-ish frame.
MOTION_A = {
    "shoulder_width": 90.0,
    "torso_length": 130.0,
    "center": (300.0, 380.0),
    "drift": (0.35, 0.0),
    "phase": 0.0,
    "frame_size": (720, 900),
}

#: Motion B: the SAME motion at roughly 2.3x the apparent size, lower and
#: further right in a taller frame. Normalization must erase this difference.
MOTION_B = {
    "shoulder_width": 210.0,
    "torso_length": 300.0,
    "center": (700.0, 900.0),
    "drift": (0.35 * 2.33, 0.0),
    "phase": 0.0,
    "frame_size": (1280, 1600),
}


@dataclass
class MotionFixture:
    source: MotionSource
    poses: list[PoseFrame]
    pose_dir: Path


def make_poses(
    spec: dict[str, Any],
    start: int,
    end: int,
    *,
    phase_offset: float = 0.0,
    frame_offset: int = 0,
    confidence: float = 0.95,
) -> list[PoseFrame]:
    """Generate a synthetic pose sequence with the given body scale.

    ``frame_offset`` shifts the figure in time. Two sequences whose frame
    numbering differs by exactly that offset contain identical poses, which is
    how a *known* compatible anchor pair is constructed for the matcher tests.
    """
    adapter = MockPoseAdapter(
        shoulder_width=spec["shoulder_width"],
        torso_length=spec["torso_length"],
        center=spec["center"],
        drift=spec["drift"],
        phase=spec["phase"] + phase_offset,
        frame_offset=frame_offset,
        confidence=confidence,
    )
    return [adapter.pose_at(index, fps=FPS) for index in range(start, end)]


#: Frame offset applied to Motion B so that B[f] shows the same body pose as
#: A[f + KNOWN_ANCHOR_OFFSET]. The matcher must discover this.
KNOWN_ANCHOR_OFFSET = 45


def drop_joint(poses: list[PoseFrame], joint: str, frames: range) -> list[PoseFrame]:
    """Remove a joint from selected frames, to exercise the gap handling."""
    out: list[PoseFrame] = []
    for pose in poses:
        if pose.frame_index in frames:
            body = {k: v for k, v in pose.body.items() if k != joint}
            out.append(pose.model_copy(update={"body": body}))
        else:
            out.append(pose)
    return out


def blur_confidence(poses: list[PoseFrame], frames: range, confidence: float) -> list[PoseFrame]:
    """Lower every joint's confidence on selected frames."""
    out: list[PoseFrame] = []
    for pose in poses:
        if pose.frame_index in frames:
            body = {
                name: Joint2D(x=j.x, y=j.y, confidence=confidence) for name, j in pose.body.items()
            }
            out.append(pose.model_copy(update={"body": body}))
        else:
            out.append(pose)
    return out


def register_motion_source(
    context: ServiceContext,
    *,
    motion_id: str,
    spec: dict[str, Any],
    start: int,
    end: int,
    poses: list[PoseFrame] | None = None,
    authorized: bool = True,
    display_name: str | None = None,
) -> MotionFixture:
    """Create a motion source with pose data on disk, bypassing ffprobe."""
    root = context.data_root.resolve("motion_sources", motion_id)
    pose_dir = root / "pose"
    pose_dir.mkdir(parents=True, exist_ok=True)

    sequence = poses if poses is not None else make_poses(spec, start, end)
    save_pose_sequence(pose_dir, sequence)

    # A stand-in source file so the hash field describes something real. The
    # pipeline never reads a motion reference's pixels, so a placeholder is
    # honest here rather than a shortcut.
    placeholder = root / "source_placeholder.bin"
    placeholder.write_bytes(f"synthetic-motion-reference::{motion_id}\n".encode())

    width, height = spec["frame_size"]
    record = MotionSource(
        id=motion_id,
        version=1,
        display_name=display_name or f"Synthetic motion {motion_id}",
        source_video_path=str(placeholder),
        source_sha256=sha256_file(placeholder),
        video=VideoSpec(
            width=width,
            height=height,
            fps=FPS,
            duration_s=end / FPS,
            frame_count=end,
            codec="h264",
            pixel_format="yuv420p",
            constant_frame_rate=True,
            avg_frame_rate=f"{int(FPS)}/1",
            r_frame_rate=f"{int(FPS)}/1",
        ),
        selected_range=FrameRange(start=start, end=end),
        pose_dir=context.relative(pose_dir),
        pose_origin="imported",
        quality=measure_pose_quality(sequence, context.config),
        usage_rights=MotionUsageRights(
            motion_use_authorized=authorized,
            rights_holder="Test Fixture",
            license="test-only",
        ),
        status=MotionSourceStatus.READY,
    )
    saved = context.repos.motion_sources.save(record)
    return MotionFixture(source=saved, poses=sequence, pose_dir=pose_dir)


def make_motion_pair(
    context: ServiceContext,
    *,
    a_range: tuple[int, int] = (0, 60),
    b_range: tuple[int, int] = (0, 60),
    b_phase_offset: float = 0.0,
    b_frame_offset: int = KNOWN_ANCHOR_OFFSET,
) -> tuple[MotionFixture, MotionFixture]:
    """The canonical test pair: same motion, very different apparent scale.

    Motion B is time-shifted by ``b_frame_offset``, so B[f] is the same body
    pose as A[f + offset]. A genuinely compatible join therefore exists and the
    anchor matcher has a correct answer to find.
    """
    a = register_motion_source(
        context,
        motion_id="mot_a",
        spec=MOTION_A,
        start=a_range[0],
        end=a_range[1],
        display_name="Motion A (small, high in frame)",
    )
    b = register_motion_source(
        context,
        motion_id="mot_b",
        spec=MOTION_B,
        start=b_range[0],
        end=b_range[1],
        poses=make_poses(
            MOTION_B,
            b_range[0],
            b_range[1],
            phase_offset=b_phase_offset,
            frame_offset=b_frame_offset,
        ),
        display_name="Motion B (large, low in frame)",
    )
    return a, b


def make_hero(context: ServiceContext, *, hero_id: str = "hero_test") -> Any:
    """Register a Hero Character with real reference image files."""
    import cv2
    import numpy as np

    from app.pipeline.master_create import HeroOptions, register_hero

    staging = context.data_root.resolve("tmp", f"{hero_id}_refs")
    staging.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, color in enumerate(((60, 180, 240), (80, 160, 220))):
        image = np.full((640, 480, 3), 235, dtype=np.uint8)
        cv2.rectangle(image, (120, 90), (360, 540), color, -1)
        cv2.circle(image, (240, 150), 70, (170, 190, 215), -1)
        path = staging / f"hero_{index}.png"
        cv2.imwrite(str(path), image)
        paths.append(path)

    return register_hero(
        context,
        HeroOptions(
            display_name="Hero Character v1",
            reference_images=paths,
            subject_kind="synthetic",
            hero_id=hero_id,
        ),
    )


def canonical_shoulder_widths(poses: list[PoseFrame]) -> list[float]:
    return [w for w in (p.shoulder_width(0.0) for p in poses) if w]


def relative_signature(poses: list[PoseFrame], joint: str) -> list[tuple[float, float]]:
    """A joint's offset from the torso centre, per frame."""
    out: list[tuple[float, float]] = []
    for pose in poses:
        center = pose.torso_center(0.0)
        target = pose.joint(joint)
        if center is None or target is None:
            out.append((math.nan, math.nan))
            continue
        out.append((target.x - center[0], target.y - center[1]))
    return out


__all__ = [
    "FPS",
    "KNOWN_ANCHOR_OFFSET",
    "MOTION_A",
    "MOTION_B",
    "TEST_PROFILE",
    "MotionFixture",
    "blur_confidence",
    "canonical_shoulder_widths",
    "drop_joint",
    "make_hero",
    "make_motion_pair",
    "make_poses",
    "register_motion_source",
    "relative_signature",
    "write_test_profile",
]
