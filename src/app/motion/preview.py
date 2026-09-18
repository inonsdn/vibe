"""Skeleton preview rendering.

The preview exists so an operator can judge a composition — especially a
bridge — **before** spending GPU hours animating a character. It draws stick
figures on a flat background: no source imagery is read, so a preview can never
leak a reference person's appearance.

Each frame is labelled with its output index and its origin (``segment``,
``bridge`` or ``anchor``), which is what makes an off-by-one at a join visible
rather than merely measurable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.errors import ValidationError
from app.media.frames import DEFAULT_TEMPLATE, ffmpeg_pattern, frame_path, write_frame
from app.motion.pose_format import PoseFrame, PoseOrigin
from app.motion.skeleton import SKELETON_EDGES

_FONT = cv2.FONT_HERSHEY_SIMPLEX

#: BGR colours per frame origin, so the bridge is obvious at a glance.
ORIGIN_COLORS: dict[str, tuple[int, int, int]] = {
    "segment": (90, 220, 120),
    "bridge": (60, 170, 255),
    "anchor": (80, 80, 255),
}


@dataclass
class PreviewSettings:
    width: int = 1080
    height: int = 1920
    fps: float = 30.0
    background: tuple[int, int, int] = (24, 22, 28)
    joint_radius: int = 7
    bone_thickness: int = 4
    draw_labels: bool = True
    scale: float = 0.5


def draw_pose(
    canvas: np.ndarray,
    pose: PoseFrame,
    *,
    color: tuple[int, int, int],
    joint_radius: int = 7,
    bone_thickness: int = 4,
) -> np.ndarray:
    """Draw one skeleton onto ``canvas`` (modified in place and returned)."""
    height, width = canvas.shape[:2]

    def point(name: str) -> tuple[int, int] | None:
        joint = pose.joint(name)
        if joint is None:
            return None
        x, y = round(joint.x), round(joint.y)
        if not (-width <= x <= 2 * width and -height <= y <= 2 * height):
            return None
        return (x, y)

    for a_name, b_name in SKELETON_EDGES:
        a, b = point(a_name), point(b_name)
        if a and b:
            cv2.line(canvas, a, b, color, bone_thickness, cv2.LINE_AA)

    for name in pose.body:
        centre = point(name)
        if centre is None:
            continue
        joint = pose.body[name]
        # Low-confidence joints are drawn hollow, so a guessed pose looks guessed.
        if joint.confidence >= 0.5:
            cv2.circle(canvas, centre, joint_radius, color, -1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, centre, joint_radius, color, 2, cv2.LINE_AA)
    return canvas


def render_preview_frame(
    pose: PoseFrame,
    settings: PreviewSettings,
    *,
    origin_label: str = "segment",
    caption: str | None = None,
) -> np.ndarray:
    canvas = np.empty((settings.height, settings.width, 3), dtype=np.uint8)
    canvas[:, :] = settings.background

    # A centre guide makes body-centre drift at a join obvious by eye.
    cv2.line(
        canvas,
        (settings.width // 2, 0),
        (settings.width // 2, settings.height),
        (52, 50, 58),
        1,
        cv2.LINE_AA,
    )

    color = ORIGIN_COLORS.get(origin_label, ORIGIN_COLORS["segment"])
    draw_pose(
        canvas,
        pose,
        color=color,
        joint_radius=settings.joint_radius,
        bone_thickness=settings.bone_thickness,
    )

    if settings.draw_labels:
        bar = max(44, settings.height // 28)
        cv2.rectangle(
            canvas, (0, settings.height - bar), (settings.width, settings.height), (0, 0, 0), -1
        )
        text = caption or f"{pose.frame_index}  {origin_label}"
        cv2.putText(
            canvas,
            text,
            (14, settings.height - bar // 3),
            _FONT,
            max(0.6, bar / 46.0),
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas


def render_skeleton_preview(
    poses: list[PoseFrame],
    output_path: str | os.PathLike[str],
    settings: PreviewSettings | None = None,
    *,
    frames_dir: str | os.PathLike[str] | None = None,
    anchor_frames: set[int] | None = None,
    ffmpeg_binary: str = "ffmpeg",
    keep_frames: bool = False,
) -> dict[str, Any]:
    """Render a skeleton preview video. Returns manifest-ready metadata."""
    from app.media import ffmpeg

    options = settings or PreviewSettings()
    if not poses:
        raise ValidationError("Cannot render a preview of an empty composition")

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = (
        Path(frames_dir) if frames_dir is not None else target.parent / f".{target.stem}_frames"
    )
    staging.mkdir(parents=True, exist_ok=True)

    anchors = anchor_frames or set()
    first_index = poses[0].frame_index
    for pose in poses:
        if pose.frame_index in anchors:
            label = "anchor"
        elif pose.origin is PoseOrigin.BRIDGE:
            label = "bridge"
        else:
            label = "segment"
        write_frame(
            frame_path(staging, pose.frame_index, DEFAULT_TEMPLATE),
            render_preview_frame(pose, options, origin_label=label),
        )

    argv = ffmpeg.build_encode_from_frames_command(
        staging / ffmpeg_pattern(DEFAULT_TEMPLATE),
        target,
        ffmpeg=ffmpeg_binary,
        fps=options.fps,
        width=int(options.width * options.scale) // 2 * 2,
        height=int(options.height * options.scale) // 2 * 2,
        crf=26,
        preset="veryfast",
        start_number=first_index,
        faststart=True,
    )
    ffmpeg.run_command(argv)

    if not keep_frames and frames_dir is None:
        for entry in staging.iterdir():
            entry.unlink(missing_ok=True)
        staging.rmdir()

    return {
        "path": str(target),
        "frame_count": len(poses),
        "first_frame": first_index,
        "last_frame": poses[-1].frame_index,
        "fps": options.fps,
        "ffmpeg_command": argv,
        "anchor_frames": sorted(anchors),
    }


__all__ = [
    "ORIGIN_COLORS",
    "PreviewSettings",
    "draw_pose",
    "render_preview_frame",
    "render_skeleton_preview",
]
