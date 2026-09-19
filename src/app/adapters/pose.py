"""2D/3D body pose estimation.

Two distinct roles, deliberately kept apart:

**Garment pipeline.** Pose is *control and QA metadata only*. It is used to
decide which garment views a motion requires, to sanity-check that motion was
preserved, and to drive future pose-conditioned renderers. It never regenerates
the human — the master video's pixels remain the source of truth.

**Motion Composition.** Pose is the *product*. A motion reference contributes
its geometry and nothing else, so extraction quality directly determines what a
synthetic master can be.

Output contract: one JSON per frame in the format defined by
:mod:`app.motion.pose_format` — ``{"schema_version", "frame_index",
"timestamp_s", "body": {joint: {x, y, confidence}}, …}``.

**Input contract.** Two entry points, because the two callers genuinely differ:

``run(frames_dir=…)``
    The generic :class:`~app.adapters.base.AnalysisAdapter` shape, for a template
    whose frames are already extracted on disk.
``estimate_sequence(video_path=…, frame_range=…)``
    The Motion Composition entry point. A motion reference has **no** extracted
    frames — deliberately, so its pixels are never copied — so the adapter is
    given the video file itself plus the exact frame range to read.

An earlier version passed ``Path(video).parent`` as ``frames_dir``, which is
neither the video nor a controlled frame sequence and could contain anything
else the operator happened to leave in that folder.

**Frame numbering.** Written poses keep the *source* clip's frame indices, so
``frame_000042.json`` is frame 42 of the reference video regardless of where the
selected range starts. Downstream normalization relies on that alignment.

Three implementations live here:

* :class:`PoseAdapter` — the interface a real model must satisfy
* ``pose_stub`` — the registered default: reports ``not_implemented`` and
  **raises** rather than fabricating production pose data
* :class:`MockPoseAdapter` — deterministic synthetic poses for tests, which
  refuses to run unless explicitly constructed, so it can never be mistaken for
  a real detector
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from app.adapters.base import (
    AdapterCapability,
    AdapterKind,
    AdapterStatus,
    AnalysisAdapter,
    NotImplementedAdapter,
    registry,
)


class PoseAdapter(AnalysisAdapter):
    """Interface a future pose implementation must satisfy."""

    kind = AdapterKind.POSE

    def estimate_sequence(
        self,
        *,
        video_path: Path,
        output_dir: Path,
        frame_indices: list[int],
        fps: float,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Estimate poses for ``frame_indices`` of ``video_path``.

        ``frame_indices`` are indices in the *source clip*, and the written pose
        files must use those same indices — not a 0-based rewrite of them.

        Implementations must be deterministic for a fixed input, must write the
        internal pose JSON format into ``output_dir``, and must treat
        ``video_path`` as read-only.
        """
        self.require_available()
        raise AssertionError("unreachable")  # pragma: no cover


class _PoseAdapterStub(NotImplementedAdapter, PoseAdapter):
    """Documented placeholder; :meth:`run` raises rather than faking output."""


pose_stub = _PoseAdapterStub(
    AdapterKind.POSE,
    "pose-estimation",
    reason="No pose model has been selected or installed yet.",
    expected_outputs=("frame_{index:06d}.json (internal pose format)",),
    requires_gpu=True,
    estimated_vram_mb=1024,
    candidate_models=("DWPose", "RTMPose", "ViTPose", "OpenPose", "MediaPipe Pose"),
    integration_notes=(
        "Used as control/QA metadata by the garment pipeline, and as the primary "
        "product by Motion Composition. Judge candidates on temporal stability "
        "and on wrist accuracy: a jittering wrist is the most visible artefact "
        "at a motion join. Poses may also be produced externally and imported "
        "with `app motion import-pose`, which is how the system runs today."
    ),
)

registry.register(pose_stub)


class MockPoseAdapter(PoseAdapter):
    """Deterministic synthetic pose generator, for tests only.

    It is **not** registered in the process-wide registry, so no pipeline can
    pick it up by accident; a caller must construct it deliberately. Its
    capability report says plainly that its output is synthetic.

    The figure it produces is a simple articulated body with a travelling root
    and swinging limbs — enough structure for normalization, anchor matching,
    bridging and QC to be exercised meaningfully, and obviously not a detector.

    Limbs are placed by **angle at a fixed bone length**, so every bone keeps a
    constant length across the whole sequence, exactly as a real skeleton does.
    An earlier version offset joints along axes, which made forearm length vary
    with the swing — fixture noise that masqueraded as bridge distortion.
    """

    kind = AdapterKind.POSE
    name = "mock-pose"

    def __init__(
        self,
        *,
        shoulder_width: float = 120.0,
        torso_length: float = 180.0,
        center: tuple[float, float] = (480.0, 640.0),
        drift: tuple[float, float] = (0.0, 0.0),
        phase: float = 0.0,
        frame_offset: int = 0,
        confidence: float = 0.95,
    ) -> None:
        self.shoulder_width = shoulder_width
        self.torso_length = torso_length
        self.center = center
        self.drift = drift
        self.phase = phase
        # Shifts the whole figure in time. Because the body has two oscillators
        # at different frequencies, only a shared time shift produces genuinely
        # identical poses in two sequences -- which is what an anchor test needs.
        self.frame_offset = frame_offset
        self.confidence = confidence

    def capability(self) -> AdapterCapability:
        return AdapterCapability(
            kind=self.kind,
            name=self.name,
            status=AdapterStatus.AVAILABLE,
            reason=(
                "Deterministic synthetic pose generator for tests. Its output is "
                "NOT a detection and must never be used as production pose data."
            ),
            requires_gpu=False,
            estimated_vram_mb=0,
            expected_outputs=("frame_{index:06d}.json (internal pose format)",),
            notes={"synthetic": True, "for_tests_only": True},
        )

    def pose_at(self, frame_index: int, *, fps: float = 30.0) -> Any:
        """Build one synthetic pose. Pure function of ``frame_index``."""
        from app.motion.pose_format import Joint2D, PoseFrame

        shifted = frame_index + self.frame_offset
        t = shifted / max(fps, 1e-6)
        swing = math.sin(self.phase + t * 2.4)
        lift = math.cos(self.phase + t * 1.7)

        cx = self.center[0] + self.drift[0] * shifted
        cy = self.center[1] + self.drift[1] * shifted
        half = self.shoulder_width / 2.0
        torso = self.torso_length / 2.0

        shoulder_y = cy - torso
        hip_y = cy + torso
        hip_half = half * 0.72
        upper_arm = self.shoulder_width * 0.55
        forearm = self.shoulder_width * 0.52
        thigh = self.torso_length * 0.62
        shin = self.torso_length * 0.58
        head = self.shoulder_width * 0.30

        def joint(x: float, y: float, scale: float = 1.0) -> Joint2D:
            return Joint2D(x=x, y=y, confidence=min(1.0, self.confidence * scale))

        def limb(
            origin: tuple[float, float], angle_rad: float, length: float
        ) -> tuple[float, float]:
            """Place a joint at a fixed distance and a varying angle."""
            return (
                origin[0] + math.cos(angle_rad) * length,
                origin[1] + math.sin(angle_rad) * length,
            )

        left_shoulder = (cx - half, shoulder_y)
        right_shoulder = (cx + half, shoulder_y)
        left_hip = (cx - hip_half, hip_y)
        right_hip = (cx + hip_half, hip_y)

        # Angles are measured from +x (screen right), y growing downward.
        left_upper = math.radians(115.0) + 0.22 * swing
        right_upper = math.radians(65.0) - 0.22 * swing
        left_elbow = limb(left_shoulder, left_upper, upper_arm)
        right_elbow = limb(right_shoulder, right_upper, upper_arm)
        left_wrist = limb(left_elbow, left_upper + 0.34 * lift, forearm)
        right_wrist = limb(right_elbow, right_upper - 0.34 * lift, forearm)

        left_leg = math.radians(96.0) + 0.10 * swing
        right_leg = math.radians(84.0) - 0.10 * swing
        left_knee = limb(left_hip, left_leg, thigh)
        right_knee = limb(right_hip, right_leg, thigh)
        left_ankle = limb(left_knee, left_leg + 0.06 * lift, shin)
        right_ankle = limb(right_knee, right_leg - 0.06 * lift, shin)

        body = {
            "nose": joint(cx, shoulder_y - head * 1.6),
            "left_eye": joint(cx - head * 0.34, shoulder_y - head * 1.85),
            "right_eye": joint(cx + head * 0.34, shoulder_y - head * 1.85),
            "left_ear": joint(cx - head * 0.68, shoulder_y - head * 1.7),
            "right_ear": joint(cx + head * 0.68, shoulder_y - head * 1.7),
            "left_shoulder": joint(*left_shoulder),
            "right_shoulder": joint(*right_shoulder),
            "left_elbow": joint(*left_elbow),
            "right_elbow": joint(*right_elbow),
            "left_wrist": joint(*left_wrist),
            "right_wrist": joint(*right_wrist),
            "left_hip": joint(*left_hip),
            "right_hip": joint(*right_hip),
            "left_knee": joint(*left_knee),
            "right_knee": joint(*right_knee),
            "left_ankle": joint(*left_ankle),
            "right_ankle": joint(*right_ankle),
        }

        xs = [j.x for j in body.values()]
        ys = [j.y for j in body.values()]
        margin = self.shoulder_width * 0.15
        bbox = (
            min(xs) - margin,
            min(ys) - margin,
            (max(xs) - min(xs)) + 2 * margin,
            (max(ys) - min(ys)) + 2 * margin,
        )

        return PoseFrame(
            frame_index=frame_index,
            timestamp_s=t,
            body=body,
            source_bbox=bbox,
        )

    def _write(
        self, output_dir: Path, frame_indices: list[int], fps: float, source: str
    ) -> dict[str, Any]:
        from app.motion.pose_format import save_pose_sequence

        # Poses keep the SOURCE clip's frame indices, so a range starting at 12
        # writes frame_000012.json, not frame_000000.json.
        poses = [self.pose_at(index, fps=fps) for index in frame_indices]
        written = save_pose_sequence(output_dir, poses)
        return {
            "adapter": self.name,
            "synthetic": True,
            "frames": len(written),
            "first_frame": written[0] if written else None,
            "last_frame": written[-1] if written else None,
            "output_dir": str(output_dir),
            "input": source,
        }

    def run(
        self,
        *,
        frames_dir: Path,
        output_dir: Path,
        frame_indices: list[int],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Frame-directory entry point (the generic adapter shape)."""
        return self._write(
            output_dir, frame_indices, float((options or {}).get("fps", 30.0)), str(frames_dir)
        )

    def estimate_sequence(
        self,
        *,
        video_path: Path,
        output_dir: Path,
        frame_indices: list[int],
        fps: float,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Video entry point, used by Motion Composition."""
        return self._write(output_dir, frame_indices, fps, str(video_path))


__all__ = ["MockPoseAdapter", "PoseAdapter", "pose_stub"]
