"""Deterministic mock character animator: no GPU, no weights, no network.

It draws the Hero Character as a flat-shaded articulated figure following the
composed pose, on a stable background. That is enough to exercise the whole
master-creation path — chunking, resume, QC, manifests and the acceptance gate
— while being impossible to mistake for photoreal output.

Determinism comes from deriving everything from ``(seed, frame_index,
hero_key)``. No state carries between chunks, so animating frames 40-60 alone
produces exactly the pixels a single-pass run would, which is what makes resume
bit-exact.

The background is a function of the hero key alone, not of the frame index, so
the background-consistency QC check has something real to measure.

**Context mode is ``none``, deliberately.** Every frame here is a pure function
of ``(hero, pose, seed, frame_index)``, so there is nothing for this backend to
condition on and nothing it could honestly claim to have consumed. Declaring
``last_frame`` or ``sequence`` would make the manifest record context frames the
renderer never looked at, which is exactly the dishonesty
:class:`~app.backends.animator.base.ContextMode` exists to prevent. A real
animator declares what it really reads; the pipeline then gathers exactly that
much and no more. The gathering path for the other two modes is covered by stub
backends in the test suite.
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from app.backends.animator.base import (
    AnimationChunkRequest,
    AnimationChunkResult,
    AnimatorCapabilities,
    AnimatorContext,
    CharacterAnimatorBackend,
    ContextMode,
)
from app.backends.base import HealthStatus
from app.core.config import AppConfig
from app.core.determinism import derive_seed
from app.core.logging import get_logger
from app.motion.pose_format import PoseFrame
from app.motion.skeleton import SKELETON_EDGES

logger = get_logger(__name__)

BACKEND_NAME = "mock"
BACKEND_VERSION = "1.0.0"


class MockAnimatorBackend(CharacterAnimatorBackend):
    """A visible, reproducible stand-in for a real character animator."""

    name = BACKEND_NAME
    version = BACKEND_VERSION

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    # -- interface --------------------------------------------------------
    def healthcheck(self) -> HealthStatus:
        return HealthStatus(
            healthy=True,
            detail="Mock animator is always available (no GPU, weights or network).",
            version=BACKEND_VERSION,
            extra={"requires_network": False, "requires_gpu": False},
        )

    def capabilities(self) -> AnimatorCapabilities:
        return AnimatorCapabilities(
            name=BACKEND_NAME,
            version=BACKEND_VERSION,
            requires_gpu=False,
            requires_model_weights=False,
            deterministic=True,
            context_mode=ContextMode.NONE,
            max_chunk_frames=256,
            recommended_chunk_frames=self._config.animator.chunk_frames,
            recommended_overlap_frames=self._config.animator.overlap_frames,
            expected_vram_mb=0,
            supports_low_vram=True,
            produces_photoreal=False,
            notes={
                "purpose": (
                    "Exercises master creation end to end without any AI model. "
                    "Output is intentionally schematic, not photoreal."
                ),
                "determinism": "pixels derive from (seed, frame_index, hero_key)",
                "context": (
                    "None. Each frame is independent, so chunk boundaries are "
                    "continuous without conditioning. A real animator will "
                    "declare last_frame or sequence."
                ),
            },
        )

    def prepare(self, context: AnimatorContext) -> dict[str, Any]:
        info = {
            "backend": BACKEND_NAME,
            "version": BACKEND_VERSION,
            "hero_key": context.hero.version_key(),
            "palette": [list(color) for color in self._palette(context)],
            "background": list(self._background_color(context)),
            "requires_network": False,
            "produces_photoreal": False,
        }
        logger.info("mock_animator_prepare", extra={"event": "mock_animator_prepare", **info})
        return info

    def animate_chunk(
        self, context: AnimatorContext, request: AnimationChunkRequest
    ) -> AnimationChunkResult:
        started = time.perf_counter()
        frames: dict[int, np.ndarray] = {}
        for pose in request.poses:
            if not request.start_frame <= pose.frame_index < request.end_frame:
                continue
            frames[pose.frame_index] = self._paint(context, pose, request.seed)

        missing = sorted(set(request.frame_indices) - set(frames))
        if missing:
            from app.core.errors import BackendError

            raise BackendError(
                "Chunk is missing pose data for some frames",
                chunk_index=request.chunk_index,
                missing=missing[:16],
                missing_count=len(missing),
            )

        return AnimationChunkResult(
            chunk_index=request.chunk_index,
            frames=frames,
            seed=request.seed,
            backend_metadata={
                "deterministic": True,
                # Truthfully zero: see the module docstring. The pipeline hands
                # this backend no context because it declares ContextMode.NONE.
                "context_frames_used": 0,
                "conditioned_on_previous": False,
            },
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    def collect_artifacts(self, context: AnimatorContext) -> dict[str, Any]:
        return {"backend": BACKEND_NAME, "version": BACKEND_VERSION, "artifacts": []}

    # -- painting ---------------------------------------------------------
    def _palette(self, context: AnimatorContext) -> tuple[tuple[int, int, int], ...]:
        """Character colours derived from the hero key: stable and distinct."""
        base = derive_seed("hero-palette", context.hero.version_key())
        skin = (
            140 + (base % 60),
            165 + ((base >> 8) % 50),
            195 + ((base >> 16) % 40),
        )
        garment = (
            40 + ((base >> 4) % 120),
            40 + ((base >> 12) % 120),
            120 + ((base >> 20) % 100),
        )
        hair = (28 + (base % 30), 32 + ((base >> 6) % 30), 48 + ((base >> 10) % 40))
        return (skin, garment, hair)

    def _background_color(self, context: AnimatorContext) -> tuple[int, int, int]:
        base = derive_seed("hero-background", context.hero.version_key())
        return (30 + (base % 25), 28 + ((base >> 8) % 25), 34 + ((base >> 16) % 25))

    def _paint(self, context: AnimatorContext, pose: PoseFrame, chunk_seed: int) -> np.ndarray:
        """Draw one frame. Pure function of (hero, pose, seed, frame index)."""
        height, width = context.frame_shape
        skin, garment, hair = self._palette(context)
        background = self._background_color(context)

        frame: np.ndarray = np.empty((height, width, 3), dtype=np.uint8)
        frame[:, :] = background

        # A fixed vignette keyed to the hero, so the background is identical in
        # every frame and background drift is measurable.
        ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
        xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
        falloff = np.clip(1.15 - 0.45 * (xs * xs + ys * ys), 0.0, 1.0)
        frame = np.clip(frame.astype(np.float32) * falloff[:, :, None], 0, 255).astype(np.uint8)

        def point(name: str) -> tuple[int, int] | None:
            joint = pose.joint(name)
            if joint is None:
                return None
            return (round(joint.x), round(joint.y))

        shoulder_width = pose.shoulder_width(0.0) or 100.0
        limb = max(6, round(shoulder_width * 0.20))

        # Torso as a filled polygon, so the figure reads as a body rather than a
        # stick figure and the garment region is a real area.
        torso_names = ("left_shoulder", "right_shoulder", "right_hip", "left_hip")
        torso_points = [point(name) for name in torso_names]
        if all(p is not None for p in torso_points):
            cv2.fillPoly(
                frame,
                [np.array([p for p in torso_points if p], dtype=np.int32)],
                garment,
            )

        for a_name, b_name in SKELETON_EDGES:
            a, b = point(a_name), point(b_name)
            if a is None or b is None:
                continue
            is_head = a_name in {"nose", "left_eye", "right_eye"} or b_name in {
                "left_ear",
                "right_ear",
            }
            color = hair if is_head else skin
            cv2.line(frame, a, b, color, limb, cv2.LINE_AA)

        head = point("nose")
        if head is not None:
            radius = max(8, round(shoulder_width * 0.34))
            cv2.circle(frame, head, radius, skin, -1, cv2.LINE_AA)
            cv2.ellipse(
                frame,
                (head[0], head[1] - radius // 3),
                (radius, radius // 2),
                0,
                180,
                360,
                hair,
                -1,
                cv2.LINE_AA,
            )

        # A one-pixel marker row keyed to (seed, frame) makes chunk boundaries and
        # duplicated frames trivially detectable in tests and contact sheets.
        marker = derive_seed("frame-marker", chunk_seed, pose.frame_index) % 256
        frame[0:1, :, :] = np.uint8(marker)
        return frame


__all__ = ["BACKEND_NAME", "BACKEND_VERSION", "MockAnimatorBackend"]
