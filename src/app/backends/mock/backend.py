"""Mock garment renderer: no GPU, no weights, no network, fully deterministic.

Its job is to exercise the *whole* system — orchestration, compositing,
transition, resume, export, QC, manifests — while being obviously synthetic to
the eye. It paints a bold procedural pattern whose colours derive from the
garment's declared dominant colours, so different garments look different and
the same garment always looks identical.

Determinism comes from deriving every value from ``(seed, frame_index,
garment_key)``: no RNG state is carried between frames, so rendering frame 40
alone produces the same pixels as rendering frames 30..50 in sequence. That is
exactly the property the resume tests check.
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from app.backends.base import (
    BackendCapabilities,
    FrameRequest,
    FrameResult,
    HealthStatus,
    RenderContext,
    RendererBackend,
)
from app.core.config import AppConfig
from app.core.determinism import derive_seed
from app.core.logging import get_logger

logger = get_logger(__name__)

BACKEND_NAME = "mock"
BACKEND_VERSION = "1.0.0"

_FALLBACK_PALETTE: tuple[tuple[int, int, int], ...] = (
    (231, 76, 60),
    (41, 128, 185),
    (39, 174, 96),
    (241, 196, 15),
    (142, 68, 173),
    (26, 188, 156),
)


def _parse_hex_color(value: str) -> tuple[int, int, int] | None:
    text = value.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6:
        return None
    try:
        return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
    except ValueError:
        return None


class MockBackend(RendererBackend):
    """A visible, reproducible stand-in for a real garment model."""

    name = BACKEND_NAME
    version = BACKEND_VERSION

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    # -- interface --------------------------------------------------------
    def healthcheck(self) -> HealthStatus:
        return HealthStatus(
            healthy=True,
            detail="Mock backend is always available (no GPU, weights or network).",
            version=BACKEND_VERSION,
            extra={"requires_network": False, "requires_gpu": False},
        )

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name=BACKEND_NAME,
            version=BACKEND_VERSION,
            requires_gpu=False,
            requires_model_weights=False,
            deterministic=True,
            supports_windows=True,
            supports_resume=True,
            max_frame_window=256,
            expected_vram_mb=0,
            supports_low_vram=True,
            supports_tiling=False,
            notes={
                "purpose": (
                    "Exercises the full pipeline without any AI model. "
                    "Output is intentionally synthetic-looking."
                ),
                "determinism": "pixels derive from (seed, frame_index, garment_key)",
            },
        )

    def prepare(self, context: RenderContext) -> dict[str, Any]:
        palette = self._palette(context)
        info = {
            "backend": BACKEND_NAME,
            "version": BACKEND_VERSION,
            "palette": [list(color) for color in palette],
            "garment_key": context.garment.version_key(),
            "pattern": self._pattern_name(context),
            "requires_network": False,
        }
        logger.info("mock_prepare", extra={"event": "mock_prepare", **info})
        return info

    def render_frame(self, context: RenderContext, request: FrameRequest) -> FrameResult:
        started = time.perf_counter()
        image = self._paint(context, request)
        duration_ms = int((time.perf_counter() - started) * 1000)
        return FrameResult(
            frame_index=request.frame_index,
            image=image,
            seed=request.seed,
            backend_metadata={
                "pattern": self._pattern_name(context),
                "deterministic": True,
            },
            duration_ms=duration_ms,
        )

    def collect_artifacts(self, context: RenderContext) -> dict[str, Any]:
        return {"backend": BACKEND_NAME, "version": BACKEND_VERSION, "artifacts": []}

    # -- painting ---------------------------------------------------------
    def _palette(self, context: RenderContext) -> tuple[tuple[int, int, int], ...]:
        """Colours from the garment's declared dominant colours, else derived."""
        colors: list[tuple[int, int, int]] = []
        for value in context.garment.dominant_colors:
            parsed = _parse_hex_color(value)
            if parsed is not None:
                colors.append(parsed)
        if colors:
            return tuple(colors)
        # Deterministic pick so a garment without declared colours is still
        # stable and visually distinct from its neighbours.
        offset = derive_seed("palette", context.garment.version_key()) % len(_FALLBACK_PALETTE)
        rotated = _FALLBACK_PALETTE[offset:] + _FALLBACK_PALETTE[:offset]
        return rotated[:3]

    def _pattern_name(self, context: RenderContext) -> str:
        names = ("stripes", "checker", "chevron", "dots")
        index = derive_seed("pattern", context.garment.version_key()) % len(names)
        return names[index]

    def _paint(self, context: RenderContext, request: FrameRequest) -> np.ndarray:
        """Build a full-frame image with a bold pattern in the garment region.

        The pattern is drawn across the whole frame; the pipeline composites it
        through the effective mask, which is deliberately the only thing that
        decides where those pixels land.
        """
        height, width = request.source.shape[:2]
        palette = self._palette(context)
        pattern = self._pattern_name(context)

        # Per-frame phase, derived (not accumulated) so any frame can be
        # rendered in isolation and match a sequential run.
        phase = derive_seed("phase", request.seed, request.frame_index) % 64
        band = max(8, (min(height, width) // 24) + (phase % 5))

        ys = np.arange(height, dtype=np.int32)[:, None]
        xs = np.arange(width, dtype=np.int32)[None, :]

        if pattern == "stripes":
            index = ((xs + phase) // band) % len(palette)
        elif pattern == "checker":
            index = (((xs + phase) // band) + ((ys + phase) // band)) % len(palette)
        elif pattern == "chevron":
            index = ((np.abs(((xs + ys + phase) % (2 * band)) - band)) // max(1, band // 2)) % len(
                palette
            )
        else:  # dots
            index = (((xs + phase) // band) * 3 + ((ys + phase) // band) * 5) % len(palette)

        index = np.broadcast_to(index, (height, width))
        lookup = np.array([color[::-1] for color in palette], dtype=np.uint8)  # RGB -> BGR
        image = lookup[index]

        # A little shading so the garment is not a flat sticker, derived from the
        # source luma so lighting roughly tracks the original frame.
        luma = cv2.cvtColor(request.source, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        shading = 0.65 + 0.7 * luma[:, :, None]
        shaded = np.clip(image.astype(np.float32) * shading, 0, 255)

        # A per-frame marker stripe makes the transition frame trivially
        # identifiable in a contact sheet.
        marker = np.uint8((request.frame_index * 7 + phase) % 256)
        shaded[0:2, :, :] = marker
        return np.floor(shaded + 0.5).astype(np.uint8)


__all__ = ["BACKEND_NAME", "BACKEND_VERSION", "MockBackend"]
