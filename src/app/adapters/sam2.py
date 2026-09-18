"""Video mask propagation (SAM 2 class of models).

Purpose in this system: an operator annotates the garment region on a handful
of keyframes; the adapter propagates those masks across every reveal frame,
producing the per-frame garment masks the compositor needs.

Required properties for a candidate model:

* temporally stable boundaries (a jittering hem produces visible flicker),
* ability to accept positive/negative point or box prompts per keyframe,
* ability to run in windows so an 8GB GPU never holds the whole clip,
* deterministic output for a fixed prompt set and seed.

Output contract: one 8-bit grayscale PNG per frame in ``output_dir``, named
``frame_{index:06d}.png``, same dimensions as the source frame, using the
project mask conventions (0 immutable / 255 editable).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.adapters.base import (
    AdapterKind,
    AnalysisAdapter,
    NotImplementedAdapter,
    registry,
)


class VideoMaskPropagationAdapter(AnalysisAdapter):
    """Interface a future implementation must satisfy."""

    kind = AdapterKind.VIDEO_MASK_PROPAGATION

    def propagate(
        self,
        *,
        frames_dir: Path,
        output_dir: Path,
        frame_indices: list[int],
        keyframe_prompts: dict[int, Any],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Propagate operator keyframe prompts across ``frame_indices``.

        ``keyframe_prompts`` maps a frame index to that frame's prompt payload
        (points, boxes or an authored mask). Implementations must be
        deterministic for a given prompt set.
        """
        self.require_available()
        raise AssertionError("unreachable")  # pragma: no cover


class _VideoMaskPropagationAdapterStub(NotImplementedAdapter, VideoMaskPropagationAdapter):
    """Documented placeholder; :meth:`run` raises rather than faking output."""


sam2_stub = _VideoMaskPropagationAdapterStub(
    AdapterKind.VIDEO_MASK_PROPAGATION,
    "sam2-video-propagation",
    reason="No mask-propagation model has been selected or installed yet.",
    expected_outputs=("frame_{index:06d}.png (grayscale mask per frame)",),
    requires_gpu=True,
    estimated_vram_mb=4096,
    candidate_models=("SAM 2 (video predictor)", "SAM 2.1", "Cutie", "DEVA", "XMem"),
    integration_notes=(
        "Expected to be prompted with operator keyframe annotations and run "
        "in frame windows. Must write masks only; never touch source_frames."
    ),
)

registry.register(sam2_stub)

__all__ = ["VideoMaskPropagationAdapter", "sam2_stub"]
