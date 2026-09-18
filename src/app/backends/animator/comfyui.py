"""ComfyUI character animator backend.

Complete as plumbing, deliberately incomplete as a character animator: it
submits whatever workflow the operator authors, maps logical inputs onto it via
a contract, waits, and collects frames. It contains no model names, no
checkpoint paths, and nothing that triggers a download.

It reuses :class:`~app.backends.comfyui.client.ComfyUIClient`, so it inherits
the same guarantees: remote endpoints refused in the constructor, no proxy
inheritance, bounded waits, and node binding by *title* rather than node id.

``produces_photoreal`` is reported as ``False`` and ``deterministic`` as
``False`` until a specific workflow has been integrated and measured. Claiming
either before that would be a lie the manifest would then record.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from app.backends.animator.base import (
    AnimationChunkRequest,
    AnimationChunkResult,
    AnimatorCapabilities,
    AnimatorContext,
    CharacterAnimatorBackend,
)
from app.backends.base import HealthStatus
from app.backends.comfyui.client import ComfyUIClient
from app.backends.comfyui.workflow import (
    WorkflowContract,
    find_contract,
    load_contract,
    load_workflow,
)
from app.core.config import AppConfig
from app.core.errors import BackendError, BackendUnavailableError
from app.core.hashing import sha256_file
from app.core.logging import get_logger
from app.media.frames import frame_filename, read_frame, write_frame
from app.motion.pose_format import save_pose_frame
from app.motion.preview import PreviewSettings, render_preview_frame

logger = get_logger(__name__)

BACKEND_NAME = "comfyui"
BACKEND_VERSION = "0.1.0"


class ComfyUIAnimatorBackend(CharacterAnimatorBackend):
    """Submits a locally authored ComfyUI character-animation workflow."""

    name = BACKEND_NAME
    version = BACKEND_VERSION

    def __init__(self, config: AppConfig, *, client: ComfyUIClient | None = None) -> None:
        self._config = config
        self._comfy_config = config.comfyui
        # Constructing the client validates the endpoint against the offline
        # policy, so a remote URL fails here rather than mid-animation.
        self._client = client or ComfyUIClient(config.comfyui)
        self._contract: WorkflowContract | None = None
        self._graph: dict[str, Any] | None = None
        self._prompts: dict[int, str] = {}

    # -- interface --------------------------------------------------------
    def healthcheck(self) -> HealthStatus:
        try:
            stats = self._client.health()
        except (BackendError, BackendUnavailableError) as exc:
            return HealthStatus(
                healthy=False,
                detail=exc.message,
                version=BACKEND_VERSION,
                extra={"base_url": self._client.base_url, **exc.details},
            )
        system = stats.get("system", {}) if isinstance(stats, dict) else {}
        return HealthStatus(
            healthy=True,
            detail="ComfyUI reachable on localhost.",
            version=str(system.get("comfyui_version", "unknown")),
            extra={"base_url": self._client.base_url},
        )

    def capabilities(self) -> AnimatorCapabilities:
        return AnimatorCapabilities(
            name=BACKEND_NAME,
            version=BACKEND_VERSION,
            requires_gpu=True,
            requires_model_weights=True,
            deterministic=False,
            supports_context_frames=True,
            max_chunk_frames=self._config.animator.max_chunk_frames,
            recommended_chunk_frames=self._config.animator.chunk_frames,
            recommended_overlap_frames=self._config.animator.overlap_frames,
            expected_vram_mb=self._config.runtime.vram_budget_mb,
            supports_low_vram=True,
            produces_photoreal=False,
            notes={
                "base_url": self._client.base_url,
                "workflow_id": self._config.animator.workflow_id,
                "model_status": (
                    "No character animation model is integrated. This backend "
                    "submits whatever workflow the operator supplies and never "
                    "downloads weights."
                ),
                "determinism": (
                    "A property of the selected workflow, not of this adapter. "
                    "Verify by animating one frame twice and comparing hashes."
                ),
                "photorealism": (
                    "Unverified. Do not claim photoreal character animation "
                    "until a real workflow has been integrated and reviewed."
                ),
            },
        )

    def prepare(self, context: AnimatorContext) -> dict[str, Any]:
        workflow_id = context.candidate.workflow_id or self._config.animator.workflow_id
        contract_path = find_contract(self._config.workflows_dir(), workflow_id)
        contract = load_contract(contract_path)
        workflow_path = context.workflow_path or (
            self._config.workflows_dir() / contract.workflow_file
        )
        graph = load_workflow(workflow_path)

        problems = contract.validate_against(graph)
        if problems:
            raise BackendError(
                "Character animation workflow does not satisfy its contract",
                workflow_id=workflow_id,
                problems=problems,
                hint=(
                    "Title the nodes in ComfyUI to match the contract's "
                    "node_title values, or bind by node_id."
                ),
            )

        health = self.healthcheck()
        if not health.healthy:
            raise BackendUnavailableError(
                "ComfyUI is not reachable; cannot prepare the master candidate.",
                detail=health.detail,
                base_url=self._client.base_url,
            )

        missing = self._client.missing_node_types(graph)
        if missing:
            raise BackendError(
                "The local ComfyUI installation is missing node types this "
                "workflow needs. Install the required custom nodes yourself; "
                "this application never downloads anything.",
                workflow_id=workflow_id,
                missing_node_types=missing,
            )

        self._contract = contract
        self._graph = graph
        info = {
            "backend": BACKEND_NAME,
            "version": BACKEND_VERSION,
            "base_url": self._client.base_url,
            "workflow_id": contract.workflow_id,
            "workflow_path": str(workflow_path),
            "workflow_sha256": sha256_file(workflow_path),
            "contract_path": str(contract_path),
            "contract_sha256": contract.contract_sha256(),
            "requires_model_nodes": contract.requires_model_nodes,
            "comfyui_version": health.version,
            "produces_photoreal": False,
        }
        logger.info("comfyui_animator_prepare", extra={"event": "comfyui_animator_prepare", **info})
        return info

    def animate_chunk(
        self, context: AnimatorContext, request: AnimationChunkRequest
    ) -> AnimationChunkResult:
        if self._contract is None or self._graph is None:
            raise BackendError("prepare() must be called before animate_chunk()")

        started = time.perf_counter()
        staged = self._stage_chunk_inputs(context, request)
        values = self._build_values(context, request, staged)
        graph = self._contract.apply(self._graph, values)

        handle = self._client.submit(graph)
        self._prompts[request.chunk_index] = handle.prompt_id
        outcome = self._client.wait(handle.prompt_id)

        images = outcome.images
        if not images:
            raise BackendError(
                "ComfyUI returned no frames for the chunk",
                chunk_index=request.chunk_index,
                prompt_id=handle.prompt_id,
                status=outcome.status,
                hint="Check that the workflow contains a SaveImage node.",
            )
        if len(images) != request.frame_count:
            raise BackendError(
                "ComfyUI returned a different number of frames than the chunk requested",
                chunk_index=request.chunk_index,
                expected=request.frame_count,
                returned=len(images),
                hint="The workflow's batch size must follow the supplied frame count.",
            )

        frames: dict[int, np.ndarray] = {}
        for offset, image in enumerate(images):
            frame_index = request.start_frame + offset
            destination = context.frames_dir / frame_filename(frame_index)
            self._client.fetch_output(image, destination)
            decoded = read_frame(destination)
            if decoded.shape[:2] != context.frame_shape:
                raise BackendError(
                    "ComfyUI returned a frame whose dimensions do not match the "
                    "canonical profile",
                    frame_index=frame_index,
                    returned=list(decoded.shape[:2]),
                    expected=list(context.frame_shape),
                )
            frames[frame_index] = decoded

        return AnimationChunkResult(
            chunk_index=request.chunk_index,
            frames=frames,
            seed=request.seed,
            backend_metadata={
                "prompt_id": handle.prompt_id,
                "elapsed_s": round(outcome.elapsed_s, 3),
                "context_frames_used": len(request.context_frames),
            },
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    def resume(self, context: AnimatorContext) -> dict[str, Any]:
        """Interrupt any stale prompt, then re-prepare."""
        interrupted = False
        try:
            state = self._client.queue_state()
            if state.get("queue_running", []):
                self._client.interrupt()
                interrupted = True
        except (BackendError, BackendUnavailableError) as exc:
            logger.warning(
                "comfyui_animator_resume_queue_unavailable",
                extra={"event": "comfyui_animator_resume_queue_unavailable", "error": exc.message},
            )
        info = self.prepare(context)
        return {"resumed": True, "interrupted_running_prompt": interrupted, **info}

    def collect_artifacts(self, context: AnimatorContext) -> dict[str, Any]:
        return {
            "backend": BACKEND_NAME,
            "version": BACKEND_VERSION,
            "prompt_ids": {str(k): v for k, v in sorted(self._prompts.items())},
        }

    def close(self) -> None:
        self._client.close()

    # -- helpers ----------------------------------------------------------
    def _stage_chunk_inputs(
        self, context: AnimatorContext, request: AnimationChunkRequest
    ) -> dict[str, str]:
        """Write pose control images, pose JSON and context frames for a chunk."""
        staging = context.candidate_dir / "comfy_inputs" / f"chunk_{request.chunk_index:04d}"
        staging.mkdir(parents=True, exist_ok=True)
        preview = PreviewSettings(
            width=context.candidate.width,
            height=context.candidate.height,
            fps=context.candidate.fps,
            draw_labels=False,
        )

        first_pose: Path | None = None
        for pose in request.poses:
            control = staging / f"pose_{frame_filename(pose.frame_index)}"
            write_frame(control, render_preview_frame(pose, preview))
            save_pose_frame(staging / f"pose_{pose.frame_index:06d}.json", pose)
            if first_pose is None:
                first_pose = control

        context_path: Path | None = None
        if request.context_frames:
            context_path = staging / "context_last.png"
            write_frame(context_path, request.context_frames[-1])

        staged: dict[str, str] = {}
        if first_pose is not None:
            staged["pose"] = (
                self._client.upload_image(first_pose, subfolder="garment_replacer_motion")
                if self._comfy_config.upload_inputs
                else str(first_pose)
            )
        if context_path is not None:
            staged["source_frame"] = (
                self._client.upload_image(context_path, subfolder="garment_replacer_motion")
                if self._comfy_config.upload_inputs
                else str(context_path)
            )
        return staged

    def _build_values(
        self,
        context: AnimatorContext,
        request: AnimationChunkRequest,
        staged: dict[str, str],
    ) -> dict[str, Any]:
        """Map candidate data onto the contract's declared logical inputs."""
        assert self._contract is not None
        hero_uploads: dict[str, str] = {}
        for path in context.hero_image_paths[:1]:
            if path.is_file():
                hero_uploads["garment_reference"] = (
                    self._client.upload_image(path, subfolder="garment_replacer_motion")
                    if self._comfy_config.upload_inputs
                    else str(path)
                )

        candidates: dict[str, Any] = {
            **staged,
            **hero_uploads,
            "seed": request.seed,
            "width": context.candidate.width,
            "height": context.candidate.height,
            "batch_size": request.frame_count,
            "frame_index": request.start_frame,
            "output_prefix": f"{context.candidate.id}_chunk{request.chunk_index:04d}",
            "steps": int(request.settings.get("steps", 20)),
            "denoise": float(request.settings.get("denoise", 1.0)),
            "guidance_scale": float(request.settings.get("guidance_scale", 5.0)),
            "prompt": str(request.settings.get("prompt", "")),
            "negative_prompt": str(request.settings.get("negative_prompt", "")),
        }
        declared = {binding.logical_name for binding in self._contract.bindings}
        return {name: value for name, value in candidates.items() if name in declared}


__all__ = ["BACKEND_NAME", "BACKEND_VERSION", "ComfyUIAnimatorBackend"]
