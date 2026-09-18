"""ComfyUI renderer backend.

This backend is complete as *plumbing* and deliberately incomplete as a garment
model: it submits whatever workflow the operator supplies, maps logical inputs
onto that workflow via its contract, waits, and collects outputs. It contains no
model names, no checkpoint paths and nothing that would trigger a download.

Until a real garment workflow exists, ``prepare`` reports
``requires_model_nodes`` from the contract and ``render_frame`` fails with an
actionable error if the graph's node classes are not installed — it never
fabricates a frame.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from app.backends.base import (
    BackendCapabilities,
    FrameRequest,
    FrameResult,
    HealthStatus,
    RenderContext,
    RendererBackend,
)
from app.backends.comfyui.client import ComfyUIClient, PromptHandle
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
from app.media.masks import save_mask

logger = get_logger(__name__)

BACKEND_NAME = "comfyui"
BACKEND_VERSION = "0.1.0"


class ComfyUIBackend(RendererBackend):
    """Submits a locally authored ComfyUI workflow, one frame at a time."""

    name = BACKEND_NAME
    version = BACKEND_VERSION

    def __init__(
        self,
        config: AppConfig,
        *,
        client: ComfyUIClient | None = None,
    ) -> None:
        self._config = config
        self._comfy_config = config.comfyui
        # Constructing the client validates the endpoint against the offline
        # policy, so a remote URL fails here rather than mid-render.
        self._client = client or ComfyUIClient(config.comfyui)
        self._contract: WorkflowContract | None = None
        self._graph: dict[str, Any] | None = None
        self._uploads: dict[str, str] = {}
        self._prompts: dict[int, PromptHandle] = {}

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
        devices = stats.get("devices", []) if isinstance(stats, dict) else []
        return HealthStatus(
            healthy=True,
            detail="ComfyUI reachable on localhost.",
            version=str(system.get("comfyui_version", "unknown")),
            extra={
                "base_url": self._client.base_url,
                "python": system.get("python_version"),
                "devices": [
                    {
                        "name": device.get("name"),
                        "vram_total_mb": _to_mb(device.get("vram_total")),
                        "vram_free_mb": _to_mb(device.get("vram_free")),
                    }
                    for device in devices
                    if isinstance(device, dict)
                ],
            },
        )

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name=BACKEND_NAME,
            version=BACKEND_VERSION,
            requires_gpu=True,
            requires_model_weights=True,
            deterministic=False,  # depends entirely on the chosen workflow
            supports_windows=True,
            supports_resume=True,
            max_frame_window=self._config.backend.frame_window,
            expected_vram_mb=self._config.runtime.vram_budget_mb,
            supports_low_vram=True,
            supports_tiling=True,
            notes={
                "base_url": self._client.base_url,
                "workflow_id": self._comfy_config.workflow_id,
                "model_status": (
                    "No garment model is integrated yet. This backend submits "
                    "whatever workflow the operator supplies and never downloads "
                    "weights."
                ),
                "determinism": (
                    "Determinism is a property of the selected workflow; seeds are "
                    "supplied per frame but sampler/node behaviour must be verified."
                ),
            },
        )

    def prepare(self, context: RenderContext) -> dict[str, Any]:
        """Load and validate the workflow + contract, and stage inputs."""
        workflow_id = context.job.workflow_id or self._comfy_config.workflow_id
        contract_path = find_contract(self._config.workflows_dir(), workflow_id)
        contract = load_contract(contract_path)
        workflow_path = context.workflow_path or (
            self._config.workflows_dir() / contract.workflow_file
        )
        graph = load_workflow(workflow_path)

        problems = contract.validate_against(graph)
        if problems:
            raise BackendError(
                "Workflow does not satisfy its contract",
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
                "ComfyUI is not reachable; cannot prepare the job.",
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
        self._uploads = {}
        if self._comfy_config.upload_inputs:
            self._uploads = self._upload_garment_references(context)

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
            "uploaded_inputs": dict(self._uploads),
            "comfyui_version": health.version,
        }
        logger.info("comfyui_prepare", extra={"event": "comfyui_prepare", **info})
        return info

    def render_frame(self, context: RenderContext, request: FrameRequest) -> FrameResult:
        if self._contract is None or self._graph is None:
            raise BackendError("prepare() must be called before render_frame()")

        started = time.perf_counter()
        staged = self._stage_frame_inputs(context, request)
        values = self._build_values(context, request, staged)
        graph = self._contract.apply(self._graph, values)

        handle = self._client.submit(graph)
        self._prompts[request.frame_index] = handle
        outcome = self._client.wait(handle.prompt_id)
        if not outcome.completed and not outcome.images:
            raise BackendError(
                "ComfyUI did not produce an output image for the frame",
                frame_index=request.frame_index,
                prompt_id=handle.prompt_id,
                status=outcome.status,
            )
        images = outcome.images
        if not images:
            raise BackendError(
                "ComfyUI reported completion but returned no images",
                frame_index=request.frame_index,
                prompt_id=handle.prompt_id,
                hint="Check that the workflow contains a SaveImage node.",
            )

        destination = context.raw_frames_dir / frame_filename(request.frame_index)
        self._client.fetch_output(images[0], destination)
        image = read_frame(destination)
        if image.shape[:2] != request.source.shape[:2]:
            raise BackendError(
                "ComfyUI returned a frame whose dimensions do not match the source",
                frame_index=request.frame_index,
                returned=list(image.shape[:2]),
                expected=list(request.source.shape[:2]),
            )

        return FrameResult(
            frame_index=request.frame_index,
            image=image,
            seed=request.seed,
            backend_metadata={
                "prompt_id": handle.prompt_id,
                "prompt_number": handle.number,
                "output_filename": images[0].get("filename"),
                "elapsed_s": round(outcome.elapsed_s, 3),
            },
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    def resume(self, context: RenderContext) -> dict[str, Any]:
        """Re-attach after an interruption.

        Any prompt still queued from a previous run is interrupted rather than
        left to write frames behind the pipeline's back, then ``prepare`` is
        re-run so the contract and staged inputs are valid again.
        """
        interrupted = False
        try:
            state = self._client.queue_state()
            running = state.get("queue_running", []) or []
            if running:
                self._client.interrupt()
                interrupted = True
        except (BackendError, BackendUnavailableError) as exc:
            logger.warning(
                "comfyui_resume_queue_unavailable",
                extra={"event": "comfyui_resume_queue_unavailable", "error": exc.message},
            )
        info = self.prepare(context)
        return {"resumed": True, "interrupted_running_prompt": interrupted, **info}

    def collect_artifacts(self, context: RenderContext) -> dict[str, Any]:
        return {
            "backend": BACKEND_NAME,
            "version": BACKEND_VERSION,
            "prompt_ids": {
                str(index): handle.prompt_id for index, handle in sorted(self._prompts.items())
            },
            "uploaded_inputs": dict(self._uploads),
        }

    def close(self) -> None:
        self._client.close()

    # -- helpers ----------------------------------------------------------
    def _upload_garment_references(self, context: RenderContext) -> dict[str, str]:
        uploads: dict[str, str] = {}
        for image in context.garment.images:
            path = next(
                (p for p in context.garment_image_paths if p.name == Path(image.path).name),
                None,
            )
            if path is None or not path.is_file():
                continue
            key = f"garment_{image.view.value}"
            if key in uploads:
                continue
            uploads[key] = self._client.upload_image(path, subfolder="garment_replacer")
        return uploads

    def _stage_frame_inputs(self, context: RenderContext, request: FrameRequest) -> dict[str, str]:
        """Write per-frame source/mask files and upload them if configured."""
        staging = context.job_dir / "comfy_inputs"
        staging.mkdir(parents=True, exist_ok=True)
        source_path = staging / f"source_{frame_filename(request.frame_index)}"
        mask_path = staging / f"mask_{frame_filename(request.frame_index)}"
        write_frame(source_path, request.source)
        save_mask(mask_path, request.effective_mask)

        staged = {"source_frame": str(source_path), "mask": str(mask_path)}
        if self._comfy_config.upload_inputs:
            staged["source_frame"] = self._client.upload_image(
                source_path, subfolder="garment_replacer"
            )
            staged["mask"] = self._client.upload_image(mask_path, subfolder="garment_replacer")
        return staged

    def _build_values(
        self,
        context: RenderContext,
        request: FrameRequest,
        staged: dict[str, str],
    ) -> dict[str, Any]:
        """Map pipeline data onto the contract's declared logical inputs."""
        assert self._contract is not None
        settings = context.job.settings
        candidates: dict[str, Any] = {
            "source_frame": staged["source_frame"],
            "mask": staged["mask"],
            "seed": request.seed,
            "steps": settings.steps,
            "denoise": settings.denoise_strength,
            "guidance_scale": settings.guidance_scale,
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "width": context.template.video.width,
            "height": context.template.video.height,
            "frame_index": request.frame_index,
            "batch_size": 1,
            "output_prefix": f"{context.job.id}_{request.frame_index:06d}",
        }
        for view_key, logical in (
            ("garment_front", "garment_reference"),
            ("garment_back", "garment_reference_back"),
            ("garment_side", "garment_reference_side"),
        ):
            if view_key in self._uploads:
                candidates[logical] = self._uploads[view_key]

        declared = {binding.logical_name for binding in self._contract.bindings}
        return {name: value for name, value in candidates.items() if name in declared}


def _to_mb(value: Any) -> int | None:
    try:
        return int(int(value) / (1024 * 1024))
    except (TypeError, ValueError):
        return None


__all__ = ["BACKEND_NAME", "BACKEND_VERSION", "ComfyUIBackend"]
