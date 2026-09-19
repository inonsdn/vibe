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

**Sequences, not stills.** A chunk is 8-24 output frames, so every pose in the
chunk is staged as one ordered asset (a numbered PNG run, or a visually lossless
video where the contract asks for one) and the whole gathered context tail is
staged the same way. The backend refuses to submit a contract that cannot carry
them, and the frame counts it reports come from the staged assets rather than
from the request -- so the manifest records what the workflow actually received.
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
    ContextMode,
)
from app.backends.base import HealthStatus
from app.backends.comfyui.client import ComfyUIClient
from app.backends.comfyui.sequence import (
    SEQUENCE_KIND_VIDEO,
    StagedSequence,
    encode_sequence_video,
    stage_frame_sequence,
)
from app.backends.comfyui.workflow import (
    WorkflowBinding,
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

#: Root of the per-candidate staging tree inside ComfyUI's ``input/`` folder.
UPLOAD_SUBFOLDER = "garment_replacer_motion"


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
            # Declared, and then actually honoured: `_stage_chunk_inputs` stages
            # every context frame the pipeline hands over, in order, and refuses
            # to submit a contract that has nowhere to put them.
            context_mode=ContextMode.SEQUENCE,
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
        # Structural checks first: a malformed chunk must never reach ComfyUI,
        # because a workflow that silently animates the wrong frames produces
        # output nothing downstream can detect as wrong.
        request.validate()
        self._assert_contract_supports(request)

        staged = self._stage_chunk_inputs(context, request)
        self._assert_sequences_match(request, staged)
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
        identities = [
            (str(image.get("subfolder", "")), str(image.get("filename", "")))
            for image in images
            if isinstance(image, dict)
        ]
        if len(identities) == len(images) and len(set(identities)) != len(identities):
            duplicates = sorted({i for i in identities if identities.count(i) > 1})
            raise BackendError(
                "ComfyUI returned the same output image more than once; the "
                "chunk would contain duplicated frames",
                chunk_index=request.chunk_index,
                duplicates=[f"{sub}/{name}" if sub else name for sub, name in duplicates][:8],
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

        pose_sequence = staged.get("pose_sequence")
        context_sequence = staged.get("context_sequence")
        return AnimationChunkResult(
            chunk_index=request.chunk_index,
            frames=frames,
            seed=request.seed,
            backend_metadata={
                "prompt_id": handle.prompt_id,
                "elapsed_s": round(outcome.elapsed_s, 3),
                "context_mode": ContextMode.SEQUENCE.value,
                # The number of frames actually handed to the workflow, taken
                # from the staged asset rather than from the request, so the
                # manifest cannot claim context the workflow never received.
                "context_frames_used": context_sequence.count if context_sequence else 0,
                "pose_frames_submitted": pose_sequence.count if pose_sequence else 0,
                "pose_sequence_kind": pose_sequence.kind if pose_sequence else None,
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

    # -- validation -------------------------------------------------------
    def _assert_contract_supports(self, request: AnimationChunkRequest) -> None:
        """Refuse to submit a contract that cannot express this chunk."""
        assert self._contract is not None
        missing = self._contract.missing_animation_inputs()
        if missing:
            raise BackendError(
                "The character animation contract does not declare the inputs a "
                "multi-frame chunk needs. Without them the workflow receives one "
                "pose and decides its own frame count.",
                workflow_id=self._contract.workflow_id,
                missing_inputs=list(missing),
                declared=sorted(self._contract.declared_inputs()),
                hint=(
                    "Bind pose_sequence (kind: sequence_dir or sequence_video) "
                    "and batch_size in the contract YAML."
                ),
            )
        if not request.context_frames:
            return
        mode = self.capabilities().context_mode
        if mode is ContextMode.SEQUENCE and not self._contract.declares("context_sequence"):
            raise BackendError(
                "This backend consumes a context sequence, but the contract has "
                "nowhere to put one. Refusing to submit and silently drop the "
                "continuity signal.",
                workflow_id=self._contract.workflow_id,
                context_frames=len(request.context_frames),
                hint="Bind context_sequence (kind: sequence_dir or sequence_video).",
            )
        if mode is ContextMode.LAST_FRAME and not (
            self._contract.declares("context_frame") or self._contract.declares("source_frame")
        ):
            raise BackendError(
                "This backend consumes one context frame, but the contract "
                "declares neither context_frame nor source_frame.",
                workflow_id=self._contract.workflow_id,
                hint="Bind context_frame in the contract YAML.",
            )

    @staticmethod
    def _assert_sequences_match(
        request: AnimationChunkRequest, staged: dict[str, StagedSequence]
    ) -> None:
        """Every requested frame must be represented, once, in the right place."""
        poses = staged.get("pose_sequence")
        if poses is None or list(poses.frame_indices) != request.frame_indices:
            raise BackendError(
                "The staged pose sequence does not match the chunk frame by frame",
                chunk_index=request.chunk_index,
                expected_count=request.frame_count,
                staged_count=poses.count if poses else 0,
                expected_first=request.start_frame,
                expected_last=request.end_frame - 1,
                staged_first=poses.first_index if poses else None,
                staged_last=poses.last_index if poses else None,
            )
        contexts = staged.get("context_sequence")
        staged_context = list(contexts.frame_indices) if contexts else []
        if staged_context != request.context_frame_indices:
            raise BackendError(
                "The staged context sequence does not match the context frames "
                "the pipeline gathered",
                chunk_index=request.chunk_index,
                expected=request.context_frame_indices,
                staged=staged_context,
            )

    # -- staging ----------------------------------------------------------
    def _stage_chunk_inputs(
        self, context: AnimatorContext, request: AnimationChunkRequest
    ) -> dict[str, StagedSequence]:
        """Write the chunk's pose and context sequences as deterministic assets.

        One asset per logical sequence input, containing **every** frame of the
        chunk in order. The pose JSON is written alongside as an audit trail; it
        is not what the workflow consumes.
        """
        assert self._contract is not None
        staging = context.candidate_dir / "comfy_inputs" / f"chunk_{request.chunk_index:04d}"
        staging.mkdir(parents=True, exist_ok=True)
        preview = PreviewSettings(
            width=context.candidate.width,
            height=context.candidate.height,
            fps=context.candidate.fps,
            draw_labels=False,
        )

        poses_json = staging / "poses"
        poses_json.mkdir(parents=True, exist_ok=True)
        for pose in request.poses:
            save_pose_frame(poses_json / f"pose_{pose.frame_index:06d}.json", pose)

        staged: dict[str, StagedSequence] = {}
        staged["pose_sequence"] = self._stage_one(
            context,
            request,
            logical_name="pose_sequence",
            directory=staging / "pose_sequence",
            prefix="pose",
            frames=[
                (pose.frame_index, render_preview_frame(pose, preview)) for pose in request.poses
            ],
        )
        if request.context_frames and self._contract.declares("context_sequence"):
            staged["context_sequence"] = self._stage_one(
                context,
                request,
                logical_name="context_sequence",
                directory=staging / "context_sequence",
                prefix="context",
                frames=list(
                    zip(request.context_frame_indices, request.context_frames, strict=True)
                ),
            )
        return staged

    def _stage_one(
        self,
        context: AnimatorContext,
        request: AnimationChunkRequest,
        *,
        logical_name: str,
        directory: Path,
        prefix: str,
        frames: list[tuple[int, np.ndarray]],
    ) -> StagedSequence:
        assert self._contract is not None
        sequence = stage_frame_sequence(directory, frames, logical_name=logical_name, prefix=prefix)
        binding = self._contract.binding(logical_name)
        if binding is not None and binding.kind == SEQUENCE_KIND_VIDEO:
            sequence = encode_sequence_video(
                sequence,
                directory.parent / f"{prefix}_sequence.mp4",
                fps=context.candidate.fps,
                ffmpeg_binary=self._config.runtime.ffmpeg_binary,
            )
        return sequence

    def _upload_subfolder(self, context: AnimatorContext, chunk_index: int, leaf: str) -> str:
        return f"{UPLOAD_SUBFOLDER}/{context.candidate.id}/chunk_{chunk_index:04d}/{leaf}"

    def _sequence_value(
        self,
        context: AnimatorContext,
        request: AnimationChunkRequest,
        sequence: StagedSequence,
        binding: WorkflowBinding,
    ) -> str:
        """Stage a sequence into ComfyUI and return the value its node reads.

        A directory binding gets an input-relative folder name; every frame in it
        is uploaded first, in order. A video binding gets the uploaded file name.
        With ``upload_inputs`` off, both get the local path, which only works
        when ComfyUI runs on this machine -- which it must.
        """
        subfolder = self._upload_subfolder(context, request.chunk_index, sequence.logical_name)
        if binding.kind == SEQUENCE_KIND_VIDEO:
            video = sequence.video_path
            if video is None:  # pragma: no cover - guarded by _stage_one
                raise BackendError(
                    "A sequence_video binding has no encoded video",
                    logical_name=sequence.logical_name,
                )
            if not self._comfy_config.upload_inputs:
                return str(video)
            return self._client.upload_file(video, subfolder=subfolder)
        if not self._comfy_config.upload_inputs:
            return str(sequence.directory)
        for path in sequence.files:
            self._client.upload_file(path, subfolder=subfolder)
        return subfolder

    # -- value mapping ----------------------------------------------------
    def _build_values(
        self,
        context: AnimatorContext,
        request: AnimationChunkRequest,
        staged: dict[str, StagedSequence],
    ) -> dict[str, Any]:
        """Map candidate data onto the contract's declared logical inputs."""
        assert self._contract is not None
        declared = self._contract.declared_inputs()
        candidates: dict[str, Any] = {}

        for logical_name, sequence in staged.items():
            binding = self._contract.binding(logical_name)
            if binding is None:
                continue
            candidates[logical_name] = self._sequence_value(context, request, sequence, binding)

        # A LAST_FRAME-style contract still gets the frame it asks for, taken
        # from the tail of the same gathered context -- never a different frame.
        if request.context_frames:
            for single in ("context_frame", "source_frame"):
                if single in declared:
                    tail = (
                        context.candidate_dir
                        / "comfy_inputs"
                        / f"chunk_{request.chunk_index:04d}"
                        / "context_last.png"
                    )
                    write_frame(tail, request.context_frames[-1])
                    candidates[single] = (
                        self._client.upload_file(
                            tail,
                            subfolder=self._upload_subfolder(
                                context, request.chunk_index, "context_frame"
                            ),
                        )
                        if self._comfy_config.upload_inputs
                        else str(tail)
                    )

        hero_slots = ("hero_reference", "hero_reference_back", "hero_reference_side")
        for slot, path in zip(hero_slots, context.hero_image_paths, strict=False):
            if slot in declared and path.is_file():
                candidates[slot] = (
                    self._client.upload_file(
                        path,
                        subfolder=self._upload_subfolder(context, request.chunk_index, "hero"),
                    )
                    if self._comfy_config.upload_inputs
                    else str(path)
                )
        # Legacy contracts reused `garment_reference` for the hero image before
        # `hero_reference` existed; keep them working rather than silently
        # leaving the character reference unbound.
        if "garment_reference" in declared and "hero_reference" in candidates:
            candidates["garment_reference"] = candidates["hero_reference"]

        candidates.update(
            {
                "seed": request.seed,
                "width": context.candidate.width,
                "height": context.candidate.height,
                "fps": context.candidate.fps,
                "batch_size": request.frame_count,
                "frame_count": request.frame_count,
                "start_frame": request.start_frame,
                "frame_index": request.start_frame,
                "output_prefix": f"{context.candidate.id}_chunk{request.chunk_index:04d}",
                "steps": int(request.settings.get("steps", 20)),
                "denoise": float(request.settings.get("denoise", 1.0)),
                "guidance_scale": float(request.settings.get("guidance_scale", 5.0)),
                "prompt": str(request.settings.get("prompt", "")),
                "negative_prompt": str(request.settings.get("negative_prompt", "")),
            }
        )
        return {name: value for name, value in candidates.items() if name in declared}


__all__ = ["BACKEND_NAME", "BACKEND_VERSION", "UPLOAD_SUBFOLDER", "ComfyUIAnimatorBackend"]
