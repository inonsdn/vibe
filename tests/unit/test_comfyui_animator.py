"""The ComfyUI animator's *sequence* contract, inspected on the wire.

Every test here drives a real ``ComfyUIAnimatorBackend`` against an in-process
``httpx.MockTransport``. Nothing is running, nothing is downloaded, no socket is
opened — but the workflow graph the backend would have submitted is captured and
examined, which is the only way to prove the plumbing represents a real
multi-frame animation workflow rather than a single still.

The failure these tests exist to prevent: staging 24 poses, uploading one, and
recording "24 frames, 16 context frames" in the manifest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from app.backends.animator.base import (
    AnimationChunkRequest,
    AnimatorContext,
    ContextMode,
)
from app.backends.animator.comfyui import UPLOAD_SUBFOLDER, ComfyUIAnimatorBackend
from app.backends.comfyui.client import ComfyUIClient
from app.backends.comfyui.sequence import MANIFEST_NAME, stage_frame_sequence
from app.core.config import REPO_ROOT, ComfyUIConfig
from app.core.errors import BackendError, ValidationError
from app.domain.master import MasterCandidate, MasterOrigin
from tests import motion_fixtures as mf

WORKFLOWS_DIR = REPO_ROOT / "workflows" / "comfyui"
WIDTH, HEIGHT, FPS = 90, 160, 30.0
CHUNK_START, CHUNK_END = 24, 36
CONTEXT_FRAMES = 4


class FakeComfy:
    """A ComfyUI that records everything it is handed."""

    def __init__(self, *, returned_images: int | None = None, duplicate: bool = False) -> None:
        self.uploads: list[tuple[str, str]] = []  # (subfolder, filename)
        self.graphs: list[dict[str, Any]] = []
        self.returned_images = returned_images
        self.duplicate = duplicate
        self._png = _png_bytes()

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/system_stats":
            return httpx.Response(200, json={"system": {"comfyui_version": "0.3.0"}})
        if path == "/object_info":
            return httpx.Response(
                200,
                json={
                    name: {}
                    for name in (
                        "LoadImagesFromDirectory",
                        "LoadImage",
                        "ImageScale",
                        "ImageBlend",
                        "RepeatImageBatch",
                        "SaveImage",
                    )
                },
            )
        if path == "/upload/image":
            content = request.content.decode("latin-1")
            subfolder = _multipart_field(content, "subfolder")
            filename = _multipart_filename(content)
            self.uploads.append((subfolder, filename))
            return httpx.Response(200, json={"name": filename, "subfolder": subfolder})
        if path == "/prompt":
            import json as _json

            self.graphs.append(_json.loads(request.content)["prompt"])
            return httpx.Response(200, json={"prompt_id": f"p{len(self.graphs)}", "number": 1})
        if path.startswith("/history/"):
            prompt_id = path.rsplit("/", 1)[-1]
            count = (
                self.returned_images
                if self.returned_images is not None
                else CHUNK_END - CHUNK_START
            )
            images = [
                {
                    "filename": f"out_{0 if self.duplicate else i:05d}.png",
                    "subfolder": "",
                    "type": "output",
                }
                for i in range(count)
            ]
            return httpx.Response(
                200,
                json={
                    prompt_id: {
                        "status": {"completed": True},
                        "outputs": {"7": {"images": images}},
                    }
                },
            )
        if path == "/view":
            return httpx.Response(200, content=self._png, headers={"content-type": "image/png"})
        if path == "/queue":
            return httpx.Response(200, json={"queue_running": [], "queue_pending": []})
        return httpx.Response(404, json={"error": path})  # pragma: no cover


def _png_bytes() -> bytes:
    import cv2

    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", frame)
    assert ok
    return bytes(buffer)


def _multipart_field(content: str, name: str) -> str:
    marker = f'name="{name}"'
    if marker not in content:
        return ""
    after = content.split(marker, 1)[1]
    body = after.split("\r\n\r\n", 1)[1]
    return body.split("\r\n", 1)[0]


def _multipart_filename(content: str) -> str:
    marker = 'filename="'
    if marker not in content:
        return ""  # pragma: no cover
    return content.split(marker, 1)[1].split('"', 1)[0]


@pytest.fixture
def backend_and_comfy(config, tmp_path):
    comfy = FakeComfy()
    comfy_config = ComfyUIConfig(upload_inputs=True)
    http = httpx.Client(
        base_url=comfy_config.base_url,
        transport=httpx.MockTransport(comfy.handler),
        timeout=5.0,
        trust_env=False,
    )
    client = ComfyUIClient(comfy_config, client=http)
    backend = ComfyUIAnimatorBackend(
        config.model_copy(update={"comfyui": comfy_config}), client=client
    )
    yield backend, comfy
    backend.close()


def make_animator_context(context, tmp_path) -> AnimatorContext:
    hero = mf.make_hero(context)
    composition = _composition(context)
    profile = context.repos.skeleton_profiles.get(
        composition.skeleton_profile_id, composition.skeleton_profile_version
    )
    candidate_dir = tmp_path / "candidate"
    frames_dir = candidate_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    candidate = MasterCandidate(
        id="mst_wire",
        display_name="Wire test",
        origin=MasterOrigin.SYNTHETIC,
        composition_id=composition.id,
        composition_version=composition.version,
        hero_character_id=hero.id,
        hero_character_version=hero.version,
        backend_name="comfyui",
        workflow_id="character_animate_placeholder",
        seed=11,
        frame_count=CHUNK_END,
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        frames_dir="masters/mst_wire/frames",
    )
    return AnimatorContext(
        candidate=candidate,
        composition=composition,
        hero=hero,
        profile=profile,
        config=context.config,
        candidate_dir=candidate_dir,
        frames_dir=frames_dir,
        pose_dir=candidate_dir / "poses",
        hero_image_paths=[context.absolute(p) for p in hero.reference_images],
        workflow_path=WORKFLOWS_DIR / "character_animate_placeholder.json",
    )


def _composition(context):
    from app.pipeline.motion_compose import ComposeOptions, SegmentSpec, compose_motion

    a, _ = mf.make_motion_pair(context, a_range=(0, CHUNK_END), b_range=(0, CHUNK_END))
    return compose_motion(
        context,
        ComposeOptions(
            display_name="wire",
            segments=[SegmentSpec(motion_source_id=a.source.id, exposed_views=["front"])],
            joins=[],
            make_preview=False,
        ),
    ).composition


def make_request(context, animator_context, *, context_frames: int = CONTEXT_FRAMES):
    from app.motion.pose_format import load_pose_sequence

    poses = load_pose_sequence(context.absolute(animator_context.composition.composed_pose_dir))
    by_index = {pose.frame_index: pose for pose in poses}
    first_context = CHUNK_START - context_frames
    return AnimationChunkRequest(
        chunk_index=1,
        start_frame=CHUNK_START,
        end_frame=CHUNK_END,
        poses=[by_index[i] for i in range(CHUNK_START, CHUNK_END)],
        seed=4242,
        context_frames=[
            np.full((HEIGHT, WIDTH, 3), i % 251, dtype=np.uint8)
            for i in range(first_context, CHUNK_START)
        ],
        context_poses=[by_index[i] for i in range(first_context, CHUNK_START)],
    )


# ---------------------------------------------------------------------------
# capabilities
# ---------------------------------------------------------------------------
def test_the_backend_declares_a_sequence_context_mode(backend_and_comfy) -> None:
    backend, _ = backend_and_comfy
    capabilities = backend.capabilities()
    assert capabilities.context_mode is ContextMode.SEQUENCE
    assert capabilities.context_frames_for(16) == 16
    assert capabilities.as_dict()["context_mode"] == "sequence"


@pytest.mark.parametrize(
    ("mode", "configured", "expected"),
    [
        (ContextMode.NONE, 16, 0),
        (ContextMode.LAST_FRAME, 16, 1),
        (ContextMode.LAST_FRAME, 0, 0),
        (ContextMode.SEQUENCE, 16, 16),
    ],
)
def test_context_frames_for_never_overstates(mode, configured, expected) -> None:
    from app.backends.animator.base import AnimatorCapabilities

    capabilities = AnimatorCapabilities(
        name="x",
        version="1",
        requires_gpu=False,
        requires_model_weights=False,
        deterministic=True,
        context_mode=mode,
        max_chunk_frames=64,
        recommended_chunk_frames=24,
        recommended_overlap_frames=configured,
    )
    assert capabilities.context_frames_for(configured) == expected
    assert capabilities.supports_context_frames is (mode is not ContextMode.NONE)


# ---------------------------------------------------------------------------
# what actually goes over the wire
# ---------------------------------------------------------------------------
@pytest.fixture
def submitted(context, tmp_path, backend_and_comfy):
    backend, comfy = backend_and_comfy
    animator_context = make_animator_context(context, tmp_path)
    request = make_request(context, animator_context)
    backend.prepare(animator_context)
    result = backend.animate_chunk(animator_context, request)
    return backend, comfy, animator_context, request, result


def test_every_pose_in_the_chunk_is_uploaded_in_order(submitted) -> None:
    _, comfy, _, request, _ = submitted
    pose_uploads = [name for sub, name in comfy.uploads if sub.endswith("/pose_sequence")]
    assert len(pose_uploads) == request.frame_count
    # Ordinal naming is what makes a directory loader read them in frame order.
    assert pose_uploads == [f"pose_{i:05d}.png" for i in range(request.frame_count)]
    assert len(set(pose_uploads)) == len(pose_uploads), "no frame uploaded twice"


def test_the_pose_manifest_maps_ordinals_back_to_absolute_frames(submitted) -> None:
    import json

    _, _, animator_context, request, _ = submitted
    manifest = (
        animator_context.candidate_dir
        / "comfy_inputs"
        / f"chunk_{request.chunk_index:04d}"
        / "pose_sequence"
        / MANIFEST_NAME
    )
    payload = json.loads(manifest.read_text())
    assert payload["frame_indices"] == request.frame_indices
    assert payload["first_frame"] == CHUNK_START
    assert payload["last_frame"] == CHUNK_END - 1
    assert payload["count"] == request.frame_count


def test_batch_size_is_actually_bound_into_the_graph(submitted) -> None:
    """It used to be computed and then filtered out for not being declared."""
    _, comfy, _, request, _ = submitted
    graph = comfy.graphs[-1]
    batch_node = _node_titled(graph, "HERO_BATCH")
    assert batch_node["inputs"]["amount"] == request.frame_count

    pose_node = _node_titled(graph, "POSE_SEQUENCE")
    assert pose_node["inputs"]["image_load_cap"] == request.frame_count
    assert pose_node["inputs"]["directory"].endswith("/pose_sequence")


def test_the_context_sequence_is_bound_not_just_its_last_frame(submitted) -> None:
    _, comfy, _, request, result = submitted
    graph = comfy.graphs[-1]
    context_node = _node_titled(graph, "CONTEXT_SEQUENCE")
    assert context_node["inputs"]["directory"].endswith("/context_sequence")

    uploaded = [name for sub, name in comfy.uploads if sub.endswith("/context_sequence")]
    assert uploaded == [f"context_{i:05d}.png" for i in range(CONTEXT_FRAMES)]
    assert result.backend_metadata["context_frames_used"] == CONTEXT_FRAMES
    assert result.backend_metadata["pose_frames_submitted"] == request.frame_count
    assert result.backend_metadata["context_mode"] == "sequence"


def test_uploads_land_under_a_per_candidate_subfolder(submitted) -> None:
    _, comfy, animator_context, _request, _ = submitted
    expected = f"{UPLOAD_SUBFOLDER}/{animator_context.candidate.id}/chunk_0001"
    assert comfy.uploads, "nothing was staged"
    for subfolder, _name in comfy.uploads:
        assert subfolder.startswith(expected), subfolder


def test_returned_frames_are_mapped_to_absolute_indices(submitted) -> None:
    _, _, _, request, result = submitted
    assert sorted(result.frames) == request.frame_indices


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------
def test_a_contract_without_a_pose_sequence_is_refused(context, tmp_path, backend_and_comfy):
    """A still-image contract cannot express a chunk, and saying so beats
    animating one pose for twelve frames."""
    from app.backends.comfyui.workflow import WorkflowBinding, WorkflowContract

    backend, _ = backend_and_comfy
    animator_context = make_animator_context(context, tmp_path)
    request = make_request(context, animator_context)
    backend.prepare(animator_context)
    backend._contract = WorkflowContract(
        workflow_id="stills_only",
        description="",
        workflow_file="character_animate_placeholder.json",
        bindings=(
            WorkflowBinding(
                logical_name="pose",
                input_name="image",
                node_title="POSE_SEQUENCE",
                kind="image_upload",
            ),
        ),
    )
    with pytest.raises(BackendError, match="does not declare"):
        backend.animate_chunk(animator_context, request)


def test_a_contract_with_nowhere_for_context_is_refused(context, tmp_path, backend_and_comfy):
    from app.backends.comfyui.workflow import load_contract

    backend, _ = backend_and_comfy
    animator_context = make_animator_context(context, tmp_path)
    request = make_request(context, animator_context)
    backend.prepare(animator_context)
    contract = load_contract(WORKFLOWS_DIR / "character_animate_placeholder.contract.yaml")
    backend._contract = _contract_without(contract, "context_sequence")

    with pytest.raises(BackendError, match="nowhere to put"):
        backend.animate_chunk(animator_context, request)


def _contract_without(contract, logical_name):
    """The same contract with one binding removed."""
    import dataclasses

    return dataclasses.replace(
        contract,
        bindings=tuple(b for b in contract.bindings if b.logical_name != logical_name),
    )


def test_a_short_output_is_refused(context, tmp_path, config) -> None:
    comfy = FakeComfy(returned_images=3)
    backend = _backend_for(config, comfy)
    animator_context = make_animator_context(context, tmp_path)
    request = make_request(context, animator_context)
    backend.prepare(animator_context)
    with pytest.raises(BackendError, match="different number of frames"):
        backend.animate_chunk(animator_context, request)
    backend.close()


def test_duplicated_outputs_are_refused(context, tmp_path, config) -> None:
    comfy = FakeComfy(duplicate=True)
    backend = _backend_for(config, comfy)
    animator_context = make_animator_context(context, tmp_path)
    request = make_request(context, animator_context)
    backend.prepare(animator_context)
    with pytest.raises(BackendError, match="more than once"):
        backend.animate_chunk(animator_context, request)
    backend.close()


def _backend_for(config, comfy: FakeComfy) -> ComfyUIAnimatorBackend:
    comfy_config = ComfyUIConfig(upload_inputs=True)
    http = httpx.Client(
        base_url=comfy_config.base_url,
        transport=httpx.MockTransport(comfy.handler),
        timeout=5.0,
        trust_env=False,
    )
    return ComfyUIAnimatorBackend(
        config.model_copy(update={"comfyui": comfy_config}),
        client=ComfyUIClient(comfy_config, client=http),
    )


def _node_titled(graph: dict[str, Any], title: str) -> dict[str, Any]:
    for node in graph.values():
        if isinstance(node, dict) and node.get("_meta", {}).get("title") == title:
            return node
    raise AssertionError(f"no node titled {title!r} in the submitted graph")


# ---------------------------------------------------------------------------
# request-level structural validation
# ---------------------------------------------------------------------------
def test_a_chunk_whose_poses_do_not_cover_it_is_rejected() -> None:
    from app.motion.pose_format import PoseFrame

    poses = [PoseFrame(frame_index=i, timestamp_s=i / 30.0) for i in (0, 1, 3)]
    request = AnimationChunkRequest(chunk_index=0, start_frame=0, end_frame=3, poses=poses, seed=1)
    with pytest.raises(ValidationError, match="does not cover"):
        request.validate()


def test_context_poses_must_immediately_precede_the_chunk() -> None:
    from app.motion.pose_format import PoseFrame

    request = AnimationChunkRequest(
        chunk_index=1,
        start_frame=10,
        end_frame=12,
        poses=[PoseFrame(frame_index=i, timestamp_s=i / 30.0) for i in (10, 11)],
        seed=1,
        context_frames=[np.zeros((2, 2, 3), dtype=np.uint8)] * 2,
        context_poses=[PoseFrame(frame_index=i, timestamp_s=i / 30.0) for i in (4, 5)],
    )
    with pytest.raises(ValidationError, match="immediately preceding"):
        request.validate()


# ---------------------------------------------------------------------------
# the staging primitive
# ---------------------------------------------------------------------------
def test_staging_removes_stale_frames_from_a_previous_run(tmp_path: Path) -> None:
    directory = tmp_path / "seq"
    long_run = [(i, np.zeros((4, 4, 3), dtype=np.uint8)) for i in range(5)]
    stage_frame_sequence(directory, long_run, logical_name="pose_sequence", prefix="pose")
    short_run = long_run[:2]
    staged = stage_frame_sequence(directory, short_run, logical_name="pose_sequence", prefix="pose")
    on_disk = sorted(p.name for p in directory.glob("pose_*.png"))
    assert on_disk == ["pose_00000.png", "pose_00001.png"]
    assert staged.count == 2


def test_staging_an_empty_sequence_is_refused(tmp_path: Path) -> None:
    with pytest.raises(BackendError, match="empty sequence"):
        stage_frame_sequence(tmp_path / "seq", [], logical_name="pose_sequence")


def test_the_client_refuses_to_upload_a_type_it_cannot_stage(tmp_path: Path, config) -> None:
    comfy = FakeComfy()
    comfy_config = ComfyUIConfig()
    http = httpx.Client(
        base_url=comfy_config.base_url,
        transport=httpx.MockTransport(comfy.handler),
        timeout=5.0,
        trust_env=False,
    )
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"not an image")
    with (
        ComfyUIClient(comfy_config, client=http) as client,
        pytest.raises(BackendError, match="does not stage"),
    ):
        client.upload_file(weights)


def test_a_video_sequence_binding_uploads_one_file(tmp_path: Path) -> None:
    """The alternative protocol: one visually lossless clip instead of a run."""
    from app.backends.comfyui.sequence import encode_sequence_video
    from tests.conftest import ffmpeg_available

    if not ffmpeg_available():
        pytest.skip("ffmpeg is not installed")

    directory = tmp_path / "seq"
    frames = [(i, np.full((16, 16, 3), i * 8, dtype=np.uint8)) for i in range(6)]
    staged = stage_frame_sequence(directory, frames, logical_name="pose_sequence", prefix="pose")
    encoded = encode_sequence_video(staged, tmp_path / "pose_sequence.mp4", fps=30.0)
    assert encoded.video_path is not None and encoded.video_path.is_file()
    assert encoded.frame_indices == tuple(range(6))
    assert encoded.kind == "sequence_video"
