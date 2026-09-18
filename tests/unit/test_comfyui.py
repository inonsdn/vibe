"""ComfyUI backend: requirement 12 — remote URLs are rejected by default.

Every test here uses an in-process ``httpx.MockTransport``: no socket is opened,
no ComfyUI needs to be running, and the conftest network guard would fail the
test if one were.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.backends.comfyui.client import ComfyUIClient, assert_local_endpoint
from app.backends.comfyui.workflow import (
    WorkflowBinding,
    find_contract,
    load_contract,
    load_workflow,
)
from app.core.config import REPO_ROOT, ComfyUIConfig, load_config
from app.core.errors import (
    BackendError,
    BackendUnavailableError,
    ConfigError,
    OfflinePolicyError,
    ValidationError,
)

WORKFLOWS_DIR = REPO_ROOT / "workflows" / "comfyui"

REMOTE_URLS = [
    "http://example.com:8188",
    "https://comfy.example.org",
    "http://10.0.0.5:8188",
    "http://192.168.1.50:8188",
    "http://172.16.4.4:8188",
    "http://8.8.8.8",
    "http://comfy.internal.lan:8188",
    "http://[2001:db8::1]:8188",
]

LOCAL_URLS = [
    "http://127.0.0.1:8188",
    "http://localhost:8188",
    "http://127.0.0.2:8188",
    "http://[::1]:8188",
]


# -- endpoint policy (requirement 12) -------------------------------------
@pytest.mark.parametrize("url", REMOTE_URLS)
def test_remote_urls_are_rejected_by_default(url: str) -> None:
    config = ComfyUIConfig(base_url=url)
    with pytest.raises(OfflinePolicyError) as exc:
        assert_local_endpoint(url, config)
    assert "non-local" in exc.value.message.lower() or "allowlist" in exc.value.message.lower()


@pytest.mark.parametrize("url", LOCAL_URLS)
def test_loopback_urls_are_accepted(url: str) -> None:
    assert assert_local_endpoint(url, ComfyUIConfig(base_url=url)) == url.rstrip("/")


def test_remote_url_needs_both_allow_remote_and_allowlist() -> None:
    url = "http://192.168.1.50:8188"
    only_flag = ComfyUIConfig(base_url=url, allow_remote=True)
    with pytest.raises(OfflinePolicyError, match="allowlist"):
        assert_local_endpoint(url, only_flag)

    both = ComfyUIConfig(base_url=url, allow_remote=True, allowed_hosts=("192.168.1.50",))
    assert assert_local_endpoint(url, both) == url


def test_non_http_scheme_is_rejected() -> None:
    config = ComfyUIConfig()
    with pytest.raises(OfflinePolicyError, match="http"):
        assert_local_endpoint("ws://127.0.0.1:8188", config)


def test_client_construction_refuses_a_remote_endpoint() -> None:
    """The refusal happens in the constructor, before any request is possible."""
    with pytest.raises(OfflinePolicyError):
        ComfyUIClient(ComfyUIConfig(base_url="http://example.com:8188"))


def test_backend_construction_refuses_a_remote_endpoint() -> None:
    from app.backends.comfyui.backend import ComfyUIBackend

    config = load_config(
        overrides={"comfyui": {"base_url": "http://example.com:8188"}}, use_env=False
    )
    with pytest.raises(OfflinePolicyError):
        ComfyUIBackend(config)


def test_default_config_is_loopback() -> None:
    config = load_config(use_env=False)
    assert config.comfyui.allow_remote is False
    assert assert_local_endpoint(config.comfyui.base_url, config.comfyui)


# -- client behaviour (mock transport, no sockets) ------------------------
def client_with(handler: Any, **config_kwargs: Any) -> ComfyUIClient:
    config = ComfyUIConfig(**config_kwargs)
    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(base_url=config.base_url, transport=transport, timeout=5.0)
    return ComfyUIClient(config, client=http_client)


def test_health_reports_stats() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/system_stats"
        return httpx.Response(
            200,
            json={
                "system": {"comfyui_version": "0.3.0", "python_version": "3.11"},
                "devices": [
                    {"name": "Test GPU", "vram_total": 8 * 1024**3, "vram_free": 7 * 1024**3}
                ],
            },
        )

    with client_with(handler) as client:
        assert client.is_healthy()
        assert client.health()["system"]["comfyui_version"] == "0.3.0"


def test_unreachable_comfyui_gives_an_actionable_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with client_with(handler) as client, pytest.raises(BackendUnavailableError) as exc:
        client.health()
    assert "ComfyUI" in exc.value.message
    assert "hint" in exc.value.details


def test_submit_returns_prompt_identifiers() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "prompt" in body and "client_id" in body
        return httpx.Response(200, json={"prompt_id": "abc123", "number": 4})

    with client_with(handler) as client:
        handle = client.submit({"1": {"class_type": "X"}})
        assert handle.prompt_id == "abc123"
        assert handle.number == 4


def test_rejected_workflow_reports_node_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"error": "invalid prompt", "node_errors": {"3": "missing input"}},
        )

    with client_with(handler) as client:
        with pytest.raises(BackendError) as exc:
            client.submit({"1": {}})
        assert exc.value.details["node_errors"] == {"3": "missing input"}


def test_wait_polls_until_history_appears() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/history"):
            calls["count"] += 1
            if calls["count"] < 3:
                return httpx.Response(200, json={})
            return httpx.Response(
                200,
                json={
                    "p1": {
                        "status": {"completed": True},
                        "outputs": {"6": {"images": [{"filename": "out.png", "type": "output"}]}},
                    }
                },
            )
        if request.url.path == "/queue":
            return httpx.Response(200, json={"queue_running": [[0, "p1"]], "queue_pending": []})
        raise AssertionError(f"unexpected path {request.url.path}")

    with client_with(handler, poll_interval_s=0.0) as client:
        outcome = client.wait("p1", sleep=lambda _s: None)
    assert outcome.completed
    assert outcome.images[0]["filename"] == "out.png"


def test_wait_times_out_with_an_actionable_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/history"):
            return httpx.Response(200, json={})
        return httpx.Response(200, json={"queue_running": [[0, "p1"]], "queue_pending": []})

    with (
        client_with(handler, poll_interval_s=0.0, job_timeout_s=0.0) as client,
        pytest.raises(BackendError, match="Timed out"),
    ):
        client.wait("p1", sleep=lambda _s: None)


def test_vanished_prompt_is_reported_as_incomplete() -> None:
    """A prompt that disappears must not be mistaken for success."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/history"):
            return httpx.Response(200, json={})
        return httpx.Response(200, json={"queue_running": [], "queue_pending": []})

    with client_with(handler, poll_interval_s=0.0) as client:
        outcome = client.wait("gone", sleep=lambda _s: None)
    assert not outcome.completed
    assert "disappeared" in outcome.status["error"]


def test_missing_node_types_are_listed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"LoadImage": {}, "SaveImage": {}})

    graph = {
        "1": {"class_type": "LoadImage"},
        "2": {"class_type": "SomeExoticCustomNode"},
        "3": {"class_type": "SaveImage"},
    }
    with client_with(handler) as client:
        assert client.missing_node_types(graph) == ["SomeExoticCustomNode"]


def test_fetch_output_writes_the_file(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/view"
        return httpx.Response(200, content=b"PNGDATA")

    with client_with(handler) as client:
        target = client.fetch_output(
            {"filename": "out.png", "subfolder": "", "type": "output"},
            tmp_path / "frame.png",
        )
    assert target.read_bytes() == b"PNGDATA"


# -- workflow contract ----------------------------------------------------
def test_placeholder_workflow_satisfies_its_contract() -> None:
    contract = load_contract(find_contract(WORKFLOWS_DIR, "garment_replace_placeholder"))
    graph = load_workflow(WORKFLOWS_DIR / contract.workflow_file)
    assert contract.validate_against(graph) == []


def test_placeholder_workflow_needs_no_model_nodes() -> None:
    """Requirement: the example must not need real model nodes to be testable."""
    contract = load_contract(find_contract(WORKFLOWS_DIR, "garment_replace_placeholder"))
    graph = load_workflow(WORKFLOWS_DIR / contract.workflow_file)
    assert contract.requires_model_nodes is False
    node_types = {node["class_type"] for node in graph.values()}
    core_only = {
        "LoadImage",
        "LoadImageMask",
        "ImageScale",
        "ImageCompositeMasked",
        "SaveImage",
    }
    assert node_types <= core_only, f"unexpected node types: {node_types - core_only}"


def test_bindings_resolve_by_title_not_by_node_id() -> None:
    """Renumbering the graph must not break the integration."""
    contract = load_contract(find_contract(WORKFLOWS_DIR, "garment_replace_placeholder"))
    graph = load_workflow(WORKFLOWS_DIR / contract.workflow_file)
    renumbered = {f"n{key}": value for key, value in graph.items()}
    # Re-point internal links so the graph stays coherent.
    for node in renumbered.values():
        for name, value in list(node.get("inputs", {}).items()):
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                node["inputs"][name] = [f"n{value[0]}", value[1]]
    assert contract.validate_against(renumbered) == []


def test_apply_writes_values_into_the_bound_nodes() -> None:
    contract = load_contract(find_contract(WORKFLOWS_DIR, "garment_replace_placeholder"))
    graph = load_workflow(WORKFLOWS_DIR / contract.workflow_file)
    patched = contract.apply(
        graph,
        {
            "source_frame": "src.png",
            "garment_reference": "grm.png",
            "mask": "mask.png",
            "width": 1080,
            "height": 1920,
            "output_prefix": "job/frame",
        },
    )
    assert patched["1"]["inputs"]["image"] == "src.png"
    assert patched["3"]["inputs"]["image"] == "mask.png"
    assert patched["4"]["inputs"]["width"] == 1080
    assert patched["6"]["inputs"]["filename_prefix"] == "job/frame"
    # The original graph is untouched.
    assert graph["1"]["inputs"]["image"] == "placeholder_source.png"


def test_apply_requires_every_required_input() -> None:
    contract = load_contract(find_contract(WORKFLOWS_DIR, "garment_replace_placeholder"))
    graph = load_workflow(WORKFLOWS_DIR / contract.workflow_file)
    with pytest.raises(ValidationError, match="Missing required workflow input"):
        contract.apply(graph, {"source_frame": "a.png"})


def test_apply_rejects_undeclared_logical_inputs() -> None:
    contract = load_contract(find_contract(WORKFLOWS_DIR, "garment_replace_placeholder"))
    graph = load_workflow(WORKFLOWS_DIR / contract.workflow_file)
    with pytest.raises(ValidationError, match="does not declare"):
        contract.apply(
            graph,
            {
                "source_frame": "a.png",
                "garment_reference": "b.png",
                "mask": "c.png",
                "width": 1,
                "height": 2,
                "pose": "p.json",
            },
        )


def test_missing_node_title_is_an_actionable_error() -> None:
    binding = WorkflowBinding(logical_name="seed", input_name="seed", node_title="NOT_PRESENT")
    with pytest.raises(ValidationError, match="bound title"):
        binding.locate({"1": {"class_type": "X", "_meta": {"title": "OTHER"}}})


def test_ambiguous_node_title_is_rejected() -> None:
    binding = WorkflowBinding(logical_name="seed", input_name="seed", node_title="DUP")
    graph = {
        "1": {"class_type": "X", "_meta": {"title": "DUP"}},
        "2": {"class_type": "Y", "_meta": {"title": "DUP"}},
    }
    with pytest.raises(ValidationError, match="ambiguous"):
        binding.locate(graph)


def test_editor_format_workflow_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "editor.json"
    path.write_text(json.dumps({"nodes": [], "links": []}), encoding="utf-8")
    with pytest.raises(ConfigError, match="API format"):
        load_workflow(path)


def test_contract_with_unknown_logical_input_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.contract.yaml"
    path.write_text(
        "workflow_id: bad\nworkflow_file: bad.json\ninputs:\n"
        "  - logical_name: teleport_garment\n    node_title: X\n    input_name: y\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="cannot supply"):
        load_contract(path)


def test_missing_contract_lists_what_is_available(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as exc:
        find_contract(WORKFLOWS_DIR, "does_not_exist")
    assert "garment_replace_placeholder.contract.yaml" in exc.value.details["available"]


def test_contract_hash_is_stable() -> None:
    path = find_contract(WORKFLOWS_DIR, "garment_replace_placeholder")
    first = load_contract(path).contract_sha256()
    assert first == load_contract(path).contract_sha256()
    assert len(first) == 64
