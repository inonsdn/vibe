"""Offline guarantees: requirement 15 — no network, no cloud integrations."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from app.core.config import load_config
from app.offline.verify import (
    CLOUD_INTEGRATIONS,
    NETWORK_FEATURES,
    REQUIRED_EXECUTABLES,
    check_endpoints,
    check_executables,
    check_gpu,
    check_paths,
    render_text_report,
    verify_offline,
)
from tests.conftest import NetworkAccessAttempted


def test_no_cloud_integrations_are_declared() -> None:
    """A positive, testable statement: there are none, and none may be added."""
    assert CLOUD_INTEGRATIONS == ()
    assert NETWORK_FEATURES == ()


def test_network_guard_blocks_outbound_connections() -> None:
    """Proves the conftest guard is active for the whole suite."""
    with pytest.raises(NetworkAccessAttempted):
        socket.create_connection(("example.com", 80), timeout=1)
    with pytest.raises(NetworkAccessAttempted):
        socket.socket().connect(("1.1.1.1", 443))


def test_loopback_is_still_permitted() -> None:
    """The guard must not break in-process test servers."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    try:
        client = socket.socket()
        client.settimeout(2)
        client.connect(server.getsockname())
        client.close()
    finally:
        server.close()


def test_default_endpoints_are_all_local(config) -> None:
    for endpoint in check_endpoints(config):
        assert endpoint.is_local, f"{endpoint.name} is not local: {endpoint.url}"
        assert endpoint.allowed


def test_remote_comfyui_endpoint_is_reported_as_refused() -> None:
    config = load_config(
        overrides={"comfyui": {"base_url": "http://example.com:8188"}}, use_env=False
    )
    comfy = next(e for e in check_endpoints(config) if e.name == "comfyui")
    assert not comfy.is_local
    assert not comfy.allowed


def test_non_loopback_api_bind_requires_explicit_opt_in() -> None:
    config = load_config(overrides={"api": {"host": "0.0.0.0"}}, use_env=False)
    api = next(e for e in check_endpoints(config) if e.name == "api_bind")
    assert not api.is_local
    assert not api.allowed

    explicit = load_config(
        overrides={"api": {"host": "0.0.0.0", "allow_remote_bind": True}}, use_env=False
    )
    api2 = next(e for e in check_endpoints(explicit) if e.name == "api_bind")
    assert api2.allowed


def test_required_executables_are_reported() -> None:
    found, missing = check_executables()
    for name in REQUIRED_EXECUTABLES:
        assert name in found
    assert all(name in REQUIRED_EXECUTABLES for name in missing)


def test_paths_are_created_and_writable(config) -> None:
    results, problems = check_paths(config)
    assert problems == []
    assert results["data_root"]["writable"]
    for name in ("templates", "garments", "jobs", "exports", "logs", "db"):
        assert results[name]["writable"], name


def test_gpu_check_never_requires_cuda() -> None:
    """Requirement: report GPU info when available, without needing CUDA."""
    info = check_gpu()
    assert "cuda_available" in info
    assert isinstance(info["devices"], list)
    assert info["note"]


def test_verify_offline_reports_adapters_and_backends(config) -> None:
    report = verify_offline(config, probe_comfyui=False)
    assert report.ok, report.problems
    assert len(report.adapters) == 7
    assert all(entry["status"] == "not_implemented" for entry in report.adapters.values())
    assert "mock" in report.backends
    assert report.backends["mock"]["capabilities"]["requires_model_weights"] is False
    assert report.provenance["config_hash"]


def test_verify_offline_text_report_mentions_no_cloud(config) -> None:
    text = render_text_report(verify_offline(config, probe_comfyui=False))
    assert "Cloud integrations" in text
    assert "none" in text


def test_verify_offline_fails_when_a_remote_endpoint_is_configured(data_root: Path) -> None:
    config = load_config(
        overrides={
            "paths": {"data_root": str(data_root)},
            "comfyui": {"base_url": "http://example.com:8188"},
        },
        use_env=False,
    )
    report = verify_offline(config, probe_comfyui=False)
    assert not report.ok
    assert any("comfyui" in problem for problem in report.problems)
