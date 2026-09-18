"""Environment-level regressions.

Requirement 12: dependency installation uses the tested versions.
Requirement 13: proxy environment variables cannot affect localhost behaviour.
Requirement 14: the network guard blocks DNS as well as socket connections.
"""

from __future__ import annotations

import json
import re
import socket
from importlib import metadata
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from app.api.deps import close as close_api_context
from app.api.main import create_app
from app.backends.comfyui.client import ComfyUIClient
from app.backends.registry import available_backends, create_backend
from app.cli.main import app as cli_app
from app.core.config import REPO_ROOT, ComfyUIConfig
from tests.conftest import NetworkAccessAttempted

#: Values chosen to break loudly if they are ever honoured: a SOCKS scheme
#: requires the optional `socksio` package, and the hosts do not resolve.
HOSTILE_PROXIES = {
    "HTTP_PROXY": "http://proxy.invalid:9",
    "HTTPS_PROXY": "http://proxy.invalid:9",
    "ALL_PROXY": "socks5://proxy.invalid:1080",
    "http_proxy": "http://proxy.invalid:9",
    "https_proxy": "http://proxy.invalid:9",
    "all_proxy": "socks5://proxy.invalid:1080",
    "NO_PROXY": "",
    "no_proxy": "",
}


@pytest.fixture
def hostile_proxy_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Set every proxy variable to something that cannot possibly work."""
    for name, value in HOSTILE_PROXIES.items():
        monkeypatch.setenv(name, value)
    return dict(HOSTILE_PROXIES)


# ---------------------------------------------------------------------------
# requirement 13: proxy variables must not reach localhost traffic
# ---------------------------------------------------------------------------
def test_internally_owned_client_ignores_environment(hostile_proxy_env) -> None:
    """The client we construct must not trust the environment at all."""
    client = ComfyUIClient(ComfyUIConfig())
    try:
        assert client._client.trust_env is False
    finally:
        client.close()


def test_socks_proxy_variable_does_not_require_socksio(hostile_proxy_env) -> None:
    """A SOCKS variable must not turn `socksio` into a hard dependency.

    httpx raises ImportError when it honours a socks5:// proxy without socksio
    installed, so simply constructing the client proves the variable is ignored.
    """
    with pytest.raises(metadata.PackageNotFoundError):
        metadata.version("socksio")  # precondition: the package really is absent

    client = ComfyUIClient(ComfyUIConfig())
    client.close()


def test_backend_capability_inspection_succeeds_under_proxies(config, hostile_proxy_env) -> None:
    for name in available_backends():
        backend = create_backend(name, config)
        try:
            capabilities = backend.capabilities().as_dict()
            assert capabilities["name"] == name
        finally:
            backend.close()


def test_animator_capability_inspection_succeeds_under_proxies(config, hostile_proxy_env) -> None:
    from app.backends.animator.registry import available_animators, create_animator

    for name in available_animators():
        backend = create_animator(name, config)
        try:
            assert backend.capabilities().as_dict()["name"] == name
        finally:
            backend.close()


def test_api_backends_endpoint_succeeds_under_proxies(config, hostile_proxy_env) -> None:
    """The server side must not consult the environment for a loopback probe."""
    from fastapi.testclient import TestClient

    application = create_app(config, configure_logs=False)
    try:
        with TestClient(application) as client:
            response = client.get("/jobs/backends")
            assert response.status_code == 200
            payload = response.json()["backends"]
            assert payload["mock"]["capabilities"]["requires_model_weights"] is False
            # Reported unavailable because nothing is listening -- not because a
            # proxy variable broke the client.
            assert payload["comfyui"]["health"]["healthy"] is False
            detail = payload["comfyui"]["health"]["detail"]
            assert "socksio" not in detail and "proxy" not in detail.lower()
    finally:
        close_api_context()


def test_master_animators_endpoint_succeeds_under_proxies(config, hostile_proxy_env) -> None:
    from fastapi.testclient import TestClient

    application = create_app(config, configure_logs=False)
    try:
        with TestClient(application) as client:
            response = client.get("/master/animators")
            assert response.status_code == 200
            animators = response.json()["animators"]
            assert animators["mock"]["health"]["healthy"] is True
    finally:
        close_api_context()


def test_cli_job_backends_succeeds_under_proxies(data_root: Path, hostile_proxy_env) -> None:
    runner = CliRunner()
    result = runner.invoke(cli_app, ["--data-root", str(data_root), "--json", "job", "backends"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["mock"]["health"]["healthy"] is True


def test_cli_master_animators_succeeds_under_proxies(data_root: Path, hostile_proxy_env) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli_app, ["--data-root", str(data_root), "--json", "doctor", "--no-probe-comfyui"]
    )
    assert result.exit_code == 0, result.output


def test_mock_transport_is_used_under_proxies(hostile_proxy_env) -> None:
    """A supplied mock transport must still serve localhost requests."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"system": {"comfyui_version": "test"}})

    config = ComfyUIConfig()
    http_client = httpx.Client(
        base_url=config.base_url,
        transport=httpx.MockTransport(handler),
        trust_env=False,
    )
    with ComfyUIClient(config, client=http_client) as client:
        assert client.is_healthy()
    assert seen and seen[0].startswith("http://127.0.0.1:8188/")


def test_a_trust_env_client_really_would_break(hostile_proxy_env) -> None:
    """The regression, demonstrated.

    With ALL_PROXY set to a socks5 URL, httpx refuses to build a client that
    trusts the environment unless the optional `socksio` package is installed.
    That is exactly what the unpatched code did on every ComfyUI call, and it is
    why trust_env=False is a fix rather than a preference.
    """
    with pytest.raises(ImportError, match="socksio"):
        httpx.Client(base_url="http://127.0.0.1:8188", trust_env=True)


def test_injected_client_is_used_verbatim(hostile_proxy_env) -> None:
    """A caller that injects a client owns that decision; we do not override it."""
    injected = httpx.Client(base_url="http://127.0.0.1:8188", trust_env=False)
    client = ComfyUIClient(ComfyUIConfig(), client=injected)
    try:
        assert client._client is injected
    finally:
        client.close()


def test_offline_verify_succeeds_under_proxies(config, hostile_proxy_env) -> None:
    from app.offline.verify import verify_offline

    report = verify_offline(config, probe_comfyui=True)
    assert report.ok, report.problems
    comfy = report.backends.get("comfyui", {}).get("runtime", {})
    assert comfy.get("available") is False


# ---------------------------------------------------------------------------
# requirement 14: the guard blocks DNS too
# ---------------------------------------------------------------------------
def test_dns_lookup_for_a_public_host_is_blocked() -> None:
    with pytest.raises(NetworkAccessAttempted):
        socket.getaddrinfo("example.com", 80)
    with pytest.raises(NetworkAccessAttempted):
        socket.gethostbyname("example.com")


def test_socket_connection_to_a_public_host_is_blocked() -> None:
    with pytest.raises(NetworkAccessAttempted):
        socket.create_connection(("example.com", 80), timeout=1)
    with pytest.raises(NetworkAccessAttempted):
        socket.socket().connect(("93.184.216.34", 80))


def test_localhost_names_and_numeric_loopback_still_resolve() -> None:
    assert socket.getaddrinfo("localhost", 80)
    assert socket.getaddrinfo("127.0.0.1", 80)
    assert socket.getaddrinfo("127.0.0.53", 80)  # any loopback address
    assert socket.gethostbyname("localhost")


def test_loopback_connections_still_work() -> None:
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


def test_public_bind_address_is_not_treated_as_loopback() -> None:
    """0.0.0.0 is a bind wildcard, not a loopback destination."""
    from tests.conftest import _is_loopback_host

    assert not _is_loopback_host("0.0.0.0")
    assert _is_loopback_host("127.0.0.1")
    assert _is_loopback_host("::1")


def test_testclient_still_works_under_the_guard(config) -> None:
    from fastapi.testclient import TestClient

    application = create_app(config, configure_logs=False)
    with TestClient(application) as client:
        assert client.get("/health").status_code == 200


def test_mock_comfy_transport_still_works_under_the_guard() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"system": {}})

    config = ComfyUIConfig()
    with ComfyUIClient(
        config,
        client=httpx.Client(
            base_url=config.base_url, transport=httpx.MockTransport(handler), trust_env=False
        ),
    ) as client:
        assert client.is_healthy()


# ---------------------------------------------------------------------------
# requirement 12: the tested dependency versions
# ---------------------------------------------------------------------------
CONSTRAINTS = REPO_ROOT / "constraints" / "tested-py311.txt"


def parse_constraints() -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in CONSTRAINTS.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name, _, version = stripped.partition("==")
        if version:
            pins[name.strip().lower()] = version.strip()
    return pins


def test_constraints_file_exists_and_pins_the_media_stack() -> None:
    assert CONSTRAINTS.is_file()
    pins = parse_constraints()
    assert "numpy" in pins
    assert "opencv-python-headless" in pins


def test_installed_media_stack_matches_the_pins() -> None:
    """The versions under test are the versions the constraints file claims."""
    pins = parse_constraints()
    for name in ("numpy", "opencv-python-headless"):
        assert metadata.version(name) == pins[name], (
            f"{name} {metadata.version(name)} is installed but the constraints "
            f"file pins {pins[name]}; update constraints/tested-py311.txt after "
            "re-running the suite."
        )


def test_pyproject_bounds_exclude_the_known_bad_combination() -> None:
    """NumPy 2.5 + OpenCV 5 aborted the interpreter; the bounds must exclude it."""
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    numpy_spec = re.search(r'"numpy([^"]*)"', text)
    opencv_spec = re.search(r'"opencv-python-headless([^"]*)"', text)
    assert numpy_spec and opencv_spec
    assert "<2.3" in numpy_spec.group(1), "numpy needs an upper bound below 2.3"
    assert "<5" in opencv_spec.group(1), "opencv needs an upper bound below 5"


def test_installed_versions_satisfy_the_declared_bounds() -> None:
    numpy_version = tuple(int(p) for p in metadata.version("numpy").split(".")[:2])
    opencv_version = int(metadata.version("opencv-python-headless").split(".")[0])
    assert (1, 26) <= numpy_version < (2, 3), f"numpy {numpy_version} is outside the tested range"
    assert 4 <= opencv_version < 5, f"opencv {opencv_version} is outside the tested range"


def test_media_stack_imports_without_aborting() -> None:
    """The regression this guards: `import cv2` killed the process (exit 135)."""
    import cv2
    import numpy as np

    array = np.zeros((8, 8, 3), dtype=np.uint8)
    assert cv2.cvtColor(array, cv2.COLOR_BGR2GRAY).shape == (8, 8)
