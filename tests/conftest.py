"""Shared pytest fixtures and the no-network guard.

The whole suite must pass with:

* no GPU,
* no model weights,
* no ComfyUI running,
* no network access.

The last one is enforced, not assumed: :func:`_block_network` patches the socket
layer so any attempt to reach a non-loopback address fails the test that made
it. Loopback is allowed because FastAPI's ``TestClient`` and the ComfyUI
transport tests operate in-process, and blocking it would break the test
harness rather than catch a real network call.
"""

from __future__ import annotations

import shutil
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from app.core.config import AppConfig
from app.media import ffmpeg as ffmpeg_module
from app.pipeline.context import ServiceContext
from tests import fixtures

_ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0", "testserver"}


class NetworkAccessAttempted(AssertionError):
    """Raised when a test tries to open a non-loopback connection."""


def _is_local(address: Any) -> bool:
    if isinstance(address, str):  # AF_UNIX
        return True
    if isinstance(address, tuple) and address:
        return str(address[0]) in _ALLOWED_HOSTS
    return False


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that attempts an outbound network connection."""
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_create = socket.create_connection

    def guard_connect(self: socket.socket, address: Any) -> Any:
        if not _is_local(address):
            raise NetworkAccessAttempted(
                f"Outbound network access is forbidden in tests: {address!r}"
            )
        return real_connect(self, address)

    def guard_connect_ex(self: socket.socket, address: Any) -> Any:
        if not _is_local(address):
            raise NetworkAccessAttempted(
                f"Outbound network access is forbidden in tests: {address!r}"
            )
        return real_connect_ex(self, address)

    def guard_create(address: Any, *args: Any, **kwargs: Any) -> Any:
        if not _is_local(address):
            raise NetworkAccessAttempted(
                f"Outbound network access is forbidden in tests: {address!r}"
            )
        return real_create(address, *args, **kwargs)

    def guard_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if str(host) not in _ALLOWED_HOSTS:
            raise NetworkAccessAttempted(f"DNS resolution is forbidden in tests: {host!r}")
        return socket.getaddrinfo.__wrapped__(host, *args, **kwargs)  # pragma: no cover

    monkeypatch.setattr(socket.socket, "connect", guard_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guard_connect_ex)
    monkeypatch.setattr(socket, "create_connection", guard_create)


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture
def config(data_root: Path) -> AppConfig:
    return fixtures.test_config(data_root)


@pytest.fixture
def context(config: AppConfig) -> Iterator[ServiceContext]:
    """A ServiceContext on an isolated temp data root and a fresh database."""
    service = ServiceContext.create(config=config, configure_logs=False)
    try:
        yield service
    finally:
        service.close()


@pytest.fixture
def template(context: ServiceContext) -> fixtures.TemplateFixture:
    return fixtures.make_template(context)


@pytest.fixture
def garment(context: ServiceContext) -> Any:
    return fixtures.make_garment(context)


@pytest.fixture
def ready_pair(
    context: ServiceContext,
    template: fixtures.TemplateFixture,
    garment: Any,
) -> Any:
    """A template/garment pair with a stored READY compatibility report."""
    from app.domain.enums import ImageViewType
    from app.pipeline import compat_service

    report = compat_service.check_compatibility(
        context,
        template.template.id,
        garment.id,
        options=compat_service.CheckOptions(
            exposed_views=[ImageViewType.FRONT, ImageViewType.BACK, ImageViewType.SIDE]
        ),
    )
    return template, garment, report


def ffmpeg_available() -> bool:
    return ffmpeg_module.tool_available("ffmpeg") and ffmpeg_module.tool_available("ffprobe")


requires_ffmpeg = pytest.mark.skipif(
    not ffmpeg_available(),
    reason="ffmpeg/ffprobe not on PATH (install FFmpeg to run encode/probe tests)",
)


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    if not ffmpeg_available():
        pytest.skip("ffmpeg not available")
    return fixtures.write_synthetic_video(tmp_path / "master.mp4")


@pytest.fixture(autouse=True)
def _cleanup_repo_data() -> Iterator[None]:
    """Guard against a test accidentally writing into the repo's data/ dir."""
    repo_data = Path(__file__).resolve().parents[1] / "data"
    before = {p.name for p in repo_data.iterdir()} if repo_data.is_dir() else set()
    yield
    if repo_data.is_dir():
        for entry in repo_data.iterdir():
            if entry.name not in before and entry.name not in {".gitignore", "README.md"}:
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    entry.unlink(missing_ok=True)
