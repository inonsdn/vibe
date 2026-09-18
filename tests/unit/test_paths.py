"""Path containment: requirement 13 — invalid paths cannot escape the data root."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.errors import PathSecurityError
from app.core.paths import DataRoot, is_safe_relative, safe_identifier

ESCAPE_ATTEMPTS = [
    "../secrets.txt",
    "../../etc/passwd",
    "a/../../b",
    "./../outside",
    "subdir/../../../tmp/evil",
    "/etc/passwd",
    "/",
    "\\\\server\\share\\file",
    "//server/share/file",
    "C:\\Windows\\System32\\config",
    "..",
    "templates/../../..",
]


@pytest.fixture
def root(tmp_path: Path) -> DataRoot:
    data_root = DataRoot(tmp_path / "data")
    data_root.ensure()
    return data_root


@pytest.mark.parametrize("candidate", ESCAPE_ATTEMPTS)
def test_escape_attempts_are_rejected(root: DataRoot, candidate: str) -> None:
    with pytest.raises(PathSecurityError):
        root.resolve(candidate)


def test_nul_byte_is_rejected(root: DataRoot) -> None:
    with pytest.raises(PathSecurityError):
        root.resolve("templates/\x00evil")


def test_empty_path_is_rejected(root: DataRoot) -> None:
    with pytest.raises(PathSecurityError):
        root.resolve("   ")


def test_legitimate_relative_paths_resolve_inside(root: DataRoot) -> None:
    resolved = root.resolve("templates", "tpl_1", "source_frames", "frame_000000.png")
    assert resolved.is_relative_to(root.path)
    assert resolved.name == "frame_000000.png"


def test_absolute_path_inside_root_is_allowed(root: DataRoot) -> None:
    inside = root.path / "jobs" / "job_1"
    assert root.resolve(inside) == inside.resolve()


def test_dotdot_that_stays_inside_is_allowed(root: DataRoot) -> None:
    # Resolution is judged on the result, not on the presence of "..".
    assert root.resolve("templates/tpl_1/../tpl_2") == (root.path / "templates" / "tpl_2")


def test_symlink_escape_is_rejected(root: DataRoot, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    link = root.path / "escape_link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip("symlinks not supported on this platform/user")
    with pytest.raises(PathSecurityError):
        root.resolve("escape_link/file.txt")


def test_relativize_round_trips(root: DataRoot) -> None:
    absolute = root.resolve("templates/tpl_1")
    assert root.relativize(absolute).as_posix() == "templates/tpl_1"


def test_contains(root: DataRoot, tmp_path: Path) -> None:
    assert root.contains(root.path / "jobs")
    assert not root.contains(tmp_path / "elsewhere")


@pytest.mark.parametrize("value", ["..", "../x", "/abs", "C:\\x", "", "CON", "NUL.txt", "a/../b"])
def test_is_safe_relative_rejects(value: str) -> None:
    assert not is_safe_relative(value)


@pytest.mark.parametrize("value", ["frame_000001.png", "sub/dir/file.json", "a.b.c"])
def test_is_safe_relative_accepts(value: str) -> None:
    assert is_safe_relative(value)


@pytest.mark.parametrize("value", ["../x", "a/b", ".", "..", "", "NUL", "with space"])
def test_safe_identifier_rejects(value: str) -> None:
    with pytest.raises(PathSecurityError):
        safe_identifier(value)


def test_safe_identifier_accepts() -> None:
    assert safe_identifier("tpl_20260101T000000Z_abc123") == "tpl_20260101T000000Z_abc123"


def test_job_and_template_dirs_are_confined(root: DataRoot) -> None:
    assert root.job_dir("job_1").is_relative_to(root.path)
    with pytest.raises(PathSecurityError):
        root.template_dir("../escape")
