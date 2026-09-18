"""Provenance capture: what code, config and tools produced an artifact.

Everything here is local-only: a ``git`` invocation in the repo directory,
``importlib.metadata`` for dependency versions, and ``ffmpeg -version``. No
network calls, ever.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from app.core.config import AppConfig
from app.version import APP_NAME, APP_VERSION, LAYOUT_VERSION, PREPROCESSING_VERSION

TRACKED_PACKAGES: tuple[str, ...] = (
    "pydantic",
    "fastapi",
    "numpy",
    "opencv-python-headless",
    "opencv-python",
    "PyYAML",
    "typer",
    "httpx",
    "uvicorn",
)

_GIT_TIMEOUT_S = 10.0


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    binary = shutil.which(cmd[0])
    if binary is None:
        return None
    try:
        result = subprocess.run(
            [binary, *cmd[1:]],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def git_info(repo_root: Path | None = None) -> dict[str, Any]:
    """Return git commit/branch/dirty state, or ``available: False``."""
    root = repo_root or Path(__file__).resolve().parents[3]
    if not (root / ".git").exists():
        return {"available": False}
    commit = _run(["git", "rev-parse", "HEAD"], root)
    if commit is None:
        return {"available": False}
    status = _run(["git", "status", "--porcelain"], root)
    return {
        "available": True,
        "commit": commit,
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], root),
        "dirty": bool(status),
        "describe": _run(["git", "describe", "--always", "--dirty", "--tags"], root),
    }


def dependency_versions(packages: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in packages:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out


def ffmpeg_version(binary: str = "ffmpeg") -> str | None:
    banner = _run([binary, "-version"])
    if not banner:
        return None
    return banner.splitlines()[0].strip()


def platform_info() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    }


def collect(config: AppConfig, *, repo_root: Path | None = None) -> dict[str, Any]:
    """Collect the full provenance block recorded in every job manifest."""
    root = repo_root or config.repo_root
    return {
        "app": {
            "name": APP_NAME,
            "version": APP_VERSION,
            "preprocessing_version": PREPROCESSING_VERSION,
            "layout_version": LAYOUT_VERSION,
        },
        "git": git_info(root),
        "platform": platform_info(),
        "dependencies": dependency_versions(),
        "ffmpeg": ffmpeg_version(config.runtime.ffmpeg_binary),
        "ffprobe": ffmpeg_version(config.runtime.ffprobe_binary),
        "config_hash": config.config_hash(),
    }


__all__ = [
    "TRACKED_PACKAGES",
    "collect",
    "dependency_versions",
    "ffmpeg_version",
    "git_info",
    "platform_info",
]
