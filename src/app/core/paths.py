"""Traversal-safe path handling.

All media and artifact paths handed to the system by an operator (CLI flag, API
payload, YAML config) are resolved through :class:`DataRoot`. Anything that
resolves outside the configured root is rejected with
:class:`~app.core.errors.PathSecurityError`.

The rules are deliberately strict:

* absolute paths are allowed only if they are inside the root,
* ``..`` segments are allowed only if the *resolved* result stays inside,
* symlinks are resolved before the containment check,
* NUL bytes and Windows drive/UNC hops are rejected outright.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from pathlib import Path, PurePosixPath, PureWindowsPath

from app.core.errors import PathSecurityError

#: Directories created under the data root at bootstrap.
RUNTIME_SUBDIRS: tuple[str, ...] = (
    "templates",
    "garments",
    "jobs",
    "cache",
    "exports",
    "logs",
    "tmp",
    "db",
)

_WINDOWS_RESERVED = re.compile(
    r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\.|$)",
    re.IGNORECASE,
)


def _reject_hostile_literal(raw: str) -> None:
    if "\x00" in raw:
        raise PathSecurityError("Path contains a NUL byte", path=raw.replace("\x00", "\\0"))
    if raw.strip() == "":
        raise PathSecurityError("Path is empty")
    # Reject UNC paths outright: they always leave the local data root.
    if raw.startswith("\\\\") or raw.startswith("//"):
        raise PathSecurityError("UNC / network paths are not allowed", path=raw)


def is_safe_relative(candidate: str) -> bool:
    """Return ``True`` if ``candidate`` is a purely relative, downward path.

    Used for path-like values coming from *metadata* (e.g. a mask filename in an
    import manifest) where absolute paths make no sense at all.
    """
    if not candidate or "\x00" in candidate:
        return False
    for flavour in (PurePosixPath(candidate), PureWindowsPath(candidate)):
        if flavour.is_absolute() or flavour.drive or flavour.anchor:
            return False
        if any(part == ".." for part in flavour.parts):
            return False
    return not _WINDOWS_RESERVED.match(Path(candidate).name)


class DataRoot:
    """A validated directory that confines every runtime path."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        _reject_hostile_literal(str(root))
        self._root = Path(root).expanduser().resolve()

    @property
    def path(self) -> Path:
        return self._root

    def __fspath__(self) -> str:  # pragma: no cover - trivial
        return str(self._root)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"DataRoot({str(self._root)!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DataRoot) and other._root == self._root

    def __hash__(self) -> int:
        return hash(self._root)

    # -- creation ---------------------------------------------------------
    def ensure(self, subdirs: Iterable[str] = RUNTIME_SUBDIRS) -> None:
        """Create the root and its documented runtime subdirectories."""
        self._root.mkdir(parents=True, exist_ok=True)
        for name in subdirs:
            self.resolve(name).mkdir(parents=True, exist_ok=True)

    # -- resolution -------------------------------------------------------
    def resolve(self, *parts: str | os.PathLike[str]) -> Path:
        """Resolve ``parts`` inside the root, rejecting any escape.

        Both relative and absolute inputs are accepted; absolute inputs must
        already live inside the root.
        """
        if not parts:
            return self._root
        raw = str(PurePosixPath(*[str(p) for p in parts])) if len(parts) > 1 else str(parts[0])
        _reject_hostile_literal(raw)

        candidate = Path(raw).expanduser()
        # A Windows drive letter on a POSIX host (or a drive different from the
        # root's) can never be inside the root; catch it before resolve() turns
        # it into a confusing relative path.
        win = PureWindowsPath(raw)
        if win.drive and win.drive.lower() != PureWindowsPath(self._root).drive.lower():
            raise PathSecurityError("Path points at a different drive", path=raw)

        resolved = (candidate if candidate.is_absolute() else self._root / candidate).resolve()
        if not self.contains(resolved):
            raise PathSecurityError(
                "Path escapes the configured data root",
                path=raw,
                resolved=str(resolved),
                data_root=str(self._root),
            )
        return resolved

    def contains(self, candidate: str | os.PathLike[str]) -> bool:
        """Return ``True`` if ``candidate`` resolves inside the root."""
        try:
            resolved = Path(candidate).expanduser().resolve()
        except OSError:  # pragma: no cover - platform dependent
            return False
        return resolved == self._root or self._root in resolved.parents

    def relativize(self, candidate: str | os.PathLike[str]) -> Path:
        """Return ``candidate`` as a path relative to the root (validated)."""
        return self.resolve(candidate).relative_to(self._root)

    # -- well-known locations --------------------------------------------
    def template_dir(self, template_id: str) -> Path:
        return self.resolve("templates", _safe_id(template_id))

    def garment_dir(self, garment_id: str) -> Path:
        return self.resolve("garments", _safe_id(garment_id))

    def job_dir(self, job_id: str) -> Path:
        return self.resolve("jobs", _safe_id(job_id))

    def export_dir(self) -> Path:
        return self.resolve("exports")

    def cache_dir(self) -> Path:
        return self.resolve("cache")

    def tmp_dir(self) -> Path:
        return self.resolve("tmp")

    def db_path(self, filename: str = "app.db") -> Path:
        if not is_safe_relative(filename):
            raise PathSecurityError("Unsafe database filename", path=filename)
        return self.resolve("db", filename)


_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _safe_id(value: str) -> str:
    """Validate an identifier used as a directory name."""
    if not _ID_RE.match(value or ""):
        raise PathSecurityError("Identifier is not a safe directory name", identifier=value)
    if value in {".", ".."} or _WINDOWS_RESERVED.match(value):
        raise PathSecurityError("Identifier is a reserved name", identifier=value)
    return value


def safe_identifier(value: str) -> str:
    """Public wrapper around identifier validation."""
    return _safe_id(value)


__all__ = ["RUNTIME_SUBDIRS", "DataRoot", "is_safe_relative", "safe_identifier"]
