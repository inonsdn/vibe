"""Deterministic hashing helpers.

Hashes are the backbone of reproducibility: source videos, extracted frames,
masks, garment images, configs, workflows and outputs are all identified by
SHA-256. ``sha256_json`` gives a canonical hash for structured settings so two
jobs with the same settings hash the same regardless of key order.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

CHUNK_SIZE = 1024 * 1024


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Stream a file through SHA-256 (safe for multi-GB video)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    """Canonical JSON: sorted keys, no insignificant whitespace, UTF-8 safe."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, set | frozenset):
        return sorted(str(v) for v in value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_dir(
    directory: str | os.PathLike[str],
    *,
    patterns: Iterable[str] = ("*",),
    relative_to: str | os.PathLike[str] | None = None,
) -> str:
    """Hash a directory's contents deterministically.

    The digest covers the sorted relative POSIX path of every matched file plus
    that file's own digest, so renames and reorders are detected.
    """
    root = Path(directory)
    base = Path(relative_to) if relative_to is not None else root
    files: set[Path] = set()
    for pattern in patterns:
        files.update(p for p in root.rglob(pattern) if p.is_file())
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda p: p.relative_to(base).as_posix()):
        digest.update(path.relative_to(base).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def hash_manifest(paths: Mapping[str, str | os.PathLike[str]]) -> dict[str, str]:
    """Hash a named set of files; missing files map to ``"missing"``."""
    out: dict[str, str] = {}
    for name, path in paths.items():
        candidate = Path(path)
        out[name] = sha256_file(candidate) if candidate.is_file() else "missing"
    return out


def short(digest: str, length: int = 12) -> str:
    return digest[:length]


__all__ = [
    "canonical_json",
    "hash_manifest",
    "sha256_bytes",
    "sha256_dir",
    "sha256_file",
    "sha256_json",
    "sha256_text",
    "short",
]
