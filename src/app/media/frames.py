"""Frame sequence I/O and validation.

Frames are stored as lossless PNG named by their *absolute index in the master
video* (``frame_000123.png``). Using absolute indices everywhere — not
per-segment offsets — is what removes the classic off-by-one class of bugs when
the intro and reveal segments are joined.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.core.errors import ValidationError
from app.core.hashing import sha256_file
from app.core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_TEMPLATE = "frame_{index:06d}.png"
#: PNG compression level 1 keeps writes fast; PNG is lossless at every level.
PNG_PARAMS = (cv2.IMWRITE_PNG_COMPRESSION, 1)


def frame_filename(index: int, template: str = DEFAULT_TEMPLATE) -> str:
    if index < 0:
        raise ValidationError("Frame index must be non-negative", index=index)
    return template.format(index=index)


def frame_path(
    directory: str | os.PathLike[str], index: int, template: str = DEFAULT_TEMPLATE
) -> Path:
    return Path(directory) / frame_filename(index, template)


def ffmpeg_pattern(template: str = DEFAULT_TEMPLATE) -> str:
    """Translate a Python frame template into an ffmpeg ``%0Nd`` pattern."""
    if "{index:06d}" in template:
        return template.replace("{index:06d}", "%06d")
    if "{index}" in template:
        return template.replace("{index}", "%d")
    raise ValidationError("Unsupported frame filename template", template=template)


def write_frame(path: str | os.PathLike[str], image: np.ndarray) -> Path:
    """Write a BGR or grayscale frame losslessly."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if image.dtype != np.uint8:
        raise ValidationError("Frames must be uint8", dtype=str(image.dtype))
    if not cv2.imwrite(str(target), image, list(PNG_PARAMS)):
        raise ValidationError("Failed to write frame", path=str(target))
    return target


def read_frame(path: str | os.PathLike[str], *, grayscale: bool = False) -> np.ndarray:
    """Read a frame; raises rather than returning ``None`` like cv2 does."""
    target = Path(path)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(target), flag)
    if image is None:
        raise ValidationError("Frame could not be read", path=str(target))
    return image


def read_frame_at(
    directory: str | os.PathLike[str],
    index: int,
    *,
    template: str = DEFAULT_TEMPLATE,
    grayscale: bool = False,
) -> np.ndarray:
    return read_frame(frame_path(directory, index, template), grayscale=grayscale)


def list_frame_indices(
    directory: str | os.PathLike[str], *, template: str = DEFAULT_TEMPLATE
) -> list[int]:
    """Indices present in a frame directory, ascending."""
    folder = Path(directory)
    if not folder.is_dir():
        return []
    prefix, _, suffix = template.partition("{index:06d}")
    if not suffix:
        prefix, _, suffix = template.partition("{index}")
    indices: list[int] = []
    for entry in folder.iterdir():
        name = entry.name
        if not entry.is_file() or not name.startswith(prefix) or not name.endswith(suffix):
            continue
        digits = name[len(prefix) : len(name) - len(suffix)]
        if digits.isdigit():
            indices.append(int(digits))
    return sorted(indices)


def iter_frames(
    directory: str | os.PathLike[str],
    indices: Iterable[int],
    *,
    template: str = DEFAULT_TEMPLATE,
    grayscale: bool = False,
) -> Iterator[tuple[int, np.ndarray]]:
    for index in indices:
        yield index, read_frame_at(directory, index, template=template, grayscale=grayscale)


@dataclass
class SequenceReport:
    """Outcome of validating a frame directory against an expected range."""

    directory: str
    expected_start: int
    expected_end: int
    present: list[int]
    missing: list[int]
    unexpected: list[int]
    wrong_size: list[int]
    width: int | None
    height: int | None

    @property
    def ok(self) -> bool:
        return not (self.missing or self.wrong_size)

    @property
    def count(self) -> int:
        return len(self.present)

    def as_dict(self) -> dict[str, object]:
        return {
            "directory": self.directory,
            "expected_start": self.expected_start,
            "expected_end": self.expected_end,
            "present_count": self.count,
            "missing": self.missing[:64],
            "missing_count": len(self.missing),
            "unexpected": self.unexpected[:64],
            "unexpected_count": len(self.unexpected),
            "wrong_size": self.wrong_size[:64],
            "wrong_size_count": len(self.wrong_size),
            "width": self.width,
            "height": self.height,
            "ok": self.ok,
        }


def validate_sequence(
    directory: str | os.PathLike[str],
    start: int,
    end: int,
    *,
    template: str = DEFAULT_TEMPLATE,
    expect_width: int | None = None,
    expect_height: int | None = None,
    check_sizes: bool = True,
) -> SequenceReport:
    """Verify that ``[start, end)`` exists on disk with consistent dimensions."""
    folder = Path(directory)
    present = list_frame_indices(folder, template=template)
    present_set = set(present)
    expected = set(range(start, end))
    width, height = expect_width, expect_height
    wrong_size: list[int] = []

    if check_sizes:
        for index in sorted(expected & present_set):
            path = frame_path(folder, index, template)
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                wrong_size.append(index)
                continue
            frame_h, frame_w = image.shape[:2]
            if width is None or height is None:
                width, height = frame_w, frame_h
            elif (frame_w, frame_h) != (width, height):
                wrong_size.append(index)

    return SequenceReport(
        directory=str(folder),
        expected_start=start,
        expected_end=end,
        present=sorted(present_set),
        missing=sorted(expected - present_set),
        unexpected=sorted(present_set - expected),
        wrong_size=wrong_size,
        width=width,
        height=height,
    )


def hash_sequence(
    directory: str | os.PathLike[str],
    indices: Iterable[int],
    *,
    template: str = DEFAULT_TEMPLATE,
) -> dict[int, str]:
    """SHA-256 per frame, keyed by absolute frame index."""
    return {index: sha256_file(frame_path(directory, index, template)) for index in indices}


def copy_frames(
    source_dir: str | os.PathLike[str],
    dest_dir: str | os.PathLike[str],
    indices: Iterable[int],
    *,
    template: str = DEFAULT_TEMPLATE,
    overwrite: bool = False,
) -> list[int]:
    """Byte-exact copy of selected frames (used for intro reuse).

    The copy is a raw byte copy, not a decode/re-encode, so intro frames are
    bit-identical across every garment job by construction.
    """
    src = Path(source_dir)
    dst = Path(dest_dir)
    dst.mkdir(parents=True, exist_ok=True)
    copied: list[int] = []
    for index in indices:
        source = frame_path(src, index, template)
        target = frame_path(dst, index, template)
        if not source.is_file():
            raise ValidationError("Source frame missing during copy", index=index, path=str(source))
        if target.exists() and not overwrite:
            copied.append(index)
            continue
        target.write_bytes(source.read_bytes())
        copied.append(index)
    return copied


def frames_equal(path_a: str | os.PathLike[str], path_b: str | os.PathLike[str]) -> bool:
    """Byte-level equality (stronger than pixel equality: catches metadata)."""
    a, b = Path(path_a), Path(path_b)
    if not (a.is_file() and b.is_file()):
        return False
    if a.stat().st_size != b.stat().st_size:
        return False
    return a.read_bytes() == b.read_bytes()


__all__ = [
    "DEFAULT_TEMPLATE",
    "SequenceReport",
    "copy_frames",
    "ffmpeg_pattern",
    "frame_filename",
    "frame_path",
    "frames_equal",
    "hash_sequence",
    "iter_frames",
    "list_frame_indices",
    "read_frame",
    "read_frame_at",
    "validate_sequence",
    "write_frame",
]
