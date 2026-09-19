"""Staging a *sequence* of frames as one deterministic ComfyUI input.

A character animation workflow does not consume a still. It consumes an ordered
run of pose-control frames — one per output frame in the chunk — and, where the
workflow conditions on previous output, an ordered run of context frames. Handing
such a workflow a single image and then recording "24 frames, 16 context frames"
in the manifest would be a lie the manifest then carries forever.

Two on-disk protocols are supported, chosen per binding by its contract ``kind``:

``sequence_dir``
    A directory of PNGs named ``<prefix>_00000.png`` .. by **ordinal position in
    the chunk**, starting at zero, so any directory-loading node reads them in
    the right order under plain lexicographic sort. Lossless, no encoder needed.
``sequence_video``
    One visually lossless video (``-qp 0`` H.264, or FFV1 where configured)
    containing the same frames in the same order.

Both write a ``sequence.json`` manifest beside the payload recording the
*absolute* output frame index each ordinal corresponds to. Ordinal position is
what the workflow sees; the absolute index is what the rest of the system talks
in, and losing the mapping between them is how frames get silently reordered.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.errors import BackendError, MediaToolError
from app.media import ffmpeg
from app.media.frames import write_frame

#: Contract ``kind`` values that carry a whole sequence.
SEQUENCE_KIND_DIR = "sequence_dir"
SEQUENCE_KIND_VIDEO = "sequence_video"
SEQUENCE_KINDS: frozenset[str] = frozenset({SEQUENCE_KIND_DIR, SEQUENCE_KIND_VIDEO})

#: Contract ``kind`` values that carry exactly one file or scalar.
SINGLE_KINDS: frozenset[str] = frozenset({"value", "image_path", "image_upload"})

#: Every kind a contract may declare.
KNOWN_KINDS: frozenset[str] = SINGLE_KINDS | SEQUENCE_KINDS

MANIFEST_NAME = "sequence.json"


@dataclass(frozen=True)
class StagedSequence:
    """An ordered run of frames written to disk, ready to hand to ComfyUI."""

    logical_name: str
    kind: str
    directory: Path
    files: tuple[Path, ...]
    frame_indices: tuple[int, ...]
    manifest_path: Path
    video_path: Path | None = None

    @property
    def count(self) -> int:
        return len(self.frame_indices)

    @property
    def first_index(self) -> int | None:
        return self.frame_indices[0] if self.frame_indices else None

    @property
    def last_index(self) -> int | None:
        return self.frame_indices[-1] if self.frame_indices else None

    @property
    def payload(self) -> Path:
        """The single path a workflow binding points at."""
        return self.video_path if self.video_path is not None else self.directory

    def as_dict(self) -> dict[str, object]:
        return {
            "logical_name": self.logical_name,
            "kind": self.kind,
            "count": self.count,
            "first_frame": self.first_index,
            "last_frame": self.last_index,
            "frame_indices": list(self.frame_indices),
            "files": [f.name for f in self.files],
            "video": self.video_path.name if self.video_path else None,
        }


def stage_frame_sequence(
    directory: Path,
    frames: Sequence[tuple[int, np.ndarray]],
    *,
    logical_name: str,
    prefix: str | None = None,
) -> StagedSequence:
    """Write ``frames`` as an ordinal-numbered PNG sequence plus a manifest.

    ``frames`` is ``[(absolute_frame_index, image), ...]`` in the order the
    workflow must consume them. The order given is the order written; nothing
    here sorts, because "in the order the caller meant" is the whole contract.
    """
    if not frames:
        raise BackendError(
            "Refusing to stage an empty sequence",
            logical_name=logical_name,
            hint="A chunk always has at least one pose.",
        )
    directory.mkdir(parents=True, exist_ok=True)
    stem = prefix or logical_name
    # A stale run must not leave extra files a directory loader would pick up.
    for existing in directory.glob(f"{stem}_*.png"):
        existing.unlink()

    written: list[Path] = []
    indices: list[int] = []
    for ordinal, (frame_index, image) in enumerate(frames):
        target = directory / f"{stem}_{ordinal:05d}.png"
        write_frame(target, image)
        written.append(target)
        indices.append(int(frame_index))

    manifest = directory / MANIFEST_NAME
    staged = StagedSequence(
        logical_name=logical_name,
        kind=SEQUENCE_KIND_DIR,
        directory=directory,
        files=tuple(written),
        frame_indices=tuple(indices),
        manifest_path=manifest,
    )
    manifest.write_text(json.dumps(staged.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return staged


def encode_sequence_video(
    staged: StagedSequence,
    destination: Path,
    *,
    fps: float,
    ffmpeg_binary: str = "ffmpeg",
) -> StagedSequence:
    """Encode an already-staged PNG run into one visually lossless video.

    ``-qp 0`` H.264 in yuv444p: no chroma subsampling, no quantisation. A pose
    control sequence that loses colour fidelity loses joint identity, so "close
    enough" is not an option here.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    pattern = staged.directory / f"{staged.files[0].name.rsplit('_', 1)[0]}_%05d.png"
    argv = [
        ffmpeg.resolve_binary(ffmpeg_binary),
        "-hide_banner",
        "-nostdin",
        "-y",
        "-framerate",
        f"{fps:g}",
        "-start_number",
        "0",
        "-i",
        str(pattern),
        "-frames:v",
        str(staged.count),
        "-c:v",
        "libx264",
        "-qp",
        "0",
        "-pix_fmt",
        "yuv444p",
        "-an",
        str(destination),
    ]
    result = ffmpeg.run_command(argv)
    if not result.ok or not destination.is_file():
        raise MediaToolError(
            "Failed to encode the pose control sequence",
            logical_name=staged.logical_name,
            stderr=result.stderr[-2000:],
        )
    updated = StagedSequence(
        logical_name=staged.logical_name,
        kind=SEQUENCE_KIND_VIDEO,
        directory=staged.directory,
        files=staged.files,
        frame_indices=staged.frame_indices,
        manifest_path=staged.manifest_path,
        video_path=destination,
    )
    staged.manifest_path.write_text(
        json.dumps(updated.as_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return updated


__all__ = [
    "KNOWN_KINDS",
    "MANIFEST_NAME",
    "SEQUENCE_KINDS",
    "SEQUENCE_KIND_DIR",
    "SEQUENCE_KIND_VIDEO",
    "SINGLE_KINDS",
    "StagedSequence",
    "encode_sequence_video",
    "stage_frame_sequence",
]
