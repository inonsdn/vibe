"""Frame-sequence assembly: cached intro + newly dressed reveal.

The intro is never regenerated. It is byte-copied from the template's immutable
source frames into a per-template ``intro_cache``, and every job links or copies
those same bytes into its own assembly directory. Two jobs for different
garments therefore have bit-identical intro frames by construction, not by
luck — which is what the "intro frames are identical across garment jobs" test
verifies.

Frames in the assembly directory keep their **absolute** master-video index.
The encoder is then told ``-start_number <intro.start>``, so there is exactly
one place where indices could drift, and it is explicit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from app.core.errors import ValidationError
from app.core.hashing import sha256_dir, sha256_file
from app.core.logging import get_logger
from app.media.compositor import apply_flash
from app.media.frames import (
    DEFAULT_TEMPLATE,
    frame_path,
    list_frame_indices,
    read_frame,
    validate_sequence,
    write_frame,
)

logger = get_logger(__name__)


@dataclass
class IntroCache:
    directory: Path
    indices: list[int]
    digest: str
    reused: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "directory": str(self.directory),
            "frame_count": len(self.indices),
            "first_frame": self.indices[0] if self.indices else None,
            "last_frame": self.indices[-1] if self.indices else None,
            "sha256": self.digest,
            "reused_existing_cache": self.reused,
        }


def ensure_intro_cache(
    source_frames_dir: str | os.PathLike[str],
    cache_dir: str | os.PathLike[str],
    start: int,
    end: int,
    *,
    template: str = DEFAULT_TEMPLATE,
    expected_digest: str | None = None,
) -> IntroCache:
    """Populate (or validate) the template's intro frame cache.

    A byte copy, never a decode/re-encode. If the cache already holds the full
    range it is reused as-is, which is the whole point: the intro is rendered
    zero times, ever.
    """
    source = Path(source_frames_dir)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    wanted = list(range(start, end))

    present = set(list_frame_indices(cache, template=template))
    missing = [index for index in wanted if index not in present]
    reused = not missing

    for index in missing:
        origin = frame_path(source, index, template)
        if not origin.is_file():
            raise ValidationError(
                "Intro frame missing from the immutable source frames",
                frame_index=index,
                path=str(origin),
            )
        frame_path(cache, index, template).write_bytes(origin.read_bytes())

    report = validate_sequence(cache, start, end, template=template, check_sizes=False)
    if not report.ok:
        raise ValidationError(
            "Intro cache is incomplete after population",
            missing=report.missing[:16],
            missing_count=len(report.missing),
        )

    digest = sha256_dir(cache, patterns=(template.replace("{index:06d}", "*"),))
    if expected_digest is not None and digest != expected_digest:
        raise ValidationError(
            "Intro cache contents differ from the expected digest; the cached "
            "intro is supposed to be immutable",
            expected=expected_digest,
            actual=digest,
        )
    return IntroCache(directory=cache, indices=wanted, digest=digest, reused=reused)


def _link_or_copy(source: Path, target: Path) -> None:
    """Hardlink when the filesystem allows it, else byte-copy.

    Either way the assembled frame is byte-identical to its origin; the link is
    purely a disk-space optimisation for 1080x1920 PNG sequences.
    """
    if target.exists():
        return
    try:
        os.link(source, target)
    except (OSError, NotImplementedError):
        target.write_bytes(source.read_bytes())


@dataclass
class AssemblyReport:
    directory: Path
    intro_frames: list[int]
    reveal_frames: list[int]
    flash_frames: list[int] = field(default_factory=list)
    total_frames: int = 0
    transition_anchor_frame: int = 0
    frame_hashes: dict[int, str] = field(default_factory=dict)

    @property
    def first_frame(self) -> int:
        return (self.intro_frames or self.reveal_frames)[0]

    @property
    def last_frame(self) -> int:
        return (self.reveal_frames or self.intro_frames)[-1]

    def as_dict(self) -> dict[str, object]:
        return {
            "directory": str(self.directory),
            "intro_frame_count": len(self.intro_frames),
            "reveal_frame_count": len(self.reveal_frames),
            "flash_frames": self.flash_frames,
            "total_frames": self.total_frames,
            "transition_anchor_frame": self.transition_anchor_frame,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
        }


def assemble_frames(
    assembly_dir: str | os.PathLike[str],
    *,
    intro_cache_dir: str | os.PathLike[str],
    intro_start: int,
    transition_anchor: int,
    reveal_frames_dir: str | os.PathLike[str],
    reveal_end: int,
    template: str = DEFAULT_TEMPLATE,
    flash_frames: int = 0,
    flash_color: tuple[int, int, int] = (255, 255, 255),
    flash_opacity: float = 0.85,
    hash_frames: bool = True,
) -> AssemblyReport:
    """Join the cached intro and the rendered reveal into one sequence.

    The seam is exact by construction: the intro contributes
    ``[intro_start, transition_anchor)`` and the reveal contributes
    ``[transition_anchor, reveal_end)``. There is no frame that both provide and
    none that neither provides, and this function asserts both facts.
    """
    if transition_anchor < intro_start:
        raise ValidationError(
            "Transition anchor precedes the intro start",
            intro_start=intro_start,
            transition_anchor=transition_anchor,
        )
    if reveal_end <= transition_anchor:
        raise ValidationError(
            "Reveal range is empty",
            transition_anchor=transition_anchor,
            reveal_end=reveal_end,
        )
    if flash_frames and flash_frames > (reveal_end - transition_anchor):
        raise ValidationError(
            "Flash transition is longer than the reveal segment",
            flash_frames=flash_frames,
            reveal_frames=reveal_end - transition_anchor,
        )

    target = Path(assembly_dir)
    target.mkdir(parents=True, exist_ok=True)
    intro_cache = Path(intro_cache_dir)
    reveal_dir = Path(reveal_frames_dir)

    intro_indices = list(range(intro_start, transition_anchor))
    reveal_indices = list(range(transition_anchor, reveal_end))

    for index in intro_indices:
        source = frame_path(intro_cache, index, template)
        if not source.is_file():
            raise ValidationError("Cached intro frame missing", frame_index=index, path=str(source))
        _link_or_copy(source, frame_path(target, index, template))

    flashed: list[int] = []
    flash_set = set(reveal_indices[:flash_frames]) if flash_frames else set()
    for index in reveal_indices:
        source = frame_path(reveal_dir, index, template)
        if not source.is_file():
            raise ValidationError(
                "Rendered reveal frame missing", frame_index=index, path=str(source)
            )
        destination = frame_path(target, index, template)
        if index in flash_set:
            # A flash frame intentionally alters the whole frame, including
            # protected regions, so it is recorded separately and excluded from
            # the protected-pixel QC checks.
            frame = read_frame(source)
            write_frame(destination, apply_flash(frame, flash_color, flash_opacity))
            flashed.append(index)
        else:
            _link_or_copy(source, destination)

    # The seam must be exactly one frame wide in index space.
    expected = intro_indices + reveal_indices
    present = list_frame_indices(target, template=template)
    if present != expected:
        raise ValidationError(
            "Assembled sequence is not contiguous over the expected range",
            expected_first=expected[0],
            expected_last=expected[-1],
            expected_count=len(expected),
            actual_count=len(present),
            missing=sorted(set(expected) - set(present))[:16],
            unexpected=sorted(set(present) - set(expected))[:16],
        )
    if intro_indices and intro_indices[-1] + 1 != transition_anchor:
        raise ValidationError(
            "Off-by-one at the transition anchor",
            last_intro_frame=intro_indices[-1],
            transition_anchor=transition_anchor,
        )

    hashes: dict[int, str] = {}
    if hash_frames:
        hashes = {index: sha256_file(frame_path(target, index, template)) for index in expected}

    return AssemblyReport(
        directory=target,
        intro_frames=intro_indices,
        reveal_frames=reveal_indices,
        flash_frames=flashed,
        total_frames=len(expected),
        transition_anchor_frame=transition_anchor,
        frame_hashes=hashes,
    )


def verify_transition(
    assembly_dir: str | os.PathLike[str],
    intro_cache_dir: str | os.PathLike[str],
    reveal_frames_dir: str | os.PathLike[str],
    transition_anchor: int,
    *,
    template: str = DEFAULT_TEMPLATE,
    flash_frames: int = 0,
) -> dict[str, object]:
    """Prove the seam is correct: last intro frame and first reveal frame.

    Returns evidence rather than a bare boolean so the QC report can show which
    file each side came from.
    """
    assembly = Path(assembly_dir)
    last_intro = transition_anchor - 1
    result: dict[str, object] = {
        "transition_anchor_frame": transition_anchor,
        "last_intro_frame": last_intro,
        "first_reveal_frame": transition_anchor,
    }

    if last_intro >= 0:
        cached = frame_path(Path(intro_cache_dir), last_intro, template)
        assembled = frame_path(assembly, last_intro, template)
        result["last_intro_matches_cache"] = (
            cached.is_file()
            and assembled.is_file()
            and sha256_file(cached) == sha256_file(assembled)
        )
    else:
        result["last_intro_matches_cache"] = None

    rendered = frame_path(Path(reveal_frames_dir), transition_anchor, template)
    assembled_reveal = frame_path(assembly, transition_anchor, template)
    if flash_frames:
        # The first reveal frames are deliberately altered by the flash.
        result["first_reveal_matches_render"] = None
        result["first_reveal_is_flash"] = True
    else:
        result["first_reveal_matches_render"] = (
            rendered.is_file()
            and assembled_reveal.is_file()
            and sha256_file(rendered) == sha256_file(assembled_reveal)
        )
        result["first_reveal_is_flash"] = False
    return result


__all__ = [
    "AssemblyReport",
    "IntroCache",
    "assemble_frames",
    "ensure_intro_cache",
    "verify_transition",
]
