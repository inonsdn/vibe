"""Contact sheets: the human half of QC.

Two sheets are produced:

* one centred on the transition anchor, so the seam can be eyeballed
  frame-by-frame,
* one sampling the reveal segment periodically, so drift or flicker over the
  whole segment is visible at a glance.

Every tile is labelled with its absolute frame index and its origin (``intro``,
``reveal`` or ``flash``), because an unlabelled contact sheet is nearly useless
when debugging an off-by-one.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from app.core.logging import get_logger
from app.media.frames import DEFAULT_TEMPLATE, frame_path, read_frame

logger = get_logger(__name__)

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(tile: np.ndarray, text: str) -> np.ndarray:
    out = tile.copy()
    height = out.shape[0]
    bar_height = max(18, height // 12)
    cv2.rectangle(out, (0, height - bar_height), (out.shape[1], height), (0, 0, 0), -1)
    scale = max(0.35, bar_height / 34.0)
    cv2.putText(
        out,
        text,
        (4, height - max(5, bar_height // 4)),
        _FONT,
        scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def build_sheet(
    frames_dir: str | os.PathLike[str],
    indices: list[int],
    output_path: str | os.PathLike[str],
    *,
    columns: int = 6,
    tile_width: int = 240,
    template: str = DEFAULT_TEMPLATE,
    labels: dict[int, str] | None = None,
) -> Path | None:
    """Render a labelled grid of frames. Returns ``None`` if nothing existed."""
    source = Path(frames_dir)
    available = [index for index in indices if frame_path(source, index, template).is_file()]
    if not available:
        return None

    tiles: list[np.ndarray] = []
    tile_height: int | None = None
    for index in available:
        frame = read_frame(frame_path(source, index, template))
        height, width = frame.shape[:2]
        scaled_height = max(1, round(tile_width * height / width))
        resized = cv2.resize(frame, (tile_width, scaled_height), interpolation=cv2.INTER_AREA)
        tile_height = tile_height or scaled_height
        if scaled_height != tile_height:
            resized = cv2.resize(resized, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        suffix = labels.get(index, "") if labels else ""
        tiles.append(_label(resized, f"{index}" + (f"  {suffix}" if suffix else "")))

    assert tile_height is not None
    rows: list[np.ndarray] = []
    for offset in range(0, len(tiles), columns):
        row = tiles[offset : offset + columns]
        if len(row) < columns:
            blank = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
            row = row + [blank] * (columns - len(row))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), sheet, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        return None
    return target


def build_contact_sheets(
    frames_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    intro_start: int,
    transition_anchor: int,
    reveal_end: int,
    flash_frames: list[int] | None = None,
    columns: int = 6,
    tile_width: int = 240,
    transition_span: int = 6,
    period_frames: int = 24,
    template: str = DEFAULT_TEMPLATE,
) -> list[Path]:
    """Build the transition sheet and the periodic reveal sheet."""
    flash = set(flash_frames or [])
    labels: dict[int, str] = {}

    def origin(index: int) -> str:
        if index in flash:
            return "flash"
        return "intro" if index < transition_anchor else "reveal"

    transition_indices = [
        index
        for index in range(transition_anchor - transition_span, transition_anchor + transition_span)
        if intro_start <= index < reveal_end
    ]
    for index in transition_indices:
        # The anchor frame is the first reveal frame by definition, so naming it
        # "ANCHOR" is both shorter than "reveal <ANCHOR" (which clipped at the
        # tile edge) and more informative.
        labels[index] = "ANCHOR" if index == transition_anchor else origin(index)

    periodic = list(range(transition_anchor, reveal_end, max(1, period_frames)))
    if reveal_end - 1 not in periodic:
        periodic.append(reveal_end - 1)
    for index in periodic:
        labels.setdefault(index, origin(index))

    out = Path(output_dir)
    sheets: list[Path] = []
    transition_sheet = build_sheet(
        frames_dir,
        transition_indices,
        out / "contact_sheet_transition.png",
        columns=columns,
        tile_width=tile_width,
        template=template,
        labels=labels,
    )
    if transition_sheet:
        sheets.append(transition_sheet)

    reveal_sheet = build_sheet(
        frames_dir,
        periodic,
        out / "contact_sheet_reveal.png",
        columns=columns,
        tile_width=tile_width,
        template=template,
        labels=labels,
    )
    if reveal_sheet:
        sheets.append(reveal_sheet)
    return sheets


__all__ = ["build_contact_sheets", "build_sheet"]
