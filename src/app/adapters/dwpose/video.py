"""Reading specific source frames out of a reference clip.

Frames are read **sequentially** and yielded only for the requested indices.
Seeking would be faster on a long clip, but phone screen recordings are
frequently variable-frame-rate and OpenCV's seek on those lands on a keyframe
rather than the frame you asked for. An extraction that silently reads frame
147 while writing ``frame_000150.json`` is the exact failure this pipeline's
frame-index checks exist to catch, and it is cheaper to avoid it than to detect
it.

The protocol is small on purpose: tests inject a fake reader and never need a
video file.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from app.core.errors import MediaToolError, NotFoundError


@runtime_checkable
class FrameReader(Protocol):
    """Yields ``(absolute_frame_index, bgr_image)`` for the requested indices."""

    def read(
        self, video_path: Path, frame_indices: list[int]
    ) -> Iterator[tuple[int, np.ndarray]]: ...


class OpenCVFrameReader:
    """The real reader. Sequential decode, absolute indices preserved."""

    def __init__(self, *, seek_threshold: int = 0) -> None:
        # A positive threshold enables a single seek to the first requested
        # frame. It stays off by default: correctness over speed on VFR input.
        self._seek_threshold = seek_threshold

    def read(self, video_path: Path, frame_indices: list[int]) -> Iterator[tuple[int, np.ndarray]]:
        import cv2

        path = Path(video_path)
        if not path.is_file():
            raise NotFoundError("Reference video not found", path=str(path))

        wanted = sorted({int(i) for i in frame_indices})
        if not wanted:
            return

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise MediaToolError(
                "OpenCV could not open the reference video. Check the codec and "
                "that FFmpeg support is present in this OpenCV build.",
                path=str(path),
            )
        try:
            start = 0
            if self._seek_threshold and wanted[0] >= self._seek_threshold:
                capture.set(cv2.CAP_PROP_POS_FRAMES, float(wanted[0]))
                start = wanted[0]

            remaining = set(wanted)
            index = start
            last = wanted[-1]
            while remaining and index <= last:
                ok, frame = capture.read()
                if not ok:
                    break
                if index in remaining:
                    remaining.discard(index)
                    yield index, frame
                index += 1

            if remaining:
                raise MediaToolError(
                    "The reference video ended before every requested frame was "
                    "read. The selected range is longer than the clip.",
                    path=str(path),
                    missing_count=len(remaining),
                    first_missing=min(remaining),
                    last_read=index - 1,
                )
        finally:
            capture.release()


__all__ = ["FrameReader", "OpenCVFrameReader"]
