"""Fakes for testing the DWPose adapter without onnxruntime, CUDA or a video.

The fakes are deliberately *low* in the stack: they stand in for the ONNX
sessions and the frame reader, and nothing else. Every line of letterboxing,
YOLOX decoding, NMS, SimCC decoding, crop inversion, ROI restoration, subject
scoring and temporal cleanup is the real implementation, exercised exactly as
it would be against real weights.

``yolox_output`` and ``simcc_output`` are the interesting pieces: they encode a
desired answer *backwards* through the model's own output convention, so a test
that asks for a box at (100, 200, 300, 600) fails unless the decoder really
recovers it.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.adapters.dwpose.session import CPU_PROVIDER, CUDA_PROVIDER
from app.motion.skeleton import BODY_JOINTS

DETECTOR_INPUT = (64, 64)
DETECTOR_STRIDES = (8, 16, 32)
POSE_INPUT = (48, 64)
SPLIT_RATIO = 2.0
NUM_CLASSES = 80


# ---------------------------------------------------------------------------
# session fakes
# ---------------------------------------------------------------------------
@dataclass
class _Port:
    name: str


class FakeSession:
    """Implements the slice of InferenceSession the adapter actually calls."""

    def __init__(
        self,
        handler: Any,
        *,
        providers: list[str] | None = None,
        input_name: str = "input",
    ) -> None:
        self._handler = handler
        self._providers = providers or [CPU_PROVIDER]
        self._input_name = input_name
        self.calls: list[np.ndarray] = []

    def run(self, output_names: list[str] | None, input_feed: dict[str, Any]) -> list[Any]:
        batch = np.asarray(next(iter(input_feed.values())))
        self.calls.append(batch)
        result = self._handler(batch)
        return list(result) if isinstance(result, (list, tuple)) else [result]

    def get_inputs(self) -> list[_Port]:
        return [_Port(self._input_name)]

    def get_outputs(self) -> list[_Port]:
        return [_Port("output")]

    def get_providers(self) -> list[str]:
        return list(self._providers)


@dataclass
class FakeSessionFactory:
    """Hands out fake sessions and records what providers it was asked for."""

    detector: FakeSession
    pose: FakeSession
    available: tuple[str, ...] = (CPU_PROVIDER,)
    version: str = "1.99.0-fake"
    requests: list[tuple[str, list[str]]] = field(default_factory=list)
    #: Providers the created session reports, overriding the session's own.
    reports: dict[str, list[str]] = field(default_factory=dict)

    def available_providers(self) -> list[str]:
        return list(self.available)

    @property
    def runtime_version(self) -> str:
        return self.version

    def create(self, model_path: Path, providers: list[str]) -> FakeSession:
        name = Path(model_path).name
        self.requests.append((name, list(providers)))
        session = self.pose if "pose" in name or "dw" in name else self.detector
        # A real onnxruntime reports the provider it actually bound, which is
        # the first *offered* provider it supports -- not necessarily the first
        # one asked for. That difference is the whole point of the reporting.
        reported = self.reports.get(name)
        if reported is not None:
            session._providers = list(reported)
        else:
            session._providers = [providers[0]] if providers else [CPU_PROVIDER]
        return session


def cuda_factory(detector: FakeSession, pose: FakeSession) -> FakeSessionFactory:
    return FakeSessionFactory(detector=detector, pose=pose, available=(CUDA_PROVIDER, CPU_PROVIDER))


# ---------------------------------------------------------------------------
# encoding helpers: build model output that decodes to a known answer
# ---------------------------------------------------------------------------
def _grid_rows(input_size: tuple[int, int], strides: tuple[int, ...]) -> list[tuple[int, int, int]]:
    """``(stride, grid_x, grid_y)`` for every row of a YOLOX prediction table."""
    width, height = input_size
    rows: list[tuple[int, int, int]] = []
    for stride in strides:
        for gy in range(height // stride):
            for gx in range(width // stride):
                rows.append((stride, gx, gy))
    return rows


def yolox_output(
    boxes: list[tuple[float, float, float, float, float]],
    *,
    input_size: tuple[int, int] = DETECTOR_INPUT,
    strides: tuple[int, ...] = DETECTOR_STRIDES,
    num_classes: int = NUM_CLASSES,
    person_class: int = 0,
    scale: float = 1.0,
) -> np.ndarray:
    """Encode ``(x1, y1, x2, y2, score)`` boxes as a YOLOX prediction table.

    ``scale`` is the letterbox factor the adapter will invert, so boxes given in
    *source image* pixels come back out of the decoder in source pixels.
    """
    rows = _grid_rows(input_size, strides)
    table = np.zeros((len(rows), 5 + num_classes), dtype=np.float32)
    table[:, 4] = 0.0  # objectness: nothing detected anywhere by default

    for slot, (x1, y1, x2, y2, score) in enumerate(boxes):
        # Put each box on its own row of the finest stride level.
        row = slot
        stride, gx, gy = rows[row]
        cx = (x1 + x2) / 2.0 * scale
        cy = (y1 + y2) / 2.0 * scale
        w = max(x2 - x1, 1e-3) * scale
        h = max(y2 - y1, 1e-3) * scale
        table[row, 0] = cx / stride - gx
        table[row, 1] = cy / stride - gy
        table[row, 2] = np.log(w / stride)
        table[row, 3] = np.log(h / stride)
        table[row, 4] = 1.0
        table[row, 5 + person_class] = score
    return table[None, ...]


def simcc_output(
    points: list[tuple[float, float]],
    scores: list[float],
    *,
    input_size: tuple[int, int] = POSE_INPUT,
    split_ratio: float = SPLIT_RATIO,
) -> list[np.ndarray]:
    """Encode model-input-space keypoints as a pair of SimCC maps."""
    count = len(points)
    bins_x = int(input_size[0] * split_ratio)
    bins_y = int(input_size[1] * split_ratio)
    simcc_x = np.full((1, count, bins_x), -1.0, dtype=np.float32)
    simcc_y = np.full((1, count, bins_y), -1.0, dtype=np.float32)
    for index, ((x, y), score) in enumerate(zip(points, scores, strict=True)):
        ix = int(np.clip(round(x * split_ratio), 0, bins_x - 1))
        iy = int(np.clip(round(y * split_ratio), 0, bins_y - 1))
        simcc_x[0, index, ix] = score
        simcc_y[0, index, iy] = score
    return [simcc_x, simcc_y]


def canonical_body_points(
    *, count: int = 17, spread: float = 1.0, offset: tuple[float, float] = (0.0, 0.0)
) -> list[tuple[float, float]]:
    """17 distinguishable keypoints in pose-model input space.

    Every joint gets a *different* coordinate, so a mapping test can tell a
    left ear from a right eye. The layout is roughly body-shaped, which keeps
    the crop inversion honest, but its exact values do not matter — only that
    they are unique.
    """
    layout = {
        "nose": (24.0, 8.0),
        "left_eye": (21.0, 6.0),
        "right_eye": (27.0, 6.5),
        "left_ear": (18.0, 7.0),
        "right_ear": (30.0, 7.5),
        "left_shoulder": (14.0, 18.0),
        "right_shoulder": (34.0, 18.5),
        "left_elbow": (10.0, 28.0),
        "right_elbow": (38.0, 28.5),
        "left_wrist": (7.0, 38.0),
        "right_wrist": (41.0, 38.5),
        "left_hip": (17.0, 36.0),
        "right_hip": (31.0, 36.5),
        "left_knee": (16.0, 48.0),
        "right_knee": (32.0, 48.5),
        "left_ankle": (15.0, 58.0),
        "right_ankle": (33.0, 58.5),
    }
    body = [
        (layout[name][0] * spread + offset[0], layout[name][1] * spread + offset[1])
        for name in BODY_JOINTS
    ]
    extra = [(1.0 + i * 0.25, 2.0 + i * 0.25) for i in range(max(0, count - len(body)))]
    return body + extra


# ---------------------------------------------------------------------------
# frame reader fake
# ---------------------------------------------------------------------------
@dataclass
class FakeFrameReader:
    """Yields synthetic frames for exactly the indices requested."""

    width: int = 360
    height: int = 640
    #: Indices to pretend are unreadable, to exercise short-clip handling.
    truncate_after: int | None = None
    reads: list[list[int]] = field(default_factory=list)

    def read(self, video_path: Path, frame_indices: list[int]) -> Iterator[tuple[int, np.ndarray]]:
        self.reads.append(list(frame_indices))
        for index in sorted(frame_indices):
            if self.truncate_after is not None and index > self.truncate_after:
                return
            frame = np.full((self.height, self.width, 3), 32, dtype=np.uint8)
            # A per-frame tint, so a test that mixes frames up can notice.
            frame[:, :, 0] = index % 251
            yield index, frame


def touch_models(tmp_path: Path) -> tuple[Path, Path]:
    """Create placeholder .onnx files so path validation passes.

    Their contents are never read: the session factory is injected. This is
    exactly the shape of the real check — does a file exist at the configured
    path — without any weights.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    detector = tmp_path / "yolox_l_fake.onnx"
    pose = tmp_path / "dw_ll_ucoco_fake.onnx"
    detector.write_bytes(b"fake-detector-not-a-model\n")
    pose.write_bytes(b"fake-pose-not-a-model\n")
    return detector, pose


__all__ = [
    "DETECTOR_INPUT",
    "DETECTOR_STRIDES",
    "POSE_INPUT",
    "SPLIT_RATIO",
    "FakeFrameReader",
    "FakeSession",
    "FakeSessionFactory",
    "canonical_body_points",
    "cuda_factory",
    "simcc_output",
    "touch_models",
    "yolox_output",
]
