"""Person detection: letterbox, run the session, decode, NMS, keep people.

The shipped default targets the YOLOX-style detector DWPose is normally paired
with, whose ONNX export returns grid-relative predictions that must be decoded
against the stride pyramid. A detector that already returns decoded boxes is
supported by setting ``pose.detector_layout: boxes``.

Everything here is plain NumPy on arrays the session returned, so a fake
session in a test exercises exactly the same decode path a real one does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.core.errors import ValidationError


@dataclass(frozen=True)
class Detection:
    """One person box in the coordinates of the image handed to the detector."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def scaled(self, factor: float, offset: tuple[float, float] = (0.0, 0.0)) -> Detection:
        return Detection(
            x1=self.x1 * factor + offset[0],
            y1=self.y1 * factor + offset[1],
            x2=self.x2 * factor + offset[0],
            y2=self.y2 * factor + offset[1],
            score=self.score,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "x2": round(self.x2, 2),
            "y2": round(self.y2, 2),
            "score": round(self.score, 4),
        }


@dataclass(frozen=True)
class LetterboxInfo:
    """How an image was fitted into the detector's square input."""

    scale: float
    pad_x: int
    pad_y: int


def letterbox(image: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, LetterboxInfo]:
    """Resize preserving aspect ratio onto a fixed canvas, padding bottom-right.

    Top-left anchoring (rather than centring) is what YOLOX's own preprocessing
    does, and it makes the inverse transform a single scale factor with no
    offset — one fewer place for a coordinate bug to hide.
    """
    import cv2

    target_w, target_h = size
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValidationError("Cannot letterbox an empty image", shape=list(image.shape))
    scale = min(target_w / width, target_h / height)
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    canvas[:new_h, :new_w] = resized[:, :, :3]
    return canvas, LetterboxInfo(scale=scale, pad_x=0, pad_y=0)


def decode_yolox(
    raw: np.ndarray,
    *,
    input_size: tuple[int, int],
    strides: tuple[int, ...],
) -> np.ndarray:
    """Decode YOLOX grid-relative output into ``(N, 4 + 1 + C)`` xyxy rows.

    Input rows are ``[dx, dy, dw, dh, objectness, *class_scores]`` where the
    offsets are relative to each cell of the stride pyramid.
    """
    predictions = np.asarray(raw, dtype=np.float32)
    if predictions.ndim == 3:
        predictions = predictions[0]
    if predictions.ndim != 2 or predictions.shape[1] < 6:
        raise ValidationError(
            "Detector output is not a YOLOX prediction table",
            shape=list(np.asarray(raw).shape),
            expected="(N, 5 + num_classes)",
        )

    width, height = input_size
    grids: list[np.ndarray] = []
    expanded: list[np.ndarray] = []
    for stride in strides:
        gw, gh = width // stride, height // stride
        xv, yv = np.meshgrid(np.arange(gw), np.arange(gh))
        grid = np.stack((xv, yv), axis=2).reshape(1, -1, 2)
        grids.append(grid)
        expanded.append(np.full((1, grid.shape[1], 1), stride, dtype=np.float32))

    grid = np.concatenate(grids, axis=1)[0].astype(np.float32)
    stride_column = np.concatenate(expanded, axis=1)[0].astype(np.float32)
    if grid.shape[0] != predictions.shape[0]:
        raise ValidationError(
            "Detector output row count does not match the stride pyramid for "
            "this input size; check pose.detector_input_size and detector_strides",
            rows=int(predictions.shape[0]),
            expected_rows=int(grid.shape[0]),
            input_size=list(input_size),
            strides=list(strides),
        )

    decoded = predictions.copy()
    decoded[:, 0:2] = (predictions[:, 0:2] + grid) * stride_column
    decoded[:, 2:4] = np.exp(np.clip(predictions[:, 2:4], -20.0, 20.0)) * stride_column

    boxes = np.empty_like(decoded[:, :4])
    boxes[:, 0] = decoded[:, 0] - decoded[:, 2] / 2.0
    boxes[:, 1] = decoded[:, 1] - decoded[:, 3] / 2.0
    boxes[:, 2] = decoded[:, 0] + decoded[:, 2] / 2.0
    boxes[:, 3] = decoded[:, 1] + decoded[:, 3] / 2.0
    return np.concatenate([boxes, decoded[:, 4:]], axis=1)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy non-maximum suppression. Ties break by index, so it is stable."""
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    # Stable ordering: -score first, then index, so equal scores are resolved
    # deterministically rather than by NumPy's sort implementation.
    order = np.lexsort((np.arange(len(scores)), -scores))
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[current], x1[rest])
        yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest])
        yy2 = np.minimum(y2[current], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[current] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        order = rest[iou <= iou_threshold]
    return keep


def detections_from_output(
    raw: np.ndarray,
    *,
    layout: str,
    input_size: tuple[int, int],
    strides: tuple[int, ...],
    person_class: int,
    score_threshold: float,
    iou_threshold: float,
    max_detections: int,
    letterbox_info: LetterboxInfo,
) -> list[Detection]:
    """Full decode: raw session output -> person boxes in image coordinates."""
    if layout == "yolox":
        table = decode_yolox(raw, input_size=input_size, strides=strides)
    else:
        table = np.asarray(raw, dtype=np.float32)
        if table.ndim == 3:
            table = table[0]
        if table.ndim != 2 or table.shape[1] < 5:
            raise ValidationError(
                "Detector output is not an (N, >=5) box table",
                shape=list(np.asarray(raw).shape),
            )

    boxes = table[:, :4]
    objectness = table[:, 4]
    if table.shape[1] > 5:
        class_scores = table[:, 5:]
        if person_class >= class_scores.shape[1]:
            raise ValidationError(
                "detector_person_class is outside the model's class range",
                person_class=person_class,
                classes=int(class_scores.shape[1]),
            )
        scores = objectness * class_scores[:, person_class]
    else:
        # Single-class detector: objectness is the person score.
        scores = objectness

    mask = scores >= score_threshold
    boxes, scores = boxes[mask], scores[mask]
    if boxes.size == 0:
        return []

    keep = nms(boxes, scores, iou_threshold)[:max_detections]
    inverse = 1.0 / max(letterbox_info.scale, 1e-9)
    out: list[Detection] = []
    for index in keep:
        x1, y1, x2, y2 = (float(v) for v in boxes[index])
        out.append(
            Detection(
                x1=(x1 - letterbox_info.pad_x) * inverse,
                y1=(y1 - letterbox_info.pad_y) * inverse,
                x2=(x2 - letterbox_info.pad_x) * inverse,
                y2=(y2 - letterbox_info.pad_y) * inverse,
                score=float(scores[index]),
            )
        )
    return out


def preprocess(image: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, LetterboxInfo]:
    """Letterbox to the detector input and produce an NCHW float32 batch."""
    canvas, info = letterbox(image, size)
    batch = canvas.transpose(2, 0, 1)[None, ...].astype(np.float32)
    return batch, info


__all__ = [
    "Detection",
    "LetterboxInfo",
    "decode_yolox",
    "detections_from_output",
    "letterbox",
    "nms",
    "preprocess",
]
