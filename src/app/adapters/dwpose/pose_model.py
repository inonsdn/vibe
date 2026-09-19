"""Keypoint inference: crop around a person box, run the model, decode, map back.

DWPose/RTMPose exports use SimCC — two 1-D classification maps per keypoint,
one for x and one for y, at ``simcc_split_ratio`` bins per input pixel. Other
common exports return heatmaps or keypoints directly, so the output shape is
inspected and the matching decoder used. Guessing wrong here would put joints
in plausible-looking but wrong places, so an unrecognised shape is an error.

All three decoders return coordinates in *model input* pixels;
:func:`keypoints_to_image` is the single place that maps them back to the image
the crop came from.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.core.errors import ValidationError


@dataclass(frozen=True)
class CropInfo:
    """The rectangle a person was cropped from, in source-image pixels."""

    x: float
    y: float
    width: float
    height: float
    input_size: tuple[int, int]

    @property
    def scale_x(self) -> float:
        return self.width / max(self.input_size[0], 1)

    @property
    def scale_y(self) -> float:
        return self.height / max(self.input_size[1], 1)


def expand_box(
    box: tuple[float, float, float, float],
    *,
    input_size: tuple[int, int],
    padding: float,
) -> CropInfo:
    """Pad a detection box out to the model's aspect ratio, as DWPose does.

    Keeping the model's aspect ratio matters: squeezing a tall dancer into a
    square input distorts limb angles, and the distortion does not undo cleanly
    because the inverse is applied per-axis.
    """
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    width = max(x2 - x1, 1.0) * padding
    height = max(y2 - y1, 1.0) * padding

    target_w, target_h = input_size
    aspect = target_w / max(target_h, 1)
    if width / max(height, 1e-6) > aspect:
        height = width / aspect
    else:
        width = height * aspect
    return CropInfo(
        x=cx - width / 2.0, y=cy - height / 2.0, width=width, height=height, input_size=input_size
    )


def crop_and_resize(image: np.ndarray, crop: CropInfo) -> np.ndarray:
    """Cut the (possibly out-of-bounds) crop and resize to the model input.

    The crop is padded rather than clipped, because clipping would change the
    rectangle and therefore the inverse mapping. A dancer at the edge of the
    frame is a normal case in a phone recording.
    """
    import cv2

    height, width = image.shape[:2]
    x1, y1 = int(np.floor(crop.x)), int(np.floor(crop.y))
    x2, y2 = int(np.ceil(crop.x + crop.width)), int(np.ceil(crop.y + crop.height))

    pad_left, pad_top = max(0, -x1), max(0, -y1)
    pad_right, pad_bottom = max(0, x2 - width), max(0, y2 - height)
    sliced = image[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2), :3]
    if sliced.size == 0:
        sliced = np.zeros((1, 1, 3), dtype=image.dtype)
    if pad_left or pad_top or pad_right or pad_bottom:
        sliced = cv2.copyMakeBorder(
            sliced, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
    return cv2.resize(sliced, crop.input_size, interpolation=cv2.INTER_LINEAR)


def preprocess(
    image: np.ndarray,
    crop: CropInfo,
    *,
    mean: tuple[float, float, float] = (123.675, 116.28, 103.53),
    std: tuple[float, float, float] = (58.395, 57.12, 57.375),
) -> np.ndarray:
    """Crop, resize and normalise into an NCHW float32 batch."""
    patch = crop_and_resize(image, crop).astype(np.float32)
    normalised = (
        (patch - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    ).astype(np.float32)
    batch: np.ndarray = normalised.transpose(2, 0, 1)[None, ...]
    return batch


def decode_simcc(
    simcc_x: np.ndarray, simcc_y: np.ndarray, *, split_ratio: float
) -> tuple[np.ndarray, np.ndarray]:
    """SimCC argmax -> ``(K, 2)`` input-space coordinates and ``(K,)`` scores."""
    x = np.asarray(simcc_x, dtype=np.float32)
    y = np.asarray(simcc_y, dtype=np.float32)
    if x.ndim == 3:
        x = x[0]
    if y.ndim == 3:
        y = y[0]
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValidationError(
            "SimCC outputs do not have matching (K, bins) shapes",
            simcc_x=list(np.asarray(simcc_x).shape),
            simcc_y=list(np.asarray(simcc_y).shape),
        )

    ix, iy = np.argmax(x, axis=1), np.argmax(y, axis=1)
    vx, vy = np.max(x, axis=1), np.max(y, axis=1)
    # mmpose takes the smaller of the two axis confidences: a keypoint is only
    # as certain as its least certain coordinate.
    scores = np.minimum(vx, vy)
    coords = np.stack([ix / split_ratio, iy / split_ratio], axis=1).astype(np.float32)
    # A negative peak means the model found nothing; mmpose zeroes those.
    coords[scores <= 0.0] = 0.0
    return coords, np.clip(scores, 0.0, 1.0).astype(np.float32)


def decode_heatmaps(
    heatmaps: np.ndarray, input_size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """``(1, K, H, W)`` heatmaps -> input-space coordinates and scores."""
    maps = np.asarray(heatmaps, dtype=np.float32)
    if maps.ndim == 4:
        maps = maps[0]
    if maps.ndim != 3:
        raise ValidationError("Heatmap output must be (K, H, W)", shape=list(maps.shape))
    count, hm_h, hm_w = maps.shape
    flat = maps.reshape(count, -1)
    index = np.argmax(flat, axis=1)
    scores = np.max(flat, axis=1)
    ys, xs = np.divmod(index, hm_w)
    scale_x = input_size[0] / max(hm_w, 1)
    scale_y = input_size[1] / max(hm_h, 1)
    coords = np.stack([xs * scale_x, ys * scale_y], axis=1).astype(np.float32)
    return coords, np.clip(scores, 0.0, 1.0).astype(np.float32)


def decode_keypoints(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(1, K, 3)`` ``[x, y, score]`` output -> coordinates and scores."""
    data = np.asarray(array, dtype=np.float32)
    if data.ndim == 3:
        data = data[0]
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValidationError("Keypoint output must be (K, 3)", shape=list(data.shape))
    return data[:, :2].astype(np.float32), np.clip(data[:, 2], 0.0, 1.0).astype(np.float32)


def decode_outputs(
    outputs: list[np.ndarray],
    *,
    input_size: tuple[int, int],
    split_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick the decoder that matches what this model actually returned."""
    arrays = [np.asarray(o) for o in outputs]
    if len(arrays) >= 2 and arrays[0].ndim in (2, 3) and arrays[1].ndim in (2, 3):
        return decode_simcc(arrays[0], arrays[1], split_ratio=split_ratio)
    if len(arrays) == 1:
        single = arrays[0]
        if single.ndim == 4:
            return decode_heatmaps(single, input_size)
        if single.ndim == 3 and single.shape[-1] == 3:
            return decode_keypoints(single)
        if single.ndim == 2 and single.shape[-1] == 3:
            return decode_keypoints(single[None, ...])
    raise ValidationError(
        "Unrecognised pose model output. Supported: two SimCC maps, one "
        "(1, K, H, W) heatmap tensor, or one (1, K, 3) keypoint tensor.",
        shapes=[list(a.shape) for a in arrays],
    )


def keypoints_to_image(coords: np.ndarray, crop: CropInfo) -> np.ndarray:
    """Map model-input coordinates back to the image the crop was taken from."""
    mapped = np.empty_like(coords, dtype=np.float32)
    mapped[:, 0] = coords[:, 0] * crop.scale_x + crop.x
    mapped[:, 1] = coords[:, 1] * crop.scale_y + crop.y
    return mapped


__all__ = [
    "CropInfo",
    "crop_and_resize",
    "decode_heatmaps",
    "decode_keypoints",
    "decode_outputs",
    "decode_simcc",
    "expand_box",
    "keypoints_to_image",
    "preprocess",
]
