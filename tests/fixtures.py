"""Programmatic synthetic fixtures.

Everything the tests need is generated here in code: a small "master
performance" as a frame sequence (and optionally a real video file), mask sets,
and garment reference images. No binary fixtures are committed, which keeps the
repository free of media and makes the fixtures self-documenting.

Frames are deliberately tiny (96x160) so the whole suite runs in seconds while
still exercising the real code paths, including the 1080x1920-style vertical
aspect ratio.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.config import AppConfig, load_config
from app.core.hashing import sha256_dir, sha256_file
from app.domain.enums import (
    BodyCoverage,
    GarmentCategory,
    GarmentLength,
    ImageViewType,
    IngestionStatus,
    MaskKind,
    Material,
    ProcessingStatus,
    Silhouette,
    SleeveLength,
    TemplateClothingClass,
)
from app.domain.garment import GarmentAsset, GarmentReferenceImage, UsageRights
from app.domain.human_template import (
    ConsentRecord,
    FrameRange,
    HumanTemplate,
    TemplateDirectories,
    VideoSpec,
)
from app.media.frames import frame_path, write_frame
from app.media.masks import save_mask
from app.pipeline.context import ServiceContext

FRAME_WIDTH = 96
FRAME_HEIGHT = 160
FPS = 10.0
TOTAL_FRAMES = 24
ANCHOR = 12

#: Regions of the synthetic frame, as (y0, y1, x0, x1).
FACE_REGION = (8, 34, 34, 62)
GARMENT_REGION = (52, 104, 24, 72)
HAND_REGION = (74, 86, 30, 42)
LEG_REGION = (104, 150, 34, 62)


def test_config(data_root: Path, **overrides: Any) -> AppConfig:
    """A config pinned to a temp data root and the synthetic frame size."""
    base: dict[str, Any] = {
        "paths": {"data_root": str(data_root)},
        "video": {
            "width": FRAME_WIDTH,
            "height": FRAME_HEIGHT,
            "fps": FPS,
            "preset": "ultrafast",
            "crf": 20,
        },
        "mask": {"feather_radius_px": 3, "protected_dilate_px": 1},
        "runtime": {"log_to_file": False, "log_level": "WARNING"},
        "backend": {"frame_window": 4},
        "qc": {"contact_sheet_period_frames": 4, "contact_sheet_transition_span": 3},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return load_config(overrides=base, use_env=False)


# ---------------------------------------------------------------------------
# frame synthesis
# ---------------------------------------------------------------------------
def synth_frame(index: int, *, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT) -> np.ndarray:
    """A deterministic 'performance' frame: moving background + body regions.

    The content moves every frame so frozen-frame and flicker checks have real
    motion to measure, and every region has a distinct colour so a leak into
    one of them is obvious.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Background: a smooth gradient that drifts with the frame index.
    ys = np.arange(height, dtype=np.float32)[:, None]
    xs = np.arange(width, dtype=np.float32)[None, :]
    drift = index * 3.0
    background = 90 + 40 * np.sin((xs + drift) / 11.0) + 25 * np.cos((ys - drift) / 17.0)
    frame[:, :, 0] = np.clip(background, 0, 255).astype(np.uint8)
    frame[:, :, 1] = np.clip(background * 0.7 + 20, 0, 255).astype(np.uint8)
    frame[:, :, 2] = np.clip(background * 0.5 + 40, 0, 255).astype(np.uint8)

    sway = round(2.5 * np.sin(index / 2.0))

    def fill(region: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
        y0, y1, x0, x1 = region
        frame[y0:y1, x0 + sway : x1 + sway] = color

    fill(LEG_REGION, (150, 168, 196))  # skin (legs)
    fill(GARMENT_REGION, (40, 40, 140))  # base garment (dark red-ish in BGR)
    fill(FACE_REGION, (170, 190, 215))  # face
    fill(HAND_REGION, (146, 165, 192))  # hand crossing the garment

    # Hair: a band above the face that also moves.
    y0, _, x0, x1 = FACE_REGION
    frame[max(0, y0 - 8) : y0, x0 + sway - 2 : x1 + sway + 2] = (30, 35, 55)
    return frame


def _region_mask(region: tuple[int, int, int, int], index: int) -> np.ndarray:
    mask = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
    y0, y1, x0, x1 = region
    sway = round(2.5 * np.sin(index / 2.0))
    mask[y0:y1, max(0, x0 + sway) : x1 + sway] = 255
    return mask


def garment_mask(index: int) -> np.ndarray:
    """The base garment region, minus the hand that crosses it."""
    return _region_mask(GARMENT_REGION, index)


def protected_mask(index: int) -> np.ndarray:
    """Face + hair + hands + legs (skin). Never editable."""
    mask = _region_mask(FACE_REGION, index)
    hair = np.zeros_like(mask)
    y0, _, x0, x1 = FACE_REGION
    sway = round(2.5 * np.sin(index / 2.0))
    hair[max(0, y0 - 8) : y0, max(0, x0 + sway - 2) : x1 + sway + 2] = 255
    return np.maximum(
        np.maximum(mask, hair),
        np.maximum(_region_mask(HAND_REGION, index), _region_mask(LEG_REGION, index)),
    )


def occlusion_mask(index: int) -> np.ndarray:
    """The hand in front of the garment."""
    return _region_mask(HAND_REGION, index)


def expansion_mask(index: int) -> np.ndarray:
    """A modest widening of the garment region (longer sleeves / fuller hem)."""
    y0, y1, x0, x1 = GARMENT_REGION
    return _region_mask((y0 - 4, min(FRAME_HEIGHT, y1 + 6), max(0, x0 - 6), x1 + 6), index)


# ---------------------------------------------------------------------------
# template fixtures
# ---------------------------------------------------------------------------
@dataclass
class TemplateFixture:
    template: HumanTemplate
    frames_dir: Path
    mask_dirs: dict[MaskKind, Path]


def write_template_frames(
    context: ServiceContext,
    directories: TemplateDirectories,
    *,
    total_frames: int = TOTAL_FRAMES,
    with_masks: tuple[MaskKind, ...] = (MaskKind.GARMENT, MaskKind.PROTECTED, MaskKind.OCCLUSION),
    mask_range: tuple[int, int] | None = None,
) -> dict[MaskKind, Path]:
    """Write synthetic source frames and the requested mask families."""
    frames_dir = context.absolute(directories.source_frames)
    frames_dir.mkdir(parents=True, exist_ok=True)
    for index in range(total_frames):
        write_frame(frame_path(frames_dir, index), synth_frame(index))

    builders = {
        MaskKind.GARMENT: garment_mask,
        MaskKind.PROTECTED: protected_mask,
        MaskKind.OCCLUSION: occlusion_mask,
        MaskKind.EXPANSION: expansion_mask,
    }
    start, end = mask_range or (0, total_frames)
    out: dict[MaskKind, Path] = {}
    for kind in MaskKind:
        directory = context.absolute(directories.mask_dir(kind))
        directory.mkdir(parents=True, exist_ok=True)
        out[kind] = directory
        if kind not in with_masks:
            continue
        for index in range(start, end):
            save_mask(frame_path(directory, index), builders[kind](index))
    return out


def make_template(
    context: ServiceContext,
    *,
    template_id: str = "tpl_test",
    total_frames: int = TOTAL_FRAMES,
    anchor: int = ANCHOR,
    clothing_class: TemplateClothingClass = TemplateClothingClass.FITTED_SHORT,
    with_masks: tuple[MaskKind, ...] = (MaskKind.GARMENT, MaskKind.PROTECTED, MaskKind.OCCLUSION),
    mask_range: tuple[int, int] | None = None,
    status: ProcessingStatus = ProcessingStatus.READY,
    source_video: Path | None = None,
) -> TemplateFixture:
    """Build a ready-to-render template without needing ffmpeg.

    ``ingest_template`` is exercised separately (it needs ffprobe); this builder
    produces the same on-disk shape so the rest of the pipeline can be tested
    with no external tools at all.
    """
    directories = TemplateDirectories.standard(f"templates/{template_id}")
    for relative in directories.all_dirs():
        context.absolute(relative).mkdir(parents=True, exist_ok=True)
    mask_dirs = write_template_frames(
        context,
        directories,
        total_frames=total_frames,
        with_masks=with_masks,
        mask_range=mask_range,
    )
    frames_dir = context.absolute(directories.source_frames)

    if source_video is None:
        # A stand-in "source video" file so source-hash checks have something
        # real to hash; ingestion tests use an actual encoded video instead.
        source_video = context.absolute(directories.root) / "source_placeholder.bin"
        source_video.write_bytes(b"synthetic-master-performance\n")

    template = HumanTemplate(
        id=template_id,
        version=1,
        display_name="Synthetic Test Performance",
        source_video_path=str(source_video),
        source_sha256=sha256_file(source_video),
        video=VideoSpec(
            width=FRAME_WIDTH,
            height=FRAME_HEIGHT,
            fps=FPS,
            duration_s=total_frames / FPS,
            frame_count=total_frames,
            codec="h264",
            pixel_format="yuv420p",
            constant_frame_rate=True,
            avg_frame_rate=f"{int(FPS)}/1",
            r_frame_rate=f"{int(FPS)}/1",
            has_audio=False,
        ),
        intro=FrameRange(start=0, end=anchor),
        reveal=FrameRange(start=anchor, end=total_frames),
        transition_anchor_frame=anchor,
        consent=ConsentRecord(subject_kind="synthetic", adult_confirmed=True),
        template_clothing_class=clothing_class,
        directories=directories,
        source_frames_sha256=sha256_dir(frames_dir),
        extracted_frame_count=total_frames,
        status=status,
    )
    saved = context.repos.templates.save(template)
    return TemplateFixture(template=saved, frames_dir=frames_dir, mask_dirs=mask_dirs)


# ---------------------------------------------------------------------------
# garment fixtures
# ---------------------------------------------------------------------------
def write_garment_image(
    path: Path,
    *,
    color: tuple[int, int, int] = (60, 180, 240),
    width: int = 900,
    height: int = 1200,
    detail: bool = True,
) -> Path:
    """A synthetic 'product photo': flat background plus a garment shape.

    Sharp edges are drawn on purpose so the variance-of-Laplacian sharpness
    measurement lands above the quality threshold.
    """
    image = np.full((height, width, 3), 240, dtype=np.uint8)
    cv2.rectangle(
        image,
        (int(width * 0.18), int(height * 0.12)),
        (int(width * 0.82), int(height * 0.86)),
        color,
        -1,
    )
    if detail:
        for offset in range(0, width, 40):
            cv2.line(image, (offset, 0), (offset, height), (20, 20, 20), 2)
        cv2.rectangle(
            image,
            (int(width * 0.30), int(height * 0.20)),
            (int(width * 0.70), int(height * 0.30)),
            (255, 255, 255),
            -1,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    return path


def make_garment(
    context: ServiceContext,
    *,
    garment_id: str = "grm_test",
    views: tuple[ImageViewType, ...] = (
        ImageViewType.FRONT,
        ImageViewType.BACK,
        ImageViewType.SIDE,
    ),
    category: GarmentCategory = GarmentCategory.TOP,
    coverage: BodyCoverage = BodyCoverage.TORSO,
    sleeve: SleeveLength = SleeveLength.SHORT,
    length: GarmentLength = GarmentLength.MID_THIGH,
    silhouette: Silhouette = Silhouette.FITTED,
    material: Material = Material.COTTON,
    transparency: float = 0.0,
    reflectivity: float = 0.1,
    fabric_flow: float = 0.2,
    colors: tuple[str, ...] = ("#3cb4f0", "#141414"),
    image_size: tuple[int, int] = (900, 1200),
    status: IngestionStatus = IngestionStatus.READY,
    requires_underlayer: bool = False,
) -> GarmentAsset:
    """Build a garment asset with real image files on disk."""
    images_dir = context.data_root.garment_dir(garment_id) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    records: list[GarmentReferenceImage] = []
    palette = {
        ImageViewType.FRONT: (60, 180, 240),
        ImageViewType.BACK: (80, 160, 220),
        ImageViewType.SIDE: (70, 170, 230),
        ImageViewType.DETAIL: (90, 190, 250),
        ImageViewType.FLAT_LAY: (100, 200, 255),
    }
    for view in views:
        path = images_dir / f"{view.value}.png"
        write_garment_image(path, color=palette[view], width=image_size[0], height=image_size[1])
        image = cv2.imread(str(path))
        records.append(
            GarmentReferenceImage(
                path=context.relative(path),
                view=view,
                sha256=sha256_file(path),
                width=image.shape[1],
                height=image.shape[0],
                sharpness_score=0.9,
                subject_area_fraction=0.5,
            )
        )

    garment = GarmentAsset(
        id=garment_id,
        version=1,
        product_name="Synthetic Test Top",
        brand="TestBrand",
        images=records,
        category=category,
        sleeve_length=sleeve,
        garment_length=length,
        body_coverage=coverage,
        silhouette=silhouette,
        material=material,
        transparency=transparency,
        reflectivity=reflectivity,
        fabric_flow=fabric_flow,
        dominant_colors=list(colors),
        pattern_description="vertical stripes",
        requires_underlayer=requires_underlayer,
        usage_rights=UsageRights(
            license="test-only", rights_holder="Test Fixture", commercial_use_allowed=False
        ),
        status=status,
    )
    return context.repos.garments.save(garment)


# ---------------------------------------------------------------------------
# video synthesis (needs ffmpeg; used only by ingestion/compose tests)
# ---------------------------------------------------------------------------
def write_synthetic_video(
    destination: Path,
    *,
    total_frames: int = TOTAL_FRAMES,
    fps: float = FPS,
    with_audio: bool = False,
    ffmpeg_binary: str = "ffmpeg",
) -> Path:
    """Encode the synthetic frames into a real CFR video file."""
    from app.media import ffmpeg as ffmpeg_module

    staging = destination.parent / f".{destination.stem}_frames"
    staging.mkdir(parents=True, exist_ok=True)
    for index in range(total_frames):
        write_frame(frame_path(staging, index), synth_frame(index))

    destination.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-framerate",
        str(int(fps)),
        "-start_number",
        "0",
        "-i",
        str(staging / "frame_%06d.png"),
    ]
    if with_audio:
        argv += [
            "-f",
            "lavfi",
            "-t",
            f"{total_frames / fps:.6f}",
            "-i",
            "sine=frequency=440:sample_rate=48000",
        ]
    argv += [
        "-c:v",
        "libx264",
        "-crf",
        "14",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-vsync",
        "cfr",
        "-r",
        str(int(fps)),
    ]
    if with_audio:
        argv += ["-c:a", "aac", "-b:a", "96k", "-shortest"]
    argv += [str(destination)]
    ffmpeg_module.run_command(argv)
    return destination


__all__ = [
    "ANCHOR",
    "FACE_REGION",
    "FPS",
    "FRAME_HEIGHT",
    "FRAME_WIDTH",
    "GARMENT_REGION",
    "HAND_REGION",
    "LEG_REGION",
    "TOTAL_FRAMES",
    "ImageViewType",
    "TemplateFixture",
    "expansion_mask",
    "garment_mask",
    "make_garment",
    "make_template",
    "occlusion_mask",
    "protected_mask",
    "synth_frame",
    "test_config",
    "write_garment_image",
    "write_synthetic_video",
    "write_template_frames",
]
