"""Template ingestion: requirement 1 — the source video hash is preserved."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.core.errors import ImmutabilityError, NotFoundError, ValidationError
from app.core.hashing import sha256_dir, sha256_file
from app.domain.enums import MaskKind, ProcessingStatus
from app.media.frames import frame_path, list_frame_indices, read_frame, write_frame
from app.media.masks import save_mask
from app.pipeline.template_ingest import (
    IngestOptions,
    import_masks,
    ingest_template,
    validate_template,
    verify_source_immutability,
)
from tests import fixtures
from tests.conftest import requires_ffmpeg

pytestmark = requires_ffmpeg


def ingest(context, video: Path, **kwargs):
    options = IngestOptions(
        display_name=kwargs.pop("display_name", "Master Performance"),
        transition_anchor=kwargs.pop("transition_anchor", fixtures.ANCHOR),
        **kwargs,
    )
    return ingest_template(context, video, options)


def test_ingestion_records_the_source_hash(context, synthetic_video: Path) -> None:
    """Requirement 1."""
    expected = sha256_file(synthetic_video)
    result = ingest(context, synthetic_video)
    assert result.template.source_sha256 == expected

    # Re-reading through the repository gives the same hash.
    stored = context.repos.templates.get(result.template.id)
    assert stored.source_sha256 == expected
    # And the file itself was not touched.
    assert sha256_file(synthetic_video) == expected


def test_ingestion_extracts_a_contiguous_frame_sequence(context, synthetic_video: Path) -> None:
    result = ingest(context, synthetic_video)
    frames_dir = context.absolute(result.template.directories.source_frames)
    indices = list_frame_indices(frames_dir)
    assert indices == list(range(fixtures.TOTAL_FRAMES))
    assert result.template.extracted_frame_count == fixtures.TOTAL_FRAMES

    frame = read_frame(frame_path(frames_dir, 0))
    assert frame.shape[:2] == (fixtures.FRAME_HEIGHT, fixtures.FRAME_WIDTH)


def test_probed_video_properties_are_measured_not_assumed(context, synthetic_video: Path) -> None:
    result = ingest(context, synthetic_video)
    video = result.template.video
    assert (video.width, video.height) == (fixtures.FRAME_WIDTH, fixtures.FRAME_HEIGHT)
    assert video.fps == pytest.approx(fixtures.FPS, abs=0.01)
    assert video.frame_count == fixtures.TOTAL_FRAMES
    assert video.constant_frame_rate is True
    assert video.codec == "h264"
    assert result.probe["streams"]


def test_source_frames_hash_is_recorded_and_enforced(context, synthetic_video: Path) -> None:
    result = ingest(context, synthetic_video)
    frames_dir = context.absolute(result.template.directories.source_frames)
    assert result.template.source_frames_sha256 == sha256_dir(frames_dir)
    verify_source_immutability(context, result.template)

    tampered = read_frame(frame_path(frames_dir, 1))
    tampered[0, 0] = (9, 9, 9)
    write_frame(frame_path(frames_dir, 1), tampered)
    with pytest.raises(ImmutabilityError):
        verify_source_immutability(context, result.template)


def test_segmentation_is_configurable_and_validated(context, synthetic_video: Path) -> None:
    result = ingest(context, synthetic_video, transition_anchor=8, reveal_end=20)
    template = result.template
    assert template.intro.start == 0
    assert template.intro.end == 8
    assert template.transition_anchor_frame == 8
    assert template.reveal.start == 8
    assert template.reveal.end == 20


def test_anchor_beyond_the_frame_count_is_rejected(context, synthetic_video: Path) -> None:
    with pytest.raises(ValidationError, match="reveal_end"):
        ingest(context, synthetic_video, transition_anchor=fixtures.TOTAL_FRAMES + 5)


def test_anchor_at_zero_is_rejected(context, synthetic_video: Path) -> None:
    with pytest.raises(ValidationError, match="after the intro start"):
        ingest(context, synthetic_video, transition_anchor=0)


def test_reveal_end_beyond_extracted_frames_is_rejected(context, synthetic_video: Path) -> None:
    with pytest.raises(ValidationError, match="exceeds the number of extracted frames"):
        ingest(context, synthetic_video, reveal_end=fixtures.TOTAL_FRAMES + 1)


def test_missing_source_video_is_reported(context, tmp_path: Path) -> None:
    with pytest.raises(NotFoundError):
        ingest(context, tmp_path / "nope.mp4")


def test_mask_directories_are_created_with_readmes(context, synthetic_video: Path) -> None:
    result = ingest(context, synthetic_video)
    for kind in MaskKind:
        directory = context.absolute(result.template.directories.mask_dir(kind))
        assert directory.is_dir()
        readme = directory / "README.txt"
        assert readme.is_file()
        text = readme.read_text(encoding="utf-8")
        assert "0 = immutable" in text
        assert kind.value.upper() in text


def test_analysis_data_directories_are_created(context, synthetic_video: Path) -> None:
    result = ingest(context, synthetic_video)
    directories = result.template.directories
    for relative in (
        directories.pose,
        directories.depth,
        directories.optical_flow,
        directories.face_landmarks,
        directories.identity_refs,
        directories.intro_cache,
    ):
        assert context.absolute(relative).is_dir()


def test_status_starts_awaiting_masks(context, synthetic_video: Path) -> None:
    result = ingest(context, synthetic_video)
    assert result.template.status is ProcessingStatus.AWAITING_MASKS


def test_validation_fails_without_masks_then_passes_after_import(
    context, synthetic_video: Path, tmp_path: Path
) -> None:
    result = ingest(context, synthetic_video)
    template_id = result.template.id

    first = validate_template(context, template_id)
    assert not first.ok
    assert any("garment masks are missing" in problem for problem in first.problems)

    # Author masks externally, then import them.
    for kind, builder in (
        (MaskKind.GARMENT, fixtures.garment_mask),
        (MaskKind.PROTECTED, fixtures.protected_mask),
    ):
        staging = tmp_path / f"masks_{kind.value}"
        staging.mkdir()
        for index in range(fixtures.ANCHOR, fixtures.TOTAL_FRAMES):
            save_mask(frame_path(staging, index), builder(index))
        imported = import_masks(context, template_id, kind, staging)
        assert imported.imported == list(range(fixtures.ANCHOR, fixtures.TOTAL_FRAMES))

    second = validate_template(context, template_id)
    assert second.ok, second.problems
    assert context.repos.templates.get(template_id).status is ProcessingStatus.READY


def test_mask_import_rejects_wrong_dimensions(
    context, synthetic_video: Path, tmp_path: Path
) -> None:
    from app.core.errors import MaskError

    result = ingest(context, synthetic_video)
    staging = tmp_path / "bad_masks"
    staging.mkdir()
    save_mask(frame_path(staging, fixtures.ANCHOR), np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(MaskError, match="do not match the frame"):
        import_masks(context, result.template.id, MaskKind.GARMENT, staging)


def test_mask_import_skips_unparseable_filenames(
    context, synthetic_video: Path, tmp_path: Path
) -> None:
    result = ingest(context, synthetic_video)
    staging = tmp_path / "mixed_masks"
    staging.mkdir()
    save_mask(frame_path(staging, fixtures.ANCHOR), fixtures.garment_mask(fixtures.ANCHOR))
    save_mask(staging / "not_a_frame.png", fixtures.garment_mask(0))
    imported = import_masks(context, result.template.id, MaskKind.GARMENT, staging)
    assert imported.imported == [fixtures.ANCHOR]
    assert any("does not match" in message for message in imported.skipped)


def test_mask_import_refuses_frames_beyond_the_template(
    context, synthetic_video: Path, tmp_path: Path
) -> None:
    result = ingest(context, synthetic_video)
    staging = tmp_path / "far_masks"
    staging.mkdir()
    save_mask(frame_path(staging, 999), fixtures.garment_mask(0))
    with pytest.raises(ValidationError, match="No importable masks"):
        import_masks(context, result.template.id, MaskKind.GARMENT, staging)


def test_mask_import_does_not_overwrite_without_the_flag(
    context, synthetic_video: Path, tmp_path: Path
) -> None:
    result = ingest(context, synthetic_video)
    staging = tmp_path / "m"
    staging.mkdir()
    for index in range(fixtures.ANCHOR, fixtures.TOTAL_FRAMES):
        save_mask(frame_path(staging, index), fixtures.garment_mask(index))
    import_masks(context, result.template.id, MaskKind.GARMENT, staging)
    second = import_masks(context, result.template.id, MaskKind.GARMENT, staging)
    assert second.imported == []
    assert all("already exists" in message for message in second.skipped)

    third = import_masks(context, result.template.id, MaskKind.GARMENT, staging, overwrite=True)
    assert third.imported


def test_ingesting_the_same_id_twice_is_refused(context, synthetic_video: Path) -> None:
    result = ingest(context, synthetic_video, template_id="tpl_dup")
    assert result.template.id == "tpl_dup"
    with pytest.raises(ValidationError, match="already exists"):
        ingest(context, synthetic_video, template_id="tpl_dup")


def test_ingestion_is_audited(context, synthetic_video: Path) -> None:
    result = ingest(context, synthetic_video)
    events = context.repos.audit.for_entity("template", result.template.id)
    assert any(event["event"] == "template_ingested" for event in events)


def test_identity_reference_images_are_copied_and_hashed(
    context, synthetic_video: Path, tmp_path: Path
) -> None:
    reference = tmp_path / "identity.png"
    fixtures.write_garment_image(reference, width=256, height=256)
    result = ingest(context, synthetic_video, identity_reference_images=[reference])

    assert len(result.template.identity_reference_images) == 1
    relative = result.template.identity_reference_images[0]
    copied = context.absolute(relative)
    assert copied.is_file()
    assert result.template.identity_reference_hashes[relative] == sha256_file(reference)


def test_ffmpeg_commands_are_recorded(context, synthetic_video: Path) -> None:
    result = ingest(context, synthetic_video)
    assert result.commands
    assert any("-i" in command for command in result.commands)


def test_tool_versions_are_recorded(context, synthetic_video: Path) -> None:
    result = ingest(context, synthetic_video)
    assert "ffmpeg" in result.template.tool_versions
    assert result.template.preprocessing_version


def test_video_with_audio_is_detected(context, tmp_path: Path) -> None:
    video = fixtures.write_synthetic_video(tmp_path / "with_audio.mp4", with_audio=True)
    result = ingest(context, video)
    assert result.template.video.has_audio is True
    assert result.template.video.audio_codec
