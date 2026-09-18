"""End-to-end mock pipeline.

Covers requirements:

1.  source video hash is preserved
2.  intro frames are identical across garment jobs
3.  mock render changes only allowed garment pixels
4.  protected pixels remain equal to the source
8.  the transition anchor has no off-by-one error
10. identical seed and inputs give identical mock output
11. manifest hashes and settings are complete
14. final video metadata matches the configured dimensions/FPS
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.core.hashing import sha256_dir, sha256_file
from app.domain.enums import JobStatus, MaskKind
from app.media.frames import frame_path, frames_equal, list_frame_indices, read_frame
from app.media.masks import load_mask
from app.pipeline.compose import ComposeOptions, compose_job
from app.pipeline.render import JobCreateOptions, create_job, render_job
from app.pipeline.template_ingest import verify_source_immutability
from app.qc.report import QCOptions, run_qc
from tests import fixtures
from tests.conftest import requires_ffmpeg


def run_to_render(context, template, garment, *, seed: int = 1234, job_id: str | None = None):
    job = create_job(
        context,
        template.template.id,
        garment.id,
        JobCreateOptions(backend_name="mock", seed=seed, job_id=job_id),
    )
    return render_job(context, job.id)


# ---------------------------------------------------------------------------
# rendering invariants
# ---------------------------------------------------------------------------
def test_render_produces_every_reveal_frame(context, ready_pair) -> None:
    template, garment, _ = ready_pair
    outcome = run_to_render(context, template, garment)

    assert outcome.job.status is JobStatus.RENDERED
    expected = list(range(fixtures.ANCHOR, fixtures.TOTAL_FRAMES))
    assert outcome.rendered_frames == expected

    composited = context.absolute(outcome.job.artifacts.composited_frames_dir)
    assert list_frame_indices(composited) == expected


def test_only_allowed_garment_pixels_change(context, ready_pair) -> None:
    """Requirement 3."""
    template, garment, _ = ready_pair
    outcome = run_to_render(context, template, garment)
    composited = context.absolute(outcome.job.artifacts.composited_frames_dir)
    masks = context.absolute(outcome.job.artifacts.effective_masks_dir)

    changed_any = False
    for index in outcome.rendered_frames:
        source = read_frame(frame_path(template.frames_dir, index))
        output = read_frame(frame_path(composited, index))
        effective = load_mask(frame_path(masks, index), expect_shape=source.shape[:2])

        outside = effective == 0
        assert np.array_equal(output[outside], source[outside]), f"leak at frame {index}"

        inside = effective == 255
        if inside.any() and not np.array_equal(output[inside], source[inside]):
            changed_any = True
    assert changed_any, "the mock backend should have changed the garment region"


def test_protected_pixels_equal_the_source(context, ready_pair) -> None:
    """Requirement 4."""
    template, garment, _ = ready_pair
    outcome = run_to_render(context, template, garment)
    composited = context.absolute(outcome.job.artifacts.composited_frames_dir)
    protected_dir = template.mask_dirs[MaskKind.PROTECTED]

    for index in outcome.rendered_frames:
        source = read_frame(frame_path(template.frames_dir, index))
        output = read_frame(frame_path(composited, index))
        protected = load_mask(frame_path(protected_dir, index), expect_shape=source.shape[:2])
        selection = protected > 0
        assert selection.any()
        assert np.array_equal(output[selection], source[selection]), f"frame {index}"


def test_occluded_hand_pixels_are_preserved(context, ready_pair) -> None:
    template, garment, _ = ready_pair
    outcome = run_to_render(context, template, garment)
    composited = context.absolute(outcome.job.artifacts.composited_frames_dir)
    occlusion_dir = template.mask_dirs[MaskKind.OCCLUSION]

    for index in outcome.rendered_frames[:4]:
        source = read_frame(frame_path(template.frames_dir, index))
        output = read_frame(frame_path(composited, index))
        occlusion = load_mask(frame_path(occlusion_dir, index), expect_shape=source.shape[:2])
        selection = occlusion > 0
        assert np.array_equal(output[selection], source[selection])


def test_source_frames_are_never_modified(context, ready_pair) -> None:
    """Requirement 1 (source integrity)."""
    template, garment, _ = ready_pair
    before = sha256_dir(template.frames_dir)
    source_hash_before = sha256_file(Path(template.template.source_video_path))

    run_to_render(context, template, garment)

    assert sha256_dir(template.frames_dir) == before
    assert sha256_file(Path(template.template.source_video_path)) == source_hash_before
    assert before == template.template.source_frames_sha256
    verify_source_immutability(context, template.template)  # must not raise


def test_tampered_source_frames_block_a_render(context, ready_pair) -> None:
    from app.core.errors import ImmutabilityError

    template, garment, _ = ready_pair
    job = create_job(
        context,
        template.template.id,
        garment.id,
        JobCreateOptions(backend_name="mock"),
    )
    # Simulate an operator editing an "immutable" frame.
    tampered = read_frame(frame_path(template.frames_dir, 3))
    tampered[0, 0] = (1, 2, 3)
    from app.media.frames import write_frame

    write_frame(frame_path(template.frames_dir, 3), tampered)

    with pytest.raises(ImmutabilityError, match="modified since ingestion"):
        render_job(context, job.id)


def test_identical_seed_and_inputs_give_identical_output(context, ready_pair) -> None:
    """Requirement 10."""
    template, garment, _ = ready_pair
    first = run_to_render(context, template, garment, seed=777, job_id="job_det_a")
    second = run_to_render(context, template, garment, seed=777, job_id="job_det_b")

    dir_a = context.absolute(first.job.artifacts.composited_frames_dir)
    dir_b = context.absolute(second.job.artifacts.composited_frames_dir)
    for index in first.rendered_frames:
        assert frames_equal(frame_path(dir_a, index), frame_path(dir_b, index)), index


def test_different_seeds_give_different_output(context, ready_pair) -> None:
    template, garment, _ = ready_pair
    first = run_to_render(context, template, garment, seed=1, job_id="job_seed_1")
    second = run_to_render(context, template, garment, seed=2, job_id="job_seed_2")
    dir_a = context.absolute(first.job.artifacts.composited_frames_dir)
    dir_b = context.absolute(second.job.artifacts.composited_frames_dir)
    differences = [
        index
        for index in first.rendered_frames
        if not frames_equal(frame_path(dir_a, index), frame_path(dir_b, index))
    ]
    assert differences, "a different seed must change the rendered garment"


def test_per_frame_seeds_are_recorded_and_re_derivable(context, ready_pair) -> None:
    from app.pipeline.render import job_seeds

    template, garment, _ = ready_pair
    outcome = run_to_render(context, template, garment, seed=99)
    recorded = outcome.job.qc_metrics["render"]["per_frame_seeds"]
    rederived = job_seeds(outcome.job)
    assert recorded == rederived
    assert len(recorded) == outcome.job.frame_range.count


def test_render_refuses_to_start_before_the_anchor(context, ready_pair) -> None:
    from app.core.errors import ValidationError

    template, garment, _ = ready_pair
    with pytest.raises(ValidationError, match="transition anchor"):
        create_job(
            context,
            template.template.id,
            garment.id,
            JobCreateOptions(backend_name="mock", frame_start=0),
        )


def test_render_refuses_to_run_past_the_reveal_range(context, ready_pair) -> None:
    from app.core.errors import ValidationError

    template, garment, _ = ready_pair
    with pytest.raises(ValidationError, match="exceeds"):
        create_job(
            context,
            template.template.id,
            garment.id,
            JobCreateOptions(backend_name="mock", frame_end=fixtures.TOTAL_FRAMES + 5),
        )


# ---------------------------------------------------------------------------
# intro reuse (requirement 2)
# ---------------------------------------------------------------------------
@requires_ffmpeg
def test_intro_frames_are_identical_across_garment_jobs(context, template) -> None:
    """Requirement 2: two outfits share a bit-identical intro."""
    from app.domain.enums import ImageViewType
    from app.pipeline import compat_service

    results = []
    for garment_id, colors in (("grm_a", ("#ff0000",)), ("grm_b", ("#00ff00",))):
        garment = fixtures.make_garment(context, garment_id=garment_id, colors=colors)
        compat_service.check_compatibility(
            context,
            template.template.id,
            garment.id,
            options=compat_service.CheckOptions(
                exposed_views=[ImageViewType.FRONT, ImageViewType.BACK, ImageViewType.SIDE]
            ),
        )
        outcome = run_to_render(context, template, garment, job_id=f"job_{garment_id}")
        result = compose_job(context, outcome.job.id, ComposeOptions(include_audio=False))
        results.append(result)

    assembly_a = context.absolute(results[0].job.artifacts.root) / "assembly_frames"
    assembly_b = context.absolute(results[1].job.artifacts.root) / "assembly_frames"

    intro_indices = list(range(0, fixtures.ANCHOR))
    for index in intro_indices:
        assert frames_equal(
            frame_path(assembly_a, index), frame_path(assembly_b, index)
        ), f"intro frame {index} differs between garment jobs"

    # And the reveal genuinely differs, so the test is not vacuous.
    differing = [
        index
        for index in range(fixtures.ANCHOR, fixtures.TOTAL_FRAMES)
        if not frames_equal(frame_path(assembly_a, index), frame_path(assembly_b, index))
    ]
    assert differing, "different garments must produce different reveal frames"

    # Both intros are byte-identical to the immutable source frames.
    for index in intro_indices:
        assert frames_equal(frame_path(assembly_a, index), frame_path(template.frames_dir, index))


def test_intro_frames_are_never_rendered(context, ready_pair) -> None:
    """No intro frame may appear in the rendered/composited output."""
    template, garment, _ = ready_pair
    outcome = run_to_render(context, template, garment)
    composited = context.absolute(outcome.job.artifacts.composited_frames_dir)
    rendered_indices = set(list_frame_indices(composited))
    assert rendered_indices.isdisjoint(set(range(0, fixtures.ANCHOR)))


# ---------------------------------------------------------------------------
# compose + transition (requirement 8) and manifest (requirement 11)
# ---------------------------------------------------------------------------
@requires_ffmpeg
def test_transition_anchor_has_no_off_by_one(context, ready_pair) -> None:
    """Requirement 8."""
    template, garment, _ = ready_pair
    outcome = run_to_render(context, template, garment)
    result = compose_job(context, outcome.job.id, ComposeOptions(include_audio=False))

    anchor = template.template.transition_anchor_frame
    assembly = context.absolute(result.job.artifacts.root) / "assembly_frames"

    indices = list_frame_indices(assembly)
    assert indices == list(range(0, fixtures.TOTAL_FRAMES))
    assert result.assembly["intro_frame_count"] == anchor
    assert result.assembly["reveal_frame_count"] == fixtures.TOTAL_FRAMES - anchor
    assert result.transition["last_intro_frame"] == anchor - 1
    assert result.transition["first_reveal_frame"] == anchor
    assert result.transition["last_intro_matches_cache"] is True
    assert result.transition["first_reveal_matches_render"] is True

    # The frame at anchor-1 is source; the frame at anchor is rendered.
    intro_cache = context.absolute(template.template.directories.intro_cache)
    composited = context.absolute(result.job.artifacts.composited_frames_dir)
    assert frames_equal(frame_path(assembly, anchor - 1), frame_path(intro_cache, anchor - 1))
    assert frames_equal(frame_path(assembly, anchor), frame_path(composited, anchor))
    assert not frames_equal(frame_path(assembly, anchor), frame_path(template.frames_dir, anchor))


@requires_ffmpeg
def test_manifest_is_complete_and_reproducible(context, ready_pair) -> None:
    """Requirement 11."""
    template, garment, report = ready_pair
    outcome = run_to_render(context, template, garment, seed=515)
    result = compose_job(context, outcome.job.id, ComposeOptions(include_audio=False))
    manifest = result.manifest

    assert manifest.required_fields_present() == []

    # Inputs
    assert manifest.template_source_sha256 == template.template.source_sha256
    assert manifest.input_hashes["template_source_video"] == template.template.source_sha256
    assert manifest.input_hashes["template_source_frames_dir"] == (
        template.template.source_frames_sha256
    )
    assert "intro_cache_dir" in manifest.input_hashes
    assert any(key.startswith("garment_image_") for key in manifest.input_hashes)
    assert manifest.input_hashes["masks_garment"] != "absent"
    assert manifest.input_hashes["masks_protected"] != "absent"

    # Settings and identity
    assert manifest.settings.feather_radius_px == outcome.job.settings.feather_radius_px
    assert manifest.reproducibility.seed == 515
    assert manifest.reproducibility.backend_name == "mock"
    assert manifest.reproducibility.backend_version
    assert manifest.reproducibility.config_hash == context.config.config_hash()
    assert manifest.reproducibility.dependencies
    assert manifest.reproducibility.ffmpeg_version
    assert manifest.reproducibility.ffmpeg_commands
    assert manifest.reproducibility.rules_file_sha256 == report.rules_file_sha256
    assert len(manifest.reproducibility.per_frame_seeds) == outcome.job.frame_range.count

    # Outputs
    assert manifest.output_hashes["final_video"] == sha256_file(result.final_video)
    assert len(manifest.frame_checksums) == fixtures.TOTAL_FRAMES
    sources = {c.source for c in manifest.frame_checksums}
    assert sources == {"intro_cache", "rendered"}

    # Compatibility provenance
    assert manifest.compatibility_report_id == report.id
    assert manifest.compatibility_state is not None
    assert manifest.compatibility_overridden is False

    # The sidecar on disk matches the stored manifest.
    sidecar = json.loads(
        context.absolute(result.job.artifacts.manifest_path).read_text(encoding="utf-8")
    )
    assert sidecar["job_id"] == outcome.job.id
    assert sidecar["reproducibility"]["seed"] == 515

    # Digest is stable across recomputation and matches the stored digest.
    assert manifest.reproducibility_digest() == manifest.reproducibility_digest()
    assert context.repos.jobs.manifest_digest(outcome.job.id) == manifest.reproducibility_digest()


@requires_ffmpeg
def test_two_identical_jobs_share_a_reproducibility_digest(context, ready_pair) -> None:
    """Requirement 10/11: same inputs -> same manifest digest."""
    template, garment, _ = ready_pair
    digests = []
    for index, job_id in enumerate(("job_rep_a", "job_rep_b")):
        outcome = run_to_render(context, template, garment, seed=31337, job_id=job_id)
        result = compose_job(
            context,
            outcome.job.id,
            ComposeOptions(include_audio=False, output_name=f"rep_{index}.mp4"),
        )
        digests.append(result.manifest.reproducibility_digest())
    assert digests[0] == digests[1]


@requires_ffmpeg
def test_final_video_metadata_matches_the_configuration(context, ready_pair) -> None:
    """Requirement 14."""
    from app.media import ffmpeg

    template, garment, _ = ready_pair
    outcome = run_to_render(context, template, garment)
    result = compose_job(context, outcome.job.id, ComposeOptions(include_audio=False))

    probe = ffmpeg.probe(result.final_video, count_frames=True)
    stream = probe.video
    assert stream is not None
    assert (stream.width, stream.height) == (
        context.config.video.width,
        context.config.video.height,
    )
    assert stream.pix_fmt == context.config.video.pixel_format
    assert stream.codec_name == "h264"
    fps = ffmpeg.parse_frame_rate(stream.avg_frame_rate)
    assert fps == pytest.approx(template.template.video.fps, abs=0.01)
    assert stream.nb_frames == fixtures.TOTAL_FRAMES


@requires_ffmpeg
def test_preview_encode_is_optional_and_smaller(context, ready_pair) -> None:
    template, garment, _ = ready_pair
    outcome = run_to_render(context, template, garment)
    result = compose_job(
        context, outcome.job.id, ComposeOptions(include_audio=False, make_preview=True)
    )
    assert result.preview_video is not None
    assert result.preview_video.is_file()
    assert "preview_video" in result.manifest.output_hashes


@requires_ffmpeg
def test_flash_transition_is_recorded_separately(context, ready_pair) -> None:
    template, garment, _ = ready_pair
    outcome = run_to_render(context, template, garment)
    result = compose_job(
        context, outcome.job.id, ComposeOptions(include_audio=False, flash_frames=3)
    )
    flash = [c.index for c in result.manifest.frame_checksums if c.source == "flash"]
    assert flash == [fixtures.ANCHOR, fixtures.ANCHOR + 1, fixtures.ANCHOR + 2]
    assert result.manifest.flash_frames == 3


@requires_ffmpeg
def test_compose_refuses_an_incomplete_render(context, ready_pair) -> None:
    from app.core.errors import ValidationError

    template, garment, _ = ready_pair
    job = create_job(
        context, template.template.id, garment.id, JobCreateOptions(backend_name="mock")
    )
    render_job(context, job.id, max_frames=2)
    with pytest.raises(ValidationError, match="unrendered frames"):
        compose_job(context, job.id, ComposeOptions(include_audio=False))


@requires_ffmpeg
def test_compose_refuses_to_overwrite_without_the_flag(context, ready_pair) -> None:
    from app.core.errors import ValidationError

    template, garment, _ = ready_pair
    outcome = run_to_render(context, template, garment)
    compose_job(
        context, outcome.job.id, ComposeOptions(include_audio=False, output_name="fixed.mp4")
    )
    with pytest.raises(ValidationError, match="already exists"):
        compose_job(
            context, outcome.job.id, ComposeOptions(include_audio=False, output_name="fixed.mp4")
        )


# ---------------------------------------------------------------------------
# QC
# ---------------------------------------------------------------------------
@requires_ffmpeg
def test_qc_passes_for_a_clean_mock_render(context, ready_pair) -> None:
    template, garment, _ = ready_pair
    outcome = run_to_render(context, template, garment)
    compose_job(context, outcome.job.id, ComposeOptions(include_audio=False))
    report = run_qc(context, outcome.job.id, QCOptions())

    assert report.passed, [c.as_dict() for c in report.failed_checks]
    ids = {check.check_id for check in report.checks}
    for expected in (
        "protected_region_preserved",
        "background_preserved",
        "face_region_preserved",
        "garment_temporal_stability",
        "mask_boundary_leakage",
        "black_frames",
        "frozen_frames",
        "duplicate_frames",
        "frame_sequence_complete",
        "intro_reuse_integrity",
        "transition_correctness",
        "dimensions_fps_frame_count",
        "audio_video_duration",
        "encoding_valid",
        "manifest_deterministic",
    ):
        assert expected in ids, f"missing QC check: {expected}"

    by_id = {check.check_id: check for check in report.checks}
    assert by_id["protected_region_preserved"].metrics["max_diff"] == 0
    assert by_id["background_preserved"].metrics["max_diff"] == 0
    assert by_id["intro_reuse_integrity"].metrics["mismatched"] == 0
    assert by_id["transition_correctness"].metrics["anchor"] == fixtures.ANCHOR


@requires_ffmpeg
def test_qc_writes_both_reports_and_contact_sheets(context, ready_pair) -> None:
    template, garment, _ = ready_pair
    outcome = run_to_render(context, template, garment)
    compose_job(context, outcome.job.id, ComposeOptions(include_audio=False))
    run_qc(context, outcome.job.id, QCOptions())

    job = context.repos.jobs.get(outcome.job.id)
    json_report = context.absolute(job.artifacts.qc_report_path)
    text_report = context.absolute(job.artifacts.qc_report_text_path)
    assert json.loads(json_report.read_text(encoding="utf-8"))["passed"] is True
    assert "QC REPORT" in text_report.read_text(encoding="utf-8")
    assert job.artifacts.contact_sheet_paths
    for relative in job.artifacts.contact_sheet_paths:
        assert context.absolute(relative).is_file()
    assert job.status is JobStatus.COMPLETED


@requires_ffmpeg
def test_qc_detects_a_tampered_protected_region(context, ready_pair) -> None:
    """QC must catch identity alteration, not just trust the compositor."""
    from app.media.frames import write_frame

    template, garment, _ = ready_pair
    outcome = run_to_render(context, template, garment)
    compose_job(context, outcome.job.id, ComposeOptions(include_audio=False))

    composited = context.absolute(outcome.job.artifacts.composited_frames_dir)
    index = outcome.rendered_frames[0]
    frame = read_frame(frame_path(composited, index))
    y0, y1, x0, x1 = fixtures.FACE_REGION
    frame[y0:y1, x0:x1] = 0  # blank the face
    write_frame(frame_path(composited, index), frame)

    report = run_qc(context, outcome.job.id, QCOptions(write_reports=False))
    assert not report.passed
    failed_ids = {check.check_id for check in report.failed_checks}
    assert "protected_region_preserved" in failed_ids


@requires_ffmpeg
def test_qc_detects_frozen_frames(context, ready_pair) -> None:
    from app.media.frames import write_frame

    template, garment, _ = ready_pair
    outcome = run_to_render(context, template, garment)
    compose_job(context, outcome.job.id, ComposeOptions(include_audio=False))

    composited = context.absolute(outcome.job.artifacts.composited_frames_dir)
    frozen = read_frame(frame_path(composited, fixtures.ANCHOR))
    for index in outcome.rendered_frames:
        write_frame(frame_path(composited, index), frozen)

    report = run_qc(context, outcome.job.id, QCOptions(write_reports=False))
    failed_ids = {check.check_id for check in report.failed_checks}
    assert "frozen_frames" in failed_ids or "duplicate_frames" in failed_ids
