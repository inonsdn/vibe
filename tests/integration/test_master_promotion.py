"""Promoting an accepted synthetic master, and then rendering an outfit on it.

This is the seam the earlier rounds left open: ``accept_master`` set a status
and nothing else, so an "accepted" master was not something the garment
pipeline could render. These tests hold the whole path together —

    animate -> QC -> accept -> promote -> masks -> compatibility -> render ->
    compose -> QC -> final MP4

— and check the properties that make the promoted template trustworthy: the
frozen pixels are the *generated* PNGs (not a re-encode), the anchor is a real
intro/reveal split, promotion is idempotent, and a failure leaves nothing
half-built.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.errors import ConflictError, ValidationError
from app.domain.compatibility import CompatibilityState
from app.domain.enums import JobStatus, MaskKind, ProcessingStatus
from app.media.frames import frame_path, list_frame_indices, read_frame
from app.media.masks import save_mask
from app.pipeline.master_create import (
    MasterCreateOptions,
    accept_master,
    animate_master,
    create_master_candidate,
    write_master_manifest,
)
from app.pipeline.master_promote import PromoteOptions, promote_master
from app.pipeline.motion_compose import ComposeOptions, JoinSpec, SegmentSpec, compose_motion
from app.pipeline.template_ingest import import_masks, verify_source_immutability
from app.qc.motion_checks import run_master_qc
from tests import fixtures
from tests import motion_fixtures as mf
from tests.conftest import requires_ffmpeg


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def composition(context):
    a, b = mf.make_motion_pair(context, a_range=(0, 60), b_range=(0, 60))
    return compose_motion(
        context,
        ComposeOptions(
            display_name="Promotion composition",
            segments=[
                SegmentSpec(motion_source_id=a.source.id, exposed_views=["front"]),
                SegmentSpec(motion_source_id=b.source.id, exposed_views=["front"]),
            ],
            joins=[JoinSpec(bridge_frames=12)],
            composition_id="cmp_promote",
            make_preview=False,
        ),
    ).composition


@pytest.fixture
def hero(context):
    return mf.make_hero(context)


@pytest.fixture
def accepted(context, composition, hero):
    """A fully animated, QC'd, operator-accepted candidate."""
    candidate = create_master_candidate(
        context,
        MasterCreateOptions(
            display_name="Promotable master",
            composition_id=composition.id,
            hero_character_id=hero.id,
            backend_name="mock",
            seed=99,
            chunk_frames=24,
            overlap_frames=16,
        ),
    )
    animate_master(context, candidate.id)
    write_master_manifest(context, context.repos.masters.get(candidate.id))
    run_master_qc(context, candidate.id)
    return accept_master(
        context,
        candidate.id,
        accepted_by="operator",
        reason="Reviewed the contact sheet; motion and identity are correct.",
    )


def write_masks_for(
    context,
    template,
    *,
    kinds=(MaskKind.GARMENT, MaskKind.PROTECTED, MaskKind.OCCLUSION),
    tmp_path,
):
    """Deterministic masks sized to the promoted template, imported properly.

    Bands rather than fixture regions: the promoted template's dimensions come
    from the canonical skeleton profile, not from the garment fixtures, so the
    masks have to be derived from the template itself.
    """
    height, width = template.video.height, template.video.width
    results = {}
    for kind in kinds:
        staging = tmp_path / f"masks_{kind.value}"
        staging.mkdir(parents=True, exist_ok=True)
        for index in range(template.reveal.start, template.reveal.end):
            mask = np.zeros((height, width), dtype=np.uint8)
            if kind is MaskKind.GARMENT:
                mask[height // 3 : (2 * height) // 3, width // 4 : (3 * width) // 4] = 255
            elif kind is MaskKind.OCCLUSION:
                # A hand crossing the garment: swaying, so it is not a constant.
                sway = round(0.04 * width * np.sin(index / 3.0))
                left = max(0, width // 2 - width // 12 + sway)
                mask[height // 2 : height // 2 + height // 12, left : left + width // 6] = 255
            else:  # protected: a face band at the top, never editable
                mask[: height // 6, width // 3 : (2 * width) // 3] = 255
            save_mask(frame_path(staging, index), mask)
        results[kind] = import_masks(context, template.id, kind, staging)
    return results


# ---------------------------------------------------------------------------
# promotion itself
# ---------------------------------------------------------------------------
def test_acceptance_alone_does_not_promote(context, accepted) -> None:
    """The bug this whole module exists for: accepted != usable."""
    assert accepted.is_accepted
    assert accepted.promoted_template_id is None
    assert context.repos.templates.list() == []


def test_promotion_creates_a_real_template(context, accepted) -> None:
    result = promote_master(context, accepted.id, PromoteOptions(transition_anchor=40))

    template = result.template
    assert result.created is True
    assert template.transition_anchor_frame == 40
    assert (template.intro.start, template.intro.end) == (0, 40)
    assert (template.reveal.start, template.reveal.end) == (40, accepted.frame_count)
    assert template.extracted_frame_count == accepted.frame_count
    assert template.status is ProcessingStatus.AWAITING_MASKS
    assert template.consent.subject_kind == "synthetic"

    stored = context.repos.templates.get(template.id)
    assert stored.id == template.id

    reloaded = context.repos.masters.get(accepted.id)
    assert reloaded.promoted_template_id == template.id
    assert reloaded.output_hashes["source_frames_sha256"] == template.source_frames_sha256
    assert reloaded.is_promoted


def test_promoted_frames_are_the_generated_pngs_byte_for_byte(context, accepted) -> None:
    """No encode/decode round trip: every pixel survives promotion exactly.

    This is the property the whole restore-outside-the-mask guarantee rests on.
    An H.264 intermediate would quantise and subsample chroma, and every later
    claim about preserved pixels would be measuring the wrong thing.
    """
    result = promote_master(context, accepted.id, PromoteOptions(transition_anchor=40))
    candidate_frames = context.absolute(accepted.frames_dir)
    template_frames = context.absolute(result.template.directories.source_frames)

    assert list_frame_indices(template_frames) == list_frame_indices(candidate_frames)
    for index in list_frame_indices(candidate_frames):
        origin = frame_path(candidate_frames, index).read_bytes()
        frozen = frame_path(template_frames, index).read_bytes()
        assert origin == frozen, f"frame {index} was not frozen byte-for-byte"

    # ... and the hash the render path checks is the hash of those bytes.
    verify_source_immutability(context, result.template)


def test_promotion_is_idempotent(context, accepted) -> None:
    first = promote_master(context, accepted.id, PromoteOptions(transition_anchor=40))
    second = promote_master(context, accepted.id, PromoteOptions(transition_anchor=40))

    assert second.created is False
    assert second.anchor_source == "already_promoted"
    assert second.template.id == first.template.id
    assert len(context.repos.templates.list()) == 1


def test_promoting_with_a_different_anchor_is_refused(context, accepted) -> None:
    promote_master(context, accepted.id, PromoteOptions(transition_anchor=40))
    with pytest.raises(ConflictError, match="different transition anchor"):
        promote_master(context, accepted.id, PromoteOptions(transition_anchor=41))


def test_unaccepted_candidates_cannot_be_promoted(context, composition, hero) -> None:
    candidate = create_master_candidate(
        context,
        MasterCreateOptions(
            display_name="Not accepted",
            composition_id=composition.id,
            hero_character_id=hero.id,
            backend_name="mock",
            seed=7,
        ),
    )
    animate_master(context, candidate.id)
    write_master_manifest(context, context.repos.masters.get(candidate.id))
    run_master_qc(context, candidate.id)
    with pytest.raises(ConflictError, match="accepted"):
        promote_master(context, candidate.id, PromoteOptions(transition_anchor=40))


@pytest.mark.parametrize("anchor", [0, -1, 10**6])
def test_an_anchor_outside_the_sequence_is_refused(context, accepted, anchor) -> None:
    with pytest.raises(ValidationError):
        promote_master(context, accepted.id, PromoteOptions(transition_anchor=anchor))
    assert context.repos.masters.get(accepted.id).promoted_template_id is None
    assert context.repos.templates.list() == []


def test_a_failed_promotion_leaves_nothing_behind(context, accepted, monkeypatch) -> None:
    """Rollback: no template record, no template directory, no candidate link."""
    import app.pipeline.master_promote as module

    def explode(*_args, **_kwargs):
        raise RuntimeError("archive encoder fell over")

    monkeypatch.setattr(module, "_encode_archive", explode)

    with pytest.raises(RuntimeError):
        promote_master(context, accepted.id, PromoteOptions(transition_anchor=40))

    assert context.repos.templates.list() == []
    assert context.repos.masters.get(accepted.id).promoted_template_id is None
    templates_root = context.data_root.resolve("templates")
    if templates_root.exists():
        assert list(templates_root.iterdir()) == [], "a template directory was left behind"


def test_the_recommended_anchor_comes_from_the_composition(context, accepted, composition) -> None:
    """P1-4: the join already knows where the reveal should begin."""
    recommended = composition.joins[0].recommended_transition_anchor
    assert recommended is not None

    result = promote_master(context, accepted.id, PromoteOptions())
    assert result.anchor_source == "composition_join"
    assert result.transition_anchor == recommended
    assert result.template.transition_anchor_frame == recommended


# ---------------------------------------------------------------------------
# the point of all of it: a promoted master renders an outfit
# ---------------------------------------------------------------------------
@requires_ffmpeg
def test_a_promoted_master_completes_a_mock_garment_render(context, accepted, tmp_path) -> None:
    """Synthetic master -> template -> masks -> compatibility -> MP4."""
    from app.pipeline import compat_service
    from app.pipeline.compose import ComposeOptions as VideoComposeOptions
    from app.pipeline.compose import compose_job
    from app.pipeline.render import JobCreateOptions, create_job, render_job
    from app.qc.report import QCOptions, run_qc

    promotion = promote_master(context, accepted.id, PromoteOptions(transition_anchor=40))
    template = promotion.template

    write_masks_for(context, template, tmp_path=tmp_path)
    template = context.repos.templates.save(
        context.repos.templates.get(template.id).model_copy(
            update={"status": ProcessingStatus.READY}
        )
    )

    garment = fixtures.make_garment(context)
    from app.domain.enums import ImageViewType

    report = compat_service.check_compatibility(
        context,
        template.id,
        garment.id,
        options=compat_service.CheckOptions(
            exposed_views=[ImageViewType.FRONT, ImageViewType.BACK, ImageViewType.SIDE]
        ),
    )
    assert report.state is CompatibilityState.READY, (
        report.state.value,
        report.blocking_reasons,
    )

    job = create_job(
        context,
        template.id,
        garment.id,
        JobCreateOptions(backend_name="mock", seed=2024),
    )
    outcome = render_job(context, job.id)
    assert outcome.job.status is JobStatus.RENDERED
    assert outcome.rendered_frames == list(range(template.reveal.start, template.reveal.end))

    composed = compose_job(context, job.id, VideoComposeOptions())
    final = context.absolute(composed.job.artifacts.final_video_path)
    assert final.is_file() and final.stat().st_size > 0

    qc = run_qc(context, job.id, QCOptions())
    assert qc.checks, "the render must be QC'd"

    # Identity preservation still holds on a synthetic master: outside the
    # editable mask the output is the promoted template's own pixels.
    composited = context.absolute(outcome.job.artifacts.composited_frames_dir)
    masks = context.absolute(outcome.job.artifacts.effective_masks_dir)
    from app.media.masks import load_mask

    source_frames = context.absolute(template.directories.source_frames)
    changed = False
    for index in outcome.rendered_frames:
        source = read_frame(frame_path(source_frames, index))
        output = read_frame(frame_path(composited, index))
        effective = load_mask(frame_path(masks, index), expect_shape=source.shape[:2])
        outside = effective == 0
        assert np.array_equal(output[outside], source[outside]), f"leak at frame {index}"
        inside = effective == 255
        if inside.any() and not np.array_equal(output[inside], source[inside]):
            changed = True
    assert changed, "the garment region should have been repainted"
