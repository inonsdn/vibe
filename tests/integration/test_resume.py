"""Resume: requirement 9 — a resumed job skips completed frames."""

from __future__ import annotations

import numpy as np
import pytest

from app.backends.base import FrameResult
from app.backends.mock.backend import MockBackend
from app.core.errors import ConflictError
from app.domain.enums import JobStatus
from app.media.frames import hash_sequence, list_frame_indices
from app.pipeline.render import JobCreateOptions, create_job, render_job, resume_job
from tests import fixtures

REVEAL = list(range(fixtures.ANCHOR, fixtures.TOTAL_FRAMES))


def test_partial_render_pauses_and_records_a_checkpoint(context, ready_pair) -> None:
    template, garment, _ = ready_pair
    job = create_job(
        context, template.template.id, garment.id, JobCreateOptions(backend_name="mock")
    )
    outcome = render_job(context, job.id, max_frames=5)

    assert outcome.job.status is JobStatus.PAUSED
    assert len(outcome.rendered_frames) == 5
    assert outcome.job.checkpoint.completed_frames == REVEAL[:5]
    assert outcome.job.checkpoint.next_frame == REVEAL[5]
    assert outcome.job.progress.completed_frames == 5
    assert outcome.job.remaining_frames() == REVEAL[5:]


def test_resume_renders_only_the_remaining_frames(context, ready_pair) -> None:
    """Requirement 9."""
    template, garment, _ = ready_pair
    job = create_job(
        context, template.template.id, garment.id, JobCreateOptions(backend_name="mock", seed=42)
    )
    first = render_job(context, job.id, max_frames=5)
    done_first = set(first.rendered_frames)

    second = resume_job(context, job.id)
    assert set(second.skipped_frames) == done_first
    assert set(second.rendered_frames) == set(REVEAL) - done_first
    assert not set(second.rendered_frames) & done_first, "a completed frame was re-rendered"
    assert second.job.status is JobStatus.RENDERED
    assert second.job.is_complete()


def test_resumed_output_is_identical_to_a_single_pass_render(context, ready_pair) -> None:
    """Interrupting a render must not change a single output byte."""
    template, garment, _ = ready_pair

    whole = create_job(
        context,
        template.template.id,
        garment.id,
        JobCreateOptions(backend_name="mock", seed=2024, job_id="job_whole"),
    )
    render_job(context, whole.id)

    sliced = create_job(
        context,
        template.template.id,
        garment.id,
        JobCreateOptions(backend_name="mock", seed=2024, job_id="job_sliced"),
    )
    render_job(context, sliced.id, max_frames=3)
    resume_job(context, sliced.id, max_frames=4)
    resume_job(context, sliced.id)

    dir_whole = context.absolute(
        context.repos.jobs.get("job_whole").artifacts.composited_frames_dir
    )
    dir_sliced = context.absolute(
        context.repos.jobs.get("job_sliced").artifacts.composited_frames_dir
    )
    assert list_frame_indices(dir_whole) == list_frame_indices(dir_sliced) == REVEAL
    assert hash_sequence(dir_whole, REVEAL) == hash_sequence(dir_sliced, REVEAL)


def test_completed_frames_are_not_handed_to_the_backend_again(context, ready_pair) -> None:
    """Proven by instrumenting the backend, not inferred from timings."""
    template, garment, _ = ready_pair
    job = create_job(
        context, template.template.id, garment.id, JobCreateOptions(backend_name="mock")
    )
    render_job(context, job.id, max_frames=4)

    seen: list[int] = []
    original = MockBackend.render_frame

    def spy(self, render_context, request):  # type: ignore[no-untyped-def]
        seen.append(request.frame_index)
        return original(self, render_context, request)

    MockBackend.render_frame = spy  # type: ignore[method-assign]
    try:
        resume_job(context, job.id)
    finally:
        MockBackend.render_frame = original  # type: ignore[method-assign]

    assert seen == REVEAL[4:]


def test_resume_after_a_crash_mid_window(context, ready_pair, monkeypatch) -> None:
    """A hard failure mid-window leaves earlier frames usable."""
    template, garment, _ = ready_pair
    job = create_job(
        context,
        template.template.id,
        garment.id,
        JobCreateOptions(backend_name="mock", seed=8),
    )

    original = MockBackend.render_frame
    boom_at = fixtures.ANCHOR + 6

    def exploding(self, render_context, request):  # type: ignore[no-untyped-def]
        if request.frame_index == boom_at:
            raise RuntimeError("simulated backend crash")
        return original(self, render_context, request)

    monkeypatch.setattr(MockBackend, "render_frame", exploding)
    with pytest.raises(RuntimeError, match="simulated backend crash"):
        render_job(context, job.id)

    crashed = context.repos.jobs.get(job.id)
    assert crashed.status is JobStatus.FAILED
    assert crashed.error is not None
    assert crashed.status.is_resumable

    # Only frames whose composited PNG reached disk count as complete. Frames
    # rendered inside the crashed window but not yet composited are redone,
    # which is the conservative and correct choice.
    committed = set(context.repos.jobs.completed_frames(job.id))
    assert committed
    assert committed <= set(range(fixtures.ANCHOR, boom_at))

    monkeypatch.setattr(MockBackend, "render_frame", original)
    outcome = resume_job(context, job.id)
    assert outcome.job.status is JobStatus.RENDERED
    assert outcome.job.is_complete()
    assert boom_at in outcome.rendered_frames
    assert set(outcome.skipped_frames) == committed
    assert not set(outcome.rendered_frames) & committed


def test_resume_is_a_noop_when_everything_is_done(context, ready_pair) -> None:
    template, garment, _ = ready_pair
    job = create_job(
        context, template.template.id, garment.id, JobCreateOptions(backend_name="mock")
    )
    render_job(context, job.id)
    outcome = resume_job(context, job.id)
    assert outcome.rendered_frames == []
    assert outcome.backend_info["nothing_to_do"] is True
    assert outcome.job.status is JobStatus.RENDERED


def test_rerendering_a_finished_job_is_refused(context, ready_pair) -> None:
    from app.pipeline.compose import ComposeOptions, compose_job
    from tests.conftest import ffmpeg_available

    template, garment, _ = ready_pair
    job = create_job(
        context, template.template.id, garment.id, JobCreateOptions(backend_name="mock")
    )
    render_job(context, job.id)
    if not ffmpeg_available():
        pytest.skip("ffmpeg required to reach a terminal state")
    compose_job(context, job.id, ComposeOptions(include_audio=False))
    from app.qc.report import QCOptions, run_qc

    run_qc(context, job.id, QCOptions(make_contact_sheets=False))
    finished = context.repos.jobs.get(job.id)
    assert finished.status is JobStatus.COMPLETED
    with pytest.raises(ConflictError, match="already finished"):
        render_job(context, job.id)


def test_frame_checkpoints_survive_a_new_service_context(context, ready_pair, config) -> None:
    """The checkpoint is in SQLite, not in memory."""
    from app.pipeline.context import ServiceContext

    template, garment, _ = ready_pair
    job = create_job(
        context, template.template.id, garment.id, JobCreateOptions(backend_name="mock")
    )
    render_job(context, job.id, max_frames=3)
    context.close()

    with ServiceContext.create(config=config, configure_logs=False) as reopened:
        reloaded = reopened.repos.jobs.get(job.id)
        assert reopened.repos.jobs.completed_frames(job.id) == REVEAL[:3]
        assert reloaded.checkpoint.next_frame == REVEAL[3]
        outcome = resume_job(reopened, job.id)
        assert set(outcome.skipped_frames) == set(REVEAL[:3])
        assert outcome.job.is_complete()


def test_frame_hashes_are_recorded_per_frame(context, ready_pair) -> None:
    template, garment, _ = ready_pair
    job = create_job(
        context, template.template.id, garment.id, JobCreateOptions(backend_name="mock")
    )
    render_job(context, job.id)
    recorded = context.repos.jobs.frame_checksums(job.id)
    composited = context.absolute(context.repos.jobs.get(job.id).artifacts.composited_frames_dir)
    on_disk = hash_sequence(composited, REVEAL)
    assert recorded == on_disk


def test_a_backend_returning_the_wrong_frames_is_rejected(context, ready_pair, monkeypatch) -> None:
    from app.core.errors import ValidationError

    template, garment, _ = ready_pair
    job = create_job(
        context, template.template.id, garment.id, JobCreateOptions(backend_name="mock")
    )

    def wrong_window(self, render_context, request):  # type: ignore[no-untyped-def]
        from app.backends.base import WindowResult

        return WindowResult(
            results=[
                FrameResult(
                    frame_index=9999,
                    image=np.zeros((fixtures.FRAME_HEIGHT, fixtures.FRAME_WIDTH, 3), np.uint8),
                    seed=0,
                )
            ],
            window_index=request.window_index,
        )

    monkeypatch.setattr(MockBackend, "render_window", wrong_window)
    with pytest.raises(ValidationError, match="different set of frames"):
        render_job(context, job.id)
