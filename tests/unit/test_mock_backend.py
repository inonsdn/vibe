"""Mock backend: requirement 10 — identical seed and inputs give identical output."""

from __future__ import annotations

import numpy as np
import pytest

from app.backends.base import FrameRequest, RenderContext, WindowRequest
from app.backends.mock.backend import MockBackend
from app.backends.registry import available_backends, create_backend
from app.core.determinism import derive_seed, frame_seed, window_seed
from app.core.errors import ValidationError
from tests import fixtures


@pytest.fixture
def render_context(context, template, garment) -> RenderContext:
    from app.domain.human_template import FrameRange
    from app.domain.render_job import JobArtifacts, JobProgress, RenderJob

    job = RenderJob(
        id="job_mock",
        template_id=template.template.id,
        template_version=1,
        garment_id=garment.id,
        garment_version=1,
        frame_range=FrameRange(start=fixtures.ANCHOR, end=fixtures.TOTAL_FRAMES),
        transition_anchor_frame=fixtures.ANCHOR,
        backend_name="mock",
        seed=4242,
        progress=JobProgress(total_frames=fixtures.TOTAL_FRAMES - fixtures.ANCHOR),
        artifacts=JobArtifacts.standard("jobs/job_mock"),
    )
    job_dir = context.data_root.job_dir("job_mock")
    (job_dir / "raw_frames").mkdir(parents=True, exist_ok=True)
    return RenderContext(
        job=job,
        template=template.template,
        garment=garment,
        config=context.config,
        source_frames_dir=template.frames_dir,
        job_dir=job_dir,
        raw_frames_dir=job_dir / "raw_frames",
    )


def make_request(index: int, seed: int) -> FrameRequest:
    source = fixtures.synth_frame(index)
    mask = fixtures.garment_mask(index)
    return FrameRequest(frame_index=index, source=source, effective_mask=mask, seed=seed)


def test_mock_needs_no_gpu_weights_or_network(context) -> None:
    backend = MockBackend(context.config)
    capabilities = backend.capabilities()
    assert capabilities.requires_gpu is False
    assert capabilities.requires_model_weights is False
    assert capabilities.deterministic is True
    health = backend.healthcheck()
    assert health.healthy
    assert health.extra["requires_network"] is False


def test_same_seed_and_inputs_produce_identical_pixels(context, render_context) -> None:
    backend = MockBackend(context.config)
    backend.prepare(render_context)
    request = make_request(14, seed=frame_seed(4242, 14, render_context.garment.version_key()))
    first = backend.render_frame(render_context, request)
    second = backend.render_frame(render_context, request)
    assert np.array_equal(first.image, second.image)


def test_a_fresh_backend_instance_reproduces_the_same_pixels(context, render_context) -> None:
    """Determinism must not depend on instance state."""
    seed = frame_seed(4242, 16, render_context.garment.version_key())
    first = MockBackend(context.config)
    first.prepare(render_context)
    a = first.render_frame(render_context, make_request(16, seed)).image

    second = MockBackend(context.config)
    second.prepare(render_context)
    b = second.render_frame(render_context, make_request(16, seed)).image
    assert np.array_equal(a, b)


def test_rendering_a_frame_in_isolation_matches_a_sequential_run(context, render_context) -> None:
    """This is what makes resume safe: no state carries between frames."""
    backend = MockBackend(context.config)
    backend.prepare(render_context)
    garment_key = render_context.garment.version_key()

    sequential = {}
    for index in range(12, 20):
        seed = frame_seed(4242, index, garment_key)
        sequential[index] = backend.render_frame(render_context, make_request(index, seed)).image

    isolated_backend = MockBackend(context.config)
    isolated_backend.prepare(render_context)
    seed = frame_seed(4242, 17, garment_key)
    isolated = isolated_backend.render_frame(render_context, make_request(17, seed)).image
    assert np.array_equal(sequential[17], isolated)


def test_different_seeds_produce_different_pixels(context, render_context) -> None:
    backend = MockBackend(context.config)
    backend.prepare(render_context)
    a = backend.render_frame(render_context, make_request(14, 1)).image
    b = backend.render_frame(render_context, make_request(14, 2)).image
    assert not np.array_equal(a, b)


def test_different_garments_look_different(context, template) -> None:
    """Two outfits must be visually distinguishable, deterministically."""
    from app.domain.human_template import FrameRange
    from app.domain.render_job import JobArtifacts, JobProgress, RenderJob

    def context_for(garment_id: str, colors: tuple[str, ...]) -> RenderContext:
        garment = fixtures.make_garment(context, garment_id=garment_id, colors=colors)
        job = RenderJob(
            id="job_x",
            template_id=template.template.id,
            template_version=1,
            garment_id=garment.id,
            garment_version=1,
            frame_range=FrameRange(start=fixtures.ANCHOR, end=fixtures.TOTAL_FRAMES),
            transition_anchor_frame=fixtures.ANCHOR,
            backend_name="mock",
            seed=7,
            progress=JobProgress(total_frames=fixtures.TOTAL_FRAMES - fixtures.ANCHOR),
            artifacts=JobArtifacts.standard("jobs/job_x"),
        )
        job_dir = context.data_root.job_dir("job_x")
        (job_dir / "raw_frames").mkdir(parents=True, exist_ok=True)
        return RenderContext(
            job=job,
            template=template.template,
            garment=garment,
            config=context.config,
            source_frames_dir=template.frames_dir,
            job_dir=job_dir,
            raw_frames_dir=job_dir / "raw_frames",
        )

    red = context_for("grm_red", ("#ff0000",))
    blue = context_for("grm_blue", ("#0000ff",))
    backend = MockBackend(context.config)
    backend.prepare(red)
    red_frame = backend.render_frame(red, make_request(13, 7)).image
    backend.prepare(blue)
    blue_frame = backend.render_frame(blue, make_request(13, 7)).image
    assert not np.array_equal(red_frame, blue_frame)


def test_render_window_matches_per_frame_rendering(context, render_context) -> None:
    backend = MockBackend(context.config)
    backend.prepare(render_context)
    garment_key = render_context.garment.version_key()
    requests = [
        make_request(index, frame_seed(4242, index, garment_key)) for index in range(12, 16)
    ]
    window = backend.render_window(
        render_context, WindowRequest(frames=requests, window_index=0, seed=1)
    )
    assert [r.frame_index for r in window.results] == [12, 13, 14, 15]
    for result, request in zip(window.results, requests, strict=True):
        single = backend.render_frame(render_context, request)
        assert np.array_equal(result.image, single.image)


def test_output_is_a_full_frame_of_the_right_shape(context, render_context) -> None:
    backend = MockBackend(context.config)
    backend.prepare(render_context)
    result = backend.render_frame(render_context, make_request(12, 5))
    assert result.image.shape == (fixtures.FRAME_HEIGHT, fixtures.FRAME_WIDTH, 3)
    assert result.image.dtype == np.uint8


def test_prepare_reports_a_deterministic_palette(context, render_context) -> None:
    backend = MockBackend(context.config)
    first = backend.prepare(render_context)
    second = MockBackend(context.config).prepare(render_context)
    assert first["palette"] == second["palette"]
    assert first["pattern"] == second["pattern"]
    assert first["requires_network"] is False


# -- registry --------------------------------------------------------------
def test_registry_exposes_both_backends(context) -> None:
    assert set(available_backends()) == {"mock", "comfyui"}
    backend = create_backend("mock", context.config)
    assert backend.name == "mock"
    backend.close()


def test_unknown_backend_is_rejected_with_a_list(context) -> None:
    with pytest.raises(ValidationError) as exc:
        create_backend("stable-magic", context.config)
    assert "mock" in exc.value.details["available"]


# -- seed derivation -------------------------------------------------------
def test_seed_derivation_is_pure() -> None:
    assert derive_seed("a", 1) == derive_seed("a", 1)
    assert derive_seed("a", 1) != derive_seed("a", 2)
    assert 0 <= frame_seed(10, 5) <= 0xFFFFFFFF
    assert window_seed(10, 0, 8) == window_seed(10, 0, 8)
    assert window_seed(10, 0, 8) != window_seed(10, 8, 16)


def test_seed_derivation_rejects_bad_widths() -> None:
    with pytest.raises(ValueError, match="32 or 64"):
        derive_seed("a", bits=17)
