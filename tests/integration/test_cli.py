"""CLI: same core services as the API, useful help, clear failures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.cli.main import app
from tests import fixtures
from tests.conftest import requires_ffmpeg

runner = CliRunner()


def invoke(data_root: Path, *args: str):
    return runner.invoke(app, ["--data-root", str(data_root), *args])


def payload(result) -> dict:
    """Parse a --json command's stdout.

    Structured logs go to stderr, so machine-readable output stays parseable --
    which is itself part of the CLI contract and is what this asserts.
    """
    return json.loads(result.stdout)


# -- help and discoverability ---------------------------------------------
def test_top_level_help_lists_every_command_group() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for group in ("template", "garment", "compatibility", "job", "offline", "doctor"):
        assert group in result.output


@pytest.mark.parametrize(
    "command",
    [
        ["template", "--help"],
        ["template", "ingest", "--help"],
        ["template", "inspect", "--help"],
        ["template", "import-masks", "--help"],
        ["garment", "ingest", "--help"],
        ["garment", "inspect", "--help"],
        ["compatibility", "check", "--help"],
        ["compatibility", "override", "--help"],
        ["job", "create", "--help"],
        ["job", "render", "--help"],
        ["job", "resume", "--help"],
        ["job", "inspect", "--help"],
        ["job", "compose", "--help"],
        ["job", "qc", "--help"],
        ["job", "delete", "--help"],
        ["offline", "verify", "--help"],
    ],
)
def test_every_documented_command_has_help(command: list[str]) -> None:
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    assert result.output.strip()


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "garment-replacer" in result.output


def test_doctor_reports_the_environment(data_root: Path) -> None:
    result = invoke(data_root, "doctor", "--no-probe-comfyui")
    assert result.exit_code == 0, result.output
    assert "OFFLINE / ENVIRONMENT VERIFICATION" in result.output
    assert "Cloud integrations" in result.output


def test_doctor_json_output(data_root: Path) -> None:
    result = invoke(data_root, "--json", "doctor", "--no-probe-comfyui")
    assert result.exit_code == 0
    report = payload(result)
    assert report["ok"] is True
    assert report["cloud_integrations"] == []
    assert set(report["backends_available"]) == {"mock", "comfyui"}


def test_offline_verify_command(data_root: Path) -> None:
    result = invoke(data_root, "offline", "verify", "--no-probe-comfyui")
    assert result.exit_code == 0
    assert "Executables" in result.output


def test_job_backends_command(data_root: Path) -> None:
    result = invoke(data_root, "job", "backends")
    assert result.exit_code == 0
    assert "mock" in result.output
    assert "comfyui" in result.output


# -- failure modes ---------------------------------------------------------
def test_unknown_template_fails_clearly(data_root: Path) -> None:
    result = invoke(data_root, "template", "inspect", "does_not_exist")
    assert result.exit_code == 2
    assert "not_found" in result.output or "Template not found" in result.output


def test_malformed_image_option_is_explained(data_root: Path) -> None:
    result = invoke(
        data_root,
        "garment",
        "ingest",
        "--image",
        "front.png",  # missing "view=" prefix
        "--category",
        "top",
        "--coverage",
        "torso",
        "--silhouette",
        "fitted",
        "--material",
        "cotton",
    )
    assert result.exit_code == 2
    assert "view=path" in result.output


def test_unknown_view_lists_the_valid_views(data_root: Path) -> None:
    result = invoke(
        data_root,
        "garment",
        "ingest",
        "--image",
        "sideways=front.png",
        "--category",
        "top",
        "--coverage",
        "torso",
        "--silhouette",
        "fitted",
        "--material",
        "cotton",
    )
    assert result.exit_code == 2
    assert "flat_lay" in result.output


def test_invalid_enum_choice_is_rejected(data_root: Path) -> None:
    result = invoke(
        data_root,
        "garment",
        "ingest",
        "--image",
        "front=x.png",
        "--category",
        "spacesuit",
        "--coverage",
        "torso",
        "--silhouette",
        "fitted",
        "--material",
        "cotton",
    )
    assert result.exit_code != 0


# -- full operator workflow ------------------------------------------------
@requires_ffmpeg
def test_full_operator_workflow_through_the_cli(data_root: Path, tmp_path: Path) -> None:
    """The README quick start, executed end to end."""
    video = fixtures.write_synthetic_video(tmp_path / "master.mp4")

    ingested = invoke(
        data_root,
        "--json",
        "template",
        "ingest",
        str(video),
        "--name",
        "CLI Performance",
        "--transition-anchor",
        str(fixtures.ANCHOR),
        "--clothing-class",
        "fitted_short",
    )
    assert ingested.exit_code == 0, ingested.output
    template_id = payload(ingested)["template_id"]

    # Author masks externally, then import them.
    from app.media.frames import frame_path
    from app.media.masks import save_mask

    for kind, builder in (
        ("garment", fixtures.garment_mask),
        ("protected", fixtures.protected_mask),
        ("occlusion", fixtures.occlusion_mask),
    ):
        staging = tmp_path / f"masks_{kind}"
        staging.mkdir()
        for index in range(fixtures.ANCHOR, fixtures.TOTAL_FRAMES):
            save_mask(frame_path(staging, index), builder(index))
        imported = invoke(
            data_root,
            "--json",
            "template",
            "import-masks",
            template_id,
            "--kind",
            kind,
            "--from",
            str(staging),
        )
        assert imported.exit_code == 0, imported.output

    inspected = invoke(data_root, "--json", "template", "inspect", template_id)
    assert inspected.exit_code == 0
    assert payload(inspected)["validation"]["ok"] is True

    # Garment
    front = fixtures.write_garment_image(tmp_path / "front.png")
    back = fixtures.write_garment_image(tmp_path / "back.png", color=(210, 90, 60))
    side = fixtures.write_garment_image(tmp_path / "side.png", color=(120, 200, 90))
    garment_result = invoke(
        data_root,
        "--json",
        "garment",
        "ingest",
        "--image",
        f"front={front}",
        "--image",
        f"back={back}",
        "--image",
        f"side={side}",
        "--category",
        "top",
        "--coverage",
        "torso",
        "--silhouette",
        "fitted",
        "--material",
        "cotton",
        "--sleeve",
        "short",
        "--length",
        "mid_thigh",
        "--product-name",
        "CLI Top",
        "--brand",
        "TestBrand",
        "--license",
        "test-only",
        "--pattern",
        "striped",
    )
    assert garment_result.exit_code == 0, garment_result.output
    garment_id = payload(garment_result)["garment_id"]

    # Compatibility
    check = invoke(
        data_root,
        "--json",
        "compatibility",
        "check",
        "--template",
        template_id,
        "--garment",
        garment_id,
        "--exposed-view",
        "front",
        "--exposed-view",
        "back",
        "--exposed-view",
        "side",
    )
    assert check.exit_code == 0, check.output
    summary = payload(check)
    assert summary["state"] == "READY"

    # Job
    created = invoke(
        data_root,
        "--json",
        "job",
        "create",
        "--template",
        template_id,
        "--garment",
        garment_id,
        "--backend",
        "mock",
        "--seed",
        "1234",
    )
    assert created.exit_code == 0, created.output
    job_id = payload(created)["job"]["id"]

    rendered = invoke(data_root, "--json", "job", "render", job_id, "--backend", "mock")
    assert rendered.exit_code == 0, rendered.output
    render_payload = payload(rendered)
    assert render_payload["rendered_frames"] == fixtures.TOTAL_FRAMES - fixtures.ANCHOR
    assert render_payload["leaked_frames"] == []

    composed = invoke(data_root, "--json", "job", "compose", job_id, "--no-audio")
    assert composed.exit_code == 0, composed.output
    compose_payload = payload(composed)
    assert compose_payload["total_frames"] == fixtures.TOTAL_FRAMES
    assert Path(compose_payload["final_video"]).is_file()

    qc = invoke(data_root, "--json", "job", "qc", job_id)
    assert qc.exit_code == 0, qc.output
    qc_payload = payload(qc)
    assert qc_payload["passed"] is True, qc_payload["blocking_failures"]

    inspected_job = invoke(data_root, "--json", "job", "inspect", job_id)
    assert inspected_job.exit_code == 0
    assert payload(inspected_job)["remaining_frames"] == []

    listed = invoke(data_root, "--json", "job", "list")
    assert job_id in listed.output


def test_render_and_resume_through_the_cli(data_root: Path) -> None:
    from app.pipeline.context import ServiceContext

    config = fixtures.test_config(data_root)
    with ServiceContext.create(config=config, configure_logs=False) as ctx:
        template = fixtures.make_template(ctx, template_id="tpl_cli")
        garment = fixtures.make_garment(ctx, garment_id="grm_cli")
        from app.pipeline import compat_service

        compat_service.check_compatibility(
            ctx,
            template.template.id,
            garment.id,
            options=compat_service.CheckOptions(
                exposed_views=[
                    fixtures.ImageViewType.FRONT,
                    fixtures.ImageViewType.BACK,
                    fixtures.ImageViewType.SIDE,
                ]
            ),
        )

    created = invoke(
        data_root,
        "--json",
        "job",
        "create",
        "--template",
        "tpl_cli",
        "--garment",
        "grm_cli",
        "--backend",
        "mock",
    )
    assert created.exit_code == 0, created.output
    job_id = payload(created)["job"]["id"]

    partial = invoke(data_root, "--json", "job", "render", job_id, "--max-frames", "4")
    assert partial.exit_code == 0
    assert payload(partial)["status"] == "paused"

    resumed = invoke(data_root, "--json", "job", "resume", job_id)
    assert resumed.exit_code == 0
    resume_payload = payload(resumed)
    assert resume_payload["skipped_frames"] == 4
    assert resume_payload["status"] == "rendered"


def test_compatibility_check_exit_code_signals_blocking(data_root: Path) -> None:
    from app.pipeline.context import ServiceContext

    config = fixtures.test_config(data_root)
    with ServiceContext.create(config=config, configure_logs=False) as ctx:
        fixtures.make_template(ctx, template_id="tpl_x")
        fixtures.make_garment(ctx, garment_id="grm_x", views=(fixtures.ImageViewType.FRONT,))

    result = invoke(
        data_root,
        "compatibility",
        "check",
        "--template",
        "tpl_x",
        "--garment",
        "grm_x",
        "--exposed-view",
        "front",
        "--exposed-view",
        "back",
    )
    assert result.exit_code == 3
    assert "NEEDS_INPUT" in result.output
    assert "app compatibility override" in result.output


def test_job_delete_requires_confirmation(data_root: Path) -> None:
    from app.pipeline.context import ServiceContext
    from app.pipeline.render import JobCreateOptions, create_job

    config = fixtures.test_config(data_root)
    with ServiceContext.create(config=config, configure_logs=False) as ctx:
        template = fixtures.make_template(ctx, template_id="tpl_del")
        garment = fixtures.make_garment(ctx, garment_id="grm_del")
        from app.pipeline import compat_service

        compat_service.check_compatibility(
            ctx,
            template.template.id,
            garment.id,
            options=compat_service.CheckOptions(
                exposed_views=[
                    fixtures.ImageViewType.FRONT,
                    fixtures.ImageViewType.BACK,
                    fixtures.ImageViewType.SIDE,
                ]
            ),
        )
        job = create_job(
            ctx, template.template.id, garment.id, JobCreateOptions(backend_name="mock")
        )
        job_dir = ctx.absolute(job.artifacts.root)

    dry_run = invoke(data_root, "job", "delete", job.id)
    assert dry_run.exit_code == 1
    assert "would delete" in dry_run.output
    assert job_dir.is_dir(), "a dry run must not delete anything"

    confirmed = invoke(data_root, "job", "delete", job.id, "--yes")
    assert confirmed.exit_code == 0
    assert not job_dir.exists()


def test_job_delete_refuses_an_unknown_job(data_root: Path) -> None:
    result = invoke(data_root, "job", "delete", "job_not_real", "--yes")
    assert result.exit_code == 2
