"""Motion and master surfaces: the CLI and the API share the same core services."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from app.api import deps
from app.api.main import create_app
from app.cli.main import app as cli_app
from tests import motion_fixtures as mf

runner = CliRunner()


def invoke(data_root: Path, *args: str):
    return runner.invoke(cli_app, ["--data-root", str(data_root), *args])


def payload(result) -> dict:
    return json.loads(result.stdout)


@pytest.fixture
def client(config) -> Iterator[TestClient]:
    application = create_app(config, configure_logs=False)
    with TestClient(application) as test_client:
        yield test_client
    deps.close()


@pytest.fixture
def api_context(client: TestClient):
    return deps.get_context()


# ---------------------------------------------------------------------------
# CLI help and discoverability
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "command",
    [
        ["motion", "--help"],
        ["motion", "ingest", "--help"],
        ["motion", "import-pose", "--help"],
        ["motion", "inspect", "--help"],
        ["motion", "normalize", "--help"],
        ["motion", "match-anchors", "--help"],
        ["motion", "compose", "--help"],
        ["motion", "preview", "--help"],
        ["motion", "qc", "--help"],
        ["master", "--help"],
        ["master", "register-hero", "--help"],
        ["master", "create", "--help"],
        ["master", "animate", "--help"],
        ["master", "inspect", "--help"],
        ["master", "accept", "--help"],
        ["master", "qc", "--help"],
    ],
)
def test_every_motion_command_has_help(command: list[str]) -> None:
    result = runner.invoke(cli_app, command)
    assert result.exit_code == 0, result.output
    assert result.output.strip()


def test_top_level_help_lists_motion_and_master() -> None:
    result = runner.invoke(cli_app, ["--help"])
    assert result.exit_code == 0
    assert "motion" in result.output
    assert "master" in result.output


def test_master_create_refuses_a_captured_origin(data_root: Path) -> None:
    """A captured master is ingested, not created; the CLI says so."""
    result = invoke(
        data_root,
        "master",
        "create",
        "--name",
        "x",
        "--composition",
        "cmp_x",
        "--hero",
        "hero_x",
        "--origin",
        "captured",
    )
    assert result.exit_code == 2
    assert "app template ingest" in result.output


def test_malformed_segment_option_is_explained(data_root: Path) -> None:
    result = invoke(data_root, "motion", "compose", "--name", "x", "--segment", "")
    assert result.exit_code == 2


def test_malformed_anchor_option_is_explained(data_root: Path) -> None:
    result = invoke(
        data_root,
        "motion",
        "compose",
        "--name",
        "x",
        "--segment",
        "mot_a",
        "--anchor",
        "42",
    )
    assert result.exit_code == 2
    assert "prev_frame:next_frame" in result.output


# ---------------------------------------------------------------------------
# CLI workflow
# ---------------------------------------------------------------------------
def test_motion_and_master_workflow_through_the_cli(context, config, data_root: Path) -> None:
    a, b = mf.make_motion_pair(context, a_range=(0, 48), b_range=(0, 48))
    hero = mf.make_hero(context)
    context.close()

    inspected = invoke(data_root, "--json", "motion", "inspect", a.source.id)
    assert inspected.exit_code == 0, inspected.output
    assert payload(inspected)["validation"]["ok"] is True

    normalized = invoke(data_root, "--json", "motion", "normalize", b.source.id)
    assert normalized.exit_code == 0, normalized.output
    transform = payload(normalized)["canonical_transform"]
    assert transform["base_scale"] > 0

    anchors = invoke(
        data_root,
        "--json",
        "motion",
        "match-anchors",
        "--prev",
        a.source.id,
        "--next",
        b.source.id,
    )
    assert anchors.exit_code == 0, anchors.output
    candidates = payload(anchors)["candidates"]
    assert candidates and candidates[0]["score"] >= 0.0

    composed = invoke(
        data_root,
        "--json",
        "motion",
        "compose",
        "--name",
        "CLI composition",
        "--segment",
        a.source.id,
        "--segment",
        b.source.id,
        "--bridge-frames",
        "12",
        "--no-preview",
    )
    assert composed.exit_code == 0, composed.output
    composition_payload = payload(composed)
    composition_id = composition_payload["composition_id"]
    assert composition_payload["segments"] == 2
    assert composition_payload["joins"] == 1

    qc = invoke(data_root, "--json", "motion", "qc", composition_id)
    assert qc.exit_code == 0, qc.output
    assert payload(qc)["passed"] is True

    created = invoke(
        data_root,
        "--json",
        "master",
        "create",
        "--name",
        "CLI master",
        "--composition",
        composition_id,
        "--hero",
        hero.id,
        "--backend",
        "mock",
        "--seed",
        "1234",
    )
    assert created.exit_code == 0, created.output
    candidate_id = payload(created)["candidate"]["id"]

    animated = invoke(data_root, "--json", "master", "animate", candidate_id)
    assert animated.exit_code == 0, animated.output
    assert payload(animated)["status"] == "animated"

    master_qc = invoke(data_root, "--json", "master", "qc", candidate_id)
    assert master_qc.exit_code == 0, master_qc.output
    assert payload(master_qc)["passed"] is True

    # Not usable yet.
    inspected_master = invoke(data_root, "--json", "master", "inspect", candidate_id)
    assert payload(inspected_master)["candidate"]["status"] == "awaiting_acceptance"

    accepted = invoke(
        data_root,
        "--json",
        "master",
        "accept",
        candidate_id,
        "--by",
        "operator",
        "--reason",
        "reviewed the preview and QC; framing and bridge are correct",
    )
    assert accepted.exit_code == 0, accepted.output
    assert payload(accepted)["candidate"]["status"] == "accepted"

    # Accepted is not promoted: the garment pipeline needs a HumanTemplate.
    assert payload(accepted)["candidate"]["promoted_template_id"] is None

    promoted = invoke(
        data_root,
        "--json",
        "master",
        "promote",
        candidate_id,
        "--transition-anchor",
        "40",
        "--by",
        "operator",
    )
    assert promoted.exit_code == 0, promoted.output
    promotion = payload(promoted)["promotion"]
    assert promotion["created"] is True
    assert promotion["transition_anchor"] == 40
    assert promotion["intro"] == [0, 40]
    template_id = promotion["template_id"]

    listed = invoke(data_root, "--json", "template", "list")
    assert listed.exit_code == 0, listed.output
    assert any(t["id"] == template_id for t in payload(listed)["templates"])

    inspected_after = invoke(data_root, "--json", "master", "inspect", candidate_id)
    assert payload(inspected_after)["candidate"]["promoted_template_id"] == template_id


def test_cli_acceptance_is_refused_without_qc(context, data_root: Path) -> None:
    from app.pipeline.master_create import (
        MasterCreateOptions,
        animate_master,
        create_master_candidate,
    )
    from app.pipeline.motion_compose import ComposeOptions, SegmentSpec, compose_motion

    a, _ = mf.make_motion_pair(context, a_range=(0, 30), b_range=(0, 30))
    hero = mf.make_hero(context)
    composition = compose_motion(
        context,
        ComposeOptions(
            display_name="single",
            segments=[SegmentSpec(motion_source_id=a.source.id)],
            composition_id="cmp_cli_noqc",
            make_preview=False,
        ),
    )
    candidate = create_master_candidate(
        context,
        MasterCreateOptions(
            display_name="no qc",
            composition_id=composition.composition.id,
            hero_character_id=hero.id,
            backend_name="mock",
            candidate_id="mst_cli_noqc",
        ),
    )
    animate_master(context, candidate.id)
    context.close()

    result = invoke(
        data_root,
        "master",
        "accept",
        "mst_cli_noqc",
        "--by",
        "operator",
        "--reason",
        "trying to skip the review step entirely",
    )
    assert result.exit_code == 2
    assert "QC" in result.output


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
def test_motion_profile_endpoint(client: TestClient) -> None:
    profile = client.get("/motion/profile").json()["profile"]
    assert profile["skeleton_format"] == "coco_17"
    assert profile["canonical_shoulder_width"] > 0


def test_animators_endpoint(client: TestClient) -> None:
    animators = client.get("/master/animators").json()["animators"]
    assert animators["mock"]["capabilities"]["requires_model_weights"] is False
    assert animators["mock"]["capabilities"]["produces_photoreal"] is False
    assert animators["comfyui"]["capabilities"]["produces_photoreal"] is False
    assert animators["comfyui"]["health"]["healthy"] is False


def test_full_motion_flow_over_http(client: TestClient, api_context) -> None:
    a, b = mf.make_motion_pair(api_context, a_range=(0, 48), b_range=(0, 48))
    hero = mf.make_hero(api_context)

    listed = client.get("/motion/sources").json()
    assert {s["id"] for s in listed["motion_sources"]} == {a.source.id, b.source.id}

    validated = client.post(f"/motion/sources/{a.source.id}/validate").json()
    assert validated["ok"], validated["problems"]

    normalized = client.post(f"/motion/sources/{b.source.id}/normalize").json()
    assert normalized["canonical_transform"]["base_scale"] > 0

    anchors = client.get(
        "/motion/anchors",
        params={"prev_motion_id": a.source.id, "next_motion_id": b.source.id, "top": 3},
    ).json()
    assert len(anchors["candidates"]) == 3

    created = client.post(
        "/motion/compositions",
        json={
            "display_name": "API composition",
            "segments": [
                {"motion_source_id": a.source.id, "exposed_views": ["front"]},
                {"motion_source_id": b.source.id, "exposed_views": ["front"]},
            ],
            "joins": [{"bridge_frames": 12}],
            "make_preview": False,
        },
    )
    assert created.status_code == 201, created.text
    composition_id = created.json()["composition"]["id"]

    qc = client.post(f"/motion/compositions/{composition_id}/qc").json()
    assert qc["passed"] is True, qc["blocking_failures"]

    candidate = client.post(
        "/master/candidates",
        json={
            "display_name": "API master",
            "composition_id": composition_id,
            "hero_character_id": hero.id,
            "backend": "mock",
            "seed": 77,
        },
    )
    assert candidate.status_code == 201, candidate.text
    candidate_id = candidate.json()["candidate"]["id"]

    animated = client.post(f"/master/candidates/{candidate_id}/animate", json={}).json()
    assert animated["status"] == "animated"

    master_qc = client.post(f"/master/candidates/{candidate_id}/qc").json()
    assert master_qc["passed"] is True, master_qc["blocking_failures"]

    manifest = client.get(f"/master/candidates/{candidate_id}/manifest").json()
    assert manifest["digest"]
    assert manifest["manifest"]["contains_source_pixels"] is False

    # Still a candidate.
    state = client.get(f"/master/candidates/{candidate_id}").json()
    assert state["accepted"] is False

    accepted = client.post(
        f"/master/candidates/{candidate_id}/accept",
        json={
            "accepted_by": "operator",
            "reason": "reviewed the QC report and the framing; approved",
        },
    ).json()
    assert accepted["accepted"] is True
    assert accepted["candidate"]["promoted_template_id"] is None

    promoted = client.post(
        f"/master/candidates/{candidate_id}/promote",
        json={"transition_anchor": 32, "promoted_by": "operator"},
    )
    assert promoted.status_code == 200, promoted.text
    body = promoted.json()
    assert body["promotion"]["created"] is True
    assert body["template"]["transition_anchor_frame"] == 32
    template_id = body["template"]["id"]

    # Idempotent over HTTP too.
    again = client.post(
        f"/master/candidates/{candidate_id}/promote",
        json={"transition_anchor": 32},
    ).json()
    assert again["promotion"]["created"] is False
    assert again["template"]["id"] == template_id

    state = client.get(f"/master/candidates/{candidate_id}").json()
    assert state["promoted_template_id"] == template_id
    assert client.get(f"/templates/{template_id}").status_code == 200


def test_api_promotion_is_refused_before_acceptance(client: TestClient, api_context) -> None:
    from app.pipeline.master_create import (
        MasterCreateOptions,
        animate_master,
        create_master_candidate,
    )
    from app.pipeline.motion_compose import ComposeOptions, SegmentSpec, compose_motion

    a, _ = mf.make_motion_pair(api_context, a_range=(0, 30), b_range=(0, 30))
    hero = mf.make_hero(api_context)
    composition = compose_motion(
        api_context,
        ComposeOptions(
            display_name="unaccepted",
            segments=[SegmentSpec(motion_source_id=a.source.id, exposed_views=["front"])],
            joins=[],
            make_preview=False,
        ),
    ).composition
    candidate = create_master_candidate(
        api_context,
        MasterCreateOptions(
            display_name="unaccepted",
            composition_id=composition.id,
            hero_character_id=hero.id,
            backend_name="mock",
            seed=5,
        ),
    )
    animate_master(api_context, candidate.id)

    response = client.post(
        f"/master/candidates/{candidate.id}/promote", json={"transition_anchor": 10}
    )
    assert response.status_code == 409, response.text


def test_api_acceptance_requires_a_substantive_reason(client: TestClient, api_context) -> None:
    response = client.post(
        "/master/candidates/nope/accept",
        json={"accepted_by": "op", "reason": "ok"},
    )
    assert response.status_code == 422


def test_api_rejects_unknown_fields(client: TestClient) -> None:
    response = client.post(
        "/motion/compositions",
        json={"display_name": "x", "segments": [{"motion_source_id": "a"}], "oops": 1},
    )
    assert response.status_code == 422


def test_missing_motion_resources_return_404(client: TestClient) -> None:
    assert client.get("/motion/sources/nope").status_code == 404
    assert client.get("/motion/compositions/nope").status_code == 404
    assert client.get("/master/candidates/nope").status_code == 404


def test_unauthorized_motion_is_refused_over_http(client: TestClient, api_context) -> None:
    mf.register_motion_source(
        api_context,
        motion_id="mot_api_unauth",
        spec=mf.MOTION_A,
        start=0,
        end=30,
        authorized=False,
    )
    response = client.post(
        "/motion/compositions",
        json={
            "display_name": "unauthorized",
            "segments": [{"motion_source_id": "mot_api_unauth"}],
            "make_preview": False,
        },
    )
    assert response.status_code == 409
    assert response.json()["code"] == "conflict"


def test_openapi_documents_the_new_surface(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    for path in (
        "/motion/sources",
        "/motion/compositions",
        "/motion/anchors",
        "/master/candidates",
        "/master/candidates/{candidate_id}/accept",
    ):
        assert path in schema["paths"]
