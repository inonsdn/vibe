"""HTTP API: the API and CLI must use the same core services."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.api.main import create_app
from app.domain.enums import ImageViewType, MaskKind
from tests import fixtures
from tests.conftest import requires_ffmpeg


@pytest.fixture
def client(config) -> Iterator[TestClient]:
    application = create_app(config, configure_logs=False)
    with TestClient(application) as test_client:
        yield test_client
    deps.close()


@pytest.fixture
def api_context(client: TestClient):
    """The same ServiceContext the API endpoints use."""
    return deps.get_context()


# -- system ----------------------------------------------------------------
def test_health_reports_schema_and_backends(client: TestClient) -> None:
    payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert payload["schema_version"] == 3
    assert set(payload["backends"]) == {"mock", "comfyui"}
    assert payload["offline_ok"] is True


def test_offline_verify_endpoint(client: TestClient) -> None:
    payload = client.get("/offline/verify?probe_comfyui=false").json()
    assert payload["ok"] is True
    assert payload["cloud_integrations"] == []
    assert payload["network_features"] == []
    assert all(e["is_local"] for e in payload["endpoints"])


def test_adapters_endpoint_is_honest(client: TestClient) -> None:
    payload = client.get("/adapters").json()
    assert len(payload["adapters"]) == 7
    assert all(entry["status"] == "not_implemented" for entry in payload["adapters"].values())
    assert "No neural model is implemented or downloaded" in payload["note"]


def test_config_endpoint_exposes_the_hash(client: TestClient) -> None:
    payload = client.get("/config").json()
    assert len(payload["config_hash"]) == 64
    assert payload["config"]["video"]["pixel_format"] == "yuv420p"


def test_backends_endpoint_reports_capabilities(client: TestClient) -> None:
    payload = client.get("/jobs/backends").json()["backends"]
    assert payload["mock"]["capabilities"]["requires_model_weights"] is False
    assert payload["mock"]["health"]["healthy"] is True
    # ComfyUI is not running; the API says so rather than failing.
    assert payload["comfyui"]["health"]["healthy"] is False


def test_correlation_id_header_is_set(client: TestClient) -> None:
    response = client.get("/health")
    assert response.headers["x-correlation-id"]


def test_index_page_renders(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "garment-replacer" in response.text
    assert "No AI model weights" in response.text


def test_openapi_schema_is_available(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    for path in ("/templates", "/garments", "/compatibility/check", "/jobs"):
        assert path in schema["paths"]


# -- resources -------------------------------------------------------------
def test_compatibility_rules_endpoint(client: TestClient) -> None:
    payload = client.get("/compatibility/rules").json()
    assert payload["rules_version"] == "1"
    assert len(payload["rules"]) == 14
    assert len(payload["rules_file_sha256"]) == 64
    assert set(payload["rules"]) <= set(payload["implemented_rule_ids"])


def test_full_flow_over_http(client: TestClient, api_context) -> None:
    """Create -> check -> job -> render, all through the API."""
    fixtures.make_template(api_context, template_id="tpl_api")
    fixtures.make_garment(api_context, garment_id="grm_api")

    listed = client.get("/templates").json()
    assert any(entry["id"] == "tpl_api" for entry in listed["templates"])

    detail = client.get("/templates/tpl_api").json()["template"]
    assert detail["transition_anchor_frame"] == fixtures.ANCHOR

    validation = client.post("/templates/tpl_api/validate").json()
    assert validation["ok"], validation["problems"]

    check = client.post(
        "/compatibility/check",
        json={
            "template_id": "tpl_api",
            "garment_id": "grm_api",
            "exposed_views": ["front", "back", "side"],
        },
    ).json()
    assert check["state"] == "READY"
    assert check["render_allowed"] is True

    created = client.post(
        "/jobs",
        json={"template_id": "tpl_api", "garment_id": "grm_api", "backend": "mock", "seed": 5},
    )
    assert created.status_code == 201
    job_id = created.json()["job"]["id"]

    rendered = client.post(f"/jobs/{job_id}/render", json={}).json()
    assert rendered["rendered_frames"] == fixtures.TOTAL_FRAMES - fixtures.ANCHOR
    assert rendered["leaked_frames"] == []

    status = client.get(f"/jobs/{job_id}").json()
    assert status["remaining_frames"] == 0
    assert status["job"]["status"] == "rendered"


def test_blocked_compatibility_returns_409(client: TestClient, api_context) -> None:
    fixtures.make_template(api_context, template_id="tpl_block")
    fixtures.make_garment(api_context, garment_id="grm_front_only", views=(ImageViewType.FRONT,))
    check = client.post(
        "/compatibility/check",
        json={
            "template_id": "tpl_block",
            "garment_id": "grm_front_only",
            "exposed_views": ["front", "back"],
        },
    ).json()
    assert check["state"] == "NEEDS_INPUT"

    response = client.post(
        "/jobs",
        json={"template_id": "tpl_block", "garment_id": "grm_front_only", "backend": "mock"},
    )
    assert response.status_code == 409
    body = response.json()
    assert body["code"] == "compatibility_blocked"
    assert "back" in body["details"]["required_missing_views"]


def test_override_endpoint_unblocks_a_job(client: TestClient, api_context) -> None:
    fixtures.make_template(api_context, template_id="tpl_ovr")
    fixtures.make_garment(api_context, garment_id="grm_ovr", views=(ImageViewType.FRONT,))
    report_id = client.post(
        "/compatibility/check",
        json={
            "template_id": "tpl_ovr",
            "garment_id": "grm_ovr",
            "exposed_views": ["front", "back"],
        },
    ).json()["report_id"]

    overridden = client.post(
        f"/compatibility/reports/{report_id}/override",
        json={"reviewer": "operator", "reason": "the back is never shown in this edit"},
    ).json()
    assert overridden["overridden"] is True
    assert overridden["render_allowed"] is True

    created = client.post(
        "/jobs", json={"template_id": "tpl_ovr", "garment_id": "grm_ovr", "backend": "mock"}
    )
    assert created.status_code == 201


def test_missing_resources_return_404(client: TestClient) -> None:
    assert client.get("/templates/nope").status_code == 404
    assert client.get("/garments/nope").status_code == 404
    assert client.get("/jobs/nope").status_code == 404
    assert client.get("/compatibility/reports/nope").status_code == 404


def test_invalid_payloads_are_rejected(client: TestClient) -> None:
    # Unknown field (schemas forbid extras).
    assert (
        client.post("/jobs", json={"template_id": "a", "garment_id": "b", "oops": 1}).status_code
        == 422
    )
    # Missing required field.
    assert client.post("/compatibility/check", json={"template_id": "a"}).status_code == 422
    # Out-of-range value.
    assert (
        client.post(
            "/garments",
            json={
                "images": [{"path": "x.png", "view": "front"}],
                "category": "top",
                "body_coverage": "torso",
                "silhouette": "fitted",
                "material": "cotton",
                "transparency": 5.0,
            },
        ).status_code
        == 422
    )


def test_path_traversal_through_the_api_is_rejected(client: TestClient, api_context) -> None:
    """Requirement 13, over HTTP."""
    fixtures.make_template(api_context, template_id="tpl_trav")
    response = client.post(
        "/templates/tpl_trav/masks",
        json={"kind": "garment", "source_dir": "../../../etc"},
    )
    assert response.status_code in (400, 404, 422)


def test_garment_ingestion_over_http(client: TestClient, api_context, tmp_path: Path) -> None:
    front = fixtures.write_garment_image(tmp_path / "front.png")
    back = fixtures.write_garment_image(tmp_path / "back.png", color=(200, 80, 60))
    response = client.post(
        "/garments",
        json={
            "images": [
                {"path": str(front), "view": "front"},
                {"path": str(back), "view": "back"},
            ],
            "category": "top",
            "body_coverage": "torso",
            "silhouette": "fitted",
            "material": "cotton",
            "sleeve_length": "short",
            "garment_length": "hip",
            "product_name": "API Top",
            "brand": "Test",
            "license": "test-only",
            "pattern_description": "striped",
        },
    )
    assert response.status_code == 201
    garment = response.json()["garment"]
    assert {image["view"] for image in garment["images"]} == {"front", "back"}
    assert garment["dominant_colors"]


def test_mask_import_over_http(client: TestClient, api_context, tmp_path: Path) -> None:
    from app.media.frames import frame_path
    from app.media.masks import save_mask

    fixtures.make_template(api_context, template_id="tpl_masks", with_masks=(MaskKind.GARMENT,))
    staging = tmp_path / "expansion"
    staging.mkdir()
    for index in range(fixtures.ANCHOR, fixtures.TOTAL_FRAMES):
        save_mask(frame_path(staging, index), fixtures.expansion_mask(index))

    response = client.post(
        "/templates/tpl_masks/masks",
        json={"kind": "expansion", "source_dir": str(staging)},
    )
    assert response.status_code == 200
    assert response.json()["imported_count"] == fixtures.TOTAL_FRAMES - fixtures.ANCHOR


@requires_ffmpeg
def test_compose_and_qc_over_http(client: TestClient, api_context) -> None:
    fixtures.make_template(api_context, template_id="tpl_full")
    fixtures.make_garment(api_context, garment_id="grm_full")
    client.post(
        "/compatibility/check",
        json={
            "template_id": "tpl_full",
            "garment_id": "grm_full",
            "exposed_views": ["front", "back", "side"],
        },
    )
    job_id = client.post(
        "/jobs", json={"template_id": "tpl_full", "garment_id": "grm_full", "backend": "mock"}
    ).json()["job"]["id"]
    client.post(f"/jobs/{job_id}/render", json={})

    composed = client.post(f"/jobs/{job_id}/compose", json={"include_audio": False}).json()
    assert composed["total_frames"] == fixtures.TOTAL_FRAMES
    assert composed["transition"]["last_intro_matches_cache"] is True

    qc = client.post(f"/jobs/{job_id}/qc", json={}).json()
    assert qc["passed"] is True, qc["blocking_failures"]

    manifest = client.get(f"/jobs/{job_id}/manifest").json()
    assert manifest["missing_fields"] == []
    assert manifest["digest"] == composed["manifest_digest"]


def test_render_in_slices_over_http(client: TestClient, api_context) -> None:
    fixtures.make_template(api_context, template_id="tpl_slice")
    fixtures.make_garment(api_context, garment_id="grm_slice")
    client.post(
        "/compatibility/check",
        json={
            "template_id": "tpl_slice",
            "garment_id": "grm_slice",
            "exposed_views": ["front", "back", "side"],
        },
    )
    job_id = client.post(
        "/jobs", json={"template_id": "tpl_slice", "garment_id": "grm_slice", "backend": "mock"}
    ).json()["job"]["id"]

    first = client.post(f"/jobs/{job_id}/render", json={"max_frames": 4}).json()
    assert first["rendered_frames"] == 4
    assert first["status"] == "paused"

    second = client.post(f"/jobs/{job_id}/render", json={"resume": True}).json()
    assert second["skipped_frames"] == 4
    assert second["status"] == "rendered"


def test_audit_endpoint_lists_events(client: TestClient, api_context) -> None:
    fixtures.make_template(api_context, template_id="tpl_audit")
    fixtures.make_garment(api_context, garment_id="grm_audit")
    client.post(
        "/compatibility/check",
        json={
            "template_id": "tpl_audit",
            "garment_id": "grm_audit",
            "exposed_views": ["front", "back", "side"],
        },
    )
    job_id = client.post(
        "/jobs", json={"template_id": "tpl_audit", "garment_id": "grm_audit", "backend": "mock"}
    ).json()["job"]["id"]
    events = api_context.repos.audit.for_entity("job", job_id)
    assert any(event["event"] == "job_created" for event in events)
