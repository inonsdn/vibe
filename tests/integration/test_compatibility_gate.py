"""The compatibility gate in front of rendering (requirements 6 and 7)."""

from __future__ import annotations

import pytest

from app.core.errors import CompatibilityBlockedError, ConflictError, ValidationError
from app.core.ids import utc_now
from app.domain.compatibility import CompatibilityState
from app.domain.enums import GarmentCategory, ImageViewType, MaskKind, Material, SleeveLength
from app.pipeline import compat_service
from app.pipeline.render import JobCreateOptions, create_job
from tests import fixtures


def check(context, template, garment, **kwargs):
    return compat_service.check_compatibility(
        context,
        template.template.id,
        garment.id,
        options=compat_service.CheckOptions(**kwargs),
    )


def test_ready_pair_can_create_a_job(context, ready_pair) -> None:
    template, garment, report = ready_pair
    assert report.state is CompatibilityState.READY
    job = create_job(
        context, template.template.id, garment.id, JobCreateOptions(backend_name="mock")
    )
    assert job.compatibility_report_id == report.id


def test_rendering_without_a_report_is_blocked(context, template, garment) -> None:
    with pytest.raises(CompatibilityBlockedError, match="No compatibility report"):
        create_job(context, template.template.id, garment.id, JobCreateOptions(backend_name="mock"))


def test_needs_input_blocks_job_creation(context, template) -> None:
    """Requirement 7: a missing back view blocks rendering."""
    garment = fixtures.make_garment(
        context, garment_id="grm_front_only", views=(ImageViewType.FRONT,)
    )
    report = check(
        context,
        template,
        garment,
        exposed_views=[ImageViewType.FRONT, ImageViewType.BACK],
    )
    assert report.state is CompatibilityState.NEEDS_INPUT
    assert ImageViewType.BACK in report.required_missing_views

    with pytest.raises(CompatibilityBlockedError) as exc:
        create_job(context, template.template.id, garment.id, JobCreateOptions(backend_name="mock"))
    assert exc.value.details["state"] == "NEEDS_INPUT"
    assert "back" in exc.value.details["required_missing_views"]


def test_incompatible_blocks_job_creation(context, template) -> None:
    garment = fixtures.make_garment(
        context,
        garment_id="grm_bad",
        category=GarmentCategory.TOP,
        sleeve=SleeveLength.NONE,
        coverage="minimal",  # type: ignore[arg-type]
    )
    report = check(context, template, garment, exposed_views=[ImageViewType.FRONT])
    assert report.state is CompatibilityState.INCOMPATIBLE
    with pytest.raises(CompatibilityBlockedError) as exc:
        create_job(context, template.template.id, garment.id, JobCreateOptions(backend_name="mock"))
    assert exc.value.details["state"] == "INCOMPATIBLE"


def test_a_reviewed_override_unblocks_rendering(context, template) -> None:
    garment = fixtures.make_garment(
        context, garment_id="grm_needs_back", views=(ImageViewType.FRONT,)
    )
    report = check(
        context, template, garment, exposed_views=[ImageViewType.FRONT, ImageViewType.BACK]
    )
    assert not report.render_allowed(utc_now())

    overridden = compat_service.store_override(
        context,
        report.id,
        reviewer="operator",
        reason="the choreography never turns; the back is never visible",
    )
    assert overridden.render_allowed(utc_now())
    job = create_job(
        context, template.template.id, garment.id, JobCreateOptions(backend_name="mock")
    )
    assert job.compatibility_report_id == report.id

    # The audit trail records who and why.
    events = context.repos.audit.for_entity("compatibility_report", report.id)
    stored = next(e for e in events if e["event"] == "compatibility_override_stored")
    assert stored["actor"] == "operator"
    assert "never turns" in stored["details"]["reason"]


def test_an_expired_override_blocks_again(context, template) -> None:
    from datetime import timedelta

    garment = fixtures.make_garment(
        context, garment_id="grm_expiring", views=(ImageViewType.FRONT,)
    )
    report = check(
        context, template, garment, exposed_views=[ImageViewType.FRONT, ImageViewType.BACK]
    )
    overridden = compat_service.store_override(
        context,
        report.id,
        reviewer="operator",
        reason="temporary approval for a single test render",
        expires_in_hours=1,
        approved_at=utc_now() - timedelta(hours=2),
    )
    assert not overridden.render_allowed(utc_now())
    with pytest.raises(CompatibilityBlockedError):
        create_job(context, template.template.id, garment.id, JobCreateOptions(backend_name="mock"))


def test_override_requires_a_substantive_reason(context, template) -> None:
    garment = fixtures.make_garment(context, garment_id="grm_r", views=(ImageViewType.FRONT,))
    report = check(
        context, template, garment, exposed_views=[ImageViewType.FRONT, ImageViewType.BACK]
    )
    with pytest.raises(ValidationError, match="real explanation"):
        compat_service.store_override(context, report.id, reviewer="op", reason="fine")


def test_override_of_a_ready_report_is_refused(context, ready_pair) -> None:
    _, _, report = ready_pair
    with pytest.raises(ValidationError, match="already READY"):
        compat_service.store_override(
            context, report.id, reviewer="op", reason="not needed, already fine"
        )


def test_overrides_can_be_disabled_by_configuration(config, data_root) -> None:
    from app.pipeline.context import ServiceContext

    strict = config.with_overrides(compatibility={"allow_override": False})
    with ServiceContext.create(config=strict, configure_logs=False) as ctx:
        template = fixtures.make_template(ctx)
        garment = fixtures.make_garment(ctx, views=(ImageViewType.FRONT,))
        report = compat_service.check_compatibility(
            ctx,
            template.template.id,
            garment.id,
            options=compat_service.CheckOptions(
                exposed_views=[ImageViewType.FRONT, ImageViewType.BACK]
            ),
        )
        with pytest.raises(ConflictError, match="disabled by configuration"):
            compat_service.store_override(
                ctx, report.id, reviewer="op", reason="should not be permitted at all"
            )


def test_protected_mask_override_requires_explicit_acknowledgement(context, template) -> None:
    """Protected pixels stay protected unless the override names them."""
    from app.pipeline.render import _override_permits_protected_edit

    garment = fixtures.make_garment(context, garment_id="grm_p", views=(ImageViewType.FRONT,))
    report = check(
        context, template, garment, exposed_views=[ImageViewType.FRONT, ImageViewType.BACK]
    )
    compat_service.store_override(
        context,
        report.id,
        reviewer="operator",
        reason="back view is not needed for this choreography",
        acknowledged_rule_ids=["missing_views_for_motion"],
    )
    job = create_job(
        context, template.template.id, garment.id, JobCreateOptions(backend_name="mock")
    )
    assert _override_permits_protected_edit(context, job) is False

    compat_service.store_override(
        context,
        report.id,
        reviewer="operator",
        reason="deliberately permitting edits inside protected regions for a test",
        acknowledged_rule_ids=["missing_views_for_motion", "protected_mask_override"],
    )
    job2 = create_job(
        context,
        template.template.id,
        garment.id,
        JobCreateOptions(backend_name="mock", job_id="job_override"),
    )
    assert _override_permits_protected_edit(context, job2) is True


def test_evaluation_context_is_derived_from_disk(context) -> None:
    """Occlusion/expansion mask presence is measured, not declared."""
    without = fixtures.make_template(
        context,
        template_id="tpl_no_occl",
        with_masks=(MaskKind.GARMENT, MaskKind.PROTECTED),
    )
    evaluation = compat_service.build_evaluation_context(context, without.template)
    assert evaluation.has_occlusion_masks is False
    assert evaluation.has_expansion_masks is False

    with_all = fixtures.make_template(
        context,
        template_id="tpl_all",
        with_masks=(MaskKind.GARMENT, MaskKind.PROTECTED, MaskKind.OCCLUSION, MaskKind.EXPANSION),
    )
    evaluation2 = compat_service.build_evaluation_context(context, with_all.template)
    assert evaluation2.has_occlusion_masks is True
    assert evaluation2.has_expansion_masks is True
    assert evaluation2.base_garment_luminance is not None
    assert evaluation2.body_region_aspect_ratio is not None


def test_missing_occlusion_masks_block_a_render(context) -> None:
    template = fixtures.make_template(
        context,
        template_id="tpl_occl_missing",
        with_masks=(MaskKind.GARMENT, MaskKind.PROTECTED),
    )
    garment = fixtures.make_garment(context, garment_id="grm_ok")
    report = compat_service.check_compatibility(
        context,
        template.template.id,
        garment.id,
        options=compat_service.CheckOptions(
            exposed_views=[ImageViewType.FRONT, ImageViewType.BACK, ImageViewType.SIDE]
        ),
    )
    assert report.state is CompatibilityState.NEEDS_INPUT
    assert MaskKind.OCCLUSION in report.required_mask_expansions


def test_report_is_persisted_and_retrievable(context, ready_pair) -> None:
    _, _, report = ready_pair
    stored = context.repos.compatibility.get(report.id)
    assert stored.state is report.state
    latest = context.repos.compatibility.latest_for_pair(
        report.template_id, report.template_version, report.garment_id, report.garment_version
    )
    assert latest is not None and latest.id == report.id


def test_summary_is_operator_friendly(context, ready_pair) -> None:
    _, _, report = ready_pair
    summary = compat_service.summarize(report)
    assert summary["state"] == "READY"
    assert summary["render_allowed"] is True
    assert len(summary["rules"]) == len(report.results)
    assert summary["rules_version"] == "1"


def test_transparent_garment_over_a_normal_base_is_blocked(context, template) -> None:
    garment = fixtures.make_garment(
        context,
        garment_id="grm_sheer",
        material=Material.MESH,
        transparency=0.7,
    )
    report = check(
        context,
        template,
        garment,
        exposed_views=[ImageViewType.FRONT, ImageViewType.BACK, ImageViewType.SIDE],
    )
    assert report.state is CompatibilityState.NEEDS_INPUT
    assert report.recommended_template_class is not None
    with pytest.raises(CompatibilityBlockedError):
        create_job(context, template.template.id, garment.id, JobCreateOptions(backend_name="mock"))
