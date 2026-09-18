"""Compatibility as a service: evaluate, persist, override.

Separated from the rule engine so the engine itself stays a pure function of
its inputs and can be unit-tested without a database.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from app.core.errors import ConflictError, ValidationError
from app.core.ids import utc_now
from app.core.logging import get_logger, log_event
from app.domain.compatibility import CompatibilityReport, CompatibilityState, ReviewerOverride
from app.domain.enums import ImageViewType, MaskKind
from app.domain.human_template import HumanTemplate
from app.media.frames import frame_path, list_frame_indices, read_frame_at
from app.media.masks import load_mask
from app.pipeline.compatibility import EvaluationContext, evaluate, load_rule_set_for
from app.pipeline.context import ServiceContext

logger = get_logger(__name__)


@dataclass
class CheckOptions:
    #: Views the operator knows the performance exposes. ``None`` => unknown.
    exposed_views: list[ImageViewType] | None = None
    #: Measure the base garment's luminance from the source frames.
    measure_base_luminance: bool = True
    report_id: str | None = None


def build_evaluation_context(
    context: ServiceContext,
    template: HumanTemplate,
    options: CheckOptions | None = None,
) -> EvaluationContext:
    """Derive the facts the rules need from what is actually on disk."""
    opts = options or CheckOptions()
    occlusion_dir = context.absolute(template.directories.mask_dir(MaskKind.OCCLUSION))
    expansion_dir = context.absolute(template.directories.mask_dir(MaskKind.EXPANSION))

    has_occlusion = bool(list_frame_indices(occlusion_dir))
    has_expansion = bool(list_frame_indices(expansion_dir))

    luminance: float | None = None
    aspect: float | None = None
    if opts.measure_base_luminance:
        luminance, aspect = _measure_base_region(context, template)

    return EvaluationContext(
        exposed_views=set(opts.exposed_views) if opts.exposed_views is not None else None,
        has_occlusion_masks=has_occlusion,
        has_expansion_masks=has_expansion,
        base_garment_luminance=luminance,
        body_region_aspect_ratio=aspect,
        notes={
            "occlusion_mask_count": len(list_frame_indices(occlusion_dir)),
            "expansion_mask_count": len(list_frame_indices(expansion_dir)),
        },
    )


def _measure_base_region(
    context: ServiceContext, template: HumanTemplate
) -> tuple[float | None, float | None]:
    """Mean luminance and aspect ratio of the base garment region.

    Measured on the first reveal frame that has a garment mask. Returns
    ``(None, None)`` when no mask exists yet — the rules treat that as unknown
    rather than assuming anything.
    """
    frames_dir = context.absolute(template.directories.source_frames)
    mask_dir = context.absolute(template.directories.mask_dir(MaskKind.GARMENT))
    for index in range(template.reveal.start, template.reveal.end):
        mask_path = frame_path(mask_dir, index)
        if not mask_path.is_file():
            continue
        try:
            frame = read_frame_at(frames_dir, index)
            mask = load_mask(mask_path, expect_shape=frame.shape[:2])
        except Exception:
            return None, None
        selection = mask > 0
        if not selection.any():
            continue
        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        luminance = float(gray[selection].mean() / 255.0)
        ys, xs = np.nonzero(selection)
        height = float(ys.max() - ys.min() + 1)
        width = float(xs.max() - xs.min() + 1)
        return luminance, (width / height if height else None)
    return None, None


def check_compatibility(
    context: ServiceContext,
    template_id: str,
    garment_id: str,
    *,
    template_version: int | None = None,
    garment_version: int | None = None,
    options: CheckOptions | None = None,
    persist: bool = True,
) -> CompatibilityReport:
    """Evaluate a template/garment pair and (by default) store the report."""
    template = context.repos.templates.get(template_id, template_version)
    garment = context.repos.garments.get(garment_id, garment_version)
    rule_set = load_rule_set_for(context.config)
    evaluation_context = build_evaluation_context(context, template, options)

    report = evaluate(
        template,
        garment,
        rule_set,
        context=evaluation_context,
        report_id=(options.report_id if options else None),
    )
    if persist:
        report = context.repos.compatibility.save(report)
        context.repos.audit.record(
            "compatibility_checked",
            entity_type="compatibility_report",
            entity_id=report.id,
            details={
                "template": f"{template.id}@v{template.version}",
                "garment": f"{garment.id}@v{garment.version}",
                "state": report.state.value,
                "confidence": report.confidence,
                "rules_version": report.rules_version,
            },
        )
    log_event(
        logger,
        "compatibility_checked",
        report_id=report.id,
        state=report.state.value,
        confidence=report.confidence,
        blocking=len(report.blocking_reasons),
    )
    return report


def store_override(
    context: ServiceContext,
    report_id: str,
    *,
    reviewer: str,
    reason: str,
    acknowledged_rule_ids: list[str] | None = None,
    expires_in_hours: float | None = None,
    approved_at: datetime | None = None,
) -> CompatibilityReport:
    """Attach a reviewed override to a blocking compatibility report.

    The override is an additive audit record: previous overrides are kept in
    ``override_audit_trail`` and the rule results are never rewritten.
    """
    if not context.config.compatibility.allow_override:
        raise ConflictError(
            "Overrides are disabled by configuration",
            hint="Set compatibility.allow_override: true to permit reviewed overrides.",
        )
    report = context.repos.compatibility.get(report_id)
    if report.state is CompatibilityState.READY:
        raise ValidationError(
            "Report is already READY; an override would be meaningless",
            report_id=report_id,
        )
    if len(reason.strip()) < 10:
        raise ValidationError(
            "An override reason must be a real explanation (10+ characters)",
            reason=reason,
        )

    moment = approved_at or utc_now()
    override = ReviewerOverride(
        reviewer=reviewer,
        reason=reason,
        approved_at=moment,
        overridden_state=report.state,
        acknowledged_rule_ids=acknowledged_rule_ids or [r.rule_id for r in report.failed_rules()],
        expires_at=(moment + timedelta(hours=expires_in_hours)) if expires_in_hours else None,
    )
    updated = context.repos.compatibility.save(report.with_override(override))
    context.repos.audit.record(
        "compatibility_override_stored",
        actor=reviewer,
        entity_type="compatibility_report",
        entity_id=report.id,
        details={
            "state": report.state.value,
            "reason": reason,
            "acknowledged_rule_ids": override.acknowledged_rule_ids,
            "expires_at": override.expires_at.isoformat() if override.expires_at else None,
        },
    )
    log_event(
        logger,
        "compatibility_override_stored",
        report_id=report.id,
        reviewer=reviewer,
        state=report.state.value,
    )
    return updated


def summarize(report: CompatibilityReport) -> dict[str, Any]:
    """Compact, operator-facing summary of a report."""
    return {
        "report_id": report.id,
        "template": f"{report.template_id}@v{report.template_version}",
        "garment": f"{report.garment_id}@v{report.garment_version}",
        "state": report.state.value,
        "confidence": report.confidence,
        "rules_version": report.rules_version,
        "blocking_reasons": report.blocking_reasons,
        "warnings": report.warnings,
        "required_missing_views": [v.value for v in report.required_missing_views],
        "required_mask_expansions": [k.value for k in report.required_mask_expansions],
        "recommended_template_class": (
            report.recommended_template_class.value if report.recommended_template_class else None
        ),
        "overridden": bool(report.override),
        "render_allowed": report.render_allowed(utc_now()),
        "rules": [
            {
                "rule_id": result.rule_id,
                "outcome": result.outcome.value,
                "message": result.message,
            }
            for result in report.results
        ],
    }


__all__ = [
    "CheckOptions",
    "build_evaluation_context",
    "check_compatibility",
    "store_override",
    "summarize",
]
