"""Compatibility endpoints."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from app.api.deps import context_dependency
from app.api.schemas import CompatibilityCheckRequest, OverrideRequest
from app.pipeline import compat_service
from app.pipeline.compatibility import load_rule_set_for, registered_rule_ids
from app.pipeline.context import ServiceContext

router = APIRouter(prefix="/compatibility", tags=["compatibility"])

Ctx = Annotated[ServiceContext, Depends(context_dependency)]


@router.get("/rules")
def rules(context: Ctx) -> dict[str, Any]:
    rule_set = load_rule_set_for(context.config)
    return {
        "rules_version": rule_set.rules_version,
        "rules_file_sha256": rule_set.sha256(),
        "description": rule_set.description,
        "implemented_rule_ids": registered_rule_ids(),
        "rules": {
            rule_id: {
                "enabled": config.enabled,
                "severity": config.severity.value,
                "weight": config.weight,
                "params": config.params,
            }
            for rule_id, config in sorted(rule_set.rules.items())
        },
    }


@router.post("/check")
def check(payload: CompatibilityCheckRequest, context: Ctx) -> dict[str, Any]:
    report = compat_service.check_compatibility(
        context,
        payload.template_id,
        payload.garment_id,
        template_version=payload.template_version,
        garment_version=payload.garment_version,
        options=compat_service.CheckOptions(exposed_views=payload.exposed_views),
    )
    return compat_service.summarize(report)


@router.get("/reports/{report_id}")
def get_report(report_id: str, context: Ctx) -> dict[str, Any]:
    report = context.repos.compatibility.get(report_id)
    return {"report": report.to_json_dict(), "summary": compat_service.summarize(report)}


@router.post("/reports/{report_id}/override")
def override(report_id: str, payload: OverrideRequest, context: Ctx) -> dict[str, Any]:
    report = compat_service.store_override(
        context,
        report_id,
        reviewer=payload.reviewer,
        reason=payload.reason,
        acknowledged_rule_ids=payload.acknowledged_rule_ids or None,
        expires_in_hours=payload.expires_in_hours,
    )
    return compat_service.summarize(report)


__all__ = ["router"]
