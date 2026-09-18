"""Compatibility report: the gate in front of every render.

The compatibility engine is deliberately deterministic and rule-based. It runs
*before* any AI backend touches a frame, so an impossible outfit is rejected in
milliseconds instead of after twenty minutes of GPU time.

Final states:

``READY``
    Every blocking and needs-input rule passed. Render is allowed.
``NEEDS_INPUT``
    The operator must supply something (a back view, an expanded mask, missing
    product metadata). Render is blocked until fixed or overridden.
``INCOMPATIBLE``
    A hard rule failed (e.g. a garment that cannot cover the template's
    exposed skin). Render is blocked until fixed or overridden.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import Field, model_validator

from app.domain.base import DomainModel, Identifier, TimestampedModel
from app.domain.enums import ImageViewType, MaskKind, RuleSeverity, TemplateClothingClass


class CompatibilityState(StrEnum):
    READY = "READY"
    NEEDS_INPUT = "NEEDS_INPUT"
    INCOMPATIBLE = "INCOMPATIBLE"

    @property
    def blocks_render(self) -> bool:
        return self is not CompatibilityState.READY


class RuleOutcome(StrEnum):
    PASS = "pass"
    WARN = "warn"
    NEEDS_INPUT = "needs_input"
    FAIL = "fail"
    SKIPPED = "skipped"

    @property
    def severity(self) -> RuleSeverity:
        return {
            RuleOutcome.PASS: RuleSeverity.INFO,
            RuleOutcome.SKIPPED: RuleSeverity.INFO,
            RuleOutcome.WARN: RuleSeverity.WARNING,
            RuleOutcome.NEEDS_INPUT: RuleSeverity.NEEDS_INPUT,
            RuleOutcome.FAIL: RuleSeverity.BLOCKING,
        }[self]


class RuleResult(DomainModel):
    """Machine-readable outcome of a single rule."""

    rule_id: str = Field(min_length=1, max_length=64)
    outcome: RuleOutcome
    message: str
    #: Structured evidence, e.g. ``{"garment_sleeve": "long", "base": "none"}``.
    evidence: dict[str, Any] = Field(default_factory=dict)
    required_views: list[ImageViewType] = Field(default_factory=list)
    required_mask_expansions: list[MaskKind] = Field(default_factory=list)
    remediation: str | None = None
    #: Rule weight used for the aggregate confidence score.
    weight: float = Field(default=1.0, ge=0.0, le=10.0)

    @property
    def severity(self) -> RuleSeverity:
        return self.outcome.severity

    @property
    def blocks_render(self) -> bool:
        return self.outcome in {RuleOutcome.FAIL, RuleOutcome.NEEDS_INPUT}


class ReviewerOverride(DomainModel):
    """A human decision to render despite a blocking compatibility state.

    Overrides are additive audit records: they never mutate the rule results
    that produced the original state.
    """

    reviewer: str = Field(min_length=1, max_length=200)
    reason: str = Field(min_length=10, max_length=2000)
    approved_at: datetime
    overridden_state: CompatibilityState
    acknowledged_rule_ids: list[str] = Field(default_factory=list)
    expires_at: datetime | None = None

    @model_validator(mode="after")
    def _tz_aware(self) -> Self:
        for name in ("approved_at", "expires_at"):
            value = getattr(self, name)
            if value is not None and value.tzinfo is None:
                raise ValueError(f"{name} must be timezone-aware (UTC)")
        if self.overridden_state is CompatibilityState.READY:
            raise ValueError("an override only makes sense for a blocking state")
        return self

    def is_valid_at(self, moment: datetime) -> bool:
        return self.expires_at is None or self.expires_at > moment


class CompatibilityReport(TimestampedModel):
    """Aggregated result of the rule engine for one (template, garment) pair."""

    id: Identifier
    template_id: Identifier
    template_version: int = Field(ge=1)
    garment_id: Identifier
    garment_version: int = Field(ge=1)

    rules_version: str = Field(min_length=1, max_length=64)
    rules_file_sha256: str | None = None

    state: CompatibilityState
    results: list[RuleResult] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    required_missing_views: list[ImageViewType] = Field(default_factory=list)
    required_mask_expansions: list[MaskKind] = Field(default_factory=list)
    recommended_template_class: TemplateClothingClass | None = None
    confidence: float = Field(ge=0.0, le=1.0)

    override: ReviewerOverride | None = None
    override_audit_trail: list[ReviewerOverride] = Field(default_factory=list)

    @model_validator(mode="after")
    def _state_matches_results(self) -> Self:
        if not self.results:
            return self
        outcomes = {result.outcome for result in self.results}
        if RuleOutcome.FAIL in outcomes:
            expected = CompatibilityState.INCOMPATIBLE
        elif RuleOutcome.NEEDS_INPUT in outcomes:
            expected = CompatibilityState.NEEDS_INPUT
        else:
            expected = CompatibilityState.READY
        if self.state is not expected:
            raise ValueError(
                f"state {self.state.value} contradicts rule outcomes (expected {expected.value})"
            )
        return self

    # -- decisions --------------------------------------------------------
    def render_allowed(self, moment: datetime, *, allow_override: bool = True) -> bool:
        """Whether a render may proceed for this report."""
        if self.state is CompatibilityState.READY:
            return True
        if not allow_override or self.override is None:
            return False
        return self.override.is_valid_at(moment)

    def block_explanation(self) -> str:
        if self.state is CompatibilityState.READY:
            return ""
        reasons = self.blocking_reasons or [r.message for r in self.results if r.blocks_render]
        return f"{self.state.value}: " + "; ".join(reasons)

    def failed_rules(self) -> list[RuleResult]:
        return [r for r in self.results if r.blocks_render]

    def with_override(self, override: ReviewerOverride) -> CompatibilityReport:
        """Return a copy carrying ``override`` plus the appended audit trail."""
        trail = [*self.override_audit_trail]
        if self.override is not None:
            trail.append(self.override)
        return self.model_copy(update={"override": override, "override_audit_trail": trail})


__all__ = [
    "CompatibilityReport",
    "CompatibilityState",
    "ReviewerOverride",
    "RuleOutcome",
    "RuleResult",
]
