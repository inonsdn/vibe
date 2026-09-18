"""Deterministic garment/template compatibility engine.

Each rule is a small pure function of ``(template, garment, params, context)``
returning a :class:`~app.domain.compatibility.RuleResult`. The YAML file decides
which rules run, what severity a trigger carries and what the thresholds are, so
tuning behaviour never requires touching this module.

The aggregate state is the worst outcome seen:
``FAIL -> INCOMPATIBLE``, else ``NEEDS_INPUT -> NEEDS_INPUT``, else ``READY``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.config import AppConfig, load_yaml
from app.core.errors import ConfigError
from app.core.hashing import sha256_file
from app.core.ids import report_id as new_report_id
from app.core.logging import get_logger
from app.domain.compatibility import (
    CompatibilityReport,
    CompatibilityState,
    RuleOutcome,
    RuleResult,
)
from app.domain.enums import (
    CATEGORY_MIN_COVERAGE,
    COVERAGE_ORDER,
    LENGTH_ORDER,
    REFLECTIVE_MATERIALS,
    SILHOUETTE_ORDER,
    SLEEVE_ORDER,
    TEMPLATE_CLASS_COVERAGE,
    BodyCoverage,
    GarmentCategory,
    GarmentLength,
    ImageViewType,
    MaskKind,
    RuleSeverity,
    Silhouette,
    SleeveLength,
    TemplateClothingClass,
    scale_index,
)
from app.domain.garment import GarmentAsset
from app.domain.human_template import HumanTemplate

logger = get_logger(__name__)

_SEVERITY_TO_OUTCOME = {
    RuleSeverity.WARNING: RuleOutcome.WARN,
    RuleSeverity.NEEDS_INPUT: RuleOutcome.NEEDS_INPUT,
    RuleSeverity.BLOCKING: RuleOutcome.FAIL,
    RuleSeverity.INFO: RuleOutcome.PASS,
}


@dataclass
class RuleConfig:
    rule_id: str
    enabled: bool
    severity: RuleSeverity
    weight: float
    params: dict[str, Any] = field(default_factory=dict)

    def outcome(self) -> RuleOutcome:
        return _SEVERITY_TO_OUTCOME[self.severity]


@dataclass
class RuleSet:
    """A loaded, versioned rule file."""

    rules_version: str
    description: str
    category_matrix: dict[str, list[str]]
    rules: dict[str, RuleConfig]
    penalties: dict[str, float]
    confidence_floor: float
    source_path: Path | None = None

    def sha256(self) -> str | None:
        return sha256_file(self.source_path) if self.source_path else None

    def config_for(self, rule_id: str) -> RuleConfig | None:
        return self.rules.get(rule_id)


@dataclass
class EvaluationContext:
    """Extra facts the rules may consult beyond the two records themselves."""

    #: Views the performance actually exposes, when known.
    exposed_views: set[ImageViewType] | None = None
    #: Whether the template has non-empty occlusion masks on disk.
    has_occlusion_masks: bool | None = None
    #: Whether the template has non-empty expansion masks on disk.
    has_expansion_masks: bool | None = None
    #: Mean luminance of the base garment region, 0..1, when measured.
    base_garment_luminance: float | None = None
    #: Aspect ratio of the body region in the frame, when measured.
    body_region_aspect_ratio: float | None = None
    notes: dict[str, Any] = field(default_factory=dict)


RuleFunc = Callable[
    [HumanTemplate, GarmentAsset, RuleConfig, EvaluationContext, RuleSet], RuleResult
]

_RULES: dict[str, RuleFunc] = {}


def rule(rule_id: str) -> Callable[[RuleFunc], RuleFunc]:
    def decorator(func: RuleFunc) -> RuleFunc:
        _RULES[rule_id] = func
        return func

    return decorator


def registered_rule_ids() -> list[str]:
    return sorted(_RULES)


# ---------------------------------------------------------------------------
# rule-set loading
# ---------------------------------------------------------------------------
def load_rule_set(path: str | Path) -> RuleSet:
    target = Path(path)
    raw = load_yaml(target)
    rules_raw = raw.get("rules") or {}
    if not isinstance(rules_raw, dict):
        raise ConfigError("Rule file 'rules' must be a mapping", path=str(target))

    unknown = sorted(set(rules_raw) - set(_RULES))
    if unknown:
        raise ConfigError(
            "Rule file references rules that are not implemented",
            unknown=unknown,
            implemented=registered_rule_ids(),
            path=str(target),
        )

    rules: dict[str, RuleConfig] = {}
    for rule_id, entry in rules_raw.items():
        if not isinstance(entry, dict):
            raise ConfigError("Each rule entry must be a mapping", rule_id=rule_id)
        severity_raw = str(entry.get("severity", "warning"))
        try:
            severity = RuleSeverity(severity_raw)
        except ValueError as exc:
            raise ConfigError(
                "Unknown rule severity",
                rule_id=rule_id,
                severity=severity_raw,
                allowed=[s.value for s in RuleSeverity],
            ) from exc
        rules[rule_id] = RuleConfig(
            rule_id=rule_id,
            enabled=bool(entry.get("enabled", True)),
            severity=severity,
            weight=float(entry.get("weight", 1.0)),
            params=dict(entry.get("params", {}) or {}),
        )

    confidence = raw.get("confidence", {}) or {}
    return RuleSet(
        rules_version=str(raw.get("rules_version", "unversioned")),
        description=str(raw.get("description", "")),
        category_matrix={
            str(key): [str(v) for v in value]
            for key, value in (raw.get("category_matrix") or {}).items()
        },
        rules=rules,
        penalties={
            "warn": float((confidence.get("penalties") or {}).get("warn", 0.35)),
            "needs_input": float((confidence.get("penalties") or {}).get("needs_input", 0.7)),
            "fail": float((confidence.get("penalties") or {}).get("fail", 1.0)),
        },
        confidence_floor=float(confidence.get("floor", 0.0)),
        source_path=target,
    )


def load_rule_set_for(config: AppConfig) -> RuleSet:
    return load_rule_set(config.config_dir() / config.compatibility.rules_file)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _passed(rule_id: str, message: str, config: RuleConfig, **evidence: Any) -> RuleResult:
    return RuleResult(
        rule_id=rule_id,
        outcome=RuleOutcome.PASS,
        message=message,
        evidence=evidence,
        weight=config.weight,
    )


def _triggered(
    rule_id: str,
    message: str,
    config: RuleConfig,
    *,
    outcome: RuleOutcome | None = None,
    remediation: str | None = None,
    required_views: list[ImageViewType] | None = None,
    required_mask_expansions: list[MaskKind] | None = None,
    **evidence: Any,
) -> RuleResult:
    return RuleResult(
        rule_id=rule_id,
        outcome=outcome or config.outcome(),
        message=message,
        evidence=evidence,
        remediation=remediation,
        required_views=required_views or [],
        required_mask_expansions=required_mask_expansions or [],
        weight=config.weight,
    )


def _coverage_rank(value: BodyCoverage) -> int:
    return scale_index(COVERAGE_ORDER, value)


# ---------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------
@rule("category_vs_template_class")
def _category_vs_template_class(
    template: HumanTemplate,
    garment: GarmentAsset,
    config: RuleConfig,
    context: EvaluationContext,
    rule_set: RuleSet,
) -> RuleResult:
    allowed = rule_set.category_matrix.get(template.template_clothing_class.value)
    if allowed is None:
        outcome = (
            RuleOutcome.FAIL
            if config.params.get("unlisted_is_blocking", True)
            else RuleOutcome.WARN
        )
        return _triggered(
            config.rule_id,
            f"No category matrix entry for template class "
            f"{template.template_clothing_class.value!r}",
            config,
            outcome=outcome,
            remediation="Add the template class to category_matrix in the rule file.",
            template_class=template.template_clothing_class.value,
        )
    if garment.category.value in allowed:
        return _passed(
            config.rule_id,
            f"Category {garment.category.value!r} is allowed over template class "
            f"{template.template_clothing_class.value!r}",
            config,
            allowed=allowed,
        )
    return _triggered(
        config.rule_id,
        f"Garment category {garment.category.value!r} cannot be placed over a "
        f"{template.template_clothing_class.value!r} base performance",
        config,
        remediation=(
            "Use a template whose base outfit this category can cover, or choose "
            "a garment category from: " + ", ".join(allowed)
        ),
        garment_category=garment.category.value,
        template_class=template.template_clothing_class.value,
        allowed_categories=allowed,
    )


@rule("required_body_coverage")
def _required_body_coverage(
    template: HumanTemplate,
    garment: GarmentAsset,
    config: RuleConfig,
    context: EvaluationContext,
    rule_set: RuleSet,
) -> RuleResult:
    problems: list[str] = []
    garment_rank = _coverage_rank(garment.body_coverage)

    if config.params.get("enforce_category_minimum", True):
        minimum = CATEGORY_MIN_COVERAGE.get(garment.category)
        if minimum is not None and garment_rank < _coverage_rank(minimum):
            problems.append(
                f"category {garment.category.value!r} implies at least "
                f"{minimum.value!r} coverage, garment declares "
                f"{garment.body_coverage.value!r}"
            )

    base_coverage = TEMPLATE_CLASS_COVERAGE[template.template_clothing_class]
    if config.params.get("enforce_template_minimum", True) and garment_rank < _coverage_rank(
        base_coverage
    ):
        problems.append(
            f"the base outfit covers {base_coverage.value!r} but the garment only "
            f"covers {garment.body_coverage.value!r}; the original garment would "
            "remain visible"
        )

    if not problems:
        return _passed(
            config.rule_id,
            f"Coverage {garment.body_coverage.value!r} is sufficient",
            config,
            base_coverage=base_coverage.value,
        )
    return _triggered(
        config.rule_id,
        "Insufficient body coverage: " + "; ".join(problems),
        config,
        remediation=(
            "Pick a garment with greater coverage, or re-shoot the master "
            "performance with a more minimal base outfit."
        ),
        garment_coverage=garment.body_coverage.value,
        base_coverage=base_coverage.value,
        problems=problems,
    )


@rule("sleeve_mismatch")
def _sleeve_mismatch(
    template: HumanTemplate,
    garment: GarmentAsset,
    config: RuleConfig,
    context: EvaluationContext,
    rule_set: RuleSet,
) -> RuleResult:
    if garment.sleeve_length is SleeveLength.NOT_APPLICABLE:
        return RuleResult(
            rule_id=config.rule_id,
            outcome=RuleOutcome.SKIPPED,
            message="Garment has no sleeve attribute",
            weight=config.weight,
        )
    base_sleeve = _base_sleeve(template.template_clothing_class)
    garment_rank = scale_index(SLEEVE_ORDER, garment.sleeve_length)
    base_rank = scale_index(SLEEVE_ORDER, base_sleeve)
    delta = garment_rank - base_rank

    if delta < 0 and config.params.get("shorter_than_base_is_blocking", True):
        return _triggered(
            config.rule_id,
            f"Garment sleeves ({garment.sleeve_length.value}) are shorter than the "
            f"base outfit's ({base_sleeve.value}); the original sleeve would show",
            config,
            outcome=RuleOutcome.FAIL,
            remediation="Choose a garment with sleeves at least as long as the base outfit.",
            garment_sleeve=garment.sleeve_length.value,
            base_sleeve=base_sleeve.value,
        )
    allowance = int(config.params.get("max_steps_longer_without_expansion", 1))
    if delta > allowance and not context.has_expansion_masks:
        return _triggered(
            config.rule_id,
            f"Garment sleeves are {delta} steps longer than the base "
            f"({base_sleeve.value} -> {garment.sleeve_length.value}); the arm "
            "region needs an expansion mask",
            config,
            remediation="Import an expansion mask covering the arms, then re-check.",
            required_mask_expansions=[MaskKind.EXPANSION],
            garment_sleeve=garment.sleeve_length.value,
            base_sleeve=base_sleeve.value,
            steps=delta,
            allowance=allowance,
        )
    return _passed(
        config.rule_id,
        f"Sleeve length {garment.sleeve_length.value!r} is compatible",
        config,
        base_sleeve=base_sleeve.value,
        steps=delta,
    )


def _base_sleeve(clothing_class: TemplateClothingClass) -> SleeveLength:
    return {
        TemplateClothingClass.SLEEVELESS_MINIMAL: SleeveLength.NONE,
        TemplateClothingClass.FITTED_SHORT: SleeveLength.SHORT,
        TemplateClothingClass.FITTED_FULL: SleeveLength.LONG,
        TemplateClothingClass.LOOSE_SHORT: SleeveLength.SHORT,
        TemplateClothingClass.LOOSE_FULL: SleeveLength.LONG,
        TemplateClothingClass.TWO_PIECE: SleeveLength.STRAP,
        TemplateClothingClass.FULL_COVERAGE: SleeveLength.LONG,
        TemplateClothingClass.NEUTRAL_BODYSUIT: SleeveLength.NONE,
    }[clothing_class]


def _base_length(clothing_class: TemplateClothingClass) -> GarmentLength:
    return {
        TemplateClothingClass.SLEEVELESS_MINIMAL: GarmentLength.CROP,
        TemplateClothingClass.FITTED_SHORT: GarmentLength.MID_THIGH,
        TemplateClothingClass.FITTED_FULL: GarmentLength.ANKLE,
        TemplateClothingClass.LOOSE_SHORT: GarmentLength.MID_THIGH,
        TemplateClothingClass.LOOSE_FULL: GarmentLength.ANKLE,
        TemplateClothingClass.TWO_PIECE: GarmentLength.WAIST,
        TemplateClothingClass.FULL_COVERAGE: GarmentLength.ANKLE,
        TemplateClothingClass.NEUTRAL_BODYSUIT: GarmentLength.HIP,
    }[clothing_class]


@rule("hem_length_mismatch")
def _hem_length_mismatch(
    template: HumanTemplate,
    garment: GarmentAsset,
    config: RuleConfig,
    context: EvaluationContext,
    rule_set: RuleSet,
) -> RuleResult:
    if garment.garment_length is GarmentLength.NOT_APPLICABLE:
        return RuleResult(
            rule_id=config.rule_id,
            outcome=RuleOutcome.SKIPPED,
            message="Garment has no hem-length attribute",
            weight=config.weight,
        )
    base_length = _base_length(template.template_clothing_class)
    delta = scale_index(LENGTH_ORDER, garment.garment_length) - scale_index(
        LENGTH_ORDER, base_length
    )

    if delta < 0 and config.params.get("shorter_than_base_is_blocking", True):
        return _triggered(
            config.rule_id,
            f"Garment hem ({garment.garment_length.value}) is shorter than the base "
            f"outfit's ({base_length.value}); the original hem would show",
            config,
            outcome=RuleOutcome.FAIL,
            remediation="Choose a garment at least as long as the base outfit.",
            garment_length=garment.garment_length.value,
            base_length=base_length.value,
        )
    allowance = int(config.params.get("max_steps_longer_without_expansion", 1))
    if delta > allowance and not context.has_expansion_masks:
        return _triggered(
            config.rule_id,
            f"Garment hem is {delta} steps longer than the base "
            f"({base_length.value} -> {garment.garment_length.value}); the leg "
            "region needs an expansion mask",
            config,
            remediation="Import an expansion mask covering the legs, then re-check.",
            required_mask_expansions=[MaskKind.EXPANSION],
            garment_length=garment.garment_length.value,
            base_length=base_length.value,
            steps=delta,
            allowance=allowance,
        )
    return _passed(
        config.rule_id,
        f"Hem length {garment.garment_length.value!r} is compatible",
        config,
        base_length=base_length.value,
        steps=delta,
    )


@rule("silhouette_expansion")
def _silhouette_expansion(
    template: HumanTemplate,
    garment: GarmentAsset,
    config: RuleConfig,
    context: EvaluationContext,
    rule_set: RuleSet,
) -> RuleResult:
    base_silhouette = (
        Silhouette.OVERSIZED
        if template.template_clothing_class
        in {TemplateClothingClass.LOOSE_SHORT, TemplateClothingClass.LOOSE_FULL}
        else Silhouette.FITTED
    )
    delta = scale_index(SILHOUETTE_ORDER, garment.silhouette) - scale_index(
        SILHOUETTE_ORDER, base_silhouette
    )
    blocking_steps = int(config.params.get("blocking_steps", 4))
    allowance = int(config.params.get("max_steps_without_expansion", 1))

    if delta >= blocking_steps:
        return _triggered(
            config.rule_id,
            f"Silhouette {garment.silhouette.value!r} is {delta} steps larger than "
            f"the base {base_silhouette.value!r}; a fixed performance cannot "
            "support that much added volume",
            config,
            outcome=RuleOutcome.FAIL,
            remediation=(
                "Re-shoot the master performance wearing a garment of similar "
                "volume, or choose a closer-fitting garment."
            ),
            garment_silhouette=garment.silhouette.value,
            base_silhouette=base_silhouette.value,
            steps=delta,
        )
    if delta > allowance and not context.has_expansion_masks:
        return _triggered(
            config.rule_id,
            f"Silhouette {garment.silhouette.value!r} is {delta} steps larger than "
            f"the base; an expansion mask is required",
            config,
            remediation="Import an expansion mask covering the added volume.",
            required_mask_expansions=[MaskKind.EXPANSION],
            garment_silhouette=garment.silhouette.value,
            base_silhouette=base_silhouette.value,
            steps=delta,
            allowance=allowance,
        )
    return _passed(
        config.rule_id,
        f"Silhouette {garment.silhouette.value!r} needs no extra expansion",
        config,
        steps=delta,
    )


@rule("transparent_garment_requirements")
def _transparent_garment_requirements(
    template: HumanTemplate,
    garment: GarmentAsset,
    config: RuleConfig,
    context: EvaluationContext,
    rule_set: RuleSet,
) -> RuleResult:
    threshold = float(config.params.get("transparency_threshold", 0.15))
    if garment.transparency <= threshold:
        return _passed(
            config.rule_id,
            "Garment is opaque enough to hide the base outfit",
            config,
            transparency=garment.transparency,
        )
    acceptable = {
        str(value) for value in config.params.get("acceptable_template_classes", []) or []
    }
    if template.template_clothing_class.value in acceptable:
        return _passed(
            config.rule_id,
            "Transparent garment is acceptable over this template's neutral base",
            config,
            transparency=garment.transparency,
            template_class=template.template_clothing_class.value,
        )
    if config.params.get("require_underlayer_declaration", True) and garment.requires_underlayer:
        return _triggered(
            config.rule_id,
            f"Transparent garment (transparency={garment.transparency:.2f}) declares "
            "an underlayer, which this pipeline cannot synthesise on a fixed "
            "performance",
            config,
            remediation=(
                "Use a template whose base outfit can serve as the underlayer "
                "(neutral_bodysuit or sleeveless_minimal)."
            ),
            transparency=garment.transparency,
            template_class=template.template_clothing_class.value,
        )
    return _triggered(
        config.rule_id,
        f"Transparent garment (transparency={garment.transparency:.2f}) over a "
        f"{template.template_clothing_class.value!r} base: the original outfit "
        "would show through",
        config,
        remediation=(
            "Use a neutral-base template, or pick an opaque garment "
            f"(transparency <= {threshold})."
        ),
        transparency=garment.transparency,
        threshold=threshold,
        acceptable_template_classes=sorted(acceptable),
    )


@rule("reflective_material_warning")
def _reflective_material_warning(
    template: HumanTemplate,
    garment: GarmentAsset,
    config: RuleConfig,
    context: EvaluationContext,
    rule_set: RuleSet,
) -> RuleResult:
    threshold = float(config.params.get("reflectivity_threshold", 0.5))
    always = {str(v) for v in config.params.get("always_flag_materials", []) or []}
    flagged = (
        garment.reflectivity >= threshold
        or garment.material.value in always
        or garment.material in REFLECTIVE_MATERIALS
    )
    if not flagged:
        return _passed(
            config.rule_id,
            "Material is not strongly reflective",
            config,
            reflectivity=garment.reflectivity,
        )
    return _triggered(
        config.rule_id,
        f"Reflective material ({garment.material.value}, "
        f"reflectivity={garment.reflectivity:.2f}): specular highlights will not "
        "match the master lighting and may look pasted on",
        config,
        remediation=(
            "Review the reveal segment closely; consider a matte alternative or "
            "a lighting-aware workflow."
        ),
        material=garment.material.value,
        reflectivity=garment.reflectivity,
    )


@rule("high_flow_fabric_warning")
def _high_flow_fabric_warning(
    template: HumanTemplate,
    garment: GarmentAsset,
    config: RuleConfig,
    context: EvaluationContext,
    rule_set: RuleSet,
) -> RuleResult:
    threshold = float(config.params.get("fabric_flow_threshold", 0.6))
    blocking = float(config.params.get("flow_blocking_threshold", 0.95))
    if garment.fabric_flow >= blocking:
        return _triggered(
            config.rule_id,
            f"Fabric flow {garment.fabric_flow:.2f} is extreme; free-swinging "
            "fabric cannot be derived from a fixed performance",
            config,
            outcome=RuleOutcome.FAIL,
            remediation="Choose a structured garment, or shoot a dedicated performance.",
            fabric_flow=garment.fabric_flow,
        )
    if garment.fabric_flow < threshold:
        return _passed(
            config.rule_id,
            "Fabric flow is within the range a fixed performance can carry",
            config,
            fabric_flow=garment.fabric_flow,
        )
    return _triggered(
        config.rule_id,
        f"High fabric flow ({garment.fabric_flow:.2f}): the garment will follow the "
        "body rather than swing independently",
        config,
        remediation="Expect a slightly pinned look; review the reveal segment.",
        fabric_flow=garment.fabric_flow,
        threshold=threshold,
    )


@rule("missing_views_for_motion")
def _missing_views_for_motion(
    template: HumanTemplate,
    garment: GarmentAsset,
    config: RuleConfig,
    context: EvaluationContext,
    rule_set: RuleSet,
) -> RuleResult:
    always = [ImageViewType(v) for v in config.params.get("always_required", ["front"])]
    if context.exposed_views is None:
        assumed = [ImageViewType(v) for v in config.params.get("assume_when_unknown", []) or []]
        required = {*always, *assumed}
        assumption = True
    else:
        required = {*always, *context.exposed_views}
        assumption = False

    missing = sorted(required - garment.available_views, key=lambda view: view.value)
    if not missing:
        return _passed(
            config.rule_id,
            "All garment views the motion exposes are present",
            config,
            required=[v.value for v in sorted(required, key=lambda x: x.value)],
        )
    return _triggered(
        config.rule_id,
        "Missing garment reference views: "
        + ", ".join(v.value for v in missing)
        + (" (motion coverage unknown, assuming the performance turns)" if assumption else ""),
        config,
        remediation="Add the missing reference images with `app garment ingest --image`.",
        required_views=missing,
        missing_views=[v.value for v in missing],
        available_views=sorted(v.value for v in garment.available_views),
        assumption_used=assumption,
    )


@rule("hair_hand_occlusion_risk")
def _hair_hand_occlusion_risk(
    template: HumanTemplate,
    garment: GarmentAsset,
    config: RuleConfig,
    context: EvaluationContext,
    rule_set: RuleSet,
) -> RuleResult:
    if context.has_occlusion_masks:
        return _passed(
            config.rule_id,
            "Occlusion masks are present for hair/hand overlap",
            config,
        )
    if not config.params.get("require_occlusion_mask", True):
        return _passed(config.rule_id, "Occlusion masks are not required by policy", config)
    escalate = bool(config.params.get("escalate_to_needs_input_without_occlusion_mask", True))
    return _triggered(
        config.rule_id,
        "No occlusion masks found: hair strands and hands crossing the garment "
        "region will be painted over",
        config,
        outcome=RuleOutcome.NEEDS_INPUT if escalate else RuleOutcome.WARN,
        remediation=(
            "Author occlusion masks for frames where hair or hands cross the "
            "garment and import them with `app template import-masks`."
        ),
        required_mask_expansions=[MaskKind.OCCLUSION],
        has_occlusion_masks=False,
    )


@rule("base_garment_leakage_risk")
def _base_garment_leakage_risk(
    template: HumanTemplate,
    garment: GarmentAsset,
    config: RuleConfig,
    context: EvaluationContext,
    rule_set: RuleSet,
) -> RuleResult:
    base_coverage = TEMPLATE_CLASS_COVERAGE[template.template_clothing_class]
    problems: list[str] = []
    if config.params.get("require_coverage_at_least_base", True) and _coverage_rank(
        garment.body_coverage
    ) < _coverage_rank(base_coverage):
        problems.append(
            f"garment coverage {garment.body_coverage.value!r} is below the base "
            f"{base_coverage.value!r}"
        )
    if (
        config.params.get("dark_base_light_garment_warning", True)
        and context.base_garment_luminance is not None
        and context.base_garment_luminance < 0.25
        and _mean_luminance(garment) > 0.7
    ):
        problems.append(
            "a light garment over a very dark base outfit tends to bleed through "
            "at the mask boundary"
        )
    if not problems:
        return _passed(config.rule_id, "Base garment leakage risk is low", config)
    return _triggered(
        config.rule_id,
        "Base garment leakage risk: " + "; ".join(problems),
        config,
        remediation=(
            "Increase the garment mask's expansion at the boundary, or choose a "
            "garment with greater coverage."
        ),
        required_mask_expansions=[MaskKind.EXPANSION],
        problems=problems,
        base_coverage=base_coverage.value,
    )


def _mean_luminance(garment: GarmentAsset) -> float:
    """Rough perceived luminance of the declared dominant colours."""
    values: list[float] = []
    for color in garment.dominant_colors:
        text = color.lstrip("#")
        if len(text) == 3:
            text = "".join(ch * 2 for ch in text)
        if len(text) != 6:
            continue
        try:
            r, g, b = (int(text[i : i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            continue
        values.append((0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0)
    return sum(values) / len(values) if values else 0.5


@rule("source_image_quality")
def _source_image_quality(
    template: HumanTemplate,
    garment: GarmentAsset,
    config: RuleConfig,
    context: EvaluationContext,
    rule_set: RuleSet,
) -> RuleResult:
    if not garment.images:
        return _triggered(
            config.rule_id,
            "Garment has no reference images",
            config,
            outcome=RuleOutcome.FAIL,
            remediation="Ingest at least one front-view reference image.",
            required_views=[ImageViewType.FRONT],
        )
    min_side = int(config.params.get("min_image_min_side_px", 768))
    min_mp = float(config.params.get("min_megapixels", 0.5))
    min_sharpness = float(config.params.get("min_sharpness_score", 0.15))
    max_aspect = float(config.params.get("max_aspect_ratio", 3.0))

    problems: list[dict[str, Any]] = []
    for image in garment.images:
        issues: list[str] = []
        if image.min_side < min_side:
            issues.append(f"min side {image.min_side}px < {min_side}px")
        if image.megapixels < min_mp:
            issues.append(f"{image.megapixels:.2f}MP < {min_mp}MP")
        if image.sharpness_score is not None and image.sharpness_score < min_sharpness:
            issues.append(f"sharpness {image.sharpness_score:.3f} < {min_sharpness}")
        ratio = max(image.aspect_ratio, 1 / image.aspect_ratio)
        if ratio > max_aspect:
            issues.append(f"aspect ratio {ratio:.2f} > {max_aspect}")
        if issues:
            problems.append({"path": image.path, "view": image.view.value, "issues": issues})

    if not problems:
        return _passed(
            config.rule_id,
            f"All {len(garment.images)} reference images meet the quality bar",
            config,
        )
    return _triggered(
        config.rule_id,
        f"{len(problems)} of {len(garment.images)} reference images fail the quality bar",
        config,
        remediation="Replace the flagged images with higher-resolution, sharper shots.",
        problems=problems,
    )


@rule("proportion_mismatch")
def _proportion_mismatch(
    template: HumanTemplate,
    garment: GarmentAsset,
    config: RuleConfig,
    context: EvaluationContext,
    rule_set: RuleSet,
) -> RuleResult:
    front = garment.images_for(ImageViewType.FRONT)
    if not front:
        return RuleResult(
            rule_id=config.rule_id,
            outcome=RuleOutcome.SKIPPED,
            message="No front reference image to compare proportions against",
            weight=config.weight,
        )
    body_ratio = context.body_region_aspect_ratio
    if body_ratio is None:
        body_ratio = template.video.width / template.video.height
    image_ratio = front[0].aspect_ratio
    delta = max(image_ratio / body_ratio, body_ratio / image_ratio)

    blocking = float(config.params.get("blocking_aspect_ratio_delta", 3.5))
    warn_at = float(config.params.get("max_aspect_ratio_delta", 1.8))
    if delta >= blocking:
        return _triggered(
            config.rule_id,
            f"Severe proportion mismatch: reference aspect ratio differs from the "
            f"body region by {delta:.2f}x",
            config,
            outcome=RuleOutcome.FAIL,
            remediation="Use a product photo framed on a body, not an extreme flat-lay crop.",
            image_aspect_ratio=round(image_ratio, 4),
            body_aspect_ratio=round(body_ratio, 4),
            delta=round(delta, 4),
        )
    if delta < warn_at:
        return _passed(
            config.rule_id,
            "Reference proportions are close enough to the body region",
            config,
            delta=round(delta, 4),
        )
    return _triggered(
        config.rule_id,
        f"Proportion mismatch: reference aspect ratio differs from the body region "
        f"by {delta:.2f}x",
        config,
        remediation="Prefer an on-body product photo framed similarly to the performance.",
        image_aspect_ratio=round(image_ratio, 4),
        body_aspect_ratio=round(body_ratio, 4),
        delta=round(delta, 4),
    )


@rule("incomplete_product_information")
def _incomplete_product_information(
    template: HumanTemplate,
    garment: GarmentAsset,
    config: RuleConfig,
    context: EvaluationContext,
    rule_set: RuleSet,
) -> RuleResult:
    minimum = float(config.params.get("min_metadata_completeness", 0.6))
    missing: list[str] = []
    completeness = garment.metadata_completeness
    if completeness < minimum:
        missing.append(f"metadata completeness {completeness:.2f} < {minimum}")
    if config.params.get("require_usage_rights", True) and not garment.usage_rights.is_documented:
        missing.append("usage rights are undocumented (license, rights holder or source)")
    if config.params.get("require_pattern_description", False) and not garment.pattern_description:
        missing.append("pattern description is empty")

    if not missing:
        return _passed(
            config.rule_id,
            "Product information is sufficiently complete",
            config,
            completeness=round(completeness, 4),
        )
    return _triggered(
        config.rule_id,
        "Incomplete product information: " + "; ".join(missing),
        config,
        remediation="Fill in the garment metadata, then re-run the compatibility check.",
        missing=missing,
        completeness=round(completeness, 4),
    )


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------
def evaluate(
    template: HumanTemplate,
    garment: GarmentAsset,
    rule_set: RuleSet,
    *,
    context: EvaluationContext | None = None,
    report_id: str | None = None,
) -> CompatibilityReport:
    """Run every enabled rule and aggregate the result."""
    evaluation_context = context or EvaluationContext()
    results: list[RuleResult] = []

    for rule_id in sorted(rule_set.rules):
        config = rule_set.rules[rule_id]
        if not config.enabled:
            continue
        func = _RULES[rule_id]
        try:
            results.append(func(template, garment, config, evaluation_context, rule_set))
        except Exception as exc:  # a broken rule must not silently pass a garment
            logger.exception("compatibility_rule_error", extra={"rule_id": rule_id})
            results.append(
                RuleResult(
                    rule_id=rule_id,
                    outcome=RuleOutcome.FAIL,
                    message=f"Rule raised an unexpected error: {exc}",
                    evidence={"error": str(exc), "error_type": type(exc).__name__},
                    weight=config.weight,
                )
            )

    outcomes = {result.outcome for result in results}
    if RuleOutcome.FAIL in outcomes:
        state = CompatibilityState.INCOMPATIBLE
    elif RuleOutcome.NEEDS_INPUT in outcomes:
        state = CompatibilityState.NEEDS_INPUT
    else:
        state = CompatibilityState.READY

    blocking = [r.message for r in results if r.outcome is RuleOutcome.FAIL]
    needs_input = [r.message for r in results if r.outcome is RuleOutcome.NEEDS_INPUT]
    warnings = [r.message for r in results if r.outcome is RuleOutcome.WARN]

    missing_views: list[ImageViewType] = []
    expansions: list[MaskKind] = []
    for result in results:
        if not result.blocks_render:
            continue
        for view in result.required_views:
            if view not in missing_views:
                missing_views.append(view)
        for kind in result.required_mask_expansions:
            if kind not in expansions:
                expansions.append(kind)

    return CompatibilityReport(
        id=report_id or new_report_id(),
        template_id=template.id,
        template_version=template.version,
        garment_id=garment.id,
        garment_version=garment.version,
        rules_version=rule_set.rules_version,
        rules_file_sha256=rule_set.sha256(),
        state=state,
        results=results,
        blocking_reasons=[*blocking, *needs_input],
        warnings=warnings,
        required_missing_views=missing_views,
        required_mask_expansions=expansions,
        recommended_template_class=_recommend_template_class(garment),
        confidence=_confidence(results, rule_set),
    )


def _confidence(results: list[RuleResult], rule_set: RuleSet) -> float:
    total_weight = sum(r.weight for r in results) or 1.0
    penalty = 0.0
    for result in results:
        if result.outcome is RuleOutcome.WARN:
            penalty += rule_set.penalties["warn"] * result.weight
        elif result.outcome is RuleOutcome.NEEDS_INPUT:
            penalty += rule_set.penalties["needs_input"] * result.weight
        elif result.outcome is RuleOutcome.FAIL:
            penalty += rule_set.penalties["fail"] * result.weight
    value = 1.0 - (penalty / total_weight)
    return max(rule_set.confidence_floor, min(1.0, round(value, 4)))


def _recommend_template_class(garment: GarmentAsset) -> TemplateClothingClass:
    """The base outfit that would make this garment easiest to place."""
    if garment.transparency > 0.15:
        return TemplateClothingClass.NEUTRAL_BODYSUIT
    if garment.category in {GarmentCategory.BOTTOM}:
        return TemplateClothingClass.TWO_PIECE
    if garment.body_coverage in {BodyCoverage.FULL_BODY, BodyCoverage.FULL_BODY_LIMBS}:
        return TemplateClothingClass.NEUTRAL_BODYSUIT
    return TemplateClothingClass.SLEEVELESS_MINIMAL


__all__ = [
    "EvaluationContext",
    "RuleConfig",
    "RuleSet",
    "evaluate",
    "load_rule_set",
    "load_rule_set_for",
    "registered_rule_ids",
    "rule",
]
