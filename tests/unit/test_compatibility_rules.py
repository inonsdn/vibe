"""Compatibility engine: requirements 6 and 7.

6. Rules correctly return READY, NEEDS_INPUT and INCOMPATIBLE.
7. Bad or missing garment views block rendering.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import REPO_ROOT
from app.core.errors import ConfigError
from app.domain.compatibility import CompatibilityState, RuleOutcome
from app.domain.enums import (
    BodyCoverage,
    GarmentCategory,
    GarmentLength,
    ImageViewType,
    MaskKind,
    Material,
    Silhouette,
    SleeveLength,
    TemplateClothingClass,
)
from app.domain.garment import GarmentAsset, GarmentReferenceImage, UsageRights
from app.domain.human_template import (
    ConsentRecord,
    FrameRange,
    HumanTemplate,
    TemplateDirectories,
    VideoSpec,
)
from app.pipeline.compatibility import (
    EvaluationContext,
    evaluate,
    load_rule_set,
    registered_rule_ids,
)

RULES_FILE = REPO_ROOT / "config" / "compatibility_rules.v1.yaml"


@pytest.fixture
def rule_set():
    return load_rule_set(RULES_FILE)


def template(
    clothing_class: TemplateClothingClass = TemplateClothingClass.SLEEVELESS_MINIMAL,
) -> HumanTemplate:
    return HumanTemplate(
        id="tpl_1",
        display_name="T",
        source_video_path="a.mp4",
        source_sha256="0" * 64,
        video=VideoSpec(width=1080, height=1920, fps=30.0, duration_s=2.0, frame_count=60),
        intro=FrameRange(start=0, end=30),
        reveal=FrameRange(start=30, end=60),
        transition_anchor_frame=30,
        consent=ConsentRecord(subject_kind="synthetic", adult_confirmed=True),
        template_clothing_class=clothing_class,
        directories=TemplateDirectories.standard("templates/tpl_1"),
        extracted_frame_count=60,
    )


def image(view: ImageViewType, *, width: int = 1200, height: int = 1600, sharpness: float = 0.8):
    return GarmentReferenceImage(
        path=f"garments/g/{view.value}.png",
        view=view,
        sha256="a" * 64,
        width=width,
        height=height,
        sharpness_score=sharpness,
    )


def garment(**overrides) -> GarmentAsset:
    payload = {
        "id": "grm_1",
        "product_name": "Test Top",
        "brand": "Brand",
        "pattern_description": "solid",
        "dominant_colors": ["#224466"],
        "usage_rights": UsageRights(license="test", rights_holder="Fixture"),
        "images": [
            image(ImageViewType.FRONT),
            image(ImageViewType.BACK),
            image(ImageViewType.SIDE),
        ],
        "category": GarmentCategory.TOP,
        "sleeve_length": SleeveLength.SHORT,
        "garment_length": GarmentLength.HIP,
        "body_coverage": BodyCoverage.TORSO,
        "silhouette": Silhouette.FITTED,
        "material": Material.COTTON,
    }
    payload.update(overrides)
    return GarmentAsset(**payload)


def good_context() -> EvaluationContext:
    return EvaluationContext(
        exposed_views={ImageViewType.FRONT, ImageViewType.BACK, ImageViewType.SIDE},
        has_occlusion_masks=True,
        has_expansion_masks=True,
        base_garment_luminance=0.4,
        body_region_aspect_ratio=0.6,
    )


def outcome_of(report, rule_id: str) -> RuleOutcome:
    result = next(r for r in report.results if r.rule_id == rule_id)
    return result.outcome


# -- READY -----------------------------------------------------------------
def test_compatible_pair_is_ready(rule_set) -> None:
    report = evaluate(template(), garment(), rule_set, context=good_context())
    assert report.state is CompatibilityState.READY, report.blocking_reasons
    assert report.confidence == pytest.approx(1.0)
    assert report.render_allowed(__import__("app.core.ids", fromlist=["utc_now"]).utc_now())
    assert not report.blocking_reasons


def test_every_configured_rule_is_implemented(rule_set) -> None:
    assert set(rule_set.rules) <= set(registered_rule_ids())
    assert len(rule_set.rules) == 14


# -- INCOMPATIBLE ----------------------------------------------------------
def test_category_not_allowed_over_template_class_is_incompatible(rule_set) -> None:
    report = evaluate(
        template(TemplateClothingClass.FULL_COVERAGE),
        garment(category=GarmentCategory.TOP, body_coverage=BodyCoverage.FULL_BODY),
        rule_set,
        context=good_context(),
    )
    assert report.state is CompatibilityState.INCOMPATIBLE
    assert outcome_of(report, "category_vs_template_class") is RuleOutcome.FAIL


def test_insufficient_coverage_is_incompatible(rule_set) -> None:
    report = evaluate(
        template(TemplateClothingClass.FITTED_FULL),
        garment(body_coverage=BodyCoverage.MINIMAL, category=GarmentCategory.TOP),
        rule_set,
        context=good_context(),
    )
    assert report.state is CompatibilityState.INCOMPATIBLE
    assert outcome_of(report, "required_body_coverage") is RuleOutcome.FAIL


def test_shorter_sleeves_than_the_base_are_incompatible(rule_set) -> None:
    report = evaluate(
        template(TemplateClothingClass.FITTED_FULL),
        garment(
            sleeve_length=SleeveLength.SHORT,
            body_coverage=BodyCoverage.TORSO_LEGS,
            garment_length=GarmentLength.ANKLE,
        ),
        rule_set,
        context=good_context(),
    )
    assert outcome_of(report, "sleeve_mismatch") is RuleOutcome.FAIL
    assert report.state is CompatibilityState.INCOMPATIBLE


def test_shorter_hem_than_the_base_is_incompatible(rule_set) -> None:
    report = evaluate(
        template(TemplateClothingClass.FITTED_FULL),
        garment(
            garment_length=GarmentLength.CROP,
            sleeve_length=SleeveLength.LONG,
            body_coverage=BodyCoverage.TORSO_LEGS,
        ),
        rule_set,
        context=good_context(),
    )
    assert outcome_of(report, "hem_length_mismatch") is RuleOutcome.FAIL


def test_extreme_silhouette_expansion_is_incompatible(rule_set) -> None:
    report = evaluate(
        template(),
        garment(silhouette=Silhouette.VOLUMINOUS),
        rule_set,
        context=good_context(),
    )
    assert outcome_of(report, "silhouette_expansion") is RuleOutcome.FAIL


def test_extreme_fabric_flow_is_incompatible(rule_set) -> None:
    report = evaluate(template(), garment(fabric_flow=1.0), rule_set, context=good_context())
    assert outcome_of(report, "high_flow_fabric_warning") is RuleOutcome.FAIL


def test_no_reference_images_is_incompatible(rule_set) -> None:
    report = evaluate(template(), garment(images=[]), rule_set, context=good_context())
    assert outcome_of(report, "source_image_quality") is RuleOutcome.FAIL
    assert report.state is CompatibilityState.INCOMPATIBLE
    assert ImageViewType.FRONT in report.required_missing_views


def test_severe_proportion_mismatch_is_incompatible(rule_set) -> None:
    report = evaluate(
        template(),
        garment(images=[image(ImageViewType.FRONT, width=4000, height=800)]),
        rule_set,
        context=good_context(),
    )
    assert outcome_of(report, "proportion_mismatch") is RuleOutcome.FAIL


# -- NEEDS_INPUT (requirement 7) ------------------------------------------
def test_missing_back_view_blocks_rendering(rule_set) -> None:
    report = evaluate(
        template(),
        garment(images=[image(ImageViewType.FRONT), image(ImageViewType.SIDE)]),
        rule_set,
        context=good_context(),
    )
    assert report.state is CompatibilityState.NEEDS_INPUT
    assert ImageViewType.BACK in report.required_missing_views
    assert not report.render_allowed(__import__("app.core.ids", fromlist=["utc_now"]).utc_now())


def test_missing_views_are_assumed_required_when_motion_is_unknown(rule_set) -> None:
    report = evaluate(
        template(),
        garment(images=[image(ImageViewType.FRONT)]),
        rule_set,
        context=EvaluationContext(
            exposed_views=None, has_occlusion_masks=True, has_expansion_masks=True
        ),
    )
    result = next(r for r in report.results if r.rule_id == "missing_views_for_motion")
    assert result.outcome is RuleOutcome.NEEDS_INPUT
    assert result.evidence["assumption_used"] is True
    assert set(result.evidence["missing_views"]) == {"back", "side"}


def test_front_only_garment_is_ready_when_motion_exposes_only_the_front(rule_set) -> None:
    report = evaluate(
        template(),
        garment(images=[image(ImageViewType.FRONT)]),
        rule_set,
        context=EvaluationContext(
            exposed_views={ImageViewType.FRONT},
            has_occlusion_masks=True,
            has_expansion_masks=True,
        ),
    )
    assert report.state is CompatibilityState.READY, report.blocking_reasons


def test_low_resolution_images_need_input(rule_set) -> None:
    report = evaluate(
        template(),
        garment(
            images=[
                image(ImageViewType.FRONT, width=320, height=420),
                image(ImageViewType.BACK),
                image(ImageViewType.SIDE),
            ]
        ),
        rule_set,
        context=good_context(),
    )
    assert outcome_of(report, "source_image_quality") is RuleOutcome.NEEDS_INPUT
    assert report.state is CompatibilityState.NEEDS_INPUT


def test_blurry_images_need_input(rule_set) -> None:
    report = evaluate(
        template(),
        garment(
            images=[
                image(ImageViewType.FRONT, sharpness=0.02),
                image(ImageViewType.BACK),
                image(ImageViewType.SIDE),
            ]
        ),
        rule_set,
        context=good_context(),
    )
    assert outcome_of(report, "source_image_quality") is RuleOutcome.NEEDS_INPUT


def test_transparent_garment_over_a_normal_base_needs_input(rule_set) -> None:
    report = evaluate(
        template(TemplateClothingClass.FITTED_SHORT),
        garment(material=Material.MESH, transparency=0.6),
        rule_set,
        context=good_context(),
    )
    assert outcome_of(report, "transparent_garment_requirements") is RuleOutcome.NEEDS_INPUT


def test_transparent_garment_over_a_neutral_base_is_acceptable(rule_set) -> None:
    report = evaluate(
        template(TemplateClothingClass.NEUTRAL_BODYSUIT),
        garment(
            material=Material.MESH,
            transparency=0.6,
            body_coverage=BodyCoverage.FULL_BODY,
            garment_length=GarmentLength.HIP,
            sleeve_length=SleeveLength.NONE,
        ),
        rule_set,
        context=good_context(),
    )
    assert outcome_of(report, "transparent_garment_requirements") is RuleOutcome.PASS


def test_missing_occlusion_masks_need_input(rule_set) -> None:
    report = evaluate(
        template(),
        garment(),
        rule_set,
        context=EvaluationContext(
            exposed_views={ImageViewType.FRONT, ImageViewType.BACK, ImageViewType.SIDE},
            has_occlusion_masks=False,
            has_expansion_masks=True,
        ),
    )
    result = next(r for r in report.results if r.rule_id == "hair_hand_occlusion_risk")
    assert result.outcome is RuleOutcome.NEEDS_INPUT
    assert MaskKind.OCCLUSION in report.required_mask_expansions


def test_long_sleeves_without_expansion_masks_need_input(rule_set) -> None:
    report = evaluate(
        template(TemplateClothingClass.SLEEVELESS_MINIMAL),
        garment(sleeve_length=SleeveLength.LONG),
        rule_set,
        context=EvaluationContext(
            exposed_views={ImageViewType.FRONT, ImageViewType.BACK, ImageViewType.SIDE},
            has_occlusion_masks=True,
            has_expansion_masks=False,
        ),
    )
    result = next(r for r in report.results if r.rule_id == "sleeve_mismatch")
    assert result.outcome is RuleOutcome.NEEDS_INPUT
    assert MaskKind.EXPANSION in result.required_mask_expansions


def test_incomplete_product_information_needs_input(rule_set) -> None:
    report = evaluate(
        template(),
        garment(
            product_name=None,
            brand=None,
            pattern_description=None,
            dominant_colors=[],
            usage_rights=UsageRights(),
        ),
        rule_set,
        context=good_context(),
    )
    assert outcome_of(report, "incomplete_product_information") is RuleOutcome.NEEDS_INPUT


# -- warnings (do not block) ----------------------------------------------
def test_reflective_material_only_warns(rule_set) -> None:
    report = evaluate(
        template(),
        garment(material=Material.SEQUIN, reflectivity=0.9),
        rule_set,
        context=good_context(),
    )
    assert outcome_of(report, "reflective_material_warning") is RuleOutcome.WARN
    assert report.state is CompatibilityState.READY
    assert report.warnings
    assert report.confidence < 1.0


def test_high_fabric_flow_only_warns(rule_set) -> None:
    report = evaluate(template(), garment(fabric_flow=0.7), rule_set, context=good_context())
    assert outcome_of(report, "high_flow_fabric_warning") is RuleOutcome.WARN
    assert report.state is CompatibilityState.READY


# -- configurability -------------------------------------------------------
def test_thresholds_are_configurable_without_code_changes(rule_set, tmp_path: Path) -> None:
    text = (
        RULES_FILE.read_text(encoding="utf-8")
        .replace("min_image_min_side_px: 768", "min_image_min_side_px: 64")
        .replace("min_megapixels: 0.5", "min_megapixels: 0.05")
    )
    relaxed_path = tmp_path / "relaxed.yaml"
    relaxed_path.write_text(text, encoding="utf-8")
    relaxed = load_rule_set(relaxed_path)

    small = garment(
        images=[
            image(ImageViewType.FRONT, width=320, height=420),
            image(ImageViewType.BACK, width=320, height=420),
            image(ImageViewType.SIDE, width=320, height=420),
        ]
    )
    strict_report = evaluate(template(), small, rule_set, context=good_context())
    relaxed_report = evaluate(template(), small, relaxed, context=good_context())
    assert outcome_of(strict_report, "source_image_quality") is RuleOutcome.NEEDS_INPUT
    assert outcome_of(relaxed_report, "source_image_quality") is RuleOutcome.PASS


def test_disabling_a_rule_removes_it(rule_set, tmp_path: Path) -> None:
    text = RULES_FILE.read_text(encoding="utf-8").replace(
        "  reflective_material_warning:\n    enabled: true",
        "  reflective_material_warning:\n    enabled: false",
    )
    path = tmp_path / "disabled.yaml"
    path.write_text(text, encoding="utf-8")
    modified = load_rule_set(path)
    report = evaluate(
        template(), garment(material=Material.SEQUIN), modified, context=good_context()
    )
    assert all(r.rule_id != "reflective_material_warning" for r in report.results)


def test_unknown_rule_in_the_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "rules_version: '9'\nrules:\n  not_a_real_rule:\n    enabled: true\n    severity: warning\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="not implemented"):
        load_rule_set(path)


def test_unknown_severity_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad2.yaml"
    path.write_text(
        "rules_version: '9'\nrules:\n  reflective_material_warning:\n"
        "    enabled: true\n    severity: catastrophic\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="severity"):
        load_rule_set(path)


def test_rules_file_hash_is_recorded(rule_set) -> None:
    report = evaluate(template(), garment(), rule_set, context=good_context())
    assert report.rules_file_sha256 and len(report.rules_file_sha256) == 64
    assert report.rules_version == "1"


def test_recommended_template_class_is_reported(rule_set) -> None:
    report = evaluate(
        template(TemplateClothingClass.FITTED_SHORT),
        garment(material=Material.MESH, transparency=0.6),
        rule_set,
        context=good_context(),
    )
    assert report.recommended_template_class is TemplateClothingClass.NEUTRAL_BODYSUIT
