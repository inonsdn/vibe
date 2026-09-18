"""Domain schema invariants, including the transition anchor (requirement 8)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from app.core.ids import utc_now
from app.domain.compatibility import (
    CompatibilityReport,
    CompatibilityState,
    ReviewerOverride,
    RuleOutcome,
    RuleResult,
)
from app.domain.enums import JobStatus, TemplateClothingClass
from app.domain.garment import GarmentAsset
from app.domain.human_template import (
    ConsentRecord,
    FrameRange,
    HumanTemplate,
    TemplateDirectories,
    VideoSpec,
)


def video(frame_count: int = 60) -> VideoSpec:
    return VideoSpec(
        width=1080,
        height=1920,
        fps=30.0,
        duration_s=frame_count / 30.0,
        frame_count=frame_count,
    )


def make(**overrides):
    payload = {
        "id": "tpl_x",
        "display_name": "T",
        "source_video_path": "a.mp4",
        "source_sha256": "0" * 64,
        "video": video(),
        "intro": FrameRange(start=0, end=30),
        "reveal": FrameRange(start=30, end=60),
        "transition_anchor_frame": 30,
        "consent": ConsentRecord(subject_kind="synthetic", adult_confirmed=True),
        "template_clothing_class": TemplateClothingClass.FITTED_SHORT,
        "directories": TemplateDirectories.standard("templates/tpl_x"),
    }
    payload.update(overrides)
    return HumanTemplate(**payload)


# -- frame ranges ----------------------------------------------------------
def test_frame_range_is_half_open() -> None:
    fr = FrameRange(start=10, end=20)
    assert fr.count == 10
    assert fr.last == 19
    assert fr.contains(10) and fr.contains(19)
    assert not fr.contains(20)
    assert list(fr.indices())[-1] == 19


def test_frame_range_rejects_inverted_bounds() -> None:
    with pytest.raises(PydanticValidationError):
        FrameRange(start=20, end=20)
    with pytest.raises(PydanticValidationError):
        FrameRange(start=21, end=20)


# -- transition anchor (requirement 8) ------------------------------------
def test_anchor_must_equal_intro_end() -> None:
    with pytest.raises(PydanticValidationError, match=r"intro\.end"):
        make(intro=FrameRange(start=0, end=29), transition_anchor_frame=30)


def test_anchor_must_equal_reveal_start() -> None:
    """An off-by-one gap at the seam is structurally impossible."""
    with pytest.raises(PydanticValidationError, match=r"reveal\.start"):
        make(reveal=FrameRange(start=31, end=60), transition_anchor_frame=30)


def test_anchor_overlap_is_rejected() -> None:
    with pytest.raises(PydanticValidationError):
        make(reveal=FrameRange(start=29, end=60), transition_anchor_frame=30)


def test_valid_anchor_has_no_gap_and_no_overlap() -> None:
    template = make()
    assert template.intro.last == template.transition_anchor_frame - 1
    assert template.reveal.start == template.transition_anchor_frame
    assert template.intro.count + template.reveal.count == template.video.frame_count
    assert template.total_output_frames == 60


def test_reveal_cannot_exceed_frame_count() -> None:
    with pytest.raises(PydanticValidationError, match="frame_count"):
        make(reveal=FrameRange(start=30, end=61))


# -- consent ---------------------------------------------------------------
def test_consented_human_requires_a_document() -> None:
    with pytest.raises(PydanticValidationError, match="consent_document_ref"):
        ConsentRecord(subject_kind="consented_human", adult_confirmed=True)


def test_adult_confirmation_is_mandatory() -> None:
    with pytest.raises(PydanticValidationError, match="adult_confirmed"):
        ConsentRecord(subject_kind="synthetic", adult_confirmed=False)


def test_unknown_subject_kind_is_rejected() -> None:
    with pytest.raises(PydanticValidationError):
        ConsentRecord(subject_kind="scraped", adult_confirmed=True)


# -- strictness ------------------------------------------------------------
def test_extra_fields_are_rejected() -> None:
    with pytest.raises(PydanticValidationError):
        make(unexpected_field="boom")


def test_naive_timestamps_are_rejected() -> None:
    from datetime import datetime

    with pytest.raises(PydanticValidationError):
        make(created_at=datetime(2026, 1, 1))


# -- garment ---------------------------------------------------------------
def test_sheer_material_requires_transparency() -> None:
    with pytest.raises(PydanticValidationError, match="sheer"):
        GarmentAsset(
            id="g1",
            category="top",
            body_coverage="torso",
            silhouette="fitted",
            material="mesh",
            transparency=0.0,
        )


def test_dominant_colors_must_be_hex() -> None:
    with pytest.raises(PydanticValidationError, match="hex"):
        GarmentAsset(
            id="g1",
            category="top",
            body_coverage="torso",
            silhouette="fitted",
            material="cotton",
            dominant_colors=["blue"],
        )


def test_source_url_must_be_http() -> None:
    with pytest.raises(PydanticValidationError):
        GarmentAsset(
            id="g1",
            category="top",
            body_coverage="torso",
            silhouette="fitted",
            material="cotton",
            source_url="file:///etc/passwd",
        )


# -- compatibility ---------------------------------------------------------
def report(state: CompatibilityState, results: list[RuleResult]) -> CompatibilityReport:
    return CompatibilityReport(
        id="cmp_1",
        template_id="t",
        template_version=1,
        garment_id="g",
        garment_version=1,
        rules_version="1",
        state=state,
        results=results,
        confidence=0.5,
    )


def test_state_must_match_rule_outcomes() -> None:
    with pytest.raises(PydanticValidationError, match="contradicts"):
        report(
            CompatibilityState.READY,
            [RuleResult(rule_id="r", outcome=RuleOutcome.FAIL, message="no")],
        )


def test_fail_outcome_implies_incompatible() -> None:
    built = report(
        CompatibilityState.INCOMPATIBLE,
        [RuleResult(rule_id="r", outcome=RuleOutcome.FAIL, message="no")],
    )
    assert built.state.blocks_render
    assert not built.render_allowed(utc_now())


def test_override_allows_render_and_keeps_audit_trail() -> None:
    built = report(
        CompatibilityState.NEEDS_INPUT,
        [RuleResult(rule_id="r", outcome=RuleOutcome.NEEDS_INPUT, message="need back view")],
    )
    override = ReviewerOverride(
        reviewer="operator",
        reason="back view is never visible in this choreography",
        approved_at=utc_now(),
        overridden_state=CompatibilityState.NEEDS_INPUT,
    )
    updated = built.with_override(override)
    assert updated.render_allowed(utc_now())
    assert not updated.render_allowed(utc_now(), allow_override=False)

    second = override.model_copy(update={"reason": "re-reviewed after mask work"})
    twice = updated.with_override(second)
    assert len(twice.override_audit_trail) == 1
    assert twice.override is second


def test_override_of_a_ready_state_is_nonsense() -> None:
    with pytest.raises(PydanticValidationError):
        ReviewerOverride(
            reviewer="op",
            reason="not needed at all here",
            approved_at=utc_now(),
            overridden_state=CompatibilityState.READY,
        )


def test_override_reason_must_be_substantive() -> None:
    with pytest.raises(PydanticValidationError):
        ReviewerOverride(
            reviewer="op",
            reason="ok",
            approved_at=utc_now(),
            overridden_state=CompatibilityState.NEEDS_INPUT,
        )


# -- job status ------------------------------------------------------------
def test_job_status_classification() -> None:
    assert JobStatus.COMPLETED.is_terminal
    assert JobStatus.FAILED.is_resumable
    assert not JobStatus.COMPLETED.is_resumable
