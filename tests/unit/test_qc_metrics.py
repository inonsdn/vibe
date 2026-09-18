"""QC metric primitives and the assembly/transition helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.core.errors import ValidationError
from app.media.assembly import assemble_frames, ensure_intro_cache, verify_transition
from app.media.frames import frame_path, write_frame
from app.qc.checks import CheckSeverity, QCReport, failed, passed, skipped
from app.qc.contact_sheet import build_contact_sheets, build_sheet
from app.qc.metrics import (
    bbox_to_mask,
    black_frame_fraction,
    boundary_band,
    duplicate_frame_indices,
    flicker_score,
    longest_run,
    mean_abs_frame_delta,
    region_diff,
    temporal_deltas,
)
from tests import fixtures


# -- metrics ---------------------------------------------------------------
def test_region_diff_is_zero_for_identical_frames() -> None:
    frame = fixtures.synth_frame(3)
    selection = np.ones(frame.shape[:2], dtype=bool)
    stats = region_diff(frame, frame, selection)
    assert stats.max_diff == 0
    assert stats.changed_pixels == 0


def test_region_diff_restricts_to_the_selection() -> None:
    a = np.zeros((10, 10, 3), dtype=np.uint8)
    b = a.copy()
    b[2, 2] = 50
    inside = np.zeros((10, 10), dtype=bool)
    inside[0:5, 0:5] = True
    assert region_diff(a, b, inside).max_diff == 50
    assert region_diff(a, b, ~inside).max_diff == 0


def test_region_diff_rejects_mismatched_selection() -> None:
    a = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="selection shape"):
        region_diff(a, a, np.ones((4, 4), dtype=bool))


def test_black_frame_detection() -> None:
    assert black_frame_fraction(np.zeros((10, 10, 3), dtype=np.uint8)) == 1.0
    assert black_frame_fraction(np.full((10, 10, 3), 200, dtype=np.uint8)) == 0.0


def test_boundary_band_lies_outside_the_mask() -> None:
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    band = boundary_band(mask, width_px=2)
    assert band.any()
    assert not (band & (mask > 0)).any()


def test_temporal_deltas_and_frozen_run_detection() -> None:
    moving = [fixtures.synth_frame(i) for i in range(6)]
    assert all(delta > 0 for delta in temporal_deltas(moving))

    frozen = [fixtures.synth_frame(0)] * 6
    deltas = temporal_deltas(frozen)
    assert all(delta == 0 for delta in deltas)
    assert longest_run(deltas, at_or_below=0.25) == 5


def test_longest_run_resets_on_motion() -> None:
    assert longest_run([0.0, 0.0, 5.0, 0.0, 0.0, 0.0], at_or_below=0.1) == 3


def test_flicker_score_summary() -> None:
    score = flicker_score([1.0, 3.0, 2.0])
    assert score["max_delta"] == 3.0
    assert score["mean_delta"] == pytest.approx(2.0)
    assert flicker_score([])["mean_delta"] == 0.0


def test_mean_abs_frame_delta_with_selection() -> None:
    a = np.zeros((8, 8, 3), dtype=np.uint8)
    b = a.copy()
    b[0:4, :, :] = 10
    selection = np.zeros((8, 8), dtype=bool)
    selection[4:8, :] = True
    assert mean_abs_frame_delta(a, b, selection) == 0.0
    assert mean_abs_frame_delta(a, b) > 0.0


def test_duplicate_detection_groups_identical_hashes() -> None:
    groups = duplicate_frame_indices({0: "a", 1: "b", 2: "a", 3: "c", 4: "a"})
    assert groups == [[0, 2, 4]]
    assert duplicate_frame_indices({0: "a", 1: "b"}) == []


def test_bbox_to_mask_clips_to_the_frame() -> None:
    mask = bbox_to_mask((10, 10), (8, 8, 10, 10))
    assert mask[9, 9]
    assert mask.sum() == 4
    assert not bbox_to_mask((10, 10), (20, 20, 5, 5)).any()


# -- check result types ----------------------------------------------------
def test_report_pass_fail_semantics() -> None:
    report = QCReport(
        job_id="job",
        checks=[
            passed("a", "fine"),
            failed("b", "warn only", severity=CheckSeverity.WARNING),
            skipped("c", "not applicable"),
        ],
    )
    assert report.passed  # a warning does not block
    assert [c.check_id for c in report.warnings] == ["b"]

    report.checks.append(failed("d", "hard failure"))
    assert not report.passed
    assert "d" in report.summary()["failed_check_ids"]


# -- assembly --------------------------------------------------------------
def write_seq(directory: Path, indices, value_base: int = 0) -> None:
    for index in indices:
        write_frame(
            frame_path(directory, index),
            np.full((16, 12, 3), (value_base + index) % 256, dtype=np.uint8),
        )


def test_intro_cache_is_byte_copied_and_reused(tmp_path: Path) -> None:
    source = tmp_path / "source"
    cache = tmp_path / "cache"
    write_seq(source, range(10))

    first = ensure_intro_cache(source, cache, 0, 5)
    assert not first.reused
    assert len(first.indices) == 5

    second = ensure_intro_cache(source, cache, 0, 5)
    assert second.reused
    assert second.digest == first.digest


def test_intro_cache_detects_tampering(tmp_path: Path) -> None:
    source, cache = tmp_path / "source", tmp_path / "cache"
    write_seq(source, range(6))
    original = ensure_intro_cache(source, cache, 0, 4)
    write_frame(frame_path(cache, 2), np.full((16, 12, 3), 99, dtype=np.uint8))
    with pytest.raises(ValidationError, match="immutable"):
        ensure_intro_cache(source, cache, 0, 4, expected_digest=original.digest)


def test_intro_cache_requires_the_source_frames(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="Intro frame missing"):
        ensure_intro_cache(tmp_path / "missing", tmp_path / "cache", 0, 3)


def test_assembly_joins_at_the_anchor_without_gap_or_overlap(tmp_path: Path) -> None:
    intro_cache, reveal, assembly = (tmp_path / n for n in ("intro", "reveal", "assembly"))
    write_seq(intro_cache, range(0, 6))
    write_seq(reveal, range(6, 12), value_base=100)

    report = assemble_frames(
        assembly,
        intro_cache_dir=intro_cache,
        intro_start=0,
        transition_anchor=6,
        reveal_frames_dir=reveal,
        reveal_end=12,
    )
    assert report.intro_frames == list(range(6))
    assert report.reveal_frames == list(range(6, 12))
    assert report.total_frames == 12
    assert report.intro_frames[-1] + 1 == report.reveal_frames[0] == 6

    evidence = verify_transition(assembly, intro_cache, reveal, 6)
    assert evidence["last_intro_matches_cache"] is True
    assert evidence["first_reveal_matches_render"] is True
    assert evidence["last_intro_frame"] == 5
    assert evidence["first_reveal_frame"] == 6


def test_assembly_fails_on_a_missing_reveal_frame(tmp_path: Path) -> None:
    intro_cache, reveal, assembly = (tmp_path / n for n in ("intro", "reveal", "assembly"))
    write_seq(intro_cache, range(0, 4))
    write_seq(reveal, [4, 6])  # 5 is missing
    with pytest.raises(ValidationError, match="Rendered reveal frame missing"):
        assemble_frames(
            assembly,
            intro_cache_dir=intro_cache,
            intro_start=0,
            transition_anchor=4,
            reveal_frames_dir=reveal,
            reveal_end=7,
        )


def test_assembly_rejects_an_empty_reveal_range(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="Reveal range is empty"):
        assemble_frames(
            tmp_path / "assembly",
            intro_cache_dir=tmp_path / "intro",
            intro_start=0,
            transition_anchor=5,
            reveal_frames_dir=tmp_path / "reveal",
            reveal_end=5,
        )


def test_flash_frames_alter_only_the_first_reveal_frames(tmp_path: Path) -> None:
    intro_cache, reveal, assembly = (tmp_path / n for n in ("intro", "reveal", "assembly"))
    write_seq(intro_cache, range(0, 4))
    write_seq(reveal, range(4, 10), value_base=50)

    report = assemble_frames(
        assembly,
        intro_cache_dir=intro_cache,
        intro_start=0,
        transition_anchor=4,
        reveal_frames_dir=reveal,
        reveal_end=10,
        flash_frames=3,
        flash_color=(255, 255, 255),
        flash_opacity=1.0,
    )
    assert report.flash_frames == [4, 5, 6]
    import cv2

    flashed = cv2.imread(str(frame_path(assembly, 4)))
    assert (flashed == 255).all()
    unflashed = cv2.imread(str(frame_path(assembly, 7)))
    original = cv2.imread(str(frame_path(reveal, 7)))
    assert np.array_equal(unflashed, original)


def test_flash_longer_than_the_reveal_is_rejected(tmp_path: Path) -> None:
    intro_cache, reveal = tmp_path / "intro", tmp_path / "reveal"
    write_seq(intro_cache, range(0, 2))
    write_seq(reveal, range(2, 4))
    with pytest.raises(ValidationError, match="longer than the reveal"):
        assemble_frames(
            tmp_path / "assembly",
            intro_cache_dir=intro_cache,
            intro_start=0,
            transition_anchor=2,
            reveal_frames_dir=reveal,
            reveal_end=4,
            flash_frames=4,
        )


# -- contact sheets --------------------------------------------------------
def test_contact_sheets_are_generated_around_the_transition(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    for index in range(20):
        write_frame(frame_path(frames, index), fixtures.synth_frame(index))
    sheets = build_contact_sheets(
        frames,
        tmp_path / "sheets",
        intro_start=0,
        transition_anchor=10,
        reveal_end=20,
        columns=4,
        tile_width=60,
        transition_span=3,
        period_frames=4,
    )
    assert len(sheets) == 2
    names = {path.name for path in sheets}
    assert names == {"contact_sheet_transition.png", "contact_sheet_reveal.png"}
    for path in sheets:
        assert path.stat().st_size > 0


def test_contact_sheet_returns_none_when_no_frames_exist(tmp_path: Path) -> None:
    assert build_sheet(tmp_path / "empty", [1, 2], tmp_path / "out.png") is None
