"""Config, hashing, logging, db migrations, frames, ffmpeg command building."""

from __future__ import annotations

import json
import logging
from io import StringIO
from pathlib import Path

import numpy as np
import pytest

from app.core.config import AppConfig, load_config
from app.core.errors import AppError, ConfigError, ValidationError
from app.core.hashing import canonical_json, sha256_dir, sha256_file, sha256_json, sha256_text
from app.core.ids import ID_PATTERN, job_id, slugify, template_id
from app.core.logging import JsonFormatter, configure_logging, get_logger, log_context, log_event
from app.db.database import Database
from app.db.migrations_runner import available_migrations, migrate, schema_version
from app.media import ffmpeg
from app.media.frames import (
    copy_frames,
    ffmpeg_pattern,
    frame_filename,
    frame_path,
    frames_equal,
    hash_sequence,
    list_frame_indices,
    validate_sequence,
    write_frame,
)


# -- config ----------------------------------------------------------------
def test_default_config_matches_the_delivery_contract() -> None:
    config = load_config(use_env=False)
    assert (config.video.width, config.video.height) == (1080, 1920)
    assert config.video.pixel_format == "yuv420p"
    assert config.video.video_codec == "libx264"
    assert config.runtime.vram_budget_mb == 8192
    assert config.backend.default == "mock"


def test_config_hash_is_order_independent(data_root: Path) -> None:
    a = load_config(overrides={"paths": {"data_root": str(data_root)}}, use_env=False)
    b = load_config(overrides={"paths": {"data_root": str(data_root)}}, use_env=False)
    assert a.config_hash() == b.config_hash()


def test_config_hash_changes_with_a_setting() -> None:
    base = load_config(use_env=False)
    changed = base.with_overrides(video={"crf": 30})
    assert base.config_hash() != changed.config_hash()


def test_env_overrides_are_applied() -> None:
    config = load_config(
        environ={"APP_VIDEO__CRF": "23", "APP_RUNTIME__OFFLINE_ENFORCED": "false"},
        use_env=True,
    )
    assert config.video.crf == 23
    assert config.runtime.offline_enforced is False


def test_unknown_config_key_is_rejected() -> None:
    with pytest.raises(ConfigError):
        load_config(overrides={"video": {"not_a_field": 1}}, use_env=False)


def test_odd_video_dimensions_are_rejected() -> None:
    with pytest.raises(ConfigError, match="even"):
        load_config(overrides={"video": {"width": 1081}}, use_env=False)


def test_flash_frames_must_be_zero_or_two_to_four() -> None:
    for value in (0, 2, 3, 4):
        assert load_config(overrides={"transition": {"flash_frames": value}}, use_env=False)
    for value in (1, 5, 10):
        with pytest.raises(ConfigError):
            load_config(overrides={"transition": {"flash_frames": value}}, use_env=False)


def test_mask_thresholds_must_be_ordered() -> None:
    with pytest.raises(ConfigError):
        load_config(
            overrides={"mask": {"immutable_threshold": 200, "editable_threshold": 100}},
            use_env=False,
        )


def test_config_is_frozen() -> None:
    """Config objects are immutable, so a manifest's config hash stays true."""
    from pydantic import ValidationError as PydanticValidationError

    config = load_config(use_env=False)
    with pytest.raises(PydanticValidationError):
        config.video.crf = 1  # type: ignore[misc]


def test_shipped_config_file_loads() -> None:
    from app.core.config import DEFAULT_CONFIG_DIR

    assert isinstance(load_config(DEFAULT_CONFIG_DIR / "app.yaml", use_env=False), AppConfig)


# -- hashing ---------------------------------------------------------------
def test_canonical_json_is_key_order_independent() -> None:
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})
    assert sha256_json({"b": 1, "a": 2}) == sha256_json({"a": 2, "b": 1})


def test_sha256_file_matches_text_hash(tmp_path: Path) -> None:
    path = tmp_path / "f.txt"
    path.write_text("hello", encoding="utf-8")
    assert sha256_file(path) == sha256_text("hello")


def test_sha256_dir_detects_content_and_rename_changes(tmp_path: Path) -> None:
    directory = tmp_path / "d"
    directory.mkdir()
    (directory / "a.txt").write_text("1", encoding="utf-8")
    (directory / "b.txt").write_text("2", encoding="utf-8")
    original = sha256_dir(directory)

    (directory / "b.txt").write_text("3", encoding="utf-8")
    assert sha256_dir(directory) != original

    (directory / "b.txt").write_text("2", encoding="utf-8")
    assert sha256_dir(directory) == original

    (directory / "b.txt").rename(directory / "c.txt")
    assert sha256_dir(directory) != original


# -- ids -------------------------------------------------------------------
def test_generated_ids_match_the_documented_pattern() -> None:
    for identifier in (template_id(), job_id()):
        assert ID_PATTERN.match(identifier), identifier


def test_ids_are_unique() -> None:
    assert len({job_id() for _ in range(200)}) == 200


def test_slugify() -> None:
    assert slugify("Héllo World!!") == "hello-world"
    assert slugify("") == "item"


# -- logging ---------------------------------------------------------------
def test_logs_are_single_line_json() -> None:
    stream = StringIO()
    configure_logging("INFO", stream=stream)
    logger = get_logger("test")
    log_event(logger, "something_happened", frames=3)
    payload = json.loads(stream.getvalue().strip())
    assert payload["event"] == "something_happened"
    assert payload["frames"] == 3
    assert payload["level"] == "INFO"
    assert payload["ts"].endswith("Z")


def test_reserved_field_names_do_not_break_logging() -> None:
    """A structured field called 'name' must not crash the logger."""
    stream = StringIO()
    configure_logging("INFO", stream=stream)
    logger = get_logger("test")
    log_event(logger, "named", name="a-name", message="a-message", module="m")
    payload = json.loads(stream.getvalue().strip())
    assert payload["name_"] == "a-name"
    assert payload["message_"] == "a-message"


def test_log_context_attaches_fields() -> None:
    stream = StringIO()
    configure_logging("INFO", stream=stream)
    logger = get_logger("test")
    with log_context(job_id="job_1"):
        log_event(logger, "inside")
    log_event(logger, "outside")
    lines = [json.loads(line) for line in stream.getvalue().strip().splitlines()]
    assert lines[0]["job_id"] == "job_1"
    assert "job_id" not in lines[1]


def test_exception_info_is_serialised() -> None:
    stream = StringIO()
    configure_logging("INFO", stream=stream)
    logger = get_logger("test")
    try:
        raise ValueError("boom")
    except ValueError:
        logger.exception("failed")
    payload = json.loads(stream.getvalue().strip())
    assert payload["exc_type"] == "ValueError"


def test_json_formatter_handles_non_serialisable_values() -> None:
    record = logging.LogRecord("app", logging.INFO, "f", 1, "msg", None, None)
    record.weird = object()  # type: ignore[attr-defined]
    assert json.loads(JsonFormatter().format(record))["weird"]


# -- errors ----------------------------------------------------------------
def test_app_errors_carry_codes_and_details() -> None:
    error = ValidationError("bad", field="x")
    assert error.code == "validation_error"
    assert error.http_status == 422
    assert error.to_dict()["details"] == {"field": "x"}
    assert isinstance(error, AppError)


# -- migrations ------------------------------------------------------------
def test_migrations_apply_in_order() -> None:
    database = Database.open(None, run_migrations=False)
    applied = database.migrate()
    assert applied == [1, 2, 3]
    assert database.schema_version == 3
    assert database.migrate() == []  # idempotent
    database.close()


def test_migration_files_are_well_named() -> None:
    migrations = available_migrations()
    assert [m.version for m in migrations] == sorted(m.version for m in migrations)
    assert all(m.sql.strip() for m in migrations)


def test_editing_an_applied_migration_is_refused(tmp_path: Path) -> None:
    from app.core.errors import ConflictError

    directory = tmp_path / "migrations"
    directory.mkdir()
    (directory / "0001_initial.sql").write_text("CREATE TABLE a (x INTEGER);", encoding="utf-8")
    database = Database.open(None, run_migrations=False)
    migrate(database.connection, directory=directory)
    (directory / "0001_initial.sql").write_text(
        "CREATE TABLE a (x INTEGER, y INTEGER);", encoding="utf-8"
    )
    with pytest.raises(ConflictError, match="changed on disk"):
        migrate(database.connection, directory=directory)
    database.close()


def test_badly_named_migration_is_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "migrations"
    directory.mkdir()
    (directory / "first.sql").write_text("SELECT 1;", encoding="utf-8")
    with pytest.raises(ValidationError, match="snake_case"):
        available_migrations(directory)


def test_foreign_keys_are_enforced() -> None:
    import sqlite3

    database = Database.open(None)
    with pytest.raises(sqlite3.IntegrityError):
        database.execute(
            "INSERT INTO job_frames (job_id, frame_index, status, updated_at)"
            " VALUES ('nope', 0, 'composited', '2026-01-01')"
        )
    database.close()


def test_schema_version_helper() -> None:
    database = Database.open(None)
    assert schema_version(database.connection) == 3
    database.close()


# -- frames ----------------------------------------------------------------
def test_frame_filenames_are_zero_padded() -> None:
    assert frame_filename(0) == "frame_000000.png"
    assert frame_filename(123456) == "frame_123456.png"
    with pytest.raises(ValidationError):
        frame_filename(-1)


def test_ffmpeg_pattern_translation() -> None:
    assert ffmpeg_pattern() == "frame_%06d.png"
    with pytest.raises(ValidationError):
        ffmpeg_pattern("weird.png")


def test_sequence_validation_reports_gaps_and_sizes(tmp_path: Path) -> None:
    directory = tmp_path / "frames"
    for index in (0, 1, 3):
        write_frame(frame_path(directory, index), np.zeros((8, 6, 3), dtype=np.uint8))
    write_frame(frame_path(directory, 4), np.zeros((9, 6, 3), dtype=np.uint8))
    report = validate_sequence(directory, 0, 5)
    assert report.missing == [2]
    assert report.wrong_size == [4]
    assert not report.ok
    assert list_frame_indices(directory) == [0, 1, 3, 4]


def test_copy_frames_is_byte_exact(tmp_path: Path) -> None:
    source = tmp_path / "src"
    destination = tmp_path / "dst"
    for index in range(3):
        write_frame(frame_path(source, index), np.full((8, 8, 3), index * 10, dtype=np.uint8))
    copy_frames(source, destination, range(3))
    for index in range(3):
        assert frames_equal(frame_path(source, index), frame_path(destination, index))
    assert hash_sequence(source, range(3)) == hash_sequence(destination, range(3))


def test_copy_frames_reports_a_missing_source(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="Source frame missing"):
        copy_frames(tmp_path / "src", tmp_path / "dst", [5])


def test_non_uint8_frames_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="uint8"):
        write_frame(tmp_path / "x.png", np.zeros((4, 4, 3), dtype=np.float32))


# -- ffmpeg command building ----------------------------------------------
def test_encode_command_pins_the_delivery_contract() -> None:
    argv = ffmpeg.build_encode_from_frames_command(
        "/data/frames/frame_%06d.png",
        "/data/out.mp4",
        fps=30.0,
        width=1080,
        height=1920,
    )
    joined = " ".join(argv)
    assert "-c:v libx264" in joined
    assert "-pix_fmt yuv420p" in joined
    assert "scale=1080:1920" in joined
    assert "-fps_mode cfr" in joined
    assert "-movflags +faststart" in joined


def test_fractional_frame_rates_are_expressed_losslessly() -> None:
    argv = ffmpeg.build_encode_from_frames_command(
        "/f/%06d.png", "/o.mp4", fps=30000 / 1001, width=1080, height=1920
    )
    assert "30000/1001" in argv


def test_frame_extraction_selects_an_exact_frame_range() -> None:
    argv = ffmpeg.build_extract_frames_command(
        "/in.mp4", "/out/frame_%06d.png", start_frame=30, end_frame=60
    )
    joined = " ".join(argv)
    assert "select=between(n\\,30\\,59)" in joined
    assert "-start_number 30" in joined


def test_audio_command_copies_without_reencoding() -> None:
    argv = ffmpeg.build_extract_audio_command("/in.mp4", "/a.m4a", start_s=1.0, duration_s=2.0)
    assert "copy" in argv
    assert "-vn" in argv


@pytest.mark.parametrize(
    "path",
    [
        "http://example.com/x.mp4",
        "https://example.com/x.mp4",
        "rtmp://host/live",
        "udp://239.0.0.1:1234",
        "concat:a.ts|b.ts",
        "pipe:0",
    ],
)
def test_remote_ffmpeg_io_is_refused(path: str) -> None:
    from app.core.errors import MediaToolError

    with pytest.raises(MediaToolError):
        ffmpeg.assert_local_path(path)


def test_frame_rate_parsing() -> None:
    assert ffmpeg.parse_frame_rate("30/1") == 30.0
    assert ffmpeg.parse_frame_rate("30000/1001") == pytest.approx(29.97, abs=0.001)
    assert ffmpeg.parse_frame_rate("0/0") is None
    assert ffmpeg.parse_frame_rate(None) is None
    assert ffmpeg.parse_frame_rate("bogus") is None


def test_missing_binary_gives_an_actionable_error() -> None:
    from app.core.errors import MediaToolError

    with pytest.raises(MediaToolError) as exc:
        ffmpeg.resolve_binary("definitely-not-a-real-binary-xyz")
    assert "hint" in exc.value.details
