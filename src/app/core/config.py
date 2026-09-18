"""Application configuration.

Configuration is layered, most specific last:

1. built-in defaults (this module),
2. ``config/app.yaml``,
3. an optional local override file (``config/local.yaml``, gitignored),
4. ``APP_*`` environment variables,
5. explicit keyword overrides (used by tests).

The resulting object is hashed into every render manifest, so configuration
drift is always visible after the fact.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.errors import ConfigError
from app.core.hashing import sha256_json
from app.core.paths import DataRoot

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_DIR = REPO_ROOT / "config"
ENV_PREFIX = "APP_"

LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0", "[::1]"})


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=False)


class PathsConfig(StrictModel):
    data_root: str = "data"
    config_dir: str = "config"
    workflows_dir: str = "workflows/comfyui"
    db_filename: str = "app.db"


class VideoConfig(StrictModel):
    """Final-encode settings. These are contract values, not suggestions."""

    width: int = 1080
    height: int = 1920
    fps: float = 30.0
    pixel_format: str = "yuv420p"
    video_codec: str = "libx264"
    crf: int = 17
    preset: str = "slow"
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    faststart: bool = True
    frame_format: Literal["png"] = "png"
    frame_filename_template: str = "frame_{index:06d}.png"
    preview_scale: float = 0.5
    preview_crf: int = 28

    @field_validator("width", "height")
    @classmethod
    def _even_positive(cls, value: int) -> int:
        if value <= 0 or value % 2 != 0:
            raise ValueError("dimension must be a positive even number for yuv420p")
        return value

    @field_validator("fps")
    @classmethod
    def _fps_positive(cls, value: float) -> float:
        if not 1.0 <= value <= 240.0:
            raise ValueError("fps must be between 1 and 240")
        return value


class MaskConfig(StrictModel):
    """Grayscale mask conventions. See docs/mask-semantics.md."""

    immutable_value: int = 0
    editable_value: int = 255
    feather_radius_px: int = 9
    feather_sigma: float = 0.0
    expansion_dilate_px: int = 0
    protected_dilate_px: int = 2
    #: Mask pixels at or below this value are treated as fully immutable.
    immutable_threshold: int = 2
    #: Mask pixels at or above this value are treated as fully editable.
    editable_threshold: int = 253
    #: Maximum fraction of the frame the editable region may cover.
    max_editable_area_fraction: float = 0.60
    #: Minimum fraction of the frame the editable region must cover to render.
    min_editable_area_fraction: float = 0.0005
    protected_wins: bool = True

    @model_validator(mode="after")
    def _thresholds_ordered(self) -> MaskConfig:
        if self.immutable_threshold >= self.editable_threshold:
            raise ValueError("immutable_threshold must be < editable_threshold")
        return self


class TransitionConfig(StrictModel):
    flash_frames: int = 0
    flash_color: tuple[int, int, int] = (255, 255, 255)
    flash_opacity: float = 0.85

    @field_validator("flash_frames")
    @classmethod
    def _flash_range(cls, value: int) -> int:
        if value not in (0, 2, 3, 4):
            raise ValueError("flash_frames must be 0 (off) or 2-4")
        return value


class BackendConfig(StrictModel):
    default: str = "mock"
    frame_window: int = 8
    max_retries: int = 2
    checkpoint_every_frames: int = 8


class ComfyUIConfig(StrictModel):
    base_url: str = "http://127.0.0.1:8188"
    allowed_hosts: tuple[str, ...] = ("127.0.0.1", "localhost", "::1")
    allow_remote: bool = False
    request_timeout_s: float = 30.0
    poll_interval_s: float = 1.0
    job_timeout_s: float = 900.0
    workflow_id: str = "garment_replace_placeholder"
    client_id: str = "garment-replacer"
    upload_inputs: bool = True

    @field_validator("base_url")
    @classmethod
    def _looks_like_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return value.rstrip("/")


class QCConfig(StrictModel):
    """Thresholds. Intermediate (lossless) frames are checked strictly."""

    lossless_protected_max_diff: int = 0
    lossless_background_max_diff: int = 0
    compressed_protected_mean_abs_diff: float = 2.0
    compressed_protected_max_diff: int = 24
    compressed_background_mean_abs_diff: float = 2.0
    face_region_mean_abs_diff: float = 2.0
    mask_boundary_leak_max_fraction: float = 0.002
    black_frame_luma_threshold: int = 12
    black_frame_max_fraction: float = 0.995
    frozen_frame_max_run: int = 6
    frozen_frame_mean_abs_diff: float = 0.25
    garment_flicker_max_mean_delta: float = 28.0
    duration_tolerance_s: float = 0.08
    fps_tolerance: float = 0.01
    contact_sheet_columns: int = 6
    contact_sheet_tile_width: int = 240
    contact_sheet_period_frames: int = 24
    contact_sheet_transition_span: int = 6


class CompatibilityConfig(StrictModel):
    rules_file: str = "compatibility_rules.v1.yaml"
    block_on_needs_input: bool = True
    allow_override: bool = True


class APIConfig(StrictModel):
    host: str = "127.0.0.1"
    port: int = 8077
    allow_remote_bind: bool = False
    cors_origins: tuple[str, ...] = ()
    enable_web_ui: bool = True


class RuntimeConfig(StrictModel):
    log_level: str = "INFO"
    log_to_file: bool = True
    offline_enforced: bool = True
    default_seed: int = 20240501
    vram_budget_mb: int = 8192
    ram_budget_mb: int = 32768
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"
    max_concurrent_jobs: int = 1


class AppConfig(StrictModel):
    """Top-level configuration object."""

    paths: PathsConfig = Field(default_factory=PathsConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    mask: MaskConfig = Field(default_factory=MaskConfig)
    transition: TransitionConfig = Field(default_factory=TransitionConfig)
    backend: BackendConfig = Field(default_factory=BackendConfig)
    comfyui: ComfyUIConfig = Field(default_factory=ComfyUIConfig)
    qc: QCConfig = Field(default_factory=QCConfig)
    compatibility: CompatibilityConfig = Field(default_factory=CompatibilityConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    # -- derived helpers --------------------------------------------------
    @property
    def repo_root(self) -> Path:
        return REPO_ROOT

    def data_root(self) -> DataRoot:
        raw = Path(self.paths.data_root)
        base = raw if raw.is_absolute() else REPO_ROOT / raw
        return DataRoot(base)

    def config_dir(self) -> Path:
        raw = Path(self.paths.config_dir)
        return raw if raw.is_absolute() else REPO_ROOT / raw

    def workflows_dir(self) -> Path:
        raw = Path(self.paths.workflows_dir)
        return raw if raw.is_absolute() else REPO_ROOT / raw

    def config_hash(self) -> str:
        return sha256_json(self.model_dump(mode="json"))

    def with_overrides(self, **overrides: Any) -> AppConfig:
        merged = _deep_merge(self.model_dump(mode="python"), overrides)
        return AppConfig.model_validate(merged)


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce_env_value(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none", ""}:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if "," in raw:
        return tuple(part.strip() for part in raw.split(",") if part.strip())
    return raw


def _env_overrides(environ: dict[str, str], known_sections: set[str]) -> dict[str, Any]:
    """Map ``APP_SECTION__FIELD=value`` into a nested override dict."""
    out: dict[str, Any] = {}
    for key, value in environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        remainder = key[len(ENV_PREFIX) :]
        if "__" not in remainder:
            continue
        section, field = remainder.split("__", 1)
        section_l, field_l = section.lower(), field.lower()
        if section_l not in known_sections:
            continue
        out.setdefault(section_l, {})[field_l] = _coerce_env_value(value)
    return out


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a YAML mapping, raising ConfigError on anything else."""
    file = Path(path)
    try:
        text = file.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Cannot read config file: {file}", error=str(exc)) from exc
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {file}", error=str(exc)) from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Config file must contain a mapping: {file}")
    return data


def load_config(
    config_file: str | os.PathLike[str] | None = None,
    *,
    environ: dict[str, str] | None = None,
    overrides: dict[str, Any] | None = None,
    use_env: bool = True,
) -> AppConfig:
    """Build an :class:`AppConfig` from files, environment and overrides."""
    layers: dict[str, Any] = {}

    if config_file is None:
        candidate = DEFAULT_CONFIG_DIR / "app.yaml"
        config_file = candidate if candidate.is_file() else None
    if config_file is not None:
        layers = _deep_merge(layers, load_yaml(config_file))

    local = DEFAULT_CONFIG_DIR / "local.yaml"
    if local.is_file():
        layers = _deep_merge(layers, load_yaml(local))

    known = set(AppConfig.model_fields)
    if use_env:
        layers = _deep_merge(layers, _env_overrides(dict(environ or os.environ), known))
    if overrides:
        layers = _deep_merge(layers, overrides)

    try:
        return AppConfig.model_validate(layers)
    except Exception as exc:  # pydantic ValidationError
        raise ConfigError("Invalid configuration", error=str(exc)) from exc


__all__ = [
    "DEFAULT_CONFIG_DIR",
    "ENV_PREFIX",
    "LOCAL_HOSTS",
    "REPO_ROOT",
    "APIConfig",
    "AppConfig",
    "BackendConfig",
    "ComfyUIConfig",
    "CompatibilityConfig",
    "MaskConfig",
    "PathsConfig",
    "QCConfig",
    "RuntimeConfig",
    "TransitionConfig",
    "VideoConfig",
    "load_config",
    "load_yaml",
]
