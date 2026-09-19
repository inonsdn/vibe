"""Configuration, the adapter factory and the `motion extract-pose` command.

The point of these tests is that an operator who has not yet downloaded any
weights gets a clear, actionable refusal at every surface — and that when the
paths *are* configured, they arrive at the adapter from wherever they were set:
``config/local.yaml``, an environment variable, or a CLI flag.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.adapters.dwpose import DWPoseOnnxAdapter
from app.adapters.dwpose.session import CPU_PROVIDER, CUDA_PROVIDER
from app.adapters.factory import POSE_ADAPTERS, create_pose_adapter
from app.adapters.pose import MockPoseAdapter
from app.core.config import load_config
from app.core.errors import ConfigError
from tests import dwpose_fixtures as dw
from tests import motion_fixtures as mf

runner = CliRunner()


def invoke(data_root: Path, *args: str):
    from app.cli.main import app as cli_app

    return runner.invoke(cli_app, ["--data-root", str(data_root), *args])


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
def test_pose_settings_default_to_nothing_configured() -> None:
    """Out of the box the system has no weights and says so."""
    config = load_config(use_env=False)
    assert config.pose.adapter == "none"
    assert config.pose.detector_model == ""
    assert config.pose.pose_model == ""
    assert config.pose.provider == "auto"
    assert config.pose.require_requested_provider is True


def test_model_paths_come_from_config(tmp_path: Path) -> None:
    detector, pose = dw.touch_models(tmp_path)
    config = load_config(
        overrides={"pose": {"detector_model": str(detector), "pose_model": str(pose)}},
        use_env=False,
    )
    adapter = create_pose_adapter("dwpose_onnx", config)
    assert adapter.capability().available is True


def test_model_paths_come_from_the_environment(tmp_path: Path) -> None:
    """APP_POSE__DETECTOR_MODEL / APP_POSE__POSE_MODEL, as documented."""
    detector, pose = dw.touch_models(tmp_path)
    config = load_config(
        environ={
            "APP_POSE__DETECTOR_MODEL": str(detector),
            "APP_POSE__POSE_MODEL": str(pose),
            "APP_POSE__PROVIDER": "cpu",
        },
    )
    assert config.pose.detector_model == str(detector)
    assert config.pose.provider == "cpu"


def test_cli_overrides_beat_configuration(tmp_path: Path) -> None:
    detector, pose = dw.touch_models(tmp_path)
    other = tmp_path / "other_pose.onnx"
    other.write_bytes(b"x")
    config = load_config(
        overrides={"pose": {"detector_model": str(detector), "pose_model": str(pose)}},
        use_env=False,
    )
    adapter = create_pose_adapter("dwpose_onnx", config, pose_model=str(other))
    assert adapter.settings.pose_model == str(other)
    assert adapter.settings.detector_model == str(detector)


def test_an_unset_cli_flag_does_not_blank_a_configured_path(tmp_path: Path) -> None:
    detector, pose = dw.touch_models(tmp_path)
    config = load_config(
        overrides={"pose": {"detector_model": str(detector), "pose_model": str(pose)}},
        use_env=False,
    )
    adapter = create_pose_adapter("dwpose_onnx", config, pose_model=None, provider=None)
    assert adapter.settings.pose_model == str(pose)
    assert adapter.settings.provider == "auto"


def test_roi_configuration_is_validated() -> None:
    with pytest.raises(Exception, match="roi"):
        load_config(overrides={"pose": {"roi_mode": "pixels", "roi": [0, 0, 0, 0]}}, use_env=False)
    with pytest.raises(Exception, match="normalized"):
        load_config(
            overrides={"pose": {"roi_mode": "normalized", "roi": [0, 0, 2.5, 1.0]}},
            use_env=False,
        )


def test_the_shipped_config_ships_no_model_paths() -> None:
    """config/app.yaml is committed and hashed into every manifest; it must
    never carry a machine-specific path or a download URL."""
    from app.core.config import DEFAULT_CONFIG_DIR

    text = (DEFAULT_CONFIG_DIR / "app.yaml").read_text(encoding="utf-8")
    assert 'detector_model: ""' in text
    assert 'pose_model: ""' in text
    # Scoped to the pose section: the ComfyUI section legitimately names a
    # loopback URL, which is not a download.
    pose_section = text.split("\npose:", 1)[1].split("\nanimator:", 1)[0]
    for banned in ("http://", "https://", "huggingface", "download"):
        assert banned not in pose_section, banned

    config = load_config(use_env=False)
    assert config.pose.detector_model == ""
    assert config.pose.pose_model == ""
    assert config.pose.adapter == "none", "a fresh clone must not assume weights"


def test_the_example_local_config_is_shipped_and_is_only_an_example() -> None:
    """config/local.example.yaml documents the shape; config/local.yaml is
    gitignored, so no operator's paths can be committed by accident."""
    from app.core.config import DEFAULT_CONFIG_DIR

    example = DEFAULT_CONFIG_DIR / "local.example.yaml"
    assert example.is_file()
    text = example.read_text(encoding="utf-8")
    assert "detector_model" in text and "pose_model" in text
    for banned in ("http://", "https://"):
        assert banned not in text, banned

    import subprocess

    repo = Path(DEFAULT_CONFIG_DIR).parent
    ignored = subprocess.run(
        ["git", "check-ignore", "config/local.yaml"], cwd=repo, capture_output=True, text=True
    )
    assert ignored.returncode == 0, "config/local.yaml must be gitignored"
    committed = subprocess.run(
        ["git", "check-ignore", "config/local.example.yaml"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert committed.returncode != 0, "the example template must be committable"


# ---------------------------------------------------------------------------
# adapter factory
# ---------------------------------------------------------------------------
def test_the_factory_builds_each_known_adapter(tmp_path: Path) -> None:
    detector, pose = dw.touch_models(tmp_path)
    config = load_config(
        overrides={"pose": {"detector_model": str(detector), "pose_model": str(pose)}},
        use_env=False,
    )
    assert isinstance(create_pose_adapter("dwpose_onnx", config), DWPoseOnnxAdapter)
    assert isinstance(create_pose_adapter("mock", config), MockPoseAdapter)
    # "registry" is the honest stub until a real adapter is configured as the
    # process-wide default.
    assert create_pose_adapter("registry", config).name == "pose-estimation"
    assert set(POSE_ADAPTERS) == {"dwpose_onnx", "mock", "registry"}


def test_an_unknown_adapter_is_refused() -> None:
    with pytest.raises(ConfigError, match="Unknown pose adapter"):
        create_pose_adapter("openpose", load_config(use_env=False))


def test_the_mock_adapter_is_never_a_fallback_for_a_missing_model() -> None:
    """Producing synthetic poses because a weight file was absent is the exact
    failure this system exists to refuse."""
    config = load_config(use_env=False)
    adapter = create_pose_adapter("dwpose_onnx", config)
    assert isinstance(adapter, DWPoseOnnxAdapter)
    assert adapter.capability().available is False
    with pytest.raises(ConfigError, match="no model or provider settings"):
        create_pose_adapter("mock", config, detector_model="/somewhere.onnx")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_extract_pose_help_documents_the_flags(data_root: Path) -> None:
    result = invoke(data_root, "motion", "extract-pose", "--help")
    assert result.exit_code == 0
    for flag in ("--adapter", "--detector-model", "--pose-model", "--provider", "--diagnostics"):
        assert flag in result.output
    assert "never downloaded" in result.output


def test_extract_pose_refuses_without_models(context, data_root: Path) -> None:
    source = mf.register_motion_source(
        context, motion_id="mot_cli", spec=mf.MOTION_A, start=0, end=10
    )
    context.close()

    result = invoke(data_root, "motion", "extract-pose", source.source.id)
    assert result.exit_code == 2
    assert "unavailable" in result.output
    assert "never downloads" in result.output


def test_extract_pose_reports_a_bad_roi_flag(context, data_root: Path) -> None:
    result = invoke(data_root, "motion", "extract-pose", "mot_x", "--roi", "1,2,3")
    assert result.exit_code == 2
    assert "four comma-separated" in result.output


def test_extract_pose_runs_end_to_end_with_the_mock_adapter(context, data_root: Path) -> None:
    """`--adapter mock` is the offline smoke test an operator can run before
    any weights exist. It is opt-in and labelled synthetic."""
    source = mf.register_motion_source(
        context, motion_id="mot_cli_mock", spec=mf.MOTION_A, start=5, end=25
    )
    for path in source.pose_dir.glob("*.json"):
        path.unlink()
    context.close()

    result = invoke(
        data_root, "--json", "motion", "extract-pose", source.source.id, "--adapter", "mock"
    )
    assert result.exit_code == 0, result.output
    body = json.loads(result.stdout)
    assert body["imported_range"] == [5, 24]
    assert body["imported_count"] == 20


def test_extract_pose_prints_the_provider_it_used(context, data_root, tmp_path) -> None:
    """The human-readable output leads with the provider, because a silent CPU
    fallback is the difference between four minutes and ninety."""
    import app.adapters.factory as factory_module

    source = mf.register_motion_source(
        context, motion_id="mot_cli_dw", spec=mf.MOTION_A, start=0, end=6
    )
    for path in source.pose_dir.glob("*.json"):
        path.unlink()
    context.close()

    detector_path, pose_path = dw.touch_models(tmp_path)
    points = dw.canonical_body_points()
    scale = min(dw.DETECTOR_INPUT[0] / 360, dw.DETECTOR_INPUT[1] / 640)
    fake = dw.FakeSessionFactory(
        detector=dw.FakeSession(
            lambda _b: dw.yolox_output([(110.0, 150.0, 250.0, 550.0, 0.9)], scale=scale)
        ),
        pose=dw.FakeSession(lambda _b: dw.simcc_output(points, [0.8] * len(points))),
        available=(CUDA_PROVIDER, CPU_PROVIDER),
    )
    real_create = factory_module.create_pose_adapter

    def patched(name, config, **kwargs):
        kwargs.setdefault("session_factory", fake)
        kwargs.setdefault("frame_reader", dw.FakeFrameReader(width=360, height=640))
        kwargs.setdefault("detector_input_size", dw.DETECTOR_INPUT)
        kwargs.setdefault("detector_strides", dw.DETECTOR_STRIDES)
        kwargs.setdefault("pose_input_size", dw.POSE_INPUT)
        kwargs.setdefault("simcc_split_ratio", dw.SPLIT_RATIO)
        kwargs.setdefault("emit_hands", False)
        return real_create(name, config, **kwargs)

    factory_module.create_pose_adapter = patched
    try:
        result = invoke(
            data_root,
            "motion",
            "extract-pose",
            source.source.id,
            "--detector-model",
            str(detector_path),
            "--pose-model",
            str(pose_path),
            "--provider",
            "cuda",
        )
    finally:
        factory_module.create_pose_adapter = real_create

    assert result.exit_code == 0, result.output
    assert "CUDAExecutionProvider" in result.output
    assert "extracted  6 pose frame(s)" in result.output


def test_doctor_reports_the_pose_adapter_as_not_yet_configured(data_root: Path) -> None:
    result = invoke(data_root, "doctor")
    assert "pose" in result.output
    assert "not_implemented" in result.output or "missing_weights" in result.output


# ---------------------------------------------------------------------------
# `doctor` / `offline verify` answer the question an operator is asking
# ---------------------------------------------------------------------------
def test_offline_verify_reports_the_configured_pose_adapter(tmp_path: Path) -> None:
    """The registry says `not_implemented` forever, because a real adapter needs
    files only the operator can supply. This reports whether they are there."""
    from app.offline.verify import configured_pose_adapter

    unset = configured_pose_adapter(load_config(use_env=False))
    assert unset["status"] == "not_configured"
    assert unset["adapter"] == "none"
    assert "docs/dwpose-setup.md" in unset["reason"]
    assert unset["downloads"] == "none - model files are supplied by the operator"

    selected_but_absent = configured_pose_adapter(
        load_config(
            overrides={"pose": {"adapter": "dwpose_onnx", "pose_model": "/nope.onnx"}},
            use_env=False,
        )
    )
    assert selected_but_absent["status"] == "missing_weights"
    assert selected_but_absent["available"] is False
    assert selected_but_absent["detector_model_configured"] is False

    detector, pose = dw.touch_models(tmp_path)
    ready = configured_pose_adapter(
        load_config(
            overrides={
                "pose": {
                    "adapter": "dwpose_onnx",
                    "detector_model": str(detector),
                    "pose_model": str(pose),
                    "provider": "cuda",
                }
            },
            use_env=False,
        )
    )
    assert ready["status"] == "available"
    assert ready["available"] is True
    assert ready["provider_requested"] == "cuda"


def test_offline_verify_never_breaks_on_a_misconfiguration() -> None:
    """`doctor` must still run and explain itself when the config is wrong."""
    from app.offline.verify import configured_pose_adapter

    broken = configured_pose_adapter(
        load_config(overrides={"pose": {"adapter": "mock"}}, use_env=False)
    )
    assert broken["status"] in {"available", "error"}
    assert "reason" in broken


def test_the_verify_report_text_names_the_pose_adapter(tmp_path: Path) -> None:
    from app.offline.verify import render_text_report, verify_offline

    report = verify_offline(load_config(use_env=False), probe_comfyui=False)
    text = render_text_report(report)
    assert "Configured pose adapter (model files are never downloaded)" in text
    assert "not_configured" in text
    assert report.as_dict()["configured_pose_adapter"]["adapter"] == "none"
