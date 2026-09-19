"""Offline guarantees, verified rather than asserted.

What this module checks:

* every configured network endpoint, and whether it is loopback-only,
* that no cloud integration is required (there are none to find — this is a
  positive statement about the code, listed explicitly so a reviewer can
  confirm it),
* which local executables are present or missing,
* FFmpeg and ffprobe specifically,
* ComfyUI, if it happens to be running (optional),
* that every runtime data path exists and is writable,
* GPU/CUDA information *when available*, without requiring CUDA.

It never makes an outbound request. The ComfyUI probe is a loopback HTTP call
to a port on this machine and is skipped entirely when the endpoint is not
local.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.adapters.base import AdapterKind
from app.adapters.base import registry as adapter_registry
from app.backends.comfyui.client import assert_local_endpoint
from app.core.config import AppConfig
from app.core.errors import OfflinePolicyError
from app.core.logging import get_logger
from app.core.paths import RUNTIME_SUBDIRS
from app.core.provenance import collect as collect_provenance

logger = get_logger(__name__)

REQUIRED_EXECUTABLES: tuple[str, ...] = ("ffmpeg", "ffprobe")
OPTIONAL_EXECUTABLES: tuple[str, ...] = ("git", "nvidia-smi", "python")

#: Cloud services this application integrates with. Deliberately empty, and
#: asserted by a test so it cannot quietly grow.
CLOUD_INTEGRATIONS: tuple[str, ...] = ()

#: Outbound network features. Also deliberately empty.
NETWORK_FEATURES: tuple[str, ...] = ()


@dataclass
class EndpointReport:
    name: str
    url: str
    is_local: bool
    allowed: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "url": self.url,
            "is_local": self.is_local,
            "allowed": self.allowed,
            "reason": self.reason,
        }


@dataclass
class OfflineReport:
    ok: bool
    endpoints: list[EndpointReport] = field(default_factory=list)
    executables: dict[str, str | None] = field(default_factory=dict)
    missing_required: list[str] = field(default_factory=list)
    paths: dict[str, dict[str, Any]] = field(default_factory=dict)
    gpu: dict[str, Any] = field(default_factory=dict)
    adapters: dict[str, Any] = field(default_factory=dict)
    #: The pose adapter the configuration actually selects, which is not the
    #: same thing as the process-wide registry entry.
    configured_pose: dict[str, Any] = field(default_factory=dict)
    backends: dict[str, Any] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "cloud_integrations": list(CLOUD_INTEGRATIONS),
            "network_features": list(NETWORK_FEATURES),
            "endpoints": [endpoint.as_dict() for endpoint in self.endpoints],
            "executables": self.executables,
            "missing_required_executables": self.missing_required,
            "paths": self.paths,
            "gpu": self.gpu,
            "adapters": self.adapters,
            "configured_pose_adapter": self.configured_pose,
            "backends": self.backends,
            "problems": self.problems,
            "warnings": self.warnings,
            "provenance": self.provenance,
        }


def adapter_capability(kind: AdapterKind) -> dict[str, Any]:
    """One adapter's capability report, or a marker that none is registered."""
    adapter = adapter_registry.get(kind)
    return adapter.capability().as_dict() if adapter is not None else {"status": "unregistered"}


def configured_pose_adapter(config: AppConfig) -> dict[str, Any]:
    """The pose adapter ``config`` selects, and whether it could run now.

    The registry answers "what does a bare installation do?" and will say
    ``not_implemented`` forever, because a real adapter needs model files that
    only the operator can supply. This answers the question an operator is
    actually asking: are my weights where I said they were?
    """
    name = config.pose.adapter
    info: dict[str, Any] = {
        "adapter": name,
        "detector_model_configured": bool(config.pose.detector_model),
        "pose_model_configured": bool(config.pose.pose_model),
        "provider_requested": config.pose.provider,
        "downloads": "none - model files are supplied by the operator",
    }
    if name in {"none", ""}:
        info["status"] = "not_configured"
        info["reason"] = (
            "No pose adapter is selected. Poses must be imported with "
            "`app motion import-pose`, or set pose.adapter to dwpose_onnx and "
            "point pose.detector_model / pose.pose_model at local ONNX files. "
            "See docs/dwpose-setup.md."
        )
        return info

    try:
        from app.adapters.factory import create_pose_adapter

        capability = create_pose_adapter(name, config).capability()
    except Exception as exc:  # a misconfiguration must not break `doctor`
        info["status"] = "error"
        info["reason"] = str(exc)
        return info

    info["status"] = capability.status.value
    info["reason"] = capability.reason
    info["available"] = capability.available
    return info


def check_endpoints(config: AppConfig) -> list[EndpointReport]:
    """Report every configured endpoint and whether the policy allows it."""
    reports: list[EndpointReport] = []

    comfy_url = config.comfyui.base_url
    try:
        assert_local_endpoint(comfy_url, config.comfyui)
        reports.append(
            EndpointReport(
                name="comfyui",
                url=comfy_url,
                is_local=True,
                allowed=True,
                reason="loopback endpoint",
            )
        )
    except OfflinePolicyError as exc:
        allowed = bool(config.comfyui.allow_remote)
        reports.append(
            EndpointReport(
                name="comfyui",
                url=comfy_url,
                is_local=False,
                allowed=allowed,
                reason=exc.message,
            )
        )

    api_host = config.api.host
    api_local = api_host in {"127.0.0.1", "localhost", "::1"}
    reports.append(
        EndpointReport(
            name="api_bind",
            url=f"http://{api_host}:{config.api.port}",
            is_local=api_local,
            allowed=api_local or config.api.allow_remote_bind,
            reason=(
                "loopback bind"
                if api_local
                else "binds a non-loopback interface; set api.allow_remote_bind "
                "explicitly if that is intended"
            ),
        )
    )
    return reports


def check_executables() -> tuple[dict[str, str | None], list[str]]:
    found: dict[str, str | None] = {}
    for name in (*REQUIRED_EXECUTABLES, *OPTIONAL_EXECUTABLES):
        found[name] = shutil.which(name)
    missing = [name for name in REQUIRED_EXECUTABLES if found.get(name) is None]
    return found, missing


def check_paths(config: AppConfig) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Verify each runtime directory exists and is actually writable."""
    data_root = config.data_root()
    results: dict[str, dict[str, Any]] = {}
    problems: list[str] = []

    root = data_root.path
    results["data_root"] = _probe_path(root)
    if not results["data_root"]["writable"]:
        problems.append(f"data root is not writable: {root}")

    for name in RUNTIME_SUBDIRS:
        try:
            target = data_root.resolve(name)
        except Exception as exc:  # pragma: no cover - defensive
            results[name] = {"path": name, "exists": False, "writable": False, "error": str(exc)}
            problems.append(f"cannot resolve runtime path {name}: {exc}")
            continue
        results[name] = _probe_path(target)
        if not results[name]["writable"]:
            problems.append(f"runtime path is not writable: {target}")
    return results, problems


def _probe_path(path: Path) -> dict[str, Any]:
    exists = path.exists()
    writable = False
    error: str | None = None
    if exists:
        writable = os.access(path, os.W_OK)
    else:
        try:
            path.mkdir(parents=True, exist_ok=True)
            exists = True
            writable = os.access(path, os.W_OK)
        except OSError as exc:
            error = str(exc)
    if writable:
        # os.access can lie on some filesystems; do a real write.
        probe = path / ".write_probe"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            writable = False
            error = str(exc)
    payload: dict[str, Any] = {"path": str(path), "exists": exists, "writable": writable}
    if error:
        payload["error"] = error
    return payload


def check_gpu() -> dict[str, Any]:
    """Best-effort GPU report. Never required; never raises."""
    info: dict[str, Any] = {
        "platform": platform.system(),
        "cuda_available": False,
        "torch_installed": False,
        "nvidia_smi": shutil.which("nvidia-smi") is not None,
        "devices": [],
        "note": (
            "GPU information is advisory. The mock backend and the whole test "
            "suite run without a GPU or CUDA."
        ),
    }

    try:  # torch is not a dependency of this project
        import torch  # type: ignore[import-not-found]

        info["torch_installed"] = True
        info["torch_version"] = getattr(torch, "__version__", "unknown")
        available = bool(torch.cuda.is_available())
        info["cuda_available"] = available
        if available:
            info["cuda_version"] = getattr(torch.version, "cuda", None)
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                info["devices"].append(
                    {
                        "index": index,
                        "name": properties.name,
                        "total_memory_mb": int(properties.total_memory / (1024 * 1024)),
                        "capability": f"{properties.major}.{properties.minor}",
                    }
                )
    except Exception:
        pass

    if not info["devices"] and info["nvidia_smi"]:
        info.update(_nvidia_smi_devices())
    return info


def _nvidia_smi_devices() -> dict[str, Any]:
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return {}
    try:
        result = subprocess.run(
            [
                binary,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if result.returncode != 0:
        return {}
    devices: list[dict[str, Any]] = []
    for index, line in enumerate(result.stdout.strip().splitlines()):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            memory = int(float(parts[1]))
        except ValueError:
            memory = 0
        devices.append(
            {
                "index": index,
                "name": parts[0],
                "total_memory_mb": memory,
                "driver_version": parts[2] if len(parts) > 2 else None,
            }
        )
    return {"devices": devices, "source": "nvidia-smi"}


def check_comfyui(config: AppConfig) -> dict[str, Any]:
    """Probe ComfyUI on loopback. Optional; absence is not a failure."""
    from app.backends.comfyui.client import ComfyUIClient

    payload: dict[str, Any] = {
        "configured_url": config.comfyui.base_url,
        "allow_remote": config.comfyui.allow_remote,
        "available": False,
    }
    try:
        client = ComfyUIClient(config.comfyui)
    except OfflinePolicyError as exc:
        payload["error"] = exc.message
        payload["refused_by_policy"] = True
        return payload
    try:
        payload["available"] = client.is_healthy()
        if payload["available"]:
            stats = client.health()
            system = stats.get("system", {}) if isinstance(stats, dict) else {}
            payload["comfyui_version"] = system.get("comfyui_version")
    except Exception as exc:
        payload["error"] = str(exc)
    finally:
        client.close()
    return payload


def check_backends(config: AppConfig) -> dict[str, Any]:
    from app.backends.registry import available_backends, create_backend

    out: dict[str, Any] = {}
    for name in available_backends():
        try:
            backend = create_backend(name, config)
        except Exception as exc:
            out[name] = {"constructed": False, "error": str(exc)}
            continue
        try:
            out[name] = {
                "constructed": True,
                "capabilities": backend.capabilities().as_dict(),
            }
        finally:
            backend.close()
    return out


def verify_offline(
    config: AppConfig,
    *,
    probe_comfyui: bool = True,
    include_backends: bool = True,
) -> OfflineReport:
    """Run every offline/environment check and aggregate the result."""
    problems: list[str] = []
    warnings: list[str] = []

    endpoints = check_endpoints(config)
    for endpoint in endpoints:
        if not endpoint.allowed:
            problems.append(f"endpoint {endpoint.name} is not allowed: {endpoint.reason}")
        elif not endpoint.is_local:
            warnings.append(
                f"endpoint {endpoint.name} is not local but has been explicitly "
                f"allowed: {endpoint.url}"
            )

    executables, missing = check_executables()
    if missing:
        problems.append(
            "missing required executables: "
            + ", ".join(missing)
            + " (install FFmpeg and put ffmpeg/ffprobe on PATH)"
        )

    paths, path_problems = check_paths(config)
    problems.extend(path_problems)

    if CLOUD_INTEGRATIONS:  # pragma: no cover - guarded by a test
        problems.append("cloud integrations are present: " + ", ".join(CLOUD_INTEGRATIONS))
    if NETWORK_FEATURES:  # pragma: no cover - guarded by a test
        problems.append("outbound network features are present: " + ", ".join(NETWORK_FEATURES))

    adapters = {kind.value: adapter_capability(kind) for kind in AdapterKind}

    report = OfflineReport(
        ok=not problems,
        endpoints=endpoints,
        executables=executables,
        missing_required=missing,
        paths=paths,
        gpu=check_gpu(),
        adapters=adapters,
        configured_pose=configured_pose_adapter(config),
        backends=check_backends(config) if include_backends else {},
        problems=problems,
        warnings=warnings,
        provenance=collect_provenance(config),
    )
    if probe_comfyui:
        comfy = check_comfyui(config)
        report.backends.setdefault("comfyui", {})["runtime"] = comfy
        if not comfy.get("available"):
            warnings.append("ComfyUI is not running (optional; the mock backend needs nothing).")
        report.warnings = warnings
    return report


def render_text_report(report: OfflineReport) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"OFFLINE / ENVIRONMENT VERIFICATION  -  {'OK' if report.ok else 'PROBLEMS'}")
    lines.append("=" * 72)

    lines.append("")
    lines.append("Endpoints")
    for endpoint in report.endpoints:
        flag = (
            "local" if endpoint.is_local else ("allowed-remote" if endpoint.allowed else "REFUSED")
        )
        lines.append(f"  [{flag:>14}] {endpoint.name:<10} {endpoint.url}")
        if not endpoint.is_local:
            lines.append(f"                   {endpoint.reason}")

    lines.append("")
    lines.append("Cloud integrations")
    lines.append("  none" if not CLOUD_INTEGRATIONS else "  " + ", ".join(CLOUD_INTEGRATIONS))
    lines.append("Outbound network features")
    lines.append("  none" if not NETWORK_FEATURES else "  " + ", ".join(NETWORK_FEATURES))

    lines.append("")
    lines.append("Executables")
    for name, path in sorted(report.executables.items()):
        marker = "ok  " if path else "MISS"
        required = " (required)" if name in REQUIRED_EXECUTABLES else ""
        lines.append(f"  [{marker}] {name}{required}: {path or 'not found'}")

    lines.append("")
    lines.append("Data paths")
    for name, info in sorted(report.paths.items()):
        marker = "ok  " if info.get("writable") else "FAIL"
        lines.append(f"  [{marker}] {name}: {info.get('path')}")

    lines.append("")
    lines.append("GPU")
    gpu = report.gpu
    lines.append(f"  cuda_available={gpu.get('cuda_available')} torch={gpu.get('torch_installed')}")
    for device in gpu.get("devices", []):
        lines.append(
            f"  device {device.get('index')}: {device.get('name')} "
            f"{device.get('total_memory_mb')}MB"
        )
    if not gpu.get("devices"):
        lines.append("  no GPU reported (not required)")

    lines.append("")
    lines.append("Preprocessing adapters (no weights required or downloaded)")
    for kind, info in sorted(report.adapters.items()):
        lines.append(f"  {kind:<24} {info.get('status')}")

    pose = report.configured_pose
    if pose:
        lines.append("")
        lines.append("Configured pose adapter (model files are never downloaded)")
        lines.append(f"  adapter                  {pose.get('adapter')}")
        lines.append(f"  status                   {pose.get('status')}")
        lines.append(f"  detector model set       {pose.get('detector_model_configured')}")
        lines.append(f"  pose model set           {pose.get('pose_model_configured')}")
        lines.append(f"  provider requested       {pose.get('provider_requested')}")
        if pose.get("status") not in {"available"}:
            lines.append(f"  -> {pose.get('reason', '')}")

    if report.problems:
        lines.append("")
        lines.append("PROBLEMS")
        lines.extend(f"  - {problem}" for problem in report.problems)
    if report.warnings:
        lines.append("")
        lines.append("Warnings")
        lines.extend(f"  - {warning}" for warning in report.warnings)

    lines.append("")
    lines.append("=" * 72)
    return "\n".join(lines) + "\n"


__all__ = [
    "CLOUD_INTEGRATIONS",
    "NETWORK_FEATURES",
    "OPTIONAL_EXECUTABLES",
    "REQUIRED_EXECUTABLES",
    "EndpointReport",
    "OfflineReport",
    "adapter_capability",
    "check_backends",
    "check_comfyui",
    "check_endpoints",
    "check_executables",
    "check_gpu",
    "check_paths",
    "render_text_report",
    "verify_offline",
]
