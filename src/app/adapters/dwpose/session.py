"""ONNX Runtime session creation, provider resolution and honest reporting.

Three properties this module exists to guarantee:

**Nothing is downloaded.** A session is created from a local file path. If the
file is not there, the error names the path and tells the operator what to put
there — it never reaches for a hub.

**CUDA is preferred, CPU is a fallback, and the difference is recorded.** A run
that quietly dropped to CPU would look identical in the output and take twenty
times as long, so :class:`ProviderResolution` carries what was requested, what
onnxruntime actually offered, and what the session reports it is using. That
goes into the motion source record and from there into the manifest.

**onnxruntime is optional at import time.** It is imported lazily inside the
factory, so the whole application — and the whole test suite — runs without it
installed. Tests inject a fake factory instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from app.core.errors import AppError
from app.core.logging import get_logger

logger = get_logger(__name__)

CUDA_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"


class ModelFileMissingError(AppError):
    """A configured model path does not point at a readable file."""

    code = "model_file_missing"
    http_status = 400


class RuntimeMissingError(AppError):
    """onnxruntime is not installed in this environment."""

    code = "onnx_runtime_missing"
    http_status = 501


class ProviderUnavailableError(AppError):
    """CUDA was required but onnxruntime does not offer it."""

    code = "onnx_provider_unavailable"
    http_status = 400


@runtime_checkable
class OnnxSession(Protocol):
    """The slice of ``onnxruntime.InferenceSession`` this package uses."""

    def run(self, output_names: list[str] | None, input_feed: dict[str, Any]) -> list[Any]: ...

    def get_inputs(self) -> list[Any]: ...

    def get_outputs(self) -> list[Any]: ...

    def get_providers(self) -> list[str]: ...


@runtime_checkable
class OnnxSessionFactory(Protocol):
    """Creates sessions. Injected, so tests never need onnxruntime."""

    def available_providers(self) -> list[str]: ...

    def create(self, model_path: Path, providers: list[str]) -> OnnxSession: ...

    @property
    def runtime_version(self) -> str: ...


@dataclass(frozen=True)
class ProviderResolution:
    """What was asked for, what was possible, and what is actually running."""

    requested: str
    #: Providers handed to onnxruntime, in priority order.
    offered: tuple[str, ...]
    #: Everything the installed onnxruntime build supports.
    available: tuple[str, ...]
    #: What the created session reports. Authoritative.
    active: str
    fell_back: bool
    runtime_version: str = "unknown"
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def using_cuda(self) -> bool:
        return self.active == CUDA_PROVIDER

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "offered": list(self.offered),
            "available": list(self.available),
            "active": self.active,
            "using_cuda": self.using_cuda,
            "fell_back_to_cpu": self.fell_back,
            "onnxruntime_version": self.runtime_version,
            **({"notes": self.notes} if self.notes else {}),
        }


def resolve_providers(requested: str, available: list[str] | tuple[str, ...]) -> list[str]:
    """Provider priority list for a request of ``auto``, ``cuda`` or ``cpu``.

    ``auto`` and ``cuda`` both put CUDA first *when the build has it*; CPU is
    always appended so a session can still be created. The difference between
    the two is what happens when CUDA is absent, and that is the caller's
    decision (``require_requested_provider``), not this function's.
    """
    have = list(available)
    if requested == "cpu":
        return [CPU_PROVIDER]
    providers: list[str] = []
    if CUDA_PROVIDER in have:
        providers.append(CUDA_PROVIDER)
    providers.append(CPU_PROVIDER)
    return providers


def assert_model_file(path: str | Path, *, role: str, hint: str) -> Path:
    """Validate a configured model path, with an error an operator can act on."""
    text = str(path or "").strip()
    if not text:
        raise ModelFileMissingError(
            f"No {role} model is configured. This application never downloads "
            "weights; point it at a file you have already placed on disk.",
            role=role,
            hint=hint,
        )
    resolved = Path(text).expanduser()
    if not resolved.is_file():
        raise ModelFileMissingError(
            f"The configured {role} model file does not exist.",
            role=role,
            path=str(resolved),
            hint=hint,
        )
    if resolved.suffix.lower() != ".onnx":
        raise ModelFileMissingError(
            f"The configured {role} model is not an .onnx file.",
            role=role,
            path=str(resolved),
            suffix=resolved.suffix,
            hint=hint,
        )
    return resolved


class OnnxRuntimeSessionFactory:
    """The real factory. Imports onnxruntime lazily and never at module scope."""

    def __init__(self, *, intra_op_threads: int = 0) -> None:
        self._intra_op_threads = intra_op_threads
        self._module: Any = None

    # -- runtime ----------------------------------------------------------
    def _runtime(self) -> Any:
        if self._module is None:
            try:
                import onnxruntime
            except ImportError as exc:
                raise RuntimeMissingError(
                    "onnxruntime is not installed. Install onnxruntime-gpu (CUDA) "
                    "or onnxruntime (CPU) in this environment; nothing is "
                    "installed automatically.",
                    hint="pip install onnxruntime-gpu==<version tested on your box>",
                ) from exc
            self._module = onnxruntime
        return self._module

    @property
    def runtime_version(self) -> str:
        try:
            return str(self._runtime().__version__)
        except RuntimeMissingError:
            return "not_installed"

    def available_providers(self) -> list[str]:
        return list(self._runtime().get_available_providers())

    def create(self, model_path: Path, providers: list[str]) -> OnnxSession:
        runtime = self._runtime()
        options = runtime.SessionOptions()
        if self._intra_op_threads:
            options.intra_op_num_threads = self._intra_op_threads
        options.graph_optimization_level = runtime.GraphOptimizationLevel.ORT_ENABLE_ALL
        session: OnnxSession = runtime.InferenceSession(
            str(model_path), sess_options=options, providers=list(providers)
        )
        return session


def create_session(
    factory: OnnxSessionFactory,
    model_path: Path,
    *,
    requested: str,
    require_requested: bool,
    role: str,
) -> tuple[OnnxSession, ProviderResolution]:
    """Create one session and report the provider it really ended up on."""
    available = tuple(factory.available_providers())
    if requested == "cuda" and CUDA_PROVIDER not in available and require_requested:
        raise ProviderUnavailableError(
            "CUDA was requested but this onnxruntime build does not offer "
            "CUDAExecutionProvider. Install onnxruntime-gpu matching your CUDA "
            "runtime, or pass --provider cpu to accept the slow path.",
            role=role,
            available=list(available),
        )
    offered = resolve_providers(requested, available)
    session = factory.create(model_path, offered)

    reported = list(session.get_providers())
    active = reported[0] if reported else CPU_PROVIDER
    resolution = ProviderResolution(
        requested=requested,
        offered=tuple(offered),
        available=available,
        active=active,
        fell_back=active != CUDA_PROVIDER and requested in {"auto", "cuda"},
        runtime_version=factory.runtime_version,
    )
    if resolution.fell_back:
        logger.warning(
            "onnx_provider_fell_back",
            extra={
                "event": "onnx_provider_fell_back",
                "role": role,
                "requested": requested,
                "active": active,
                "available": list(available),
            },
        )
    return session, resolution


__all__ = [
    "CPU_PROVIDER",
    "CUDA_PROVIDER",
    "ModelFileMissingError",
    "OnnxRuntimeSessionFactory",
    "OnnxSession",
    "OnnxSessionFactory",
    "ProviderResolution",
    "ProviderUnavailableError",
    "RuntimeMissingError",
    "assert_model_file",
    "create_session",
    "resolve_providers",
]
