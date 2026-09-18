"""Adapter contract shared by all preprocessing components."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from app.core.errors import AppError
from app.core.logging import get_logger

logger = get_logger(__name__)


class AdapterKind(StrEnum):
    VIDEO_MASK_PROPAGATION = "video_mask_propagation"
    HUMAN_PARSING = "human_parsing"
    POSE = "pose"
    DENSEPOSE = "densepose"
    DEPTH = "depth"
    OPTICAL_FLOW = "optical_flow"
    FACE_LANDMARKS = "face_landmarks"


class AdapterStatus(StrEnum):
    AVAILABLE = "available"
    NOT_IMPLEMENTED = "not_implemented"
    MISSING_WEIGHTS = "missing_weights"
    MISSING_RUNTIME = "missing_runtime"
    ERROR = "error"


class AdapterNotAvailableError(AppError):
    code = "adapter_not_available"
    http_status = 501


@dataclass
class AdapterCapability:
    """Honest self-report used by ``app doctor`` and ``app offline verify``."""

    kind: AdapterKind
    name: str
    status: AdapterStatus
    reason: str
    requires_gpu: bool = False
    estimated_vram_mb: int | None = None
    expected_outputs: tuple[str, ...] = ()
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.status is AdapterStatus.AVAILABLE

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "name": self.name,
            "status": self.status.value,
            "available": self.available,
            "reason": self.reason,
            "requires_gpu": self.requires_gpu,
            "estimated_vram_mb": self.estimated_vram_mb,
            "expected_outputs": list(self.expected_outputs),
            "notes": self.notes,
        }


class AnalysisAdapter(abc.ABC):
    """Base class for every preprocessing adapter.

    Implementations must be *pure producers*: they read the immutable source
    frames and write into their own output directory. They must never modify
    ``source_frames`` — the ingestion pipeline enforces this by hashing the
    directory before and after any adapter run.
    """

    kind: AdapterKind
    name: str = "unnamed"

    @abc.abstractmethod
    def capability(self) -> AdapterCapability:
        """Report whether this adapter can run right now, and why not."""

    @abc.abstractmethod
    def run(
        self,
        *,
        frames_dir: Path,
        output_dir: Path,
        frame_indices: list[int],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Produce artefacts for ``frame_indices`` into ``output_dir``."""

    def available(self) -> bool:
        return self.capability().available

    def require_available(self) -> None:
        capability = self.capability()
        if not capability.available:
            raise AdapterNotAvailableError(
                f"Adapter '{capability.name}' is not available: {capability.reason}",
                kind=capability.kind.value,
                status=capability.status.value,
            )


class NotImplementedAdapter(AnalysisAdapter):
    """A documented placeholder: declares its contract, refuses to pretend.

    This is deliberately *not* a fake implementation. Calling :meth:`run`
    raises, so no pipeline can accidentally treat synthetic data as real model
    output. See ``docs/model-selection-checklist.md``.
    """

    def __init__(
        self,
        kind: AdapterKind,
        name: str,
        *,
        reason: str,
        expected_outputs: tuple[str, ...],
        requires_gpu: bool = True,
        estimated_vram_mb: int | None = None,
        candidate_models: tuple[str, ...] = (),
        integration_notes: str = "",
    ) -> None:
        self.kind = kind
        self.name = name
        self._reason = reason
        self._expected_outputs = expected_outputs
        self._requires_gpu = requires_gpu
        self._estimated_vram_mb = estimated_vram_mb
        self._candidate_models = candidate_models
        self._integration_notes = integration_notes

    def capability(self) -> AdapterCapability:
        return AdapterCapability(
            kind=self.kind,
            name=self.name,
            status=AdapterStatus.NOT_IMPLEMENTED,
            reason=self._reason,
            requires_gpu=self._requires_gpu,
            estimated_vram_mb=self._estimated_vram_mb,
            expected_outputs=self._expected_outputs,
            notes={
                "candidate_models": list(self._candidate_models),
                "integration_notes": self._integration_notes,
                "manual_alternative": (
                    "Author the artefacts externally and import them with "
                    "`app template import-masks`."
                ),
            },
        )

    def run(
        self,
        *,
        frames_dir: Path,
        output_dir: Path,
        frame_indices: list[int],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.require_available()  # always raises: status is NOT_IMPLEMENTED
        raise AssertionError("unreachable")  # pragma: no cover


class AdapterRegistry:
    """Lookup by kind. One adapter per kind; later registration wins."""

    def __init__(self) -> None:
        self._adapters: dict[AdapterKind, AnalysisAdapter] = {}

    def register(self, adapter: AnalysisAdapter) -> None:
        self._adapters[adapter.kind] = adapter

    def get(self, kind: AdapterKind) -> AnalysisAdapter | None:
        return self._adapters.get(kind)

    def require(self, kind: AdapterKind) -> AnalysisAdapter:
        adapter = self.get(kind)
        if adapter is None:
            raise AdapterNotAvailableError(
                f"No adapter registered for {kind.value}", kind=kind.value
            )
        return adapter

    def capabilities(self) -> list[AdapterCapability]:
        return [self._adapters[kind].capability() for kind in AdapterKind if kind in self._adapters]

    def as_dict(self) -> dict[str, Any]:
        return {cap.kind.value: cap.as_dict() for cap in self.capabilities()}


#: Process-wide registry, populated with stubs at import time.
registry = AdapterRegistry()


__all__ = [
    "AdapterCapability",
    "AdapterKind",
    "AdapterNotAvailableError",
    "AdapterRegistry",
    "AdapterStatus",
    "AnalysisAdapter",
    "NotImplementedAdapter",
    "registry",
]
