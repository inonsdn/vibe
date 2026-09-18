"""Adapter stubs must be honest: interfaces, not fake implementations."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.adapters import registry
from app.adapters.base import (
    AdapterKind,
    AdapterNotAvailableError,
    AdapterRegistry,
    AdapterStatus,
    NotImplementedAdapter,
)


def test_every_adapter_kind_is_registered() -> None:
    for kind in AdapterKind:
        assert registry.get(kind) is not None, f"{kind.value} has no adapter"


def test_no_adapter_claims_to_be_available() -> None:
    """Requirement: unfinished integrations are interfaces, not fakes."""
    for capability in registry.capabilities():
        assert capability.status is AdapterStatus.NOT_IMPLEMENTED
        assert not capability.available
        assert capability.reason


def test_running_a_stub_raises_rather_than_fabricating_output(tmp_path: Path) -> None:
    for kind in AdapterKind:
        adapter = registry.require(kind)
        with pytest.raises(AdapterNotAvailableError):
            adapter.run(frames_dir=tmp_path, output_dir=tmp_path, frame_indices=[0])


def test_capabilities_document_the_output_contract() -> None:
    for capability in registry.capabilities():
        assert capability.expected_outputs, f"{capability.kind} documents no outputs"
        assert capability.notes["candidate_models"]
        assert "manual_alternative" in capability.notes


def test_sam2_adapter_exposes_a_propagation_entry_point(tmp_path: Path) -> None:
    from app.adapters import sam2_stub

    with pytest.raises(AdapterNotAvailableError):
        sam2_stub.propagate(
            frames_dir=tmp_path,
            output_dir=tmp_path,
            frame_indices=[0, 1],
            keyframe_prompts={0: {"points": []}},
        )


def test_registry_require_raises_for_unregistered_kind() -> None:
    empty = AdapterRegistry()
    with pytest.raises(AdapterNotAvailableError):
        empty.require(AdapterKind.POSE)


def test_registry_as_dict_is_serialisable() -> None:
    payload = registry.as_dict()
    assert set(payload) == {kind.value for kind in AdapterKind}
    assert all("estimated_vram_mb" in entry for entry in payload.values())


def test_vram_estimates_fit_an_8gb_budget() -> None:
    """Each adapter's estimate must be plausible for an RTX 5060 8GB."""
    for capability in registry.capabilities():
        assert capability.estimated_vram_mb is not None
        assert 0 < capability.estimated_vram_mb <= 8192


def test_custom_stub_reports_its_reason() -> None:
    stub = NotImplementedAdapter(
        AdapterKind.DEPTH,
        "test-depth",
        reason="not chosen yet",
        expected_outputs=("x.png",),
    )
    capability = stub.capability()
    assert capability.reason == "not chosen yet"
    with pytest.raises(AdapterNotAvailableError):
        stub.require_available()
