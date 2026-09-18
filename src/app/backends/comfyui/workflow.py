"""Named logical inputs -> ComfyUI node/input locations.

Application code must never contain a ComfyUI node id. Instead a *contract*
file declares logical input names (``source_frame``, ``garment_reference``,
``mask``, ``seed``, ``prompt`` ...) and, for each, where in the workflow JSON
its value belongs. Swapping in a different workflow means editing that YAML,
not the Python.

A binding locates a node either by its id *or* — preferred, because ids shift
whenever a graph is edited — by its ``title`` in ``_meta``, which ComfyUI
preserves when you name a node in the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.core.errors import ConfigError, ValidationError
from app.core.hashing import sha256_file, sha256_json
from app.core.logging import get_logger

logger = get_logger(__name__)

#: Logical inputs the pipeline knows how to supply. A contract may use a subset.
KNOWN_LOGICAL_INPUTS: frozenset[str] = frozenset(
    {
        "source_frame",
        "source_frame_window",
        "garment_reference",
        "garment_reference_back",
        "garment_reference_side",
        "mask",
        "expansion_mask",
        "protected_mask",
        "pose",
        "depth",
        "seed",
        "steps",
        "denoise",
        "guidance_scale",
        "prompt",
        "negative_prompt",
        "width",
        "height",
        "frame_index",
        "batch_size",
        "output_prefix",
    }
)


@dataclass(frozen=True)
class WorkflowBinding:
    """Where one logical input lands in the workflow graph."""

    logical_name: str
    input_name: str
    node_title: str | None = None
    node_id: str | None = None
    kind: str = "value"  # value | image_path | image_upload
    required: bool = True
    description: str = ""

    def locate(self, graph: dict[str, Any]) -> str:
        """Resolve this binding to a concrete node id in ``graph``."""
        if self.node_id is not None:
            if self.node_id not in graph:
                raise ValidationError(
                    "Workflow does not contain the bound node id",
                    logical_name=self.logical_name,
                    node_id=self.node_id,
                )
            return self.node_id
        if self.node_title is None:
            raise ConfigError(
                "Binding must specify node_title or node_id",
                logical_name=self.logical_name,
            )
        matches = [
            node_id
            for node_id, node in graph.items()
            if isinstance(node, dict)
            and str(node.get("_meta", {}).get("title", "")) == self.node_title
        ]
        if not matches:
            raise ValidationError(
                "No workflow node carries the bound title",
                logical_name=self.logical_name,
                node_title=self.node_title,
                hint="Title the node in the ComfyUI UI, or bind by node_id.",
            )
        if len(matches) > 1:
            raise ValidationError(
                "Workflow node title is ambiguous",
                logical_name=self.logical_name,
                node_title=self.node_title,
                matches=matches,
            )
        return matches[0]


@dataclass
class WorkflowContract:
    """A workflow plus the mapping from logical inputs to graph locations."""

    workflow_id: str
    description: str
    workflow_file: str
    bindings: tuple[WorkflowBinding, ...]
    output_node_titles: tuple[str, ...] = ()
    requires_model_nodes: bool = True
    contract_path: Path | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def binding(self, logical_name: str) -> WorkflowBinding | None:
        return next((b for b in self.bindings if b.logical_name == logical_name), None)

    def required_inputs(self) -> tuple[str, ...]:
        return tuple(b.logical_name for b in self.bindings if b.required)

    def contract_sha256(self) -> str:
        if self.contract_path is not None and self.contract_path.is_file():
            return sha256_file(self.contract_path)
        return sha256_json(
            {
                "workflow_id": self.workflow_id,
                "bindings": [
                    [b.logical_name, b.input_name, b.node_title, b.node_id, b.kind, b.required]
                    for b in self.bindings
                ],
            }
        )

    def apply(
        self,
        graph: dict[str, Any],
        values: dict[str, Any],
        *,
        strict: bool = True,
    ) -> dict[str, Any]:
        """Return a copy of ``graph`` with ``values`` written into it.

        ``strict`` requires every declared-required binding to have a value, and
        rejects values whose logical name is not in the contract.
        """
        patched = _deep_copy_graph(graph)
        unknown = set(values) - {b.logical_name for b in self.bindings}
        if unknown and strict:
            raise ValidationError(
                "Values supplied for logical inputs the contract does not declare",
                unknown=sorted(unknown),
                declared=sorted(b.logical_name for b in self.bindings),
            )
        for binding in self.bindings:
            if binding.logical_name not in values:
                if binding.required and strict:
                    raise ValidationError(
                        "Missing required workflow input",
                        logical_name=binding.logical_name,
                    )
                continue
            node_id = binding.locate(patched)
            node = patched[node_id]
            node.setdefault("inputs", {})
            node["inputs"][binding.input_name] = values[binding.logical_name]
        return patched

    def validate_against(self, graph: dict[str, Any]) -> list[str]:
        """Return a list of problems binding this contract to ``graph``."""
        problems: list[str] = []
        for binding in self.bindings:
            try:
                binding.locate(graph)
            except (ValidationError, ConfigError) as exc:
                problems.append(f"{binding.logical_name}: {exc.message}")
        for title in self.output_node_titles:
            if not any(
                isinstance(node, dict) and node.get("_meta", {}).get("title") == title
                for node in graph.values()
            ):
                problems.append(f"output node titled {title!r} not found in workflow")
        return problems


def _deep_copy_graph(graph: dict[str, Any]) -> dict[str, Any]:
    import copy

    return copy.deepcopy(graph)


def load_workflow(path: str | Path) -> dict[str, Any]:
    """Load a ComfyUI API-format workflow (a mapping of node id -> node)."""
    import json

    target = Path(path)
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError("Cannot read workflow file", path=str(target)) from exc
    except json.JSONDecodeError as exc:
        raise ConfigError("Workflow file is not valid JSON", path=str(target)) from exc
    if not isinstance(data, dict) or not data:
        raise ConfigError("Workflow must be a non-empty JSON object", path=str(target))
    if "nodes" in data and "links" in data:
        raise ConfigError(
            "This looks like a ComfyUI *editor* workflow; export the API format "
            "instead (Workflow -> Export (API))",
            path=str(target),
        )
    return data


def load_contract(path: str | Path) -> WorkflowContract:
    """Load a workflow contract YAML file."""
    target = Path(path)
    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise ConfigError("Cannot read workflow contract", path=str(target)) from exc
    except yaml.YAMLError as exc:
        raise ConfigError("Workflow contract is not valid YAML", path=str(target)) from exc
    if not isinstance(raw, dict):
        raise ConfigError("Workflow contract must be a mapping", path=str(target))

    bindings: list[WorkflowBinding] = []
    for entry in raw.get("inputs", []) or []:
        if not isinstance(entry, dict):
            raise ConfigError("Each contract input must be a mapping", path=str(target))
        logical = str(entry.get("logical_name", "")).strip()
        if not logical:
            raise ConfigError("Contract input is missing logical_name", path=str(target))
        if logical not in KNOWN_LOGICAL_INPUTS:
            raise ConfigError(
                "Contract declares a logical input the pipeline cannot supply",
                logical_name=logical,
                known=sorted(KNOWN_LOGICAL_INPUTS),
                path=str(target),
            )
        if not entry.get("input_name"):
            raise ConfigError(
                "Contract input is missing input_name", logical_name=logical, path=str(target)
            )
        if not entry.get("node_title") and not entry.get("node_id"):
            raise ConfigError(
                "Contract input needs node_title or node_id",
                logical_name=logical,
                path=str(target),
            )
        bindings.append(
            WorkflowBinding(
                logical_name=logical,
                input_name=str(entry["input_name"]),
                node_title=entry.get("node_title"),
                node_id=str(entry["node_id"]) if entry.get("node_id") is not None else None,
                kind=str(entry.get("kind", "value")),
                required=bool(entry.get("required", True)),
                description=str(entry.get("description", "")),
            )
        )

    workflow_id = str(raw.get("workflow_id") or target.stem)
    return WorkflowContract(
        workflow_id=workflow_id,
        description=str(raw.get("description", "")),
        workflow_file=str(raw.get("workflow_file") or f"{workflow_id}.json"),
        bindings=tuple(bindings),
        output_node_titles=tuple(str(t) for t in raw.get("output_node_titles", []) or []),
        requires_model_nodes=bool(raw.get("requires_model_nodes", True)),
        contract_path=target,
        notes=dict(raw.get("notes", {}) or {}),
    )


def find_contract(workflows_dir: str | Path, workflow_id: str) -> Path:
    """Locate ``<workflows_dir>/<workflow_id>.contract.yaml``."""
    folder = Path(workflows_dir)
    candidate = folder / f"{workflow_id}.contract.yaml"
    if candidate.is_file():
        return candidate
    raise ConfigError(
        "Workflow contract not found",
        workflow_id=workflow_id,
        expected_path=str(candidate),
        available=sorted(p.name for p in folder.glob("*.contract.yaml")) if folder.is_dir() else [],
    )


__all__ = [
    "KNOWN_LOGICAL_INPUTS",
    "WorkflowBinding",
    "WorkflowContract",
    "find_contract",
    "load_contract",
    "load_workflow",
]
