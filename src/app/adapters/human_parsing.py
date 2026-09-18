"""Human body-part / clothing parsing.

Purpose in this system: derive the *protected* mask (face, hair, hands, exposed
skin) and an initial garment region from a semantic segmentation of the body.
The protected mask is the single most safety-critical artefact in the pipeline,
so a candidate model is judged primarily on its recall around hands, hair
strands and skin/garment boundaries — a false negative there leaks edits onto
the performer's body.

Output contract: a label PNG per frame plus a JSON legend mapping label values
to part names, written into ``output_dir``. The ingestion step converts labels
into the four mask families.
"""

from __future__ import annotations

from app.adapters.base import (
    AdapterKind,
    AnalysisAdapter,
    NotImplementedAdapter,
    registry,
)


class HumanParsingAdapter(AnalysisAdapter):
    """Interface a future implementation must satisfy."""

    kind = AdapterKind.HUMAN_PARSING


class _HumanParsingAdapterStub(NotImplementedAdapter, HumanParsingAdapter):
    """Documented placeholder; :meth:`run` raises rather than faking output."""


human_parsing_stub = _HumanParsingAdapterStub(
    AdapterKind.HUMAN_PARSING,
    "human-parsing",
    reason="No human-parsing model has been selected or installed yet.",
    expected_outputs=("frame_{index:06d}.png (label map)", "legend.json (label -> part name)"),
    requires_gpu=True,
    estimated_vram_mb=2048,
    candidate_models=("SCHP", "Graphonomy", "ATR/LIP-trained parsers", "Sapiens-seg"),
    integration_notes=(
        "Judged on recall around hands, hair and skin boundaries; a miss "
        "there becomes a visible identity edit."
    ),
)

registry.register(human_parsing_stub)

__all__ = ["HumanParsingAdapter", "human_parsing_stub"]
