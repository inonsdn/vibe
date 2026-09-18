"""Strict domain schemas (Pydantic v2).

These models are the data contracts of the system. Every model forbids extra
fields so a typo in a hand-edited JSON sidecar fails loudly instead of being
silently ignored.
"""

from app.domain.compatibility import (
    CompatibilityReport,
    CompatibilityState,
    ReviewerOverride,
    RuleOutcome,
    RuleResult,
)
from app.domain.enums import (
    BodyCoverage,
    GarmentCategory,
    GarmentLength,
    ImageViewType,
    IngestionStatus,
    JobStatus,
    MaskKind,
    Material,
    ProcessingStatus,
    RuleSeverity,
    Silhouette,
    SleeveLength,
    TemplateClothingClass,
)
from app.domain.garment import GarmentAsset, GarmentReferenceImage, UsageRights
from app.domain.human_template import (
    ConsentRecord,
    FrameRange,
    HumanTemplate,
    TemplateDirectories,
    VideoSpec,
)
from app.domain.manifest import (
    FrameChecksum,
    QCSummary,
    RenderManifest,
    ReproducibilityBlock,
)
from app.domain.render_job import (
    Checkpoint,
    JobArtifacts,
    JobError,
    JobProgress,
    RenderJob,
    RenderSettings,
)

__all__ = [
    "BodyCoverage",
    "Checkpoint",
    "CompatibilityReport",
    "CompatibilityState",
    "ConsentRecord",
    "FrameChecksum",
    "FrameRange",
    "GarmentAsset",
    "GarmentCategory",
    "GarmentLength",
    "GarmentReferenceImage",
    "HumanTemplate",
    "ImageViewType",
    "IngestionStatus",
    "JobArtifacts",
    "JobError",
    "JobProgress",
    "JobStatus",
    "MaskKind",
    "Material",
    "ProcessingStatus",
    "QCSummary",
    "RenderJob",
    "RenderManifest",
    "RenderSettings",
    "ReproducibilityBlock",
    "ReviewerOverride",
    "RuleOutcome",
    "RuleResult",
    "RuleSeverity",
    "Silhouette",
    "SleeveLength",
    "TemplateClothingClass",
    "TemplateDirectories",
    "UsageRights",
    "VideoSpec",
]
