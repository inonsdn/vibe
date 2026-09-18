"""Controlled vocabularies shared by schemas, rules and reports.

All enums are ``str``-valued so they serialise directly to JSON/SQLite and can
be referenced verbatim from the YAML rule files.
"""

from __future__ import annotations

from enum import StrEnum


class ProcessingStatus(StrEnum):
    """Lifecycle of a human template."""

    CREATED = "created"
    PROBING = "probing"
    EXTRACTING = "extracting"
    AWAITING_MASKS = "awaiting_masks"
    MASKS_IMPORTED = "masks_imported"
    VALIDATED = "validated"
    READY = "ready"
    FAILED = "failed"


class IngestionStatus(StrEnum):
    """Lifecycle of a garment asset."""

    CREATED = "created"
    IMAGES_IMPORTED = "images_imported"
    METADATA_INCOMPLETE = "metadata_incomplete"
    READY = "ready"
    REJECTED = "rejected"


class TemplateClothingClass(StrEnum):
    """What the master performer is actually wearing in the source footage.

    This drives most compatibility rules: a long-sleeve replacement over a
    sleeveless base leaks skin, and a short base garment under a long
    replacement needs mask expansion.
    """

    SLEEVELESS_MINIMAL = "sleeveless_minimal"
    FITTED_SHORT = "fitted_short"
    FITTED_FULL = "fitted_full"
    LOOSE_SHORT = "loose_short"
    LOOSE_FULL = "loose_full"
    TWO_PIECE = "two_piece"
    FULL_COVERAGE = "full_coverage"
    NEUTRAL_BODYSUIT = "neutral_bodysuit"


class GarmentCategory(StrEnum):
    TOP = "top"
    BOTTOM = "bottom"
    DRESS = "dress"
    OUTERWEAR = "outerwear"
    FULL_OUTFIT = "full_outfit"
    COSTUME = "costume"


class SleeveLength(StrEnum):
    NONE = "none"
    STRAP = "strap"
    CAP = "cap"
    SHORT = "short"
    ELBOW = "elbow"
    THREE_QUARTER = "three_quarter"
    LONG = "long"
    NOT_APPLICABLE = "not_applicable"


class GarmentLength(StrEnum):
    CROP = "crop"
    WAIST = "waist"
    HIP = "hip"
    MID_THIGH = "mid_thigh"
    KNEE = "knee"
    MIDI = "midi"
    ANKLE = "ankle"
    FLOOR = "floor"
    NOT_APPLICABLE = "not_applicable"


class BodyCoverage(StrEnum):
    """How much of the body the garment must cover in the output."""

    MINIMAL = "minimal"
    TORSO = "torso"
    TORSO_ARMS = "torso_arms"
    TORSO_LEGS = "torso_legs"
    FULL_BODY = "full_body"
    FULL_BODY_LIMBS = "full_body_limbs"


class Silhouette(StrEnum):
    BODYCON = "bodycon"
    FITTED = "fitted"
    STRAIGHT = "straight"
    A_LINE = "a_line"
    FLARED = "flared"
    OVERSIZED = "oversized"
    VOLUMINOUS = "voluminous"


class Material(StrEnum):
    COTTON = "cotton"
    DENIM = "denim"
    KNIT = "knit"
    LEATHER = "leather"
    LATEX = "latex"
    SATIN = "satin"
    SILK = "silk"
    SEQUIN = "sequin"
    METALLIC = "metallic"
    MESH = "mesh"
    LACE = "lace"
    TULLE = "tulle"
    CHIFFON = "chiffon"
    WOOL = "wool"
    SYNTHETIC = "synthetic"
    OTHER = "other"


class ImageViewType(StrEnum):
    FRONT = "front"
    BACK = "back"
    SIDE = "side"
    DETAIL = "detail"
    FLAT_LAY = "flat_lay"


class MaskKind(StrEnum):
    """The four mask families stored per template frame."""

    GARMENT = "garment"
    EXPANSION = "expansion"
    PROTECTED = "protected"
    OCCLUSION = "occlusion"


class RuleSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    NEEDS_INPUT = "needs_input"
    BLOCKING = "blocking"


class JobStatus(StrEnum):
    CREATED = "created"
    PREPARING = "preparing"
    RENDERING = "rendering"
    PAUSED = "paused"
    RENDERED = "rendered"
    COMPOSING = "composing"
    COMPOSED = "composed"
    QC_RUNNING = "qc_running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}

    @property
    def is_resumable(self) -> bool:
        return self in {
            JobStatus.CREATED,
            JobStatus.PREPARING,
            JobStatus.RENDERING,
            JobStatus.PAUSED,
            JobStatus.RENDERED,
            JobStatus.COMPOSING,
            JobStatus.COMPOSED,
            JobStatus.QC_RUNNING,
            JobStatus.FAILED,
        }


#: Mask directory name per kind, used by the on-disk layout.
MASK_DIR_NAMES: dict[MaskKind, str] = {
    MaskKind.GARMENT: "masks_garment",
    MaskKind.EXPANSION: "masks_expansion",
    MaskKind.PROTECTED: "masks_protected",
    MaskKind.OCCLUSION: "masks_occlusion",
}

#: Which garment categories require which minimum coverage.
CATEGORY_MIN_COVERAGE: dict[GarmentCategory, BodyCoverage] = {
    GarmentCategory.TOP: BodyCoverage.TORSO,
    GarmentCategory.BOTTOM: BodyCoverage.TORSO_LEGS,
    GarmentCategory.DRESS: BodyCoverage.TORSO_LEGS,
    GarmentCategory.OUTERWEAR: BodyCoverage.TORSO_ARMS,
    GarmentCategory.FULL_OUTFIT: BodyCoverage.FULL_BODY,
    GarmentCategory.COSTUME: BodyCoverage.FULL_BODY,
}

#: Ordered coverage scale, used for "garment needs more coverage than the base
#: template provides" comparisons.
COVERAGE_ORDER: tuple[BodyCoverage, ...] = (
    BodyCoverage.MINIMAL,
    BodyCoverage.TORSO,
    BodyCoverage.TORSO_ARMS,
    BodyCoverage.TORSO_LEGS,
    BodyCoverage.FULL_BODY,
    BodyCoverage.FULL_BODY_LIMBS,
)

#: Coverage the template's own clothing class already provides.
TEMPLATE_CLASS_COVERAGE: dict[TemplateClothingClass, BodyCoverage] = {
    TemplateClothingClass.SLEEVELESS_MINIMAL: BodyCoverage.MINIMAL,
    TemplateClothingClass.FITTED_SHORT: BodyCoverage.TORSO,
    TemplateClothingClass.FITTED_FULL: BodyCoverage.TORSO_LEGS,
    TemplateClothingClass.LOOSE_SHORT: BodyCoverage.TORSO,
    TemplateClothingClass.LOOSE_FULL: BodyCoverage.TORSO_LEGS,
    TemplateClothingClass.TWO_PIECE: BodyCoverage.TORSO,
    TemplateClothingClass.FULL_COVERAGE: BodyCoverage.FULL_BODY,
    TemplateClothingClass.NEUTRAL_BODYSUIT: BodyCoverage.FULL_BODY,
}

#: Ordered sleeve scale for mismatch comparisons.
SLEEVE_ORDER: tuple[SleeveLength, ...] = (
    SleeveLength.NONE,
    SleeveLength.STRAP,
    SleeveLength.CAP,
    SleeveLength.SHORT,
    SleeveLength.ELBOW,
    SleeveLength.THREE_QUARTER,
    SleeveLength.LONG,
)

#: Ordered hem scale for hem/length mismatch comparisons.
LENGTH_ORDER: tuple[GarmentLength, ...] = (
    GarmentLength.CROP,
    GarmentLength.WAIST,
    GarmentLength.HIP,
    GarmentLength.MID_THIGH,
    GarmentLength.KNEE,
    GarmentLength.MIDI,
    GarmentLength.ANKLE,
    GarmentLength.FLOOR,
)

#: Ordered silhouette scale for expansion comparisons.
SILHOUETTE_ORDER: tuple[Silhouette, ...] = (
    Silhouette.BODYCON,
    Silhouette.FITTED,
    Silhouette.STRAIGHT,
    Silhouette.A_LINE,
    Silhouette.FLARED,
    Silhouette.OVERSIZED,
    Silhouette.VOLUMINOUS,
)

#: Materials that behave badly under naive relighting.
REFLECTIVE_MATERIALS: frozenset[Material] = frozenset(
    {Material.LATEX, Material.SEQUIN, Material.METALLIC, Material.SATIN, Material.LEATHER}
)

#: Materials that are semi-transparent by nature.
SHEER_MATERIALS: frozenset[Material] = frozenset(
    {Material.MESH, Material.LACE, Material.TULLE, Material.CHIFFON}
)


def scale_index(scale: tuple[object, ...], value: object) -> int:
    """Index of ``value`` in ``scale``, or ``-1`` when unranked."""
    try:
        return scale.index(value)
    except ValueError:
        return -1


__all__ = [
    "CATEGORY_MIN_COVERAGE",
    "COVERAGE_ORDER",
    "LENGTH_ORDER",
    "MASK_DIR_NAMES",
    "REFLECTIVE_MATERIALS",
    "SHEER_MATERIALS",
    "SILHOUETTE_ORDER",
    "SLEEVE_ORDER",
    "TEMPLATE_CLASS_COVERAGE",
    "BodyCoverage",
    "GarmentCategory",
    "GarmentLength",
    "ImageViewType",
    "IngestionStatus",
    "JobStatus",
    "MaskKind",
    "Material",
    "ProcessingStatus",
    "RuleSeverity",
    "Silhouette",
    "SleeveLength",
    "TemplateClothingClass",
    "scale_index",
]
