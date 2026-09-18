"""Shared Pydantic base classes and field types."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.ids import utc_now

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
OptionalSha256 = Annotated[str | None, Field(default=None, pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
NonEmptyStr = Annotated[str, Field(min_length=1, max_length=4096)]


class DomainModel(BaseModel):
    """Strict base: unknown fields are an error, enums serialise as values."""

    model_config = ConfigDict(
        extra="forbid",
        use_enum_values=False,
        validate_default=True,
        str_strip_whitespace=True,
        ser_json_timedelta="float",
    )

    def to_json_dict(self) -> dict[str, Any]:
        """JSON-ready dict (enums as strings, datetimes as ISO-8601)."""
        return self.model_dump(mode="json")


class TimestampedModel(DomainModel):
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime | None = None

    @field_validator("created_at", "updated_at")
    @classmethod
    def _require_tz(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware (UTC)")
        return value


__all__ = [
    "DomainModel",
    "Identifier",
    "NonEmptyStr",
    "OptionalSha256",
    "Sha256",
    "TimestampedModel",
]
