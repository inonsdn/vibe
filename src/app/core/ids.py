"""Identifier generation.

Identifiers are readable, sortable and filesystem-safe: a UTC timestamp prefix
plus random suffix. They are validated by :func:`app.core.paths.safe_identifier`
before ever becoming a directory name.
"""

from __future__ import annotations

import re
import secrets
import unicodedata
from datetime import UTC, datetime

_SLUG_RE = re.compile(r"[^a-z0-9]+")
ID_PATTERN = re.compile(r"^[a-z]{2,6}_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{6}$")


def utc_now() -> datetime:
    return datetime.now(UTC)


def timestamp_token(moment: datetime | None = None) -> str:
    return (moment or utc_now()).strftime("%Y%m%dT%H%M%SZ")


def new_id(prefix: str, *, moment: datetime | None = None, entropy: int = 3) -> str:
    """Build an identifier such as ``tpl_20260917T120000Z_9f3a1c``."""
    clean = _SLUG_RE.sub("", prefix.lower())[:6] or "id"
    return f"{clean}_{timestamp_token(moment)}_{secrets.token_hex(entropy)}"


def template_id(moment: datetime | None = None) -> str:
    return new_id("tpl", moment=moment)


def garment_id(moment: datetime | None = None) -> str:
    return new_id("grm", moment=moment)


def job_id(moment: datetime | None = None) -> str:
    return new_id("job", moment=moment)


def report_id(moment: datetime | None = None) -> str:
    return new_id("cmp", moment=moment)


def slugify(value: str, *, max_length: int = 48, fallback: str = "item") -> str:
    """ASCII slug suitable for filenames."""
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    slug = _SLUG_RE.sub("-", normalized.lower()).strip("-")
    return (slug[:max_length].strip("-")) or fallback


__all__ = [
    "ID_PATTERN",
    "garment_id",
    "job_id",
    "new_id",
    "report_id",
    "slugify",
    "template_id",
    "timestamp_token",
    "utc_now",
]
