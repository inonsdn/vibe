"""Typed application errors.

Every error carries a short, stable ``code`` so the CLI, API and logs can all
report the same machine-readable reason.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class for all expected (non-bug) failures."""

    code = "app_error"
    http_status = 500

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}

    def __str__(self) -> str:  # pragma: no cover - trivial
        if self.details:
            return f"{self.message} ({self.details})"
        return self.message


class ConfigError(AppError):
    code = "config_error"
    http_status = 500


class ValidationError(AppError):
    code = "validation_error"
    http_status = 422


class PathSecurityError(ValidationError):
    """Raised when a path would escape the configured data root."""

    code = "path_security_error"
    http_status = 400


class NotFoundError(AppError):
    code = "not_found"
    http_status = 404


class ConflictError(AppError):
    code = "conflict"
    http_status = 409


class ImmutabilityError(AppError):
    """Raised on any attempt to modify an immutable artifact."""

    code = "immutability_violation"
    http_status = 409


class MediaToolError(AppError):
    """ffmpeg/ffprobe missing or failing."""

    code = "media_tool_error"
    http_status = 500


class BackendError(AppError):
    code = "backend_error"
    http_status = 502


class BackendUnavailableError(BackendError):
    code = "backend_unavailable"
    http_status = 503


class OfflinePolicyError(AppError):
    """Raised when a configured endpoint is not local."""

    code = "offline_policy_violation"
    http_status = 400


class CompatibilityBlockedError(AppError):
    """Raised when rendering is attempted for a non-READY compatibility state."""

    code = "compatibility_blocked"
    http_status = 409


class MaskError(ValidationError):
    code = "mask_error"


class QCError(AppError):
    code = "qc_error"


__all__ = [
    "AppError",
    "BackendError",
    "BackendUnavailableError",
    "CompatibilityBlockedError",
    "ConfigError",
    "ConflictError",
    "ImmutabilityError",
    "MaskError",
    "MediaToolError",
    "NotFoundError",
    "OfflinePolicyError",
    "PathSecurityError",
    "QCError",
    "ValidationError",
]
