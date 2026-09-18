"""QC check result types and the check registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class CheckSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class CheckResult:
    """One QC check's outcome, with the numbers behind it."""

    check_id: str
    passed: bool
    severity: CheckSeverity
    message: str
    metrics: dict[str, Any] = field(default_factory=dict)
    threshold: dict[str, Any] = field(default_factory=dict)
    #: Frames that triggered the failure, truncated for readability.
    offending_frames: list[int] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str | None = None

    @property
    def blocking(self) -> bool:
        return not self.passed and self.severity is CheckSeverity.ERROR

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "check_id": self.check_id,
            "passed": self.passed,
            "severity": self.severity.value,
            "message": self.message,
            "metrics": self.metrics,
        }
        if self.threshold:
            payload["threshold"] = self.threshold
        if self.offending_frames:
            payload["offending_frames"] = self.offending_frames[:32]
            payload["offending_frame_count"] = len(self.offending_frames)
        if self.skipped:
            payload["skipped"] = True
            payload["skip_reason"] = self.skip_reason
        return payload


def passed(
    check_id: str,
    message: str,
    *,
    metrics: dict[str, Any] | None = None,
    threshold: dict[str, Any] | None = None,
) -> CheckResult:
    return CheckResult(
        check_id=check_id,
        passed=True,
        severity=CheckSeverity.INFO,
        message=message,
        metrics=metrics or {},
        threshold=threshold or {},
    )


def failed(
    check_id: str,
    message: str,
    *,
    severity: CheckSeverity = CheckSeverity.ERROR,
    metrics: dict[str, Any] | None = None,
    threshold: dict[str, Any] | None = None,
    offending_frames: list[int] | None = None,
) -> CheckResult:
    return CheckResult(
        check_id=check_id,
        passed=False,
        severity=severity,
        message=message,
        metrics=metrics or {},
        threshold=threshold or {},
        offending_frames=offending_frames or [],
    )


def skipped(check_id: str, reason: str) -> CheckResult:
    return CheckResult(
        check_id=check_id,
        passed=True,
        severity=CheckSeverity.INFO,
        message=f"skipped: {reason}",
        skipped=True,
        skip_reason=reason,
    )


@dataclass
class QCReport:
    """The aggregate QC result for one job."""

    job_id: str
    checks: list[CheckResult] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    contact_sheets: list[str] = field(default_factory=list)
    generated_at: str = ""

    @property
    def passed(self) -> bool:
        return not any(check.blocking for check in self.checks)

    @property
    def failed_checks(self) -> list[CheckResult]:
        return [check for check in self.checks if not check.passed]

    @property
    def warnings(self) -> list[CheckResult]:
        return [
            check
            for check in self.checks
            if not check.passed and check.severity is CheckSeverity.WARNING
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "generated_at": self.generated_at,
            "passed": self.passed,
            "checks_total": len(self.checks),
            "checks_failed": len(self.failed_checks),
            "blocking_failures": [c.check_id for c in self.checks if c.blocking],
            "warnings": [c.check_id for c in self.warnings],
            "checks": [check.as_dict() for check in self.checks],
            "metrics": self.metrics,
            "contact_sheets": self.contact_sheets,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks_total": len(self.checks),
            "checks_failed": len(self.failed_checks),
            "failed_check_ids": [c.check_id for c in self.failed_checks],
        }


__all__ = ["CheckResult", "CheckSeverity", "QCReport", "failed", "passed", "skipped"]
