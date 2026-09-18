"""Repositories: the only code that reads or writes domain records.

Each repository stores the full Pydantic model as canonical JSON in ``payload``
and promotes a handful of fields to real columns for indexing. Reads always go
back through ``model_validate``, so a schema change surfaces immediately rather
than as a mysterious attribute error later.
"""

from __future__ import annotations

import builtins
import json
import sqlite3
from typing import Any

from app.core.errors import ConflictError, NotFoundError
from app.core.hashing import canonical_json
from app.core.ids import utc_now
from app.core.logging import get_logger
from app.db.database import Database
from app.domain.compatibility import CompatibilityReport, CompatibilityState
from app.domain.enums import JobStatus
from app.domain.garment import GarmentAsset
from app.domain.human_template import HumanTemplate
from app.domain.manifest import RenderManifest
from app.domain.render_job import RenderJob

logger = get_logger(__name__)


def _dumps(model: Any) -> str:
    return canonical_json(model.model_dump(mode="json"))


def _loads(row: sqlite3.Row, column: str = "payload") -> dict[str, Any]:
    return json.loads(row[column])


class AuditLog:
    """Append-only audit trail (overrides, ingests, deletions, renders)."""

    def __init__(self, database: Database) -> None:
        self._db = database

    def record(
        self,
        event: str,
        *,
        actor: str | None = None,
        entity_type: str | None = None,
        entity_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._db.execute(
            "INSERT INTO audit_events (occurred_at, actor, event, entity_type, entity_id, details)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                utc_now().isoformat(),
                actor,
                event,
                entity_type,
                entity_id,
                canonical_json(details or {}),
            ),
        )

    def for_entity(self, entity_type: str, entity_id: str) -> list[dict[str, Any]]:
        rows = self._db.query_all(
            "SELECT occurred_at, actor, event, details FROM audit_events"
            " WHERE entity_type = ? AND entity_id = ? ORDER BY id",
            (entity_type, entity_id),
        )
        return [
            {
                "occurred_at": row["occurred_at"],
                "actor": row["actor"],
                "event": row["event"],
                "details": json.loads(row["details"] or "{}"),
            }
            for row in rows
        ]


class TemplateRepository:
    def __init__(self, database: Database) -> None:
        self._db = database

    def save(self, template: HumanTemplate, *, allow_update: bool = True) -> HumanTemplate:
        """Insert or update one (id, version) row.

        ``allow_update`` exists so ingestion can refuse to silently overwrite an
        existing template version; status transitions during ingestion use the
        default.
        """
        existing = self.try_get(template.id, template.version)
        if existing is not None and not allow_update:
            raise ConflictError(
                "Template version already exists",
                template_id=template.id,
                version=template.version,
            )
        record = template.model_copy(update={"updated_at": utc_now()})
        self._db.execute(
            """
            INSERT INTO human_templates (
                id, version, display_name, source_sha256, width, height, fps,
                frame_count, transition_anchor_frame, template_clothing_class,
                status, created_at, updated_at, payload
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id, version) DO UPDATE SET
                display_name = excluded.display_name,
                source_sha256 = excluded.source_sha256,
                width = excluded.width,
                height = excluded.height,
                fps = excluded.fps,
                frame_count = excluded.frame_count,
                transition_anchor_frame = excluded.transition_anchor_frame,
                template_clothing_class = excluded.template_clothing_class,
                status = excluded.status,
                updated_at = excluded.updated_at,
                payload = excluded.payload
            """,
            (
                record.id,
                record.version,
                record.display_name,
                record.source_sha256,
                record.video.width,
                record.video.height,
                record.video.fps,
                record.video.frame_count,
                record.transition_anchor_frame,
                record.template_clothing_class.value,
                record.status.value,
                record.created_at.isoformat(),
                record.updated_at.isoformat() if record.updated_at else None,
                _dumps(record),
            ),
        )
        return record

    def try_get(self, template_id: str, version: int | None = None) -> HumanTemplate | None:
        if version is None:
            row = self._db.query_one(
                "SELECT payload FROM human_templates WHERE id = ?" " ORDER BY version DESC LIMIT 1",
                (template_id,),
            )
        else:
            row = self._db.query_one(
                "SELECT payload FROM human_templates WHERE id = ? AND version = ?",
                (template_id, version),
            )
        return HumanTemplate.model_validate(_loads(row)) if row else None

    def get(self, template_id: str, version: int | None = None) -> HumanTemplate:
        template = self.try_get(template_id, version)
        if template is None:
            raise NotFoundError("Template not found", template_id=template_id, version=version)
        return template

    def list(self, *, limit: int = 100, offset: int = 0) -> builtins.list[HumanTemplate]:
        rows = self._db.query_all(
            "SELECT payload FROM human_templates ORDER BY created_at DESC, id, version"
            " LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return [HumanTemplate.model_validate(_loads(row)) for row in rows]

    def latest_version(self, template_id: str) -> int | None:
        row = self._db.query_one(
            "SELECT MAX(version) AS v FROM human_templates WHERE id = ?", (template_id,)
        )
        return int(row["v"]) if row and row["v"] is not None else None


class GarmentRepository:
    def __init__(self, database: Database) -> None:
        self._db = database

    def save(self, garment: GarmentAsset, *, allow_update: bool = True) -> GarmentAsset:
        existing = self.try_get(garment.id, garment.version)
        if existing is not None and not allow_update:
            raise ConflictError(
                "Garment version already exists", garment_id=garment.id, version=garment.version
            )
        record = garment.model_copy(update={"updated_at": utc_now()})
        self._db.execute(
            """
            INSERT INTO garment_assets (
                id, version, product_name, brand, category, body_coverage,
                status, created_at, updated_at, payload
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id, version) DO UPDATE SET
                product_name = excluded.product_name,
                brand = excluded.brand,
                category = excluded.category,
                body_coverage = excluded.body_coverage,
                status = excluded.status,
                updated_at = excluded.updated_at,
                payload = excluded.payload
            """,
            (
                record.id,
                record.version,
                record.product_name,
                record.brand,
                record.category.value,
                record.body_coverage.value,
                record.status.value,
                record.created_at.isoformat(),
                record.updated_at.isoformat() if record.updated_at else None,
                _dumps(record),
            ),
        )
        return record

    def try_get(self, garment_id: str, version: int | None = None) -> GarmentAsset | None:
        if version is None:
            row = self._db.query_one(
                "SELECT payload FROM garment_assets WHERE id = ? ORDER BY version DESC LIMIT 1",
                (garment_id,),
            )
        else:
            row = self._db.query_one(
                "SELECT payload FROM garment_assets WHERE id = ? AND version = ?",
                (garment_id, version),
            )
        return GarmentAsset.model_validate(_loads(row)) if row else None

    def get(self, garment_id: str, version: int | None = None) -> GarmentAsset:
        garment = self.try_get(garment_id, version)
        if garment is None:
            raise NotFoundError("Garment not found", garment_id=garment_id, version=version)
        return garment

    def list(self, *, limit: int = 100, offset: int = 0) -> builtins.list[GarmentAsset]:
        rows = self._db.query_all(
            "SELECT payload FROM garment_assets ORDER BY created_at DESC, id, version"
            " LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return [GarmentAsset.model_validate(_loads(row)) for row in rows]


class CompatibilityRepository:
    def __init__(self, database: Database) -> None:
        self._db = database

    def save(self, report: CompatibilityReport) -> CompatibilityReport:
        record = report.model_copy(update={"updated_at": utc_now()})
        self._db.execute(
            """
            INSERT INTO compatibility_reports (
                id, template_id, template_version, garment_id, garment_version,
                state, rules_version, confidence, has_override,
                created_at, updated_at, payload
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                state = excluded.state,
                rules_version = excluded.rules_version,
                confidence = excluded.confidence,
                has_override = excluded.has_override,
                updated_at = excluded.updated_at,
                payload = excluded.payload
            """,
            (
                record.id,
                record.template_id,
                record.template_version,
                record.garment_id,
                record.garment_version,
                record.state.value,
                record.rules_version,
                record.confidence,
                1 if record.override else 0,
                record.created_at.isoformat(),
                record.updated_at.isoformat() if record.updated_at else None,
                _dumps(record),
            ),
        )
        return record

    def try_get(self, report_id: str) -> CompatibilityReport | None:
        row = self._db.query_one(
            "SELECT payload FROM compatibility_reports WHERE id = ?", (report_id,)
        )
        return CompatibilityReport.model_validate(_loads(row)) if row else None

    def get(self, report_id: str) -> CompatibilityReport:
        report = self.try_get(report_id)
        if report is None:
            raise NotFoundError("Compatibility report not found", report_id=report_id)
        return report

    def latest_for_pair(
        self,
        template_id: str,
        template_version: int,
        garment_id: str,
        garment_version: int,
    ) -> CompatibilityReport | None:
        row = self._db.query_one(
            "SELECT payload FROM compatibility_reports"
            " WHERE template_id = ? AND template_version = ?"
            "   AND garment_id = ? AND garment_version = ?"
            " ORDER BY created_at DESC, id DESC LIMIT 1",
            (template_id, template_version, garment_id, garment_version),
        )
        return CompatibilityReport.model_validate(_loads(row)) if row else None

    def list_by_state(
        self, state: CompatibilityState, *, limit: int = 100
    ) -> list[CompatibilityReport]:
        rows = self._db.query_all(
            "SELECT payload FROM compatibility_reports WHERE state = ?"
            " ORDER BY created_at DESC LIMIT ?",
            (state.value, limit),
        )
        return [CompatibilityReport.model_validate(_loads(row)) for row in rows]


class JobRepository:
    def __init__(self, database: Database) -> None:
        self._db = database

    def save(self, job: RenderJob) -> RenderJob:
        record = job.model_copy(update={"updated_at": utc_now()})
        self._db.execute(
            """
            INSERT INTO render_jobs (
                id, template_id, template_version, garment_id, garment_version,
                compatibility_report_id, backend_name, backend_version,
                workflow_id, workflow_sha256, seed, frame_start, frame_end,
                status, completed_frames, total_frames,
                created_at, updated_at, started_at, finished_at, payload
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                backend_version = excluded.backend_version,
                workflow_id = excluded.workflow_id,
                workflow_sha256 = excluded.workflow_sha256,
                status = excluded.status,
                completed_frames = excluded.completed_frames,
                total_frames = excluded.total_frames,
                updated_at = excluded.updated_at,
                started_at = excluded.started_at,
                finished_at = excluded.finished_at,
                payload = excluded.payload
            """,
            (
                record.id,
                record.template_id,
                record.template_version,
                record.garment_id,
                record.garment_version,
                record.compatibility_report_id,
                record.backend_name,
                record.backend_version,
                record.workflow_id,
                record.workflow_sha256,
                record.seed,
                record.frame_range.start,
                record.frame_range.end,
                record.status.value,
                record.progress.completed_frames,
                record.progress.total_frames,
                record.created_at.isoformat(),
                record.updated_at.isoformat() if record.updated_at else None,
                record.started_at.isoformat() if record.started_at else None,
                record.finished_at.isoformat() if record.finished_at else None,
                _dumps(record),
            ),
        )
        return record

    def try_get(self, job_id: str) -> RenderJob | None:
        row = self._db.query_one("SELECT payload FROM render_jobs WHERE id = ?", (job_id,))
        return RenderJob.model_validate(_loads(row)) if row else None

    def get(self, job_id: str) -> RenderJob:
        job = self.try_get(job_id)
        if job is None:
            raise NotFoundError("Job not found", job_id=job_id)
        return job

    def list(
        self, *, status: JobStatus | None = None, limit: int = 100, offset: int = 0
    ) -> builtins.list[RenderJob]:
        if status is None:
            rows = self._db.query_all(
                "SELECT payload FROM render_jobs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
        else:
            rows = self._db.query_all(
                "SELECT payload FROM render_jobs WHERE status = ?"
                " ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (status.value, limit, offset),
            )
        return [RenderJob.model_validate(_loads(row)) for row in rows]

    # -- per-frame checkpoint state ---------------------------------------
    def record_frame(
        self,
        job_id: str,
        frame_index: int,
        *,
        status: str,
        frame_sha256: str | None = None,
        frame_seed: int | None = None,
        duration_ms: int | None = None,
        error: str | None = None,
    ) -> None:
        self._db.execute(
            """
            INSERT INTO job_frames (
                job_id, frame_index, status, frame_sha256, frame_seed,
                duration_ms, attempts, error, updated_at
            ) VALUES (?,?,?,?,?,?,1,?,?)
            ON CONFLICT(job_id, frame_index) DO UPDATE SET
                status = excluded.status,
                frame_sha256 = excluded.frame_sha256,
                frame_seed = excluded.frame_seed,
                duration_ms = excluded.duration_ms,
                attempts = job_frames.attempts + 1,
                error = excluded.error,
                updated_at = excluded.updated_at
            """,
            (
                job_id,
                frame_index,
                status,
                frame_sha256,
                frame_seed,
                duration_ms,
                error,
                utc_now().isoformat(),
            ),
        )

    def completed_frames(self, job_id: str) -> builtins.list[int]:
        rows = self._db.query_all(
            "SELECT frame_index FROM job_frames WHERE job_id = ? AND status = 'composited'"
            " ORDER BY frame_index",
            (job_id,),
        )
        return [int(row["frame_index"]) for row in rows]

    def frame_checksums(self, job_id: str) -> dict[int, str]:
        rows = self._db.query_all(
            "SELECT frame_index, frame_sha256 FROM job_frames"
            " WHERE job_id = ? AND frame_sha256 IS NOT NULL ORDER BY frame_index",
            (job_id,),
        )
        return {int(row["frame_index"]): str(row["frame_sha256"]) for row in rows}

    def clear_frames(self, job_id: str) -> int:
        cursor = self._db.execute("DELETE FROM job_frames WHERE job_id = ?", (job_id,))
        return cursor.rowcount or 0

    # -- manifests ---------------------------------------------------------
    def save_manifest(self, manifest: RenderManifest) -> str:
        digest = manifest.reproducibility_digest()
        self._db.execute(
            "INSERT INTO job_manifests (job_id, digest, created_at, payload)"
            " VALUES (?,?,?,?)"
            " ON CONFLICT(job_id) DO UPDATE SET"
            "   digest = excluded.digest, payload = excluded.payload",
            (manifest.job_id, digest, utc_now().isoformat(), _dumps(manifest)),
        )
        return digest

    def get_manifest(self, job_id: str) -> RenderManifest | None:
        row = self._db.query_one("SELECT payload FROM job_manifests WHERE job_id = ?", (job_id,))
        return RenderManifest.model_validate(_loads(row)) if row else None

    def manifest_digest(self, job_id: str) -> str | None:
        row = self._db.query_one("SELECT digest FROM job_manifests WHERE job_id = ?", (job_id,))
        return str(row["digest"]) if row else None


class Repositories:
    """Bundle handed to services so they take one dependency, not five."""

    def __init__(self, database: Database) -> None:
        self.database = database
        self.templates = TemplateRepository(database)
        self.garments = GarmentRepository(database)
        self.compatibility = CompatibilityRepository(database)
        self.jobs = JobRepository(database)
        self.audit = AuditLog(database)


__all__ = [
    "AuditLog",
    "CompatibilityRepository",
    "GarmentRepository",
    "JobRepository",
    "Repositories",
    "TemplateRepository",
]
