"""Forward-only SQL migrations.

Migration files live in ``src/app/db/migrations`` and are named
``NNNN_description.sql``. Each file is applied exactly once, inside a
transaction, and recorded in ``schema_migrations`` together with the SHA-256 of
its text. If a previously applied file changes on disk, migration aborts: that
is almost always an accident, and silently ignoring it would make the schema
untrustworthy.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from app.core.errors import ConflictError, ValidationError
from app.core.hashing import sha256_text
from app.core.ids import utc_now
from app.core.logging import get_logger

logger = get_logger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
_FILENAME_RE = re.compile(r"^(\d{4})_([a-z0-9_]+)\.sql$")

_BOOTSTRAP = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    sha256     TEXT NOT NULL,
    applied_at TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    path: Path
    sql: str

    @property
    def sha256(self) -> str:
        return sha256_text(self.sql)


def available_migrations(directory: Path | None = None) -> list[Migration]:
    """Discover migration files, ordered by version, rejecting duplicates."""
    folder = directory or MIGRATIONS_DIR
    migrations: list[Migration] = []
    seen: dict[int, str] = {}
    for path in sorted(folder.glob("*.sql")):
        match = _FILENAME_RE.match(path.name)
        if not match:
            raise ValidationError(
                "Migration filename must look like 0001_snake_case.sql",
                filename=path.name,
            )
        version = int(match.group(1))
        if version in seen:
            raise ValidationError(
                "Duplicate migration version",
                version=version,
                files=[seen[version], path.name],
            )
        seen[version] = path.name
        migrations.append(
            Migration(
                version=version,
                name=match.group(2),
                path=path,
                sql=path.read_text(encoding="utf-8"),
            )
        )
    return migrations


def applied_migrations(connection: sqlite3.Connection) -> dict[int, dict[str, str]]:
    connection.executescript(_BOOTSTRAP)
    rows = connection.execute(
        "SELECT version, name, sha256, applied_at FROM schema_migrations ORDER BY version"
    ).fetchall()
    return {
        int(row["version"]): {
            "name": row["name"],
            "sha256": row["sha256"],
            "applied_at": row["applied_at"],
        }
        for row in rows
    }


def migrate(connection: sqlite3.Connection, *, directory: Path | None = None) -> list[int]:
    """Apply pending migrations; returns the versions applied in this call."""
    already = applied_migrations(connection)
    applied_now: list[int] = []

    for migration in available_migrations(directory):
        record = already.get(migration.version)
        if record is not None:
            if record["sha256"] != migration.sha256:
                raise ConflictError(
                    "An already-applied migration file has changed on disk; "
                    "create a new migration instead of editing history",
                    version=migration.version,
                    name=migration.name,
                    recorded_sha256=record["sha256"],
                    file_sha256=migration.sha256,
                )
            continue
        logger.info(
            "migration_apply",
            extra={
                "event": "migration_apply",
                "version": migration.version,
                "migration_name": migration.name,
            },
        )
        with connection:  # commit/rollback around each migration
            connection.executescript(migration.sql)
            connection.execute(
                "INSERT INTO schema_migrations (version, name, sha256, applied_at)"
                " VALUES (?, ?, ?, ?)",
                (
                    migration.version,
                    migration.name,
                    migration.sha256,
                    utc_now().isoformat(),
                ),
            )
        applied_now.append(migration.version)
    return applied_now


def schema_version(connection: sqlite3.Connection) -> int:
    applied = applied_migrations(connection)
    return max(applied) if applied else 0


__all__ = [
    "MIGRATIONS_DIR",
    "Migration",
    "applied_migrations",
    "available_migrations",
    "migrate",
    "schema_version",
]
