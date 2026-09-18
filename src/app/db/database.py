"""SQLite connection management.

Settings chosen deliberately:

* ``WAL`` journal so a long render can write checkpoints while the API reads,
* ``foreign_keys=ON`` so a template cannot be deleted out from under a job,
* ``synchronous=FULL`` because checkpoint durability matters more than speed
  (a few writes per rendered frame is nothing next to the render itself),
* ``detect_types=0`` and explicit ISO-8601 text for timestamps, avoiding
  SQLite's deprecated datetime adapters,
* ``check_same_thread=False`` plus an explicit re-entrant lock, because FastAPI
  runs synchronous endpoints in a thread pool. Writes are serialised through
  that lock rather than pooled: the whole product is a single-operator, one
  render at a time tool, so a lock is the honest model.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from app.core.logging import get_logger
from app.db.migrations_runner import migrate, schema_version

logger = get_logger(__name__)

_PRAGMAS = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA foreign_keys=ON",
    "PRAGMA synchronous=FULL",
    "PRAGMA busy_timeout=10000",
    "PRAGMA temp_store=MEMORY",
)


def connect(path: str | os.PathLike[str] | None) -> sqlite3.Connection:
    """Open a configured connection. ``None`` or ``":memory:"`` for tests."""
    target = ":memory:" if path is None else str(path)
    if target != ":memory:":
        Path(target).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        target,
        isolation_level=None,
        timeout=10.0,
        check_same_thread=False,
    )
    connection.row_factory = sqlite3.Row
    for pragma in _PRAGMAS:
        # An in-memory database cannot use WAL; ignore that one failure.
        try:
            connection.execute(pragma)
        except sqlite3.DatabaseError:  # pragma: no cover - platform dependent
            logger.debug("pragma_skipped", extra={"pragma": pragma})
    return connection


class Database:
    """Thin wrapper giving transactions, row helpers and migration state."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._lock = threading.RLock()

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection

    @classmethod
    def open(cls, path: str | os.PathLike[str] | None, *, run_migrations: bool = True) -> Database:
        connection = connect(path)
        database = cls(connection)
        if run_migrations:
            database.migrate()
        return database

    def migrate(self) -> list[int]:
        with self._lock:
            return migrate(self._connection)

    @property
    def schema_version(self) -> int:
        with self._lock:
            return schema_version(self._connection)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Explicit transaction (``isolation_level=None`` means manual BEGIN)."""
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
            else:
                self._connection.execute("COMMIT")

    def execute(self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._connection.execute(sql, params)

    def query_one(
        self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()
    ) -> sqlite3.Row | None:
        with self._lock:
            return self._connection.execute(sql, params).fetchone()

    def query_all(
        self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()
    ) -> list[sqlite3.Row]:
        with self._lock:
            return self._connection.execute(sql, params).fetchall()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def open_database(path: str | os.PathLike[str] | None, *, run_migrations: bool = True) -> Database:
    return Database.open(path, run_migrations=run_migrations)


__all__ = ["Database", "connect", "open_database"]
