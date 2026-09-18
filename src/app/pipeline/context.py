"""Service context: the single object CLI and API both build.

Holding config, data root, database and repositories together is what lets
``app/cli`` and ``app/api`` be thin: each one constructs a
:class:`ServiceContext` and calls the same pipeline functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import AppConfig, load_config
from app.core.logging import configure_logging, get_logger
from app.core.paths import DataRoot
from app.db.database import Database
from app.db.repositories import Repositories

logger = get_logger(__name__)


@dataclass
class ServiceContext:
    """Everything a pipeline function needs, constructed once per process."""

    config: AppConfig
    data_root: DataRoot
    database: Database
    repos: Repositories

    @classmethod
    def create(
        cls,
        *,
        config: AppConfig | None = None,
        config_file: str | Path | None = None,
        overrides: dict[str, Any] | None = None,
        db_path: str | Path | None = None,
        configure_logs: bool = True,
        ensure_dirs: bool = True,
    ) -> ServiceContext:
        resolved = config or load_config(config_file, overrides=overrides)
        data_root = resolved.data_root()
        if ensure_dirs:
            data_root.ensure()
        if configure_logs:
            configure_logging(
                resolved.runtime.log_level,
                log_dir=(data_root.resolve("logs") if resolved.runtime.log_to_file else None),
            )
        target = db_path if db_path is not None else data_root.db_path(resolved.paths.db_filename)
        database = Database.open(target)
        return cls(
            config=resolved,
            data_root=data_root,
            database=database,
            repos=Repositories(database),
        )

    def close(self) -> None:
        self.database.close()

    def __enter__(self) -> ServiceContext:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- path helpers (all traversal-checked) -----------------------------
    def absolute(self, relative: str | Path) -> Path:
        """Resolve a data-root-relative path stored in a record."""
        return self.data_root.resolve(relative)

    def relative(self, absolute: str | Path) -> str:
        """Convert an absolute path back to its stored, portable form."""
        return self.data_root.relativize(absolute).as_posix()


__all__ = ["ServiceContext"]
