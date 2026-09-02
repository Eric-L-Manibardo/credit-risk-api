"""SQLite engine and project data paths.

Production code gets a connection from here. Switching SQLite → PostgreSQL
later is a different URL, not a rewrite of query code.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, create_engine

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DB_PATH = DATA_DIR / "credit.db"

TABLE_NAME = "loans"


def get_engine(db_path: Path | None = None) -> Engine:
    """Return a SQLAlchemy engine for the local SQLite file."""
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{path}")
