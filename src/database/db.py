"""SQLite schema for jobs — deliberately small: one project = one video for now.
Grows into a real queue once we have more than one provider fighting for GPU time."""
from __future__ import annotations

import sqlite3

from src.utils.config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id   TEXT NOT NULL,
    job_type     TEXT NOT NULL,          -- 'script' | 'tts' | 'animation' | 'render'
    provider     TEXT NOT NULL,          -- 'groq' | 'kokoro' | 'manim' | 'ffmpeg'
    status       TEXT NOT NULL CHECK (status IN ('queued','running','success','failed','retrying')),
    retry_count  INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    started_at   TEXT,
    finished_at  TEXT,
    output_path  TEXT,
    error        TEXT
);
"""


def get_connection() -> sqlite3.Connection:
    settings = get_settings()
    db_path = settings.data_dir / "jobs.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    conn = get_connection()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()
