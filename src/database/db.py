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

-- Phase 1D: local project lifecycle registry — see
-- src/core/project_state_machine.py (transition rules) and
-- src/database/project_repository.py (this schema's read/write API).
-- Additive only: the jobs table above is untouched.
CREATE TABLE IF NOT EXISTS projects (
    project_id             TEXT PRIMARY KEY,
    manifest_path          TEXT NOT NULL,
    manifest_fingerprint   TEXT NOT NULL,
    current_stage          TEXT NOT NULL,
    last_successful_stage  TEXT NOT NULL,
    failed_stage           TEXT,
    failure_message        TEXT,
    lifecycle_version      INTEGER NOT NULL,
    created_at             TEXT NOT NULL,
    updated_at             TEXT NOT NULL,
    completed_at           TEXT,
    archived_at            TEXT,
    retry_count            INTEGER NOT NULL DEFAULT 0,
    execution_status       TEXT NOT NULL
);

-- Append-only audit trail of every stage change. transition_id's
-- autoincrement order is the deterministic chronological/event order
-- list_project_transitions() reads back in.
CREATE TABLE IF NOT EXISTS project_transitions (
    transition_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id         TEXT NOT NULL REFERENCES projects (project_id),
    from_stage         TEXT NOT NULL,
    to_stage           TEXT NOT NULL,
    occurred_at        TEXT NOT NULL,
    reason             TEXT,
    is_retry           INTEGER NOT NULL,
    lifecycle_version  INTEGER NOT NULL
);

-- Phase 2B: local artifact registry — one row per produced (or
-- to-be-produced) file a project's pipeline stages claim exist. Additive
-- only: no existing table above is modified. See src/models/artifact.py
-- (ArtifactRecord), src/database/artifact_repository.py (this table's
-- read/write API), and src/core/artifact_verifier.py (the read-only
-- checker that verifies a registered row against the real file).
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id     TEXT PRIMARY KEY,
    project_id      TEXT NOT NULL REFERENCES projects (project_id),
    kind            TEXT NOT NULL CHECK (kind IN ('audio','visual','animation','render','qc_report','overlay_render')),
    scene_id        TEXT,
    relative_path   TEXT NOT NULL,
    byte_size       INTEGER NOT NULL,
    sha256_checksum TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    metadata_json   TEXT NOT NULL DEFAULT '{}'
);

-- One registration per (project, kind, scene, path). SQLite treats NULL
-- as distinct from NULL in a plain UNIQUE constraint, so scene_id is
-- coalesced to '' here — otherwise two project-level rows (scene_id IS
-- NULL, e.g. two "render" artifacts for the same project/path) would
-- never conflict.
CREATE UNIQUE INDEX IF NOT EXISTS idx_artifacts_identity
ON artifacts (project_id, kind, COALESCE(scene_id, ''), relative_path);
"""


def get_connection() -> sqlite3.Connection:
    settings = get_settings()
    settings.data_dir.mkdir(exist_ok=True)
    db_path = settings.data_dir / "jobs.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def get_readonly_connection() -> sqlite3.Connection:
    """Open the existing jobs.db strictly read-only, via a SQLite URI
    connection (mode=ro) — for callers (the dry-run CLI command) that must
    never create the data directory, the database file, or its journal/
    WAL/SHM files. Never calls data_dir.mkdir() and never lets SQLite
    create a missing database file: if the data directory or jobs.db does
    not exist (or exists but can't be read), this raises
    sqlite3.OperationalError, same as any other read failure — callers
    already handle that as a clean, no-side-effect error."""
    settings = get_settings()
    db_path = settings.data_dir / "jobs.db"
    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def get_existing_connection() -> sqlite3.Connection:
    """Open the existing jobs.db for read-write access, via a SQLite URI
    connection (mode=rw) — for a command (verify-and-advance) that must be
    able to write on success but must never create the data directory,
    the database file, or its journal/WAL/SHM files as a side effect of
    merely checking whether a project/artifact exists. Same
    never-create-anything contract as get_readonly_connection() above,
    just not read-only: mode=rw (unlike sqlite3.connect()'s default
    mode=rwc) still refuses to create a missing database file. Never
    calls data_dir.mkdir() and never lets SQLite create a missing
    database file: if the data directory or jobs.db does not exist (or
    exists but can't be opened), this raises sqlite3.OperationalError,
    same as get_readonly_connection() — callers already handle that as a
    clean, no-side-effect error. A sidecar journal/WAL/SHM file may still
    appear once the caller actually starts a write transaction (e.g. via
    save_transition()) — that is a normal, expected part of that write,
    not something this function itself creates."""
    settings = get_settings()
    db_path = settings.data_dir / "jobs.db"
    uri = f"{db_path.resolve().as_uri()}?mode=rw"
    conn = sqlite3.connect(uri, uri=True)
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
