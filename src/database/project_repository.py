"""Phase 1D: SQLite persistence for the project lifecycle registry —
`projects` and `project_transitions` (schema in src/database/db.py's
SCHEMA, additive alongside the existing `jobs` table).

Same convention as src/core/job.py: every function here takes an already-
open `conn: sqlite3.Connection` as its first argument rather than managing
connection lifecycle itself — callers get their connection from
src/database/db.py's get_connection() (or, in tests, an isolated in-memory
connection with SCHEMA applied directly).

No provider calls, no network access — this module only reads and writes
local SQLite rows."""
from __future__ import annotations

import sqlite3
from datetime import datetime

from src.models.project_state import ProjectRecord, ProjectTransition


class ProjectAlreadyExistsError(Exception):
    """Raised by create_project() when project_id is already registered.
    Distinct from ProjectConcurrencyError, which is about a stale
    lifecycle_version on an UPDATE to an existing row, not creation."""


class ProjectConcurrencyError(Exception):
    """Raised by save_transition() when the row's lifecycle_version no
    longer matches the caller's expected_lifecycle_version — another
    writer updated this project first. No partial write occurs: the whole
    transaction (project row update + transition insert) is rolled back
    before this is raised."""


_PROJECT_COLUMNS = (
    "project_id",
    "manifest_path",
    "manifest_fingerprint",
    "current_stage",
    "last_successful_stage",
    "failed_stage",
    "failure_message",
    "lifecycle_version",
    "created_at",
    "updated_at",
    "completed_at",
    "archived_at",
    "retry_count",
    "execution_status",
)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _record_to_row_params(record: ProjectRecord) -> tuple:
    return (
        record.project_id,
        record.manifest_path,
        record.manifest_fingerprint,
        record.current_stage,
        record.last_successful_stage,
        record.failed_stage,
        record.failure_message,
        record.lifecycle_version,
        _iso(record.created_at),
        _iso(record.updated_at),
        _iso(record.completed_at),
        _iso(record.archived_at),
        record.retry_count,
        record.execution_status,
    )


def _row_to_record(row: sqlite3.Row) -> ProjectRecord:
    return ProjectRecord(**{column: row[column] for column in _PROJECT_COLUMNS})


def _row_to_transition(row: sqlite3.Row) -> ProjectTransition:
    return ProjectTransition(
        project_id=row["project_id"],
        from_stage=row["from_stage"],
        to_stage=row["to_stage"],
        occurred_at=row["occurred_at"],
        reason=row["reason"],
        is_retry=bool(row["is_retry"]),
        lifecycle_version=row["lifecycle_version"],
    )


def create_project(
    conn: sqlite3.Connection,
    record: ProjectRecord,
    initial_transition: ProjectTransition,
) -> ProjectRecord:
    """Insert a brand-new project row plus its initial transition audit
    row, atomically. Raises ProjectAlreadyExistsError if project_id is
    already registered — no row is changed in that case."""
    if initial_transition.project_id != record.project_id:
        raise ValueError("initial_transition.project_id must match record.project_id")

    try:
        with conn:
            conn.execute(
                f"INSERT INTO projects ({', '.join(_PROJECT_COLUMNS)}) "
                f"VALUES ({', '.join('?' for _ in _PROJECT_COLUMNS)})",
                _record_to_row_params(record),
            )
            conn.execute(
                "INSERT INTO project_transitions "
                "(project_id, from_stage, to_stage, occurred_at, reason, is_retry, lifecycle_version) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    initial_transition.project_id,
                    initial_transition.from_stage,
                    initial_transition.to_stage,
                    _iso(initial_transition.occurred_at),
                    initial_transition.reason,
                    int(initial_transition.is_retry),
                    initial_transition.lifecycle_version,
                ),
            )
    except sqlite3.IntegrityError as exc:
        raise ProjectAlreadyExistsError(
            f"project {record.project_id!r} is already registered"
        ) from exc

    return record


def get_project(conn: sqlite3.Connection, project_id: str) -> ProjectRecord | None:
    row = conn.execute(
        "SELECT * FROM projects WHERE project_id = ?", (project_id,)
    ).fetchone()
    return _row_to_record(row) if row is not None else None


def list_projects(conn: sqlite3.Connection) -> list[ProjectRecord]:
    rows = conn.execute(
        "SELECT * FROM projects ORDER BY created_at ASC, project_id ASC"
    ).fetchall()
    return [_row_to_record(row) for row in rows]


def list_project_transitions(conn: sqlite3.Connection, project_id: str) -> list[ProjectTransition]:
    """Returns transitions in deterministic chronological/event order
    (transition_id's autoincrement order, i.e. insertion order)."""
    rows = conn.execute(
        "SELECT project_id, from_stage, to_stage, occurred_at, reason, is_retry, lifecycle_version "
        "FROM project_transitions WHERE project_id = ? ORDER BY transition_id ASC",
        (project_id,),
    ).fetchall()
    return [_row_to_transition(row) for row in rows]


def save_transition(
    conn: sqlite3.Connection,
    expected_lifecycle_version: int,
    updated_record: ProjectRecord,
    transition: ProjectTransition,
) -> ProjectRecord:
    """Persist one transition: update the project row (only if it is
    still at expected_lifecycle_version) and insert the transition audit
    row, atomically. Raises ProjectConcurrencyError — with no partial
    write — if the row has already moved past expected_lifecycle_version
    (or does not exist)."""
    if transition.project_id != updated_record.project_id:
        raise ValueError("transition.project_id must match updated_record.project_id")

    with conn:
        cursor = conn.execute(
            """
            UPDATE projects
            SET manifest_path = ?, manifest_fingerprint = ?, current_stage = ?,
                last_successful_stage = ?, failed_stage = ?, failure_message = ?,
                lifecycle_version = ?, updated_at = ?, completed_at = ?, archived_at = ?,
                retry_count = ?, execution_status = ?
            WHERE project_id = ? AND lifecycle_version = ?
            """,
            (
                updated_record.manifest_path,
                updated_record.manifest_fingerprint,
                updated_record.current_stage,
                updated_record.last_successful_stage,
                updated_record.failed_stage,
                updated_record.failure_message,
                updated_record.lifecycle_version,
                _iso(updated_record.updated_at),
                _iso(updated_record.completed_at),
                _iso(updated_record.archived_at),
                updated_record.retry_count,
                updated_record.execution_status,
                updated_record.project_id,
                expected_lifecycle_version,
            ),
        )
        if cursor.rowcount == 0:
            raise ProjectConcurrencyError(
                f"project {updated_record.project_id!r} was not at lifecycle_version "
                f"{expected_lifecycle_version} (concurrent update, or unknown project_id) — "
                "no changes were written"
            )
        conn.execute(
            "INSERT INTO project_transitions "
            "(project_id, from_stage, to_stage, occurred_at, reason, is_retry, lifecycle_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                transition.project_id,
                transition.from_stage,
                transition.to_stage,
                _iso(transition.occurred_at),
                transition.reason,
                int(transition.is_retry),
                transition.lifecycle_version,
            ),
        )

    return updated_record
