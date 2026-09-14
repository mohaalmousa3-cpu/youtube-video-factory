"""Phase 2B: SQLite persistence for the local artifact registry — the
`artifacts` table (schema in src/database/db.py's SCHEMA, additive
alongside `jobs`, `projects`, `project_transitions`).

Same convention as src/database/project_repository.py: every function
here takes an already-open `conn: sqlite3.Connection` as its first
argument rather than managing connection lifecycle itself. No provider
calls, no network access, no non-SQLite filesystem access — this module
only reads and writes local SQLite rows. It never verifies that a
registered artifact's file actually exists or matches its recorded
checksum — that's src/core/artifact_verifier.py's job.

No delete/overwrite function is defined here on purpose: Phase 2B adds no
destructive public CLI command, and nothing in this phase needs to remove
or replace an artifact row once written."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from src.models.artifact import ArtifactRecord
from src.models.enums import ArtifactKind


class ArtifactAlreadyExistsError(Exception):
    """Raised by register_artifact() when artifact_id is already
    registered. Distinct from DuplicateArtifactRegistrationError, which is
    about a different artifact_id colliding on the same
    project/kind/scene/path identity, not the artifact_id itself."""


class DuplicateArtifactRegistrationError(Exception):
    """Raised by register_artifact() when a DIFFERENT artifact_id already
    registers the same (project_id, kind, scene_id, relative_path)
    identity — Phase 2B allows only one registration per that identity."""


class ArtifactProjectNotFoundError(Exception):
    """Raised by register_artifact() when record.project_id has no
    matching row in the `projects` table — a foreign-key violation. This
    is a different failure from a duplicate registration (the artifact
    itself may be perfectly unique) and must never be reported as one."""


class ArtifactRegistrationError(Exception):
    """Raised by register_artifact() for any sqlite3.IntegrityError that
    is neither an artifact_id collision, a (project, kind, scene, path)
    identity collision, nor a foreign-key violation — e.g. a CHECK/NOT
    NULL constraint failure. Wraps the underlying error rather than
    mislabeling it as one of the more specific cases above."""


_ARTIFACT_COLUMNS = (
    "artifact_id",
    "project_id",
    "kind",
    "scene_id",
    "relative_path",
    "byte_size",
    "sha256_checksum",
    "created_at",
    "metadata_json",
)


def _record_to_row_params(record: ArtifactRecord) -> tuple:
    return (
        record.artifact_id,
        record.project_id,
        record.kind,
        record.scene_id,
        record.relative_path,
        record.byte_size,
        record.sha256_checksum,
        record.created_at.isoformat(),
        json.dumps(record.metadata, sort_keys=True),
    )


def _row_to_record(row: sqlite3.Row) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_id=row["artifact_id"],
        project_id=row["project_id"],
        kind=row["kind"],
        scene_id=row["scene_id"],
        relative_path=row["relative_path"],
        byte_size=row["byte_size"],
        sha256_checksum=row["sha256_checksum"],
        created_at=datetime.fromisoformat(row["created_at"]),
        metadata=json.loads(row["metadata_json"]),
    )


def register_artifact(conn: sqlite3.Connection, record: ArtifactRecord) -> ArtifactRecord:
    """Insert one new artifact row, atomically. Raises:
    - ArtifactProjectNotFoundError if record.project_id has no matching
      row in `projects` (a foreign-key violation) — checked first, since
      this is a distinct failure from any duplicate below and must never
      be reported as one;
    - ArtifactAlreadyExistsError if artifact_id is already registered;
    - DuplicateArtifactRegistrationError if a different artifact_id
      already registers the same (project_id, kind, scene_id,
      relative_path) identity;
    - ArtifactRegistrationError for any other integrity failure (wraps
      the underlying sqlite3.IntegrityError).
    No row is changed in any case."""
    try:
        with conn:
            conn.execute(
                f"INSERT INTO artifacts ({', '.join(_ARTIFACT_COLUMNS)}) "
                f"VALUES ({', '.join('?' for _ in _ARTIFACT_COLUMNS)})",
                _record_to_row_params(record),
            )
    except sqlite3.IntegrityError as exc:
        # sqlite3 exposes the precise constraint via sqlite_errorname
        # (Python 3.11+) — e.g. SQLITE_CONSTRAINT_FOREIGNKEY vs
        # SQLITE_CONSTRAINT_UNIQUE/_PRIMARYKEY. Checked first and on its
        # own terms, so a missing parent project is never mistaken for a
        # duplicate registration just because it also raises
        # IntegrityError.
        error_name = getattr(exc, "sqlite_errorname", None)
        if error_name == "SQLITE_CONSTRAINT_FOREIGNKEY":
            raise ArtifactProjectNotFoundError(
                f"cannot register artifact {record.artifact_id!r}: project_id "
                f"{record.project_id!r} is not registered in the local project registry"
            ) from exc

        if get_artifact(conn, record.artifact_id) is not None:
            raise ArtifactAlreadyExistsError(
                f"artifact_id {record.artifact_id!r} is already registered"
            ) from exc

        if error_name == "SQLITE_CONSTRAINT_UNIQUE":
            raise DuplicateArtifactRegistrationError(
                f"an artifact is already registered for project_id={record.project_id!r} "
                f"kind={record.kind!r} scene_id={record.scene_id!r} "
                f"relative_path={record.relative_path!r}"
            ) from exc

        raise ArtifactRegistrationError(
            f"could not register artifact {record.artifact_id!r}: {exc}"
        ) from exc

    return record


def get_artifact(conn: sqlite3.Connection, artifact_id: str) -> ArtifactRecord | None:
    row = conn.execute(
        "SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,)
    ).fetchone()
    return _row_to_record(row) if row is not None else None


def list_artifacts_by_project(
    conn: sqlite3.Connection, project_id: str, *, kind: ArtifactKind | None = None
) -> list[ArtifactRecord]:
    """Read/list artifacts registered for one project, deterministically
    ordered by created_at then artifact_id. Optionally filtered to one
    kind."""
    if kind is None:
        rows = conn.execute(
            "SELECT * FROM artifacts WHERE project_id = ? ORDER BY created_at ASC, artifact_id ASC",
            (project_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM artifacts WHERE project_id = ? AND kind = ? "
            "ORDER BY created_at ASC, artifact_id ASC",
            (project_id, kind),
        ).fetchall()
    return [_row_to_record(row) for row in rows]


def list_artifacts_by_scene(
    conn: sqlite3.Connection, project_id: str, scene_id: str
) -> list[ArtifactRecord]:
    """Read/list artifacts registered for one specific scene of one
    project, deterministically ordered by created_at then artifact_id."""
    rows = conn.execute(
        "SELECT * FROM artifacts WHERE project_id = ? AND scene_id = ? "
        "ORDER BY created_at ASC, artifact_id ASC",
        (project_id, scene_id),
    ).fetchall()
    return [_row_to_record(row) for row in rows]
