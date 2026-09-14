"""Tests for src/database/artifact_repository.py: SQLite persistence and
uniqueness/conflict behavior. No network, no provider. Isolated in-memory
SQLite connection with src/database/db.py's SCHEMA applied directly — same
fixture pattern as tests/test_project_repository.py."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from src.database.artifact_repository import (
    ArtifactAlreadyExistsError,
    DuplicateArtifactRegistrationError,
    get_artifact,
    list_artifacts_by_project,
    list_artifacts_by_scene,
    register_artifact,
)
from src.database.db import SCHEMA
from src.models.artifact import ArtifactRecord

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
VALID_SHA256 = "a" * 64


@pytest.fixture()
def conn():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA)
    # artifacts.project_id has a FOREIGN KEY REFERENCES projects(project_id);
    # register a dummy parent row directly rather than pulling in the
    # whole manifest_builder/project_state_machine stack, which this
    # module's tests have no need of.
    connection.execute(
        "INSERT INTO projects (project_id, manifest_path, manifest_fingerprint, current_stage, "
        "last_successful_stage, lifecycle_version, created_at, updated_at, retry_count, execution_status) "
        "VALUES ('proj-abc123', 'x/manifest.json', 'fingerprint', 'planned', 'planned', 1, ?, ?, 0, 'not_executed')",
        (NOW.isoformat(), NOW.isoformat()),
    )
    connection.commit()
    yield connection
    connection.close()


def _record(**overrides) -> ArtifactRecord:
    data = dict(
        artifact_id="audio-scene-01",
        project_id="proj-abc123",
        kind="audio",
        scene_id="scene-01",
        relative_path="audio/scene-01.wav",
        byte_size=100,
        sha256_checksum=VALID_SHA256,
        created_at=NOW,
        metadata={"voice": "kokoro-af"},
    )
    data.update(overrides)
    return ArtifactRecord(**data)


# ---------------------------------------------------------------------
# register / get round trip
# ---------------------------------------------------------------------


def test_register_and_get_round_trip(conn):
    record = _record()
    register_artifact(conn, record)

    loaded = get_artifact(conn, record.artifact_id)
    assert loaded == record


def test_get_unknown_artifact_returns_none(conn):
    assert get_artifact(conn, "does-not-exist") is None


# ---------------------------------------------------------------------
# uniqueness / conflict behavior
# ---------------------------------------------------------------------


def test_duplicate_artifact_id_is_rejected_and_does_not_overwrite(conn):
    record = _record()
    register_artifact(conn, record)

    conflicting = _record(relative_path="audio/different.wav")  # same artifact_id, different path
    with pytest.raises(ArtifactAlreadyExistsError):
        register_artifact(conn, conflicting)

    reloaded = get_artifact(conn, record.artifact_id)
    assert reloaded.relative_path == record.relative_path  # untouched


def test_duplicate_project_kind_scene_path_identity_is_rejected(conn):
    record = _record(artifact_id="audio-scene-01-a")
    register_artifact(conn, record)

    same_identity_different_id = _record(artifact_id="audio-scene-01-b")
    with pytest.raises(DuplicateArtifactRegistrationError):
        register_artifact(conn, same_identity_different_id)

    assert get_artifact(conn, "audio-scene-01-b") is None  # nothing written


def test_project_level_artifacts_scene_id_none_dedup_works(conn):
    """Two project-level artifacts (scene_id=NULL) sharing project/kind/path
    must conflict too — regression check for the COALESCE(scene_id, '')
    trick in the unique index (plain NULL != NULL in SQLite would let
    duplicates slip through)."""
    render_a = _record(
        artifact_id="render-a", kind="render", scene_id=None, relative_path="render/final.mp4",
    )
    register_artifact(conn, render_a)

    render_b = _record(
        artifact_id="render-b", kind="render", scene_id=None, relative_path="render/final.mp4",
    )
    with pytest.raises(DuplicateArtifactRegistrationError):
        register_artifact(conn, render_b)


def test_different_scene_same_project_kind_path_does_not_conflict(conn):
    a = _record(artifact_id="audio-scene-01", scene_id="scene-01")
    b = _record(artifact_id="audio-scene-02", scene_id="scene-02", relative_path="audio/scene-01.wav")
    register_artifact(conn, a)
    register_artifact(conn, b)  # different scene_id -> different identity, no conflict

    assert get_artifact(conn, "audio-scene-01") is not None
    assert get_artifact(conn, "audio-scene-02") is not None


# ---------------------------------------------------------------------
# read/list by project and scene
# ---------------------------------------------------------------------


def test_list_artifacts_by_project(conn):
    a = _record(artifact_id="audio-scene-01", kind="audio", scene_id="scene-01")
    b = _record(
        artifact_id="visual-scene-01", kind="visual", scene_id="scene-01",
        relative_path="visuals/scene-01.png",
    )
    register_artifact(conn, a)
    register_artifact(conn, b)

    results = list_artifacts_by_project(conn, "proj-abc123")
    assert {r.artifact_id for r in results} == {"audio-scene-01", "visual-scene-01"}


def test_list_artifacts_by_project_filtered_by_kind(conn):
    a = _record(artifact_id="audio-scene-01", kind="audio", scene_id="scene-01")
    b = _record(
        artifact_id="visual-scene-01", kind="visual", scene_id="scene-01",
        relative_path="visuals/scene-01.png",
    )
    register_artifact(conn, a)
    register_artifact(conn, b)

    results = list_artifacts_by_project(conn, "proj-abc123", kind="audio")
    assert [r.artifact_id for r in results] == ["audio-scene-01"]


def test_list_artifacts_by_project_returns_empty_for_unknown_project(conn):
    assert list_artifacts_by_project(conn, "does-not-exist") == []


def test_list_artifacts_by_scene(conn):
    a = _record(artifact_id="audio-scene-01", kind="audio", scene_id="scene-01")
    b = _record(
        artifact_id="visual-scene-01", kind="visual", scene_id="scene-01",
        relative_path="visuals/scene-01.png",
    )
    other_scene = _record(
        artifact_id="audio-scene-02", kind="audio", scene_id="scene-02",
        relative_path="audio/scene-02.wav",
    )
    register_artifact(conn, a)
    register_artifact(conn, b)
    register_artifact(conn, other_scene)

    results = list_artifacts_by_scene(conn, "proj-abc123", "scene-01")
    assert {r.artifact_id for r in results} == {"audio-scene-01", "visual-scene-01"}


def test_metadata_round_trips_through_json_column(conn):
    record = _record(metadata={"voice": "kokoro-af", "duration": 4.2, "retried": False, "note": None})
    register_artifact(conn, record)

    loaded = get_artifact(conn, record.artifact_id)
    assert loaded.metadata == {"voice": "kokoro-af", "duration": 4.2, "retried": False, "note": None}
