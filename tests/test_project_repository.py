"""Tests for src/database/project_repository.py: SQLite persistence,
optimistic locking, and atomicity. No network, no provider.

Most tests use an isolated in-memory SQLite connection with
src/database/db.py's SCHEMA applied directly — no monkeypatching of global
config, no filesystem access at all. One test (`test_init_db_creates_...`)
additionally exercises the real init_db()/get_connection() path against a
tmp_path-rooted PROJECT_ROOT, matching tests/test_job.py's existing
fixture pattern, to prove Phase 1D's schema additions don't break that
entry point or the existing jobs table."""
import sqlite3
from datetime import datetime, timezone

import pytest

from src.core.manifest_builder import build_video_manifest
from src.core.project_state_machine import (
    create_initial_project,
    initial_transition_for,
    transition_project,
)
from src.database.db import SCHEMA
from src.database.project_repository import (
    ProjectAlreadyExistsError,
    ProjectConcurrencyError,
    create_project,
    get_project,
    list_project_transitions,
    list_projects,
    save_transition,
)
from src.utils.channel_config import get_channel_policy

FIXED_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
LATER_NOW = datetime(2026, 1, 1, 12, 5, 0, tzinfo=timezone.utc)


@pytest.fixture()
def conn():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA)
    yield connection
    connection.close()


def _valid_story_input(**overrides) -> dict:
    data = dict(
        story_id="why-we-care-what-people-think",
        title="Why We Care So Much What People Think",
        topic="social psychology",
        target_duration_seconds=480.0,
        language="en-US",
        viewer_facing_language="English",
        approval_status="approved",
    )
    data.update(overrides)
    return data


def _valid_scene_plan() -> dict:
    return dict(
        scenes=(
            dict(
                scene_id="scene-01",
                sequence=1,
                narration_text="Why does being left out sting so much?",
                scene_type="establishing",
                narrative_beat="hook",
                visual_brief="Stickman character alone on a quiet street corner at dusk.",
                motion_mode="in",
                approval_state="approved",
            ),
        ),
        role_outfits=(),
    )


def _new_project_and_transition(project_id_suffix: str = ""):
    story = _valid_story_input(story_id=f"why-we-care-what-people-think{project_id_suffix}")
    manifest = build_video_manifest(story, _valid_scene_plan(), get_channel_policy(), created_at=FIXED_NOW)
    project = create_initial_project(f"data/projects/{manifest.project_id}/manifest.json", manifest, now=FIXED_NOW)
    return project, initial_transition_for(project)


# ---------------------------------------------------------------------
# init_db compatibility
# ---------------------------------------------------------------------


def test_init_db_creates_project_tables_without_breaking_jobs(tmp_path, monkeypatch):
    from src.utils import config
    from src.database.db import get_connection, init_db
    from src.core.job import job

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    config.get_settings.cache_clear()
    try:
        init_db()
        real_conn = get_connection()
        try:
            tables = {
                row["name"]
                for row in real_conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            assert {"jobs", "projects", "project_transitions"}.issubset(tables)

            # The pre-existing jobs API still works, unmodified.
            with job(real_conn, "proj1", "tts", "kokoro") as handle:
                handle.output_path = "data/narration.wav"
            row = real_conn.execute(
                "SELECT status, output_path FROM jobs WHERE job_id = ?", (handle.job_id,)
            ).fetchone()
            assert row["status"] == "success"
            assert row["output_path"] == "data/narration.wav"
        finally:
            real_conn.close()
    finally:
        config.get_settings.cache_clear()


# ---------------------------------------------------------------------
# create_project / get_project round trip
# ---------------------------------------------------------------------


def test_create_and_get_project_round_trip(conn):
    project, initial_transition = _new_project_and_transition()

    create_project(conn, project, initial_transition)
    loaded = get_project(conn, project.project_id)

    assert loaded == project


def test_get_project_returns_none_for_unknown_id(conn):
    assert get_project(conn, "does-not-exist") is None


def test_create_project_produces_exactly_one_transition_record(conn):
    project, initial_transition = _new_project_and_transition()
    create_project(conn, project, initial_transition)

    transitions = list_project_transitions(conn, project.project_id)

    assert len(transitions) == 1
    assert transitions[0] == initial_transition


def test_create_project_duplicate_id_is_rejected_and_does_not_overwrite(conn):
    project, initial_transition = _new_project_and_transition()
    create_project(conn, project, initial_transition)

    conflicting_project = project.__class__(
        **{**project.model_dump(), "manifest_path": "data/projects/other/manifest.json"}
    )
    with pytest.raises(ProjectAlreadyExistsError):
        create_project(conn, conflicting_project, initial_transition)

    reloaded = get_project(conn, project.project_id)
    assert reloaded.manifest_path == project.manifest_path  # untouched by the rejected attempt
    assert len(list_project_transitions(conn, project.project_id)) == 1


# ---------------------------------------------------------------------
# save_transition / optimistic locking
# ---------------------------------------------------------------------


def test_save_transition_persists_stage_change_and_transition(conn):
    project, initial_transition = _new_project_and_transition()
    create_project(conn, project, initial_transition)

    updated, transition = transition_project(project, "audio_pending", now=LATER_NOW)
    result = save_transition(conn, project.lifecycle_version, updated, transition)

    assert result == updated
    reloaded = get_project(conn, project.project_id)
    assert reloaded == updated
    assert reloaded.current_stage == "audio_pending"

    transitions = list_project_transitions(conn, project.project_id)
    assert len(transitions) == 2
    assert transitions[0] == initial_transition
    assert transitions[1] == transition


def test_transitions_are_listed_in_chronological_order(conn):
    project, initial_transition = _new_project_and_transition()
    create_project(conn, project, initial_transition)

    step1, t1 = transition_project(project, "audio_pending", now=FIXED_NOW)
    save_transition(conn, project.lifecycle_version, step1, t1)

    step2, t2 = transition_project(step1, "audio_ready", now=LATER_NOW, verified=True)
    save_transition(conn, step1.lifecycle_version, step2, t2)

    transitions = list_project_transitions(conn, project.project_id)
    assert [t.to_stage for t in transitions] == ["planned", "audio_pending", "audio_ready"]


def test_save_transition_with_stale_version_raises_and_writes_nothing(conn):
    project, initial_transition = _new_project_and_transition()
    create_project(conn, project, initial_transition)

    # A first writer successfully advances the project.
    step1, t1 = transition_project(project, "audio_pending", now=FIXED_NOW)
    save_transition(conn, project.lifecycle_version, step1, t1)

    before_project = get_project(conn, project.project_id)
    before_transitions = list_project_transitions(conn, project.project_id)

    # A second writer, holding a stale copy of `project` (lifecycle_version=1,
    # already superseded by step1's version=2), tries to save its own
    # transition using that stale expected version.
    stale_update, stale_transition = transition_project(project, "audio_pending", now=LATER_NOW)
    with pytest.raises(ProjectConcurrencyError):
        save_transition(conn, project.lifecycle_version, stale_update, stale_transition)

    # No partial write: project row and transition list are byte-for-byte
    # unchanged from immediately before the rejected attempt.
    assert get_project(conn, project.project_id) == before_project
    assert list_project_transitions(conn, project.project_id) == before_transitions


def test_save_transition_unknown_project_raises(conn):
    project, _ = _new_project_and_transition()
    updated, transition = transition_project(project, "audio_pending", now=FIXED_NOW)
    with pytest.raises(ProjectConcurrencyError):
        save_transition(conn, 1, updated, transition)  # never created


# ---------------------------------------------------------------------
# list_projects
# ---------------------------------------------------------------------


def test_list_projects_returns_all_created_projects(conn):
    project_a, transition_a = _new_project_and_transition("-a")
    project_b, transition_b = _new_project_and_transition("-b")
    create_project(conn, project_a, transition_a)
    create_project(conn, project_b, transition_b)

    ids = {p.project_id for p in list_projects(conn)}
    assert ids == {project_a.project_id, project_b.project_id}
