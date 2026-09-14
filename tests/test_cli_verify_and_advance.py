"""Tests for the `verify-and-advance` CLI command (src/cli.py
cmd_verify_and_advance). Same isolated_db / _create_registered_project
pattern as tests/test_cli_verify_artifacts.py and
tests/test_verified_transition_service.py."""
from __future__ import annotations

import argparse
import builtins
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

import src.cli as cli
from src.core.manifest_builder import build_video_manifest
from src.core.manifest_store import save_manifest
from src.core.project_state_machine import create_initial_project, initial_transition_for, transition_project
from src.database.artifact_repository import register_artifact
from src.database.project_repository import create_project, get_project, list_project_transitions, save_transition
from src.models.artifact import ArtifactRecord
from src.utils.channel_config import get_channel_policy

FIXED_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    from src.utils import config

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    config.get_settings.cache_clear()
    yield tmp_path
    config.get_settings.cache_clear()


def _story_input_dict() -> dict:
    return dict(
        story_id="why-we-care-what-people-think",
        title="Why We Care So Much What People Think",
        topic="social psychology",
        target_duration_seconds=480.0,
        language="en-US",
        viewer_facing_language="English",
        approval_status="approved",
    )


def _scene_plan_dict() -> dict:
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


def _create_registered_project(projects_root: Path):
    from src.database.db import get_connection, init_db

    manifest = build_video_manifest(
        _story_input_dict(), _scene_plan_dict(), get_channel_policy(), created_at=FIXED_NOW
    )
    manifest_path = projects_root / manifest.project_id / "manifest.json"
    save_manifest(manifest, manifest_path)

    project = create_initial_project(manifest_path, manifest, now=FIXED_NOW)
    transition = initial_transition_for(project)

    init_db()
    conn = get_connection()
    try:
        create_project(conn, project, transition)
    finally:
        conn.close()
    return project.project_id, manifest_path.parent


def _advance_to_audio_pending(project_id: str) -> None:
    from src.database.db import get_connection

    conn = get_connection()
    try:
        project = get_project(conn, project_id)
        updated, transition = transition_project(project, "audio_pending", now=FIXED_NOW)
        save_transition(conn, project.lifecycle_version, updated, transition)
    finally:
        conn.close()


def _register_valid_audio_artifact(project_id: str, project_dir: Path) -> None:
    from src.database.db import get_connection

    content = b"hello world"
    audio_path = project_dir / "audio" / "scene-01.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(content)

    record = ArtifactRecord(
        artifact_id="audio-scene-01",
        project_id=project_id,
        kind="audio",
        scene_id="scene-01",
        relative_path="audio/scene-01.wav",
        byte_size=len(content),
        sha256_checksum=hashlib.sha256(content).hexdigest(),
        created_at=FIXED_NOW,
    )
    conn = get_connection()
    try:
        register_artifact(conn, record)
    finally:
        conn.close()


def _ns(**kwargs) -> argparse.Namespace:
    defaults = dict(project_id="proj-123", to="audio_ready", reason="a reason", format="text")
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------
# parser registration
# ---------------------------------------------------------------------


def test_verify_and_advance_parser_registration():
    parser = cli.build_parser()
    args = parser.parse_args(["verify-and-advance", "proj-123", "--to", "audio_ready", "--reason", "r"])
    assert args.func is cli.cmd_verify_and_advance
    assert args.format == "text"
    assert args.to == "audio_ready"
    assert args.reason == "r"


def test_verify_and_advance_parser_rejects_unsupported_target_stage():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["verify-and-advance", "proj-123", "--to", "planned", "--reason", "r"])
    with pytest.raises(SystemExit):
        parser.parse_args(["verify-and-advance", "proj-123", "--to", "render_pending", "--reason", "r"])


def test_verify_and_advance_parser_requires_to_and_reason():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["verify-and-advance", "proj-123", "--reason", "r"])
    with pytest.raises(SystemExit):
        parser.parse_args(["verify-and-advance", "proj-123", "--to", "audio_ready"])


def test_verify_and_advance_cli_choices_match_service_supported_stages():
    from src.core.verified_transition_service import SUPPORTED_TARGET_STAGES

    parser = cli.build_parser()
    for stage in SUPPORTED_TARGET_STAGES:
        args = parser.parse_args(["verify-and-advance", "proj-123", "--to", stage, "--reason", "r"])
        assert args.to == stage


# ---------------------------------------------------------------------
# happy path: writes to SQLite only on success, text + json output
# ---------------------------------------------------------------------


def test_cmd_verify_and_advance_success_commits_and_reports_ok(isolated_db, capsys):
    project_id, project_dir = _create_registered_project(isolated_db / "projects")
    _advance_to_audio_pending(project_id)
    _register_valid_audio_artifact(project_id, project_dir)

    from src.database.db import get_connection

    conn = get_connection()
    try:
        stage_before = get_project(conn, project_id).current_stage
        transitions_before = len(list_project_transitions(conn, project_id))
    finally:
        conn.close()

    rc = cli.cmd_verify_and_advance(
        _ns(project_id=project_id, to="audio_ready", reason="narration recorded", format="text")
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert "verify-and-advance: OK" in out
    assert "approved: True" in out
    assert "db_committed: True" in out

    conn = get_connection()
    try:
        project = get_project(conn, project_id)
        transitions = list_project_transitions(conn, project_id)
    finally:
        conn.close()

    assert stage_before == "audio_pending"
    assert project.current_stage == "audio_ready"  # advanced exactly one stage
    assert len(transitions) == transitions_before + 1  # exactly one transition saved
    assert transitions[-1].reason == "narration recorded"
    assert transitions[-1].from_stage == "audio_pending"
    assert transitions[-1].to_stage == "audio_ready"


def test_cmd_verify_and_advance_json_output(isolated_db, capsys):
    project_id, project_dir = _create_registered_project(isolated_db / "projects")
    _advance_to_audio_pending(project_id)
    _register_valid_audio_artifact(project_id, project_dir)

    rc = cli.cmd_verify_and_advance(
        _ns(project_id=project_id, to="audio_ready", reason="narration recorded", format="json")
    )
    out = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert out["approved"] is True
    assert out["db_committed"] is True
    assert out["to_stage"] == "audio_ready"
    assert out["lifecycle_version_after"] == out["lifecycle_version_before"] + 1


# ---------------------------------------------------------------------
# failures -> non-zero exit, nothing written
# ---------------------------------------------------------------------


def test_cmd_verify_and_advance_missing_artifact_fails_and_writes_nothing(isolated_db, capsys):
    project_id, _project_dir = _create_registered_project(isolated_db / "projects")
    _advance_to_audio_pending(project_id)
    # deliberately do not register any artifact

    from src.database.db import get_connection
    from src.utils.config import get_settings

    db_path = get_settings().data_dir / "jobs.db"
    db_bytes_before = db_path.read_bytes()

    conn = get_connection()
    try:
        before = get_project(conn, project_id)
    finally:
        conn.close()

    rc = cli.cmd_verify_and_advance(
        _ns(project_id=project_id, to="audio_ready", reason="narration recorded", format="text")
    )
    out = capsys.readouterr().out

    assert rc == 1
    assert "verify-and-advance: FAILED" in out
    assert "missing required" in out

    # Zero DB-byte changes, not just semantic equality.
    assert db_path.read_bytes() == db_bytes_before

    conn = get_connection()
    try:
        after = get_project(conn, project_id)
    finally:
        conn.close()
    assert after == before


def test_cmd_verify_and_advance_illegal_transition_zero_db_byte_changes(isolated_db, capsys):
    """Project is still 'planned' (never advanced to audio_pending); the
    requested to_stage='audio_ready' is illegal regardless of artifacts."""
    project_id, project_dir = _create_registered_project(isolated_db / "projects")
    _register_valid_audio_artifact(project_id, project_dir)

    from src.utils.config import get_settings

    db_path = get_settings().data_dir / "jobs.db"
    db_bytes_before = db_path.read_bytes()

    rc = cli.cmd_verify_and_advance(
        _ns(project_id=project_id, to="audio_ready", reason="skip attempt", format="text")
    )
    out = capsys.readouterr().out

    assert rc == 1
    assert "verify-and-advance: FAILED" in out
    assert db_path.read_bytes() == db_bytes_before


def test_cmd_verify_and_advance_unknown_project_in_existing_db_zero_db_byte_changes(isolated_db, capsys):
    from src.database.db import init_db
    from src.utils.config import get_settings

    init_db()
    db_path = get_settings().data_dir / "jobs.db"
    db_bytes_before = db_path.read_bytes()

    rc = cli.cmd_verify_and_advance(_ns(project_id="does-not-exist"))
    err = capsys.readouterr().err

    assert rc == 1
    assert "no project found" in err
    assert db_path.read_bytes() == db_bytes_before


def test_cmd_verify_and_advance_invalid_format_returns_error(isolated_db, capsys):
    rc = cli.cmd_verify_and_advance(_ns(format="yaml"))
    assert rc == 1
    assert "invalid --format" in capsys.readouterr().err


def test_cmd_verify_and_advance_empty_reason_returns_error(isolated_db, capsys):
    rc = cli.cmd_verify_and_advance(_ns(reason="   "))
    assert rc == 1
    assert "--reason is required" in capsys.readouterr().err


def test_cmd_verify_and_advance_missing_database_creates_nothing(isolated_db, capsys):
    """No data/ directory, no jobs.db, no SQLite sidecar files — this is
    the Phase 2C fix under test: cmd_verify_and_advance must never call
    init_db()/create the registry just to discover a project doesn't
    exist (or that the database doesn't exist at all yet)."""
    assert list(isolated_db.iterdir()) == []

    rc = cli.cmd_verify_and_advance(_ns(project_id="proj-does-not-exist"))
    err = capsys.readouterr().err

    assert rc == 1
    assert "verify-and-advance: FAILED" in err
    assert "no local project database found" in err
    assert list(isolated_db.iterdir()) == []  # nothing at all was created


def test_cmd_verify_and_advance_never_creates_a_project_for_an_unknown_project_id_in_an_existing_db(
    isolated_db,
):
    from src.database.db import get_connection, init_db

    init_db()  # simulate a pre-existing, already-initialized registry

    rc = cli.cmd_verify_and_advance(_ns(project_id="proj-does-not-exist"))
    assert rc == 1

    conn = get_connection()
    try:
        assert get_project(conn, "proj-does-not-exist") is None
    finally:
        conn.close()


def test_cmd_verify_and_advance_unreadable_db_returns_error(isolated_db, monkeypatch, capsys):
    import src.database.db as db

    def _boom():
        raise sqlite3.OperationalError("db not readable")

    monkeypatch.setattr(db, "get_existing_connection", _boom)

    rc = cli.cmd_verify_and_advance(_ns())
    err = capsys.readouterr().err
    assert rc == 1
    assert "verify-and-advance: FAILED" in err
    assert "no local project database found" in err


# ---------------------------------------------------------------------
# no provider/render module is ever imported
# ---------------------------------------------------------------------


def test_cmd_verify_and_advance_does_not_import_provider_or_render_modules(isolated_db):
    project_id, project_dir = _create_registered_project(isolated_db / "projects")
    _advance_to_audio_pending(project_id)
    _register_valid_audio_artifact(project_id, project_dir)

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("src.providers") or name.startswith("src.render"):
            raise AssertionError(f"verify-and-advance must not import {name}")
        return real_import(name, globals, locals, fromlist, level)

    original = builtins.__import__
    builtins.__import__ = guarded_import
    try:
        rc = cli.cmd_verify_and_advance(
            _ns(project_id=project_id, to="audio_ready", reason="narration recorded", format="text")
        )
    finally:
        builtins.__import__ = original

    assert rc == 0
