from __future__ import annotations

import argparse
import builtins
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import src.cli as cli
import pytest

from src.core.manifest_builder import build_video_manifest
from src.core.manifest_store import save_manifest
from src.core.project_state_machine import create_initial_project, initial_transition_for
from src.database.project_repository import create_project, get_project, list_project_transitions
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


def _create_registered_project(projects_root: Path) -> str:
    from src.database.db import get_connection, init_db

    manifest = build_video_manifest(
        _story_input_dict(),
        _scene_plan_dict(),
        get_channel_policy(),
        created_at=FIXED_NOW,
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
    return project.project_id


def test_dry_run_parser_registration():
    parser = cli.build_parser()
    args = parser.parse_args(["dry-run", "proj-123"])
    assert args.func is cli.cmd_dry_run
    assert args.format == "text"


def test_cmd_dry_run_text_and_json_output_are_read_only(isolated_db, tmp_path, capsys):
    project_id = _create_registered_project(tmp_path / "projects")

    from src.database.db import get_connection
    from src.utils.config import get_settings

    settings = get_settings()
    db_path = settings.data_dir / "jobs.db"
    db_before = db_path.read_bytes()

    conn = get_connection()
    try:
        project_before = get_project(conn, project_id)
        transitions_before = list_project_transitions(conn, project_id)
    finally:
        conn.close()

    manifest_path = Path(project_before.manifest_path)
    manifest_before = manifest_path.read_bytes()

    rc_text = cli.cmd_dry_run(argparse.Namespace(project_id=project_id, format="text"))
    out_text = capsys.readouterr().out

    rc_json = cli.cmd_dry_run(argparse.Namespace(project_id=project_id, format="json"))
    out_json = capsys.readouterr().out

    assert rc_text == 0
    assert "read-only planning/reporting only" in out_text
    assert rc_json == 0

    parsed = json.loads(out_json)
    assert parsed["read_only"] is True
    assert parsed["project_id"] == project_id
    assert parsed["execution_plan"]["next_stage"] == "audio_pending"

    conn = get_connection()
    try:
        project_after = get_project(conn, project_id)
        transitions_after = list_project_transitions(conn, project_id)
    finally:
        conn.close()

    assert project_before == project_after
    assert transitions_before == transitions_after
    assert manifest_before == manifest_path.read_bytes()
    assert db_before == db_path.read_bytes()


def test_cmd_dry_run_unknown_project_returns_error(isolated_db, capsys):
    rc = cli.cmd_dry_run(argparse.Namespace(project_id="does-not-exist", format="text"))
    assert rc == 1
    assert "no project found" in capsys.readouterr().err


def test_cmd_dry_run_missing_manifest_returns_error(isolated_db, tmp_path, capsys):
    project_id = _create_registered_project(tmp_path / "projects")

    from src.database.db import get_connection

    conn = get_connection()
    try:
        project = get_project(conn, project_id)
    finally:
        conn.close()

    Path(project.manifest_path).unlink()

    rc = cli.cmd_dry_run(argparse.Namespace(project_id=project_id, format="text"))
    assert rc == 1
    assert "manifest not found" in capsys.readouterr().err


def test_cmd_dry_run_invalid_manifest_returns_error(isolated_db, tmp_path, capsys):
    project_id = _create_registered_project(tmp_path / "projects")

    from src.database.db import get_connection

    conn = get_connection()
    try:
        project = get_project(conn, project_id)
    finally:
        conn.close()

    Path(project.manifest_path).write_text("{invalid json")

    rc = cli.cmd_dry_run(argparse.Namespace(project_id=project_id, format="text"))
    assert rc == 1
    assert "invalid JSON" in capsys.readouterr().err


def test_cmd_dry_run_invalid_format_returns_error(isolated_db, capsys):
    rc = cli.cmd_dry_run(argparse.Namespace(project_id="any", format="yaml"))
    assert rc == 1
    assert "invalid --format" in capsys.readouterr().err


def test_cmd_dry_run_unreadable_db_returns_error(isolated_db, monkeypatch, capsys):
    import src.database.db as db

    def _boom():
        raise sqlite3.OperationalError("db not readable")

    monkeypatch.setattr(db, "get_connection", _boom)

    rc = cli.cmd_dry_run(argparse.Namespace(project_id="any", format="text"))
    assert rc == 1
    assert "could not read project registry" in capsys.readouterr().err


def test_cmd_dry_run_does_not_import_provider_or_render_modules(isolated_db, tmp_path, monkeypatch):
    project_id = _create_registered_project(tmp_path / "projects")

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("src.providers") or name.startswith("src.render"):
            raise AssertionError(f"dry-run must not import {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    rc = cli.cmd_dry_run(argparse.Namespace(project_id=project_id, format="text"))
    assert rc == 0
