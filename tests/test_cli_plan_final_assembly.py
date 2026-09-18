"""Tests for the `plan-final-assembly` CLI command (src/cli.py
cmd_plan_final_assembly). Same isolated_db / _create_registered_project
pattern as tests/test_cli_derive_text_overlays.py. The core pure function
(src.core.final_assembly_planner.build_final_assembly_plan) is exercised
for real (it is pure, so no ffmpeg/ffprobe/provider/network call happens
anywhere in this file)."""
from __future__ import annotations

import argparse
import inspect
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

import src.cli as cli
from src.core.manifest_builder import build_video_manifest
from src.core.manifest_store import save_manifest
from src.core.project_state_machine import create_initial_project, initial_transition_for
from src.database.project_repository import create_project
from src.utils.channel_config import get_channel_policy

FIXED_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
MODULE = "src.core.final_assembly_planner"


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    from src.utils import config

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    config.get_settings.cache_clear()
    yield tmp_path
    config.get_settings.cache_clear()


def _story_input_dict(story_id: str = "why-we-care-what-people-think") -> dict:
    return dict(
        story_id=story_id,
        title="Why We Care So Much What People Think",
        topic="social psychology",
        target_duration_seconds=480.0,
        language="en-US",
        viewer_facing_language="English",
        approval_status="approved",
    )


def _scene_plan_dict(scene_ids: tuple[str, ...] = ("scene-01",)) -> dict:
    return dict(
        scenes=tuple(
            dict(
                scene_id=scene_id,
                sequence=i,
                narration_text=f"Narration for {scene_id}.",
                scene_type="narration",
                narrative_beat="setup",
                visual_brief=f"Visual brief for {scene_id}.",
                motion_mode="static",
                approval_state="approved",
            )
            for i, scene_id in enumerate(scene_ids, start=1)
        ),
        role_outfits=(),
    )


def _create_registered_project(
    projects_root: Path,
    scene_ids: tuple[str, ...] = ("scene-01",),
    story_id: str = "why-we-care-what-people-think",
):
    from src.database.db import get_connection, init_db

    manifest = build_video_manifest(
        _story_input_dict(story_id), _scene_plan_dict(scene_ids), get_channel_policy(), created_at=FIXED_NOW
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
    return project.project_id, manifest_path.parent, manifest


def _register_render_artifact(project_id, duration_seconds, *, artifact_id="render-final"):
    from src.database.artifact_repository import register_artifact
    from src.database.db import get_connection
    from src.models.artifact import ArtifactRecord

    conn = get_connection()
    try:
        register_artifact(
            conn,
            ArtifactRecord(
                artifact_id=artifact_id,
                project_id=project_id,
                kind="render",
                scene_id=None,
                relative_path="render/final.mp4",
                byte_size=2048,
                sha256_checksum="1" * 64,
                created_at=FIXED_NOW,
                metadata={"duration_seconds": duration_seconds, "source": "final-video-assembly-v1"},
            ),
        )
    finally:
        conn.close()


def _args(project_id, manifest_path, out_format="text"):
    return argparse.Namespace(
        project_id=project_id,
        manifest=str(manifest_path),
        format=out_format,
    )


# ---------------------------------------------------------------------
# 29-32: parser registration
# ---------------------------------------------------------------------


def test_parser_registers_plan_final_assembly():
    parser = cli.build_parser()
    args = parser.parse_args(["plan-final-assembly", "proj-123", "--manifest", "m.json"])
    assert args.func is cli.cmd_plan_final_assembly
    assert args.project_id == "proj-123"
    assert args.manifest == "m.json"
    assert args.format == "text"


def test_project_id_is_positional():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["plan-final-assembly", "--manifest", "m.json"])


def test_manifest_is_required():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["plan-final-assembly", "proj-123"])


def test_no_output_argument_exists():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["plan-final-assembly", "proj-123", "--manifest", "m.json", "--output", "o.json"])


def test_existing_commands_remain_registered():
    parser = cli.build_parser()
    assert parser.parse_args(["health"]).func is cli.cmd_health
    assert parser.parse_args(["dry-run", "p"]).func is cli.cmd_dry_run
    assert parser.parse_args(["resume-plan", "p"]).func is cli.cmd_resume_plan
    assert parser.parse_args(
        ["derive-text-overlays", "p", "--manifest", "m", "--output", "o"]
    ).func is cli.cmd_derive_text_overlays
    assert parser.parse_args(
        ["render-text-overlays", "p", "--manifest", "m", "--output", "o"]
    ).func is cli.cmd_render_text_overlays


# ---------------------------------------------------------------------
# 33-34: success output
# ---------------------------------------------------------------------


def test_cli_success_text_output_includes_every_field(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_render_artifact(project_id, 5.0)
    manifest_path = project_dir / "manifest.json"

    rc = cli.cmd_plan_final_assembly(_args(project_id, manifest_path))
    out = capsys.readouterr().out

    assert rc == 0
    assert "plan-final-assembly: OK" in out
    for field in (
        "project_id", "manifest_project_id_matches", "manifest_fingerprint_matches",
        "render_artifact_id", "render_ready", "render_duration_seconds", "overlay_count",
        "manifest_has_explicit_overlays", "overlay_render_artifact_id", "overlay_render_ready",
        "final_viewer_output_kind", "final_viewer_output_relative_path", "next_safe_local_command",
        "next_command_requires_explicit_paths", "blocked_reasons", "notes",
    ):
        assert f"{field}:" in out
    assert "derive-text-overlays" in out


def test_cli_success_json_output_includes_every_field(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_render_artifact(project_id, 5.0)
    manifest_path = project_dir / "manifest.json"

    rc = cli.cmd_plan_final_assembly(_args(project_id, manifest_path, out_format="json"))
    out, err = capsys.readouterr()
    payload = json.loads(out)

    assert rc == 0
    assert err == ""
    assert payload["ok"] is True
    for field in (
        "project_id", "manifest_project_id_matches", "manifest_fingerprint_matches",
        "render_artifact_id", "render_ready", "render_duration_seconds", "overlay_count",
        "manifest_has_explicit_overlays", "overlay_render_artifact_id", "overlay_render_ready",
        "final_viewer_output_kind", "final_viewer_output_relative_path", "next_safe_local_command",
        "next_command_requires_explicit_paths", "blocked_reasons", "notes",
    ):
        assert field in payload


# ---------------------------------------------------------------------
# 35-36: domain errors
# ---------------------------------------------------------------------


def test_cli_missing_project_error_to_stderr_only(isolated_db, tmp_path, capsys):
    from src.database.db import init_db

    init_db()
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")

    rc = cli.cmd_plan_final_assembly(_args("does-not-exist", manifest_path))
    out, err = capsys.readouterr()

    assert rc != 0
    assert out == ""
    assert "no project found" in err


def test_cli_json_domain_error_to_stderr_only(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    bad_manifest_path = tmp_path / "bad.json"
    bad_manifest_path.write_text("not valid json", encoding="utf-8")

    rc = cli.cmd_plan_final_assembly(_args(project_id, bad_manifest_path, out_format="json"))
    out, err = capsys.readouterr()

    assert rc != 0
    assert out == ""
    payload = json.loads(err)
    assert payload["ok"] is False
    assert payload["project_id"] == project_id


# ---------------------------------------------------------------------
# 37-38: sqlite sentinel sanitization
# ---------------------------------------------------------------------

_SENTINEL_SQLITE_ERROR = "SENTINEL-SQLITE-ERROR-C:\\sensitive\\project.db"


def test_cli_text_sqlite_error_is_sanitized(tmp_path, capsys, monkeypatch):
    def _boom():
        raise sqlite3.OperationalError(_SENTINEL_SQLITE_ERROR)

    monkeypatch.setattr("src.database.db.get_readonly_connection", _boom)

    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")
    rc = cli.cmd_plan_final_assembly(_args("proj-x", manifest_path))
    out, err = capsys.readouterr()

    assert rc != 0
    assert out == ""
    assert "no local project database found" in err
    assert _SENTINEL_SQLITE_ERROR not in err
    assert _SENTINEL_SQLITE_ERROR not in out


def test_cli_json_sqlite_error_is_sanitized(tmp_path, capsys, monkeypatch):
    def _boom():
        raise sqlite3.OperationalError(_SENTINEL_SQLITE_ERROR)

    monkeypatch.setattr("src.database.db.get_readonly_connection", _boom)

    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")
    rc = cli.cmd_plan_final_assembly(_args("proj-x", manifest_path, out_format="json"))
    out, err = capsys.readouterr()

    assert rc != 0
    assert out == ""
    payload = json.loads(err)
    assert payload["ok"] is False
    assert "no local project database found" in payload["reason"]
    for value in payload.values():
        assert _SENTINEL_SQLITE_ERROR not in str(value)


# ---------------------------------------------------------------------
# 39-41: sanitized failure paths
# ---------------------------------------------------------------------


def test_cli_missing_manifest_file_failure_is_sanitized(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")

    rc = cli.cmd_plan_final_assembly(_args(project_id, tmp_path / "does-not-exist.json"))
    out, err = capsys.readouterr()

    assert rc != 0
    assert out == ""
    assert "plan-final-assembly: FAILED" in err


def test_cli_project_manifest_mismatch_is_sanitized(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    other_id, other_dir, other_manifest = _create_registered_project(
        tmp_path / "projects2", ("scene-01",), story_id="a-completely-different-story"
    )

    rc = cli.cmd_plan_final_assembly(_args(project_id, other_dir / "manifest.json"))
    err = capsys.readouterr().err

    assert rc != 0
    assert "project_id" in err


def test_cli_fingerprint_mismatch_is_sanitized(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["source_fingerprint"] = "0" * 64
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    rc = cli.cmd_plan_final_assembly(_args(project_id, manifest_path))
    err = capsys.readouterr().err

    assert rc != 0
    assert "fingerprint" in err


# ---------------------------------------------------------------------
# 42-45: connection lifecycle / no subprocess / no provider
# ---------------------------------------------------------------------


def test_cli_opens_no_write_connection():
    source = inspect.getsource(cli.cmd_plan_final_assembly)
    assert "get_connection(" not in source


def test_cli_closes_read_connection_before_pure_planner_call(isolated_db, tmp_path, capsys, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_render_artifact(project_id, 5.0)
    manifest_path = project_dir / "manifest.json"

    from src.database import db as db_module

    real_get_readonly_connection = db_module.get_readonly_connection
    live_connections: list = []

    def _tracking_get_readonly_connection():
        conn = real_get_readonly_connection()
        live_connections.append(conn)
        return conn

    monkeypatch.setattr("src.database.db.get_readonly_connection", _tracking_get_readonly_connection)

    from src.core import final_assembly_planner as planner_module

    real_build_plan = planner_module.build_final_assembly_plan

    def _checking_build_plan(project_arg, manifest_arg, artifacts_arg):
        for conn in live_connections:
            with pytest.raises(sqlite3.ProgrammingError):
                conn.execute("SELECT 1")
        return real_build_plan(project_arg, manifest_arg, artifacts_arg)

    monkeypatch.setattr(f"{MODULE}.build_final_assembly_plan", _checking_build_plan)

    rc = cli.cmd_plan_final_assembly(_args(project_id, manifest_path))
    capsys.readouterr()

    assert rc == 0
    assert len(live_connections) == 1


def test_cli_does_not_invoke_subprocess():
    source = inspect.getsource(cli.cmd_plan_final_assembly)
    assert "subprocess" not in source


def test_cli_imports_no_provider_module():
    source = inspect.getsource(cli.cmd_plan_final_assembly)
    for forbidden in ("llm_groq", "llm_tokenrouter", "tts_kokoro", "image_qwen", "lipsync_rhubarb"):
        assert forbidden not in source


# ---------------------------------------------------------------------
# 46-47: no DB mutation
# ---------------------------------------------------------------------


def test_cli_success_changes_no_artifact_or_project_rows(isolated_db, tmp_path, capsys):
    from src.database.artifact_repository import list_artifacts_by_project
    from src.database.db import get_connection
    from src.database.project_repository import get_project

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_render_artifact(project_id, 5.0)
    manifest_path = project_dir / "manifest.json"

    conn = get_connection()
    try:
        artifacts_before = list_artifacts_by_project(conn, project_id)
        project_before = get_project(conn, project_id)
    finally:
        conn.close()

    cli.cmd_plan_final_assembly(_args(project_id, manifest_path))
    capsys.readouterr()

    conn = get_connection()
    try:
        artifacts_after = list_artifacts_by_project(conn, project_id)
        project_after = get_project(conn, project_id)
    finally:
        conn.close()

    assert artifacts_after == artifacts_before
    assert project_after == project_before


def test_cli_failure_changes_no_artifact_or_project_rows(isolated_db, tmp_path, capsys):
    from src.database.artifact_repository import list_artifacts_by_project
    from src.database.db import get_connection
    from src.database.project_repository import get_project

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = tmp_path / "does-not-exist.json"

    conn = get_connection()
    try:
        artifacts_before = list_artifacts_by_project(conn, project_id)
        project_before = get_project(conn, project_id)
    finally:
        conn.close()

    rc = cli.cmd_plan_final_assembly(_args(project_id, manifest_path))
    capsys.readouterr()

    conn = get_connection()
    try:
        artifacts_after = list_artifacts_by_project(conn, project_id)
        project_after = get_project(conn, project_id)
    finally:
        conn.close()

    assert rc != 0
    assert artifacts_after == artifacts_before
    assert project_after == project_before


# ---------------------------------------------------------------------
# 48-49: KeyboardInterrupt / unexpected exception
# ---------------------------------------------------------------------


def test_cli_keyboard_interrupt_is_not_caught(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"

    def _boom(*a, **k):
        raise KeyboardInterrupt()

    monkeypatch.setattr(f"{MODULE}.build_final_assembly_plan", _boom)

    with pytest.raises(KeyboardInterrupt):
        cli.cmd_plan_final_assembly(_args(project_id, manifest_path))


def test_cli_unexpected_exception_is_sanitized(isolated_db, tmp_path, capsys, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"

    secret_like = "SENTINEL-ORIGINAL-MESSAGE-never-printed-c3d9/etc/shadow"

    def _boom(*a, **k):
        raise RuntimeError(secret_like)

    monkeypatch.setattr(f"{MODULE}.build_final_assembly_plan", _boom)

    rc = cli.cmd_plan_final_assembly(_args(project_id, manifest_path))
    out, err = capsys.readouterr()

    assert rc != 0
    assert secret_like not in err
    assert secret_like not in out
    assert "an unexpected internal error occurred" in err


def test_cli_invalid_format_rejects(isolated_db, tmp_path, capsys):
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")

    rc = cli.cmd_plan_final_assembly(_args("proj-x", manifest_path, out_format="xml"))
    err = capsys.readouterr().err

    assert rc == 1
    assert "invalid --format" in err
