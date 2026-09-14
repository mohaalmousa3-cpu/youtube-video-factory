"""Tests for the `verify-artifacts` CLI command (src/cli.py cmd_verify_artifacts).
Same isolated_db / _create_registered_project pattern as
tests/test_cli_dry_run.py."""
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
from src.core.project_state_machine import create_initial_project, initial_transition_for
from src.database.artifact_repository import register_artifact
from src.database.project_repository import create_project, get_project, list_project_transitions
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


def _create_registered_project(projects_root: Path) -> tuple[str, Path]:
    """Mirrors test_cli_dry_run.py's helper: builds and saves a manifest,
    creates the matching ProjectRecord in SQLite. Returns (project_id,
    project_dir)."""
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


def _register_broken_render_artifact(project_id: str) -> None:
    """Registers a 'render' artifact whose file was never actually written
    — a claimed-but-missing project-level artifact."""
    from src.database.db import get_connection

    record = ArtifactRecord(
        artifact_id="render-final",
        project_id=project_id,
        kind="render",
        scene_id=None,
        relative_path="render/final.mp4",
        byte_size=12345,
        sha256_checksum="a" * 64,
        created_at=FIXED_NOW,
    )
    conn = get_connection()
    try:
        register_artifact(conn, record)
    finally:
        conn.close()


# ---------------------------------------------------------------------
# parser registration
# ---------------------------------------------------------------------


def test_verify_artifacts_parser_registration():
    parser = cli.build_parser()
    args = parser.parse_args(["verify-artifacts", "proj-123"])
    assert args.func is cli.cmd_verify_artifacts
    assert args.format == "text"
    assert args.kind is None


def test_verify_artifacts_parser_accepts_kind_and_format():
    parser = cli.build_parser()
    args = parser.parse_args(["verify-artifacts", "proj-123", "--kind", "audio", "--format", "json"])
    assert args.kind == "audio"
    assert args.format == "json"


def test_verify_artifacts_parser_rejects_unknown_kind():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["verify-artifacts", "proj-123", "--kind", "not-a-kind"])


def test_cli_artifact_kinds_is_derived_from_the_canonical_enum_not_a_second_list():
    """UltraReview finding: cli.py used to hardcode its own tuple of kind
    strings, a second source of truth alongside src.models.enums.ArtifactKind.
    This asserts the CLI's --kind choices are the SAME objects as the
    canonical Literal's args, not merely equal-by-value to a
    coincidentally-matching separate list."""
    from typing import get_args

    from src.models.enums import ArtifactKind

    assert cli._ARTIFACT_KINDS == get_args(ArtifactKind)
    for kind in cli._ARTIFACT_KINDS:
        parser = cli.build_parser()
        args = parser.parse_args(["verify-artifacts", "proj-123", "--kind", kind])
        assert args.kind == kind


# ---------------------------------------------------------------------
# happy path: text + json, strictly read-only
# ---------------------------------------------------------------------


def test_cmd_verify_artifacts_passes_and_is_read_only(isolated_db, capsys):
    project_id, project_dir = _create_registered_project(isolated_db / "projects")
    _register_valid_audio_artifact(project_id, project_dir)

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

    manifest_bytes_before = (project_dir / "manifest.json").read_bytes()
    audio_bytes_before = (project_dir / "audio" / "scene-01.wav").read_bytes()

    rc = cli.cmd_verify_artifacts(argparse.Namespace(project_id=project_id, kind=None, format="text"))
    out = capsys.readouterr().out

    assert rc == 0
    assert "read-only verification only" in out
    assert "[PASS]" in out

    rc_json = cli.cmd_verify_artifacts(argparse.Namespace(project_id=project_id, kind=None, format="json"))
    out_json = capsys.readouterr().out
    assert rc_json == 0
    parsed = json.loads(out_json)
    assert parsed["read_only"] is True
    assert parsed["passed"] is True
    assert parsed["artifact_count"] == 1

    conn = get_connection()
    try:
        project_after = get_project(conn, project_id)
        transitions_after = list_project_transitions(conn, project_id)
    finally:
        conn.close()

    assert project_before == project_after
    assert transitions_before == transitions_after
    assert db_before == db_path.read_bytes()
    assert manifest_bytes_before == (project_dir / "manifest.json").read_bytes()
    assert audio_bytes_before == (project_dir / "audio" / "scene-01.wav").read_bytes()


# ---------------------------------------------------------------------
# failures -> non-zero exit
# ---------------------------------------------------------------------


def test_cmd_verify_artifacts_missing_file_returns_nonzero(isolated_db, capsys):
    project_id, _project_dir = _create_registered_project(isolated_db / "projects")
    _register_broken_render_artifact(project_id)

    rc = cli.cmd_verify_artifacts(argparse.Namespace(project_id=project_id, kind=None, format="text"))
    out = capsys.readouterr().out

    assert rc == 1
    assert "[FAIL]" in out
    assert "does not exist" in out


def test_cmd_verify_artifacts_no_artifacts_registered_returns_nonzero(isolated_db, capsys):
    project_id, _project_dir = _create_registered_project(isolated_db / "projects")

    rc = cli.cmd_verify_artifacts(argparse.Namespace(project_id=project_id, kind=None, format="text"))
    err = capsys.readouterr().err

    assert rc == 1
    assert "no artifacts registered" in err


def test_cmd_verify_artifacts_kind_filter_with_no_matches_returns_nonzero(isolated_db, capsys):
    project_id, project_dir = _create_registered_project(isolated_db / "projects")
    _register_valid_audio_artifact(project_id, project_dir)

    rc = cli.cmd_verify_artifacts(argparse.Namespace(project_id=project_id, kind="render", format="text"))
    err = capsys.readouterr().err

    assert rc == 1
    assert "no artifacts registered of kind 'render'" in err


def test_cmd_verify_artifacts_kind_filter_matches_only_that_kind(isolated_db, capsys):
    project_id, project_dir = _create_registered_project(isolated_db / "projects")
    _register_valid_audio_artifact(project_id, project_dir)
    _register_broken_render_artifact(project_id)

    rc = cli.cmd_verify_artifacts(argparse.Namespace(project_id=project_id, kind="audio", format="json"))
    out = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert out["artifact_count"] == 1
    assert out["results"][0]["kind"] == "audio"


def test_cmd_verify_artifacts_unknown_project_returns_error(isolated_db, capsys):
    from src.database.db import init_db

    init_db()

    rc = cli.cmd_verify_artifacts(argparse.Namespace(project_id="does-not-exist", kind=None, format="text"))
    assert rc == 1
    assert "no project found" in capsys.readouterr().err


def test_cmd_verify_artifacts_invalid_format_returns_error(isolated_db, capsys):
    rc = cli.cmd_verify_artifacts(argparse.Namespace(project_id="any", kind=None, format="yaml"))
    assert rc == 1
    assert "invalid --format" in capsys.readouterr().err


def test_cmd_verify_artifacts_unreadable_db_returns_error(isolated_db, monkeypatch, capsys):
    import src.database.db as db

    def _boom():
        raise sqlite3.OperationalError("db not readable")

    monkeypatch.setattr(db, "get_readonly_connection", _boom)

    rc = cli.cmd_verify_artifacts(argparse.Namespace(project_id="any", kind=None, format="text"))
    assert rc == 1
    assert "could not read artifact registry" in capsys.readouterr().err


# ---------------------------------------------------------------------
# never creates data/, jobs.db, or any artifact file/directory
# ---------------------------------------------------------------------


def test_cmd_verify_artifacts_missing_database_fails_safely_and_creates_nothing(isolated_db, capsys):
    data_dir = isolated_db / "data"
    assert not data_dir.exists()

    rc = cli.cmd_verify_artifacts(
        argparse.Namespace(project_id="proj-does-not-exist", kind=None, format="text")
    )

    assert rc == 1
    assert "could not read artifact registry" in capsys.readouterr().err
    assert not data_dir.exists()
    assert list(isolated_db.iterdir()) == []  # nothing at all was created


def test_cmd_verify_artifacts_never_transitions_project_stage(isolated_db):
    project_id, project_dir = _create_registered_project(isolated_db / "projects")
    _register_valid_audio_artifact(project_id, project_dir)

    from src.database.db import get_connection

    conn = get_connection()
    try:
        before = get_project(conn, project_id)
    finally:
        conn.close()
    assert before.current_stage == "planned"

    cli.cmd_verify_artifacts(argparse.Namespace(project_id=project_id, kind=None, format="text"))

    conn = get_connection()
    try:
        after = get_project(conn, project_id)
    finally:
        conn.close()
    assert after == before  # byte-for-byte unchanged: stage was never touched


# ---------------------------------------------------------------------
# no provider/render imports or execution
# ---------------------------------------------------------------------


def test_cmd_verify_artifacts_does_not_import_provider_or_render_modules(isolated_db, monkeypatch):
    project_id, project_dir = _create_registered_project(isolated_db / "projects")
    _register_valid_audio_artifact(project_id, project_dir)

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("src.providers") or name.startswith("src.render"):
            raise AssertionError(f"verify-artifacts must not import {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    rc = cli.cmd_verify_artifacts(argparse.Namespace(project_id=project_id, kind=None, format="text"))
    assert rc == 0


# ---------------------------------------------------------------------
# OSError during inspection: normal non-zero failure, never a raw
# traceback (UltraReview finding)
# ---------------------------------------------------------------------


def test_cmd_verify_artifacts_reports_os_error_as_normal_failure_not_a_traceback(
    isolated_db, capsys, monkeypatch
):
    project_id, project_dir = _create_registered_project(isolated_db / "projects")
    _register_valid_audio_artifact(project_id, project_dir)

    import src.core.artifact_verifier as artifact_verifier_module

    def raising_hash(_path):
        raise OSError("simulated disk error")

    monkeypatch.setattr(artifact_verifier_module, "_sha256_of_file", raising_hash)

    rc = cli.cmd_verify_artifacts(argparse.Namespace(project_id=project_id, kind=None, format="text"))
    out = capsys.readouterr().out

    assert rc == 1  # normal failure exit code, not an uncaught-exception crash
    assert "[FAIL]" in out
    assert "could not inspect file" in out
