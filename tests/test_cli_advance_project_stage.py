"""Tests for the `advance-project-stage` CLI command (src/cli.py
cmd_advance_project_stage). Same isolated_db / _create_registered_project
pattern as tests/test_cli_verify_and_advance.py. No provider, no
filesystem artifact — only ProjectRecord + transition rows."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

import src.cli as cli
from src.core.manifest_builder import build_video_manifest
from src.core.manifest_store import save_manifest
from src.core.project_state_machine import create_initial_project, initial_transition_for, transition_project
from src.database.project_repository import create_project, get_project, list_project_transitions, save_transition
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


def _ns(**kwargs) -> argparse.Namespace:
    defaults = dict(project_id="proj-123", to="audio_pending", reason=None, format="text")
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------
# parser registration
# ---------------------------------------------------------------------


def test_parser_registration():
    parser = cli.build_parser()
    args = parser.parse_args(["advance-project-stage", "proj-123", "--to", "audio_pending"])
    assert args.func is cli.cmd_advance_project_stage
    assert args.project_id == "proj-123"
    assert args.to == "audio_pending"
    assert args.reason is None
    assert args.format == "text"


def test_parser_accepts_optional_reason():
    parser = cli.build_parser()
    args = parser.parse_args(
        ["advance-project-stage", "proj-123", "--to", "audio_pending", "--reason", "kickoff"]
    )
    assert args.reason == "kickoff"


def test_parser_requires_to():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["advance-project-stage", "proj-123"])


def test_parser_rejects_verification_required_stages():
    parser = cli.build_parser()
    for stage in ("audio_ready", "visuals_ready", "animation_ready", "rendered", "qc_passed", "completed"):
        with pytest.raises(SystemExit):
            parser.parse_args(["advance-project-stage", "proj-123", "--to", stage])


def test_parser_rejects_archived_and_failed():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["advance-project-stage", "proj-123", "--to", "archived"])
    with pytest.raises(SystemExit):
        parser.parse_args(["advance-project-stage", "proj-123", "--to", "failed"])


def test_parser_choices_match_service_allow_list():
    from src.core.stage_advance_service import NON_VERIFICATION_TARGET_STAGES

    parser = cli.build_parser()
    for stage in NON_VERIFICATION_TARGET_STAGES:
        args = parser.parse_args(["advance-project-stage", "proj-123", "--to", stage])
        assert args.to == stage


# ---------------------------------------------------------------------
# happy path: writes to SQLite only on success, text + json output
# ---------------------------------------------------------------------


def test_cli_success_commits_and_reports_ok(isolated_db, capsys):
    project_id, _project_dir = _create_registered_project(isolated_db / "projects")

    from src.database.db import get_connection

    conn = get_connection()
    try:
        stage_before = get_project(conn, project_id).current_stage
    finally:
        conn.close()

    rc = cli.cmd_advance_project_stage(
        argparse.Namespace(project_id=project_id, to="audio_pending", reason="kickoff", format="text")
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert "advance-project-stage: OK" in out
    assert "approved: True" in out
    assert "db_committed: True" in out

    conn = get_connection()
    try:
        project = get_project(conn, project_id)
        transitions = list_project_transitions(conn, project_id)
    finally:
        conn.close()

    assert stage_before == "planned"
    assert project.current_stage == "audio_pending"
    assert transitions[-1].reason == "kickoff"
    assert transitions[-1].from_stage == "planned"
    assert transitions[-1].to_stage == "audio_pending"


def test_cli_success_without_reason(isolated_db, capsys):
    project_id, _project_dir = _create_registered_project(isolated_db / "projects")

    rc = cli.cmd_advance_project_stage(
        argparse.Namespace(project_id=project_id, to="audio_pending", reason=None, format="text")
    )

    assert rc == 0


def test_cli_json_output(isolated_db, capsys):
    project_id, _project_dir = _create_registered_project(isolated_db / "projects")

    rc = cli.cmd_advance_project_stage(
        argparse.Namespace(project_id=project_id, to="audio_pending", reason="kickoff", format="json")
    )
    out = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert out["approved"] is True
    assert out["db_committed"] is True
    assert out["from_stage"] == "planned"
    assert out["to_stage"] == "audio_pending"
    assert out["reason"] == "kickoff"
    assert out["lifecycle_version_after"] == out["lifecycle_version_before"] + 1


# ---------------------------------------------------------------------
# rejections -> non-zero exit, nothing written
# ---------------------------------------------------------------------


def test_cli_rejects_verification_required_target_at_service_layer(isolated_db, capsys):
    """The parser's choices= already blocks this at the argparse layer
    (see test_parser_rejects_verification_required_stages) — this test
    calls the handler directly (bypassing the parser, the way a
    programmatic caller might) to confirm the service-layer allow-list
    check is real defense in depth, not decorative."""
    project_id, _project_dir = _create_registered_project(isolated_db / "projects")

    from src.database.db import get_connection

    conn = get_connection()
    try:
        before = get_project(conn, project_id)
    finally:
        conn.close()

    rc = cli.cmd_advance_project_stage(
        argparse.Namespace(project_id=project_id, to="audio_ready", reason=None, format="text")
    )
    out = capsys.readouterr().out

    assert rc == 1
    assert "FAILED" in out
    assert "verify-and-advance" in out

    conn = get_connection()
    try:
        after = get_project(conn, project_id)
    finally:
        conn.close()
    assert after == before


def test_cli_rejects_skip_transition_with_zero_writes(isolated_db, capsys):
    project_id, _project_dir = _create_registered_project(isolated_db / "projects")

    from src.database.db import get_connection

    conn = get_connection()
    try:
        before = get_project(conn, project_id)
    finally:
        conn.close()

    rc = cli.cmd_advance_project_stage(
        argparse.Namespace(project_id=project_id, to="visuals_pending", reason=None, format="text")
    )
    out = capsys.readouterr().out

    assert rc == 1
    assert "FAILED" in out

    conn = get_connection()
    try:
        after = get_project(conn, project_id)
    finally:
        conn.close()
    assert after == before


def test_cli_unknown_project_returns_error(isolated_db, capsys):
    from src.database.db import init_db

    init_db()  # DB exists, but this project_id is not registered in it
    rc = cli.cmd_advance_project_stage(_ns(project_id="does-not-exist"))
    err = capsys.readouterr().err

    assert rc == 1
    assert "no project found" in err


def test_cli_missing_database_has_zero_filesystem_side_effects(isolated_db, capsys):
    assert list(isolated_db.iterdir()) == []

    rc = cli.cmd_advance_project_stage(_ns(project_id="proj-does-not-exist"))
    err = capsys.readouterr().err

    assert rc == 1
    assert "advance-project-stage: FAILED" in err
    assert "no local project database found" in err
    assert list(isolated_db.iterdir()) == []


def test_cli_invalid_format_returns_error(isolated_db, capsys):
    rc = cli.cmd_advance_project_stage(_ns(format="yaml"))
    err = capsys.readouterr().err
    assert rc == 1
    assert "invalid --format" in err


# ---------------------------------------------------------------------
# end-to-end: advance-project-stage feeding into verify-and-advance
# ---------------------------------------------------------------------


def test_cli_end_to_end_advance_then_verify_and_advance(isolated_db, capsys):
    from src.database.artifact_repository import register_artifact
    from src.models.artifact import ArtifactRecord
    import hashlib

    project_id, project_dir = _create_registered_project(isolated_db / "projects")

    rc_advance = cli.cmd_advance_project_stage(
        argparse.Namespace(project_id=project_id, to="audio_pending", reason=None, format="text")
    )
    capsys.readouterr()
    assert rc_advance == 0

    content = b"hello world"
    audio_path = project_dir / "audio" / "scene-01.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(content)

    from src.database.db import get_connection

    conn = get_connection()
    try:
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
        register_artifact(conn, record)
    finally:
        conn.close()

    rc_verify = cli.cmd_verify_and_advance(
        argparse.Namespace(project_id=project_id, to="audio_ready", reason="narration recorded", format="text")
    )
    out_verify = capsys.readouterr().out

    assert rc_verify == 0
    assert "verify-and-advance: OK" in out_verify

    conn = get_connection()
    try:
        project = get_project(conn, project_id)
    finally:
        conn.close()
    assert project.current_stage == "audio_ready"
