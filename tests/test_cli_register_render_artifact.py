"""Tests for the `register-render-artifact` CLI command (src/cli.py
cmd_register_render_artifact). Same isolated_db / _create_registered_project
pattern as tests/test_cli_register_animation_artifact.py. Uses the REAL
local ffmpeg/ffprobe binary (no mocking) except where a test is
specifically about a database failure. No --scene-id anywhere — "render"
is project-level."""
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

import src.cli as cli
from src.core.manifest_builder import build_video_manifest
from src.core.manifest_store import save_manifest
from src.core.project_state_machine import create_initial_project, initial_transition_for, transition_project
from src.database.project_repository import create_project, get_project, save_transition
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


def _scene_plan_dict(scene_ids: tuple[str, ...] = ("scene-01",)) -> dict:
    return dict(
        scenes=tuple(
            dict(
                scene_id=scene_id,
                sequence=i,
                narration_text="Why does being left out sting so much?",
                scene_type="establishing",
                narrative_beat="hook",
                visual_brief="Stickman character alone on a quiet street corner at dusk.",
                motion_mode="in",
                approval_state="approved",
            )
            for i, scene_id in enumerate(scene_ids, start=1)
        ),
        role_outfits=(),
    )


def _create_registered_project(projects_root: Path, scene_ids: tuple[str, ...] = ("scene-01",)):
    from src.database.db import get_connection, init_db

    manifest = build_video_manifest(
        _story_input_dict(), _scene_plan_dict(scene_ids), get_channel_policy(), created_at=FIXED_NOW
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


def _advance_to_render_pending(project_id: str) -> None:
    from src.database.db import get_connection

    conn = get_connection()
    try:
        stages = [
            ("audio_pending", False), ("audio_ready", True),
            ("visuals_pending", False), ("visuals_ready", True),
            ("animation_pending", False), ("animation_ready", True),
            ("render_pending", False),
        ]
        for stage, verified in stages:
            project = get_project(conn, project_id)
            updated, transition = transition_project(project, stage, now=FIXED_NOW, verified=verified)
            save_transition(conn, project.lifecycle_version, updated, transition)
    finally:
        conn.close()


def _real_mp4(path: Path, duration_seconds: float = 1.0, color: str = "red") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from src.utils.config import get_settings

    ffmpeg = get_settings().ffmpeg_path
    result = subprocess.run(
        [
            ffmpeg, "-y",
            "-f", "lavfi", "-i", f"color=c={color}:size=64x64:rate=5:duration={duration_seconds}",
            "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
            "-t", str(duration_seconds),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def _real_audio_only_mp4(path: Path, duration_seconds: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from src.utils.config import get_settings

    ffmpeg = get_settings().ffmpeg_path
    result = subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", str(duration_seconds),
         "-c:a", "aac", str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def _ns(**kwargs) -> argparse.Namespace:
    defaults = dict(project_id="proj-123", file="video.mp4", format="text")
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------
# parser registration
# ---------------------------------------------------------------------


def test_register_render_artifact_parser_registration():
    parser = cli.build_parser()
    args = parser.parse_args(["register-render-artifact", "proj-123", "--file", "video.mp4"])
    assert args.func is cli.cmd_register_render_artifact
    assert args.project_id == "proj-123"
    assert args.file == "video.mp4"
    assert args.format == "text"
    assert not hasattr(args, "scene_id")


def test_register_render_artifact_requires_file():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["register-render-artifact", "proj-123"])


def test_register_render_artifact_rejects_scene_id_flag():
    """--scene-id must not exist for this command — "render" is
    project-level, not scene-level."""
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["register-render-artifact", "proj-123", "--scene-id", "scene-01", "--file", "video.mp4"]
        )


# ---------------------------------------------------------------------
# success / idempotency / conflict
# ---------------------------------------------------------------------


def test_cli_success_registers_artifact_and_copies_file(isolated_db, tmp_path, capsys):
    project_id, project_dir = _create_registered_project(tmp_path / "projects")
    source = tmp_path / "source.mp4"
    _real_mp4(source)

    rc = cli.cmd_register_render_artifact(
        argparse.Namespace(project_id=project_id, file=str(source), format="text")
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert "register-render-artifact: OK" in out
    assert (project_dir / "render" / "final.mp4").exists()


def test_cli_success_json_output_shape(isolated_db, tmp_path, capsys):
    project_id, project_dir = _create_registered_project(tmp_path / "projects")
    source = tmp_path / "source.mp4"
    _real_mp4(source, duration_seconds=1.5)

    rc = cli.cmd_register_render_artifact(
        argparse.Namespace(project_id=project_id, file=str(source), format="json")
    )
    out = capsys.readouterr().out

    assert rc == 0
    payload = json.loads(out)
    assert payload["ok"] is True
    assert payload["idempotent"] is False
    assert payload["copied"] is True
    assert payload["project_id"] == project_id
    assert payload["artifact_id"] == "render-final"
    assert payload["relative_path"] == "render/final.mp4"
    assert payload["duration_seconds"] == pytest.approx(1.5, abs=0.3)
    assert payload["artifact"]["kind"] == "render"
    assert payload["artifact"]["scene_id"] is None
    assert payload["artifact"]["metadata"]["source"] == "external"


def test_cli_duplicate_identical_run_is_idempotent(isolated_db, tmp_path, capsys):
    project_id, project_dir = _create_registered_project(tmp_path / "projects")
    source = tmp_path / "source.mp4"
    _real_mp4(source)
    ns = argparse.Namespace(project_id=project_id, file=str(source), format="text")

    rc1 = cli.cmd_register_render_artifact(ns)
    capsys.readouterr()
    rc2 = cli.cmd_register_render_artifact(ns)
    out2 = capsys.readouterr().out

    assert rc1 == 0
    assert rc2 == 0
    assert "idempotent no-op" in out2

    from src.database.db import get_connection
    from src.database.artifact_repository import list_artifacts_by_project

    conn = get_connection()
    try:
        artifacts = list_artifacts_by_project(conn, project_id, kind="render")
    finally:
        conn.close()
    assert len(artifacts) == 1


def test_cli_duplicate_different_content_is_rejected(isolated_db, tmp_path, capsys):
    project_id, project_dir = _create_registered_project(tmp_path / "projects")
    source_a = tmp_path / "a.mp4"
    _real_mp4(source_a, color="red")
    cli.cmd_register_render_artifact(argparse.Namespace(project_id=project_id, file=str(source_a), format="text"))
    capsys.readouterr()

    destination = project_dir / "render" / "final.mp4"
    bytes_before = destination.read_bytes()

    source_b = tmp_path / "b.mp4"
    _real_mp4(source_b, color="blue")
    rc = cli.cmd_register_render_artifact(
        argparse.Namespace(project_id=project_id, file=str(source_b), format="text")
    )
    out = capsys.readouterr().out  # an ordinary rejection reports via the result, on stdout

    assert rc == 1
    assert "FAILED" in out
    assert destination.read_bytes() == bytes_before


# ---------------------------------------------------------------------
# video validation via the CLI
# ---------------------------------------------------------------------


def test_cli_malformed_video_returns_error(isolated_db, tmp_path, capsys):
    project_id, project_dir = _create_registered_project(tmp_path / "projects")
    garbage = tmp_path / "garbage.mp4"
    garbage.write_bytes(b"not a video, just junk")

    rc = cli.cmd_register_render_artifact(
        argparse.Namespace(project_id=project_id, file=str(garbage), format="text")
    )
    out = capsys.readouterr().out

    assert rc == 1
    assert "could not measure video duration" in out
    assert not (project_dir / "render").exists()


def test_cli_audio_only_video_returns_error(isolated_db, tmp_path, capsys):
    project_id, project_dir = _create_registered_project(tmp_path / "projects")
    audio_only = tmp_path / "audio_only.mp4"
    _real_audio_only_mp4(audio_only)

    rc = cli.cmd_register_render_artifact(
        argparse.Namespace(project_id=project_id, file=str(audio_only), format="text")
    )
    out = capsys.readouterr().out

    assert rc == 1
    assert "does not contain a video stream" in out
    assert not (project_dir / "render").exists()


# ---------------------------------------------------------------------
# failure paths with zero side effects
# ---------------------------------------------------------------------


def test_cli_unknown_project_returns_error(isolated_db, tmp_path, capsys):
    from src.database.db import init_db

    init_db()  # DB exists, but this project_id is not registered in it
    source = tmp_path / "source.mp4"
    _real_mp4(source)

    rc = cli.cmd_register_render_artifact(
        argparse.Namespace(project_id="does-not-exist", file=str(source), format="text")
    )
    err = capsys.readouterr().err

    assert rc == 1
    assert "no project found" in err


def test_cli_missing_database_has_zero_filesystem_side_effects(isolated_db, capsys):
    data_dir = isolated_db / "data"
    assert not data_dir.exists()

    rc = cli.cmd_register_render_artifact(
        argparse.Namespace(project_id="proj-x", file="video.mp4", format="text")
    )
    err = capsys.readouterr().err

    assert rc == 1
    assert "no local project database found" in err
    assert not data_dir.exists()
    assert list(isolated_db.iterdir()) == []


def test_cli_invalid_format_returns_error(isolated_db, capsys):
    rc = cli.cmd_register_render_artifact(_ns(format="yaml"))
    err = capsys.readouterr().err
    assert rc == 1
    assert "invalid --format" in err


# ---------------------------------------------------------------------
# end-to-end with verify-and-advance, via the CLI
# ---------------------------------------------------------------------


def test_cli_end_to_end_register_then_verify_and_advance(isolated_db, tmp_path, capsys):
    project_id, project_dir = _create_registered_project(tmp_path / "projects", scene_ids=("scene-01",))
    _advance_to_render_pending(project_id)

    source = tmp_path / "source.mp4"
    _real_mp4(source)
    rc_register = cli.cmd_register_render_artifact(
        argparse.Namespace(project_id=project_id, file=str(source), format="text")
    )
    capsys.readouterr()
    assert rc_register == 0

    rc_advance = cli.cmd_verify_and_advance(
        argparse.Namespace(project_id=project_id, to="rendered", reason="smoke test", format="text")
    )
    out_advance = capsys.readouterr().out

    assert rc_advance == 0
    assert "verify-and-advance: OK" in out_advance

    from src.database.db import get_connection

    conn = get_connection()
    try:
        project = get_project(conn, project_id)
    finally:
        conn.close()
    assert project.current_stage == "rendered"


def test_cli_no_render_artifact_prevents_rendered(isolated_db, tmp_path, capsys):
    project_id, project_dir = _create_registered_project(tmp_path / "projects", scene_ids=("scene-01",))
    _advance_to_render_pending(project_id)

    rc_advance = cli.cmd_verify_and_advance(
        argparse.Namespace(project_id=project_id, to="rendered", reason="smoke test", format="text")
    )
    out_advance = capsys.readouterr().out

    assert rc_advance == 1
    assert "missing required" in out_advance

    from src.database.db import get_connection

    conn = get_connection()
    try:
        project = get_project(conn, project_id)
    finally:
        conn.close()
    assert project.current_stage == "render_pending"  # never advanced
