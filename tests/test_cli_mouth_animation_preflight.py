"""Tests for the `mouth-animation-preflight` CLI command (src/cli.py
cmd_mouth_animation_preflight). Same isolated_db / _create_registered_project
pattern as tests/test_cli_build_upscaled_ken_burns.py. This command never
opens/decodes audio, so the registered "audio" artifact is a plain
temporary file containing arbitrary non-audio bytes, inserted directly via
register_artifact() (bypassing the real, ffprobe-based audio registrar) —
no ffmpeg, no real WAV fixture, no network anywhere in this file. The
lipsync_rhubarb.health_check() call is always mocked — no real Rhubarb
binary either."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

import src.cli as cli
from src.core.manifest_builder import build_video_manifest
from src.core.manifest_store import save_manifest
from src.core.project_state_machine import create_initial_project, initial_transition_for
from src.database.project_repository import create_project
from src.models.artifact import ArtifactRecord
from src.utils.channel_config import get_channel_policy

FIXED_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
PREFLIGHT_MODULE = "src.core.mouth_animation_preflight"


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
    projects_root: Path, scene_ids: tuple[str, ...] = ("scene-01",), story_id: str = "why-we-care-what-people-think"
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


def _register_real_visual(project_dir, project_id, scene_id, size=(200, 100)) -> None:
    from PIL import Image

    from src.core.visual_artifact_registrar import register_visual_artifact
    from src.database.db import get_connection
    from src.database.project_repository import get_project

    source = project_dir.parent / f"source-{scene_id}.png"
    Image.new("RGB", size, color=(200, 150, 100)).save(source, format="PNG")

    conn = get_connection()
    try:
        project = get_project(conn, project_id)
        manifest_path = project.manifest_path
        from src.core.manifest_store import load_manifest

        manifest = load_manifest(Path(manifest_path))
        result = register_visual_artifact(conn, project, manifest, scene_id, source, now=FIXED_NOW)
        assert result.ok, result.reasons
    finally:
        conn.close()


def _register_fake_audio(project_dir, project_id, scene_id) -> None:
    """Directly inserts an ArtifactRecord for kind='audio' and writes a
    plain non-empty file with arbitrary bytes at the expected relative
    path — bypassing the real audio registrar (which requires ffprobe/real
    audio) since this command never opens or decodes the audio file."""
    from src.database.artifact_repository import register_artifact
    from src.database.db import get_connection

    audio_path = project_dir / "audio" / f"{scene_id}.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    content = b"arbitrary non-audio bytes for a preflight test fixture"
    audio_path.write_bytes(content)
    checksum = hashlib.sha256(content).hexdigest()

    conn = get_connection()
    try:
        register_artifact(
            conn,
            ArtifactRecord(
                artifact_id=f"audio-{scene_id}",
                project_id=project_id,
                kind="audio",
                scene_id=scene_id,
                relative_path=f"audio/{scene_id}.wav",
                byte_size=len(content),
                sha256_checksum=checksum,
                created_at=FIXED_NOW,
                metadata={"duration_seconds": 1.0, "source": "external"},
            ),
        )
    finally:
        conn.close()


def _args(project_id, scene_id, manifest_path, mouth_box="100,50,40,20", out_format="text"):
    return argparse.Namespace(
        project_id=project_id, scene_id=scene_id, manifest=str(manifest_path), mouth_box=mouth_box, format=out_format
    )


# ---------------------------------------------------------------------
# parser registration
# ---------------------------------------------------------------------


def test_parser_registration():
    parser = cli.build_parser()
    args = parser.parse_args(
        ["mouth-animation-preflight", "proj-123", "scene-01", "--manifest", "m.json", "--mouth-box", "1,2,3,4"]
    )
    assert args.func is cli.cmd_mouth_animation_preflight
    assert args.project_id == "proj-123"
    assert args.scene_id == "scene-01"
    assert args.manifest == "m.json"
    assert args.mouth_box == "1,2,3,4"
    assert args.format == "text"


@pytest.mark.parametrize(
    "argv",
    [
        ["mouth-animation-preflight", "proj-123", "scene-01", "--mouth-box", "1,2,3,4"],
        ["mouth-animation-preflight", "proj-123", "scene-01", "--manifest", "m.json"],
        ["mouth-animation-preflight", "proj-123", "--manifest", "m.json", "--mouth-box", "1,2,3,4"],
        ["mouth-animation-preflight", "--manifest", "m.json", "--mouth-box", "1,2,3,4"],
    ],
)
def test_parser_requires_all_arguments(argv):
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(argv)


# ---------------------------------------------------------------------
# success
# ---------------------------------------------------------------------


def test_cli_success_text_mode(isolated_db, tmp_path, capsys, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_visual(project_dir, project_id, "scene-01", size=(200, 100))
    _register_fake_audio(project_dir, project_id, "scene-01")
    monkeypatch.setattr(f"{PREFLIGHT_MODULE}.health_check", lambda: True)

    manifest_path = project_dir / "manifest.json"
    rc = cli.cmd_mouth_animation_preflight(_args(project_id, "scene-01", manifest_path, "100,50,40,20"))
    out = capsys.readouterr().out

    assert rc == 0
    assert "mouth-animation-preflight: OK" in out
    assert "preflight passed" in out
    assert "no mouth-animation generation performed" in out
    assert f"project_id: {project_id}" in out
    assert "scene_id: scene-01" in out
    assert "visual: visuals/scene-01.png" in out
    assert "audio: audio/scene-01.wav" in out
    assert "image_size: 200x100" in out
    assert "mouth_box: 100,50,40,20" in out


def test_cli_success_json_mode(isolated_db, tmp_path, capsys, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_visual(project_dir, project_id, "scene-01", size=(200, 100))
    _register_fake_audio(project_dir, project_id, "scene-01")
    monkeypatch.setattr(f"{PREFLIGHT_MODULE}.health_check", lambda: True)

    manifest_path = project_dir / "manifest.json"
    rc = cli.cmd_mouth_animation_preflight(
        _args(project_id, "scene-01", manifest_path, "100,50,40,20", out_format="json")
    )
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert rc == 0
    assert payload == {
        "ok": True,
        "project_id": project_id,
        "scene_id": "scene-01",
        "visual_relative_path": "visuals/scene-01.png",
        "audio_relative_path": "audio/scene-01.wav",
        "image_width": 200,
        "image_height": 100,
        "mouth_box": {"center_x": 100, "center_y": 50, "width": 40, "height": 20},
    }


def test_cli_never_writes_to_the_database(isolated_db, tmp_path, capsys, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_visual(project_dir, project_id, "scene-01", size=(200, 100))
    _register_fake_audio(project_dir, project_id, "scene-01")
    monkeypatch.setattr(f"{PREFLIGHT_MODULE}.health_check", lambda: True)

    from src.utils.config import get_settings

    db_path = get_settings().data_dir / "jobs.db"
    bytes_before = db_path.read_bytes()

    manifest_path = project_dir / "manifest.json"
    cli.cmd_mouth_animation_preflight(_args(project_id, "scene-01", manifest_path, "100,50,40,20"))
    capsys.readouterr()

    assert db_path.read_bytes() == bytes_before


def test_cli_closes_readonly_connection_before_pillow_and_rhubarb_checks(isolated_db, tmp_path, capsys, monkeypatch):
    """Proves closure the same way test_cli_build_upscaled_ken_burns.py's
    analogous test does: a closed sqlite3 connection raises
    ProgrammingError on further use, so the mocked Pillow/Rhubarb checks
    try to reuse the CLI's own connection object and record whether that
    raised."""
    import sqlite3

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_visual(project_dir, project_id, "scene-01", size=(200, 100))
    _register_fake_audio(project_dir, project_id, "scene-01")

    import src.database.db as db_module

    real_get_readonly_connection = db_module.get_readonly_connection
    holder = {}

    def _spy_get_readonly_connection():
        conn = real_get_readonly_connection()
        holder["conn"] = conn
        return conn

    monkeypatch.setattr("src.database.db.get_readonly_connection", _spy_get_readonly_connection)

    was_closed = {"validate_image_and_box": None, "check_rhubarb_available": None}

    def _check_closed(key):
        conn = holder["conn"]
        try:
            conn.execute("SELECT 1")
            was_closed[key] = False
        except sqlite3.ProgrammingError:
            was_closed[key] = True

    def _fake_validate_image_and_box(source_visual_path, mouth_box):
        _check_closed("validate_image_and_box")
        return (200, 100)

    def _fake_check_rhubarb_available():
        _check_closed("check_rhubarb_available")

    monkeypatch.setattr(f"{PREFLIGHT_MODULE}.validate_image_and_box", _fake_validate_image_and_box)
    monkeypatch.setattr(f"{PREFLIGHT_MODULE}.check_rhubarb_available", _fake_check_rhubarb_available)

    manifest_path = project_dir / "manifest.json"
    rc = cli.cmd_mouth_animation_preflight(_args(project_id, "scene-01", manifest_path, "100,50,40,20"))
    capsys.readouterr()

    assert rc == 0
    assert was_closed["validate_image_and_box"] is True
    assert was_closed["check_rhubarb_available"] is True


# ---------------------------------------------------------------------
# rejections -> rc=1, stable JSON shape, no DB/file mutation
# ---------------------------------------------------------------------


def test_cli_unknown_project_returns_error(isolated_db, tmp_path, capsys):
    from src.database.db import init_db

    init_db()
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")

    rc = cli.cmd_mouth_animation_preflight(_args("does-not-exist", "scene-01", manifest_path))
    err = capsys.readouterr().err

    assert rc == 1
    assert "unknown project_id" in err


def test_cli_missing_database_returns_error(isolated_db, tmp_path, capsys):
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")

    rc = cli.cmd_mouth_animation_preflight(_args("proj-x", "scene-01", manifest_path))
    err = capsys.readouterr().err

    assert rc == 1
    assert "no local project database found" in err


# ---------------------------------------------------------------------
# sanitized sqlite3.Error during the DB-lookup block (after the read-only
# connection already opened successfully) — distinct from the
# connection-open failure above, which keeps its own specialized message.
# ---------------------------------------------------------------------


def test_cli_sqlite_error_during_project_lookup_is_sanitized_text_mode(isolated_db, tmp_path, capsys, monkeypatch):
    import sqlite3

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"

    sentinel = "SENTINEL-SQLITE-ERROR-9f2c7a1b"

    def _boom(*args, **kwargs):
        raise sqlite3.OperationalError(sentinel)

    monkeypatch.setattr("src.database.project_repository.get_project", _boom)

    pillow_called = {"value": False}
    rhubarb_called = {"value": False}
    monkeypatch.setattr(
        f"{PREFLIGHT_MODULE}.validate_image_and_box",
        lambda *a, **k: pillow_called.__setitem__("value", True) or (0, 0),
    )
    monkeypatch.setattr(
        f"{PREFLIGHT_MODULE}.check_rhubarb_available", lambda: rhubarb_called.__setitem__("value", True)
    )

    from src.utils.config import get_settings

    db_path = get_settings().data_dir / "jobs.db"
    bytes_before = db_path.read_bytes()

    rc = cli.cmd_mouth_animation_preflight(_args(project_id, "scene-01", manifest_path))
    err = capsys.readouterr().err

    assert rc == 1
    assert "local project database could not be read" in err
    assert sentinel not in err
    assert db_path.read_bytes() == bytes_before
    assert pillow_called["value"] is False
    assert rhubarb_called["value"] is False


def test_cli_sqlite_error_during_project_lookup_is_sanitized_json_mode(isolated_db, tmp_path, capsys, monkeypatch):
    import sqlite3

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"

    sentinel = "SENTINEL-SQLITE-ERROR-json-4d8e"

    def _boom(*args, **kwargs):
        raise sqlite3.OperationalError(sentinel)

    monkeypatch.setattr("src.database.project_repository.get_project", _boom)

    rc = cli.cmd_mouth_animation_preflight(_args(project_id, "scene-01", manifest_path, out_format="json"))
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert rc == 1
    assert payload == {
        "ok": False,
        "project_id": project_id,
        "scene_id": "scene-01",
        "reason": "local project database could not be read",
    }
    assert sentinel not in out


def test_cli_sqlite_error_during_artifact_lookup_is_sanitized(isolated_db, tmp_path, capsys, monkeypatch):
    import sqlite3

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_visual(project_dir, project_id, "scene-01")
    manifest_path = project_dir / "manifest.json"

    sentinel = "SENTINEL-SQLITE-ERROR-artifact-lookup-7c1a"

    def _boom(*args, **kwargs):
        raise sqlite3.OperationalError(sentinel)

    monkeypatch.setattr("src.database.artifact_repository.list_artifacts_by_scene", _boom)

    rc = cli.cmd_mouth_animation_preflight(_args(project_id, "scene-01", manifest_path))
    err = capsys.readouterr().err

    assert rc == 1
    assert "local project database could not be read" in err
    assert sentinel not in err


def test_cli_sqlite_error_during_project_lookup_still_closes_connection(isolated_db, tmp_path, capsys, monkeypatch):
    import sqlite3

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"

    def _boom(*args, **kwargs):
        raise sqlite3.OperationalError("boom")

    monkeypatch.setattr("src.database.project_repository.get_project", _boom)

    import src.database.db as db_module

    real_get_readonly_connection = db_module.get_readonly_connection
    holder = {}

    def _spy_get_readonly_connection():
        conn = real_get_readonly_connection()
        holder["conn"] = conn
        return conn

    monkeypatch.setattr("src.database.db.get_readonly_connection", _spy_get_readonly_connection)

    rc = cli.cmd_mouth_animation_preflight(_args(project_id, "scene-01", manifest_path))
    capsys.readouterr()

    assert rc == 1
    with pytest.raises(sqlite3.ProgrammingError):
        holder["conn"].execute("SELECT 1")


def test_cli_malformed_manifest_file_rejects(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = tmp_path / "bad.json"
    manifest_path.write_text("not valid json", encoding="utf-8")

    rc = cli.cmd_mouth_animation_preflight(_args(project_id, "scene-01", manifest_path))
    err = capsys.readouterr().err

    assert rc == 1
    assert "could not load manifest" in err


def test_cli_wrong_project_manifest_rejects(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    other_id, other_dir, other_manifest = _create_registered_project(
        tmp_path / "projects2", story_id="a-completely-different-story"
    )

    rc = cli.cmd_mouth_animation_preflight(_args(project_id, "scene-01", other_dir / "manifest.json"))
    err = capsys.readouterr().err

    assert rc == 1
    assert "does not match project_id" in err


def test_cli_stale_fingerprint_manifest_rejects(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["source_fingerprint"] = "0" * 64
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(raw), encoding="utf-8")

    rc = cli.cmd_mouth_animation_preflight(_args(project_id, "scene-01", tampered_path))
    err = capsys.readouterr().err

    assert rc == 1
    assert "source_fingerprint does not match" in err


def test_cli_unknown_scene_rejects(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"

    rc = cli.cmd_mouth_animation_preflight(_args(project_id, "scene-does-not-exist", manifest_path))
    err = capsys.readouterr().err

    assert rc == 1
    assert "unknown scene_id" in err


def test_cli_no_visual_artifact_rejects(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"

    rc = cli.cmd_mouth_animation_preflight(_args(project_id, "scene-01", manifest_path))
    err = capsys.readouterr().err

    assert rc == 1
    assert "no registered visual artifact found" in err


def test_cli_duplicate_visual_artifact_rejects(isolated_db, tmp_path, capsys):
    from src.database.artifact_repository import register_artifact
    from src.database.db import get_connection

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_visual(project_dir, project_id, "scene-01")

    conn = get_connection()
    try:
        register_artifact(
            conn,
            ArtifactRecord(
                artifact_id="visual-scene-01-extra",
                project_id=project_id,
                kind="visual",
                scene_id="scene-01",
                relative_path="visuals/scene-01-extra.png",
                byte_size=1,
                sha256_checksum="1" * 64,
                created_at=FIXED_NOW,
                metadata={"width": 1, "height": 1, "format": "PNG", "source": "external"},
            ),
        )
    finally:
        conn.close()

    manifest_path = project_dir / "manifest.json"
    rc = cli.cmd_mouth_animation_preflight(_args(project_id, "scene-01", manifest_path))
    err = capsys.readouterr().err

    assert rc == 1
    assert "multiple registered visual artifacts" in err


def test_cli_missing_visual_file_on_disk_rejects(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_visual(project_dir, project_id, "scene-01")
    (project_dir / "visuals" / "scene-01.png").unlink()

    manifest_path = project_dir / "manifest.json"
    rc = cli.cmd_mouth_animation_preflight(_args(project_id, "scene-01", manifest_path))
    err = capsys.readouterr().err

    assert rc == 1
    assert "missing on disk" in err


def test_cli_no_audio_artifact_rejects(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_visual(project_dir, project_id, "scene-01")

    manifest_path = project_dir / "manifest.json"
    rc = cli.cmd_mouth_animation_preflight(_args(project_id, "scene-01", manifest_path))
    err = capsys.readouterr().err

    assert rc == 1
    assert "no registered audio artifact found" in err


def test_cli_duplicate_audio_artifact_rejects(isolated_db, tmp_path, capsys):
    from src.database.artifact_repository import register_artifact
    from src.database.db import get_connection

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_visual(project_dir, project_id, "scene-01")
    _register_fake_audio(project_dir, project_id, "scene-01")

    conn = get_connection()
    try:
        register_artifact(
            conn,
            ArtifactRecord(
                artifact_id="audio-scene-01-extra",
                project_id=project_id,
                kind="audio",
                scene_id="scene-01",
                relative_path="audio/scene-01-extra.wav",
                byte_size=1,
                sha256_checksum="2" * 64,
                created_at=FIXED_NOW,
                metadata={"duration_seconds": 1.0, "source": "external"},
            ),
        )
    finally:
        conn.close()

    manifest_path = project_dir / "manifest.json"
    rc = cli.cmd_mouth_animation_preflight(_args(project_id, "scene-01", manifest_path))
    err = capsys.readouterr().err

    assert rc == 1
    assert "multiple registered audio artifacts" in err


def test_cli_missing_audio_file_on_disk_rejects(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_visual(project_dir, project_id, "scene-01")
    _register_fake_audio(project_dir, project_id, "scene-01")
    (project_dir / "audio" / "scene-01.wav").unlink()

    manifest_path = project_dir / "manifest.json"
    rc = cli.cmd_mouth_animation_preflight(_args(project_id, "scene-01", manifest_path))
    err = capsys.readouterr().err

    assert rc == 1
    assert "missing on disk" in err


def test_cli_invalid_mouth_box_syntax_rejects_before_any_db_lookup(isolated_db, tmp_path, capsys):
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")

    rc = cli.cmd_mouth_animation_preflight(
        _args("proj-x", "scene-01", manifest_path, mouth_box="not,enough")
    )
    err = capsys.readouterr().err

    assert rc == 1
    assert "exactly four comma-separated integers" in err


def test_cli_non_integer_mouth_box_rejects(isolated_db, tmp_path, capsys):
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")

    rc = cli.cmd_mouth_animation_preflight(
        _args("proj-x", "scene-01", manifest_path, mouth_box="1,2,3.5,4")
    )
    err = capsys.readouterr().err

    assert rc == 1
    assert "base-10 integers" in err


def test_cli_non_positive_mouth_box_dimension_rejects(isolated_db, tmp_path, capsys):
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")

    rc = cli.cmd_mouth_animation_preflight(
        _args("proj-x", "scene-01", manifest_path, mouth_box="1,2,0,4")
    )
    err = capsys.readouterr().err

    assert rc == 1
    assert "width and height must be positive" in err


def test_cli_negative_mouth_box_coordinate_rejects(isolated_db, tmp_path, capsys):
    """A '-' sign never matches the strict ^[0-9]+$ token grammar, so this
    is rejected as an invalid token (base-10 integers), not via the (now
    unreachable) explicit negative-value check."""
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")

    rc = cli.cmd_mouth_animation_preflight(
        _args("proj-x", "scene-01", manifest_path, mouth_box="-1,2,3,4")
    )
    err = capsys.readouterr().err

    assert rc == 1
    assert "base-10 integers" in err


def test_cli_unicode_arabic_indic_mouth_box_digit_rejects(isolated_db, tmp_path, capsys):
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")

    rc = cli.cmd_mouth_animation_preflight(
        _args("proj-x", "scene-01", manifest_path, mouth_box="٤٥,50,40,20")
    )
    err = capsys.readouterr().err

    assert rc == 1
    assert "base-10 integers" in err


def test_cli_whitespace_mouth_box_token_rejects(isolated_db, tmp_path, capsys):
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")

    rc = cli.cmd_mouth_animation_preflight(
        _args("proj-x", "scene-01", manifest_path, mouth_box=" 100,50,40,20")
    )
    err = capsys.readouterr().err

    assert rc == 1
    assert "base-10 integers" in err


def test_cli_mouth_box_out_of_bounds_rejects(isolated_db, tmp_path, capsys, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_visual(project_dir, project_id, "scene-01", size=(200, 100))
    _register_fake_audio(project_dir, project_id, "scene-01")
    monkeypatch.setattr(f"{PREFLIGHT_MODULE}.health_check", lambda: True)

    manifest_path = project_dir / "manifest.json"
    # right = (195 - 20) + 40 = 215 > 200
    rc = cli.cmd_mouth_animation_preflight(_args(project_id, "scene-01", manifest_path, "195,50,40,20"))
    err = capsys.readouterr().err

    assert rc == 1
    assert "extends outside the image bounds" in err


def test_cli_odd_sized_mouth_box_crossing_edge_rejects(isolated_db, tmp_path, capsys, monkeypatch):
    """The explicitly required odd-dimension edge-crossing demonstration
    case: width=41 (odd), positioned so the RIGHT edge crosses the image
    boundary under integer floor-division geometry."""
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_visual(project_dir, project_id, "scene-01", size=(200, 100))
    _register_fake_audio(project_dir, project_id, "scene-01")
    monkeypatch.setattr(f"{PREFLIGHT_MODULE}.health_check", lambda: True)

    manifest_path = project_dir / "manifest.json"
    # width=41 (odd): left = 190 - (41 // 2) = 190 - 20 = 170; right = 170 + 41 = 211 > 200
    rc = cli.cmd_mouth_animation_preflight(_args(project_id, "scene-01", manifest_path, "190,50,41,23"))
    err = capsys.readouterr().err

    assert rc == 1
    assert "extends outside the image bounds" in err


def test_cli_rhubarb_unavailable_rejects(isolated_db, tmp_path, capsys, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_visual(project_dir, project_id, "scene-01", size=(200, 100))
    _register_fake_audio(project_dir, project_id, "scene-01")
    monkeypatch.setattr(f"{PREFLIGHT_MODULE}.health_check", lambda: False)

    manifest_path = project_dir / "manifest.json"
    rc = cli.cmd_mouth_animation_preflight(_args(project_id, "scene-01", manifest_path, "100,50,40,20"))
    err = capsys.readouterr().err

    assert rc == 1
    assert "Rhubarb lip-sync tool is unavailable" in err


def test_cli_invalid_format_rejects(isolated_db, tmp_path, capsys):
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")

    rc = cli.cmd_mouth_animation_preflight(_args("proj-x", "scene-01", manifest_path, out_format="xml"))
    err = capsys.readouterr().err

    assert rc == 1
    assert "invalid --format" in err


def test_cli_json_failure_has_stable_shape(isolated_db, tmp_path, capsys):
    from src.database.db import init_db

    init_db()
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")

    rc = cli.cmd_mouth_animation_preflight(
        _args("does-not-exist", "scene-01", manifest_path, out_format="json")
    )
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert rc == 1
    assert payload["ok"] is False
    assert payload["project_id"] == "does-not-exist"
    assert payload["scene_id"] == "scene-01"
    assert isinstance(payload["reason"], str) and payload["reason"]


def test_cli_database_bytes_identical_before_and_after_successful_preflight(
    isolated_db, tmp_path, capsys, monkeypatch
):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_visual(project_dir, project_id, "scene-01", size=(200, 100))
    _register_fake_audio(project_dir, project_id, "scene-01")
    monkeypatch.setattr(f"{PREFLIGHT_MODULE}.health_check", lambda: True)

    from src.utils.config import get_settings

    db_path = get_settings().data_dir / "jobs.db"
    bytes_before = db_path.read_bytes()

    manifest_path = project_dir / "manifest.json"
    rc = cli.cmd_mouth_animation_preflight(_args(project_id, "scene-01", manifest_path, "100,50,40,20"))
    capsys.readouterr()

    assert rc == 0
    assert db_path.read_bytes() == bytes_before
