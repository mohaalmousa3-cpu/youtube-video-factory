"""Tests for the `build-scene-audio` CLI command (src/cli.py
cmd_build_scene_audio). Same isolated_db / _create_registered_project
pattern as tests/test_cli_build_scene_image.py. KokoroProvider is always
faked at src.core.scene_audio_generation.KokoroProvider — never constructed
for real, never any real synthesis or network call anywhere in this file."""
from __future__ import annotations

import argparse
import json
import wave
from datetime import datetime, timezone
from pathlib import Path

import pytest

import src.cli as cli
from src.core.cost_guard import PaidApprovalRequiredError
from src.core.manifest_builder import build_video_manifest
from src.core.manifest_store import save_manifest
from src.core.project_state_machine import create_initial_project, initial_transition_for
from src.database.project_repository import create_project
from src.utils.channel_config import get_channel_policy

FIXED_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
GENERATION_MODULE = "src.core.scene_audio_generation"


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


def _write_real_wav(path: Path, duration_seconds: float = 0.5, sample_rate: int = 24000) -> None:
    num_frames = int(duration_seconds * sample_rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_frames)


class _WritesValidWavProvider:
    def __init__(self):
        pass

    def synthesize(self, text, out_path, voice="af_bella", speed=0.92, sentence_pause=0.45, clause_pause=0.18):
        _write_real_wav(out_path)
        return out_path


def _explode_if_constructed(*args, **kwargs):
    raise AssertionError("KokoroProvider must never be constructed once a preceding check has already failed")


def _args(
    project_id,
    scene_id,
    manifest_path,
    output,
    voice=None,
    speed=None,
    sentence_pause=None,
    clause_pause=None,
    out_format="text",
):
    return argparse.Namespace(
        project_id=project_id,
        scene_id=scene_id,
        manifest=str(manifest_path),
        output=str(output),
        voice=voice,
        speed=speed,
        sentence_pause=sentence_pause,
        clause_pause=clause_pause,
        format=out_format,
    )


# ---------------------------------------------------------------------
# parser registration
# ---------------------------------------------------------------------


def test_parser_registration():
    parser = cli.build_parser()
    args = parser.parse_args(
        ["build-scene-audio", "proj-123", "scene-01", "--manifest", "m.json", "--output", "out.wav"]
    )
    assert args.func is cli.cmd_build_scene_audio
    assert args.project_id == "proj-123"
    assert args.scene_id == "scene-01"
    assert args.manifest == "m.json"
    assert args.output == "out.wav"
    assert args.voice is None
    assert args.speed is None
    assert args.sentence_pause is None
    assert args.clause_pause is None
    assert args.format == "text"


@pytest.mark.parametrize(
    "argv",
    [
        ["build-scene-audio", "proj-123", "scene-01", "--output", "out.wav"],
        ["build-scene-audio", "proj-123", "scene-01", "--manifest", "m.json"],
    ],
)
def test_parser_requires_manifest_and_output(argv):
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(argv)


def test_existing_commands_remain_registered():
    parser = cli.build_parser()
    assert parser.parse_args([
        "build-scene-image", "p", "s", "--manifest", "m", "--reference-image", "r", "--output", "o",
        "--include-character", "true",
    ]).func is cli.cmd_build_scene_image
    assert parser.parse_args(["health"]).func is cli.cmd_health


# ---------------------------------------------------------------------
# success
# ---------------------------------------------------------------------


def test_cli_success_text_mode_matches_exactly(isolated_db, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(f"{GENERATION_MODULE}.KokoroProvider", _WritesValidWavProvider)
    monkeypatch.setattr(f"{GENERATION_MODULE}.require_paid_approval", lambda *a, **k: None)

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    output = tmp_path / "out.wav"

    rc = cli.cmd_build_scene_audio(_args(project_id, "scene-01", manifest_path, output))
    out = capsys.readouterr().out

    assert rc == 0
    expected = (
        "build-scene-audio: OK (scene audio generated; not registered)\n"
        f"  project_id: {project_id}\n"
        "  scene_id: scene-01\n"
        f"  output: {output}\n"
        "  voice: af_bella\n"
        "  speed: 0.92\n"
        "  sentence_pause: 0.45\n"
        "  clause_pause: 0.18\n"
    )
    assert out == expected
    assert output.exists()


def test_cli_success_json_mode_matches_exactly(isolated_db, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(f"{GENERATION_MODULE}.KokoroProvider", _WritesValidWavProvider)
    monkeypatch.setattr(f"{GENERATION_MODULE}.require_paid_approval", lambda *a, **k: None)

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    output = tmp_path / "out.wav"

    rc = cli.cmd_build_scene_audio(
        _args(
            project_id, "scene-01", manifest_path, output,
            voice="af_sky", speed="1.1", sentence_pause="0.3", clause_pause="0.1",
            out_format="json",
        )
    )
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert rc == 0
    assert payload == {
        "ok": True,
        "project_id": project_id,
        "scene_id": "scene-01",
        "output": str(output),
        "voice": "af_sky",
        "speed": 1.1,
        "sentence_pause": 0.3,
        "clause_pause": 0.1,
        "registered": False,
    }


# ---------------------------------------------------------------------
# rejections that must block before provider construction
# ---------------------------------------------------------------------


def test_cli_unknown_project_returns_error(isolated_db, tmp_path, capsys, monkeypatch):
    from src.database.db import init_db

    monkeypatch.setattr(f"{GENERATION_MODULE}.KokoroProvider", _explode_if_constructed)
    init_db()
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")
    output = tmp_path / "out.wav"

    rc = cli.cmd_build_scene_audio(_args("does-not-exist", "scene-01", manifest_path, output))
    err = capsys.readouterr().err

    assert rc == 1
    assert "unknown project_id" in err
    assert not output.exists()


def test_cli_missing_database_returns_error(isolated_db, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(f"{GENERATION_MODULE}.KokoroProvider", _explode_if_constructed)
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")
    output = tmp_path / "out.wav"

    rc = cli.cmd_build_scene_audio(_args("proj-x", "scene-01", manifest_path, output))
    err = capsys.readouterr().err

    assert rc == 1
    assert "no local project database found" in err


def test_cli_malformed_manifest_file_rejects(isolated_db, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(f"{GENERATION_MODULE}.KokoroProvider", _explode_if_constructed)
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = tmp_path / "bad.json"
    manifest_path.write_text("not valid json", encoding="utf-8")
    output = tmp_path / "out.wav"

    rc = cli.cmd_build_scene_audio(_args(project_id, "scene-01", manifest_path, output))
    err = capsys.readouterr().err

    assert rc == 1
    assert "could not load manifest" in err


def test_cli_wrong_project_manifest_rejects(isolated_db, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(f"{GENERATION_MODULE}.KokoroProvider", _explode_if_constructed)
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    other_id, other_dir, other_manifest = _create_registered_project(
        tmp_path / "projects2", story_id="a-completely-different-story"
    )
    output = tmp_path / "out.wav"

    rc = cli.cmd_build_scene_audio(_args(project_id, "scene-01", other_dir / "manifest.json", output))
    err = capsys.readouterr().err

    assert rc == 1
    assert "does not match project_id" in err


def test_cli_stale_fingerprint_manifest_rejects(isolated_db, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(f"{GENERATION_MODULE}.KokoroProvider", _explode_if_constructed)
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["source_fingerprint"] = "0" * 64
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(raw), encoding="utf-8")
    output = tmp_path / "out.wav"

    rc = cli.cmd_build_scene_audio(_args(project_id, "scene-01", tampered_path, output))
    err = capsys.readouterr().err

    assert rc == 1
    assert "source_fingerprint does not match" in err


def test_cli_unknown_scene_rejects(isolated_db, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(f"{GENERATION_MODULE}.KokoroProvider", _explode_if_constructed)
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    output = tmp_path / "out.wav"

    rc = cli.cmd_build_scene_audio(_args(project_id, "scene-does-not-exist", manifest_path, output))
    err = capsys.readouterr().err

    assert rc == 1
    assert "unknown scene_id" in err


def test_cli_output_already_exists_rejects_and_is_never_overwritten(isolated_db, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(f"{GENERATION_MODULE}.KokoroProvider", _explode_if_constructed)
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    output = tmp_path / "out.wav"
    original_bytes = b"already here, must not change"
    output.write_bytes(original_bytes)

    rc = cli.cmd_build_scene_audio(_args(project_id, "scene-01", manifest_path, output))
    err = capsys.readouterr().err

    assert rc == 1
    assert "--output already exists" in err
    assert output.read_bytes() == original_bytes


def test_cli_invalid_speed_rejects(isolated_db, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(f"{GENERATION_MODULE}.KokoroProvider", _explode_if_constructed)
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    output = tmp_path / "out.wav"

    rc = cli.cmd_build_scene_audio(_args(project_id, "scene-01", manifest_path, output, speed="not-a-number"))
    err = capsys.readouterr().err

    assert rc == 1
    assert "must be valid decimal numbers" in err


def test_cli_invalid_format_rejects(isolated_db, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(f"{GENERATION_MODULE}.KokoroProvider", _explode_if_constructed)
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")
    output = tmp_path / "out.wav"

    rc = cli.cmd_build_scene_audio(_args("proj-x", "scene-01", manifest_path, output, out_format="xml"))
    err = capsys.readouterr().err

    assert rc == 1
    assert "invalid --format" in err


def test_cli_json_failure_has_stable_shape(isolated_db, tmp_path, capsys, monkeypatch):
    from src.database.db import init_db

    monkeypatch.setattr(f"{GENERATION_MODULE}.KokoroProvider", _explode_if_constructed)
    init_db()
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")
    output = tmp_path / "out.wav"

    rc = cli.cmd_build_scene_audio(
        _args("does-not-exist", "scene-01", manifest_path, output, out_format="json")
    )
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert rc == 1
    assert payload["ok"] is False
    assert payload["project_id"] == "does-not-exist"
    assert payload["scene_id"] == "scene-01"
    assert isinstance(payload["reason"], str) and payload["reason"]


# ---------------------------------------------------------------------
# paid-approval denial — surfaced cleanly, provider never constructed
# ---------------------------------------------------------------------


def test_cli_paid_approval_denial_reports_clean_failure_text_mode(isolated_db, tmp_path, capsys, monkeypatch):
    def _fake_require(service_name, proposals_path, *, is_paid):
        raise PaidApprovalRequiredError("denied")

    monkeypatch.setattr(f"{GENERATION_MODULE}.require_paid_approval", _fake_require)
    monkeypatch.setattr(f"{GENERATION_MODULE}.KokoroProvider", _explode_if_constructed)

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    output = tmp_path / "out.wav"

    rc = cli.cmd_build_scene_audio(_args(project_id, "scene-01", manifest_path, output))
    err = capsys.readouterr().err

    assert rc == 1
    assert "build-scene-audio: FAILED — kokoro is not approved for a paid provider call" in err
    assert "Traceback" not in err
    assert not output.exists()


def test_cli_paid_approval_denial_reports_clean_failure_json_mode(isolated_db, tmp_path, capsys, monkeypatch):
    def _fake_require(service_name, proposals_path, *, is_paid):
        raise PaidApprovalRequiredError("denied")

    monkeypatch.setattr(f"{GENERATION_MODULE}.require_paid_approval", _fake_require)
    monkeypatch.setattr(f"{GENERATION_MODULE}.KokoroProvider", _explode_if_constructed)

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    output = tmp_path / "out.wav"

    rc = cli.cmd_build_scene_audio(
        _args(project_id, "scene-01", manifest_path, output, out_format="json")
    )
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert rc == 1
    assert payload == {
        "ok": False,
        "project_id": project_id,
        "scene_id": "scene-01",
        "reason": "kokoro is not approved for a paid provider call",
    }
    assert not output.exists()


# ---------------------------------------------------------------------
# no manifest/SQLite/artifact mutation, success or failure
# ---------------------------------------------------------------------


def test_cli_never_writes_to_the_database_or_manifest_on_success(isolated_db, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(f"{GENERATION_MODULE}.KokoroProvider", _WritesValidWavProvider)
    monkeypatch.setattr(f"{GENERATION_MODULE}.require_paid_approval", lambda *a, **k: None)

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    output = tmp_path / "out.wav"

    from src.utils.config import get_settings

    db_path = get_settings().data_dir / "jobs.db"
    bytes_before = db_path.read_bytes()
    manifest_bytes_before = manifest_path.read_bytes()

    rc = cli.cmd_build_scene_audio(_args(project_id, "scene-01", manifest_path, output))
    capsys.readouterr()

    assert rc == 0
    assert db_path.read_bytes() == bytes_before
    assert manifest_path.read_bytes() == manifest_bytes_before

    from src.database.artifact_repository import list_artifacts_by_scene
    from src.database.db import get_connection

    conn = get_connection()
    try:
        artifacts = list_artifacts_by_scene(conn, project_id, "scene-01")
    finally:
        conn.close()
    assert artifacts == []


def test_cli_never_writes_to_the_database_or_manifest_on_failure(isolated_db, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(f"{GENERATION_MODULE}.KokoroProvider", _explode_if_constructed)
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    output = tmp_path / "out.wav"
    output.write_bytes(b"already here")

    from src.utils.config import get_settings

    db_path = get_settings().data_dir / "jobs.db"
    bytes_before = db_path.read_bytes()
    manifest_bytes_before = manifest_path.read_bytes()

    rc = cli.cmd_build_scene_audio(_args(project_id, "scene-01", manifest_path, output))
    capsys.readouterr()

    assert rc == 1
    assert db_path.read_bytes() == bytes_before
    assert manifest_path.read_bytes() == manifest_bytes_before

    from src.database.artifact_repository import list_artifacts_by_scene
    from src.database.db import get_connection

    conn = get_connection()
    try:
        artifacts = list_artifacts_by_scene(conn, project_id, "scene-01")
    finally:
        conn.close()
    assert artifacts == []
