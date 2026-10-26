"""Tests for the `build-composite-scene-clip` CLI command (src/cli.py
cmd_build_composite_scene_clip). Same isolated_db / _create_registered_project
/ _args() pattern as tests/test_cli_build_mouth_animated_clip.py. The core
pipeline (src.core.composite_scene_generation.build_composite_scene_clip) is
always mocked in these tests — no real Rhubarb binary, no real ffmpeg
subprocess call from this file, except in the one optional, doubly-gated
integration test at the bottom of test_composite_scene_generation.py (not
duplicated here)."""
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
GENERATION_MODULE = "src.core.composite_scene_generation"


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

    from src.core.manifest_store import load_manifest
    from src.core.visual_artifact_registrar import register_visual_artifact
    from src.database.db import get_connection
    from src.database.project_repository import get_project

    source = project_dir.parent / f"source-{scene_id}.png"
    Image.new("RGB", size, color=(200, 150, 100)).save(source, format="PNG")

    conn = get_connection()
    try:
        project = get_project(conn, project_id)
        manifest = load_manifest(Path(project.manifest_path))
        result = register_visual_artifact(conn, project, manifest, scene_id, source, now=FIXED_NOW)
        assert result.ok, result.reasons
    finally:
        conn.close()


def _register_fake_audio(project_dir, project_id, scene_id) -> None:
    """Directly inserts an ArtifactRecord for kind='audio' and writes a
    plain non-empty file with arbitrary bytes at the expected relative
    path — same technique as test_cli_build_mouth_animated_clip.py, since
    this command never decodes the audio file itself (only ffmpeg/Rhubarb,
    both mocked in these tests, ever read it)."""
    from src.database.artifact_repository import register_artifact
    from src.database.db import get_connection

    audio_path = project_dir / "audio" / f"{scene_id}.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    content = b"arbitrary non-audio bytes for a generation test fixture"
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


def _args(
    project_id,
    scene_id,
    manifest_path,
    mouth_box="100,50,40,20",
    limb_box="10,10,40,90,25,15",
    sway_amplitude_degrees="4.0",
    sway_cycles="1.0",
    output="out.mp4",
    out_format="text",
):
    return argparse.Namespace(
        project_id=project_id,
        scene_id=scene_id,
        manifest=str(manifest_path),
        mouth_box=mouth_box,
        limb_box=limb_box,
        sway_amplitude_degrees=sway_amplitude_degrees,
        sway_cycles=sway_cycles,
        output=str(output),
        format=out_format,
    )


def _mock_pipeline_success(monkeypatch):
    def _fake_pipeline(
        scene, source_visual_path, source_audio_path, raw_mouth_box, raw_limb_box, raw_amplitude_degrees, raw_sway_cycles, out_path
    ):
        Path(out_path).write_bytes(b"fake-mp4-bytes")
        return out_path

    monkeypatch.setattr(f"{GENERATION_MODULE}.build_composite_scene_clip", _fake_pipeline)


# ---------------------------------------------------------------------
# parser registration
# ---------------------------------------------------------------------


def test_parser_registration():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "build-composite-scene-clip",
            "proj-123",
            "scene-01",
            "--manifest",
            "m.json",
            "--mouth-box",
            "1,2,3,4",
            "--limb-box",
            "5,6,7,8,9,10",
            "--sway-amplitude-degrees",
            "4.0",
            "--output",
            "out.mp4",
        ]
    )
    assert args.func is cli.cmd_build_composite_scene_clip
    assert args.project_id == "proj-123"
    assert args.scene_id == "scene-01"
    assert args.manifest == "m.json"
    assert args.mouth_box == "1,2,3,4"
    assert args.limb_box == "5,6,7,8,9,10"
    assert args.sway_amplitude_degrees == "4.0"
    assert args.sway_cycles == "1.0"
    assert args.output == "out.mp4"
    assert args.format == "text"


def test_parser_has_no_frame_count_option():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "build-composite-scene-clip",
                "proj-123",
                "scene-01",
                "--manifest",
                "m.json",
                "--mouth-box",
                "1,2,3,4",
                "--limb-box",
                "5,6,7,8,9,10",
                "--sway-amplitude-degrees",
                "4.0",
                "--frame-count",
                "10",
                "--output",
                "out.mp4",
            ]
        )


@pytest.mark.parametrize(
    "argv",
    [
        [
            "build-composite-scene-clip",
            "proj-123",
            "scene-01",
            "--limb-box",
            "5,6,7,8,9,10",
            "--sway-amplitude-degrees",
            "4.0",
            "--output",
            "out.mp4",
        ],
        [
            "build-composite-scene-clip",
            "proj-123",
            "scene-01",
            "--manifest",
            "m.json",
            "--sway-amplitude-degrees",
            "4.0",
            "--output",
            "out.mp4",
        ],
        [
            "build-composite-scene-clip",
            "proj-123",
            "scene-01",
            "--manifest",
            "m.json",
            "--mouth-box",
            "1,2,3,4",
            "--limb-box",
            "5,6,7,8,9,10",
            "--output",
            "out.mp4",
        ],
        [
            "build-composite-scene-clip",
            "proj-123",
            "scene-01",
            "--manifest",
            "m.json",
            "--mouth-box",
            "1,2,3,4",
            "--limb-box",
            "5,6,7,8,9,10",
            "--sway-amplitude-degrees",
            "4.0",
        ],
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
    _register_real_visual(project_dir, project_id, "scene-01")
    _register_fake_audio(project_dir, project_id, "scene-01")
    _mock_pipeline_success(monkeypatch)

    manifest_path = project_dir / "manifest.json"
    output_path = tmp_path / "clip.mp4"
    rc = cli.cmd_build_composite_scene_clip(_args(project_id, "scene-01", manifest_path, output=output_path))
    out = capsys.readouterr().out

    assert rc == 0
    assert "build-composite-scene-clip: OK" in out
    assert "audio-muxed composite clip generated; not registered" in out
    assert f"project_id: {project_id}" in out
    assert "scene_id: scene-01" in out
    assert str(output_path) in out
    assert output_path.exists()


def test_cli_success_json_mode(isolated_db, tmp_path, capsys, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_visual(project_dir, project_id, "scene-01")
    _register_fake_audio(project_dir, project_id, "scene-01")
    _mock_pipeline_success(monkeypatch)

    manifest_path = project_dir / "manifest.json"
    output_path = tmp_path / "clip.mp4"
    rc = cli.cmd_build_composite_scene_clip(
        _args(project_id, "scene-01", manifest_path, output=output_path, out_format="json")
    )
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert rc == 0
    assert payload == {
        "ok": True,
        "project_id": project_id,
        "scene_id": "scene-01",
        "output": str(output_path),
        "audio_muxed": True,
        "registered": False,
    }


def test_cli_never_writes_to_the_database(isolated_db, tmp_path, capsys, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_visual(project_dir, project_id, "scene-01")
    _register_fake_audio(project_dir, project_id, "scene-01")
    _mock_pipeline_success(monkeypatch)

    from src.utils.config import get_settings

    db_path = get_settings().data_dir / "jobs.db"
    bytes_before = db_path.read_bytes()

    manifest_path = project_dir / "manifest.json"
    output_path = tmp_path / "clip.mp4"
    cli.cmd_build_composite_scene_clip(_args(project_id, "scene-01", manifest_path, output=output_path))
    capsys.readouterr()

    assert db_path.read_bytes() == bytes_before


def test_cli_closes_readonly_connection_before_pipeline_is_invoked(isolated_db, tmp_path, capsys, monkeypatch):
    import sqlite3

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_visual(project_dir, project_id, "scene-01")
    _register_fake_audio(project_dir, project_id, "scene-01")

    import src.database.db as db_module

    real_get_readonly_connection = db_module.get_readonly_connection
    holder = {}

    def _spy_get_readonly_connection():
        conn = real_get_readonly_connection()
        holder["conn"] = conn
        return conn

    monkeypatch.setattr("src.database.db.get_readonly_connection", _spy_get_readonly_connection)

    was_closed = {"value": None}

    def _fake_pipeline(
        scene, source_visual_path, source_audio_path, raw_mouth_box, raw_limb_box, raw_amplitude_degrees, raw_sway_cycles, out_path
    ):
        conn = holder["conn"]
        try:
            conn.execute("SELECT 1")
            was_closed["value"] = False
        except sqlite3.ProgrammingError:
            was_closed["value"] = True
        Path(out_path).write_bytes(b"fake-mp4-bytes")
        return out_path

    monkeypatch.setattr(f"{GENERATION_MODULE}.build_composite_scene_clip", _fake_pipeline)

    manifest_path = project_dir / "manifest.json"
    output_path = tmp_path / "clip.mp4"
    rc = cli.cmd_build_composite_scene_clip(_args(project_id, "scene-01", manifest_path, output=output_path))
    capsys.readouterr()

    assert rc == 0
    assert was_closed["value"] is True


def test_cli_no_output_created_by_cli_itself_on_failure(isolated_db, tmp_path, capsys):
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")
    output_path = tmp_path / "clip.mp4"

    rc = cli.cmd_build_composite_scene_clip(_args("does-not-exist", "scene-01", manifest_path, output=output_path))
    capsys.readouterr()

    assert rc == 1
    assert not output_path.exists()


# ---------------------------------------------------------------------
# rejections -> rc=1, no new output file, no DB mutation
# ---------------------------------------------------------------------


def test_cli_unknown_project_returns_error(isolated_db, tmp_path, capsys):
    from src.database.db import init_db

    init_db()
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")

    rc = cli.cmd_build_composite_scene_clip(_args("does-not-exist", "scene-01", manifest_path))
    err = capsys.readouterr().err

    assert rc == 1
    assert "unknown project_id" in err


def test_cli_missing_database_returns_error(isolated_db, tmp_path, capsys):
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")

    rc = cli.cmd_build_composite_scene_clip(_args("proj-x", "scene-01", manifest_path))
    err = capsys.readouterr().err

    assert rc == 1
    assert "no local project database found" in err


def test_cli_sqlite_error_during_lookup_is_sanitized(isolated_db, tmp_path, capsys, monkeypatch):
    import sqlite3

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"

    sentinel = "SENTINEL-SQLITE-ERROR-composite-gen-7a1e"

    def _boom(*args, **kwargs):
        raise sqlite3.OperationalError(sentinel)

    monkeypatch.setattr("src.database.project_repository.get_project", _boom)

    rc = cli.cmd_build_composite_scene_clip(_args(project_id, "scene-01", manifest_path))
    err = capsys.readouterr().err

    assert rc == 1
    assert "local project database could not be read" in err
    assert sentinel not in err


def test_cli_malformed_manifest_file_rejects(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = tmp_path / "bad.json"
    manifest_path.write_text("not valid json", encoding="utf-8")

    rc = cli.cmd_build_composite_scene_clip(_args(project_id, "scene-01", manifest_path))
    err = capsys.readouterr().err

    assert rc == 1
    assert "could not load manifest" in err


def test_cli_wrong_project_manifest_rejects(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    other_id, other_dir, other_manifest = _create_registered_project(
        tmp_path / "projects2", story_id="a-completely-different-story"
    )

    rc = cli.cmd_build_composite_scene_clip(_args(project_id, "scene-01", other_dir / "manifest.json"))
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

    rc = cli.cmd_build_composite_scene_clip(_args(project_id, "scene-01", tampered_path))
    err = capsys.readouterr().err

    assert rc == 1
    assert "source_fingerprint does not match" in err


def test_cli_unknown_scene_rejects(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"

    rc = cli.cmd_build_composite_scene_clip(_args(project_id, "scene-does-not-exist", manifest_path))
    err = capsys.readouterr().err

    assert rc == 1
    assert "unknown scene_id" in err


def test_cli_no_visual_artifact_rejects(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"

    rc = cli.cmd_build_composite_scene_clip(_args(project_id, "scene-01", manifest_path))
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
    rc = cli.cmd_build_composite_scene_clip(_args(project_id, "scene-01", manifest_path))
    err = capsys.readouterr().err

    assert rc == 1
    assert "multiple registered visual artifacts" in err


def test_cli_missing_visual_file_on_disk_rejects(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_visual(project_dir, project_id, "scene-01")
    (project_dir / "visuals" / "scene-01.png").unlink()

    manifest_path = project_dir / "manifest.json"
    rc = cli.cmd_build_composite_scene_clip(_args(project_id, "scene-01", manifest_path))
    err = capsys.readouterr().err

    assert rc == 1
    assert "missing on disk" in err


def test_cli_no_audio_artifact_rejects(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_visual(project_dir, project_id, "scene-01")

    manifest_path = project_dir / "manifest.json"
    rc = cli.cmd_build_composite_scene_clip(_args(project_id, "scene-01", manifest_path))
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
    rc = cli.cmd_build_composite_scene_clip(_args(project_id, "scene-01", manifest_path))
    err = capsys.readouterr().err

    assert rc == 1
    assert "multiple registered audio artifacts" in err


def test_cli_missing_audio_file_on_disk_rejects(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_visual(project_dir, project_id, "scene-01")
    _register_fake_audio(project_dir, project_id, "scene-01")
    (project_dir / "audio" / "scene-01.wav").unlink()

    manifest_path = project_dir / "manifest.json"
    rc = cli.cmd_build_composite_scene_clip(_args(project_id, "scene-01", manifest_path))
    err = capsys.readouterr().err

    assert rc == 1
    assert "missing on disk" in err


def test_cli_invalid_format_rejects(isolated_db, tmp_path, capsys):
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")

    rc = cli.cmd_build_composite_scene_clip(_args("proj-x", "scene-01", manifest_path, out_format="xml"))
    err = capsys.readouterr().err

    assert rc == 1
    assert "invalid --format" in err


def test_cli_core_error_text_mode(isolated_db, tmp_path, capsys, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_visual(project_dir, project_id, "scene-01")
    _register_fake_audio(project_dir, project_id, "scene-01")

    from src.core.composite_scene_generation import CompositeSceneGenerationError

    def _fake_pipeline(
        scene, source_visual_path, source_audio_path, raw_mouth_box, raw_limb_box, raw_amplitude_degrees, raw_sway_cycles, out_path
    ):
        raise CompositeSceneGenerationError("mouth cue generation failed")

    monkeypatch.setattr(f"{GENERATION_MODULE}.build_composite_scene_clip", _fake_pipeline)

    manifest_path = project_dir / "manifest.json"
    output_path = tmp_path / "clip.mp4"
    rc = cli.cmd_build_composite_scene_clip(_args(project_id, "scene-01", manifest_path, output=output_path))
    err = capsys.readouterr().err

    assert rc == 1
    assert "build-composite-scene-clip: FAILED — mouth cue generation failed" in err
    assert not output_path.exists()


def test_cli_core_error_json_mode(isolated_db, tmp_path, capsys, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_visual(project_dir, project_id, "scene-01")
    _register_fake_audio(project_dir, project_id, "scene-01")

    from src.core.composite_scene_generation import CompositeSceneGenerationError

    def _fake_pipeline(
        scene, source_visual_path, source_audio_path, raw_mouth_box, raw_limb_box, raw_amplitude_degrees, raw_sway_cycles, out_path
    ):
        raise CompositeSceneGenerationError("composite scene render failed")

    monkeypatch.setattr(f"{GENERATION_MODULE}.build_composite_scene_clip", _fake_pipeline)

    manifest_path = project_dir / "manifest.json"
    output_path = tmp_path / "clip.mp4"
    rc = cli.cmd_build_composite_scene_clip(
        _args(project_id, "scene-01", manifest_path, output=output_path, out_format="json")
    )
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert rc == 1
    assert payload == {
        "ok": False,
        "project_id": project_id,
        "scene_id": "scene-01",
        "reason": "composite scene render failed",
    }
    assert "audio_muxed" not in payload
    assert "registered" not in payload


def test_cli_json_success_includes_audio_muxed_and_registered_fields(isolated_db, tmp_path, capsys, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_visual(project_dir, project_id, "scene-01")
    _register_fake_audio(project_dir, project_id, "scene-01")
    _mock_pipeline_success(monkeypatch)

    manifest_path = project_dir / "manifest.json"
    output_path = tmp_path / "clip.mp4"
    rc = cli.cmd_build_composite_scene_clip(
        _args(project_id, "scene-01", manifest_path, output=output_path, out_format="json")
    )
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert rc == 0
    assert payload["audio_muxed"] is True
    assert payload["registered"] is False


def test_cli_json_failure_never_includes_success_only_fields(isolated_db, tmp_path, capsys):
    from src.database.db import init_db

    init_db()
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")

    rc = cli.cmd_build_composite_scene_clip(
        _args("does-not-exist", "scene-01", manifest_path, out_format="json")
    )
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert rc == 1
    assert "audio_muxed" not in payload
    assert "registered" not in payload
