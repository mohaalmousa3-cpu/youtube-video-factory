"""Tests for the `build-upscaled-ken-burns` CLI command (src/cli.py
cmd_build_upscaled_ken_burns). Same isolated_db / _create_registered_project
pattern as tests/test_cli_finalize_scene_timing.py. Uses a real, local,
Pillow-generated PNG for the registered visual artifact (no network, no
paid provider) and the real, pure finalize_scene_timing() to build the
enriched manifest in memory. The upscale/render pipeline itself is always
mocked in these tests (src.core.ken_burns_upscale_pipeline's
health_check/upscale_image/ken_burns_clip) — no real Real-ESRGAN binary or
ffmpeg subprocess call from this file, except in the one optional,
doubly-gated integration test at the bottom."""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

import src.cli as cli
from src.core.manifest_builder import build_video_manifest
from src.core.manifest_store import save_manifest
from src.core.project_state_machine import create_initial_project, initial_transition_for
from src.core.scene_timing_finalizer import finalize_scene_timing
from src.database.project_repository import create_project
from src.models.artifact import ArtifactRecord
from src.utils.channel_config import get_channel_policy

FIXED_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
PIPELINE_MODULE = "src.core.ken_burns_upscale_pipeline"


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


def _scene_plan_dict(scene_ids: tuple[str, ...] = ("scene-01",), motion_mode: str = "static") -> dict:
    local_fallback = "static" if motion_mode == "manual_flow" else None
    return dict(
        scenes=tuple(
            dict(
                scene_id=scene_id,
                sequence=i,
                narration_text=f"Narration for {scene_id}.",
                scene_type="narration",
                narrative_beat="setup",
                visual_brief=f"Visual brief for {scene_id}.",
                motion_mode=motion_mode,
                local_fallback_motion_mode=local_fallback,
                approval_state="approved",
            )
            for i, scene_id in enumerate(scene_ids, start=1)
        ),
        role_outfits=(),
    )


def _create_registered_project(
    projects_root: Path,
    scene_ids: tuple[str, ...] = ("scene-01",),
    motion_mode: str = "static",
    story_id: str = "why-we-care-what-people-think",
):
    from src.database.db import get_connection, init_db

    manifest = build_video_manifest(
        _story_input_dict(story_id),
        _scene_plan_dict(scene_ids, motion_mode=motion_mode),
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
    return project.project_id, manifest_path.parent, manifest


def _enriched_manifest_path(tmp_path, manifest, scene_durations: dict[str, float]) -> Path:
    audio_artifacts = [
        ArtifactRecord(
            artifact_id=f"audio-{scene_id}",
            project_id=manifest.project_id,
            kind="audio",
            scene_id=scene_id,
            relative_path=f"audio/{scene_id}.wav",
            byte_size=1024,
            sha256_checksum="0" * 64,
            created_at=FIXED_NOW,
            metadata={"duration_seconds": duration, "source": "external"},
        )
        for scene_id, duration in scene_durations.items()
    ]
    enriched = finalize_scene_timing(manifest, audio_artifacts)
    path = tmp_path / "enriched.json"
    save_manifest(enriched, path)
    return path


def _register_real_visual(tmp_path, project_id, manifest, scene_id) -> None:
    from PIL import Image

    from src.core.visual_artifact_registrar import register_visual_artifact
    from src.database.db import get_connection
    from src.database.project_repository import get_project

    source = tmp_path / f"source-{scene_id}.png"
    Image.new("RGB", (16, 16), color=(200, 150, 100)).save(source, format="PNG")

    conn = get_connection()
    try:
        project = get_project(conn, project_id)
        result = register_visual_artifact(conn, project, manifest, scene_id, source, now=FIXED_NOW)
        assert result.ok, result.reasons
    finally:
        conn.close()


def _mock_pipeline_success(monkeypatch):
    monkeypatch.setattr(f"{PIPELINE_MODULE}.health_check", lambda: True)

    def _fake_upscale(src, dst, *args, **kwargs):
        Path(dst).write_bytes(b"fake-upscaled")
        return dst

    def _fake_ken_burns(image_path, duration, out_path, **kwargs):
        Path(out_path).write_bytes(b"fake-mp4-bytes")
        return out_path

    monkeypatch.setattr(f"{PIPELINE_MODULE}.upscale_image", _fake_upscale)
    monkeypatch.setattr(f"{PIPELINE_MODULE}.ken_burns_clip", _fake_ken_burns)


# ---------------------------------------------------------------------
# parser registration
# ---------------------------------------------------------------------


def test_parser_registration():
    parser = cli.build_parser()
    args = parser.parse_args(
        ["build-upscaled-ken-burns", "proj-123", "scene-01", "--manifest", "m.json", "--output", "out.mp4"]
    )
    assert args.func is cli.cmd_build_upscaled_ken_burns
    assert args.project_id == "proj-123"
    assert args.scene_id == "scene-01"
    assert args.manifest == "m.json"
    assert args.output == "out.mp4"
    assert args.format == "text"


@pytest.mark.parametrize(
    "argv",
    [
        ["build-upscaled-ken-burns", "proj-123", "scene-01", "--output", "out.mp4"],
        ["build-upscaled-ken-burns", "proj-123", "scene-01", "--manifest", "m.json"],
        ["build-upscaled-ken-burns", "proj-123", "--manifest", "m.json", "--output", "out.mp4"],
        ["build-upscaled-ken-burns", "--manifest", "m.json", "--output", "out.mp4"],
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
    _register_real_visual(tmp_path, project_id, manifest, "scene-01")
    manifest_path = _enriched_manifest_path(tmp_path, manifest, {"scene-01": 4.5})
    _mock_pipeline_success(monkeypatch)

    output_path = tmp_path / "clip.mp4"
    rc = cli.cmd_build_upscaled_ken_burns(
        argparse.Namespace(
            project_id=project_id,
            scene_id="scene-01",
            manifest=str(manifest_path),
            output=str(output_path),
            format="text",
        )
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert "build-upscaled-ken-burns: OK" in out
    assert "not registered" in out
    assert f"project_id: {project_id}" in out
    assert "scene_id: scene-01" in out
    assert str(output_path) in out
    assert output_path.exists()


def test_cli_success_json_mode(isolated_db, tmp_path, capsys, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_visual(tmp_path, project_id, manifest, "scene-01")
    manifest_path = _enriched_manifest_path(tmp_path, manifest, {"scene-01": 4.5})
    _mock_pipeline_success(monkeypatch)

    output_path = tmp_path / "clip.mp4"
    rc = cli.cmd_build_upscaled_ken_burns(
        argparse.Namespace(
            project_id=project_id,
            scene_id="scene-01",
            manifest=str(manifest_path),
            output=str(output_path),
            format="json",
        )
    )
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert rc == 0
    assert payload == {
        "ok": True,
        "project_id": project_id,
        "scene_id": "scene-01",
        "output": str(output_path),
    }


def test_cli_never_writes_to_the_database(isolated_db, tmp_path, capsys, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_visual(tmp_path, project_id, manifest, "scene-01")
    manifest_path = _enriched_manifest_path(tmp_path, manifest, {"scene-01": 4.5})
    _mock_pipeline_success(monkeypatch)

    from src.utils.config import get_settings

    db_path = get_settings().data_dir / "jobs.db"
    bytes_before = db_path.read_bytes()

    output_path = tmp_path / "clip.mp4"
    cli.cmd_build_upscaled_ken_burns(
        argparse.Namespace(
            project_id=project_id,
            scene_id="scene-01",
            manifest=str(manifest_path),
            output=str(output_path),
            format="text",
        )
    )
    capsys.readouterr()

    assert db_path.read_bytes() == bytes_before


def test_cli_closes_readonly_connection_before_pipeline_is_invoked(isolated_db, tmp_path, capsys, monkeypatch):
    """sqlite3.Connection.close is a read-only slot on the instance (can't
    be monkeypatched directly), so this proves closure indirectly: a
    closed sqlite3 connection raises ProgrammingError on any further use —
    the fake pipeline tries to use the CLI's own connection object and
    records whether that raised."""
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_visual(tmp_path, project_id, manifest, "scene-01")
    manifest_path = _enriched_manifest_path(tmp_path, manifest, {"scene-01": 4.5})

    import src.database.db as db_module

    real_get_readonly_connection = db_module.get_readonly_connection
    holder = {}

    def _spy_get_readonly_connection():
        conn = real_get_readonly_connection()
        holder["conn"] = conn
        return conn

    monkeypatch.setattr("src.database.db.get_readonly_connection", _spy_get_readonly_connection)

    was_closed_when_pipeline_ran = {"value": None}

    def _fake_pipeline(scene, source_visual_path, out_path, **kwargs):
        conn = holder["conn"]
        try:
            conn.execute("SELECT 1")
            was_closed_when_pipeline_ran["value"] = False
        except sqlite3.ProgrammingError:
            was_closed_when_pipeline_ran["value"] = True
        Path(out_path).write_bytes(b"fake-mp4-bytes")
        return out_path

    monkeypatch.setattr(f"{PIPELINE_MODULE}.build_upscaled_ken_burns_clip", _fake_pipeline)

    output_path = tmp_path / "clip.mp4"
    rc = cli.cmd_build_upscaled_ken_burns(
        argparse.Namespace(
            project_id=project_id,
            scene_id="scene-01",
            manifest=str(manifest_path),
            output=str(output_path),
            format="text",
        )
    )
    capsys.readouterr()

    assert rc == 0
    assert was_closed_when_pipeline_ran["value"] is True


# ---------------------------------------------------------------------
# rejections -> rc=1, no new output file, no artifact/DB/manifest touched
# ---------------------------------------------------------------------


def test_cli_unknown_project_returns_error(isolated_db, tmp_path, capsys):
    from src.database.db import init_db

    init_db()
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")
    output_path = tmp_path / "clip.mp4"

    rc = cli.cmd_build_upscaled_ken_burns(
        argparse.Namespace(
            project_id="does-not-exist",
            scene_id="scene-01",
            manifest=str(manifest_path),
            output=str(output_path),
            format="text",
        )
    )
    err = capsys.readouterr().err

    assert rc == 1
    assert "unknown project_id" in err
    assert not output_path.exists()


def test_cli_missing_database_has_zero_filesystem_side_effects(isolated_db, tmp_path, capsys):
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")
    output_path = tmp_path / "clip.mp4"

    # isolated_db and tmp_path are the same directory in this fixture setup
    # (both resolve to pytest's per-test tmp_path); snapshot AFTER writing
    # our own test-setup file so the assertion below proves the COMMAND
    # itself creates nothing, not that the directory is pristine.
    before = sorted(str(p) for p in isolated_db.rglob("*"))

    rc = cli.cmd_build_upscaled_ken_burns(
        argparse.Namespace(
            project_id="proj-x",
            scene_id="scene-01",
            manifest=str(manifest_path),
            output=str(output_path),
            format="text",
        )
    )
    err = capsys.readouterr().err

    assert rc == 1
    assert "no local project database found" in err
    assert sorted(str(p) for p in isolated_db.rglob("*")) == before
    assert not output_path.exists()


def test_cli_malformed_manifest_file_rejects(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("not valid json", encoding="utf-8")
    output_path = tmp_path / "clip.mp4"

    rc = cli.cmd_build_upscaled_ken_burns(
        argparse.Namespace(
            project_id=project_id,
            scene_id="scene-01",
            manifest=str(manifest_path),
            output=str(output_path),
            format="text",
        )
    )
    err = capsys.readouterr().err

    assert rc == 1
    assert "could not load manifest" in err
    assert not output_path.exists()


def test_cli_wrong_project_manifest_rejects(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    other_id, other_dir, other_manifest = _create_registered_project(
        tmp_path / "projects2", scene_ids=("scene-01",), story_id="a-completely-different-story"
    )
    manifest_path = _enriched_manifest_path(tmp_path, other_manifest, {"scene-01": 3.0})
    output_path = tmp_path / "clip.mp4"

    rc = cli.cmd_build_upscaled_ken_burns(
        argparse.Namespace(
            project_id=project_id,
            scene_id="scene-01",
            manifest=str(manifest_path),
            output=str(output_path),
            format="text",
        )
    )
    err = capsys.readouterr().err

    assert rc == 1
    assert "does not match project_id" in err
    assert not output_path.exists()


def test_cli_stale_fingerprint_manifest_rejects(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = _enriched_manifest_path(tmp_path, manifest, {"scene-01": 3.0})
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["source_fingerprint"] = "0" * 64
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    output_path = tmp_path / "clip.mp4"

    rc = cli.cmd_build_upscaled_ken_burns(
        argparse.Namespace(
            project_id=project_id,
            scene_id="scene-01",
            manifest=str(manifest_path),
            output=str(output_path),
            format="text",
        )
    )
    err = capsys.readouterr().err

    assert rc == 1
    assert "source_fingerprint does not match" in err
    assert not output_path.exists()


def test_cli_unknown_scene_rejects(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = _enriched_manifest_path(tmp_path, manifest, {"scene-01": 3.0})
    output_path = tmp_path / "clip.mp4"

    rc = cli.cmd_build_upscaled_ken_burns(
        argparse.Namespace(
            project_id=project_id,
            scene_id="scene-does-not-exist",
            manifest=str(manifest_path),
            output=str(output_path),
            format="text",
        )
    )
    err = capsys.readouterr().err

    assert rc == 1
    assert "unknown scene_id" in err
    assert not output_path.exists()


def test_cli_no_visual_artifact_rejects(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = _enriched_manifest_path(tmp_path, manifest, {"scene-01": 3.0})
    output_path = tmp_path / "clip.mp4"

    rc = cli.cmd_build_upscaled_ken_burns(
        argparse.Namespace(
            project_id=project_id,
            scene_id="scene-01",
            manifest=str(manifest_path),
            output=str(output_path),
            format="text",
        )
    )
    err = capsys.readouterr().err

    assert rc == 1
    assert "no registered visual artifact found" in err
    assert not output_path.exists()


def test_cli_duplicate_visual_artifact_rejects(isolated_db, tmp_path, capsys):
    from src.database.artifact_repository import register_artifact
    from src.database.db import get_connection

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_visual(tmp_path, project_id, manifest, "scene-01")
    manifest_path = _enriched_manifest_path(tmp_path, manifest, {"scene-01": 3.0})

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

    output_path = tmp_path / "clip.mp4"
    rc = cli.cmd_build_upscaled_ken_burns(
        argparse.Namespace(
            project_id=project_id,
            scene_id="scene-01",
            manifest=str(manifest_path),
            output=str(output_path),
            format="text",
        )
    )
    err = capsys.readouterr().err

    assert rc == 1
    assert "multiple registered visual artifacts" in err
    assert not output_path.exists()


def test_cli_missing_visual_file_on_disk_rejects(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_visual(tmp_path, project_id, manifest, "scene-01")
    manifest_path = _enriched_manifest_path(tmp_path, manifest, {"scene-01": 3.0})

    (project_dir / "visuals" / "scene-01.png").unlink()

    output_path = tmp_path / "clip.mp4"
    rc = cli.cmd_build_upscaled_ken_burns(
        argparse.Namespace(
            project_id=project_id,
            scene_id="scene-01",
            manifest=str(manifest_path),
            output=str(output_path),
            format="text",
        )
    )
    err = capsys.readouterr().err

    assert rc == 1
    assert "missing on disk" in err
    assert not output_path.exists()


def test_cli_manual_flow_motion_mode_rejects(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(
        tmp_path / "projects", motion_mode="manual_flow"
    )
    _register_real_visual(tmp_path, project_id, manifest, "scene-01")
    manifest_path = _enriched_manifest_path(tmp_path, manifest, {"scene-01": 3.0})

    output_path = tmp_path / "clip.mp4"
    rc = cli.cmd_build_upscaled_ken_burns(
        argparse.Namespace(
            project_id=project_id,
            scene_id="scene-01",
            manifest=str(manifest_path),
            output=str(output_path),
            format="text",
        )
    )
    err = capsys.readouterr().err

    assert rc == 1
    assert "manual_flow" in err
    assert not output_path.exists()


def test_cli_invalid_timing_rejects(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_visual(tmp_path, project_id, manifest, "scene-01")
    # Enriched manifest built with NO audio artifacts at all -> finalize_scene_timing
    # itself would raise, so instead save the CANONICAL (un-enriched) manifest as
    # --manifest, whose measured_audio_duration_seconds is still None.
    manifest_path = tmp_path / "unenriched.json"
    save_manifest(manifest, manifest_path)

    output_path = tmp_path / "clip.mp4"
    rc = cli.cmd_build_upscaled_ken_burns(
        argparse.Namespace(
            project_id=project_id,
            scene_id="scene-01",
            manifest=str(manifest_path),
            output=str(output_path),
            format="text",
        )
    )
    err = capsys.readouterr().err

    assert rc == 1
    assert "measured_audio_duration_seconds" in err
    assert not output_path.exists()


def test_cli_unavailable_upscaler_rejects(isolated_db, tmp_path, capsys, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_visual(tmp_path, project_id, manifest, "scene-01")
    manifest_path = _enriched_manifest_path(tmp_path, manifest, {"scene-01": 3.0})

    monkeypatch.setattr(f"{PIPELINE_MODULE}.health_check", lambda: False)

    def _boom(*args, **kwargs):
        raise AssertionError("must not be called")

    monkeypatch.setattr(f"{PIPELINE_MODULE}.upscale_image", _boom)
    monkeypatch.setattr(f"{PIPELINE_MODULE}.ken_burns_clip", _boom)

    output_path = tmp_path / "clip.mp4"
    rc = cli.cmd_build_upscaled_ken_burns(
        argparse.Namespace(
            project_id=project_id,
            scene_id="scene-01",
            manifest=str(manifest_path),
            output=str(output_path),
            format="text",
        )
    )
    err = capsys.readouterr().err

    assert rc == 1
    assert "unavailable" in err
    assert not output_path.exists()


def test_cli_existing_output_rejects_and_leaves_it_byte_identical(isolated_db, tmp_path, capsys, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_visual(tmp_path, project_id, manifest, "scene-01")
    manifest_path = _enriched_manifest_path(tmp_path, manifest, {"scene-01": 3.0})
    _mock_pipeline_success(monkeypatch)

    output_path = tmp_path / "clip.mp4"
    output_path.write_bytes(b"pre-existing content nobody asked to touch")
    bytes_before = output_path.read_bytes()

    rc = cli.cmd_build_upscaled_ken_burns(
        argparse.Namespace(
            project_id=project_id,
            scene_id="scene-01",
            manifest=str(manifest_path),
            output=str(output_path),
            format="text",
        )
    )
    err = capsys.readouterr().err

    assert rc == 1
    assert "already exists" in err
    assert output_path.read_bytes() == bytes_before


def test_cli_json_failure_has_stable_shape(isolated_db, tmp_path, capsys):
    from src.database.db import init_db

    init_db()
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")
    output_path = tmp_path / "clip.mp4"

    rc = cli.cmd_build_upscaled_ken_burns(
        argparse.Namespace(
            project_id="does-not-exist",
            scene_id="scene-01",
            manifest=str(manifest_path),
            output=str(output_path),
            format="json",
        )
    )
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert rc == 1
    assert payload["ok"] is False
    assert payload["project_id"] == "does-not-exist"
    assert payload["scene_id"] == "scene-01"
    assert isinstance(payload["reason"], str) and payload["reason"]


# ---------------------------------------------------------------------
# optional integration test — real health_check() gates on both sides;
# skips cleanly in this checkout (Real-ESRGAN/model assets are absent)
# ---------------------------------------------------------------------


def test_real_pipeline_integration(isolated_db, tmp_path, capsys):
    from src.providers import image_upscale
    from src.render import ffmpeg_render

    if not (ffmpeg_render.health_check() and image_upscale.health_check()):
        pytest.skip("real ffmpeg and/or Real-ESRGAN not available in this environment")

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_visual(tmp_path, project_id, manifest, "scene-01")
    manifest_path = _enriched_manifest_path(tmp_path, manifest, {"scene-01": 1.0})

    output_path = tmp_path / "clip.mp4"
    rc = cli.cmd_build_upscaled_ken_burns(
        argparse.Namespace(
            project_id=project_id,
            scene_id="scene-01",
            manifest=str(manifest_path),
            output=str(output_path),
            format="text",
        )
    )
    capsys.readouterr()

    assert rc == 0
    assert output_path.exists()
    assert ffmpeg_render.get_duration_seconds(output_path) == pytest.approx(1.0, abs=0.3)
