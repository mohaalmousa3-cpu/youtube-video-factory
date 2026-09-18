"""Tests for the `derive-text-overlays` CLI command (src/cli.py
cmd_derive_text_overlays). Same isolated_db / _create_registered_project
pattern as tests/test_cli_finalize_scene_timing.py. The core pure function
(src.core.overlay_derivation.derive_text_overlays) is mocked for the
stdout/stderr-boundary tests and exercised for real (via
finalize_scene_timing(), itself pure) for the end-to-end/lifecycle tests —
no real ffmpeg/ffprobe/provider/network call anywhere in this file, since
neither derive-text-overlays nor its inputs require real media for this
phase."""
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
MODULE = "src.core.overlay_derivation"


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


def _scene_plan_dict(scene_ids: tuple[str, ...] = ("scene-01", "scene-02")) -> dict:
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


def _create_registered_project(projects_root: Path, scene_ids: tuple[str, ...] = ("scene-01", "scene-02")):
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
    return project.project_id, manifest_path.parent, manifest


def _finalized_manifest(manifest, durations: dict[str, float]):
    from src.core.scene_timing_finalizer import finalize_scene_timing
    from src.models.artifact import ArtifactRecord

    artifacts = [
        ArtifactRecord(
            artifact_id=f"audio-{scene_id}",
            project_id=manifest.project_id,
            kind="audio",
            scene_id=scene_id,
            relative_path=f"audio/{scene_id}.wav",
            byte_size=1,
            sha256_checksum="0" * 64,
            created_at=FIXED_NOW,
            metadata={"duration_seconds": duration, "source": "external"},
        )
        for scene_id, duration in durations.items()
    ]
    return finalize_scene_timing(manifest, artifacts)


def _register_render_artifact(
    project_id, duration_seconds, *, artifact_id="render-final", relative_path="render/final.mp4"
):
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
                relative_path=relative_path,
                byte_size=2048,
                sha256_checksum="1" * 64,
                created_at=FIXED_NOW,
                metadata={"duration_seconds": duration_seconds, "source": "final-video-assembly-v1"},
            ),
        )
    finally:
        conn.close()


def _args(project_id, manifest_path, output, out_format="text"):
    return argparse.Namespace(
        project_id=project_id,
        manifest=str(manifest_path),
        output=str(output),
        format=out_format,
    )


# ---------------------------------------------------------------------
# 37-41: parser registration
# ---------------------------------------------------------------------


def test_parser_registers_derive_text_overlays():
    parser = cli.build_parser()
    args = parser.parse_args(
        ["derive-text-overlays", "proj-123", "--manifest", "m.json", "--output", "out.json"]
    )
    assert args.func is cli.cmd_derive_text_overlays
    assert args.project_id == "proj-123"
    assert args.manifest == "m.json"
    assert args.output == "out.json"
    assert args.format == "text"


def test_project_id_is_positional():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["derive-text-overlays", "--manifest", "m.json", "--output", "out.json"])


def test_manifest_is_required():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["derive-text-overlays", "proj-123", "--output", "out.json"])


def test_output_is_required():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["derive-text-overlays", "proj-123", "--manifest", "m.json"])


def test_existing_commands_remain_registered():
    parser = cli.build_parser()
    assert parser.parse_args(["health"]).func is cli.cmd_health
    assert parser.parse_args(
        ["finalize-scene-timing", "p", "--output", "o"]
    ).func is cli.cmd_finalize_scene_timing
    assert parser.parse_args(
        ["render-text-overlays", "p", "--manifest", "m", "--output", "o"]
    ).func is cli.cmd_render_text_overlays


# ---------------------------------------------------------------------
# 42-60: real end-to-end lifecycle behavior (real project/DB fixtures;
# the core derive_text_overlays() itself is pure, so no ffmpeg/ffprobe/
# provider/network call happens anywhere below)
# ---------------------------------------------------------------------


def test_cli_success_writes_derived_manifest(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    enriched_input = _finalized_manifest(manifest, {"scene-01": 5.0, "scene-02": 7.0})
    input_manifest_path = tmp_path / "input.json"
    save_manifest(enriched_input, input_manifest_path)
    _register_render_artifact(project_id, 12.0)

    output_path = tmp_path / "derived.json"
    rc = cli.cmd_derive_text_overlays(_args(project_id, input_manifest_path, output_path))
    out = capsys.readouterr().out

    assert rc == 0
    assert "derive-text-overlays: OK" in out
    assert f"project_id: {project_id}" in out
    assert "scenes_with_new_overlays: 2" in out
    assert "scenes_with_existing_overlays: 0" in out
    assert "derived_overlay_count: 2" in out
    assert "preserved_overlay_count: 0" in out
    assert "source_render_artifact_id: render-final" in out
    assert output_path.exists()


def test_cli_success_json_mode(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    enriched_input = _finalized_manifest(manifest, {"scene-01": 5.0})
    input_manifest_path = tmp_path / "input.json"
    save_manifest(enriched_input, input_manifest_path)
    _register_render_artifact(project_id, 5.0)

    output_path = tmp_path / "derived.json"
    rc = cli.cmd_derive_text_overlays(_args(project_id, input_manifest_path, output_path, out_format="json"))
    out, err = capsys.readouterr()
    payload = json.loads(out)

    assert rc == 0
    assert err == ""
    assert payload["ok"] is True
    assert payload["project_id"] == project_id
    assert payload["scenes_with_new_overlays"] == 1
    assert payload["derived_overlay_count"] == 1
    assert payload["source_render_artifact_id"] == "render-final"
    assert payload["source_render_duration_seconds"] == 5.0
    assert payload["source_fingerprint"] == manifest.source_fingerprint


def test_cli_text_domain_error_goes_to_stderr_only(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    # Not finalized — missing measured_audio_duration_seconds.
    input_manifest_path = tmp_path / "input.json"
    save_manifest(manifest, input_manifest_path)
    _register_render_artifact(project_id, 5.0)

    output_path = tmp_path / "derived.json"
    rc = cli.cmd_derive_text_overlays(_args(project_id, input_manifest_path, output_path))
    out, err = capsys.readouterr()

    assert rc != 0
    assert out == ""
    assert "derive-text-overlays: FAILED" in err
    assert not output_path.exists()


def test_cli_json_domain_error_goes_to_stderr_only(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    input_manifest_path = tmp_path / "input.json"
    save_manifest(manifest, input_manifest_path)
    _register_render_artifact(project_id, 5.0)

    output_path = tmp_path / "derived.json"
    rc = cli.cmd_derive_text_overlays(_args(project_id, input_manifest_path, output_path, out_format="json"))
    out, err = capsys.readouterr()

    assert rc != 0
    assert out == ""
    payload = json.loads(err)
    assert payload["ok"] is False
    assert payload["project_id"] == project_id


def test_cli_unexpected_exception_is_sanitized(isolated_db, tmp_path, capsys, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    enriched_input = _finalized_manifest(manifest, {"scene-01": 5.0})
    input_manifest_path = tmp_path / "input.json"
    save_manifest(enriched_input, input_manifest_path)
    _register_render_artifact(project_id, 5.0)

    secret_like = "SENTINEL-ORIGINAL-MESSAGE-never-printed-c3d9/etc/shadow"

    def _boom(*a, **k):
        raise RuntimeError(secret_like)

    monkeypatch.setattr(f"{MODULE}.derive_text_overlays", _boom)

    output_path = tmp_path / "derived.json"
    rc = cli.cmd_derive_text_overlays(_args(project_id, input_manifest_path, output_path))
    out, err = capsys.readouterr()

    assert rc != 0
    assert secret_like not in err
    assert secret_like not in out
    assert "an unexpected internal error occurred" in err
    assert not output_path.exists()


def test_cli_keyboard_interrupt_is_not_caught(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    enriched_input = _finalized_manifest(manifest, {"scene-01": 5.0})
    input_manifest_path = tmp_path / "input.json"
    save_manifest(enriched_input, input_manifest_path)
    _register_render_artifact(project_id, 5.0)

    def _boom(*a, **k):
        raise KeyboardInterrupt()

    monkeypatch.setattr(f"{MODULE}.derive_text_overlays", _boom)

    output_path = tmp_path / "derived.json"
    with pytest.raises(KeyboardInterrupt):
        cli.cmd_derive_text_overlays(_args(project_id, input_manifest_path, output_path))


def test_cli_invalid_format_rejects(isolated_db, tmp_path, capsys):
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")

    rc = cli.cmd_derive_text_overlays(
        _args("proj-x", manifest_path, tmp_path / "out.json", out_format="xml")
    )
    err = capsys.readouterr().err

    assert rc == 1
    assert "invalid --format" in err


# ---------------------------------------------------------------------
# sqlite3.Error from get_readonly_connection() must never expose raw
# exception text — same sanitized "no local project database found"
# message the final-video-assembly and text-overlay-renderer command
# paths already use.
# ---------------------------------------------------------------------

_SENTINEL_SQLITE_ERROR = "SENTINEL-SQLITE-ERROR-C:\\sensitive\\project.db"


def test_cli_text_sqlite_error_is_sanitized(tmp_path, capsys, monkeypatch):
    def _boom():
        raise sqlite3.OperationalError(_SENTINEL_SQLITE_ERROR)

    monkeypatch.setattr("src.database.db.get_readonly_connection", _boom)

    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")
    rc = cli.cmd_derive_text_overlays(_args("proj-x", manifest_path, tmp_path / "out.json"))
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
    rc = cli.cmd_derive_text_overlays(
        _args("proj-x", manifest_path, tmp_path / "out.json", out_format="json")
    )
    out, err = capsys.readouterr()

    assert rc != 0
    assert out == ""
    payload = json.loads(err)
    assert payload["ok"] is False
    assert "no local project database found" in payload["reason"]
    assert _SENTINEL_SQLITE_ERROR not in err
    for value in payload.values():
        assert _SENTINEL_SQLITE_ERROR not in str(value)


# ---------------------------------------------------------------------
# 48-50: output path conflicts
# ---------------------------------------------------------------------


def test_cli_rejects_output_equal_to_canonical_manifest_path(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    enriched_input = _finalized_manifest(manifest, {"scene-01": 5.0})
    input_manifest_path = tmp_path / "input.json"
    save_manifest(enriched_input, input_manifest_path)
    _register_render_artifact(project_id, 5.0)

    canonical_path = project_dir / "manifest.json"
    bytes_before = canonical_path.read_bytes()

    rc = cli.cmd_derive_text_overlays(_args(project_id, input_manifest_path, canonical_path))
    err = capsys.readouterr().err

    assert rc != 0
    assert "must not be the project's own canonical manifest path" in err
    assert canonical_path.read_bytes() == bytes_before


def test_cli_rejects_output_equal_to_input_manifest_path(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    enriched_input = _finalized_manifest(manifest, {"scene-01": 5.0})
    input_manifest_path = tmp_path / "input.json"
    save_manifest(enriched_input, input_manifest_path)
    _register_render_artifact(project_id, 5.0)
    bytes_before = input_manifest_path.read_bytes()

    rc = cli.cmd_derive_text_overlays(_args(project_id, input_manifest_path, input_manifest_path))
    err = capsys.readouterr().err

    assert rc != 0
    assert "must not be the same path as --manifest" in err
    assert input_manifest_path.read_bytes() == bytes_before


def test_cli_rejects_pre_existing_output_path(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    enriched_input = _finalized_manifest(manifest, {"scene-01": 5.0})
    input_manifest_path = tmp_path / "input.json"
    save_manifest(enriched_input, input_manifest_path)
    _register_render_artifact(project_id, 5.0)

    output_path = tmp_path / "derived.json"
    output_path.write_bytes(b"pre-existing content nobody asked to touch")
    bytes_before = output_path.read_bytes()

    rc = cli.cmd_derive_text_overlays(_args(project_id, input_manifest_path, output_path))
    err = capsys.readouterr().err

    assert rc != 0
    assert "already exists" in err
    assert output_path.read_bytes() == bytes_before


# ---------------------------------------------------------------------
# 51-52: render artifact resolution
# ---------------------------------------------------------------------


def test_cli_missing_render_artifact_rejected(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    enriched_input = _finalized_manifest(manifest, {"scene-01": 5.0})
    input_manifest_path = tmp_path / "input.json"
    save_manifest(enriched_input, input_manifest_path)
    # No render artifact registered at all.

    output_path = tmp_path / "derived.json"
    rc = cli.cmd_derive_text_overlays(_args(project_id, input_manifest_path, output_path))
    err = capsys.readouterr().err

    assert rc != 0
    assert "no 'render' artifact" in err
    assert not output_path.exists()


def test_cli_ambiguous_render_artifact_rejected(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    enriched_input = _finalized_manifest(manifest, {"scene-01": 5.0})
    input_manifest_path = tmp_path / "input.json"
    save_manifest(enriched_input, input_manifest_path)
    _register_render_artifact(project_id, 5.0, artifact_id="render-final-a", relative_path="render/final-a.mp4")
    _register_render_artifact(project_id, 5.0, artifact_id="render-final-b", relative_path="render/final-b.mp4")

    output_path = tmp_path / "derived.json"
    rc = cli.cmd_derive_text_overlays(_args(project_id, input_manifest_path, output_path))
    err = capsys.readouterr().err

    assert rc != 0
    assert "2 'render' artifacts" in err
    assert not output_path.exists()


# ---------------------------------------------------------------------
# 53-56: connection lifecycle / no subprocess / no provider
# ---------------------------------------------------------------------


def test_cli_opens_no_write_connection():
    source = inspect.getsource(cli.cmd_derive_text_overlays)
    assert "get_connection(" not in source


def test_cli_closes_read_connection_before_pure_derivation_call(isolated_db, tmp_path, capsys, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    enriched_input = _finalized_manifest(manifest, {"scene-01": 5.0})
    input_manifest_path = tmp_path / "input.json"
    save_manifest(enriched_input, input_manifest_path)
    _register_render_artifact(project_id, 5.0)

    from src.database import db as db_module

    real_get_readonly_connection = db_module.get_readonly_connection
    live_connections: list = []

    def _tracking_get_readonly_connection():
        conn = real_get_readonly_connection()
        live_connections.append(conn)
        return conn

    monkeypatch.setattr("src.database.db.get_readonly_connection", _tracking_get_readonly_connection)

    from src.core import overlay_derivation as derivation_module

    real_derive = derivation_module.derive_text_overlays

    def _checking_derive(manifest_arg, render_artifact_arg):
        for conn in live_connections:
            with pytest.raises(sqlite3.ProgrammingError):
                conn.execute("SELECT 1")
        return real_derive(manifest_arg, render_artifact_arg)

    monkeypatch.setattr(f"{MODULE}.derive_text_overlays", _checking_derive)

    output_path = tmp_path / "derived.json"
    rc = cli.cmd_derive_text_overlays(_args(project_id, input_manifest_path, output_path))
    capsys.readouterr()

    assert rc == 0
    assert len(live_connections) == 1


def test_cli_does_not_invoke_subprocess():
    source = inspect.getsource(cli.cmd_derive_text_overlays)
    assert "subprocess" not in source


def test_cli_imports_no_provider_module():
    source = inspect.getsource(cli.cmd_derive_text_overlays)
    for forbidden in ("llm_groq", "llm_tokenrouter", "tts_kokoro", "image_qwen", "lipsync_rhubarb"):
        assert forbidden not in source


# ---------------------------------------------------------------------
# 57-60: file/DB non-mutation proof
# ---------------------------------------------------------------------


def test_cli_input_manifest_file_unchanged_after_success(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    enriched_input = _finalized_manifest(manifest, {"scene-01": 5.0})
    input_manifest_path = tmp_path / "input.json"
    save_manifest(enriched_input, input_manifest_path)
    _register_render_artifact(project_id, 5.0)
    bytes_before = input_manifest_path.read_bytes()

    output_path = tmp_path / "derived.json"
    cli.cmd_derive_text_overlays(_args(project_id, input_manifest_path, output_path))
    capsys.readouterr()

    assert input_manifest_path.read_bytes() == bytes_before


def test_cli_output_manifest_exists_and_is_valid(isolated_db, tmp_path, capsys):
    from src.models.manifest import VideoManifest

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    enriched_input = _finalized_manifest(manifest, {"scene-01": 5.0})
    input_manifest_path = tmp_path / "input.json"
    save_manifest(enriched_input, input_manifest_path)
    _register_render_artifact(project_id, 5.0)

    output_path = tmp_path / "derived.json"
    rc = cli.cmd_derive_text_overlays(_args(project_id, input_manifest_path, output_path))
    capsys.readouterr()

    assert rc == 0
    assert output_path.exists()
    saved = VideoManifest.model_validate(json.loads(output_path.read_text(encoding="utf-8")))
    assert saved.project_id == project_id
    assert len(saved.scene_plan.scenes[0].text_overlays) == 1


def test_cli_success_changes_no_artifact_or_project_rows(isolated_db, tmp_path, capsys):
    from src.database.artifact_repository import list_artifacts_by_project
    from src.database.db import get_connection
    from src.database.project_repository import get_project

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    enriched_input = _finalized_manifest(manifest, {"scene-01": 5.0})
    input_manifest_path = tmp_path / "input.json"
    save_manifest(enriched_input, input_manifest_path)
    _register_render_artifact(project_id, 5.0)

    conn = get_connection()
    try:
        artifacts_before = list_artifacts_by_project(conn, project_id)
        project_before = get_project(conn, project_id)
    finally:
        conn.close()

    output_path = tmp_path / "derived.json"
    cli.cmd_derive_text_overlays(_args(project_id, input_manifest_path, output_path))
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

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    input_manifest_path = tmp_path / "input.json"
    save_manifest(manifest, input_manifest_path)  # not finalized -> guaranteed failure
    _register_render_artifact(project_id, 5.0)

    conn = get_connection()
    try:
        artifacts_before = list_artifacts_by_project(conn, project_id)
        project_before = get_project(conn, project_id)
    finally:
        conn.close()

    output_path = tmp_path / "derived.json"
    rc = cli.cmd_derive_text_overlays(_args(project_id, input_manifest_path, output_path))
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
