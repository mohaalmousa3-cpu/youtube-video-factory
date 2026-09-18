"""Tests for the `build-final-local` CLI command (src/cli.py
cmd_build_final_local). Same isolated_db / _create_registered_project
pattern as tests/test_cli_plan_final_assembly.py. All underlying stage
functions are exercised through the real, already-tested
src.core.local_resume_orchestrator.build_final_local() — only
assemble_final_video/render_text_overlays are faked here (same fakes as
tests/test_local_resume_orchestrator.py) so no ffmpeg/ffprobe/provider/
network call happens anywhere in this file."""
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
from src.core.scene_timing_finalizer import finalize_scene_timing
from src.database.artifact_repository import register_artifact
from src.database.project_repository import create_project
from src.models.artifact import ArtifactRecord
from src.utils.channel_config import get_channel_policy

FIXED_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
ORCHESTRATOR_MODULE = "src.core.local_resume_orchestrator"


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


def _create_registered_project(projects_root: Path, scene_ids: tuple[str, ...] = ("scene-01",)):
    from src.database.db import get_connection, init_db

    manifest = build_video_manifest(
        _story_input_dict(), _scene_plan_dict(scene_ids), get_channel_policy(), created_at=FIXED_NOW
    )
    audio_artifacts = [
        ArtifactRecord(
            artifact_id=f"audio-{sid}",
            project_id=manifest.project_id,
            kind="audio",
            scene_id=sid,
            relative_path=f"audio/{sid}.wav",
            byte_size=1,
            sha256_checksum="0" * 64,
            created_at=FIXED_NOW,
            metadata={"duration_seconds": 5.0, "source": "external"},
        )
        for sid in scene_ids
    ]
    manifest = finalize_scene_timing(manifest, audio_artifacts)

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


def _register_render_artifact(project_id, duration_seconds=5.0, *, artifact_id="render-final"):
    from src.database.db import get_connection

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


def _register_overlay_render_artifact(project_id, render_artifact_id, manifest_fingerprint, overlay_count=0):
    from src.database.db import get_connection

    conn = get_connection()
    try:
        register_artifact(
            conn,
            ArtifactRecord(
                artifact_id="overlay-render-final",
                project_id=project_id,
                kind="overlay_render",
                scene_id=None,
                relative_path="overlay_render/final.mp4",
                byte_size=2048,
                sha256_checksum="2" * 64,
                created_at=FIXED_NOW,
                metadata={
                    "duration_seconds": 5.0,
                    "source": "text-overlay-renderer-v1",
                    "overlay_count": overlay_count,
                    "source_render_artifact_id": render_artifact_id,
                    "source_render_sha256": "1" * 64,
                    "source_render_relative_path": "render/final.mp4",
                    "manifest_fingerprint": manifest_fingerprint,
                },
            ),
        )
    finally:
        conn.close()


def _fake_assemble_final_video():
    def _fake(project_id, manifest_path, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-render-bytes")
        _register_render_artifact(project_id)
        return None

    return _fake


def _fake_render_text_overlays():
    def _fake(project_id, manifest_path, output_path):
        from src.core.manifest_store import load_manifest

        manifest = load_manifest(Path(manifest_path))
        overlay_count = sum(len(s.text_overlays) for s in manifest.scene_plan.scenes)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-overlay-bytes")
        _register_overlay_render_artifact(project_id, "render-final", manifest.source_fingerprint, overlay_count)
        return None

    return _fake


def _args(project_id, timed_manifest, derived, render_out, overlay_out, out_format="text"):
    return argparse.Namespace(
        project_id=project_id,
        timed_manifest=str(timed_manifest),
        derived_manifest_output=str(derived),
        render_output=str(render_out),
        overlay_output=str(overlay_out),
        format=out_format,
    )


# ---------------------------------------------------------------------
# parser registration
# ---------------------------------------------------------------------


def test_parser_registers_build_final_local():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "build-final-local", "proj-123",
            "--timed-manifest", "t.json",
            "--derived-manifest-output", "d.json",
            "--render-output", "r.mp4",
            "--overlay-output", "o.mp4",
        ]
    )
    assert args.func is cli.cmd_build_final_local
    assert args.project_id == "proj-123"
    assert args.format == "text"


def test_project_id_is_positional():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "build-final-local",
                "--timed-manifest", "t.json", "--derived-manifest-output", "d.json",
                "--render-output", "r.mp4", "--overlay-output", "o.mp4",
            ]
        )


def test_all_four_paths_are_required():
    parser = cli.build_parser()
    for missing in ("--timed-manifest", "--derived-manifest-output", "--render-output", "--overlay-output"):
        full = {
            "--timed-manifest": "t.json",
            "--derived-manifest-output": "d.json",
            "--render-output": "r.mp4",
            "--overlay-output": "o.mp4",
        }
        del full[missing]
        argv = ["build-final-local", "proj-123"]
        for k, v in full.items():
            argv += [k, v]
        with pytest.raises(SystemExit):
            parser.parse_args(argv)


def test_existing_commands_remain_registered():
    parser = cli.build_parser()
    assert parser.parse_args(["health"]).func is cli.cmd_health
    assert parser.parse_args(
        ["plan-final-assembly", "p", "--manifest", "m"]
    ).func is cli.cmd_plan_final_assembly
    assert parser.parse_args(
        ["assemble-final-video", "p", "--manifest", "m", "--output", "o"]
    ).func is cli.cmd_assemble_final_video


# ---------------------------------------------------------------------
# success output
# ---------------------------------------------------------------------


def test_cli_full_success_text_output(isolated_db, tmp_path, capsys, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    monkeypatch.setattr(f"{ORCHESTRATOR_MODULE}.assemble_final_video", _fake_assemble_final_video())
    monkeypatch.setattr(f"{ORCHESTRATOR_MODULE}.render_text_overlays", _fake_render_text_overlays())

    rc = cli.cmd_build_final_local(
        _args(
            project_id, project_dir / "manifest.json",
            tmp_path / "d.json", tmp_path / "r.mp4", tmp_path / "o.mp4",
        )
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert "build-final-local: OK" in out
    for field in (
        "render_status", "render_artifact_id", "render_output_path",
        "derived_manifest_status", "derived_manifest_path",
        "overlay_render_status", "overlay_render_artifact_id", "overlay_render_output_path",
        "stopped_at_stage", "blocked_reasons", "notes",
    ):
        assert f"{field}:" in out


def test_cli_full_success_json_output(isolated_db, tmp_path, capsys, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    monkeypatch.setattr(f"{ORCHESTRATOR_MODULE}.assemble_final_video", _fake_assemble_final_video())
    monkeypatch.setattr(f"{ORCHESTRATOR_MODULE}.render_text_overlays", _fake_render_text_overlays())

    rc = cli.cmd_build_final_local(
        _args(
            project_id, project_dir / "manifest.json",
            tmp_path / "d.json", tmp_path / "r.mp4", tmp_path / "o.mp4",
            out_format="json",
        )
    )
    out, err = capsys.readouterr()
    payload = json.loads(out)

    assert rc == 0
    assert err == ""
    assert payload["ok"] is True
    assert payload["stopped_at_stage"] is None


def test_cli_partial_result_still_prints_payload_with_nonzero_exit(isolated_db, tmp_path, capsys, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")

    def _boom(project_id, manifest_path, output_path):
        from src.core.final_video_assembly import FinalVideoAssemblyError

        raise FinalVideoAssemblyError("assembly exploded")

    monkeypatch.setattr(f"{ORCHESTRATOR_MODULE}.assemble_final_video", _boom)

    rc = cli.cmd_build_final_local(
        _args(
            project_id, project_dir / "manifest.json",
            tmp_path / "d.json", tmp_path / "r.mp4", tmp_path / "o.mp4",
            out_format="json",
        )
    )
    out, err = capsys.readouterr()
    payload = json.loads(out)

    assert rc != 0
    assert err == ""
    assert payload["ok"] is False
    assert payload["render_status"] == "failed"
    assert payload["stopped_at_stage"] == "assemble_final_video"


# ---------------------------------------------------------------------
# domain errors (up-front LocalResumeOrchestratorError)
# ---------------------------------------------------------------------


def test_cli_missing_project_error_to_stderr_only(isolated_db, tmp_path, capsys):
    from src.database.db import init_db

    init_db()
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")

    rc = cli.cmd_build_final_local(
        _args("does-not-exist", manifest_path, tmp_path / "d.json", tmp_path / "r.mp4", tmp_path / "o.mp4")
    )
    out, err = capsys.readouterr()

    assert rc != 0
    assert out == ""
    assert "unknown project_id" in err


def test_cli_json_domain_error_to_stderr_only(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")

    rc = cli.cmd_build_final_local(
        _args(
            project_id, project_dir / "manifest.json",
            project_dir / "manifest.json",  # collides with --timed-manifest
            tmp_path / "r.mp4", tmp_path / "o.mp4",
            out_format="json",
        )
    )
    out, err = capsys.readouterr()

    assert rc != 0
    assert out == ""
    payload = json.loads(err)
    assert payload["ok"] is False
    assert payload["project_id"] == project_id


# ---------------------------------------------------------------------
# sqlite sentinel sanitization
# ---------------------------------------------------------------------

_SENTINEL_SQLITE_ERROR = "SENTINEL-SQLITE-ERROR-C:\\sensitive\\project.db"


def test_cli_text_sqlite_error_is_sanitized(tmp_path, capsys, monkeypatch):
    def _boom():
        raise sqlite3.OperationalError(_SENTINEL_SQLITE_ERROR)

    monkeypatch.setattr("src.database.db.get_readonly_connection", _boom)

    rc = cli.cmd_build_final_local(
        _args("proj-x", tmp_path / "t.json", tmp_path / "d.json", tmp_path / "r.mp4", tmp_path / "o.mp4")
    )
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

    rc = cli.cmd_build_final_local(
        _args(
            "proj-x", tmp_path / "t.json", tmp_path / "d.json", tmp_path / "r.mp4", tmp_path / "o.mp4",
            out_format="json",
        )
    )
    out, err = capsys.readouterr()

    assert rc != 0
    assert out == ""
    payload = json.loads(err)
    assert payload["ok"] is False
    assert "no local project database found" in payload["reason"]
    for value in payload.values():
        assert _SENTINEL_SQLITE_ERROR not in str(value)


# ---------------------------------------------------------------------
# connection lifecycle / no subprocess / no provider
# ---------------------------------------------------------------------


def test_cli_opens_no_connection_of_its_own():
    source = inspect.getsource(cli.cmd_build_final_local)
    assert "get_connection(" not in source
    assert "get_readonly_connection(" not in source


def test_cli_does_not_invoke_subprocess():
    source = inspect.getsource(cli.cmd_build_final_local)
    assert "subprocess" not in source


def test_cli_imports_no_provider_module():
    source = inspect.getsource(cli.cmd_build_final_local)
    for forbidden in ("llm_groq", "llm_tokenrouter", "tts_kokoro", "image_qwen", "lipsync_rhubarb"):
        assert forbidden not in source


def test_cli_invalid_format_rejected(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    rc = cli.cmd_build_final_local(
        _args(
            project_id, project_dir / "manifest.json",
            tmp_path / "d.json", tmp_path / "r.mp4", tmp_path / "o.mp4",
            out_format="xml",
        )
    )
    out, err = capsys.readouterr()
    assert rc != 0
    assert out == ""
    assert "invalid --format" in err


def test_cli_unexpected_exception_sanitized(isolated_db, tmp_path, capsys, monkeypatch):
    def _boom(**kwargs):
        raise RuntimeError("SENTINEL-unexpected-secret-detail")

    monkeypatch.setattr(f"{ORCHESTRATOR_MODULE}.build_final_local", _boom)
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")

    rc = cli.cmd_build_final_local(
        _args(project_id, project_dir / "manifest.json", tmp_path / "d.json", tmp_path / "r.mp4", tmp_path / "o.mp4")
    )
    out, err = capsys.readouterr()

    assert rc != 0
    assert out == ""
    assert "an unexpected internal error occurred" in err
    assert "SENTINEL-unexpected-secret-detail" not in err


def test_cli_keyboard_interrupt_not_caught(isolated_db, tmp_path, monkeypatch):
    def _boom(**kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(f"{ORCHESTRATOR_MODULE}.build_final_local", _boom)
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")

    with pytest.raises(KeyboardInterrupt):
        cli.cmd_build_final_local(
            _args(
                project_id, project_dir / "manifest.json",
                tmp_path / "d.json", tmp_path / "r.mp4", tmp_path / "o.mp4",
            )
        )
