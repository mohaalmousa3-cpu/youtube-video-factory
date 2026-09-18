"""Tests for the `verify-final-output` CLI command (src/cli.py
cmd_verify_final_output). Same isolated_db / _create_registered_project
pattern as tests/test_cli_plan_final_assembly.py. The two ffprobe probe
functions inside src.core.final_qc_gate are monkeypatched throughout (same
approach as tests/test_final_qc_gate.py), so no ffmpeg/ffprobe/provider/
network call happens anywhere in this file."""
from __future__ import annotations

import argparse
import hashlib
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
GATE_MODULE = "src.core.final_qc_gate"


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    from src.utils import config

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    config.get_settings.cache_clear()
    yield tmp_path
    config.get_settings.cache_clear()


@pytest.fixture(autouse=True)
def default_probe(monkeypatch):
    monkeypatch.setattr(f"{GATE_MODULE}._probe_stream_types", lambda path: frozenset({"video", "audio"}))
    monkeypatch.setattr(f"{GATE_MODULE}._probe_duration_seconds", lambda path: 5.0)


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


def _register_artifact(record) -> None:
    from src.database.artifact_repository import register_artifact
    from src.database.db import get_connection

    conn = get_connection()
    try:
        register_artifact(conn, record)
    finally:
        conn.close()


def _write_file(path: Path, content: bytes = b"dummy-bytes") -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return len(content), hashlib.sha256(content).hexdigest()


def _register_valid_render(project_dir: Path, project_id: str, *, duration_seconds=5.0, artifact_id="render-final"):
    from src.models.artifact import ArtifactRecord

    size, checksum = _write_file(project_dir / "render" / "final.mp4")
    record = ArtifactRecord(
        artifact_id=artifact_id,
        project_id=project_id,
        kind="render",
        scene_id=None,
        relative_path="render/final.mp4",
        byte_size=size,
        sha256_checksum=checksum,
        created_at=FIXED_NOW,
        metadata={"duration_seconds": duration_seconds, "source": "final-video-assembly-v1"},
    )
    _register_artifact(record)
    return record


def _register_fake_render(project_id: str, *, duration_seconds=5.0, artifact_id="render-final"):
    """A registered 'render' artifact whose file is never written — used
    only for CLI-plumbing tests that don't care whether QC itself passes."""
    from src.models.artifact import ArtifactRecord

    record = ArtifactRecord(
        artifact_id=artifact_id,
        project_id=project_id,
        kind="render",
        scene_id=None,
        relative_path="render/final.mp4",
        byte_size=1,
        sha256_checksum="1" * 64,
        created_at=FIXED_NOW,
        metadata={"duration_seconds": duration_seconds, "source": "final-video-assembly-v1"},
    )
    _register_artifact(record)
    return record


def _args(project_id, manifest_path, report_output, *, require_overlays="false", out_format="text"):
    return argparse.Namespace(
        project_id=project_id,
        manifest=str(manifest_path),
        report_output=str(report_output),
        require_overlays=require_overlays,
        format=out_format,
    )


# ---------------------------------------------------------------------
# 54-59: parser registration
# ---------------------------------------------------------------------


def test_parser_registers_verify_final_output():
    parser = cli.build_parser()
    args = parser.parse_args(
        ["verify-final-output", "proj-123", "--manifest", "m.json", "--report-output", "r.json"]
    )
    assert args.func is cli.cmd_verify_final_output
    assert args.project_id == "proj-123"
    assert args.require_overlays == "false"
    assert args.format == "text"


def test_project_id_is_positional():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["verify-final-output", "--manifest", "m.json", "--report-output", "r.json"])


def test_manifest_is_required():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["verify-final-output", "proj-123", "--report-output", "r.json"])


def test_report_output_is_required():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["verify-final-output", "proj-123", "--manifest", "m.json"])


def test_require_overlays_true_false_parsing():
    parser = cli.build_parser()
    args_true = parser.parse_args(
        ["verify-final-output", "p", "--manifest", "m", "--report-output", "r", "--require-overlays", "true"]
    )
    assert args_true.require_overlays == "true"
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["verify-final-output", "p", "--manifest", "m", "--report-output", "r", "--require-overlays", "maybe"]
        )


def test_existing_commands_remain_registered():
    parser = cli.build_parser()
    assert parser.parse_args(["health"]).func is cli.cmd_health
    assert parser.parse_args(
        ["plan-final-assembly", "p", "--manifest", "m"]
    ).func is cli.cmd_plan_final_assembly
    assert parser.parse_args(
        ["build-final-local", "p", "--timed-manifest", "a", "--derived-manifest-output", "b",
         "--render-output", "c", "--overlay-output", "d"]
    ).func is cli.cmd_build_final_local
    assert parser.parse_args(
        ["register-qc-report-artifact", "p", "--file", "f"]
    ).func is cli.cmd_register_qc_report_artifact


# ---------------------------------------------------------------------
# 60-63: pass/fail output behavior
# ---------------------------------------------------------------------


def test_passed_text_report_to_stdout_exit_zero(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_valid_render(project_dir, project_id)
    report_output = tmp_path / "report.json"

    rc = cli.cmd_verify_final_output(_args(project_id, project_dir / "manifest.json", report_output))
    out, err = capsys.readouterr()

    assert rc == 0
    assert err == ""
    assert "verify-final-output: PASS" in out
    assert report_output.exists()


def test_passed_json_report_to_stdout_exit_zero(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_valid_render(project_dir, project_id)
    report_output = tmp_path / "report.json"

    rc = cli.cmd_verify_final_output(
        _args(project_id, project_dir / "manifest.json", report_output, out_format="json")
    )
    out, err = capsys.readouterr()

    assert rc == 0
    assert err == ""
    payload = json.loads(out)
    assert payload["passed"] is True
    assert report_output.exists()


def test_failed_qc_text_report_stdout_report_exists_nonzero(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    # no render registered at all -> QC report fails, but is still produced
    report_output = tmp_path / "report.json"

    rc = cli.cmd_verify_final_output(_args(project_id, project_dir / "manifest.json", report_output))
    out, err = capsys.readouterr()

    assert rc != 0
    assert err == ""
    assert "verify-final-output: FAIL" in out
    assert report_output.exists()
    payload = json.loads(report_output.read_text(encoding="utf-8"))
    assert payload["passed"] is False


def test_failed_qc_json_report_stdout_report_exists_nonzero(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    report_output = tmp_path / "report.json"

    rc = cli.cmd_verify_final_output(
        _args(project_id, project_dir / "manifest.json", report_output, out_format="json")
    )
    out, err = capsys.readouterr()

    assert rc != 0
    assert err == ""
    payload = json.loads(out)
    assert payload["passed"] is False
    assert report_output.exists()


def test_invalid_input_error_to_stderr_no_report_file(isolated_db, tmp_path, capsys):
    from src.database.db import init_db

    init_db()
    report_output = tmp_path / "report.json"
    rc = cli.cmd_verify_final_output(_args("does-not-exist", tmp_path / "m.json", report_output))
    out, err = capsys.readouterr()

    assert rc != 0
    assert out == ""
    assert "no project found" in err
    assert not report_output.exists()


# ---------------------------------------------------------------------
# 65-67: sanitized failures
# ---------------------------------------------------------------------

_SENTINEL_SQLITE_ERROR = "SENTINEL-SQLITE-ERROR-C:\\sensitive\\project.db"


def test_cli_text_sqlite_error_is_sanitized(tmp_path, capsys, monkeypatch):
    def _boom():
        raise sqlite3.OperationalError(_SENTINEL_SQLITE_ERROR)

    monkeypatch.setattr("src.database.db.get_readonly_connection", _boom)
    report_output = tmp_path / "report.json"

    rc = cli.cmd_verify_final_output(_args("proj-x", tmp_path / "m.json", report_output))
    out, err = capsys.readouterr()

    assert rc != 0
    assert out == ""
    assert "no local project database found" in err
    assert _SENTINEL_SQLITE_ERROR not in err
    assert not report_output.exists()


def test_cli_json_sqlite_error_is_sanitized(tmp_path, capsys, monkeypatch):
    def _boom():
        raise sqlite3.OperationalError(_SENTINEL_SQLITE_ERROR)

    monkeypatch.setattr("src.database.db.get_readonly_connection", _boom)
    report_output = tmp_path / "report.json"

    rc = cli.cmd_verify_final_output(
        _args("proj-x", tmp_path / "m.json", report_output, out_format="json")
    )
    out, err = capsys.readouterr()

    assert rc != 0
    assert out == ""
    payload = json.loads(err)
    assert payload["ok"] is False
    assert "no local project database found" in payload["reason"]
    for value in payload.values():
        assert _SENTINEL_SQLITE_ERROR not in str(value)


def test_manifest_project_mismatch_sanitized(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    other_id, other_dir, other_manifest = _create_registered_project(
        tmp_path / "projects2", ("scene-01",), story_id="a-completely-different-story"
    )
    report_output = tmp_path / "report.json"

    rc = cli.cmd_verify_final_output(_args(project_id, other_dir / "manifest.json", report_output))
    err = capsys.readouterr().err

    assert rc != 0
    assert "project_id" in err or "fingerprint" in err
    assert not report_output.exists()


# ---------------------------------------------------------------------
# 68-73: report-output collision protection
# ---------------------------------------------------------------------


def test_report_output_existing_path_rejected_and_preserved(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    report_output = tmp_path / "report.json"
    report_output.write_text('{"sentinel": true}', encoding="utf-8")
    before = report_output.read_bytes()

    rc = cli.cmd_verify_final_output(_args(project_id, project_dir / "manifest.json", report_output))
    err = capsys.readouterr().err

    assert rc != 0
    assert "--report-output already exists" in err
    assert report_output.read_bytes() == before


def test_report_output_alias_canonical_manifest_rejected(isolated_db, tmp_path, capsys):
    # The canonical manifest is necessarily an already-existing real file,
    # so aliasing it always also trips the (checked first) "already exists"
    # rejection — the safety property under test is that it is rejected
    # and never overwritten, regardless of which check message fires.
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    before = manifest_path.read_bytes()

    rc = cli.cmd_verify_final_output(_args(project_id, manifest_path, manifest_path))
    err = capsys.readouterr().err

    assert rc != 0
    assert "already exists" in err or "alias" in err
    assert manifest_path.read_bytes() == before


def test_report_output_alias_supplied_manifest_rejected(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    other_manifest = tmp_path / "other-manifest.json"
    save_manifest(manifest, other_manifest)
    before = other_manifest.read_bytes()

    rc = cli.cmd_verify_final_output(_args(project_id, other_manifest, other_manifest))
    err = capsys.readouterr().err

    assert rc != 0
    assert "already exists" in err or "alias" in err
    assert other_manifest.read_bytes() == before


def test_report_output_alias_render_path_rejected(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    report_output = project_dir / "render" / "final.mp4"

    rc = cli.cmd_verify_final_output(_args(project_id, project_dir / "manifest.json", report_output))
    err = capsys.readouterr().err

    assert rc != 0
    assert "alias" in err


def test_report_output_alias_overlay_render_path_rejected(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    report_output = project_dir / "overlay_render" / "final.mp4"

    rc = cli.cmd_verify_final_output(_args(project_id, project_dir / "manifest.json", report_output))
    err = capsys.readouterr().err

    assert rc != 0
    assert "alias" in err


def test_report_output_alias_registered_scene_artifact_rejected(isolated_db, tmp_path, capsys):
    from src.models.artifact import ArtifactRecord

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_artifact(
        ArtifactRecord(
            artifact_id="audio-scene-01",
            project_id=project_id,
            kind="audio",
            scene_id="scene-01",
            relative_path="audio/scene-01.wav",
            byte_size=1,
            sha256_checksum="9" * 64,
            created_at=FIXED_NOW,
            metadata={"duration_seconds": 5.0, "source": "external"},
        )
    )
    report_output = project_dir / "audio" / "scene-01.wav"

    rc = cli.cmd_verify_final_output(_args(project_id, project_dir / "manifest.json", report_output))
    err = capsys.readouterr().err

    assert rc != 0
    assert "alias" in err


# ---------------------------------------------------------------------
# 74-80: atomic write / connection lifecycle / safety
# ---------------------------------------------------------------------


def test_atomic_write_leaves_no_partial_file_on_write_failure(isolated_db, tmp_path, capsys, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_valid_render(project_dir, project_id)
    report_output = tmp_path / "nested" / "report.json"

    def _boom(src, dst):
        raise OSError("disk full (simulated)")

    monkeypatch.setattr("src.cli.os.replace", _boom)

    rc = cli.cmd_verify_final_output(_args(project_id, project_dir / "manifest.json", report_output))
    err = capsys.readouterr().err

    assert rc != 0
    assert "could not write" in err
    assert not report_output.exists()
    leftover_tmp_files = list(report_output.parent.glob(".*.tmp")) if report_output.parent.exists() else []
    assert leftover_tmp_files == []


def test_db_connection_closed_before_verification_ffprobe_and_write(isolated_db, tmp_path, capsys, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_valid_render(project_dir, project_id)
    report_output = tmp_path / "report.json"

    from src.database import db as db_module

    real_get_readonly_connection = db_module.get_readonly_connection
    live_connections: list = []

    def _tracking_get_readonly_connection():
        conn = real_get_readonly_connection()
        live_connections.append(conn)
        return conn

    monkeypatch.setattr("src.database.db.get_readonly_connection", _tracking_get_readonly_connection)

    from src.core import final_qc_gate as gate_module

    real_verify = gate_module.verify_final_output

    def _checking_verify(**kwargs):
        import sqlite3 as _sqlite3

        for conn in live_connections:
            with pytest.raises(_sqlite3.ProgrammingError):
                conn.execute("SELECT 1")
        return real_verify(**kwargs)

    monkeypatch.setattr(f"{GATE_MODULE}.verify_final_output", _checking_verify)

    rc = cli.cmd_verify_final_output(_args(project_id, project_dir / "manifest.json", report_output))
    capsys.readouterr()

    assert rc == 0
    assert len(live_connections) == 1


def test_cli_opens_no_write_connection():
    source = inspect.getsource(cli.cmd_verify_final_output)
    assert "get_connection(" not in source


def test_cli_imports_no_provider_module():
    source = inspect.getsource(cli.cmd_verify_final_output)
    for forbidden in ("llm_groq", "llm_tokenrouter", "tts_kokoro", "image_qwen", "lipsync_rhubarb"):
        assert forbidden not in source


def test_cli_makes_no_network_call():
    source = inspect.getsource(cli.cmd_verify_final_output)
    for forbidden in ("requests.", "urllib", "http.client", "socket."):
        assert forbidden not in source


def test_keyboard_interrupt_not_caught(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")

    def _boom(**kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(f"{GATE_MODULE}.verify_final_output", _boom)
    report_output = tmp_path / "report.json"

    with pytest.raises(KeyboardInterrupt):
        cli.cmd_verify_final_output(_args(project_id, project_dir / "manifest.json", report_output))


def test_unexpected_exception_sanitized(isolated_db, tmp_path, capsys, monkeypatch):
    def _boom(**kwargs):
        raise RuntimeError("SENTINEL-unexpected-secret-detail")

    monkeypatch.setattr(f"{GATE_MODULE}.verify_final_output", _boom)
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    report_output = tmp_path / "report.json"

    rc = cli.cmd_verify_final_output(_args(project_id, project_dir / "manifest.json", report_output))
    out, err = capsys.readouterr()

    assert rc != 0
    assert out == ""
    assert "an unexpected internal error occurred" in err
    assert "SENTINEL-unexpected-secret-detail" not in err
    assert not report_output.exists()


# ---------------------------------------------------------------------
# 81-82: no row mutation
# ---------------------------------------------------------------------


def _snapshot_rows(project_id):
    from src.database.artifact_repository import list_artifacts_by_project
    from src.database.db import get_connection
    from src.database.project_repository import get_project

    conn = get_connection()
    try:
        project = get_project(conn, project_id)
        artifacts = list_artifacts_by_project(conn, project_id)
    finally:
        conn.close()
    return project, artifacts


def test_no_rows_change_on_passed_report(isolated_db, tmp_path):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_valid_render(project_dir, project_id)
    before_project, before_artifacts = _snapshot_rows(project_id)

    rc = cli.cmd_verify_final_output(_args(project_id, project_dir / "manifest.json", tmp_path / "report.json"))

    after_project, after_artifacts = _snapshot_rows(project_id)
    assert rc == 0
    assert after_project == before_project
    assert after_artifacts == before_artifacts


def test_no_rows_change_on_failed_report(isolated_db, tmp_path):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    before_project, before_artifacts = _snapshot_rows(project_id)

    rc = cli.cmd_verify_final_output(_args(project_id, project_dir / "manifest.json", tmp_path / "report.json"))

    after_project, after_artifacts = _snapshot_rows(project_id)
    assert rc != 0
    assert after_project == before_project
    assert after_artifacts == before_artifacts
