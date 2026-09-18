"""Tests for the `run-final-qc-workflow` CLI command (src/cli.py
cmd_run_final_qc_workflow). Same isolated_db / project-fixture pattern as
tests/test_cli_verify_final_output.py. Real SQLite, the REAL
register_qc_report_artifact()/verify_and_advance(); only
verify_final_output() is faked (a canned FinalQcReport), so no ffmpeg/
ffprobe/provider/network call happens anywhere in this file."""
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
from src.core.final_qc_gate import FinalQcProbeError, FinalQcReport, QcCheckResult
from src.core.manifest_builder import build_video_manifest
from src.core.manifest_store import save_manifest
from src.core.project_state_machine import create_initial_project, initial_transition_for
from src.core.qc_report_artifact_registrar import QcReportArtifactRegistrationResult
from src.database.artifact_repository import register_artifact
from src.database.db import get_connection, init_db
from src.database.project_repository import create_project
from src.models.artifact import ArtifactRecord
from src.utils.channel_config import get_channel_policy

FIXED_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
WORKFLOW = "src.core.qc_workflow"
REASON = "QC reviewed and passed"


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


def _scene_plan_dict(scene_ids=("scene-01",)) -> dict:
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


def _create_project(projects_root: Path, *, stage: str = "qc_pending"):
    manifest = build_video_manifest(
        _story_input_dict(), _scene_plan_dict(), get_channel_policy(), created_at=FIXED_NOW
    )
    manifest_path = projects_root / manifest.project_id / "manifest.json"
    save_manifest(manifest, manifest_path)
    project = create_initial_project(manifest_path, manifest, now=FIXED_NOW)
    transition = initial_transition_for(project)
    if stage != "planned":
        project = project.model_copy(update={"current_stage": stage, "last_successful_stage": "rendered"})
    init_db()
    conn = get_connection()
    try:
        create_project(conn, project, transition)
    finally:
        conn.close()
    return project.project_id, manifest_path.parent, manifest


def _register_render(project_dir: Path, project_id: str) -> None:
    content = b"dummy-render-bytes"
    path = project_dir / "render" / "final.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    conn = get_connection()
    try:
        register_artifact(
            conn,
            ArtifactRecord(
                artifact_id="render-final", project_id=project_id, kind="render", scene_id=None,
                relative_path="render/final.mp4", byte_size=len(content),
                sha256_checksum=hashlib.sha256(content).hexdigest(), created_at=FIXED_NOW,
                metadata={"duration_seconds": 5.0, "source": "final-video-assembly-v1"},
            ),
        )
    finally:
        conn.close()


def _canned_report(project_id: str, *, passed: bool = True) -> FinalQcReport:
    return FinalQcReport(
        project_id=project_id,
        passed=passed,
        require_overlays=False,
        render_checks=(QcCheckResult("render_registered", passed, "render", "canned"),),
        overlay_render_checks=(),
        overlay_render_present=False,
        viewer_facing_output_kind="render" if passed else None,
        viewer_facing_output_relative_path="render/final.mp4" if passed else None,
        blocking_reasons=() if passed else ("render is not trustworthy",),
        warnings=("overlay_render is absent; render/final.mp4 is the base viewer-facing output",) if passed else (),
        generated_at="2026-01-01T12:00:00+00:00",
    )


def _patch_verify(monkeypatch, *, passed: bool = True):
    monkeypatch.setattr(
        f"{WORKFLOW}.verify_final_output", lambda **kw: _canned_report(kw["project"].project_id, passed=passed)
    )


def _args(project_id, manifest, report_output, *, reason=REASON, require_overlays="false", out_format="text"):
    return argparse.Namespace(
        project_id=project_id,
        manifest=str(manifest),
        report_output=str(report_output),
        reason=reason,
        require_overlays=require_overlays,
        format=out_format,
    )


def _snapshot():
    conn = get_connection()
    try:
        return {
            table: [tuple(r) for r in conn.execute(f"SELECT * FROM {table}")]
            for table in ("projects", "artifacts", "project_transitions", "jobs")
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------
# parser registration
# ---------------------------------------------------------------------


def _full_argv(**omit):
    values = {
        "--manifest": "m.json", "--report-output": "r.json", "--reason": "ok",
    }
    for key in omit:
        values.pop("--" + key.replace("_", "-"))
    argv = ["run-final-qc-workflow", "proj-123"]
    for key, value in values.items():
        argv += [key, value]
    return argv


def test_parser_registers_run_final_qc_workflow():
    args = cli.build_parser().parse_args(_full_argv())
    assert args.func is cli.cmd_run_final_qc_workflow
    assert args.project_id == "proj-123"
    assert args.require_overlays == "false"
    assert args.format == "text"


def test_project_id_is_positional():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            ["run-final-qc-workflow", "--manifest", "m", "--report-output", "r", "--reason", "ok"]
        )


@pytest.mark.parametrize("missing", ["manifest", "report_output", "reason"])
def test_required_options(missing):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(_full_argv(**{missing: True}))


def test_require_overlays_parsing():
    parser = cli.build_parser()
    assert parser.parse_args(_full_argv() + ["--require-overlays", "true"]).require_overlays == "true"
    with pytest.raises(SystemExit):
        parser.parse_args(_full_argv() + ["--require-overlays", "maybe"])


def test_existing_commands_remain_registered():
    parser = cli.build_parser()
    assert parser.parse_args(["health"]).func is cli.cmd_health
    assert parser.parse_args(
        ["verify-final-output", "p", "--manifest", "m", "--report-output", "r"]
    ).func is cli.cmd_verify_final_output
    assert parser.parse_args(["register-qc-report-artifact", "p", "--file", "f"]).func is cli.cmd_register_qc_report_artifact
    assert parser.parse_args(
        ["verify-and-advance", "p", "--to", "qc_passed", "--reason", "r"]
    ).func is cli.cmd_verify_and_advance


# ---------------------------------------------------------------------
# success / partial output
# ---------------------------------------------------------------------


def test_full_success_text_stdout_exit_zero(isolated_db, monkeypatch, tmp_path, capsys):
    project_id, project_dir, manifest = _create_project(tmp_path / "projects")
    _register_render(project_dir, project_id)
    _patch_verify(monkeypatch)
    report_output = tmp_path / "out" / "report.json"

    rc = cli.cmd_run_final_qc_workflow(_args(project_id, project_dir / "manifest.json", report_output))
    out, err = capsys.readouterr()

    assert rc == 0
    assert err == ""
    assert "run-final-qc-workflow: OK" in out
    for field in (
        "qc_report_generated", "qc_report_passed", "qc_report_path", "qc_report_artifact_registered",
        "qc_report_artifact_id", "lifecycle_advanced", "resulting_project_stage", "stopped_at_step",
        "blocked_reasons", "warnings", "verification_report_summary",
    ):
        assert f"{field}:" in out
    assert report_output.exists()


def test_full_success_json_stdout_exit_zero(isolated_db, monkeypatch, tmp_path, capsys):
    project_id, project_dir, manifest = _create_project(tmp_path / "projects")
    _register_render(project_dir, project_id)
    _patch_verify(monkeypatch)

    rc = cli.cmd_run_final_qc_workflow(
        _args(project_id, project_dir / "manifest.json", tmp_path / "out" / "report.json", out_format="json")
    )
    out, err = capsys.readouterr()
    payload = json.loads(out)

    assert rc == 0 and err == ""
    assert payload["ok"] is True
    assert payload["lifecycle_advanced"] is True
    assert payload["resulting_project_stage"] == "qc_passed"
    assert payload["stopped_at_step"] is None
    assert payload["verification_report_summary"]["passed"] is True


@pytest.mark.parametrize("out_format", ["text", "json"])
def test_failed_qc_partial_result_to_stdout_nonzero_report_exists(isolated_db, monkeypatch, tmp_path, capsys, out_format):
    project_id, project_dir, manifest = _create_project(tmp_path / "projects")
    _patch_verify(monkeypatch, passed=False)
    report_output = tmp_path / "out" / "report.json"

    rc = cli.cmd_run_final_qc_workflow(
        _args(project_id, project_dir / "manifest.json", report_output, out_format=out_format)
    )
    out, err = capsys.readouterr()

    assert rc != 0
    assert err == ""
    assert report_output.exists()
    assert json.loads(report_output.read_text(encoding="utf-8"))["passed"] is False
    if out_format == "json":
        payload = json.loads(out)
        assert payload["ok"] is False
        assert payload["stopped_at_step"] == "verify_final_output"
        assert payload["qc_report_artifact_registered"] is False
    else:
        assert "run-final-qc-workflow: PARTIAL" in out
        assert "verify_final_output" in out


@pytest.mark.parametrize("out_format", ["text", "json"])
def test_registration_partial_result_to_stdout_nonzero(isolated_db, monkeypatch, tmp_path, capsys, out_format):
    project_id, project_dir, manifest = _create_project(tmp_path / "projects")
    _register_render(project_dir, project_id)
    _patch_verify(monkeypatch)
    monkeypatch.setattr(
        f"{WORKFLOW}.register_qc_report_artifact",
        lambda *a, **k: QcReportArtifactRegistrationResult(
            project_id=project_id, artifact_id="qc-report-final", relative_path="qc/report.json", ok=False,
            idempotent=False, copied=False, passed=True, artifact=None, reasons=("rejected by registrar",),
        ),
    )
    report_output = tmp_path / "out" / "report.json"

    rc = cli.cmd_run_final_qc_workflow(
        _args(project_id, project_dir / "manifest.json", report_output, out_format=out_format)
    )
    out, err = capsys.readouterr()

    assert rc != 0 and err == ""
    assert report_output.exists()
    assert "register_qc_report_artifact" in out
    assert "rejected by registrar" in out


@pytest.mark.parametrize("out_format", ["text", "json"])
def test_transition_partial_result_to_stdout_nonzero(isolated_db, monkeypatch, tmp_path, capsys, out_format):
    project_id, project_dir, manifest = _create_project(tmp_path / "projects", stage="rendered")
    _register_render(project_dir, project_id)
    _patch_verify(monkeypatch)
    report_output = tmp_path / "out" / "report.json"

    rc = cli.cmd_run_final_qc_workflow(
        _args(project_id, project_dir / "manifest.json", report_output, out_format=out_format)
    )
    out, err = capsys.readouterr()

    assert rc != 0 and err == ""
    assert report_output.exists()
    assert "verify_and_advance" in out
    if out_format == "json":
        payload = json.loads(out)
        assert payload["qc_report_artifact_registered"] is True
        assert payload["lifecycle_advanced"] is False


def test_exit_zero_requires_every_success_condition(monkeypatch, capsys):
    from src.core.qc_workflow import QcWorkflowResult

    def _result(**overrides):
        base = dict(
            project_id="p", qc_report_generated=True, qc_report_passed=True, qc_report_path="x",
            qc_report_artifact_registered=True, qc_report_artifact_id="a", lifecycle_advanced=True,
            resulting_project_stage="qc_passed", stopped_at_step=None, blocked_reasons=(), warnings=(),
            verification_report_summary=None,
        )
        base.update(overrides)
        return QcWorkflowResult(**base)

    args = _args("p", "m.json", "r.json", out_format="json")
    for overrides, expected_rc in (
        ({}, 0),
        ({"qc_report_passed": False}, 1),
        ({"qc_report_artifact_registered": False}, 1),
        ({"lifecycle_advanced": False}, 1),
        ({"resulting_project_stage": "qc_pending"}, 1),
    ):
        monkeypatch.setattr(f"{WORKFLOW}.run_qc_workflow", lambda **kw: _result(**overrides))
        assert cli.cmd_run_final_qc_workflow(args) == expected_rc
        capsys.readouterr()


def test_existing_report_resume_via_cli(isolated_db, monkeypatch, tmp_path, capsys):
    project_id, project_dir, manifest = _create_project(tmp_path / "projects")
    _register_render(project_dir, project_id)
    content = b'{"passed": true}'
    qc_path = project_dir / "qc" / "report.json"
    qc_path.parent.mkdir(parents=True)
    qc_path.write_bytes(content)
    conn = get_connection()
    try:
        register_artifact(
            conn,
            ArtifactRecord(
                artifact_id="qc-report-final", project_id=project_id, kind="qc_report", scene_id=None,
                relative_path="qc/report.json", byte_size=len(content),
                sha256_checksum=hashlib.sha256(content).hexdigest(), created_at=FIXED_NOW,
                metadata={"passed": True, "source": "external"},
            ),
        )
    finally:
        conn.close()
    monkeypatch.setattr(f"{WORKFLOW}.verify_final_output", lambda **k: (_ for _ in ()).throw(AssertionError("no")))
    report_output = tmp_path / "out" / "report.json"

    rc = cli.cmd_run_final_qc_workflow(
        _args(project_id, project_dir / "manifest.json", report_output, out_format="json")
    )
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["qc_report_generated"] is False
    assert payload["qc_report_path"] == str(qc_path.resolve())
    assert not report_output.exists()


# ---------------------------------------------------------------------
# preflight / sanitized errors
# ---------------------------------------------------------------------


def test_preflight_error_text_to_stderr_no_report(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_project(tmp_path / "projects")
    report_output = tmp_path / "out" / "report.json"

    rc = cli.cmd_run_final_qc_workflow(
        _args(project_id, project_dir / "manifest.json", report_output, reason="   ")
    )
    out, err = capsys.readouterr()

    assert rc != 0 and out == ""
    assert "run-final-qc-workflow: FAILED" in err
    assert not report_output.exists()


def test_preflight_error_json_to_stderr_no_report(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_project(tmp_path / "projects")
    report_output = tmp_path / "out" / "report.json"
    report_output.parent.mkdir(parents=True)
    report_output.write_text("keep-me", encoding="utf-8")

    rc = cli.cmd_run_final_qc_workflow(
        _args(project_id, project_dir / "manifest.json", report_output, out_format="json")
    )
    out, err = capsys.readouterr()

    assert rc != 0 and out == ""
    payload = json.loads(err)
    assert payload["ok"] is False and payload["project_id"] == project_id
    assert report_output.read_text(encoding="utf-8") == "keep-me"


def test_probe_error_reported_to_stderr_no_report(isolated_db, monkeypatch, tmp_path, capsys):
    project_id, project_dir, manifest = _create_project(tmp_path / "projects")
    monkeypatch.setattr(
        f"{WORKFLOW}.verify_final_output", lambda **k: (_ for _ in ()).throw(FinalQcProbeError("ffprobe unusable"))
    )
    report_output = tmp_path / "out" / "report.json"

    rc = cli.cmd_run_final_qc_workflow(_args(project_id, project_dir / "manifest.json", report_output))
    out, err = capsys.readouterr()

    assert rc != 0 and out == ""
    assert "ffprobe unusable" in err
    assert not report_output.exists()


_SENTINEL_SQLITE_ERROR = "SENTINEL-SQLITE-ERROR-C:\\sensitive\\project.db"


def test_sqlite_sentinel_sanitized_text(tmp_path, monkeypatch, capsys):
    def _boom():
        raise sqlite3.OperationalError(_SENTINEL_SQLITE_ERROR)

    monkeypatch.setattr(f"{WORKFLOW}.get_readonly_connection", _boom)
    rc = cli.cmd_run_final_qc_workflow(_args("proj-x", tmp_path / "m.json", tmp_path / "r.json"))
    out, err = capsys.readouterr()

    assert rc != 0 and out == ""
    assert "no local project database found" in err
    assert _SENTINEL_SQLITE_ERROR not in err


def test_sqlite_sentinel_sanitized_json(tmp_path, monkeypatch, capsys):
    def _boom():
        raise sqlite3.OperationalError(_SENTINEL_SQLITE_ERROR)

    monkeypatch.setattr(f"{WORKFLOW}.get_readonly_connection", _boom)
    rc = cli.cmd_run_final_qc_workflow(
        _args("proj-x", tmp_path / "m.json", tmp_path / "r.json", out_format="json")
    )
    out, err = capsys.readouterr()

    assert rc != 0 and out == ""
    payload = json.loads(err)
    assert "no local project database found" in payload["reason"]
    for value in payload.values():
        assert _SENTINEL_SQLITE_ERROR not in str(value)


def test_invalid_format_rejected(tmp_path, capsys):
    rc = cli.cmd_run_final_qc_workflow(_args("p", "m", "r", out_format="xml"))
    out, err = capsys.readouterr()
    assert rc != 0 and out == ""
    assert "invalid --format" in err


def test_unexpected_exception_sanitized(monkeypatch, tmp_path, capsys):
    def _boom(**kwargs):
        raise RuntimeError("SENTINEL-unexpected-secret-detail")

    monkeypatch.setattr(f"{WORKFLOW}.run_qc_workflow", _boom)
    rc = cli.cmd_run_final_qc_workflow(_args("p", "m", "r"))
    out, err = capsys.readouterr()

    assert rc != 0 and out == ""
    assert "an unexpected internal error occurred" in err
    assert "SENTINEL-unexpected-secret-detail" not in err


def test_keyboard_interrupt_not_caught(monkeypatch):
    def _boom(**kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(f"{WORKFLOW}.run_qc_workflow", _boom)
    with pytest.raises(KeyboardInterrupt):
        cli.cmd_run_final_qc_workflow(_args("p", "m", "r"))


# ---------------------------------------------------------------------
# source-level / row-change safety
# ---------------------------------------------------------------------


def test_cli_opens_no_db_connection_itself():
    source = inspect.getsource(cli.cmd_run_final_qc_workflow)
    for forbidden in ("get_connection(", "get_readonly_connection(", "get_existing_connection("):
        assert forbidden not in source


def test_cli_imports_no_provider_module():
    source = inspect.getsource(cli.cmd_run_final_qc_workflow)
    for forbidden in ("llm_groq", "llm_tokenrouter", "tts_kokoro", "image_qwen", "lipsync_rhubarb", "subprocess"):
        assert forbidden not in source


def test_no_rows_change_on_preflight_failure(isolated_db, tmp_path):
    project_id, project_dir, manifest = _create_project(tmp_path / "projects")
    _register_render(project_dir, project_id)
    report_output = tmp_path / "report.json"
    report_output.write_text("existing", encoding="utf-8")
    before = _snapshot()

    rc = cli.cmd_run_final_qc_workflow(_args(project_id, project_dir / "manifest.json", report_output))

    assert rc != 0
    assert _snapshot() == before
    assert report_output.read_text(encoding="utf-8") == "existing"
