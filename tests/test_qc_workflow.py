"""Tests for src/core/qc_workflow.py — QC WORKFLOW WRAPPER V1. Real SQLite
(isolated per test via the isolated_db fixture, same convention as
tests/test_local_resume_orchestrator.py) and the REAL register_qc_report_
artifact()/verify_and_advance() throughout — only verify_final_output()'s
own ffprobe probe functions (or, for a few tests, verify_final_output()
itself) are faked, so no ffmpeg/ffprobe/provider/network call happens
anywhere in this file. The already-merged Final QC Gate's own real-FFmpeg
test (tests/test_final_qc_gate.py) is deliberately not duplicated here."""
from __future__ import annotations

import hashlib
import inspect
import json
import socket
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.core.final_qc_gate import FinalQcReport, QcCheckResult
from src.core.manifest_builder import build_video_manifest
from src.core.manifest_store import save_manifest
from src.core.project_state_machine import create_initial_project, initial_transition_for
from src.core.qc_report_artifact_registrar import QcReportArtifactRegistrationResult, _validate_qc_report
from src.core.qc_workflow import (
    QcWorkflowArtifactTopologyError,
    QcWorkflowError,
    QcWorkflowProjectManifestMismatchError,
    QcWorkflowReportOutputConflictError,
    QcWorkflowResult,
    run_qc_workflow,
)
from src.database.artifact_repository import list_artifacts_by_project, register_artifact
from src.database.db import get_connection, get_existing_connection, get_readonly_connection, init_db
from src.database.project_repository import create_project, get_project
from src.models.artifact import ArtifactRecord
from src.utils.channel_config import get_channel_policy

FIXED_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
MODULE = "src.core.qc_workflow"
GATE = "src.core.final_qc_gate"
REASON = "QC reviewed and passed"


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    from src.utils import config

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    config.get_settings.cache_clear()
    yield tmp_path
    config.get_settings.cache_clear()


@pytest.fixture()
def default_probe(monkeypatch):
    monkeypatch.setattr(f"{GATE}._probe_stream_types", lambda path: frozenset({"video", "audio"}))
    monkeypatch.setattr(f"{GATE}._probe_duration_seconds", lambda path: 5.0)


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


def _overlay_dict(**overrides) -> dict:
    data = dict(
        text="The Cyberball Game",
        position="lower_third",
        style_id="lower_third_primary",
        viewer_facing_language="English",
        deterministic=True,
        start_seconds=0.0,
        end_seconds=1.0,
    )
    data.update(overrides)
    return data


def _scene_plan_dict(scene_ids=("scene-01",), overlays_by_scene=None) -> dict:
    overlays_by_scene = overlays_by_scene or {}
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
                text_overlays=tuple(overlays_by_scene.get(scene_id, ())),
            )
            for i, scene_id in enumerate(scene_ids, start=1)
        ),
        role_outfits=(),
    )


def _create_project(
    projects_root: Path,
    *,
    stage: str = "qc_pending",
    scene_ids=("scene-01",),
    overlays_by_scene=None,
    story_id: str = "why-we-care-what-people-think",
):
    manifest = build_video_manifest(
        _story_input_dict(story_id), _scene_plan_dict(scene_ids, overlays_by_scene), get_channel_policy(),
        created_at=FIXED_NOW,
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


def _write_file(path: Path, content: bytes) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return len(content), hashlib.sha256(content).hexdigest()


def _register(record: ArtifactRecord) -> None:
    conn = get_connection()
    try:
        register_artifact(conn, record)
    finally:
        conn.close()


def _register_render(project_dir: Path, project_id: str, duration=5.0) -> ArtifactRecord:
    size, checksum = _write_file(project_dir / "render" / "final.mp4", b"dummy-render-bytes")
    record = ArtifactRecord(
        artifact_id="render-final",
        project_id=project_id,
        kind="render",
        scene_id=None,
        relative_path="render/final.mp4",
        byte_size=size,
        sha256_checksum=checksum,
        created_at=FIXED_NOW,
        metadata={"duration_seconds": duration, "source": "final-video-assembly-v1"},
    )
    _register(record)
    return record


def _register_overlay(project_dir: Path, project_id: str, render: ArtifactRecord, manifest, overlay_count: int):
    size, checksum = _write_file(project_dir / "overlay_render" / "final.mp4", b"dummy-overlay-bytes")
    record = ArtifactRecord(
        artifact_id="overlay-render-final",
        project_id=project_id,
        kind="overlay_render",
        scene_id=None,
        relative_path="overlay_render/final.mp4",
        byte_size=size,
        sha256_checksum=checksum,
        created_at=FIXED_NOW,
        metadata={
            "duration_seconds": 5.0,
            "source": "text-overlay-renderer-v1",
            "overlay_count": overlay_count,
            "source_render_artifact_id": render.artifact_id,
            "source_render_sha256": render.sha256_checksum,
            "source_render_relative_path": "render/final.mp4",
            "manifest_fingerprint": manifest.source_fingerprint,
        },
    )
    _register(record)
    return record


def _register_existing_qc_report(project_dir: Path, project_id: str, content: bytes, *, artifact_id="qc-report-final",
                                 relative_path="qc/report.json", write_file=True) -> ArtifactRecord:
    if write_file:
        size, checksum = _write_file(project_dir / relative_path, content)
    else:
        size, checksum = len(content), hashlib.sha256(content).hexdigest()
    record = ArtifactRecord(
        artifact_id=artifact_id,
        project_id=project_id,
        kind="qc_report",
        scene_id=None,
        relative_path=relative_path,
        byte_size=size,
        sha256_checksum=checksum,
        created_at=FIXED_NOW,
        metadata={"passed": True, "source": "external"},
    )
    _register(record)
    return record


def _canned_report(project_id: str, *, passed: bool = True, blocking=(), warnings=(), overlay=False) -> FinalQcReport:
    return FinalQcReport(
        project_id=project_id,
        passed=passed,
        require_overlays=False,
        render_checks=(QcCheckResult("render_registered", passed, "render", "canned"),),
        overlay_render_checks=(),
        overlay_render_present=overlay,
        viewer_facing_output_kind=("overlay_render" if overlay else "render") if passed else None,
        viewer_facing_output_relative_path=(
            ("overlay_render/final.mp4" if overlay else "render/final.mp4") if passed else None
        ),
        blocking_reasons=tuple(blocking),
        warnings=tuple(warnings),
        generated_at="2026-01-01T12:00:00+00:00",
    )


def _patch_verify(monkeypatch, report_factory, calls: list | None = None):
    def _fake(**kwargs):
        if calls is not None:
            calls.append(kwargs)
        return report_factory(kwargs["project"].project_id)

    monkeypatch.setattr(f"{MODULE}.verify_final_output", _fake)


def _run(project_id, project_dir, report_output, *, require_overlays=False, reason=REASON, manifest_path=None):
    return run_qc_workflow(
        project_id=project_id,
        manifest_path=manifest_path or (project_dir / "manifest.json"),
        report_output_path=report_output,
        require_overlays=require_overlays,
        reason=reason,
    )


def _snapshot():
    conn = get_connection()
    try:
        rows = {
            "projects": [tuple(r) for r in conn.execute("SELECT * FROM projects ORDER BY project_id")],
            "artifacts": [tuple(r) for r in conn.execute("SELECT * FROM artifacts ORDER BY artifact_id")],
            "transitions": [tuple(r) for r in conn.execute("SELECT * FROM project_transitions ORDER BY transition_id")],
            "jobs": [tuple(r) for r in conn.execute("SELECT * FROM jobs")],
        }
    finally:
        conn.close()
    return rows


def _stage(project_id):
    conn = get_readonly_connection()
    try:
        return get_project(conn, project_id).current_stage
    finally:
        conn.close()


def _artifacts(project_id, kind=None):
    conn = get_readonly_connection()
    try:
        return list_artifacts_by_project(conn, project_id, kind=kind)
    finally:
        conn.close()


# ---------------------------------------------------------------------
# 1-2, 17: full success paths
# ---------------------------------------------------------------------


def test_full_success_overlay_enhanced_output(isolated_db, default_probe, tmp_path):
    project_id, project_dir, manifest = _create_project(
        tmp_path / "projects", overlays_by_scene={"scene-01": [_overlay_dict()]}
    )
    render = _register_render(project_dir, project_id)
    _register_overlay(project_dir, project_id, render, manifest, overlay_count=1)
    report_output = tmp_path / "out" / "report.json"

    result = _run(project_id, project_dir, report_output, require_overlays=True)

    assert result.stopped_at_step is None
    assert result.lifecycle_advanced is True
    assert result.resulting_project_stage == "qc_passed"
    assert result.verification_report_summary["viewer_facing_output_kind"] == "overlay_render"
    assert _stage(project_id) == "qc_passed"
    assert report_output.exists()


def test_full_success_render_only_require_overlays_false(isolated_db, default_probe, tmp_path):
    project_id, project_dir, manifest = _create_project(tmp_path / "projects")
    _register_render(project_dir, project_id)

    result = _run(project_id, project_dir, tmp_path / "out" / "report.json", require_overlays=False)

    assert result.lifecycle_advanced is True
    assert result.verification_report_summary["viewer_facing_output_kind"] == "render"
    assert any("overlay_render is absent" in w for w in result.warnings)


def test_full_success_result_fields_exact(isolated_db, monkeypatch, tmp_path):
    project_id, project_dir, manifest = _create_project(tmp_path / "projects")
    _register_render(project_dir, project_id)
    _patch_verify(monkeypatch, lambda pid: _canned_report(pid, warnings=("w1",)))
    report_output = tmp_path / "out" / "report.json"

    result = _run(project_id, project_dir, report_output)

    assert result.project_id == project_id
    assert result.qc_report_generated is True
    assert result.qc_report_passed is True
    assert result.qc_report_path == str(report_output.resolve())
    assert result.qc_report_artifact_registered is True
    assert result.qc_report_artifact_id == "qc-report-final"
    assert result.lifecycle_advanced is True
    assert result.resulting_project_stage == "qc_passed"
    assert result.stopped_at_step is None
    assert result.blocked_reasons == ()
    assert result.warnings == ("w1",)
    assert result.verification_report_summary["passed"] is True


def test_result_is_immutable(isolated_db, monkeypatch, tmp_path):
    project_id, project_dir, manifest = _create_project(tmp_path / "projects")
    _register_render(project_dir, project_id)
    _patch_verify(monkeypatch, lambda pid: _canned_report(pid))
    result = _run(project_id, project_dir, tmp_path / "out" / "report.json")
    assert isinstance(result, QcWorkflowResult)
    with pytest.raises(Exception):
        result.lifecycle_advanced = False  # type: ignore[misc]


# ---------------------------------------------------------------------
# 3-4, 19, 26: failed QC
# ---------------------------------------------------------------------


def test_render_only_require_overlays_true_writes_failed_report_no_registration(isolated_db, default_probe, tmp_path):
    project_id, project_dir, manifest = _create_project(tmp_path / "projects")
    _register_render(project_dir, project_id)
    report_output = tmp_path / "out" / "report.json"
    before = _snapshot()

    result = _run(project_id, project_dir, report_output, require_overlays=True)

    assert result.stopped_at_step == "verify_final_output"
    assert result.qc_report_generated is True
    assert result.qc_report_passed is False
    assert result.qc_report_artifact_registered is False
    assert result.lifecycle_advanced is False
    assert "overlay_render is required but missing" in result.blocked_reasons
    assert json.loads(report_output.read_text(encoding="utf-8"))["passed"] is False
    assert _snapshot() == before


def test_invalid_render_writes_failed_report_no_registration_no_transition(isolated_db, default_probe, tmp_path):
    project_id, project_dir, manifest = _create_project(tmp_path / "projects")
    record = _register_render(project_dir, project_id)
    (project_dir / "render" / "final.mp4").write_bytes(b"corrupted-after-registration")
    report_output = tmp_path / "out" / "report.json"
    before = _snapshot()

    result = _run(project_id, project_dir, report_output)

    assert result.qc_report_passed is False
    assert result.qc_report_artifact_registered is False
    assert result.lifecycle_advanced is False
    assert report_output.exists()
    assert _snapshot() == before
    assert record.artifact_id == "render-final"


def test_failed_report_blocking_reasons_and_warnings_preserved(isolated_db, monkeypatch, tmp_path):
    project_id, project_dir, manifest = _create_project(tmp_path / "projects")
    _patch_verify(
        monkeypatch,
        lambda pid: _canned_report(pid, passed=False, blocking=("b1", "b2"), warnings=("w1",)),
    )
    result = _run(project_id, project_dir, tmp_path / "out" / "report.json")
    assert result.blocked_reasons == ("b1", "b2")
    assert result.warnings == ("w1",)
    assert result.verification_report_summary["blocking_reasons"] == ["b1", "b2"]


# ---------------------------------------------------------------------
# 5-7: preflight
# ---------------------------------------------------------------------


def test_report_output_already_exists_is_preflight_error(isolated_db, monkeypatch, tmp_path):
    project_id, project_dir, manifest = _create_project(tmp_path / "projects")
    report_output = tmp_path / "out" / "report.json"
    report_output.parent.mkdir(parents=True)
    report_output.write_text("sentinel", encoding="utf-8")
    monkeypatch.setattr(f"{MODULE}.verify_final_output", lambda **k: (_ for _ in ()).throw(AssertionError("no")))
    before = _snapshot()

    with pytest.raises(QcWorkflowReportOutputConflictError):
        _run(project_id, project_dir, report_output)

    assert report_output.read_text(encoding="utf-8") == "sentinel"
    assert _snapshot() == before


@pytest.mark.parametrize("target", ["canonical_manifest", "supplied_manifest", "render", "overlay", "artifact"])
def test_report_output_aliases_are_preflight_errors(isolated_db, tmp_path, target):
    project_id, project_dir, manifest = _create_project(tmp_path / "projects")
    supplied = tmp_path / "supplied-manifest.json"
    save_manifest(manifest, supplied)
    _register(
        ArtifactRecord(
            artifact_id="audio-scene-01", project_id=project_id, kind="audio", scene_id="scene-01",
            relative_path="audio/scene-01.wav", byte_size=1, sha256_checksum="5" * 64, created_at=FIXED_NOW,
            metadata={"duration_seconds": 5.0, "source": "external"},
        )
    )
    targets = {
        "canonical_manifest": project_dir / "manifest.json",
        "supplied_manifest": supplied,
        "render": project_dir / "render" / "final.mp4",
        "overlay": project_dir / "overlay_render" / "final.mp4",
        "artifact": project_dir / "audio" / "scene-01.wav",
    }
    before_bytes = supplied.read_bytes()

    with pytest.raises(QcWorkflowReportOutputConflictError):
        _run(project_id, project_dir, targets[target], manifest_path=supplied)

    assert supplied.read_bytes() == before_bytes
    assert not (project_dir / "audio").exists()


@pytest.mark.parametrize("reason", ["", "   ", "\n\t"])
def test_empty_reason_rejected(isolated_db, tmp_path, reason):
    project_id, project_dir, manifest = _create_project(tmp_path / "projects")
    with pytest.raises(QcWorkflowError):
        _run(project_id, project_dir, tmp_path / "out" / "report.json", reason=reason)
    assert not (tmp_path / "out").exists()


def test_empty_project_id_rejected(tmp_path):
    with pytest.raises(QcWorkflowError):
        run_qc_workflow(
            project_id="", manifest_path=tmp_path / "m.json", report_output_path=tmp_path / "r.json",
            require_overlays=False, reason=REASON,
        )


def test_project_manifest_mismatch_rejected(isolated_db, tmp_path):
    project_id, project_dir, manifest = _create_project(tmp_path / "projects")
    other_id, other_dir, other_manifest = _create_project(
        tmp_path / "projects2", story_id="a-completely-different-story"
    )
    with pytest.raises(QcWorkflowProjectManifestMismatchError):
        _run(project_id, project_dir, tmp_path / "out" / "r.json", manifest_path=other_dir / "manifest.json")


# ---------------------------------------------------------------------
# 8-11: existing qc_report artifact
# ---------------------------------------------------------------------


def test_existing_passed_report_skips_generation_and_attempts_transition(isolated_db, monkeypatch, tmp_path):
    project_id, project_dir, manifest = _create_project(tmp_path / "projects")
    _register_render(project_dir, project_id)
    _register_existing_qc_report(project_dir, project_id, b'{"passed": true}')
    monkeypatch.setattr(f"{MODULE}.verify_final_output", lambda **k: (_ for _ in ()).throw(AssertionError("no")))
    monkeypatch.setattr(
        f"{MODULE}.register_qc_report_artifact", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no"))
    )
    report_output = tmp_path / "out" / "report.json"

    result = _run(project_id, project_dir, report_output)

    assert result.qc_report_generated is False
    assert result.qc_report_passed is True
    assert result.qc_report_artifact_registered is True
    assert result.qc_report_artifact_id == "qc-report-final"
    assert result.qc_report_path == str((project_dir / "qc" / "report.json").resolve())
    assert result.lifecycle_advanced is True
    assert result.resulting_project_stage == "qc_passed"
    assert not report_output.exists()
    assert not report_output.parent.exists()


def test_existing_passed_false_report_blocks_without_transition(isolated_db, tmp_path):
    project_id, project_dir, manifest = _create_project(tmp_path / "projects")
    _register_render(project_dir, project_id)
    _register_existing_qc_report(project_dir, project_id, b'{"passed": false}')

    result = _run(project_id, project_dir, tmp_path / "out" / "report.json")

    assert result.stopped_at_step == "qc_report_preflight"
    assert result.qc_report_passed is False
    assert result.lifecycle_advanced is False
    assert _stage(project_id) == "qc_pending"


@pytest.mark.parametrize(
    "content",
    [None, b"not json", b"[1, 2]", b'{"nope": true}', b'{"passed": "yes"}'],
    ids=["missing-file", "invalid-json", "non-object", "no-passed", "non-bool-passed"],
)
def test_existing_unusable_report_blocks(isolated_db, tmp_path, content):
    project_id, project_dir, manifest = _create_project(tmp_path / "projects")
    _register_render(project_dir, project_id)
    if content is None:
        _register_existing_qc_report(project_dir, project_id, b'{"passed": true}', write_file=False)
    else:
        _register_existing_qc_report(project_dir, project_id, content)

    result = _run(project_id, project_dir, tmp_path / "out" / "report.json")

    assert result.stopped_at_step == "qc_report_preflight"
    assert result.qc_report_passed is None
    assert result.lifecycle_advanced is False
    assert len(result.blocked_reasons) == 1
    assert _stage(project_id) == "qc_pending"


def test_existing_report_unsafe_path_helper_blocks():
    from src.core.qc_workflow import _peek_existing_report

    passed, error = _peek_existing_report(None)
    assert passed is None and "unsafe" in error


def test_more_than_one_qc_report_artifact_blocks(isolated_db, tmp_path):
    project_id, project_dir, manifest = _create_project(tmp_path / "projects")
    _register_render(project_dir, project_id)
    _register_existing_qc_report(project_dir, project_id, b'{"passed": true}')
    _register_existing_qc_report(
        project_dir, project_id, b'{"passed": true, "x": 1}', artifact_id="qc-report-second",
        relative_path="qc/report2.json",
    )

    result = _run(project_id, project_dir, tmp_path / "out" / "report.json")

    assert result.stopped_at_step == "qc_report_topology"
    assert result.lifecycle_advanced is False
    assert _stage(project_id) == "qc_pending"


# ---------------------------------------------------------------------
# 12-16, 22, 28: partial failures — no rollback, no retry
# ---------------------------------------------------------------------


def test_registration_rejection_keeps_report_and_skips_transition(isolated_db, monkeypatch, tmp_path):
    project_id, project_dir, manifest = _create_project(tmp_path / "projects")
    _register_render(project_dir, project_id)
    _patch_verify(monkeypatch, lambda pid: _canned_report(pid))
    # A different file already sits at the canonical destination -> the real
    # registrar rejects (never overwrites).
    _write_file(project_dir / "qc" / "report.json", b"pre-existing different content")
    report_output = tmp_path / "out" / "report.json"

    result = _run(project_id, project_dir, report_output)

    assert result.stopped_at_step == "register_qc_report_artifact"
    assert result.qc_report_generated is True
    assert result.qc_report_passed is True
    assert result.qc_report_artifact_registered is False
    assert result.lifecycle_advanced is False
    assert report_output.exists()
    assert (project_dir / "qc" / "report.json").read_bytes() == b"pre-existing different content"
    assert _artifacts(project_id, "qc_report") == []
    assert _stage(project_id) == "qc_pending"


def test_registration_unexpected_exception_returns_structured_partial_result(isolated_db, monkeypatch, tmp_path):
    project_id, project_dir, manifest = _create_project(tmp_path / "projects")
    _register_render(project_dir, project_id)
    _patch_verify(monkeypatch, lambda pid: _canned_report(pid))

    def _boom(*args, **kwargs):
        raise RuntimeError("SENTINEL-unexpected-registration-detail")

    monkeypatch.setattr(f"{MODULE}.register_qc_report_artifact", _boom)
    report_output = tmp_path / "out" / "report.json"

    result = _run(project_id, project_dir, report_output)

    assert result.stopped_at_step == "register_qc_report_artifact"
    assert result.qc_report_generated is True
    assert result.lifecycle_advanced is False
    assert "SENTINEL" not in " ".join(result.blocked_reasons)
    assert report_output.exists()
    assert _stage(project_id) == "qc_pending"


def test_transition_rejection_after_registration_keeps_everything(isolated_db, monkeypatch, tmp_path):
    project_id, project_dir, manifest = _create_project(tmp_path / "projects")
    # No render artifact registered -> verify_and_advance rejects for real.
    _patch_verify(monkeypatch, lambda pid: _canned_report(pid))
    report_output = tmp_path / "out" / "report.json"

    result = _run(project_id, project_dir, report_output)

    assert result.stopped_at_step == "verify_and_advance"
    assert result.qc_report_artifact_registered is True
    assert result.qc_report_artifact_id == "qc-report-final"
    assert result.lifecycle_advanced is False
    assert any("render" in reason for reason in result.blocked_reasons)
    assert report_output.exists()
    assert (project_dir / "qc" / "report.json").exists()
    assert len(_artifacts(project_id, "qc_report")) == 1
    assert _stage(project_id) == "qc_pending"


def test_lifecycle_version_conflict_returns_partial_result(isolated_db, monkeypatch, tmp_path):
    project_id, project_dir, manifest = _create_project(tmp_path / "projects")
    _register_render(project_dir, project_id)
    _patch_verify(monkeypatch, lambda pid: _canned_report(pid))

    from src.core import qc_workflow as module

    real_verify_and_advance = module.verify_and_advance

    def _conflicting(conn, project, manifest_arg, artifacts, to_stage, reason, **kwargs):
        other = get_connection()
        try:
            with other:
                other.execute(
                    "UPDATE projects SET lifecycle_version = lifecycle_version + 1 WHERE project_id = ?",
                    (project_id,),
                )
        finally:
            other.close()
        return real_verify_and_advance(conn, project, manifest_arg, artifacts, to_stage, reason, **kwargs)

    monkeypatch.setattr(f"{MODULE}.verify_and_advance", _conflicting)
    report_output = tmp_path / "out" / "report.json"

    result = _run(project_id, project_dir, report_output)

    assert result.stopped_at_step == "verify_and_advance"
    assert result.lifecycle_advanced is False
    assert result.qc_report_artifact_registered is True
    assert report_output.exists()
    assert len(_artifacts(project_id, "qc_report")) == 1
    assert _stage(project_id) == "qc_pending"


def test_wrong_current_stage_returns_partial_result(isolated_db, monkeypatch, tmp_path):
    project_id, project_dir, manifest = _create_project(tmp_path / "projects", stage="rendered")
    _register_render(project_dir, project_id)
    _patch_verify(monkeypatch, lambda pid: _canned_report(pid))
    report_output = tmp_path / "out" / "report.json"

    result = _run(project_id, project_dir, report_output)

    assert result.stopped_at_step == "verify_and_advance"
    assert result.lifecycle_advanced is False
    assert result.resulting_project_stage == "rendered"
    assert result.blocked_reasons
    assert report_output.exists()
    assert len(_artifacts(project_id, "qc_report")) == 1
    assert _stage(project_id) == "rendered"


def test_transition_unexpected_exception_returns_partial_result(isolated_db, monkeypatch, tmp_path):
    project_id, project_dir, manifest = _create_project(tmp_path / "projects")
    _register_render(project_dir, project_id)
    _patch_verify(monkeypatch, lambda pid: _canned_report(pid))

    def _boom(*args, **kwargs):
        raise RuntimeError("SENTINEL-transition-detail")

    monkeypatch.setattr(f"{MODULE}.verify_and_advance", _boom)
    result = _run(project_id, project_dir, tmp_path / "out" / "report.json")

    assert result.stopped_at_step == "verify_and_advance"
    assert result.qc_report_artifact_registered is True
    assert "SENTINEL" not in " ".join(result.blocked_reasons)


def test_no_automatic_retry_loop(isolated_db, monkeypatch, tmp_path):
    project_id, project_dir, manifest = _create_project(tmp_path / "projects")
    _register_render(project_dir, project_id)
    verify_calls: list = []
    register_calls: list = []
    transition_calls: list = []
    _patch_verify(monkeypatch, lambda pid: _canned_report(pid), verify_calls)

    def _rejecting_register(*args, **kwargs):
        register_calls.append(1)
        return QcReportArtifactRegistrationResult(
            project_id=project_id, artifact_id="qc-report-final", relative_path="qc/report.json", ok=False,
            idempotent=False, copied=False, passed=True, artifact=None, reasons=("rejected",),
        )

    monkeypatch.setattr(f"{MODULE}.register_qc_report_artifact", _rejecting_register)
    monkeypatch.setattr(
        f"{MODULE}.verify_and_advance", lambda *a, **k: transition_calls.append(1)
    )

    result = _run(project_id, project_dir, tmp_path / "out" / "report.json")

    assert result.stopped_at_step == "register_qc_report_artifact"
    assert len(verify_calls) == 1
    assert register_calls == [1]
    assert transition_calls == []


def test_atomic_write_failure_leaves_no_report_and_no_temp_file(isolated_db, monkeypatch, tmp_path):
    project_id, project_dir, manifest = _create_project(tmp_path / "projects")
    _register_render(project_dir, project_id)
    _patch_verify(monkeypatch, lambda pid: _canned_report(pid))

    def _boom(src, dst):
        raise OSError("disk full (simulated)")

    monkeypatch.setattr(f"{MODULE}.os.replace", _boom)
    report_output = tmp_path / "out" / "report.json"
    before = _snapshot()

    result = _run(project_id, project_dir, report_output)

    assert result.stopped_at_step == "report_write"
    assert result.qc_report_generated is False
    assert not report_output.exists()
    assert list(report_output.parent.glob(".*.tmp")) == []
    assert _snapshot() == before


def test_generated_report_payload_is_accepted_by_existing_registrar_validator(isolated_db, default_probe, tmp_path):
    project_id, project_dir, manifest = _create_project(tmp_path / "projects")
    _register_render(project_dir, project_id)
    report_output = tmp_path / "out" / "report.json"

    _run(project_id, project_dir, report_output, require_overlays=False)

    data = json.loads(report_output.read_text(encoding="utf-8"))
    assert isinstance(data["passed"], bool)
    assert _validate_qc_report(report_output) == (True, None)


# ---------------------------------------------------------------------
# 23: live DB-connection lifecycle
# ---------------------------------------------------------------------


def test_db_connections_closed_before_each_stage(isolated_db, monkeypatch, tmp_path):
    project_id, project_dir, manifest = _create_project(tmp_path / "projects")
    _register_render(project_dir, project_id)
    tracked: list = []

    def _tracking(factory):
        def _open():
            conn = factory()
            tracked.append(conn)
            return conn

        return _open

    monkeypatch.setattr(f"{MODULE}.get_readonly_connection", _tracking(get_readonly_connection))
    monkeypatch.setattr(f"{MODULE}.get_existing_connection", _tracking(get_existing_connection))

    def _all_closed(exclude=None):
        assert tracked
        for conn in tracked:
            if conn is exclude:
                continue
            with pytest.raises(sqlite3.ProgrammingError):
                conn.execute("SELECT 1")

    from src.core import qc_workflow as module

    def _fake_verify(**kwargs):
        _all_closed()
        return _canned_report(kwargs["project"].project_id)

    real_write = module._write_report_atomic

    def _checking_write(path, text):
        _all_closed()
        return real_write(path, text)

    real_register = module.register_qc_report_artifact

    def _checking_register(conn, *args, **kwargs):
        _all_closed(exclude=conn)
        return real_register(conn, *args, **kwargs)

    real_transition = module.verify_and_advance

    def _checking_transition(conn, *args, **kwargs):
        _all_closed(exclude=conn)
        return real_transition(conn, *args, **kwargs)

    monkeypatch.setattr(f"{MODULE}.verify_final_output", _fake_verify)
    monkeypatch.setattr(f"{MODULE}._write_report_atomic", _checking_write)
    monkeypatch.setattr(f"{MODULE}.register_qc_report_artifact", _checking_register)
    monkeypatch.setattr(f"{MODULE}.verify_and_advance", _checking_transition)

    result = _run(project_id, project_dir, tmp_path / "out" / "report.json")

    assert result.lifecycle_advanced is True
    assert len(tracked) == 3  # preflight read, registration write, transition write
    _all_closed()


def test_db_connections_closed_on_existing_report_resume_path(isolated_db, monkeypatch, tmp_path):
    project_id, project_dir, manifest = _create_project(tmp_path / "projects")
    _register_render(project_dir, project_id)
    _register_existing_qc_report(project_dir, project_id, b'{"passed": true}')
    tracked: list = []
    real_ro, real_rw = get_readonly_connection, get_existing_connection

    def _open(factory):
        def _f():
            conn = factory()
            tracked.append(conn)
            return conn

        return _f

    monkeypatch.setattr(f"{MODULE}.get_readonly_connection", _open(real_ro))
    monkeypatch.setattr(f"{MODULE}.get_existing_connection", _open(real_rw))

    result = _run(project_id, project_dir, tmp_path / "out" / "report.json")

    assert result.lifecycle_advanced is True
    assert len(tracked) == 2
    for conn in tracked:
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")


# ---------------------------------------------------------------------
# 24-25: source-level safety
# ---------------------------------------------------------------------


def _function_source() -> str:
    import src.core.qc_workflow as m

    return "\n".join(
        inspect.getsource(obj)
        for _name, obj in vars(m).items()
        if inspect.isfunction(obj) and obj.__module__ == m.__name__
    )


def test_wrapper_has_no_provider_ffmpeg_subprocess_network_or_verifier_calls():
    import src.core.qc_workflow as m

    source = _function_source()
    for forbidden in (
        "subprocess", "ffmpeg", "ffprobe", "artifact_verifier", "verify_artifact", "llm_groq",
        "llm_tokenrouter", "tts_kokoro", "image_qwen", "lipsync_rhubarb", "requests.", "urllib", "socket.",
        "build_final_local", "assemble_final_video", "derive_text_overlays", "render_text_overlays",
    ):
        assert forbidden not in source.lower() if forbidden.islower() else forbidden not in source
    assert not hasattr(m, "subprocess")
    assert not hasattr(m, "verify_artifact")


def test_wrapper_has_no_direct_lifecycle_transition_logic():
    source = _function_source()
    for forbidden in (
        "transition_project", "save_transition", "stage_advance_service", "advance_project_stage",
        "ready_for_manual_publish", '"completed"', "qc_pending", "current_stage ==", "lifecycle_version",
    ):
        assert forbidden not in source


def test_wrapper_has_no_destructive_rollback():
    source = _function_source()
    assert "rmtree" not in source
    assert ".unlink(" not in source.replace("os.unlink(", "")  # no Path.unlink anywhere
    assert source.count("os.unlink(") == 1  # only this call's own temp-file cleanup


# ---------------------------------------------------------------------
# 26-27: row-change discipline
# ---------------------------------------------------------------------


def test_only_expected_rows_change_on_full_success(isolated_db, monkeypatch, tmp_path):
    project_id, project_dir, manifest = _create_project(tmp_path / "projects")
    _register_render(project_dir, project_id)
    _patch_verify(monkeypatch, lambda pid: _canned_report(pid))
    before = _snapshot()

    _run(project_id, project_dir, tmp_path / "out" / "report.json")
    after = _snapshot()

    assert after["jobs"] == before["jobs"]
    new_artifacts = [row for row in after["artifacts"] if row not in before["artifacts"]]
    assert len(new_artifacts) == 1 and new_artifacts[0][2] == "qc_report"
    assert len(after["transitions"]) == len(before["transitions"]) + 1
    assert after["transitions"][-1][2:4] == ("qc_pending", "qc_passed")
    assert after["projects"] != before["projects"]
    assert _stage(project_id) == "qc_passed"


# ---------------------------------------------------------------------
# Wrapper-level real-SQLite integration
# ---------------------------------------------------------------------


def test_wrapper_level_sqlite_integration(isolated_db, monkeypatch, tmp_path):
    project_id, project_dir, manifest = _create_project(tmp_path / "projects", stage="qc_pending")
    _register_render(project_dir, project_id)
    _patch_verify(monkeypatch, lambda pid: _canned_report(pid, warnings=("note",)))
    report_output = tmp_path / "out" / "report.json"
    before = _snapshot()

    socket_calls: list = []
    real_init = socket.socket.__init__

    def _tracking_init(self, *args, **kwargs):
        socket_calls.append(1)
        return real_init(self, *args, **kwargs)

    socket.socket.__init__ = _tracking_init
    try:
        result = _run(project_id, project_dir, report_output, reason="QC gate passed; manual review OK")
    finally:
        socket.socket.__init__ = real_init

    assert socket_calls == []
    assert result.lifecycle_advanced is True and result.stopped_at_step is None

    assert report_output.exists()
    payload = json.loads(report_output.read_text(encoding="utf-8"))
    assert payload["passed"] is True

    qc_artifacts = _artifacts(project_id, "qc_report")
    assert len(qc_artifacts) == 1
    artifact = qc_artifacts[0]
    assert artifact.artifact_id == "qc-report-final"
    assert artifact.relative_path == "qc/report.json"
    assert artifact.scene_id is None
    assert dict(artifact.metadata) == {"passed": True, "source": "external"}
    assert artifact.sha256_checksum == hashlib.sha256((project_dir / "qc" / "report.json").read_bytes()).hexdigest()

    assert _stage(project_id) == "qc_passed"
    after = _snapshot()
    assert len(after["transitions"]) == len(before["transitions"]) + 1
    transition = after["transitions"][-1]
    assert transition[2:4] == ("qc_pending", "qc_passed")
    assert transition[5] == "QC gate passed; manual review OK"


def test_rerun_after_transition_failure_reuses_existing_registered_report(isolated_db, monkeypatch, tmp_path):
    """Resume-safety: first run registers the report but the transition is
    rejected (project at 'rendered'); after the operator corrects the stage,
    a rerun with a NEW report-output path must reuse the already-registered
    passed=true report — no regeneration, no new file, no re-registration."""
    project_id, project_dir, manifest = _create_project(tmp_path / "projects", stage="rendered")
    _register_render(project_dir, project_id)
    verify_calls: list = []
    _patch_verify(monkeypatch, lambda pid: _canned_report(pid), verify_calls)

    first = _run(project_id, project_dir, tmp_path / "out" / "first.json")
    assert first.stopped_at_step == "verify_and_advance"

    conn = get_connection()
    try:
        with conn:
            conn.execute("UPDATE projects SET current_stage = 'qc_pending' WHERE project_id = ?", (project_id,))
    finally:
        conn.close()

    second_output = tmp_path / "out" / "second.json"
    second = _run(project_id, project_dir, second_output)

    assert second.lifecycle_advanced is True
    assert second.qc_report_generated is False
    assert not second_output.exists()
    assert len(verify_calls) == 1
    assert len(_artifacts(project_id, "qc_report")) == 1
