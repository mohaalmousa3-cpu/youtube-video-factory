"""Tests for src/core/qc_report_artifact_registrar.py: Phase 2H's
registration-first, project-level QC Report Artifact Registration. Same
isolated_db / build-manifest-then-register-project pattern as
tests/test_render_artifact_registrar.py. No network, no provider, no
ffmpeg/Pillow — this registrar only needs stdlib json."""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.core.manifest_builder import build_video_manifest
from src.core.manifest_store import save_manifest
from src.core.project_state_machine import create_initial_project, initial_transition_for
from src.core.qc_report_artifact_registrar import (
    QcReportArtifactRegistrationError,
    register_qc_report_artifact,
)
from src.core.verified_transition_service import verify_and_advance
from src.database.artifact_repository import list_artifacts_by_project
from src.database.db import SCHEMA
from src.utils.channel_config import get_channel_policy

FIXED_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture()
def conn():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA)
    yield connection
    connection.close()


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


def _registered_project(conn, tmp_path: Path, scene_ids: tuple[str, ...] = ("scene-01",)):
    from src.database.project_repository import create_project

    manifest = build_video_manifest(
        _story_input_dict(), _scene_plan_dict(scene_ids), get_channel_policy(), created_at=FIXED_NOW
    )
    project_dir = tmp_path / "projects" / manifest.project_id
    manifest_path = project_dir / "manifest.json"
    save_manifest(manifest, manifest_path)

    project = create_initial_project(manifest_path, manifest, now=FIXED_NOW)
    transition = initial_transition_for(project)
    create_project(conn, project, transition)
    return project, manifest, project_dir


def _write_report(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------


def test_valid_report_registration_passed_true(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.json"
    _write_report(source, b'{"passed": true}')

    result = register_qc_report_artifact(conn, project, manifest, source, now=FIXED_NOW)

    assert result.ok is True
    assert result.idempotent is False
    assert result.copied is True
    assert result.artifact_id == "qc-report-final"
    assert result.relative_path == "qc/report.json"
    assert result.passed is True

    destination = project_dir / "qc" / "report.json"
    assert destination.exists()
    assert destination.read_bytes() == source.read_bytes()

    stored = list_artifacts_by_project(conn, project.project_id, kind="qc_report")
    assert len(stored) == 1
    assert stored[0].artifact_id == "qc-report-final"
    assert stored[0].kind == "qc_report"
    assert stored[0].scene_id is None
    assert stored[0].sha256_checksum == _sha256(destination)
    assert stored[0].byte_size == destination.stat().st_size
    assert stored[0].metadata["source"] == "external"
    assert stored[0].metadata["passed"] is True
    assert stored[0].created_at == FIXED_NOW


def test_valid_report_registration_passed_false_still_registers(conn, tmp_path):
    """A structurally valid report saying passed:false is real, retained
    audit data — registration itself never gates on the value; only
    verify_and_advance(to='qc_passed') does (see
    test_verified_transition_service.py)."""
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.json"
    _write_report(source, b'{"passed": false}')

    result = register_qc_report_artifact(conn, project, manifest, source, now=FIXED_NOW)

    assert result.ok is True
    assert result.passed is False

    stored = list_artifacts_by_project(conn, project.project_id, kind="qc_report")
    assert len(stored) == 1
    assert stored[0].metadata["passed"] is False


# ---------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------


def test_identical_repeat_is_a_no_write_idempotent_success(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.json"
    _write_report(source, b'{"passed": true}')

    first = register_qc_report_artifact(conn, project, manifest, source, now=FIXED_NOW)
    assert first.ok and not first.idempotent

    destination = project_dir / "qc" / "report.json"
    bytes_before = destination.read_bytes()
    mtime_before = destination.stat().st_mtime_ns
    rows_before = list_artifacts_by_project(conn, project.project_id, kind="qc_report")

    second = register_qc_report_artifact(conn, project, manifest, source, now=FIXED_NOW)

    assert second.ok is True
    assert second.idempotent is True
    assert second.copied is False
    assert second.passed is True
    assert second.reasons == ()

    rows_after = list_artifacts_by_project(conn, project.project_id, kind="qc_report")
    assert rows_after == rows_before  # no new/changed DB row
    assert destination.read_bytes() == bytes_before  # file untouched
    assert destination.stat().st_mtime_ns == mtime_before  # not rewritten


def test_different_second_source_is_rejected_and_leaves_original_unchanged(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source_a = tmp_path / "a.json"
    _write_report(source_a, b'{"passed": true}')
    first = register_qc_report_artifact(conn, project, manifest, source_a, now=FIXED_NOW)
    assert first.ok

    destination = project_dir / "qc" / "report.json"
    bytes_before = destination.read_bytes()
    row_before = list_artifacts_by_project(conn, project.project_id, kind="qc_report")[0]

    source_b = tmp_path / "b.json"
    _write_report(source_b, b'{"passed": false}')  # genuinely different content

    second = register_qc_report_artifact(conn, project, manifest, source_b, now=FIXED_NOW)

    assert second.ok is False
    assert second.reasons
    assert "already registered" in second.reasons[0]

    assert destination.read_bytes() == bytes_before
    row_after = list_artifacts_by_project(conn, project.project_id, kind="qc_report")[0]
    assert row_after == row_before


# ---------------------------------------------------------------------
# Zero-write rejection cases
# ---------------------------------------------------------------------


def test_mismatched_manifest_project_id_raises_and_causes_zero_writes(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.json"
    _write_report(source, b'{"passed": true}')

    mismatched_manifest = manifest.model_copy(update={"project_id": "proj-does-not-match"})

    with pytest.raises(QcReportArtifactRegistrationError):
        register_qc_report_artifact(conn, project, mismatched_manifest, source, now=FIXED_NOW)

    assert list_artifacts_by_project(conn, project.project_id, kind="qc_report") == []
    assert not (project_dir / "qc").exists()


def test_missing_source_file_causes_zero_writes(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)

    result = register_qc_report_artifact(conn, project, manifest, tmp_path / "does-not-exist.json", now=FIXED_NOW)

    assert result.ok is False
    assert "does not exist" in result.reasons[0]
    assert not (project_dir / "qc").exists()


def test_directory_source_causes_zero_writes(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    a_dir = tmp_path / "a_directory.json"
    a_dir.mkdir()

    result = register_qc_report_artifact(conn, project, manifest, a_dir, now=FIXED_NOW)

    assert result.ok is False
    assert "not a regular file" in result.reasons[0]
    assert not (project_dir / "qc").exists()


def test_zero_byte_source_causes_zero_writes(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    empty = tmp_path / "empty.json"
    empty.write_bytes(b"")

    result = register_qc_report_artifact(conn, project, manifest, empty, now=FIXED_NOW)

    assert result.ok is False
    assert "empty" in result.reasons[0]
    assert not (project_dir / "qc").exists()


# ---------------------------------------------------------------------
# Report structure validation: malformed, missing/non-boolean passed
# ---------------------------------------------------------------------


def test_malformed_json_source_causes_zero_writes(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    bad = tmp_path / "bad.json"
    _write_report(bad, b"this is not JSON at all, just junk bytes 1234567890")

    result = register_qc_report_artifact(conn, project, manifest, bad, now=FIXED_NOW)

    assert result.ok is False
    assert "is not valid JSON" in result.reasons[0]
    assert not (project_dir / "qc").exists()
    assert list_artifacts_by_project(conn, project.project_id, kind="qc_report") == []


def test_non_object_json_source_causes_zero_writes(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    bad = tmp_path / "bad.json"
    _write_report(bad, b"[1, 2, 3]")

    result = register_qc_report_artifact(conn, project, manifest, bad, now=FIXED_NOW)

    assert result.ok is False
    assert "must be a JSON object" in result.reasons[0]
    assert not (project_dir / "qc").exists()
    assert list_artifacts_by_project(conn, project.project_id, kind="qc_report") == []


def test_missing_passed_field_causes_zero_writes(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    bad = tmp_path / "bad.json"
    _write_report(bad, b'{"notes": "looks fine"}')

    result = register_qc_report_artifact(conn, project, manifest, bad, now=FIXED_NOW)

    assert result.ok is False
    assert "missing a 'passed' field" in result.reasons[0]
    assert not (project_dir / "qc").exists()
    assert list_artifacts_by_project(conn, project.project_id, kind="qc_report") == []


def test_non_boolean_passed_field_causes_zero_writes(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    bad = tmp_path / "bad.json"
    _write_report(bad, b'{"passed": "yes"}')  # a string, not a real boolean

    result = register_qc_report_artifact(conn, project, manifest, bad, now=FIXED_NOW)

    assert result.ok is False
    assert "must be a boolean" in result.reasons[0]
    assert not (project_dir / "qc").exists()
    assert list_artifacts_by_project(conn, project.project_id, kind="qc_report") == []


def test_numeric_passed_field_is_not_accepted_as_boolean(conn, tmp_path):
    """1/0 are ints in JSON, not real booleans — Python's bool is a subtype
    of int, so this specifically proves the check does not accidentally
    accept 1/0 as truthy/falsy stand-ins for true/false."""
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    bad = tmp_path / "bad.json"
    _write_report(bad, b'{"passed": 1}')

    result = register_qc_report_artifact(conn, project, manifest, bad, now=FIXED_NOW)

    assert result.ok is False
    assert "must be a boolean" in result.reasons[0]
    assert not (project_dir / "qc").exists()


# ---------------------------------------------------------------------
# Pre-existing destination file, no matching DB record
# ---------------------------------------------------------------------


def test_different_pre_existing_destination_is_never_overwritten(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    destination = project_dir / "qc" / "report.json"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"pre-existing content nobody asked to touch")
    bytes_before = destination.read_bytes()

    source = tmp_path / "source.json"
    _write_report(source, b'{"passed": true}')

    result = register_qc_report_artifact(conn, project, manifest, source, now=FIXED_NOW)

    assert result.ok is False
    assert "already exists with different content" in result.reasons[0]
    assert destination.read_bytes() == bytes_before
    assert list_artifacts_by_project(conn, project.project_id, kind="qc_report") == []


def test_identical_pre_existing_destination_is_registered_without_rewriting(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.json"
    _write_report(source, b'{"passed": true}')

    destination = project_dir / "qc" / "report.json"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(source.read_bytes())  # byte-identical, placed by some other step
    mtime_before = destination.stat().st_mtime_ns

    result = register_qc_report_artifact(conn, project, manifest, source, now=FIXED_NOW)

    assert result.ok is True
    assert result.copied is False  # never re-copied
    assert result.idempotent is False  # this IS a fresh DB registration, just without a copy
    assert destination.stat().st_mtime_ns == mtime_before

    stored = list_artifacts_by_project(conn, project.project_id, kind="qc_report")
    assert len(stored) == 1
    assert stored[0].sha256_checksum == _sha256(source)


# ---------------------------------------------------------------------
# Symlink / path-safety escape
# ---------------------------------------------------------------------


def test_symlinked_qc_directory_escape_is_rejected(conn, tmp_path):
    """If `qc/` itself were a symlink pointing outside the project
    directory, the shared path-safety check must catch it before any
    write — same defense-in-depth artifact_verifier.py already relies on,
    reused here via src.core.path_safety."""
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    outside = tmp_path / "outside_qc"
    outside.mkdir()
    try:
        (project_dir / "qc").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"cannot create symlinks in this environment: {exc}")

    source = tmp_path / "source.json"
    _write_report(source, b'{"passed": true}')

    result = register_qc_report_artifact(conn, project, manifest, source, now=FIXED_NOW)

    assert result.ok is False
    assert "unsafe" in result.reasons[0]
    assert list(outside.iterdir()) == []  # nothing was written through the symlink
    assert list_artifacts_by_project(conn, project.project_id, kind="qc_report") == []


# ---------------------------------------------------------------------
# Cleanup on register_artifact() failure
# ---------------------------------------------------------------------


def test_register_artifact_failure_cleans_up_only_a_freshly_copied_destination(conn, tmp_path, monkeypatch):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.json"
    _write_report(source, b'{"passed": true}')

    def _boom(_conn, _record):
        raise sqlite3.OperationalError("simulated database failure")

    monkeypatch.setattr("src.core.qc_report_artifact_registrar.register_artifact", _boom)

    result = register_qc_report_artifact(conn, project, manifest, source, now=FIXED_NOW)

    assert result.ok is False
    assert "simulated database failure" in result.reasons[0]
    destination = project_dir / "qc" / "report.json"
    assert not destination.exists()  # the freshly-copied file was cleaned up
    if destination.parent.exists():
        assert list(destination.parent.iterdir()) == []  # no orphan left behind either
    assert list_artifacts_by_project(conn, project.project_id, kind="qc_report") == []


def test_register_artifact_failure_never_removes_a_pre_existing_destination(conn, tmp_path, monkeypatch):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.json"
    _write_report(source, b'{"passed": true}')

    destination = project_dir / "qc" / "report.json"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(source.read_bytes())  # pre-existing, byte-identical

    def _boom(_conn, _record):
        raise sqlite3.OperationalError("simulated database failure")

    monkeypatch.setattr("src.core.qc_report_artifact_registrar.register_artifact", _boom)

    result = register_qc_report_artifact(conn, project, manifest, source, now=FIXED_NOW)

    assert result.ok is False
    assert destination.exists()  # never deleted — this call did not create it
    assert destination.read_bytes() == source.read_bytes()


def test_unexpected_register_artifact_failure_propagates_and_cleans_up_fresh_copy(conn, tmp_path, monkeypatch):
    """An UNEXPECTED exception (not one of register_artifact()'s
    documented failure types) must never be silently downgraded into an
    ordinary rejected result — it propagates unchanged. The freshly-copied
    destination is still cleaned up best-effort first."""
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.json"
    _write_report(source, b'{"passed": true}')

    def _boom(_conn, _record):
        raise RuntimeError("totally unexpected bug")

    monkeypatch.setattr("src.core.qc_report_artifact_registrar.register_artifact", _boom)

    with pytest.raises(RuntimeError, match="totally unexpected bug"):
        register_qc_report_artifact(conn, project, manifest, source, now=FIXED_NOW)

    destination = project_dir / "qc" / "report.json"
    assert not destination.exists()  # the freshly-copied file was still cleaned up
    assert list_artifacts_by_project(conn, project.project_id, kind="qc_report") == []


def test_unexpected_register_artifact_failure_never_removes_a_pre_existing_destination(conn, tmp_path, monkeypatch):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.json"
    _write_report(source, b'{"passed": true}')

    destination = project_dir / "qc" / "report.json"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(source.read_bytes())  # pre-existing, byte-identical
    bytes_before = destination.read_bytes()

    def _boom(_conn, _record):
        raise RuntimeError("totally unexpected bug")

    monkeypatch.setattr("src.core.qc_report_artifact_registrar.register_artifact", _boom)

    with pytest.raises(RuntimeError, match="totally unexpected bug"):
        register_qc_report_artifact(conn, project, manifest, source, now=FIXED_NOW)

    assert destination.exists()  # never deleted — this call did not create it
    assert destination.read_bytes() == bytes_before  # byte-for-byte unchanged


# ---------------------------------------------------------------------
# End-to-end with verify_and_advance (registration layer only — the
# semantic passed=true gate itself is tested exhaustively in
# tests/test_verified_transition_service.py)
# ---------------------------------------------------------------------


def test_end_to_end_register_true_report_with_render_then_verify_and_advance_succeeds(conn, tmp_path):
    from src.core.project_state_machine import transition_project
    from src.database.project_repository import save_transition, get_project
    from src.core.render_artifact_registrar import register_render_artifact
    import subprocess

    project, manifest, project_dir = _registered_project(conn, tmp_path, scene_ids=("scene-01", "scene-02"))
    stages = [
        ("audio_pending", False), ("audio_ready", True),
        ("visuals_pending", False), ("visuals_ready", True),
        ("animation_pending", False), ("animation_ready", True),
        ("render_pending", False), ("rendered", True),
        ("qc_pending", False),
    ]
    for stage, verified in stages:
        updated, transition = transition_project(project, stage, now=FIXED_NOW, verified=verified)
        save_transition(conn, project.lifecycle_version, updated, transition)
        project = get_project(conn, project.project_id)

    # A real render, via Phase 2G's own registrar (not re-implemented here).
    from src.utils.config import get_settings

    render_source = tmp_path / "final.mp4"
    ffmpeg = get_settings().ffmpeg_path
    subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i", "color=c=red:size=64x64:rate=5:duration=1",
         "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", "1",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(render_source)],
        capture_output=True, text=True, check=True,
    )
    render_result = register_render_artifact(conn, project, manifest, render_source, now=FIXED_NOW)
    assert render_result.ok, render_result.reasons

    qc_source = tmp_path / "report.json"
    _write_report(qc_source, b'{"passed": true}')
    qc_result = register_qc_report_artifact(conn, project, manifest, qc_source, now=FIXED_NOW)
    assert qc_result.ok, qc_result.reasons

    artifacts = list_artifacts_by_project(conn, project.project_id)
    outcome = verify_and_advance(conn, project, manifest, artifacts, "qc_passed", "smoke test")

    assert outcome.approved is True
    assert outcome.db_committed is True

    final = get_project(conn, project.project_id)
    assert final.current_stage == "qc_passed"
