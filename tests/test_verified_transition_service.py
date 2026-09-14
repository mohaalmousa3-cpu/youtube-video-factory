"""Tests for src/core/verified_transition_service.py — Phase 2C's sole
approved path for advancing a project into a verification-required
lifecycle stage. Same isolated_db / build-manifest-then-register-project
pattern as tests/test_cli_verify_artifacts.py."""
from __future__ import annotations

import builtins
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.core.manifest_builder import build_video_manifest
from src.core.manifest_store import save_manifest
from src.core.project_state_machine import create_initial_project, initial_transition_for, transition_project
from src.core.verified_transition_service import (
    SUPPORTED_TARGET_STAGES,
    VerifiedTransitionServiceError,
    verify_and_advance,
)
from src.database.artifact_repository import list_artifacts_by_project, register_artifact
from src.database.project_repository import (
    create_project,
    get_project,
    list_project_transitions,
    save_transition,
)
from src.models.artifact import ArtifactRecord
from src.utils.channel_config import get_channel_policy

FIXED_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


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


def _scene_plan_dict() -> dict:
    return dict(
        scenes=(
            dict(
                scene_id="scene-01",
                sequence=1,
                narration_text="Why does being left out sting so much?",
                scene_type="establishing",
                narrative_beat="hook",
                visual_brief="Stickman character alone on a quiet street corner at dusk.",
                motion_mode="in",
                approval_state="approved",
            ),
            dict(
                scene_id="scene-02",
                sequence=2,
                narration_text="Because belonging kept our ancestors alive.",
                scene_type="narration",
                narrative_beat="setup",
                visual_brief="Stickman character among a small warm group around a fire.",
                motion_mode="static",
                approval_state="approved",
            ),
        ),
        role_outfits=(),
    )


def _create_registered_project(projects_root: Path, *, story_id: str = "why-we-care-what-people-think"):
    """Mirrors test_cli_verify_artifacts.py's helper: builds and saves a
    manifest, creates the matching ProjectRecord in SQLite. Returns
    (project_id, project_dir, manifest)."""
    from src.database.db import get_connection, init_db

    manifest = build_video_manifest(
        _story_input_dict(story_id), _scene_plan_dict(), get_channel_policy(), created_at=FIXED_NOW
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


def _get_conn():
    from src.database.db import get_connection

    return get_connection()


def _write_and_register(conn, project_id, project_dir, kind, scene_id, rel, *, artifact_id=None, content=b"data"):
    path = project_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    record = ArtifactRecord(
        artifact_id=artifact_id or f"{kind}-{scene_id or 'project'}",
        project_id=project_id,
        kind=kind,
        scene_id=scene_id,
        relative_path=rel,
        byte_size=len(content),
        sha256_checksum=hashlib.sha256(content).hexdigest(),
        created_at=FIXED_NOW,
    )
    register_artifact(conn, record)
    return record


def _advance_plain(conn, project, to_stage, *, reason=None, verified=False, now=FIXED_NOW):
    """Advance through a non-verification-required '_pending' stage using
    project_state_machine/project_repository directly — outside this
    service's scope (a '_pending' stage claims nothing completed yet)."""
    updated, transition = transition_project(project, to_stage, now=now, reason=reason, verified=verified)
    save_transition(conn, project.lifecycle_version, updated, transition)
    return updated


# ---------------------------------------------------------------------
# success: audio_ready, visuals_ready, animation_ready, rendered,
# qc_passed, and completed's explicit-reason/verified guard preserved —
# one continuous journey, each checkpoint tested fail-then-succeed so the
# missing-artifact case for every stage is covered on the way through.
# ---------------------------------------------------------------------


def test_verify_and_advance_full_pipeline_fail_then_succeed_at_each_checkpoint(isolated_db):
    project_id, project_dir, manifest = _create_registered_project(isolated_db / "projects")
    conn = _get_conn()
    try:
        project = get_project(conn, project_id)
        assert project.current_stage == "planned"
        project = _advance_plain(conn, project, "audio_pending")

        # audio_ready: scene-02's audio missing -> rejected, nothing written
        _write_and_register(conn, project_id, project_dir, "audio", "scene-01", "audio/scene-01.wav")
        before = get_project(conn, project_id)
        before_transitions = list_project_transitions(conn, project_id)
        result = verify_and_advance(
            conn, project, manifest, list_artifacts_by_project(conn, project_id), "audio_ready", "narration recorded"
        )
        assert result.approved is False
        assert result.db_committed is False
        assert any("scene-02" in r for r in result.reasons)
        assert get_project(conn, project_id) == before
        assert list_project_transitions(conn, project_id) == before_transitions

        _write_and_register(conn, project_id, project_dir, "audio", "scene-02", "audio/scene-02.wav")
        result = verify_and_advance(
            conn, project, manifest, list_artifacts_by_project(conn, project_id), "audio_ready", "narration recorded"
        )
        assert result.approved is True
        assert result.db_committed is True
        assert result.lifecycle_version_after == project.lifecycle_version + 1
        assert len(result.required_artifacts) == 2
        assert len(result.artifact_verification_results) == 2
        assert all(r.passed for r in result.artifact_verification_results)
        project = get_project(conn, project_id)
        assert project.current_stage == "audio_ready"
        assert project.last_successful_stage == "audio_ready"
        assert list_project_transitions(conn, project_id)[-1].reason == "narration recorded"

        project = _advance_plain(conn, project, "visuals_pending")

        # visuals_ready: nothing registered yet -> both scenes missing
        result = verify_and_advance(
            conn, project, manifest, list_artifacts_by_project(conn, project_id), "visuals_ready", "images generated"
        )
        assert result.approved is False
        assert len(result.reasons) == 2

        _write_and_register(conn, project_id, project_dir, "visual", "scene-01", "visual/scene-01.png")
        _write_and_register(conn, project_id, project_dir, "visual", "scene-02", "visual/scene-02.png")
        result = verify_and_advance(
            conn, project, manifest, list_artifacts_by_project(conn, project_id), "visuals_ready", "images generated"
        )
        assert result.approved and result.db_committed
        project = get_project(conn, project_id)
        assert project.current_stage == "visuals_ready"

        project = _advance_plain(conn, project, "animation_pending")

        _write_and_register(conn, project_id, project_dir, "animation", "scene-01", "animation/scene-01.json")
        result = verify_and_advance(
            conn, project, manifest, list_artifacts_by_project(conn, project_id), "animation_ready", "animation complete"
        )
        assert result.approved is False
        assert any("scene-02" in r for r in result.reasons)

        _write_and_register(conn, project_id, project_dir, "animation", "scene-02", "animation/scene-02.json")
        result = verify_and_advance(
            conn, project, manifest, list_artifacts_by_project(conn, project_id), "animation_ready", "animation complete"
        )
        assert result.approved and result.db_committed
        project = get_project(conn, project_id)
        assert project.current_stage == "animation_ready"

        project = _advance_plain(conn, project, "render_pending")

        result = verify_and_advance(
            conn, project, manifest, list_artifacts_by_project(conn, project_id), "rendered", "final video assembled"
        )
        assert result.approved is False
        assert any("render" in r for r in result.reasons)

        _write_and_register(
            conn, project_id, project_dir, "render", None, "render/final.mp4", artifact_id="render-final"
        )
        result = verify_and_advance(
            conn, project, manifest, list_artifacts_by_project(conn, project_id), "rendered", "final video assembled"
        )
        assert result.approved and result.db_committed
        project = get_project(conn, project_id)
        assert project.current_stage == "rendered"

        project = _advance_plain(conn, project, "qc_pending")

        result = verify_and_advance(
            conn, project, manifest, list_artifacts_by_project(conn, project_id), "qc_passed", "QC review passed"
        )
        assert result.approved is False
        assert any("qc_report" in r for r in result.reasons)

        # Phase 2H: qc_passed additionally requires the verified qc_report
        # FILE's own JSON to say passed=true — this real round trip proves
        # a genuine passed:true report satisfies that gate end to end.
        _write_and_register(
            conn, project_id, project_dir, "qc_report", None, "qc/report.json", artifact_id="qc-report-final",
            content=b'{"passed": true}',
        )
        result = verify_and_advance(
            conn, project, manifest, list_artifacts_by_project(conn, project_id), "qc_passed", "QC review passed"
        )
        assert result.approved and result.db_committed
        assert len(result.required_artifacts) == 2
        project = get_project(conn, project_id)
        assert project.current_stage == "qc_passed"
        assert project.last_successful_stage == "qc_passed"

        project = _advance_plain(conn, project, "ready_for_manual_publish")

        # "completed" is never automated: a missing reason is rejected...
        result = verify_and_advance(
            conn, project, manifest, list_artifacts_by_project(conn, project_id), "completed", ""
        )
        assert result.approved is False
        project_unchanged = get_project(conn, project_id)
        assert project_unchanged.current_stage == "ready_for_manual_publish"

        # ...and only an explicit reason completes it, via the same
        # guarded write path as every other verified transition.
        result = verify_and_advance(
            conn, project, manifest, list_artifacts_by_project(conn, project_id), "completed", "manual completion recorded"
        )
        assert result.approved and result.db_committed
        assert result.required_artifacts == ()
        project = get_project(conn, project_id)
        assert project.current_stage == "completed"
        assert project.completed_at is not None
    finally:
        conn.close()


# ---------------------------------------------------------------------
# rejections that must never touch the database
# ---------------------------------------------------------------------


def test_verify_and_advance_rejects_target_stage_that_is_not_verification_required(isolated_db):
    project_id, _project_dir, manifest = _create_registered_project(isolated_db / "projects")
    conn = _get_conn()
    try:
        project = get_project(conn, project_id)
        before = get_project(conn, project_id)

        result = verify_and_advance(conn, project, manifest, [], "render_pending", "not a real checkpoint")

        assert result.approved is False
        assert result.db_committed is False
        assert "not a verification-required target stage" in result.reasons[0]
        assert get_project(conn, project_id) == before
    finally:
        conn.close()


def test_verify_and_advance_rejects_illegal_skip_even_with_valid_artifacts(isolated_db):
    """Every required artifact is present and would verify cleanly, but the
    project is still at 'planned' — skipping straight to 'audio_ready'
    must be rejected by the state machine, not silently allowed just
    because the artifacts exist."""
    project_id, project_dir, manifest = _create_registered_project(isolated_db / "projects")
    conn = _get_conn()
    try:
        project = get_project(conn, project_id)
        assert project.current_stage == "planned"
        _write_and_register(conn, project_id, project_dir, "audio", "scene-01", "audio/scene-01.wav")
        _write_and_register(conn, project_id, project_dir, "audio", "scene-02", "audio/scene-02.wav")

        before = get_project(conn, project_id)
        before_transitions = list_project_transitions(conn, project_id)
        result = verify_and_advance(
            conn, project, manifest, list_artifacts_by_project(conn, project_id), "audio_ready", "skip attempt"
        )

        assert result.approved is False
        assert result.db_committed is False
        assert get_project(conn, project_id) == before
        assert list_project_transitions(conn, project_id) == before_transitions
    finally:
        conn.close()


def test_verify_and_advance_rejects_regression_to_earlier_stage(isolated_db):
    project_id, project_dir, manifest = _create_registered_project(isolated_db / "projects")
    conn = _get_conn()
    try:
        project = get_project(conn, project_id)
        project = _advance_plain(conn, project, "audio_pending")
        _write_and_register(conn, project_id, project_dir, "audio", "scene-01", "audio/scene-01.wav")
        _write_and_register(conn, project_id, project_dir, "audio", "scene-02", "audio/scene-02.wav")
        result = verify_and_advance(
            conn, project, manifest, list_artifacts_by_project(conn, project_id), "audio_ready", "narration recorded"
        )
        assert result.db_committed
        project = get_project(conn, project_id)
        project = _advance_plain(conn, project, "visuals_pending")

        before = get_project(conn, project_id)
        before_transitions = list_project_transitions(conn, project_id)
        result = verify_and_advance(
            conn, project, manifest, list_artifacts_by_project(conn, project_id), "audio_ready", "trying to regress"
        )

        assert result.approved is False
        assert result.db_committed is False
        assert get_project(conn, project_id) == before
        assert list_project_transitions(conn, project_id) == before_transitions
    finally:
        conn.close()


def test_verify_and_advance_requires_non_empty_reason(isolated_db):
    project_id, project_dir, manifest = _create_registered_project(isolated_db / "projects")
    conn = _get_conn()
    try:
        project = get_project(conn, project_id)
        project = _advance_plain(conn, project, "audio_pending")
        _write_and_register(conn, project_id, project_dir, "audio", "scene-01", "audio/scene-01.wav")
        _write_and_register(conn, project_id, project_dir, "audio", "scene-02", "audio/scene-02.wav")

        result = verify_and_advance(
            conn, project, manifest, list_artifacts_by_project(conn, project_id), "audio_ready", "   "
        )

        assert result.approved is False
        assert any("reason" in r for r in result.reasons)
        assert get_project(conn, project_id).current_stage == "audio_pending"
    finally:
        conn.close()


def test_verify_and_advance_invalid_checksum_rejects_and_writes_nothing(isolated_db):
    project_id, project_dir, manifest = _create_registered_project(isolated_db / "projects")
    conn = _get_conn()
    try:
        project = get_project(conn, project_id)
        project = _advance_plain(conn, project, "audio_pending")
        _write_and_register(conn, project_id, project_dir, "audio", "scene-01", "audio/scene-01.wav")

        content = b"real audio bytes"
        path = project_dir / "audio" / "scene-02.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        register_artifact(
            conn,
            ArtifactRecord(
                artifact_id="audio-scene-02",
                project_id=project_id,
                kind="audio",
                scene_id="scene-02",
                relative_path="audio/scene-02.wav",
                byte_size=len(content),
                sha256_checksum="0" * 64,
                created_at=FIXED_NOW,
            ),
        )

        before = get_project(conn, project_id)
        result = verify_and_advance(
            conn, project, manifest, list_artifacts_by_project(conn, project_id), "audio_ready", "narration recorded"
        )

        assert result.approved is False
        assert any("checksum mismatch" in r for r in result.reasons)
        assert get_project(conn, project_id) == before
    finally:
        conn.close()


def test_verify_and_advance_wrong_byte_size_rejects(isolated_db):
    project_id, project_dir, manifest = _create_registered_project(isolated_db / "projects")
    conn = _get_conn()
    try:
        project = get_project(conn, project_id)
        project = _advance_plain(conn, project, "audio_pending")
        _write_and_register(conn, project_id, project_dir, "audio", "scene-01", "audio/scene-01.wav")

        content = b"real audio bytes"
        path = project_dir / "audio" / "scene-02.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        register_artifact(
            conn,
            ArtifactRecord(
                artifact_id="audio-scene-02",
                project_id=project_id,
                kind="audio",
                scene_id="scene-02",
                relative_path="audio/scene-02.wav",
                byte_size=len(content) + 5,
                sha256_checksum=hashlib.sha256(content).hexdigest(),
                created_at=FIXED_NOW,
            ),
        )

        before = get_project(conn, project_id)
        result = verify_and_advance(
            conn, project, manifest, list_artifacts_by_project(conn, project_id), "audio_ready", "narration recorded"
        )

        assert result.approved is False
        assert any("byte_size mismatch" in r for r in result.reasons)
        assert get_project(conn, project_id) == before
    finally:
        conn.close()


def test_verify_and_advance_unsafe_path_rejects_without_touching_db(isolated_db):
    """Mirrors tests/test_artifact_verifier.py's model_construct() bypass
    technique: proves the service relies on artifact_verifier's real
    filesystem-aware safety check rather than trusting ArtifactRecord's
    string-only validator alone."""
    project_id, project_dir, manifest = _create_registered_project(isolated_db / "projects")
    conn = _get_conn()
    try:
        project = get_project(conn, project_id)
        project = _advance_plain(conn, project, "audio_pending")
        _write_and_register(conn, project_id, project_dir, "audio", "scene-01", "audio/scene-01.wav")

        outside = project_dir.parent / "outside-secret.wav"
        outside.write_bytes(b"secret")
        bad_artifact = ArtifactRecord.model_construct(
            artifact_id="audio-scene-02",
            project_id=project_id,
            kind="audio",
            scene_id="scene-02",
            relative_path=str(outside),
            byte_size=6,
            sha256_checksum=hashlib.sha256(b"secret").hexdigest(),
            created_at=FIXED_NOW,
            metadata={},
        )

        before = get_project(conn, project_id)
        artifacts = list(list_artifacts_by_project(conn, project_id)) + [bad_artifact]
        result = verify_and_advance(conn, project, manifest, artifacts, "audio_ready", "narration recorded")

        assert result.approved is False
        assert any("unsafe" in r for r in result.reasons)
        assert get_project(conn, project_id) == before
    finally:
        conn.close()


def test_verify_and_advance_duplicate_artifacts_for_same_scene_rejects(isolated_db):
    project_id, project_dir, manifest = _create_registered_project(isolated_db / "projects")
    conn = _get_conn()
    try:
        project = get_project(conn, project_id)
        project = _advance_plain(conn, project, "audio_pending")
        _write_and_register(
            conn, project_id, project_dir, "audio", "scene-01", "audio/scene-01.wav", artifact_id="audio-01-take-a"
        )
        _write_and_register(
            conn,
            project_id,
            project_dir,
            "audio",
            "scene-01",
            "audio/scene-01-take2.wav",
            artifact_id="audio-01-take-b",
        )
        _write_and_register(conn, project_id, project_dir, "audio", "scene-02", "audio/scene-02.wav")

        before = get_project(conn, project_id)
        result = verify_and_advance(
            conn, project, manifest, list_artifacts_by_project(conn, project_id), "audio_ready", "narration recorded"
        )

        assert result.approved is False
        assert any("duplicate/conflicting" in r for r in result.reasons)
        assert get_project(conn, project_id) == before
    finally:
        conn.close()


def test_verify_and_advance_artifact_project_id_mismatch_rejects(isolated_db):
    project_id, project_dir, manifest = _create_registered_project(isolated_db / "projects")
    conn = _get_conn()
    try:
        project = get_project(conn, project_id)
        project = _advance_plain(conn, project, "audio_pending")
        _write_and_register(conn, project_id, project_dir, "audio", "scene-02", "audio/scene-02.wav")

        content = b"scene one audio"
        path = project_dir / "audio" / "scene-01.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        # Never registered via register_artifact() (that would fail the
        # project foreign-key check) — passed directly to exercise the
        # per-artifact project/manifest identity check that
        # src/core/artifact_verifier.py performs, without this service
        # re-implementing it.
        mismatched = ArtifactRecord(
            artifact_id="audio-scene-01",
            project_id="a-totally-different-project",
            kind="audio",
            scene_id="scene-01",
            relative_path="audio/scene-01.wav",
            byte_size=len(content),
            sha256_checksum=hashlib.sha256(content).hexdigest(),
            created_at=FIXED_NOW,
        )

        before = get_project(conn, project_id)
        artifacts = list(list_artifacts_by_project(conn, project_id)) + [mismatched]
        result = verify_and_advance(conn, project, manifest, artifacts, "audio_ready", "narration recorded")

        assert result.approved is False
        assert any("does not match" in r for r in result.reasons)
        assert get_project(conn, project_id) == before
    finally:
        conn.close()


def test_verify_and_advance_raises_on_manifest_project_id_mismatch(isolated_db):
    project_id, _dir_a, _manifest_a = _create_registered_project(isolated_db / "projects-a")
    _other_id, _dir_b, manifest_b = _create_registered_project(
        isolated_db / "projects-b", story_id="a-completely-different-story"
    )
    conn = _get_conn()
    try:
        project = get_project(conn, project_id)
        with pytest.raises(VerifiedTransitionServiceError):
            verify_and_advance(conn, project, manifest_b, [], "audio_ready", "mismatched manifest")
    finally:
        conn.close()


def test_verify_and_advance_stale_lifecycle_version_causes_no_partial_write(isolated_db):
    project_id, project_dir, manifest = _create_registered_project(isolated_db / "projects")
    conn = _get_conn()
    try:
        project = get_project(conn, project_id)
        project = _advance_plain(conn, project, "audio_pending")
        _write_and_register(conn, project_id, project_dir, "audio", "scene-01", "audio/scene-01.wav")
        _write_and_register(conn, project_id, project_dir, "audio", "scene-02", "audio/scene-02.wav")

        # Simulate a concurrent writer bumping the row's lifecycle_version
        # out from under our in-hand `project` object — save_transition's
        # own optimistic-lock WHERE clause must catch this.
        with conn:
            conn.execute(
                "UPDATE projects SET lifecycle_version = ? WHERE project_id = ?",
                (project.lifecycle_version + 1, project_id),
            )

        before = get_project(conn, project_id)
        before_transitions = list_project_transitions(conn, project_id)

        result = verify_and_advance(
            conn, project, manifest, list_artifacts_by_project(conn, project_id), "audio_ready", "narration recorded"
        )

        assert result.approved is True  # verification + state-machine legality both passed
        assert result.db_committed is False
        assert any("lifecycle_version" in r for r in result.reasons)
        assert result.lifecycle_version_after is None
        assert get_project(conn, project_id) == before
        assert list_project_transitions(conn, project_id) == before_transitions
    finally:
        conn.close()


def test_verify_and_advance_saves_exactly_one_transition_on_success(isolated_db):
    project_id, project_dir, manifest = _create_registered_project(isolated_db / "projects")
    conn = _get_conn()
    try:
        project = get_project(conn, project_id)
        project = _advance_plain(conn, project, "audio_pending")
        _write_and_register(conn, project_id, project_dir, "audio", "scene-01", "audio/scene-01.wav")
        _write_and_register(conn, project_id, project_dir, "audio", "scene-02", "audio/scene-02.wav")

        before_count = len(list_project_transitions(conn, project_id))
        result = verify_and_advance(
            conn, project, manifest, list_artifacts_by_project(conn, project_id), "audio_ready", "narration recorded"
        )
        after = list_project_transitions(conn, project_id)

        assert result.db_committed is True
        assert len(after) == before_count + 1
        assert after[-1].to_stage == "audio_ready"
        assert after[-1].lifecycle_version == result.lifecycle_version_after
    finally:
        conn.close()


def test_verify_and_advance_zero_changes_anywhere_when_verification_fails(isolated_db):
    project_id, project_dir, manifest = _create_registered_project(isolated_db / "projects")
    conn = _get_conn()
    try:
        project = get_project(conn, project_id)
        project = _advance_plain(conn, project, "audio_pending")
        _write_and_register(conn, project_id, project_dir, "audio", "scene-01", "audio/scene-01.wav")

        from src.utils.config import get_settings

        db_path = get_settings().data_dir / "jobs.db"
        db_bytes_before = db_path.read_bytes()
        manifest_bytes_before = (project_dir / "manifest.json").read_bytes()
        audio_bytes_before = (project_dir / "audio" / "scene-01.wav").read_bytes()
        files_before = sorted(
            p.relative_to(project_dir).as_posix() for p in project_dir.rglob("*") if p.is_file()
        )

        result = verify_and_advance(
            conn, project, manifest, list_artifacts_by_project(conn, project_id), "audio_ready", "narration recorded"
        )

        assert result.approved is False
        assert db_path.read_bytes() == db_bytes_before
        assert (project_dir / "manifest.json").read_bytes() == manifest_bytes_before
        assert (project_dir / "audio" / "scene-01.wav").read_bytes() == audio_bytes_before
        files_after = sorted(
            p.relative_to(project_dir).as_posix() for p in project_dir.rglob("*") if p.is_file()
        )
        assert files_after == files_before
    finally:
        conn.close()


# ---------------------------------------------------------------------
# no provider/render/network/subprocess module is ever imported
# ---------------------------------------------------------------------


def test_verify_and_advance_does_not_import_provider_render_network_or_subprocess_modules(
    isolated_db, monkeypatch
):
    project_id, project_dir, manifest = _create_registered_project(isolated_db / "projects")
    conn = _get_conn()
    try:
        project = get_project(conn, project_id)
        project = _advance_plain(conn, project, "audio_pending")
        _write_and_register(conn, project_id, project_dir, "audio", "scene-01", "audio/scene-01.wav")
        _write_and_register(conn, project_id, project_dir, "audio", "scene-02", "audio/scene-02.wav")

        real_import = builtins.__import__
        forbidden_prefixes = ("src.providers", "src.render", "subprocess", "socket")

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.startswith(forbidden_prefixes):
                raise AssertionError(f"verify_and_advance must not import {name}")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        result = verify_and_advance(
            conn, project, manifest, list_artifacts_by_project(conn, project_id), "audio_ready", "narration recorded"
        )
        assert result.db_committed is True
    finally:
        conn.close()


def test_supported_target_stages_matches_spec_exactly():
    assert SUPPORTED_TARGET_STAGES == {
        "audio_ready",
        "visuals_ready",
        "animation_ready",
        "rendered",
        "qc_passed",
        "completed",
    }


# ---------------------------------------------------------------------
# Phase 2H: qc_passed-only semantic gate on the verified qc_report file
# ---------------------------------------------------------------------


def _project_at_qc_pending_with_render(conn, project_id, project_dir):
    """Fast-forward a project straight to qc_pending with a render
    artifact already registered. verify_and_advance() only ever checks
    the CURRENT target stage's own requirement, never earlier stages', so
    no audio/visual/animation artifacts need to exist for these tests —
    only the render half of qc_passed's two-artifact requirement, plus
    whatever qc_report each test registers itself."""
    project = get_project(conn, project_id)
    for stage, verified in [
        ("audio_pending", False), ("audio_ready", True),
        ("visuals_pending", False), ("visuals_ready", True),
        ("animation_pending", False), ("animation_ready", True),
        ("render_pending", False), ("rendered", True),
        ("qc_pending", False),
    ]:
        updated, transition = transition_project(project, stage, now=FIXED_NOW, verified=verified)
        save_transition(conn, project.lifecycle_version, updated, transition)
        project = get_project(conn, project_id)
    _write_and_register(conn, project_id, project_dir, "render", None, "render/final.mp4", artifact_id="render-final")
    return project


def test_qc_passed_rejects_report_with_passed_false_and_writes_nothing(isolated_db):
    """A structurally valid, checksum-matching qc_report saying
    passed:false is real, registered data — but must never by itself
    satisfy qc_passed."""
    project_id, project_dir, manifest = _create_registered_project(isolated_db / "projects")
    conn = _get_conn()
    try:
        project = _project_at_qc_pending_with_render(conn, project_id, project_dir)
        _write_and_register(
            conn, project_id, project_dir, "qc_report", None, "qc/report.json", artifact_id="qc-report-final",
            content=b'{"passed": false}',
        )
        before = get_project(conn, project_id)
        before_transitions = list_project_transitions(conn, project_id)

        result = verify_and_advance(
            conn, project, manifest, list_artifacts_by_project(conn, project_id), "qc_passed", "QC review"
        )

        assert result.approved is False
        assert result.db_committed is False
        assert any("passed=false" in r for r in result.reasons)
        assert get_project(conn, project_id) == before  # zero DB write
        assert list_project_transitions(conn, project_id) == before_transitions
    finally:
        conn.close()


def test_qc_passed_tampered_report_fails_checksum_before_semantic_parsing(isolated_db):
    """The report is registered as passed:true (so IF semantic parsing
    were reached, it would pass), but its on-disk bytes are changed
    afterward to different-but-still-valid passed:true JSON — a different
    checksum either way. The rejection reason must be the structural
    checksum mismatch, never a semantic 'passed' message, proving
    structural verification runs (and can reject) before semantic parsing
    is ever attempted."""
    project_id, project_dir, manifest = _create_registered_project(isolated_db / "projects")
    conn = _get_conn()
    try:
        project = _project_at_qc_pending_with_render(conn, project_id, project_dir)
        _write_and_register(
            conn, project_id, project_dir, "qc_report", None, "qc/report.json", artifact_id="qc-report-final",
            content=b'{"passed": true}',
        )
        # Tamper with the file after registration — still valid,
        # still-true JSON, just different bytes than what was registered.
        (project_dir / "qc" / "report.json").write_bytes(b'{"passed": true, "note": "tampered"}')

        result = verify_and_advance(
            conn, project, manifest, list_artifacts_by_project(conn, project_id), "qc_passed", "QC review"
        )

        assert result.approved is False
        assert result.db_committed is False
        assert any("checksum mismatch" in r for r in result.reasons)
        assert not any("passed=false" in r for r in result.reasons)
    finally:
        conn.close()


def test_qc_passed_missing_qc_report_still_rejects_with_render_present(isolated_db):
    """Regression check: render alone (Phase 2G's own requirement) must
    still be insufficient for qc_passed without any qc_report at all —
    unchanged, pre-existing behavior, not new."""
    project_id, project_dir, manifest = _create_registered_project(isolated_db / "projects")
    conn = _get_conn()
    try:
        project = _project_at_qc_pending_with_render(conn, project_id, project_dir)

        result = verify_and_advance(
            conn, project, manifest, list_artifacts_by_project(conn, project_id), "qc_passed", "QC review"
        )

        assert result.approved is False
        assert result.db_committed is False
        assert any("qc_report" in r for r in result.reasons)
    finally:
        conn.close()


def test_qc_passed_accepts_report_with_passed_true(isolated_db):
    project_id, project_dir, manifest = _create_registered_project(isolated_db / "projects")
    conn = _get_conn()
    try:
        project = _project_at_qc_pending_with_render(conn, project_id, project_dir)
        _write_and_register(
            conn, project_id, project_dir, "qc_report", None, "qc/report.json", artifact_id="qc-report-final",
            content=b'{"passed": true}',
        )

        result = verify_and_advance(
            conn, project, manifest, list_artifacts_by_project(conn, project_id), "qc_passed", "QC review"
        )

        assert result.approved is True
        assert result.db_committed is True
        project = get_project(conn, project_id)
        assert project.current_stage == "qc_passed"
    finally:
        conn.close()
