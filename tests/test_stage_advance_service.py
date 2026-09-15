"""Tests for src/core/stage_advance_service.py — Phase 2I's sole approved
path for a single non-verification forward move. Same isolated_db /
build-manifest-then-register-project pattern as
tests/test_verified_transition_service.py; no artifacts are needed here
at all, only ProjectRecord + transition rows."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.core.manifest_builder import build_video_manifest
from src.core.manifest_store import save_manifest
from src.core.project_state_machine import (
    create_initial_project,
    initial_transition_for,
    mark_project_failed,
    transition_project,
)
from src.core.stage_advance_service import NON_VERIFICATION_TARGET_STAGES, advance_project_stage
from src.database.project_repository import (
    ProjectConcurrencyError,
    create_project,
    get_project,
    list_project_transitions,
    save_transition,
)
from src.utils.channel_config import get_channel_policy

FIXED_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

# The full forward chain past "planned", each entry's own verified
# requirement — used only to fast-forward test setup via
# transition_project()/save_transition() directly, never via the code
# under test.
_CHAIN: tuple[tuple[str, bool], ...] = (
    ("audio_pending", False),
    ("audio_ready", True),
    ("visuals_pending", False),
    ("visuals_ready", True),
    ("animation_pending", False),
    ("animation_ready", True),
    ("render_pending", False),
    ("rendered", True),
    ("qc_pending", False),
    ("qc_passed", True),
    ("ready_for_manual_publish", False),
)

_VERIFICATION_REQUIRED_STAGES = ("audio_ready", "visuals_ready", "animation_ready", "rendered", "qc_passed")


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
        ),
        role_outfits=(),
    )


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    from src.utils import config

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    config.get_settings.cache_clear()
    yield tmp_path
    config.get_settings.cache_clear()


def _create_registered_project(projects_root: Path):
    from src.database.db import get_connection, init_db

    manifest = build_video_manifest(
        _story_input_dict(), _scene_plan_dict(), get_channel_policy(), created_at=FIXED_NOW
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
    return project.project_id


def _get_conn():
    from src.database.db import get_connection

    return get_connection()


def _fast_forward_before(conn, project_id, target_stage):
    """Walk the project, via transition_project()+save_transition()
    directly (test setup only — never the code under test), up to but not
    including `target_stage`. Returns the resulting ProjectRecord."""
    project = get_project(conn, project_id)
    for stage, verified in _CHAIN:
        if stage == target_stage:
            break
        updated, transition = transition_project(project, stage, now=FIXED_NOW, verified=verified)
        save_transition(conn, project.lifecycle_version, updated, transition)
        project = get_project(conn, project_id)
    return project


# ---------------------------------------------------------------------
# every legal transition succeeds
# ---------------------------------------------------------------------


@pytest.mark.parametrize("to_stage", sorted(NON_VERIFICATION_TARGET_STAGES))
def test_allows_each_legal_non_verification_transition(isolated_db, to_stage):
    project_id = _create_registered_project(isolated_db / "projects")
    conn = _get_conn()
    try:
        project = _fast_forward_before(conn, project_id, to_stage)
        before_version = project.lifecycle_version

        result = advance_project_stage(conn, project, to_stage, reason="routine advance")

        assert result.approved is True
        assert result.db_committed is True
        assert result.reasons == ()
        assert result.lifecycle_version_before == before_version
        assert result.lifecycle_version_after == before_version + 1

        updated = get_project(conn, project_id)
        assert updated.current_stage == to_stage
        assert updated.lifecycle_version == before_version + 1

        transitions = list_project_transitions(conn, project_id)
        assert transitions[-1].to_stage == to_stage
        assert transitions[-1].reason == "routine advance"
        assert transitions[-1].is_retry is False
    finally:
        conn.close()


def test_reason_is_optional_and_defaults_to_none(isolated_db):
    project_id = _create_registered_project(isolated_db / "projects")
    conn = _get_conn()
    try:
        project = get_project(conn, project_id)
        result = advance_project_stage(conn, project, "audio_pending")

        assert result.approved is True
        assert result.reason is None
        transitions = list_project_transitions(conn, project_id)
        assert transitions[-1].reason is None
    finally:
        conn.close()


# ---------------------------------------------------------------------
# every verification-required / out-of-scope target is rejected, with
# zero writes, regardless of current_stage
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "to_stage",
    [*_VERIFICATION_REQUIRED_STAGES, "completed", "archived", "failed"],
)
def test_rejects_every_out_of_scope_target_with_zero_writes(isolated_db, to_stage):
    project_id = _create_registered_project(isolated_db / "projects")
    conn = _get_conn()
    try:
        before = get_project(conn, project_id)
        before_transitions = list_project_transitions(conn, project_id)

        result = advance_project_stage(conn, before, to_stage, reason="attempt")

        assert result.approved is False
        assert result.db_committed is False
        assert result.reasons
        assert any("verify-and-advance" in r or "not a non-verification target stage" in r for r in result.reasons)

        after = get_project(conn, project_id)
        assert after == before
        assert list_project_transitions(conn, project_id) == before_transitions
    finally:
        conn.close()


# ---------------------------------------------------------------------
# illegal transitions from an otherwise-legal current stage: skip,
# regression — caught via transition_project()'s own
# ProjectStateTransitionError, not reimplemented
# ---------------------------------------------------------------------


def test_rejects_skip_transition_with_zero_writes(isolated_db):
    """From "planned", the only legal target is "audio_pending" — skipping
    straight to "visuals_pending" (itself a legal MEMBER of
    NON_VERIFICATION_TARGET_STAGES, just not project's actual next stage)
    must still be rejected."""
    project_id = _create_registered_project(isolated_db / "projects")
    conn = _get_conn()
    try:
        before = get_project(conn, project_id)

        result = advance_project_stage(conn, before, "visuals_pending")

        assert result.approved is False
        assert result.db_committed is False
        assert any("not an allowed transition" in r for r in result.reasons)
        assert get_project(conn, project_id) == before
    finally:
        conn.close()


def test_rejects_regression_with_zero_writes(isolated_db):
    project_id = _create_registered_project(isolated_db / "projects")
    conn = _get_conn()
    try:
        project = _fast_forward_before(conn, project_id, "render_pending")
        before = get_project(conn, project_id)

        result = advance_project_stage(conn, project, "audio_pending")  # regression

        assert result.approved is False
        assert result.db_committed is False
        assert any("not an allowed transition" in r for r in result.reasons)
        assert get_project(conn, project_id) == before
    finally:
        conn.close()


# ---------------------------------------------------------------------
# terminal / failed current_stage
# ---------------------------------------------------------------------


def test_rejects_from_archived_current_stage(isolated_db):
    project_id = _create_registered_project(isolated_db / "projects")
    conn = _get_conn()
    try:
        project = get_project(conn, project_id)
        archived, transition = transition_project(project, "archived", now=FIXED_NOW, reason="shelved")
        save_transition(conn, project.lifecycle_version, archived, transition)
        before = get_project(conn, project_id)

        result = advance_project_stage(conn, before, "audio_pending")

        assert result.approved is False
        assert result.db_committed is False
        assert any("archived" in r and "terminal" in r for r in result.reasons)
        assert get_project(conn, project_id) == before
    finally:
        conn.close()


def test_rejects_from_completed_current_stage(isolated_db):
    project_id = _create_registered_project(isolated_db / "projects")
    conn = _get_conn()
    try:
        project = _fast_forward_before(conn, project_id, "completed")
        updated, transition = transition_project(
            project, "completed", now=FIXED_NOW, reason="manual completion recorded", verified=True
        )
        save_transition(conn, project.lifecycle_version, updated, transition)
        before = get_project(conn, project_id)

        result = advance_project_stage(conn, before, "audio_pending")

        assert result.approved is False
        assert result.db_committed is False
        assert any("completed" in r for r in result.reasons)
        assert get_project(conn, project_id) == before
    finally:
        conn.close()


def test_rejects_from_failed_current_stage_even_when_to_stage_matches_failed_stage(isolated_db):
    """transition_project() itself would legally accept this as a RETRY —
    advance-project-stage deliberately excludes retries anyway, via its
    own explicit current_stage=='failed' guard."""
    project_id = _create_registered_project(isolated_db / "projects")
    conn = _get_conn()
    try:
        project = get_project(conn, project_id)  # current_stage == "planned"
        updated, transition = transition_project(project, "audio_pending", now=FIXED_NOW)
        save_transition(conn, project.lifecycle_version, updated, transition)
        project = get_project(conn, project_id)

        failed, fail_transition = mark_project_failed(project, "audio_pending", "tts provider down", now=FIXED_NOW)
        save_transition(conn, project.lifecycle_version, failed, fail_transition)
        before = get_project(conn, project_id)
        assert before.current_stage == "failed"
        assert before.failed_stage == "audio_pending"

        result = advance_project_stage(conn, before, "audio_pending")  # matches failed_stage

        assert result.approved is False
        assert result.db_committed is False
        assert any("does not perform retries" in r for r in result.reasons)
        assert get_project(conn, project_id) == before
    finally:
        conn.close()


def test_rejects_from_failed_current_stage_for_an_unrelated_target_too(isolated_db):
    project_id = _create_registered_project(isolated_db / "projects")
    conn = _get_conn()
    try:
        project = get_project(conn, project_id)
        updated, transition = transition_project(project, "audio_pending", now=FIXED_NOW)
        save_transition(conn, project.lifecycle_version, updated, transition)
        project = get_project(conn, project_id)
        failed, fail_transition = mark_project_failed(project, "audio_pending", "boom", now=FIXED_NOW)
        save_transition(conn, project.lifecycle_version, failed, fail_transition)
        before = get_project(conn, project_id)

        result = advance_project_stage(conn, before, "visuals_pending")

        assert result.approved is False
        assert result.db_committed is False
        assert any("does not perform retries" in r for r in result.reasons)
        assert get_project(conn, project_id) == before
    finally:
        conn.close()


# ---------------------------------------------------------------------
# concurrency: stale lifecycle_version
# ---------------------------------------------------------------------


def test_stale_lifecycle_version_is_reported_not_written(isolated_db):
    project_id = _create_registered_project(isolated_db / "projects")
    conn = _get_conn()
    try:
        project = get_project(conn, project_id)  # lifecycle_version == 1

        # Simulate a concurrent writer: advance the row to version 2
        # behind this in-memory `project`'s back.
        updated, transition = transition_project(project, "audio_pending", now=FIXED_NOW)
        save_transition(conn, project.lifecycle_version, updated, transition)

        # `project` is still the stale version=1 snapshot.
        result = advance_project_stage(conn, project, "audio_pending")

        assert result.approved is True  # the in-memory transition itself was legal
        assert result.db_committed is False  # but the write lost the optimistic-lock race
        assert result.reasons

        after = get_project(conn, project_id)
        assert after.lifecycle_version == 2  # unchanged by the failed second attempt
        assert after.current_stage == "audio_pending"  # from the FIRST (winning) write only
    finally:
        conn.close()


# ---------------------------------------------------------------------
# NON_VERIFICATION_TARGET_STAGES is exactly the six specified stages
# ---------------------------------------------------------------------


def test_non_verification_target_stages_matches_spec_exactly():
    assert NON_VERIFICATION_TARGET_STAGES == {
        "audio_pending",
        "visuals_pending",
        "animation_pending",
        "render_pending",
        "qc_pending",
        "ready_for_manual_publish",
    }
