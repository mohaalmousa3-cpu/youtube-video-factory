"""Tests for src/core/project_state_machine.py: pure local logic, no
SQLite, no provider, no network. A real VideoManifest (built via Phase
1C's build_video_manifest against the real ChannelPolicy) supplies
project_id/source_fingerprint for create_initial_project — everything
after that is pure ProjectRecord/ProjectTransition/ResumePlan logic."""
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.core.manifest_builder import build_video_manifest
from src.core.project_state_machine import (
    ProjectStateTransitionError,
    build_resume_plan,
    create_initial_project,
    initial_transition_for,
    mark_project_failed,
    transition_project,
)
from src.models.project_state import ProjectRecord, ProjectTransition, ResumePlan
from src.utils.channel_config import get_channel_policy

FIXED_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
LATER_NOW = datetime(2026, 1, 1, 12, 5, 0, tzinfo=timezone.utc)


def _valid_story_input(**overrides) -> dict:
    data = dict(
        story_id="why-we-care-what-people-think",
        title="Why We Care So Much What People Think",
        topic="social psychology",
        target_duration_seconds=480.0,
        language="en-US",
        viewer_facing_language="English",
        approval_status="approved",
    )
    data.update(overrides)
    return data


def _valid_scene_plan() -> dict:
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


def _make_manifest():
    return build_video_manifest(
        _valid_story_input(), _valid_scene_plan(), get_channel_policy(), created_at=FIXED_NOW
    )


def _initial_project() -> ProjectRecord:
    manifest = _make_manifest()
    return create_initial_project("data/projects/demo/manifest.json", manifest, now=FIXED_NOW)


def _advance(project: ProjectRecord, to_stage: str, **kwargs) -> ProjectRecord:
    updated, _ = transition_project(project, to_stage, now=FIXED_NOW, **kwargs)
    return updated


# ---------------------------------------------------------------------
# create_initial_project
# ---------------------------------------------------------------------


def test_create_initial_project_is_planned():
    project = _initial_project()
    assert project.current_stage == "planned"
    assert project.last_successful_stage == "planned"
    assert project.lifecycle_version == 1
    assert project.execution_status == "not_executed"
    assert project.failed_stage is None
    assert project.failure_message is None
    assert project.completed_at is None
    assert project.archived_at is None
    assert project.retry_count == 0


def test_create_initial_project_copies_project_id_and_fingerprint_from_manifest():
    manifest = _make_manifest()
    project = create_initial_project("data/projects/demo/manifest.json", manifest, now=FIXED_NOW)
    assert project.project_id == manifest.project_id
    assert project.manifest_fingerprint == manifest.source_fingerprint


def test_create_initial_project_uses_injected_now():
    project = _initial_project()
    assert project.created_at == FIXED_NOW
    assert project.updated_at == FIXED_NOW


def test_initial_transition_for_records_creation_into_planned():
    project = _initial_project()
    transition = initial_transition_for(project)
    assert transition.project_id == project.project_id
    assert transition.from_stage == "planned"
    assert transition.to_stage == "planned"
    assert transition.is_retry is False
    assert transition.lifecycle_version == 1


# ---------------------------------------------------------------------
# Canonical route (with success verification explicitly simulated)
# ---------------------------------------------------------------------


def test_canonical_route_reaches_completed():
    project = _initial_project()
    project = _advance(project, "audio_pending")
    project = _advance(project, "audio_ready", verified=True)
    project = _advance(project, "visuals_pending")
    project = _advance(project, "visuals_ready", verified=True)
    project = _advance(project, "animation_pending")
    project = _advance(project, "animation_ready", verified=True)
    project = _advance(project, "render_pending")
    project = _advance(project, "rendered", verified=True)
    project = _advance(project, "qc_pending")
    project = _advance(project, "qc_passed", verified=True)
    project = _advance(project, "ready_for_manual_publish")
    project = _advance(project, "completed", verified=True, reason="manual completion recorded")

    assert project.current_stage == "completed"
    assert project.last_successful_stage == "completed"
    assert project.completed_at == FIXED_NOW
    assert project.lifecycle_version == 13  # 1 initial + 12 transitions


def test_planned_skips_script_approved_directly_to_audio_pending():
    project = _initial_project()
    updated, transition = transition_project(project, "audio_pending", now=FIXED_NOW)
    assert updated.current_stage == "audio_pending"
    assert transition.from_stage == "planned"
    assert transition.to_stage == "audio_pending"


def test_script_approved_is_never_a_reachable_target():
    project = _initial_project()
    with pytest.raises(ProjectStateTransitionError):
        transition_project(project, "script_approved", now=FIXED_NOW)


# ---------------------------------------------------------------------
# Forbidden skips / regressions / direct jumps
# ---------------------------------------------------------------------


def test_forbidden_skip_raises():
    project = _initial_project()
    with pytest.raises(ProjectStateTransitionError):
        transition_project(project, "visuals_pending", now=FIXED_NOW)  # skips audio_* entirely


def test_regression_raises():
    project = _initial_project()
    project = _advance(project, "audio_pending")
    project = _advance(project, "audio_ready", verified=True)
    with pytest.raises(ProjectStateTransitionError):
        transition_project(project, "planned", now=FIXED_NOW)


def test_planned_to_completed_directly_raises():
    project = _initial_project()
    with pytest.raises(ProjectStateTransitionError):
        transition_project(project, "completed", now=FIXED_NOW, verified=True, reason="skip ahead")


# ---------------------------------------------------------------------
# Verification gate
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "from_stage, to_stage",
    [
        ("audio_pending", "audio_ready"),
        ("visuals_pending", "visuals_ready"),
        ("animation_pending", "animation_ready"),
        ("render_pending", "rendered"),
        ("qc_pending", "qc_passed"),
    ],
)
def test_ready_stage_requires_verified_true(from_stage, to_stage):
    project = _initial_project()
    # Walk up to and including `from_stage`.
    route = ["audio_pending", "audio_ready", "visuals_pending", "visuals_ready",
             "animation_pending", "animation_ready", "render_pending", "rendered",
             "qc_pending", "qc_passed"]
    for stage in route:
        verified = stage in ("audio_ready", "visuals_ready", "animation_ready", "rendered", "qc_passed")
        project = _advance(project, stage, verified=verified)
        if stage == from_stage:
            break

    assert project.current_stage == from_stage

    with pytest.raises(ProjectStateTransitionError):
        transition_project(project, to_stage, now=FIXED_NOW)  # verified defaults to False

    updated, _ = transition_project(project, to_stage, now=FIXED_NOW, verified=True)
    assert updated.current_stage == to_stage


def test_completed_requires_verified_and_reason():
    project = _initial_project()
    for stage, verified in [
        ("audio_pending", False), ("audio_ready", True), ("visuals_pending", False),
        ("visuals_ready", True), ("animation_pending", False), ("animation_ready", True),
        ("render_pending", False), ("rendered", True), ("qc_pending", False),
        ("qc_passed", True), ("ready_for_manual_publish", False),
    ]:
        project = _advance(project, stage, verified=verified)

    with pytest.raises(ProjectStateTransitionError):
        transition_project(project, "completed", now=FIXED_NOW, verified=True)  # no reason

    with pytest.raises(ProjectStateTransitionError):
        transition_project(project, "completed", now=FIXED_NOW, reason="manual completion recorded")  # not verified

    updated, _ = transition_project(
        project, "completed", now=FIXED_NOW, verified=True, reason="manual completion recorded"
    )
    assert updated.current_stage == "completed"


# ---------------------------------------------------------------------
# Failure / retry semantics
# ---------------------------------------------------------------------


def test_failure_from_audio_pending_records_correctly():
    project = _initial_project()
    project = _advance(project, "audio_pending")

    updated, transition = mark_project_failed(
        project, "audio_pending", "Kokoro synthesis crashed", now=FIXED_NOW
    )

    assert updated.current_stage == "failed"
    assert updated.failed_stage == "audio_pending"
    assert updated.failure_message == "Kokoro synthesis crashed"
    assert updated.last_successful_stage == "planned"  # unchanged by the failure
    assert transition.from_stage == "audio_pending"
    assert transition.to_stage == "failed"


def test_planned_cannot_be_marked_failed():
    """"planned" means a valid, approved manifest exists — a registry
    fact, not an in-progress execution stage — so there is nothing
    in-flight to fail. Only actual execution stages (audio_pending,
    visuals_pending, animation_pending, render_pending, qc_pending,
    ready_for_manual_publish) are fail-able."""
    project = _initial_project()

    with pytest.raises(ProjectStateTransitionError):
        mark_project_failed(project, "planned", "should be rejected", now=FIXED_NOW)

    # The rejected call left the project completely unchanged.
    unchanged = _initial_project()
    assert project == unchanged
    assert project.current_stage == "planned"
    assert project.failed_stage is None
    assert project.failure_message is None
    assert project.lifecycle_version == 1


def test_mark_project_failed_rejects_mismatched_failed_stage():
    project = _initial_project()
    project = _advance(project, "audio_pending")
    with pytest.raises(ProjectStateTransitionError):
        mark_project_failed(project, "visuals_pending", "wrong stage", now=FIXED_NOW)


def test_mark_project_failed_rejects_already_verified_stage():
    project = _initial_project()
    project = _advance(project, "audio_pending")
    project = _advance(project, "audio_ready", verified=True)
    with pytest.raises(ProjectStateTransitionError):
        mark_project_failed(project, "audio_ready", "cannot un-verify", now=FIXED_NOW)


def test_resume_plan_from_failed_returns_retry_stage():
    project = _initial_project()
    project = _advance(project, "audio_pending")
    failed, _ = mark_project_failed(project, "audio_pending", "boom", now=FIXED_NOW)

    plan = build_resume_plan(failed)

    assert plan.action == "retry_stage"
    assert plan.next_stage == "audio_pending"


def test_retry_clears_failure_fields_without_marking_success():
    project = _initial_project()
    project = _advance(project, "audio_pending")
    failed, _ = mark_project_failed(project, "audio_pending", "boom", now=FIXED_NOW)

    retried, transition = transition_project(failed, "audio_pending", now=LATER_NOW)

    assert retried.current_stage == "audio_pending"  # not audio_ready — no false success
    assert retried.failed_stage is None
    assert retried.failure_message is None
    assert retried.retry_count == 1
    assert transition.is_retry is True
    assert transition.from_stage == "failed"
    assert transition.to_stage == "audio_pending"


def test_retrying_a_stage_other_than_failed_stage_is_rejected():
    project = _initial_project()
    project = _advance(project, "audio_pending")
    failed, _ = mark_project_failed(project, "audio_pending", "boom", now=FIXED_NOW)

    with pytest.raises(ProjectStateTransitionError):
        transition_project(failed, "visuals_pending", now=FIXED_NOW)


def test_after_retry_normal_forward_rules_continue():
    project = _initial_project()
    project = _advance(project, "audio_pending")
    failed, _ = mark_project_failed(project, "audio_pending", "boom", now=FIXED_NOW)
    retried, _ = transition_project(failed, "audio_pending", now=LATER_NOW)

    updated, _ = transition_project(retried, "audio_ready", now=LATER_NOW, verified=True)
    assert updated.current_stage == "audio_ready"


# ---------------------------------------------------------------------
# Terminal states: archived, completed
# ---------------------------------------------------------------------


def test_archiving_requires_reason():
    project = _initial_project()
    with pytest.raises(ProjectStateTransitionError):
        transition_project(project, "archived", now=FIXED_NOW)


def test_archived_rejects_every_later_transition():
    project = _initial_project()
    archived, _ = transition_project(project, "archived", now=FIXED_NOW, reason="abandoned")

    assert archived.current_stage == "archived"
    assert archived.archived_at == FIXED_NOW

    for target in ("audio_pending", "planned", "completed", "archived"):
        with pytest.raises(ProjectStateTransitionError):
            transition_project(archived, target, now=LATER_NOW, verified=True, reason="x")


def test_archiving_is_allowed_from_an_active_in_progress_stage():
    project = _initial_project()
    project = _advance(project, "audio_pending")
    archived, transition = transition_project(project, "archived", now=FIXED_NOW, reason="priorities changed")
    assert archived.current_stage == "archived"
    assert transition.from_stage == "audio_pending"


def test_archiving_is_allowed_from_failed():
    project = _initial_project()
    project = _advance(project, "audio_pending")
    failed, _ = mark_project_failed(project, "audio_pending", "boom", now=FIXED_NOW)
    archived, _ = transition_project(failed, "archived", now=LATER_NOW, reason="giving up")
    assert archived.current_stage == "archived"
    assert archived.failed_stage is None  # cleared — failed_stage only makes sense while failed


def test_completed_only_allows_archiving_afterward():
    project = _initial_project()
    for stage, verified, reason in [
        ("audio_pending", False, None), ("audio_ready", True, None),
        ("visuals_pending", False, None), ("visuals_ready", True, None),
        ("animation_pending", False, None), ("animation_ready", True, None),
        ("render_pending", False, None), ("rendered", True, None),
        ("qc_pending", False, None), ("qc_passed", True, None),
        ("ready_for_manual_publish", False, None),
        ("completed", True, "manual completion recorded"),
    ]:
        project = _advance(project, stage, verified=verified, reason=reason)

    with pytest.raises(ProjectStateTransitionError):
        transition_project(project, "audio_pending", now=LATER_NOW)

    archived, _ = transition_project(project, "archived", now=LATER_NOW, reason="cleanup")
    assert archived.current_stage == "archived"


# ---------------------------------------------------------------------
# ResumePlan
# ---------------------------------------------------------------------


def test_resume_plan_for_planned_starts_audio_pending():
    plan = build_resume_plan(_initial_project())
    assert plan.action == "start_stage"
    assert plan.next_stage == "audio_pending"


def test_resume_plan_for_qc_passed_starts_ready_for_manual_publish():
    project = _initial_project()
    for stage, verified in [
        ("audio_pending", False), ("audio_ready", True), ("visuals_pending", False),
        ("visuals_ready", True), ("animation_pending", False), ("animation_ready", True),
        ("render_pending", False), ("rendered", True), ("qc_pending", False), ("qc_passed", True),
    ]:
        project = _advance(project, stage, verified=verified)

    plan = build_resume_plan(project)
    assert plan.action == "start_stage"
    assert plan.next_stage == "ready_for_manual_publish"


def test_resume_plan_for_ready_for_manual_publish_requires_manual_intervention():
    project = _initial_project()
    for stage, verified in [
        ("audio_pending", False), ("audio_ready", True), ("visuals_pending", False),
        ("visuals_ready", True), ("animation_pending", False), ("animation_ready", True),
        ("render_pending", False), ("rendered", True), ("qc_pending", False), ("qc_passed", True),
        ("ready_for_manual_publish", False),
    ]:
        project = _advance(project, stage, verified=verified)

    plan = build_resume_plan(project)
    assert plan.action == "manual_intervention_required"
    assert plan.next_stage is None


def test_resume_plan_for_completed_is_terminal():
    project = _initial_project()
    for stage, verified, reason in [
        ("audio_pending", False, None), ("audio_ready", True, None),
        ("visuals_pending", False, None), ("visuals_ready", True, None),
        ("animation_pending", False, None), ("animation_ready", True, None),
        ("render_pending", False, None), ("rendered", True, None),
        ("qc_pending", False, None), ("qc_passed", True, None),
        ("ready_for_manual_publish", False, None),
        ("completed", True, "manual completion recorded"),
    ]:
        project = _advance(project, stage, verified=verified, reason=reason)

    plan = build_resume_plan(project)
    assert plan.action == "terminal"
    assert plan.next_stage is None


def test_resume_plan_for_archived_is_terminal():
    project = _initial_project()
    archived, _ = transition_project(project, "archived", now=FIXED_NOW, reason="abandoned")
    plan = build_resume_plan(archived)
    assert plan.action == "terminal"
    assert plan.next_stage is None


# ---------------------------------------------------------------------
# Immutability and internal coherence
# ---------------------------------------------------------------------


def test_project_record_is_frozen():
    project = _initial_project()
    with pytest.raises(ValidationError):
        project.current_stage = "audio_pending"


def test_project_transition_is_frozen():
    project = _initial_project()
    _, transition = transition_project(project, "audio_pending", now=FIXED_NOW)
    with pytest.raises(ValidationError):
        transition.to_stage = "audio_ready"


def test_resume_plan_is_frozen():
    plan = build_resume_plan(_initial_project())
    with pytest.raises(ValidationError):
        plan.action = "terminal"


def test_timestamp_injection_is_deterministic():
    manifest = _make_manifest()
    first = create_initial_project("data/projects/demo/manifest.json", manifest, now=FIXED_NOW)
    second = create_initial_project("data/projects/demo/manifest.json", manifest, now=FIXED_NOW)
    assert first.created_at == second.created_at == FIXED_NOW


@pytest.mark.parametrize(
    "overrides",
    [
        dict(current_stage="planned", failed_stage="audio_pending"),
        dict(current_stage="planned", failure_message="oops"),
        dict(current_stage="failed", failed_stage=None),
        dict(current_stage="planned", archived_at=FIXED_NOW),
        dict(current_stage="archived", archived_at=None),
        dict(current_stage="planned", completed_at=FIXED_NOW),
        dict(current_stage="completed", completed_at=None),
    ],
)
def test_project_record_rejects_incoherent_field_combinations(overrides):
    base = dict(
        project_id="proj-x",
        manifest_path="data/projects/x/manifest.json",
        manifest_fingerprint="f" * 64,
        current_stage="planned",
        last_successful_stage="planned",
        failed_stage=None,
        failure_message=None,
        lifecycle_version=1,
        created_at=FIXED_NOW,
        updated_at=FIXED_NOW,
        completed_at=None,
        archived_at=None,
        retry_count=0,
        execution_status="not_executed",
    )
    base.update(overrides)
    with pytest.raises(ValidationError):
        ProjectRecord(**base)
