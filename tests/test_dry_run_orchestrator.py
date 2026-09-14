from __future__ import annotations

from datetime import datetime, timezone

from src.core.dry_run_orchestrator import DryRunOrchestratorError, build_dry_run_report
from src.core.manifest_builder import build_video_manifest
from src.core.project_state_machine import (
    create_initial_project,
    initial_transition_for,
    mark_project_failed,
    transition_project,
)
from src.utils.channel_config import get_channel_policy

FIXED_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
LATER_NOW = datetime(2026, 1, 1, 12, 5, 0, tzinfo=timezone.utc)


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


def _scene_plan_dict(*, manual_flow: bool = False) -> dict:
    motion_mode = "manual_flow" if manual_flow else "in"
    local_fallback = "pan_lr" if manual_flow else None
    return dict(
        scenes=(
            dict(
                scene_id="scene-01",
                sequence=1,
                narration_text="Why does being left out sting so much?",
                scene_type="establishing",
                narrative_beat="hook",
                visual_brief="Stickman character alone on a quiet street corner at dusk.",
                motion_mode=motion_mode,
                local_fallback_motion_mode=local_fallback,
                approval_state="approved",
            ),
        ),
        role_outfits=(),
    )


def _new_project(*, manual_flow: bool = False):
    manifest = build_video_manifest(
        _story_input_dict(),
        _scene_plan_dict(manual_flow=manual_flow),
        get_channel_policy(),
        created_at=FIXED_NOW,
    )
    project = create_initial_project(f"data/projects/{manifest.project_id}/manifest.json", manifest, now=FIXED_NOW)
    transitions = [initial_transition_for(project)]
    return project, transitions, manifest


def test_dry_run_for_planned_project_reports_next_stage_and_read_only():
    project, transitions, manifest = _new_project()

    report = build_dry_run_report(project, transitions, manifest, get_channel_policy())

    assert report.read_only is True
    assert report.execution_plan.lifecycle_status == "active"
    assert report.execution_plan.next_action == "start_stage"
    assert report.execution_plan.next_stage == "audio_pending"
    assert report.execution_plan.planned_steps[0].stage == "audio_pending"
    assert "Checkpoint stages" in report.execution_plan.requirements[0]


def test_dry_run_for_in_progress_project_keeps_current_pending_stage():
    project, transitions, manifest = _new_project()
    updated, transition = transition_project(project, "audio_pending", now=LATER_NOW)
    transitions.append(transition)

    report = build_dry_run_report(updated, transitions, manifest, get_channel_policy())

    assert report.current_stage == "audio_pending"
    assert report.execution_plan.next_stage == "audio_pending"
    assert report.execution_plan.planned_steps[0].stage == "audio_pending"


def test_dry_run_for_failed_project_reports_retry_guidance():
    project, transitions, manifest = _new_project()
    updated, transition = transition_project(project, "audio_pending", now=LATER_NOW)
    transitions.append(transition)
    failed, failed_transition = mark_project_failed(updated, "audio_pending", "kokoro timeout", now=LATER_NOW)
    transitions.append(failed_transition)

    report = build_dry_run_report(failed, transitions, manifest, get_channel_policy())

    assert report.execution_plan.lifecycle_status == "failed"
    assert report.execution_plan.next_action == "retry_stage"
    assert report.execution_plan.next_stage == "audio_pending"
    assert any("failed" in block.lower() for block in report.execution_plan.blocks)


def test_dry_run_for_completed_and_archived_projects_reports_terminal():
    project, transitions, manifest = _new_project()
    audio_pending, t1 = transition_project(project, "audio_pending", now=LATER_NOW)
    audio_ready, t2 = transition_project(audio_pending, "audio_ready", now=LATER_NOW, verified=True)
    visuals_pending, t3 = transition_project(audio_ready, "visuals_pending", now=LATER_NOW)
    visuals_ready, t4 = transition_project(visuals_pending, "visuals_ready", now=LATER_NOW, verified=True)
    animation_pending, t5 = transition_project(visuals_ready, "animation_pending", now=LATER_NOW)
    animation_ready, t6 = transition_project(animation_pending, "animation_ready", now=LATER_NOW, verified=True)
    render_pending, t7 = transition_project(animation_ready, "render_pending", now=LATER_NOW)
    rendered, t8 = transition_project(render_pending, "rendered", now=LATER_NOW, verified=True)
    qc_pending, t9 = transition_project(rendered, "qc_pending", now=LATER_NOW)
    qc_passed, t10 = transition_project(qc_pending, "qc_passed", now=LATER_NOW, verified=True)
    ready, t11 = transition_project(qc_passed, "ready_for_manual_publish", now=LATER_NOW)
    completed, t12 = transition_project(
        ready, "completed", now=LATER_NOW, verified=True, reason="manual completion recorded"
    )
    transitions.extend([t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12])

    completed_report = build_dry_run_report(completed, transitions, manifest, get_channel_policy())
    assert completed_report.execution_plan.lifecycle_status == "completed"
    assert completed_report.execution_plan.next_action == "terminal"
    assert completed_report.execution_plan.next_stage is None

    archived, archived_transition = transition_project(completed, "archived", now=LATER_NOW, reason="cleanup")
    archived_report = build_dry_run_report(
        archived, transitions + [archived_transition], manifest, get_channel_policy()
    )
    assert archived_report.execution_plan.lifecycle_status == "archived"
    assert archived_report.execution_plan.next_action == "terminal"
    assert archived_report.execution_plan.next_stage is None


def test_dry_run_reports_policy_constraints_and_manual_flow_local_fallback_requirement():
    project, transitions, manifest = _new_project(manual_flow=True)

    report = build_dry_run_report(project, transitions, manifest, get_channel_policy())

    decisions = {d.key: d for d in report.policy_decisions}
    assert decisions["paid_services"].status == "enforced"
    assert "disabled" in decisions["automatic_payment"].detail.lower()
    assert "disabled" in decisions["veo_api"].detail.lower()
    assert "disabled" in decisions["background_music"].detail.lower()
    assert decisions["manual_flow_local_fallback"].status == "required"
    assert any("manual_flow" in requirement for requirement in report.execution_plan.requirements)


def test_dry_run_report_is_deterministic_when_transitions_are_unsorted():
    project, transitions, manifest = _new_project()
    step1, t1 = transition_project(project, "audio_pending", now=FIXED_NOW)
    step2, t2 = transition_project(step1, "audio_ready", now=LATER_NOW, verified=True)

    report_a = build_dry_run_report(step2, [t2, transitions[0], t1], manifest, get_channel_policy())
    report_b = build_dry_run_report(step2, [transitions[0], t1, t2], manifest, get_channel_policy())

    assert report_a.model_dump(mode="json") == report_b.model_dump(mode="json")


def test_dry_run_rejects_manifest_mismatch():
    project, transitions, manifest = _new_project()
    other_project, _, _ = _new_project(manual_flow=True)

    bad_project = project.__class__(
        **{**project.model_dump(), "manifest_fingerprint": other_project.manifest_fingerprint}
    )

    try:
        build_dry_run_report(bad_project, transitions, manifest, get_channel_policy())
    except DryRunOrchestratorError as exc:
        assert "fingerprint" in str(exc)
    else:
        raise AssertionError("expected DryRunOrchestratorError")
