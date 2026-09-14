"""Phase 2A: local, read-only dry-run orchestration.

Consumes existing ProjectRecord + transition history + validated
VideoManifest + policy data and emits a deterministic execution plan.
Never executes stages, writes storage, or calls providers/renderers.
"""
from __future__ import annotations

from src.core.project_state_machine import build_resume_plan
from src.models.manifest import VideoManifest
from src.models.orchestration import DryRunReport, ExecutionPlan, PlannedStep, PolicyDecision
from src.models.project_state import ProjectRecord, ProjectTransition, ProjectStage
from src.utils.channel_config import ChannelPolicy


class DryRunOrchestratorError(Exception):
    """Raised when an execution plan cannot be derived from inconsistent input."""


_STAGE_ROUTE: tuple[ProjectStage, ...] = (
    "planned",
    "audio_pending",
    "audio_ready",
    "visuals_pending",
    "visuals_ready",
    "animation_pending",
    "animation_ready",
    "render_pending",
    "rendered",
    "qc_pending",
    "qc_passed",
    "ready_for_manual_publish",
    "completed",
)

_VERIFICATION_REQUIRED_STAGES = frozenset(
    {
        "audio_ready",
        "visuals_ready",
        "animation_ready",
        "rendered",
        "qc_passed",
        "completed",
    }
)

_NON_INVOKED_INTEGRATIONS: tuple[str, ...] = (
    "Groq LLM",
    "TokenRouter LLM fallback",
    "Kokoro TTS",
    "Qwen Image",
    "Google Flow",
    "Veo",
    "Rhubarb Lip Sync",
    "Real-ESRGAN",
    "Manim renderer",
    "FFmpeg renderer",
    "YouTube publish",
    "Payment/billing",
    "Network/external APIs",
)


def _stage_index(stage: ProjectStage) -> int:
    try:
        return _STAGE_ROUTE.index(stage)
    except ValueError as exc:
        raise DryRunOrchestratorError(f"stage {stage!r} is not on the canonical execution route") from exc


def _planned_steps(start_stage: ProjectStage) -> tuple[PlannedStep, ...]:
    start_idx = _stage_index(start_stage)
    steps = []
    for i, stage in enumerate(_STAGE_ROUTE[start_idx:], start=1):
        steps.append(
            PlannedStep(
                order=i,
                stage=stage,
                action="start_stage",
                verification_required=stage in _VERIFICATION_REQUIRED_STAGES,
                explanation=(
                    f"{stage!r} requires explicit verification before lifecycle advancement."
                    if stage in _VERIFICATION_REQUIRED_STAGES
                    else f"{stage!r} is a planned execution checkpoint and is not run in dry-run mode."
                ),
            )
        )
    return tuple(steps)


def _lifecycle_status(project: ProjectRecord) -> str:
    if project.current_stage == "failed":
        return "failed"
    if project.current_stage == "completed":
        return "completed"
    if project.current_stage == "archived":
        return "archived"
    return "active"


def build_dry_run_report(
    project: ProjectRecord,
    transitions: list[ProjectTransition],
    manifest: VideoManifest,
    channel_policy: ChannelPolicy,
) -> DryRunReport:
    """Build a deterministic, read-only execution plan from existing state."""
    if manifest.project_id != project.project_id:
        raise DryRunOrchestratorError(
            f"manifest project_id {manifest.project_id!r} does not match project {project.project_id!r}"
        )

    if manifest.source_fingerprint != project.manifest_fingerprint:
        raise DryRunOrchestratorError(
            "manifest fingerprint does not match project registry fingerprint"
        )

    for transition in transitions:
        if transition.project_id != project.project_id:
            raise DryRunOrchestratorError("transition history contains a different project_id")

    resume_plan = build_resume_plan(project)

    blocks: list[str] = []
    if project.current_stage == "failed":
        blocks.append(
            f"Project is failed at {project.failed_stage!r}; resolve failure and retry that stage."
        )
    elif project.current_stage == "archived":
        blocks.append("Project is archived (terminal); no execution resume is allowed.")
    elif project.current_stage == "completed":
        blocks.append("Project is completed; no further execution stages are available.")

    warnings = [
        "Read-only dry-run only: no assets/providers/rendering/publishing/payment actions are invoked.",
        "This report does not assert that any artifact was produced or independently verified.",
    ]

    requirements = [
        "Checkpoint stages (audio_ready, visuals_ready, animation_ready, rendered, qc_passed, completed) require explicit verification before transition.",
    ]

    manual_flow_scene_ids = tuple(
        scene.scene_id for scene in manifest.scene_plan.scenes if scene.motion_mode == "manual_flow"
    )
    if channel_policy.motion.require_local_fallback_for_manual_flow and manual_flow_scene_ids:
        requirements.append(
            "Local fallback motion is required for manual_flow scenes: " + ", ".join(manual_flow_scene_ids)
        )

    next_stage = resume_plan.next_stage
    steps = _planned_steps(next_stage) if next_stage is not None else ()

    policy_decisions = (
        PolicyDecision(
            key="paid_services",
            status="enforced" if not channel_policy.budget.paid_services_enabled else "required",
            detail=(
                "Paid services are disabled by channel policy."
                if not channel_policy.budget.paid_services_enabled
                else "Paid services require explicit user approval."
            ),
        ),
        PolicyDecision(
            key="automatic_payment",
            status="enforced" if not channel_policy.budget.automatic_payment_allowed else "required",
            detail=(
                "Automatic payment is disabled by policy."
                if not channel_policy.budget.automatic_payment_allowed
                else "Automatic payment is enabled and requires governance review."
            ),
        ),
        PolicyDecision(
            key="veo_api",
            status="enforced" if not channel_policy.motion.veo_api_enabled else "required",
            detail=(
                "Veo is disabled by policy."
                if not channel_policy.motion.veo_api_enabled
                else "Veo is enabled by policy configuration."
            ),
        ),
        PolicyDecision(
            key="background_music",
            status="enforced" if not channel_policy.audio.background_music_enabled else "required",
            detail=(
                "Background music is disabled by policy."
                if not channel_policy.audio.background_music_enabled
                else "Background music is enabled by policy configuration."
            ),
        ),
        PolicyDecision(
            key="manual_flow_local_fallback",
            status=(
                "required"
                if channel_policy.motion.require_local_fallback_for_manual_flow and manual_flow_scene_ids
                else "not_applicable"
            ),
            detail=(
                "Each manual_flow scene must keep a local fallback motion mode."
                if channel_policy.motion.require_local_fallback_for_manual_flow and manual_flow_scene_ids
                else "No manual_flow scene requires fallback handling in this manifest."
            ),
        ),
    )

    sorted_transitions = sorted(
        transitions,
        key=lambda t: (t.lifecycle_version, t.occurred_at.isoformat(), t.from_stage, t.to_stage),
    )

    return DryRunReport(
        project_id=project.project_id,
        manifest_path=project.manifest_path,
        current_stage=project.current_stage,
        last_successful_stage=project.last_successful_stage,
        lifecycle_version=project.lifecycle_version,
        transition_count=len(sorted_transitions),
        manifest_matches_project_record=True,
        execution_plan=ExecutionPlan(
            lifecycle_status=_lifecycle_status(project),
            current_stage=project.current_stage,
            next_action=resume_plan.action,
            next_stage=resume_plan.next_stage,
            explanation=resume_plan.explanation,
            planned_steps=steps,
            requirements=tuple(requirements),
            blocks=tuple(blocks),
            warnings=tuple(warnings),
        ),
        policy_decisions=policy_decisions,
        non_invoked_integrations=_NON_INVOKED_INTEGRATIONS,
    )
