"""Phase 1D: pure, local, deterministic project lifecycle state machine.

No filesystem writes, no database access, no provider calls — every
function here takes a ProjectRecord (and plain arguments) and returns a
brand-new ProjectRecord plus a ProjectTransition audit record. Persisting
either is src/database/project_repository.py's job, not this module's.

Phase 1D tracks FUTURE production stages (TTS, images, animation/Flow,
render, QC, manual publish) — it never runs any of that work itself.
execution_status on every ProjectRecord this module produces is always
"not_executed".

Design summary (see docs/spec-v4/IMPLEMENTATION-PLAN.md's Phase 1D entry
for the full write-up):

- The canonical forward route is a single linear chain (_FORWARD_TRANSITIONS)
  — no branching, no reordering by a caller.
- "script_approved" is never actually assigned as a stage: Phase 1C's
  ManifestBuilder already requires an approved StoryInput and every scene
  approved before a manifest can exist, so "planned" already implies that
  prerequisite is satisfied, and planned transitions straight to
  "audio_pending" (tested explicitly in tests/test_project_state_machine.py).
- Stages that CLAIM completed work a future verifier hasn't run yet
  (_VERIFICATION_REQUIRED_STAGES) can only be reached with transition_project's
  verified=True — Phase 1D itself never passes verified=True in any
  non-test code path; only a future artifact-verification stage should.
- Only stages representing in-progress/attempted work (_FAILABLE_STAGES) can
  fail; an already-verified "_ready"/"rendered"/"qc_passed"/"completed"
  stage cannot (there is deliberately no "unverify" operation in Phase 1D).
- "archived" is reachable (with an explicit reason) from any non-archived
  state, including "completed" and "failed" — an administrative closure
  action, distinct from the production pipeline's forward/retry rules —
  and has no outgoing transitions at all once reached.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.models.manifest import VideoManifest
from src.models.project_state import ProjectRecord, ProjectStage, ProjectTransition, ResumePlan


class ProjectStateTransitionError(Exception):
    """Raised for any illegal transition attempt: a skip, a regression, a
    transition out of a terminal state, a retry targeting the wrong stage,
    an unverified claim of completed work, or a terminal transition made
    without its required explicit reason. Callers should treat this as
    fatal — a transition is never partially applied."""


# The single allowed forward route. Each key has exactly one valid target —
# a straight line, matching the canonical route in the Phase 1D task spec.
# "script_approved" deliberately has no entry: planned's only forward
# target is "audio_pending".
_FORWARD_TRANSITIONS: dict[str, str] = {
    "planned": "audio_pending",
    "audio_pending": "audio_ready",
    "audio_ready": "visuals_pending",
    "visuals_pending": "visuals_ready",
    "visuals_ready": "animation_pending",
    "animation_pending": "animation_ready",
    "animation_ready": "render_pending",
    "render_pending": "rendered",
    "rendered": "qc_pending",
    "qc_pending": "qc_passed",
    "qc_passed": "ready_for_manual_publish",
    "ready_for_manual_publish": "completed",
}

# The same route, used by build_resume_plan() to find "the next stage
# after this one" for a stage that has already completed.
_STAGE_ORDER: tuple[str, ...] = (
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

# Stages that CLAIM a future artifact/measurement exists (TTS audio +
# measured duration, generated images, completed animation/Flow work, a
# rendered file, a passed QC check) or that the project is fully done.
# transition_project() refuses to enter any of these without verified=True.
# "ready_for_manual_publish" is deliberately NOT in this set: reaching it
# is already gated behind qc_passed having been verified, so it is a
# bookkeeping consequence of a prior verified fact, not a new claim.
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

# Only a stage representing in-progress/attempted work can fail. A
# "_ready"/"rendered"/"qc_passed"/"completed" stage represents an
# already-verified fact — Phase 1D has no "unverify" operation, so those
# stages are deliberately excluded here (disjoint from
# _VERIFICATION_REQUIRED_STAGES by construction).
_FAILABLE_STAGES = frozenset(
    {
        "planned",
        "audio_pending",
        "visuals_pending",
        "animation_pending",
        "render_pending",
        "qc_pending",
        "ready_for_manual_publish",
    }
)


def _now(now: datetime | None) -> datetime:
    return now if now is not None else datetime.now(timezone.utc)


def _replace(record: ProjectRecord, **changes: object) -> ProjectRecord:
    """Build a new, fully-validated ProjectRecord from `record` with
    `changes` applied. Deliberately NOT model_copy(update=...): model_copy
    bypasses ProjectRecord's cross-field coherence validator entirely, and
    these coherence rules (failed_stage only set while failed,
    archived_at only set while archived, ...) are exactly what must hold
    after every transition. Going through the normal constructor means
    pydantic re-checks them on every single transition, not just at
    initial creation."""
    data = {
        "project_id": record.project_id,
        "manifest_path": record.manifest_path,
        "manifest_fingerprint": record.manifest_fingerprint,
        "current_stage": record.current_stage,
        "last_successful_stage": record.last_successful_stage,
        "failed_stage": record.failed_stage,
        "failure_message": record.failure_message,
        "lifecycle_version": record.lifecycle_version,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "completed_at": record.completed_at,
        "archived_at": record.archived_at,
        "retry_count": record.retry_count,
        "execution_status": record.execution_status,
    }
    data.update(changes)
    return ProjectRecord(**data)


def create_initial_project(
    manifest_path: str | Path,
    manifest: VideoManifest,
    *,
    now: datetime | None = None,
) -> ProjectRecord:
    """Create the one and only initial ProjectRecord Phase 1D may create:
    "planned", copying project_id/fingerprint from `manifest`. Pure — does
    not write manifest_path or anything else to disk; the caller is
    expected to have already validated/saved the manifest via
    src/core/manifest_builder.py and src/core/manifest_store.py."""
    when = _now(now)
    return ProjectRecord(
        project_id=manifest.project_id,
        manifest_path=str(manifest_path),
        manifest_fingerprint=manifest.source_fingerprint,
        current_stage="planned",
        last_successful_stage="planned",
        failed_stage=None,
        failure_message=None,
        lifecycle_version=1,
        created_at=when,
        updated_at=when,
        completed_at=None,
        archived_at=None,
        retry_count=0,
        execution_status="not_executed",
    )


def initial_transition_for(project: ProjectRecord) -> ProjectTransition:
    """Build the creation audit record for a freshly created ProjectRecord
    (as returned by create_initial_project). from_stage and to_stage are
    both "planned" — a project is created directly into that state, so
    there is no "before creation" stage to record. This is a convenience
    for callers of src/database/project_repository.py's create_project(),
    which takes an initial_transition as an explicit parameter (per the
    Phase 1D task spec's repository API) rather than synthesizing one
    itself."""
    return ProjectTransition(
        project_id=project.project_id,
        from_stage=project.current_stage,
        to_stage=project.current_stage,
        occurred_at=project.created_at,
        reason="project created from a validated, approved VideoManifest",
        is_retry=False,
        lifecycle_version=project.lifecycle_version,
    )


def transition_project(
    project: ProjectRecord,
    to_stage: ProjectStage,
    *,
    now: datetime | None = None,
    reason: str | None = None,
    verified: bool = False,
) -> tuple[ProjectRecord, ProjectTransition]:
    """Attempt one forward transition, retry, or archive action.

    - Normal forward moves follow _FORWARD_TRANSITIONS exactly (no
      skipping, no regression, no caller-defined routes).
    - If `project.current_stage == "failed"`, the ONLY legal `to_stage` is
      `project.failed_stage` itself (a retry) — anything else raises.
    - `to_stage` in _VERIFICATION_REQUIRED_STAGES requires `verified=True`;
      Phase 1D never passes this itself outside tests simulating a future
      verifier (see this module's docstring).
    - `to_stage == "completed"` additionally requires a non-empty `reason`
      ("manual completion recorded" or similar) — Phase 1D never completes
      a project automatically.
    - `to_stage == "archived"` is allowed from any state except
      "archived" itself (already terminal) and requires a non-empty
      `reason`; it does not require `verified`.

    Raises ProjectStateTransitionError on any illegal request. Never
    partially applies a transition — either both the returned ProjectRecord
    and ProjectTransition reflect the new state, or an exception is raised
    and nothing is returned."""
    current = project.current_stage
    when = _now(now)

    if current == "archived":
        raise ProjectStateTransitionError(
            "project is archived, which is terminal — no further transitions are allowed"
        )

    if to_stage == "archived":
        if not reason:
            raise ProjectStateTransitionError(
                "archiving a project requires an explicit reason"
            )
        updated = _replace(
            project,
            current_stage="archived",
            failed_stage=None,
            failure_message=None,
            lifecycle_version=project.lifecycle_version + 1,
            updated_at=when,
            archived_at=when,
        )
        transition = ProjectTransition(
            project_id=project.project_id,
            from_stage=current,
            to_stage="archived",
            occurred_at=when,
            reason=reason,
            is_retry=False,
            lifecycle_version=updated.lifecycle_version,
        )
        return updated, transition

    if current == "completed":
        raise ProjectStateTransitionError(
            "project is completed and does not transition further in Phase 1D "
            "(archiving it is still allowed — see to_stage='archived')"
        )

    is_retry = False
    if current == "failed":
        if to_stage != project.failed_stage:
            raise ProjectStateTransitionError(
                f"project is 'failed' at {project.failed_stage!r}; only retrying that exact "
                f"stage is permitted, not {to_stage!r}"
            )
        is_retry = True
    else:
        expected_next = _FORWARD_TRANSITIONS.get(current)
        if expected_next is None or to_stage != expected_next:
            raise ProjectStateTransitionError(
                f"{current!r} -> {to_stage!r} is not an allowed transition "
                f"(expected {expected_next!r})"
            )

    if to_stage in _VERIFICATION_REQUIRED_STAGES and not verified:
        raise ProjectStateTransitionError(
            f"transitioning to {to_stage!r} claims completed/measured work and requires "
            "verified=True (a future artifact verifier — Phase 1D never sets this itself "
            "outside of a test simulating one)"
        )

    if to_stage == "completed" and not reason:
        raise ProjectStateTransitionError(
            "transitioning to 'completed' requires an explicit reason (e.g. "
            "'manual completion recorded') — Phase 1D never completes a project automatically"
        )

    updated = _replace(
        project,
        current_stage=to_stage,
        last_successful_stage=to_stage,
        failed_stage=None,
        failure_message=None,
        lifecycle_version=project.lifecycle_version + 1,
        updated_at=when,
        completed_at=when if to_stage == "completed" else None,
        retry_count=project.retry_count + (1 if is_retry else 0),
    )
    transition = ProjectTransition(
        project_id=project.project_id,
        from_stage=current,
        to_stage=to_stage,
        occurred_at=when,
        reason=reason,
        is_retry=is_retry,
        lifecycle_version=updated.lifecycle_version,
    )
    return updated, transition


def mark_project_failed(
    project: ProjectRecord,
    failed_stage: ProjectStage,
    message: str | None,
    *,
    now: datetime | None = None,
) -> tuple[ProjectRecord, ProjectTransition]:
    """Record that `failed_stage` (which must equal `project.current_stage`
    — you can only fail the stage you were actually attempting) failed,
    with an optional owner-facing `message`. Never marks the failed stage
    as successful; `last_successful_stage` is left exactly as it was."""
    if project.current_stage not in _FAILABLE_STAGES:
        raise ProjectStateTransitionError(
            f"cannot fail from {project.current_stage!r} — only an in-progress stage "
            f"({sorted(_FAILABLE_STAGES)}) can fail in Phase 1D"
        )
    if failed_stage != project.current_stage:
        raise ProjectStateTransitionError(
            f"failed_stage {failed_stage!r} must match the project's current stage "
            f"{project.current_stage!r}"
        )

    when = _now(now)
    updated = _replace(
        project,
        current_stage="failed",
        failed_stage=failed_stage,
        failure_message=message,
        lifecycle_version=project.lifecycle_version + 1,
        updated_at=when,
    )
    transition = ProjectTransition(
        project_id=project.project_id,
        from_stage=project.current_stage,
        to_stage="failed",
        occurred_at=when,
        reason=message,
        is_retry=False,
        lifecycle_version=updated.lifecycle_version,
    )
    return updated, transition


def _next_stage_in_order(stage: str) -> ProjectStage:
    try:
        idx = _STAGE_ORDER.index(stage)
    except ValueError as exc:
        raise ProjectStateTransitionError(
            f"{stage!r} has no defined position in the canonical stage order"
        ) from exc
    if idx + 1 >= len(_STAGE_ORDER):
        raise ProjectStateTransitionError(f"{stage!r} has no next stage — it is already the end")
    return _STAGE_ORDER[idx + 1]  # type: ignore[return-value]


def build_resume_plan(project: ProjectRecord) -> ResumePlan:
    """Report what a future orchestrator should do next for `project`.
    Pure inspection only — never mutates `project`, never touches storage,
    never executes a stage."""
    stage = project.current_stage

    if stage == "archived":
        return ResumePlan(
            project_id=project.project_id,
            action="terminal",
            next_stage=None,
            explanation="Project is archived (terminal) and cannot resume.",
            last_successful_stage=project.last_successful_stage,
        )

    if stage == "completed":
        return ResumePlan(
            project_id=project.project_id,
            action="terminal",
            next_stage=None,
            explanation="Project is completed; there is nothing further to do.",
            last_successful_stage=project.last_successful_stage,
        )

    if stage == "failed":
        return ResumePlan(
            project_id=project.project_id,
            action="retry_stage",
            next_stage=project.failed_stage,
            explanation=f"Project failed while attempting {project.failed_stage!r}; retry that stage.",
            last_successful_stage=project.last_successful_stage,
        )

    if stage == "ready_for_manual_publish":
        return ResumePlan(
            project_id=project.project_id,
            action="manual_intervention_required",
            next_stage=None,
            explanation=(
                "Final output is ready; publishing is a manual, out-of-band action "
                "Phase 1D never automates."
            ),
            last_successful_stage=project.last_successful_stage,
        )

    if stage.endswith("_pending"):
        return ResumePlan(
            project_id=project.project_id,
            action="start_stage",
            next_stage=stage,
            explanation=f"{stage!r} has not completed yet; (re)start it.",
            last_successful_stage=project.last_successful_stage,
        )

    # "planned", or a "_ready"/"rendered"/"qc_passed" stage: that stage is
    # done, so the next stage in the canonical order is what starts next.
    next_stage = _next_stage_in_order(stage)
    return ResumePlan(
        project_id=project.project_id,
        action="start_stage",
        next_stage=next_stage,
        explanation=f"{stage!r} is complete; start {next_stage!r} next.",
        last_successful_stage=project.last_successful_stage,
    )
