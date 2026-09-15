"""Phase 2I: Non-Verification Stage Advance Service.

The sole approved local path for moving a project exactly one step along
its OWN forward chain into a stage that claims no completed/measured work
— the six "bookkeeping" stages src/core/project_state_machine.py's forward
chain reaches WITHOUT verified=True:

    planned          -> audio_pending
    audio_ready      -> visuals_pending
    visuals_ready    -> animation_pending
    animation_ready  -> render_pending
    rendered         -> qc_pending
    qc_passed        -> ready_for_manual_publish

Every other forward target (audio_ready, visuals_ready, animation_ready,
rendered, qc_passed, completed) CLAIMS a future artifact/measurement
exists or that the project is fully done — those are
src/core/verified_transition_service.py's verify_and_advance() territory
exclusively, gated on real artifact verification (and, for "completed",
an explicit --reason). This module never accepts any of those six as a
target: NON_VERIFICATION_TARGET_STAGES below is the complete, exhaustive
allow-list, checked before src/core/project_state_machine.py's
transition_project() is ever called, and this module always calls it with
verified=False, hard-coded — never accepting a verified argument from a
caller. "archived" is out of scope here (different semantics —
administrative closure — no command owns it yet); attempting it is
rejected by the same allow-list check. A currently-"failed" project is
rejected by its own explicit check below: transition_project() would
otherwise legally accept a RETRY back into project.failed_stage whenever
that happens to be one of NON_VERIFICATION_TARGET_STAGES, but retrying is
a distinct, not-yet-built capability this command deliberately excludes.

Reuses, never re-implements: transition_project() for the single legal
next-stage computation itself (a skip, a regression, or acting from a
terminal (archived/completed) current_stage is caught via its own
ProjectStateTransitionError, not duplicated here) and save_transition()
for the one guarded UPDATE + INSERT under project.lifecycle_version's
optimistic lock. Local filesystem and local SQLite only — no artifact
file, manifest, or queue file is ever read or written; no provider, no
network, no subprocess.

Every ordinary rejection (a target outside NON_VERIFICATION_TARGET_STAGES,
an illegal/skip/regression transition, a stale lifecycle_version) is
reported via the returned StageAdvanceResult with approved=False (or
db_committed=False) — this module does not raise for those."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime

from src.core.project_state_machine import ProjectStateTransitionError, transition_project
from src.database.project_repository import ProjectConcurrencyError, save_transition
from src.models.project_state import ProjectRecord, ProjectStage

# The complete, exhaustive set of stages this service will ever enter —
# exactly the forward-chain targets NOT in
# project_state_machine._VERIFICATION_REQUIRED_STAGES. Deliberately
# re-listed here (not imported) as a hard-coded allow-list: this set is
# this module's own contract, independent of whatever
# project_state_machine.py's private set happens to contain internally.
NON_VERIFICATION_TARGET_STAGES: frozenset[str] = frozenset(
    {
        "audio_pending",
        "visuals_pending",
        "animation_pending",
        "render_pending",
        "qc_pending",
        "ready_for_manual_publish",
    }
)


@dataclass(frozen=True)
class StageAdvanceResult:
    """The one typed result advance_project_stage() always returns,
    success or failure. Never itself mutates a manifest, an artifact, or
    anything beyond the project's own row + transition audit trail."""

    project_id: str
    from_stage: ProjectStage
    to_stage: ProjectStage
    reason: str | None
    approved: bool
    db_committed: bool
    reasons: tuple[str, ...]
    lifecycle_version_before: int
    lifecycle_version_after: int | None


def advance_project_stage(
    conn: sqlite3.Connection,
    project: ProjectRecord,
    to_stage: ProjectStage,
    *,
    reason: str | None = None,
    now: datetime | None = None,
) -> StageAdvanceResult:
    """Attempt exactly one non-verification forward move for `project`,
    into `to_stage`. Writes nothing to SQLite unless `to_stage` is in
    NON_VERIFICATION_TARGET_STAGES AND transition_project() accepts it as
    project's own single legal next stage — in every other case this
    returns a rejected result before save_transition() is ever reached."""
    lifecycle_before = project.lifecycle_version

    def _rejected(reasons: tuple[str, ...]) -> StageAdvanceResult:
        return StageAdvanceResult(
            project_id=project.project_id,
            from_stage=project.current_stage,
            to_stage=to_stage,
            reason=reason,
            approved=False,
            db_committed=False,
            reasons=reasons,
            lifecycle_version_before=lifecycle_before,
            lifecycle_version_after=None,
        )

    if to_stage not in NON_VERIFICATION_TARGET_STAGES:
        return _rejected(
            (
                f"{to_stage!r} is not a non-verification target stage supported by "
                f"advance-project-stage (expected one of {sorted(NON_VERIFICATION_TARGET_STAGES)}); "
                "a stage that claims completed/measured work — including 'completed' itself — "
                "must go through verify-and-advance instead",
            )
        )

    # A "failed" project can legally retry back into its own failed_stage
    # via transition_project() (its is_retry branch) — but retrying is a
    # distinct future capability this command deliberately does not offer;
    # blocked explicitly here rather than left to fall through, since
    # transition_project() itself would otherwise accept a retry whenever
    # failed_stage happens to be one of NON_VERIFICATION_TARGET_STAGES.
    if project.current_stage == "failed":
        return _rejected(
            (
                f"project is currently 'failed' at {project.failed_stage!r}; "
                "advance-project-stage does not perform retries — that is a distinct, "
                "not-yet-built capability, not this command's scope",
            )
        )

    try:
        updated_record, transition = transition_project(
            project, to_stage, now=now, reason=reason, verified=False
        )
    except ProjectStateTransitionError as exc:
        return _rejected((str(exc),))

    try:
        save_transition(conn, lifecycle_before, updated_record, transition)
    except ProjectConcurrencyError as exc:
        return StageAdvanceResult(
            project_id=project.project_id,
            from_stage=project.current_stage,
            to_stage=to_stage,
            reason=reason,
            approved=True,
            db_committed=False,
            reasons=(str(exc),),
            lifecycle_version_before=lifecycle_before,
            lifecycle_version_after=None,
        )

    return StageAdvanceResult(
        project_id=project.project_id,
        from_stage=project.current_stage,
        to_stage=to_stage,
        reason=reason,
        approved=True,
        db_committed=True,
        reasons=(),
        lifecycle_version_before=lifecycle_before,
        lifecycle_version_after=updated_record.lifecycle_version,
    )
