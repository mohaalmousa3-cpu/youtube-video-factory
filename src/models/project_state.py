"""Phase 1D: typed data models for one video project's local production
lifecycle — how far a project has gotten from a planned VideoManifest
toward a finished, manually-published video.

This module is pure data (frozen, strict Pydantic v2 models), same style
as src/models/manifest.py. The actual transition RULES (what stage may
follow what, retry/failure semantics, the verification gate) live in
src/core/project_state_machine.py; persistence lives in
src/database/project_repository.py. Nothing here calls a provider, writes
to disk, or touches SQLite.

`ProjectStage` is intentionally NOT the same type as
src/models/enums.py's `ManifestLifecycleState` (that one is VideoManifest's
own coarse planned/in_production/complete/archived field, unchanged since
Phase 1C) — a ProjectStage is the much more granular real-production
lifecycle Phase 1D introduces on top of a manifest that is already
"planned"."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from src.models.common import FrozenStrictModel
from src.models.enums import ExecutionStatus

# The full finite state set, in the order Required Stages were listed in
# the Phase 1D task spec. "script_approved" is part of this type for
# completeness/documentation but is never actually assigned as a stage by
# src/core/project_state_machine.py — see that module's docstring for why:
# Phase 1C's ManifestBuilder already requires an approved StoryInput and
# every scene approved before a manifest can exist at all, so by the time
# a project reaches "planned" that prerequisite is already satisfied, and
# planned transitions straight to "audio_pending".
ProjectStage = Literal[
    "planned",
    "script_approved",
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
    "failed",
    "archived",
]

# What build_resume_plan() recommends a future orchestrator do next.
# "no_op" is part of the type for API completeness (the task spec lists it
# as one of five possible actions) but build_resume_plan() never returns it
# for any ProjectStage Phase 1D defines — every reachable stage maps to one
# of the other four actions. Kept for forward compatibility rather than
# invented a scenario just to exercise it.
ResumeAction = Literal[
    "no_op",
    "start_stage",
    "retry_stage",
    "manual_intervention_required",
    "terminal",
]


class ProjectRecord(FrozenStrictModel):
    """One video project's current lifecycle state. Immutable — every
    transition in project_state_machine.py returns a brand-new
    ProjectRecord rather than mutating this one; the project_repository
    module is what actually persists an updated record over the old one in
    SQLite, under an optimistic lock."""

    project_id: str = Field(min_length=1)
    manifest_path: str = Field(min_length=1)
    manifest_fingerprint: str = Field(min_length=1)

    current_stage: ProjectStage
    # Reaching "planned" already IS a successful checkpoint (a valid,
    # approved manifest exists), so a freshly created project's
    # last_successful_stage is "planned", not some placeholder/None — this
    # keeps the field non-optional and avoids a separate "no stage yet"
    # sentinel meaning to thread through every consumer.
    last_successful_stage: ProjectStage

    failed_stage: ProjectStage | None = None
    # Owner-facing only — may be Arabic or English (see project policy);
    # never read by the render pipeline, never viewer-facing content, and
    # explicitly optional even while current_stage == "failed" (a failure
    # can be recorded without a human-readable explanation yet).
    failure_message: str | None = None

    # Optimistic-lock / audit counter: starts at 1, incremented by exactly
    # 1 on every transition. src/database/project_repository.py's
    # save_transition() requires the caller's expected value to still
    # match the stored row before writing.
    lifecycle_version: int = Field(ge=1)

    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    archived_at: datetime | None = None

    retry_count: int = Field(ge=0, default=0)

    # Always "not_executed" in Phase 1D — this layer only tracks stage
    # bookkeeping, it never calls a provider (see
    # src/core/project_state_machine.py's module docstring).
    execution_status: ExecutionStatus

    @model_validator(mode="after")
    def _fields_are_internally_coherent(self) -> "ProjectRecord":
        if self.current_stage == "failed":
            if self.failed_stage is None:
                raise ValueError("failed_stage must be set when current_stage is 'failed'")
        else:
            if self.failed_stage is not None:
                raise ValueError("failed_stage must be None unless current_stage is 'failed'")
            if self.failure_message is not None:
                raise ValueError("failure_message must be None unless current_stage is 'failed'")

        if self.current_stage == "archived":
            if self.archived_at is None:
                raise ValueError("archived_at must be set when current_stage is 'archived'")
        elif self.archived_at is not None:
            raise ValueError("archived_at must be None unless current_stage is 'archived'")

        if self.current_stage == "completed":
            if self.completed_at is None:
                raise ValueError("completed_at must be set when current_stage is 'completed'")
        elif self.current_stage == "archived":
            # completed_at is a historical timestamp here: None if archived
            # before ever completing, or the original completion time if
            # archived afterward — either is valid, it is never cleared.
            pass
        elif self.completed_at is not None:
            raise ValueError(
                "completed_at must be None unless current_stage is 'completed' or 'archived'"
            )

        return self


class ProjectTransition(FrozenStrictModel):
    """An append-only audit record of one stage change. Owner-facing audit
    text only — never secrets, never viewer-facing script content.

    Deterministic ordering key: (project_id, lifecycle_version) — the
    lifecycle_version this transition produced on the ProjectRecord side.
    The SQLite repository additionally assigns its own autoincrement row
    id for storage purposes, but that is a storage detail, not part of
    this pure model."""

    project_id: str = Field(min_length=1)
    from_stage: ProjectStage
    to_stage: ProjectStage
    occurred_at: datetime
    reason: str | None = None
    is_retry: bool = False
    lifecycle_version: int = Field(ge=1)


class ResumePlan(FrozenStrictModel):
    """A pure, read-only recommendation of what a future orchestrator
    would do next for a project. build_resume_plan() never executes
    anything and never mutates a ProjectRecord — it only inspects one and
    reports what SHOULD happen next."""

    project_id: str = Field(min_length=1)
    action: ResumeAction
    next_stage: ProjectStage | None = None
    explanation: str = Field(min_length=1)
    last_successful_stage: ProjectStage
