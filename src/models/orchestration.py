"""Phase 2A dry-run orchestration models.

Read-only execution planning/reporting only: these models describe what
would happen next for an existing project without running any stage,
writing any file, mutating SQLite, or calling external integrations.
"""
from __future__ import annotations

from typing import Literal

from pydantic import Field

from src.models.common import FrozenStrictModel
from src.models.project_state import ProjectStage, ResumeAction


class PlannedStep(FrozenStrictModel):
    order: int = Field(ge=1)
    stage: ProjectStage
    action: ResumeAction
    verification_required: bool
    explanation: str = Field(min_length=1)


class PolicyDecision(FrozenStrictModel):
    key: str = Field(min_length=1)
    status: Literal["enforced", "required", "not_applicable"]
    detail: str = Field(min_length=1)


class ExecutionPlan(FrozenStrictModel):
    mode: Literal["dry_run_read_only"] = "dry_run_read_only"
    lifecycle_status: Literal["active", "failed", "completed", "archived"]
    current_stage: ProjectStage
    next_action: ResumeAction
    next_stage: ProjectStage | None = None
    explanation: str = Field(min_length=1)
    planned_steps: tuple[PlannedStep, ...] = ()
    requirements: tuple[str, ...] = ()
    blocks: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class DryRunReport(FrozenStrictModel):
    read_only: Literal[True] = True
    project_id: str = Field(min_length=1)
    manifest_path: str = Field(min_length=1)
    current_stage: ProjectStage
    last_successful_stage: ProjectStage
    lifecycle_version: int = Field(ge=1)
    transition_count: int = Field(ge=0)
    manifest_matches_project_record: bool
    execution_plan: ExecutionPlan
    policy_decisions: tuple[PolicyDecision, ...] = ()
    non_invoked_integrations: tuple[str, ...] = ()
