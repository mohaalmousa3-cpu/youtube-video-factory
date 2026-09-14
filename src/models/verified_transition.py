"""Phase 2C: the typed outcome of the Verified State Transition Service.

Pure data (frozen, strict Pydantic v2), same convention as
src/models/orchestration.py and src/models/artifact.py. Nothing here reads
a file, writes to disk, or touches SQLite — see
src/core/verified_transition_service.py for the logic that populates one
of these. There is no separate "request" model: the service function's own
parameters (project_id via `project`, `to_stage`, `reason`) already ARE the
request, and VerifiedTransitionResult echoes them back alongside the
outcome so one value fully reports what was asked AND what happened."""
from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from src.models.artifact import ArtifactVerificationResult
from src.models.common import FrozenStrictModel
from src.models.enums import ArtifactKind
from src.models.project_state import ProjectStage


class ArtifactRequirement(FrozenStrictModel):
    """One minimum-artifact requirement a verification-required target
    stage imposes: exactly one verified artifact of `kind`, either at
    project scope or scoped to one `scene_id`."""

    kind: ArtifactKind
    scope: Literal["project", "scene"]
    scene_id: str | None = None

    @model_validator(mode="after")
    def _scene_id_matches_scope(self) -> "ArtifactRequirement":
        if self.scope == "scene" and self.scene_id is None:
            raise ValueError("scene_id is required when scope is 'scene'")
        if self.scope == "project" and self.scene_id is not None:
            raise ValueError("scene_id must be None when scope is 'project'")
        return self


class VerifiedTransitionResult(FrozenStrictModel):
    """The one typed result src/core/verified_transition_service.py's
    verify_and_advance() always returns, success or failure. Never itself
    mutates a file, the artifact registry, or a ProjectRecord — it only
    reports what verify_and_advance() already did (or refused to do)."""

    project_id: str = Field(min_length=1)
    from_stage: ProjectStage
    to_stage: ProjectStage
    reason: str

    approved: bool
    required_artifacts: tuple[ArtifactRequirement, ...]
    artifact_verification_results: tuple[ArtifactVerificationResult, ...]
    db_committed: bool
    reasons: tuple[str, ...] = ()

    lifecycle_version_before: int = Field(ge=1)
    lifecycle_version_after: int | None = None

    @model_validator(mode="after")
    def _fields_are_internally_coherent(self) -> "VerifiedTransitionResult":
        if not self.approved:
            if self.db_committed:
                raise ValueError("db_committed cannot be True when approved is False")
            if not self.reasons:
                raise ValueError("reasons must be non-empty when approved is False")
            if self.lifecycle_version_after is not None:
                raise ValueError("lifecycle_version_after must be None when approved is False")
            return self

        if self.db_committed:
            if self.reasons:
                raise ValueError("reasons must be empty when db_committed is True")
            if self.lifecycle_version_after != self.lifecycle_version_before + 1:
                raise ValueError(
                    "lifecycle_version_after must equal lifecycle_version_before + 1 "
                    "when db_committed is True"
                )
        else:
            if self.lifecycle_version_after is not None:
                raise ValueError("lifecycle_version_after must be None unless db_committed is True")
            if not self.reasons:
                raise ValueError(
                    "reasons must explain why an approved transition was not committed "
                    "(e.g. a stale lifecycle_version)"
                )
        return self
