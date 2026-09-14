"""Phase 2C: Verified State Transition Service.

The sole approved local path for advancing a project into a
verification-required lifecycle stage. For a requested target stage it:

  1. maps the target stage to its minimum required registered artifacts,
  2. verifies each one using Phase 2B's src/core/artifact_verifier.py
     (never re-implementing checksum/path/scene validation here), and
  3. only if every requirement is met, performs exactly one guarded
     transition via src/core/project_state_machine.py's
     transition_project(verified=True, ...) followed by
     src/database/project_repository.py's save_transition() under the
     project's existing optimistic lock.

Local filesystem and local SQLite only. No LLM, TTS, image generation,
Flow, Veo, renderer, FFmpeg, YouTube, network/API, subprocess, payment, or
provider call anywhere in this module — and it never creates, edits,
moves, deletes, or registers an artifact file, and never writes a manifest
or queue file.

Every ordinary rejection (illegal transition, missing/duplicate artifact,
failed verification, stale lifecycle_version, ...) is reported via the
returned VerifiedTransitionResult with approved=False (or
db_committed=False) — this module does not raise for those. It raises
VerifiedTransitionServiceError only for a caller/input inconsistency it
will not silently work around (a manifest that does not belong to the
given project)."""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from src.core.artifact_verifier import verify_artifact
from src.core.project_state_machine import ProjectStateTransitionError, transition_project
from src.database.project_repository import ProjectConcurrencyError, save_transition
from src.models.artifact import ArtifactRecord, ArtifactVerificationResult
from src.models.manifest import VideoManifest
from src.models.project_state import ProjectRecord, ProjectStage
from src.models.verified_transition import ArtifactRequirement, VerifiedTransitionResult


class VerifiedTransitionServiceError(Exception):
    """Raised only for a project/manifest identity mismatch — a caller
    bug, not a normal verification outcome. Ordinary verification
    failures (missing artifacts, illegal transition, stale
    lifecycle_version, a missing reason, ...) are reported via a returned
    VerifiedTransitionResult instead; see this module's docstring."""


# Every target stage verify_and_advance() accepts, and the minimum
# artifact registration each one requires. "completed" requires no
# additional artifact registration here: reaching it is already gated on
# qc_passed having been verified earlier, and transition_project() itself
# still enforces the explicit reason + verified=True it has always
# required — this service adds no new completion path, it only routes an
# explicit, already-legal completion through the same guarded write path
# as every other verified transition.
_SCENE_LEVEL_TARGETS: dict[str, str] = {
    "audio_ready": "audio",
    "visuals_ready": "visual",
    "animation_ready": "animation",
}
_PROJECT_LEVEL_TARGETS: dict[str, tuple[str, ...]] = {
    "rendered": ("render",),
    "qc_passed": ("render", "qc_report"),
    "completed": (),
}
SUPPORTED_TARGET_STAGES: frozenset[str] = frozenset(_SCENE_LEVEL_TARGETS) | frozenset(_PROJECT_LEVEL_TARGETS)


def _required_artifacts(to_stage: str, manifest: VideoManifest) -> tuple[ArtifactRequirement, ...]:
    scene_kind = _SCENE_LEVEL_TARGETS.get(to_stage)
    if scene_kind is not None:
        return tuple(
            ArtifactRequirement(kind=scene_kind, scope="scene", scene_id=scene.scene_id)
            for scene in manifest.scene_plan.scenes
        )
    return tuple(
        ArtifactRequirement(kind=kind, scope="project", scene_id=None)
        for kind in _PROJECT_LEVEL_TARGETS.get(to_stage, ())
    )


def _matching(
    artifacts: tuple[ArtifactRecord, ...], kind: str, scene_id: str | None
) -> list[ArtifactRecord]:
    return [a for a in artifacts if a.kind == kind and a.scene_id == scene_id]


def verify_and_advance(
    conn: sqlite3.Connection,
    project: ProjectRecord,
    manifest: VideoManifest,
    artifacts: "list[ArtifactRecord] | tuple[ArtifactRecord, ...]",
    to_stage: ProjectStage,
    reason: str,
    *,
    now: datetime | None = None,
) -> VerifiedTransitionResult:
    """Verify every artifact `to_stage` requires against `project`'s own
    manifest, then — only if every one passes — perform exactly one
    guarded transition_project(verified=True) + save_transition() under
    project.lifecycle_version's optimistic lock. Writes nothing to SQLite
    if any requirement or verification fails; save_transition's own
    `with conn:` transaction ensures a failed/stale write leaves no
    partial change."""
    if manifest.project_id != project.project_id:
        raise VerifiedTransitionServiceError(
            f"manifest project_id {manifest.project_id!r} does not match project {project.project_id!r}"
        )
    if manifest.source_fingerprint != project.manifest_fingerprint:
        raise VerifiedTransitionServiceError(
            "manifest fingerprint does not match the project registry's recorded fingerprint"
        )

    lifecycle_before = project.lifecycle_version

    def _rejected(
        reasons: tuple[str, ...],
        required: tuple[ArtifactRequirement, ...] = (),
        results: tuple[ArtifactVerificationResult, ...] = (),
    ) -> VerifiedTransitionResult:
        return VerifiedTransitionResult(
            project_id=project.project_id,
            from_stage=project.current_stage,
            to_stage=to_stage,
            reason=reason,
            approved=False,
            required_artifacts=required,
            artifact_verification_results=results,
            db_committed=False,
            reasons=reasons,
            lifecycle_version_before=lifecycle_before,
            lifecycle_version_after=None,
        )

    if not reason or not reason.strip():
        return _rejected(("an explicit non-empty reason is required for a verified transition",))

    if to_stage not in SUPPORTED_TARGET_STAGES:
        return _rejected(
            (
                f"{to_stage!r} is not a verification-required target stage supported by "
                f"verify-and-advance (expected one of {sorted(SUPPORTED_TARGET_STAGES)})",
            )
        )

    required = _required_artifacts(to_stage, manifest)
    artifacts_tuple = tuple(artifacts)
    project_dir = Path(project.manifest_path).resolve().parent

    reasons: list[str] = []
    results: list[ArtifactVerificationResult] = []
    for req in required:
        matches = _matching(artifacts_tuple, req.kind, req.scene_id)
        scope_label = f"scene {req.scene_id!r}" if req.scope == "scene" else "project level"
        if not matches:
            reasons.append(f"missing required {req.kind!r} artifact for {scope_label}")
            continue
        if len(matches) > 1:
            reasons.append(
                f"duplicate/conflicting {req.kind!r} artifacts registered for {scope_label}: "
                f"{sorted(a.artifact_id for a in matches)}"
            )
            continue
        result = verify_artifact(project_dir, matches[0], manifest)
        results.append(result)
        if not result.passed:
            reasons.extend(f"{matches[0].artifact_id}: {r}" for r in result.reasons)

    if reasons:
        return _rejected(tuple(reasons), required=required, results=tuple(results))

    try:
        updated_record, transition = transition_project(
            project, to_stage, now=now, reason=reason, verified=True
        )
    except ProjectStateTransitionError as exc:
        return _rejected((str(exc),), required=required, results=tuple(results))

    try:
        save_transition(conn, lifecycle_before, updated_record, transition)
    except ProjectConcurrencyError as exc:
        return VerifiedTransitionResult(
            project_id=project.project_id,
            from_stage=project.current_stage,
            to_stage=to_stage,
            reason=reason,
            approved=True,
            required_artifacts=required,
            artifact_verification_results=tuple(results),
            db_committed=False,
            reasons=(str(exc),),
            lifecycle_version_before=lifecycle_before,
            lifecycle_version_after=None,
        )

    return VerifiedTransitionResult(
        project_id=project.project_id,
        from_stage=project.current_stage,
        to_stage=to_stage,
        reason=reason,
        approved=True,
        required_artifacts=required,
        artifact_verification_results=tuple(results),
        db_committed=True,
        reasons=(),
        lifecycle_version_before=lifecycle_before,
        lifecycle_version_after=updated_record.lifecycle_version,
    )
