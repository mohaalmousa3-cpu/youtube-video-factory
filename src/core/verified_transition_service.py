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
given project).

Phase 2H addition — a targeted semantic gate, `to_stage == "qc_passed"`
only: structural verification (existence/size/checksum/path-safety, via
artifact_verifier.py, completely unchanged) proves a qc_report artifact's
*file* matches what was registered; it says nothing about the report's
content. A registered qc_report may legitimately say {"passed": false} —
Phase 2H's registrar retains that as real audit data — but must never by
itself be enough to reach qc_passed. So, only after every structural
requirement (both "render" and "qc_report") has already passed, this
module additionally reads the verified qc_report file (never
ArtifactRecord.metadata, which is mutable DB state, not the source of
truth) and requires its JSON to be an object with `passed` strictly
`True`. Every other target stage (audio_ready, visuals_ready,
animation_ready, rendered, completed) is completely unaffected — this
check is gated strictly on to_stage == "qc_passed"."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from src.core.artifact_verifier import verify_artifact
from src.core.path_safety import resolve_under_project_dir
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


def _qc_report_passed(path: Path | None) -> str | None:
    """Read and parse the VERIFIED qc_report file at `path` (already
    proven, by the structural check that runs before this is ever called,
    to exist on disk with a checksum matching its registered
    ArtifactRecord) and require its JSON to be an object with `passed`
    strictly equal to boolean True. Returns None if it does; otherwise a
    single, concise rejection reason string — never raises. `path` is
    None only if the artifact's own relative_path somehow failed the
    project-relative safety check (structurally already impossible to
    reach here, since the structural loop above would already have
    rejected it first — handled defensively anyway, not left to crash)."""
    if path is None:
        return "qc_report: relative_path is unsafe"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return f"qc_report: could not read verified report file at {path}: {exc}"
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return f"qc_report: verified report file at {path} is not valid JSON: {exc}"
    if not isinstance(data, dict):
        return f"qc_report: verified report file at {path} must be a JSON object at the top level"
    if "passed" not in data or not isinstance(data["passed"], bool):
        return f"qc_report: verified report file at {path} must have a boolean 'passed' field"
    if data["passed"] is not True:
        return f"qc_report: verified report file at {path} records passed=false — QC did not pass"
    return None


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

    # Phase 2H: qc_passed-only semantic gate — see module docstring.
    # Structural verification (above) proves the qc_report FILE matches
    # what was registered; it says nothing about its content. Only once
    # every structural requirement (render AND qc_report) has already
    # passed do we additionally require the verified report's own JSON to
    # say passed=true — never falling back to ArtifactRecord.metadata.
    if to_stage == "qc_passed" and not reasons:
        qc_artifact = _matching(artifacts_tuple, "qc_report", None)[0]
        resolved = resolve_under_project_dir(project_dir, qc_artifact.relative_path)
        semantic_error = _qc_report_passed(resolved)
        if semantic_error is not None:
            reasons.append(semantic_error)

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
