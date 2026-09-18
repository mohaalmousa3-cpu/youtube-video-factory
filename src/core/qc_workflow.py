"""QC WORKFLOW WRAPPER V1: local-only coordination of the QC-report-to-
qc_passed tail of the pipeline.

Given a project_id and an explicit manifest/report-output path, this module
sequences four already-existing, already-hardened local contracts, in
order, and never reimplements any of them:

  1. src.core.final_qc_gate.verify_final_output() — computes the QC
     verdict; never writes anything itself.
  2. An atomic report-file write (temp file + os.replace()) — the same
     small pattern src.cli.cmd_verify_final_output and
     src.core.manifest_store.save_manifest() already use, duplicated here
     rather than imported (this codebase's established convention for
     small, self-contained write helpers).
  3. src.core.qc_report_artifact_registrar.register_qc_report_artifact() —
     registers the report as the project's one qc_report artifact; never
     called when the report says passed=false (see module docstring for
     why: qc_report is a one-slot, no-overwrite artifact identity, so
     eagerly registering a failure would permanently block ever
     registering a later, corrected passed=true report).
  4. src.core.verified_transition_service.verify_and_advance() — the sole
     existing path to "qc_passed"; this module never duplicates its
     lifecycle-transition rules (current-stage legality, lifecycle-version
     optimistic locking, the passed=true semantic gate) and never checks
     project.current_stage itself before calling it.

Resume-safe: if a qc_report artifact is already registered for this
project, this module never regenerates a new report (FinalQcReport's own
generated_at timestamp would make a fresh report byte-different from an
already-registered one even when every substantive field is identical,
which would make a completely valid rerun look like "different content"
to the registrar's own no-overwrite check) — it reads the existing report
file directly and, if it already says passed=true, proceeds straight to
verify_and_advance() using the existing registration.

No provider, no paid service, no network call, no direct FFmpeg/ffprobe
call, no subprocess import, no direct artifact_verifier import, no video/
audio/image generation, no call to build-final-local/assemble-final-video/
derive-text-overlays/render-text-overlays, no publishing, and no
lifecycle move beyond "qc_passed" (never "ready_for_manual_publish" or
"completed") anywhere in this module.

An ORDINARY outcome short of full success — QC failed, an existing
qc_report is unusable, registration is rejected, or the transition is
rejected — is never raised; it is represented in the returned
QcWorkflowResult (stopped_at_step + blocked_reasons). This module raises
only for invalid caller input, an unsafe/colliding --report-output path,
a project/manifest identity mismatch, or an impossible artifact topology
(more than one registered qc_report artifact, or the propagated
final_qc_gate.ArtifactTopologyError for more than one render/
overlay_render artifact) — in every one of these cases nothing has been
attempted yet, so there is no partial state to report."""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path

from src.core.final_qc_gate import (
    ArtifactTopologyError as FinalQcArtifactTopologyError,
    ProjectManifestMismatchError as FinalQcProjectManifestMismatchError,
    verify_final_output,
)
from src.core.manifest_store import ManifestStoreError, load_manifest
from src.core.path_safety import resolve_under_project_dir
from src.core.qc_report_artifact_registrar import QcReportArtifactRegistrationError, register_qc_report_artifact
from src.core.verified_transition_service import VerifiedTransitionServiceError, verify_and_advance
from src.database.artifact_repository import list_artifacts_by_project
from src.database.db import get_existing_connection, get_readonly_connection
from src.database.project_repository import get_project
from src.models.manifest import VideoManifest
from src.models.project_state import ProjectRecord

_RENDER_RELATIVE_PATH = "render/final.mp4"
_OVERLAY_RENDER_RELATIVE_PATH = "overlay_render/final.mp4"


class QcWorkflowError(Exception):
    """Base class for every reason run_qc_workflow() cannot even begin —
    invalid caller input or an unsafe/preflight condition detected before
    any file is written or any database write is attempted. Never raised
    for an ordinary QC failure, registration rejection, lifecycle
    rejection, or lifecycle-version conflict; those are represented in a
    returned QcWorkflowResult instead."""


class QcWorkflowProjectManifestMismatchError(QcWorkflowError):
    """The supplied manifest's project_id or source_fingerprint does not
    match the loaded ProjectRecord."""


class QcWorkflowReportOutputConflictError(QcWorkflowError):
    """--report-output already exists, or aliases the canonical project
    manifest, the supplied --manifest, render/final.mp4,
    overlay_render/final.mp4, or a registered artifact's own file."""


class QcWorkflowArtifactTopologyError(QcWorkflowError):
    """An impossible artifact topology: more than one registered render or
    overlay_render artifact (propagated from
    final_qc_gate.ArtifactTopologyError, never reimplemented here)."""


@dataclass(frozen=True)
class QcWorkflowResult:
    """The one typed, immutable result run_qc_workflow() always returns on
    any outcome short of an up-front QcWorkflowError — including a
    partial completion. Every path/identity field is a string (never a
    Path) or None when nothing applies. `verification_report_summary` is
    a plain JSON-safe dict (the same payload written to --report-output,
    or None when no report was generated this run), never the
    FinalQcReport object itself."""

    project_id: str
    qc_report_generated: bool
    qc_report_passed: bool | None
    qc_report_path: str | None
    qc_report_artifact_registered: bool
    qc_report_artifact_id: str | None
    lifecycle_advanced: bool
    resulting_project_stage: str | None
    stopped_at_step: str | None
    blocked_reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    verification_report_summary: dict | None


def _load_project_and_artifacts(project_id: str) -> tuple[ProjectRecord, list]:
    try:
        conn = get_readonly_connection()
    except sqlite3.Error as exc:
        raise QcWorkflowError("no local project database found") from exc
    try:
        project = get_project(conn, project_id)
        if project is None:
            raise QcWorkflowError(f"unknown project_id {project_id!r}")
        artifacts = list_artifacts_by_project(conn, project_id)
    finally:
        conn.close()
    return project, artifacts


def _norm(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def _validate_report_output_path(
    project: ProjectRecord, artifacts: list, manifest_path: Path, report_output_path: Path
) -> None:
    if report_output_path.exists():
        raise QcWorkflowReportOutputConflictError("--report-output already exists")

    project_dir_resolved = Path(project.manifest_path).resolve().parent
    protected = {
        "the project's canonical manifest path": _norm(Path(project.manifest_path)),
        "the supplied --manifest path": _norm(manifest_path),
        "render/final.mp4 under the project directory": _norm(project_dir_resolved / _RENDER_RELATIVE_PATH),
        "overlay_render/final.mp4 under the project directory": _norm(
            project_dir_resolved / _OVERLAY_RENDER_RELATIVE_PATH
        ),
    }
    for artifact in artifacts:
        resolved = resolve_under_project_dir(project_dir_resolved, artifact.relative_path)
        if resolved is not None:
            protected[f"registered {artifact.kind} artifact {artifact.artifact_id!r}"] = _norm(resolved)

    report_output_normcase = _norm(report_output_path)
    for label, path_n in protected.items():
        if report_output_normcase == path_n:
            raise QcWorkflowReportOutputConflictError(f"--report-output must not alias {label}")


def _qc_report_payload(report) -> dict:
    """The same JSON-safe shape src.cli.cmd_verify_final_output's own
    private payload builder already produces — duplicated here (not
    imported: core/ never depends on cli.py) since this module needs it
    to actually write the file, not just print it."""
    return {
        "passed": report.passed,
        "project_id": report.project_id,
        "require_overlays": report.require_overlays,
        "render_checks": [
            {"check_id": c.check_id, "passed": c.passed, "subject": c.subject, "message": c.message}
            for c in report.render_checks
        ],
        "overlay_render_checks": [
            {"check_id": c.check_id, "passed": c.passed, "subject": c.subject, "message": c.message}
            for c in report.overlay_render_checks
        ],
        "overlay_render_present": report.overlay_render_present,
        "viewer_facing_output_kind": report.viewer_facing_output_kind,
        "viewer_facing_output_relative_path": report.viewer_facing_output_relative_path,
        "blocking_reasons": list(report.blocking_reasons),
        "warnings": list(report.warnings),
        "generated_at": report.generated_at,
    }


def _write_report_atomic(report_output_path: Path, text: str) -> None:
    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(report_output_path.parent), prefix=f".{report_output_path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_name, report_output_path)
    except OSError:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _peek_existing_report(path: Path | None) -> tuple[bool | None, str | None]:
    """Read an already-registered qc_report artifact's own file (never its
    ArtifactRecord.metadata, which is mutable DB state, not the source of
    truth — same principle verified_transition_service.py's own
    _qc_report_passed() already applies) and extract its `passed` value.
    Only a structural, filesystem/JSON-level check (existence, valid
    JSON, a boolean `passed` field) — never a checksum/byte-size
    comparison, which stays exclusively the verifier module's job and is
    never invoked here. Returns (passed_bool, None) on a structurally
    valid report or (None, error_message) on any failure."""
    if path is None:
        return None, "existing qc_report artifact's relative_path is unsafe"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, f"could not read existing qc_report file: {exc.__class__.__name__}"
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None, "existing qc_report file is not valid JSON"
    if not isinstance(data, dict):
        return None, "existing qc_report file must be a JSON object at the top level"
    if "passed" not in data or not isinstance(data["passed"], bool):
        return None, "existing qc_report file must have a boolean 'passed' field"
    return data["passed"], None


def _attempt_transition(
    *,
    project_id: str,
    manifest: VideoManifest,
    reason: str,
    qc_report_generated: bool,
    qc_report_path: str | None,
    qc_report_artifact_id: str,
    warnings: tuple[str, ...],
    verification_report_summary: dict | None,
) -> QcWorkflowResult:
    """STEP 7: attempt verify_and_advance(to_stage='qc_passed') using a
    fresh, short-lived write connection, reached only after a passed=true
    qc_report artifact is already known to be registered (either just
    registered this run, or found already valid). Never calls
    the non-verified stage-advance service, never targets any stage beyond qc_passed, never
    duplicates verify_and_advance's own current-stage or lifecycle-version
    checks."""

    def _result(**overrides) -> QcWorkflowResult:
        base = dict(
            project_id=project_id,
            qc_report_generated=qc_report_generated,
            qc_report_passed=True,
            qc_report_path=qc_report_path,
            qc_report_artifact_registered=True,
            qc_report_artifact_id=qc_report_artifact_id,
            lifecycle_advanced=False,
            resulting_project_stage=None,
            stopped_at_step="verify_and_advance",
            blocked_reasons=(),
            warnings=warnings,
            verification_report_summary=verification_report_summary,
        )
        base.update(overrides)
        return QcWorkflowResult(**base)

    try:
        conn = get_existing_connection()
    except sqlite3.Error:
        return _result(blocked_reasons=("no local project database found",))

    try:
        project = get_project(conn, project_id)
        if project is None:
            return _result(blocked_reasons=(f"unknown project_id {project_id!r}",))
        artifacts = list_artifacts_by_project(conn, project_id)
        try:
            transition_result = verify_and_advance(conn, project, manifest, artifacts, "qc_passed", reason)
        except VerifiedTransitionServiceError as exc:
            return _result(resulting_project_stage=project.current_stage, blocked_reasons=(str(exc),))
    except Exception:
        # Never hides partial state: the report file and registered
        # qc_report artifact are already real and are left untouched.
        return _result(blocked_reasons=("verified transition failed unexpectedly",))
    finally:
        conn.close()

    if transition_result.approved and transition_result.db_committed:
        return _result(
            lifecycle_advanced=True,
            resulting_project_stage="qc_passed",
            stopped_at_step=None,
            blocked_reasons=(),
        )

    return _result(
        resulting_project_stage=project.current_stage,
        blocked_reasons=transition_result.reasons,
    )


def run_qc_workflow(
    *,
    project_id: str,
    manifest_path: Path,
    report_output_path: Path,
    require_overlays: bool,
    reason: str,
) -> QcWorkflowResult:
    """Coordinate verify_final_output() -> atomic report write ->
    register_qc_report_artifact() -> verify_and_advance(to='qc_passed')
    for `project_id`. Resume-safe: skips generation/registration entirely
    when a valid, already-registered, passed=true qc_report artifact
    already exists. Raises QcWorkflowError subclasses only for invalid
    input or a preflight-level path/topology problem; every ordinary
    outcome is returned as a QcWorkflowResult."""
    if not project_id:
        raise QcWorkflowError("project_id must not be empty")
    if not reason or not reason.strip():
        raise QcWorkflowError("reason must not be empty")

    manifest_path = Path(manifest_path)
    report_output_path = Path(report_output_path)

    project, artifacts = _load_project_and_artifacts(project_id)

    if not manifest_path.exists():
        raise QcWorkflowError("--manifest does not exist")
    if not manifest_path.is_file():
        raise QcWorkflowError("--manifest is not a regular file")

    try:
        manifest = load_manifest(manifest_path)
    except ManifestStoreError as exc:
        raise QcWorkflowError("could not load --manifest (unreadable or invalid)") from exc

    if manifest.project_id != project_id:
        raise QcWorkflowProjectManifestMismatchError(
            f"--manifest project_id {manifest.project_id!r} does not match project_id {project_id!r}"
        )
    if manifest.source_fingerprint != project.manifest_fingerprint:
        raise QcWorkflowProjectManifestMismatchError(
            "--manifest source_fingerprint does not match the project registry's recorded fingerprint"
        )

    _validate_report_output_path(project, artifacts, manifest_path, report_output_path)

    project_dir_resolved = Path(project.manifest_path).resolve().parent
    qc_report_artifacts = [a for a in artifacts if a.kind == "qc_report"]

    if len(qc_report_artifacts) > 1:
        return QcWorkflowResult(
            project_id=project_id,
            qc_report_generated=False,
            qc_report_passed=None,
            qc_report_path=None,
            qc_report_artifact_registered=False,
            qc_report_artifact_id=None,
            lifecycle_advanced=False,
            resulting_project_stage=project.current_stage,
            stopped_at_step="qc_report_topology",
            blocked_reasons=(
                f"project {project_id!r} has {len(qc_report_artifacts)} 'qc_report' artifacts "
                "(expected at most 1)",
            ),
            warnings=(),
            verification_report_summary=None,
        )

    if len(qc_report_artifacts) == 1:
        existing = qc_report_artifacts[0]
        resolved = resolve_under_project_dir(project_dir_resolved, existing.relative_path)
        existing_path = str(resolved) if resolved is not None else None
        passed, error = _peek_existing_report(resolved)

        if error is not None:
            return QcWorkflowResult(
                project_id=project_id,
                qc_report_generated=False,
                qc_report_passed=None,
                qc_report_path=existing_path,
                qc_report_artifact_registered=True,
                qc_report_artifact_id=existing.artifact_id,
                lifecycle_advanced=False,
                resulting_project_stage=project.current_stage,
                stopped_at_step="qc_report_preflight",
                blocked_reasons=(error,),
                warnings=(),
                verification_report_summary=None,
            )

        if not passed:
            return QcWorkflowResult(
                project_id=project_id,
                qc_report_generated=False,
                qc_report_passed=False,
                qc_report_path=existing_path,
                qc_report_artifact_registered=True,
                qc_report_artifact_id=existing.artifact_id,
                lifecycle_advanced=False,
                resulting_project_stage=project.current_stage,
                stopped_at_step="qc_report_preflight",
                blocked_reasons=(
                    "existing qc_report artifact records passed=false — QC did not pass",
                ),
                warnings=(),
                verification_report_summary=None,
            )

        return _attempt_transition(
            project_id=project_id,
            manifest=manifest,
            reason=reason,
            qc_report_generated=False,
            qc_report_path=existing_path,
            qc_report_artifact_id=existing.artifact_id,
            warnings=(),
            verification_report_summary=None,
        )

    # No existing qc_report artifact — normal path: generate a fresh report.
    try:
        report = verify_final_output(
            project=project,
            manifest=manifest,
            artifacts=artifacts,
            project_dir=project_dir_resolved,
            require_overlays=require_overlays,
        )
    except FinalQcProjectManifestMismatchError as exc:
        raise QcWorkflowProjectManifestMismatchError(str(exc)) from exc
    except FinalQcArtifactTopologyError as exc:
        raise QcWorkflowArtifactTopologyError(str(exc)) from exc

    payload = _qc_report_payload(report)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"

    try:
        _write_report_atomic(report_output_path, text)
    except OSError:
        return QcWorkflowResult(
            project_id=project_id,
            qc_report_generated=False,
            qc_report_passed=report.passed,
            qc_report_path=None,
            qc_report_artifact_registered=False,
            qc_report_artifact_id=None,
            lifecycle_advanced=False,
            resulting_project_stage=project.current_stage,
            stopped_at_step="report_write",
            blocked_reasons=("could not write --report-output",),
            warnings=report.warnings,
            verification_report_summary=None,
        )

    qc_report_path = str(report_output_path.resolve())

    if not report.passed:
        return QcWorkflowResult(
            project_id=project_id,
            qc_report_generated=True,
            qc_report_passed=False,
            qc_report_path=qc_report_path,
            qc_report_artifact_registered=False,
            qc_report_artifact_id=None,
            lifecycle_advanced=False,
            resulting_project_stage=project.current_stage,
            stopped_at_step="verify_final_output",
            blocked_reasons=report.blocking_reasons,
            warnings=report.warnings,
            verification_report_summary=payload,
        )

    # passed=True: register.
    def _registration_failed(reasons: tuple[str, ...], stage: str | None) -> QcWorkflowResult:
        return QcWorkflowResult(
            project_id=project_id,
            qc_report_generated=True,
            qc_report_passed=True,
            qc_report_path=qc_report_path,
            qc_report_artifact_registered=False,
            qc_report_artifact_id=None,
            lifecycle_advanced=False,
            resulting_project_stage=stage,
            stopped_at_step="register_qc_report_artifact",
            blocked_reasons=reasons,
            warnings=report.warnings,
            verification_report_summary=payload,
        )

    try:
        conn = get_existing_connection()
    except sqlite3.Error:
        return _registration_failed(("no local project database found",), project.current_stage)

    try:
        reg_project = get_project(conn, project_id)
        if reg_project is None:
            return _registration_failed((f"unknown project_id {project_id!r}",), None)
        try:
            reg_result = register_qc_report_artifact(conn, reg_project, manifest, report_output_path)
        except QcReportArtifactRegistrationError as exc:
            return _registration_failed((str(exc),), reg_project.current_stage)
    except Exception:
        # Never hides partial state: the report file stays on disk untouched.
        return _registration_failed(("qc_report registration failed unexpectedly",), project.current_stage)
    finally:
        conn.close()

    if not reg_result.ok:
        return _registration_failed(reg_result.reasons, reg_project.current_stage)

    return _attempt_transition(
        project_id=project_id,
        manifest=manifest,
        reason=reason,
        qc_report_generated=True,
        qc_report_path=qc_report_path,
        qc_report_artifact_id=reg_result.artifact_id,
        warnings=report.warnings,
        verification_report_summary=payload,
    )
