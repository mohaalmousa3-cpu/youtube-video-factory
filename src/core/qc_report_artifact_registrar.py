"""Phase 2H: registration-first, project-level QC Report Artifact
Registration.

Registers a single, already-produced local JSON file as THE project's
canonical "qc_report" artifact, so a project can later work toward Phase
2C's verify-and-advance(..., to_stage="qc_passed", ...) requirement of one
verified qc_report artifact (alongside the render artifact — see below)
for the whole project. This module never runs QC itself (no provider
call, no rendering, no comparison against any real output); never mutates
the manifest or ProjectRecord.manifest_fingerprint; and never advances a
project's lifecycle stage — see src/core/verified_transition_service.py
for that. It is also stage-agnostic: it never checks or cares what
current_stage the project is in.

Like Phase 2G's render, "qc_report" is a PROJECT-level ArtifactKind
(src/models/artifact.py's PROJECT_LEVEL_ARTIFACT_KINDS), so this module
takes no scene_id at all — there is exactly one QC report per project,
fixed and deterministic:

    artifact_id:    qc-report-final
    kind:           qc_report
    scene_id:       None
    relative_path:  qc/report.json

scene_id MUST be None here — enforced by ArtifactRecord itself
(src/models/artifact.py's _scene_association_matches_kind model validator
raises ValueError if a project-level kind carries a scene_id), the same
mechanism Phase 2G's render relies on. Existing-registration lookups
accordingly use src/database/artifact_repository.list_artifacts_by_project()
(filtered to kind="qc_report"), not list_artifacts_by_scene().

TWO VALIDATION LAYERS — read this before assuming "passed" is enforced
here, it deliberately is not:

  Layer 1 (THIS module, at registration time): the source file must be
  real, parseable JSON, a top-level object, with a `passed` field that is
  a genuine boolean. Either true or false is accepted and registered —
  this layer only asks "is this a well-formed report," never "did QC
  pass." A registered {"passed": false} report is real, retained audit
  data, not an error.

  Layer 2 (src/core/verified_transition_service.py, Phase 2H's other
  change, NOT in this module): only when advancing to "qc_passed"
  specifically, that module additionally re-reads the verified on-disk
  report and requires passed to be strictly True before allowing the
  transition. That is the ONLY place "passed" gates anything — this
  registrar never refuses to register a passed:false report, and never
  reads or writes ProjectRecord/lifecycle state itself.

Local filesystem and local SQLite only — no ffprobe, no Pillow, no
renderer, no provider, no network: this is the first of the four
registrars needing no external validation library at all, just stdlib
`json`.

Write paths, exactly three — identical shape to the three prior
project/scene-level registrars:
  1. mkdir <project_dir>/qc/ (only once every validation has passed)
  2. copy the source file to <project_dir>/qc/report.json (only for a
     genuinely fresh registration — never when the destination already
     has byte-identical content)
  3. register_artifact() — exactly one INSERT, and only after (1) and (2)
     (if needed) have already succeeded

Every ordinary rejection (invalid source file, unreadable/malformed/
non-object JSON, a missing or non-boolean `passed` field, an existing
record/file with different content, a documented register_artifact()
failure — ArtifactAlreadyExistsError, DuplicateArtifactRegistrationError,
ArtifactProjectNotFoundError, ArtifactRegistrationError, or any
sqlite3.Error, ...) is reported via the returned
QcReportArtifactRegistrationResult with ok=False — this module does not
raise for those. It raises QcReportArtifactRegistrationError only for a
caller/input inconsistency it will not silently work around (a manifest
that does not belong to the given project), the same split
src/core/verified_transition_service.py and the three prior registrars
use. An UNEXPECTED exception from register_artifact() (anything outside
that documented set) is a different case: this call's own freshly-copied
file (never a pre-existing one) is still cleaned up best-effort, but the
exception itself is re-raised unchanged rather than folded into a result —
an undocumented failure is a bug to surface, not an ordinary outcome to
report.

Deliberately NOT shared with audio/visual/animation/render_artifact_registrar.py
in this phase: _sha256_of_file() and the fresh-copy cleanup helper below
are intentional, small, self-contained duplicates (a fifth copy now), not
imports — a scoped choice, not an oversight; a shared-helper extraction is
left for a later, deliberate maintenance refactor, same as every prior
phase's own stated choice."""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.core.path_safety import resolve_under_project_dir
from src.database.artifact_repository import (
    ArtifactAlreadyExistsError,
    ArtifactProjectNotFoundError,
    ArtifactRegistrationError,
    DuplicateArtifactRegistrationError,
    list_artifacts_by_project,
    register_artifact,
)
from src.models.artifact import ArtifactRecord
from src.models.manifest import VideoManifest
from src.models.project_state import ProjectRecord

_CHUNK_SIZE = 1024 * 1024

_ARTIFACT_ID = "qc-report-final"
_RELATIVE_PATH = "qc/report.json"


class QcReportArtifactRegistrationError(Exception):
    """Raised only for a project/manifest identity mismatch — a caller
    bug, not a normal registration outcome. Ordinary rejections (invalid
    source file, malformed/missing/non-boolean 'passed', a conflicting
    existing record/file, ...) are reported via a returned
    QcReportArtifactRegistrationResult instead; see this module's
    docstring."""


@dataclass(frozen=True)
class QcReportArtifactRegistrationResult:
    """The one typed result register_qc_report_artifact() always returns,
    success or failure. Never itself mutates the manifest or a
    ProjectRecord's lifecycle state — it only reports what this call
    already did (or refused to do) to the artifact registry and, at most,
    one file. `passed` here is only the value THIS report claims — it is
    never the qc_passed transition's own gate; see this module's
    docstring."""

    project_id: str
    artifact_id: str
    relative_path: str
    ok: bool
    idempotent: bool
    copied: bool
    passed: bool | None
    artifact: ArtifactRecord | None
    reasons: tuple[str, ...] = ()


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_qc_report(source_file: Path) -> tuple[bool | None, str | None]:
    """Validate `source_file` is a structurally real QC report: readable,
    parseable JSON, a top-level object, with a strict boolean `passed`
    field. Either True or False is accepted here — see this module's
    docstring for why the value itself is never gated at registration
    time. Returns (passed_bool, None) on success or (None, error_message)
    on any failure."""
    try:
        text = source_file.read_text(encoding="utf-8")
    except OSError as exc:
        return None, f"could not read report file at {source_file}: {exc}"
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"report file at {source_file} is not valid JSON: {exc}"
    if not isinstance(data, dict):
        return None, f"report file at {source_file} must be a JSON object at the top level"
    if "passed" not in data:
        return None, f"report file at {source_file} is missing a 'passed' field"
    if not isinstance(data["passed"], bool):
        return None, (
            f"report file at {source_file}'s 'passed' field must be a boolean "
            f"(got {type(data['passed']).__name__})"
        )
    return data["passed"], None


def _cleanup_fresh_copy(destination: Path, created_fresh_copy: bool) -> None:
    """Delete `destination` only if THIS call is the one that just created
    it fresh — never a pre-existing file the call found already in place.
    Best-effort: swallows any failure while deleting so cleanup itself can
    never mask whatever original exception/rejection triggered it."""
    if not created_fresh_copy:
        return
    try:
        destination.unlink(missing_ok=True)
    except OSError:
        pass


def register_qc_report_artifact(
    conn: sqlite3.Connection,
    project: ProjectRecord,
    manifest: VideoManifest,
    source_file: Path,
    *,
    now: datetime | None = None,
) -> QcReportArtifactRegistrationResult:
    """Register `source_file` as this project's canonical, project-level
    "qc_report" artifact. Writes nothing to SQLite or the filesystem on
    any rejection path — see this module's docstring for the exact three
    write points a successful, non-idempotent call may reach.
    Stage-agnostic: never reads or checks project.current_stage. Takes no
    scene_id — "qc_report" is project-level, not scene-level. Accepts
    passed:true AND passed:false reports equally; see this module's
    docstring for where (and only where) passed actually gates anything."""
    when = now if now is not None else datetime.now(timezone.utc)
    if manifest.project_id != project.project_id:
        raise QcReportArtifactRegistrationError(
            f"manifest project_id {manifest.project_id!r} does not match project {project.project_id!r}"
        )
    if manifest.source_fingerprint != project.manifest_fingerprint:
        raise QcReportArtifactRegistrationError(
            "manifest fingerprint does not match the project registry's recorded fingerprint"
        )

    def _result(
        *,
        ok: bool,
        idempotent: bool = False,
        copied: bool = False,
        passed: bool | None = None,
        artifact: ArtifactRecord | None = None,
        reasons: tuple[str, ...] = (),
    ) -> QcReportArtifactRegistrationResult:
        return QcReportArtifactRegistrationResult(
            project_id=project.project_id,
            artifact_id=_ARTIFACT_ID,
            relative_path=_RELATIVE_PATH,
            ok=ok,
            idempotent=idempotent,
            copied=copied,
            passed=passed,
            artifact=artifact,
            reasons=reasons,
        )

    if not source_file.exists():
        return _result(ok=False, reasons=(f"source file does not exist at {source_file}",))
    if not source_file.is_file():
        return _result(ok=False, reasons=(f"source file at {source_file} is not a regular file",))
    source_size = source_file.stat().st_size
    if source_size == 0:
        return _result(ok=False, reasons=(f"source file at {source_file} is empty",))
    source_checksum = _sha256_of_file(source_file)

    project_dir_resolved = Path(project.manifest_path).resolve().parent
    destination = resolve_under_project_dir(project_dir_resolved, _RELATIVE_PATH)
    if destination is None:
        return _result(
            ok=False,
            reasons=(
                f"relative_path {_RELATIVE_PATH!r} is unsafe: it resolves outside "
                f"the project directory {project_dir_resolved}",
            ),
        )

    # 1. Already registered? Compare against the INCOMING file's checksum
    # only (never re-derived from whatever currently sits at `destination`
    # on disk) — an identical match is a pure no-op: no file touched, no
    # DB write. A mismatch is rejected outright; Phase 2H never overwrites
    # an existing registration. Project-level lookup (no scene to scope
    # by) — at most one "qc_report" row can ever exist per project by
    # construction (the artifacts table's own unique index on
    # (project_id, kind, COALESCE(scene_id,''), relative_path), with both
    # scene_id and relative_path fixed constants here).
    existing_matches = [a for a in list_artifacts_by_project(conn, project.project_id, kind="qc_report")]
    if existing_matches:
        existing = existing_matches[0]
        if existing.sha256_checksum == source_checksum:
            return _result(ok=True, idempotent=True, passed=existing.metadata.get("passed"), artifact=existing)
        return _result(
            ok=False,
            reasons=(
                f"a qc_report artifact is already registered for this project ({existing.artifact_id!r}) "
                "with different content — Phase 2H never overwrites an existing artifact registration",
            ),
        )

    # 2. No existing record. Does the destination file already exist
    # (placed there by some earlier, non-CLI step)? Register in place
    # without copying if its bytes already match; refuse to overwrite if
    # they don't. `created_fresh_copy` tracks whether THIS call is the one
    # that put a new file on disk, so a later register_artifact() failure
    # only ever cleans up a file this call itself created — never a
    # pre-existing one.
    created_fresh_copy = False
    if destination.exists():
        if not destination.is_file():
            return _result(ok=False, reasons=(f"destination at {destination} exists and is not a regular file",))
        dest_size = destination.stat().st_size
        dest_checksum = _sha256_of_file(destination)
        if dest_checksum != source_checksum or dest_size != source_size:
            return _result(
                ok=False,
                reasons=(
                    f"destination {destination} already exists with different content — "
                    "Phase 2H never overwrites an existing file",
                ),
            )
        final_checksum, final_size = dest_checksum, dest_size
    else:
        final_checksum, final_size = None, None  # computed after the copy below

    passed, report_error = _validate_qc_report(source_file)
    if report_error is not None:
        return _result(ok=False, reasons=(report_error,))

    if final_checksum is None:
        # Fresh path: nothing exists yet at `destination` — create qc/
        # only now, after every validation (including report structure
        # validation) has already passed.
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, destination)
        created_fresh_copy = True
        final_size = destination.stat().st_size
        final_checksum = _sha256_of_file(destination)  # hash the COPY, not the source

    record = ArtifactRecord(
        artifact_id=_ARTIFACT_ID,
        project_id=project.project_id,
        kind="qc_report",
        scene_id=None,
        relative_path=_RELATIVE_PATH,
        byte_size=final_size,
        sha256_checksum=final_checksum,
        created_at=when,
        metadata={"passed": passed, "source": "external"},
    )

    try:
        register_artifact(conn, record)
    except (
        ArtifactAlreadyExistsError,
        DuplicateArtifactRegistrationError,
        ArtifactProjectNotFoundError,
        ArtifactRegistrationError,
        sqlite3.Error,
    ) as exc:
        # A documented, expected failure mode of register_artifact() — an
        # ordinary rejection, reported via the result, never raised.
        _cleanup_fresh_copy(destination, created_fresh_copy)
        return _result(ok=False, passed=passed, reasons=(str(exc),))
    except Exception:
        # Anything else is undocumented/unexpected: still clean up a
        # freshly-copied file (never a pre-existing one), but this is a
        # bug to surface, not an ordinary outcome — re-raise unchanged
        # rather than silently downgrading it into a rejected result.
        _cleanup_fresh_copy(destination, created_fresh_copy)
        raise

    return _result(
        ok=True,
        copied=created_fresh_copy,
        passed=passed,
        artifact=record,
    )
