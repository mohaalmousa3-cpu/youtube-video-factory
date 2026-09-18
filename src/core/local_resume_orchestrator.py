"""LOCAL RESUME ORCHESTRATOR V1: local-only coordination of the render ->
overlay-derivation -> overlay-rendering tail of the pipeline.

Given a project_id and four explicit paths (--timed-manifest,
--derived-manifest-output, --render-output, --overlay-output), this module
decides — using src.core.final_assembly_planner.build_final_assembly_plan()
as its decision engine, never reimplemented here — which of the three
already-existing, already-hardened local production functions still need to
run, and calls them in order:

  1. src.core.final_video_assembly.assemble_final_video()
  2. src.core.overlay_derivation.derive_text_overlays() (+ save_manifest())
  3. src.core.text_overlay_render.render_text_overlays()

This module never implements media generation itself, never invokes FFmpeg/
ffprobe directly, never imports or calls a provider, and never calls
src.core.artifact_verifier — reuse based on the planner's strict metadata/
lineage contract only (see build_final_local()'s own docstring). It never
writes a manifest itself (save_manifest() is the only writer, called
exactly once, only in the derivation-created case) and never touches
project lifecycle state, the `jobs` table, or the database schema.

Connection lifecycle: this module opens only short-lived, read-only SQLite
connections of its own (to load the ProjectRecord and its registered
artifacts, and to re-check them after a stage completes), always closed
before calling assemble_final_video(), derive_text_overlays(),
save_manifest(), or render_text_overlays() — each of those either manages
its own connection internally (the first and third) or performs no I/O at
all (the second and fourth)."""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.core.final_assembly_planner import FinalAssemblyPlannerError, build_final_assembly_plan
from src.core.final_video_assembly import FinalVideoAssemblyError, assemble_final_video
from src.core.manifest_store import ManifestStoreError, load_manifest, save_manifest
from src.core.overlay_derivation import OverlayDerivationError, derive_text_overlays
from src.core.path_safety import resolve_under_project_dir
from src.core.text_overlay_render import TextOverlayRenderError, render_text_overlays
from src.database.artifact_repository import list_artifacts_by_project
from src.database.db import get_readonly_connection
from src.database.project_repository import get_project
from src.models.project_state import ProjectRecord

_RENDER_RELATIVE_PATH = "render/final.mp4"
_OVERLAY_RENDER_RELATIVE_PATH = "overlay_render/final.mp4"

_INTEGRITY_NOTE = (
    "artifact reuse is based on the existing planner's strict metadata/lineage contract, not an "
    "on-disk checksum verification — run verify-artifacts PROJECT_ID first if filesystem integrity "
    "of an existing render or overlay_render file is uncertain"
)

RenderStatus = Literal["render_reused", "render_created", "render_missing", "blocked", "failed"]
DerivedManifestStatus = Literal[
    "derived_manifest_reused", "derived_manifest_created", "not_needed", "blocked", "failed"
]
OverlayRenderStatus = Literal[
    "overlay_render_reused", "overlay_render_created", "not_attempted", "blocked", "failed"
]


class LocalResumeOrchestratorError(Exception):
    """Raised only for invalid caller input or a path-collision condition
    detected before any stage runs (or an impossible internal-consistency
    failure between two planner checkpoints) — never merely because a
    prerequisite stage hasn't completed yet or a stage's own call failed;
    those are reported via the returned LocalResumeResult instead."""


@dataclass(frozen=True)
class LocalResumeResult:
    """The one typed, immutable result build_final_local() always returns
    on any outcome short of an up-front LocalResumeOrchestratorError —
    including a stage failure partway through. Every path field is a
    string (never a Path) or None when nothing was written/reused there."""

    project_id: str
    render_status: RenderStatus
    render_artifact_id: str | None
    render_output_path: str | None
    derived_manifest_status: DerivedManifestStatus
    derived_manifest_path: str | None
    overlay_render_status: OverlayRenderStatus
    overlay_render_artifact_id: str | None
    overlay_render_output_path: str | None
    stopped_at_stage: str | None
    blocked_reasons: tuple[str, ...]
    notes: tuple[str, ...]


def _norm(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def _load_project_and_artifacts(project_id: str) -> tuple[ProjectRecord, list]:
    try:
        conn = get_readonly_connection()
    except sqlite3.Error as exc:
        raise LocalResumeOrchestratorError("no local project database found") from exc
    try:
        project = get_project(conn, project_id)
        if project is None:
            raise LocalResumeOrchestratorError(f"unknown project_id {project_id!r}")
        artifacts = list_artifacts_by_project(conn, project_id)
    finally:
        conn.close()
    return project, artifacts


def _validate_paths(
    project: ProjectRecord,
    artifacts: list,
    timed_manifest_path: Path,
    derived_manifest_output_path: Path,
    render_output_path: Path,
    overlay_output_path: Path,
) -> None:
    """STEP 4: pairwise distinctness of the four supplied paths, plus
    alias checks of the three OUTPUT paths against every protected path
    (the canonical manifest, render/final.mp4, overlay_render/final.mp4,
    and every registered artifact's own file). Applies unconditionally,
    regardless of which stages will end up skipped. Never creates a
    parent directory, never touches the filesystem beyond .resolve()."""
    supplied = {
        "--timed-manifest": timed_manifest_path,
        "--derived-manifest-output": derived_manifest_output_path,
        "--render-output": render_output_path,
        "--overlay-output": overlay_output_path,
    }
    items = list(supplied.items())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if _norm(items[i][1]) == _norm(items[j][1]):
                raise LocalResumeOrchestratorError(
                    f"{items[i][0]} and {items[j][0]} must not resolve to the same path"
                )

    project_dir_resolved = Path(project.manifest_path).resolve().parent
    protected = {
        "the project's canonical manifest path": _norm(Path(project.manifest_path)),
        "render/final.mp4 under the project directory": _norm(project_dir_resolved / _RENDER_RELATIVE_PATH),
        "overlay_render/final.mp4 under the project directory": _norm(
            project_dir_resolved / _OVERLAY_RENDER_RELATIVE_PATH
        ),
    }
    for artifact in artifacts:
        resolved = resolve_under_project_dir(project_dir_resolved, artifact.relative_path)
        if resolved is not None:
            protected[f"registered {artifact.kind} artifact {artifact.artifact_id!r}"] = _norm(resolved)

    for label in ("--derived-manifest-output", "--render-output", "--overlay-output"):
        path_n = _norm(supplied[label])
        for protected_label, protected_n in protected.items():
            if path_n == protected_n:
                raise LocalResumeOrchestratorError(f"{label} must not alias {protected_label}")


def _with_note(notes: tuple[str, ...], extra: str) -> tuple[str, ...]:
    return notes + (extra,) if extra not in notes else notes


def build_final_local(
    *,
    project_id: str,
    timed_manifest_path: Path,
    derived_manifest_output_path: Path,
    render_output_path: Path,
    overlay_output_path: Path,
) -> LocalResumeResult:
    """Coordinate assemble_final_video() -> derive_text_overlays() ->
    render_text_overlays() for `project_id`, skipping any stage whose
    output already validly exists per build_final_assembly_plan(), and
    never overwriting an existing caller file. Raises
    LocalResumeOrchestratorError for invalid input or a path collision
    detected before any stage runs; any ordinary stage failure is instead
    reported via the returned LocalResumeResult (stopped_at_stage +
    blocked_reasons), never raised."""
    if not project_id:
        raise LocalResumeOrchestratorError("project_id must not be empty")

    timed_manifest_path = Path(timed_manifest_path)
    derived_manifest_output_path = Path(derived_manifest_output_path)
    render_output_path = Path(render_output_path)
    overlay_output_path = Path(overlay_output_path)

    project, artifacts = _load_project_and_artifacts(project_id)

    if not timed_manifest_path.exists():
        raise LocalResumeOrchestratorError("--timed-manifest does not exist")
    if not timed_manifest_path.is_file():
        raise LocalResumeOrchestratorError("--timed-manifest is not a regular file")

    _validate_paths(
        project, artifacts, timed_manifest_path, derived_manifest_output_path, render_output_path, overlay_output_path
    )

    try:
        timed_manifest = load_manifest(timed_manifest_path)
    except ManifestStoreError as exc:
        raise LocalResumeOrchestratorError("could not load --timed-manifest (unreadable or invalid)") from exc

    try:
        plan = build_final_assembly_plan(project, timed_manifest, artifacts)
    except FinalAssemblyPlannerError as exc:
        raise LocalResumeOrchestratorError(str(exc)) from exc

    # B: a valid overlay_render already exists — full skip, nothing is
    # written, render_output/derived_manifest_output/overlay_output are
    # all ignored (per STEP 4's resume exception).
    if plan.overlay_render_ready:
        return LocalResumeResult(
            project_id=project_id,
            render_status="render_reused",
            render_artifact_id=plan.render_artifact_id,
            render_output_path=None,
            derived_manifest_status="not_needed",
            derived_manifest_path=None,
            overlay_render_status="overlay_render_reused",
            overlay_render_artifact_id=plan.overlay_render_artifact_id,
            overlay_render_output_path=None,
            stopped_at_stage=None,
            blocked_reasons=(),
            notes=("overlay_render/final.mp4 is the enhanced viewer-facing output.", _INTEGRITY_NOTE),
        )

    # From here on overlay_render is not ready, so overlay_output_path
    # will be needed eventually — hard-stop up front if it already exists,
    # before touching anything else (STEP 4.4/4.8).
    if overlay_output_path.exists():
        raise LocalResumeOrchestratorError("--overlay-output already exists")

    notes: tuple[str, ...] = ()

    # C/D: render handling.
    if plan.render_ready:
        render_status: RenderStatus = "render_reused"
        render_output_result: str | None = None
        notes = _with_note(notes, _INTEGRITY_NOTE)
    else:
        if render_output_path.exists():
            raise LocalResumeOrchestratorError("--render-output already exists")
        try:
            assemble_final_video(project_id, timed_manifest_path, render_output_path)
        except FinalVideoAssemblyError as exc:
            return LocalResumeResult(
                project_id=project_id,
                render_status="failed",
                render_artifact_id=None,
                render_output_path=None,
                derived_manifest_status="not_attempted",
                derived_manifest_path=None,
                overlay_render_status="not_attempted",
                overlay_render_artifact_id=None,
                overlay_render_output_path=None,
                stopped_at_stage="assemble_final_video",
                blocked_reasons=(str(exc),),
                notes=(),
            )

        project, artifacts = _load_project_and_artifacts(project_id)
        try:
            timed_manifest = load_manifest(timed_manifest_path)
        except ManifestStoreError as exc:
            raise LocalResumeOrchestratorError(
                "--timed-manifest could not be reloaded after assemble_final_video succeeded"
            ) from exc
        try:
            plan = build_final_assembly_plan(project, timed_manifest, artifacts)
        except FinalAssemblyPlannerError as exc:
            raise LocalResumeOrchestratorError(str(exc)) from exc
        if not plan.render_ready:
            raise LocalResumeOrchestratorError(
                "assemble_final_video succeeded but the planner does not report a valid render afterward"
            )
        render_status = "render_created"
        render_output_result = str(render_output_path.resolve())

    render_artifact = next(a for a in artifacts if a.kind == "render")

    # E: derived-manifest handling.
    if plan.manifest_has_explicit_overlays:
        manifest_to_render_path = timed_manifest_path
        manifest_to_render = timed_manifest
        derived_status: DerivedManifestStatus = "not_needed"
        derived_path_result: str | None = str(timed_manifest_path.resolve())
    elif not derived_manifest_output_path.exists():
        # E2: derive fresh.
        try:
            derived_manifest = derive_text_overlays(timed_manifest, render_artifact)
        except OverlayDerivationError as exc:
            return LocalResumeResult(
                project_id=project_id,
                render_status=render_status,
                render_artifact_id=render_artifact.artifact_id,
                render_output_path=render_output_result,
                derived_manifest_status="failed",
                derived_manifest_path=None,
                overlay_render_status="not_attempted",
                overlay_render_artifact_id=None,
                overlay_render_output_path=None,
                stopped_at_stage="derive_text_overlays",
                blocked_reasons=(str(exc),),
                notes=notes,
            )
        try:
            save_manifest(derived_manifest, derived_manifest_output_path)
        except ManifestStoreError as exc:
            return LocalResumeResult(
                project_id=project_id,
                render_status=render_status,
                render_artifact_id=render_artifact.artifact_id,
                render_output_path=render_output_result,
                derived_manifest_status="failed",
                derived_manifest_path=None,
                overlay_render_status="not_attempted",
                overlay_render_artifact_id=None,
                overlay_render_output_path=None,
                stopped_at_stage="derive_text_overlays",
                blocked_reasons=(str(exc),),
                notes=notes,
            )

        try:
            reloaded = load_manifest(derived_manifest_output_path)
        except ManifestStoreError as exc:
            raise LocalResumeOrchestratorError(
                "the just-written derived manifest could not be reloaded for validation"
            ) from exc
        if reloaded.project_id != project_id or reloaded.source_fingerprint != timed_manifest.source_fingerprint:
            raise LocalResumeOrchestratorError(
                "the just-written derived manifest failed post-write identity validation"
            )
        if not any(scene.text_overlays for scene in reloaded.scene_plan.scenes):
            raise LocalResumeOrchestratorError(
                "the just-written derived manifest unexpectedly has no explicit overlays"
            )

        manifest_to_render_path = derived_manifest_output_path
        manifest_to_render = reloaded
        derived_status = "derived_manifest_created"
        derived_path_result = str(derived_manifest_output_path.resolve())
    else:
        # E3: an existing derived-manifest file — reuse only if it is
        # exactly what deterministic derivation would produce right now;
        # never overwritten either way.
        try:
            existing_derived = load_manifest(derived_manifest_output_path)
        except ManifestStoreError:
            return LocalResumeResult(
                project_id=project_id,
                render_status=render_status,
                render_artifact_id=render_artifact.artifact_id,
                render_output_path=render_output_result,
                derived_manifest_status="blocked",
                derived_manifest_path=None,
                overlay_render_status="not_attempted",
                overlay_render_artifact_id=None,
                overlay_render_output_path=None,
                stopped_at_stage="derived_manifest_validation",
                blocked_reasons=(
                    "--derived-manifest-output already exists but could not be loaded as a valid manifest",
                ),
                notes=notes,
            )

        try:
            expected_derived = derive_text_overlays(timed_manifest, render_artifact)
        except OverlayDerivationError as exc:
            return LocalResumeResult(
                project_id=project_id,
                render_status=render_status,
                render_artifact_id=render_artifact.artifact_id,
                render_output_path=render_output_result,
                derived_manifest_status="blocked",
                derived_manifest_path=None,
                overlay_render_status="not_attempted",
                overlay_render_artifact_id=None,
                overlay_render_output_path=None,
                stopped_at_stage="derived_manifest_validation",
                blocked_reasons=(str(exc),),
                notes=notes,
            )

        if existing_derived.model_dump(mode="json") != expected_derived.model_dump(mode="json"):
            return LocalResumeResult(
                project_id=project_id,
                render_status=render_status,
                render_artifact_id=render_artifact.artifact_id,
                render_output_path=render_output_result,
                derived_manifest_status="blocked",
                derived_manifest_path=None,
                overlay_render_status="not_attempted",
                overlay_render_artifact_id=None,
                overlay_render_output_path=None,
                stopped_at_stage="derived_manifest_validation",
                blocked_reasons=(
                    "existing derived manifest does not match deterministic derivation from the "
                    "supplied timed manifest and render artifact",
                ),
                notes=notes,
            )

        manifest_to_render_path = derived_manifest_output_path
        manifest_to_render = existing_derived
        derived_status = "derived_manifest_reused"
        derived_path_result = str(derived_manifest_output_path.resolve())
        notes = _with_note(notes, _INTEGRITY_NOTE)

    # F: overlay rendering.
    try:
        render_text_overlays(project_id, manifest_to_render_path, overlay_output_path)
    except TextOverlayRenderError as exc:
        return LocalResumeResult(
            project_id=project_id,
            render_status=render_status,
            render_artifact_id=render_artifact.artifact_id,
            render_output_path=render_output_result,
            derived_manifest_status=derived_status,
            derived_manifest_path=derived_path_result,
            overlay_render_status="failed",
            overlay_render_artifact_id=None,
            overlay_render_output_path=None,
            stopped_at_stage="render_text_overlays",
            blocked_reasons=(str(exc),),
            notes=notes,
        )

    _project, artifacts = _load_project_and_artifacts(project_id)
    try:
        final_plan = build_final_assembly_plan(project, manifest_to_render, artifacts)
    except FinalAssemblyPlannerError as exc:
        raise LocalResumeOrchestratorError(str(exc)) from exc
    if not final_plan.overlay_render_ready:
        raise LocalResumeOrchestratorError(
            "render_text_overlays succeeded but the planner does not report a valid overlay_render afterward"
        )

    return LocalResumeResult(
        project_id=project_id,
        render_status=render_status,
        render_artifact_id=render_artifact.artifact_id,
        render_output_path=render_output_result,
        derived_manifest_status=derived_status,
        derived_manifest_path=derived_path_result,
        overlay_render_status="overlay_render_created",
        overlay_render_artifact_id=final_plan.overlay_render_artifact_id,
        overlay_render_output_path=str(overlay_output_path.resolve()),
        stopped_at_stage=None,
        blocked_reasons=(),
        notes=notes,
    )
