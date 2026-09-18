"""FINAL ASSEMBLY PLANNER V1: pure, local, read-only planning report for
the render -> overlay-derivation -> overlay-rendering tail of the
pipeline — the part src.core.dry_run_orchestrator's existing
build_dry_run_report() cannot see, since neither the "overlay_render"
ArtifactKind nor the manifest-file-based overlay-derivation step is part
of src.models.project_state.ProjectStage's lifecycle model at all.

This module complements, and never replaces, the existing read-only
planning commands (dry-run, resume-plan, verify-artifacts) and the
existing write-capable production commands (assemble-final-video,
derive-text-overlays, render-text-overlays, verify-and-advance). It never
calls any of them.

build_final_assembly_plan() has no filesystem I/O, no database I/O, no
subprocess call, no FFmpeg/ffprobe, no provider/LLM import, no network
call, and never mutates any input or writes a manifest/artifact/lifecycle
transition/job record — it only reasons over a ProjectRecord, a
VideoManifest, and an already-fetched sequence of ArtifactRecord objects
the caller supplies (src.cli.cmd_plan_final_assembly owns all I/O, exactly
mirroring every other Phase 2 planning module's own split, e.g.
src.core.dry_run_orchestrator.build_dry_run_report() and
src.core.scene_timing_finalizer.finalize_scene_timing())."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from src.models.artifact import ArtifactRecord
from src.models.manifest import VideoManifest
from src.models.project_state import ProjectRecord

_RENDER_RELATIVE_PATH = "render/final.mp4"
_OVERLAY_RENDER_RELATIVE_PATH = "overlay_render/final.mp4"
_OVERLAY_RENDER_SOURCE = "text-overlay-renderer-v1"

_LIFECYCLE_NOTE = (
    "render/final.mp4 remains the current lifecycle-recognized project output; "
    "overlay_render/final.mp4 is an optional enhanced viewer-facing artifact outside "
    "the current ProjectStage lifecycle model."
)


class FinalAssemblyPlannerError(Exception):
    """Raised only for an invalid caller input or an impossible internal
    contract — never merely because a prerequisite is absent (that is
    represented in FinalAssemblyPlan.blocked_reasons/next_safe_local_command
    instead). Covers: manifest/project identity mismatch, an artifact with
    an invalid kind/scene association/relative_path/metadata passed into
    the planner, or duplicate artifact identities that make project state
    ambiguous (more than one "render" or "overlay_render" artifact)."""


@dataclass(frozen=True)
class FinalAssemblyPlan:
    """The one typed, immutable result build_final_assembly_plan() always
    returns on success. Every field is a scalar or a tuple of scalars."""

    project_id: str
    manifest_project_id_matches: bool
    manifest_fingerprint_matches: bool
    render_artifact_id: str | None
    render_ready: bool
    render_duration_seconds: float | None
    overlay_count: int
    manifest_has_explicit_overlays: bool
    overlay_render_artifact_id: str | None
    overlay_render_ready: bool
    final_viewer_output_kind: str | None
    final_viewer_output_relative_path: str | None
    next_safe_local_command: str | None
    next_command_requires_explicit_paths: bool
    blocked_reasons: tuple[str, ...]
    notes: tuple[str, ...]


def _validate_duration(record: ArtifactRecord, label: str) -> float:
    if "duration_seconds" not in record.metadata:
        raise FinalAssemblyPlannerError(
            f"{label} artifact {record.artifact_id!r} metadata is missing a 'duration_seconds' value"
        )
    value = record.metadata["duration_seconds"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FinalAssemblyPlannerError(
            f"{label} artifact {record.artifact_id!r} has a non-numeric duration_seconds value"
        )
    if not math.isfinite(value):
        raise FinalAssemblyPlannerError(
            f"{label} artifact {record.artifact_id!r} has a non-finite duration_seconds value"
        )
    if value <= 0:
        raise FinalAssemblyPlannerError(
            f"{label} artifact {record.artifact_id!r} has a non-positive duration_seconds value"
        )
    return float(value)


def _validate_render_artifact(record: ArtifactRecord, project: ProjectRecord) -> float:
    if record.scene_id is not None:
        raise FinalAssemblyPlannerError(
            f"render artifact {record.artifact_id!r} has a non-null scene_id — 'render' must be "
            "project-level"
        )
    if record.project_id != project.project_id:
        raise FinalAssemblyPlannerError(
            f"render artifact {record.artifact_id!r} project_id {record.project_id!r} does not "
            f"match project {project.project_id!r}"
        )
    if record.relative_path != _RENDER_RELATIVE_PATH:
        raise FinalAssemblyPlannerError(
            f"render artifact {record.artifact_id!r} relative_path must be exactly "
            f"{_RENDER_RELATIVE_PATH!r}, got {record.relative_path!r}"
        )
    return _validate_duration(record, "render")


def _validate_overlay_render_artifact(
    record: ArtifactRecord,
    project: ProjectRecord,
    render_artifact: ArtifactRecord,
    manifest: VideoManifest,
    overlay_count: int,
) -> None:
    if record.scene_id is not None:
        raise FinalAssemblyPlannerError(
            f"overlay_render artifact {record.artifact_id!r} has a non-null scene_id — "
            "'overlay_render' must be project-level"
        )
    if record.project_id != project.project_id:
        raise FinalAssemblyPlannerError(
            f"overlay_render artifact {record.artifact_id!r} project_id {record.project_id!r} "
            f"does not match project {project.project_id!r}"
        )
    if record.relative_path != _OVERLAY_RENDER_RELATIVE_PATH:
        raise FinalAssemblyPlannerError(
            f"overlay_render artifact {record.artifact_id!r} relative_path must be exactly "
            f"{_OVERLAY_RENDER_RELATIVE_PATH!r}, got {record.relative_path!r}"
        )
    _validate_duration(record, "overlay_render")

    if record.metadata.get("source") != _OVERLAY_RENDER_SOURCE:
        raise FinalAssemblyPlannerError(
            f"overlay_render artifact {record.artifact_id!r} metadata['source'] must be exactly "
            f"{_OVERLAY_RENDER_SOURCE!r}"
        )
    if record.metadata.get("source_render_artifact_id") != render_artifact.artifact_id:
        raise FinalAssemblyPlannerError(
            f"overlay_render artifact {record.artifact_id!r} metadata['source_render_artifact_id'] "
            f"does not match the resolved render artifact {render_artifact.artifact_id!r}"
        )
    if record.metadata.get("source_render_sha256") != render_artifact.sha256_checksum:
        raise FinalAssemblyPlannerError(
            f"overlay_render artifact {record.artifact_id!r} metadata['source_render_sha256'] "
            "does not match the resolved render artifact's checksum"
        )
    if record.metadata.get("source_render_relative_path") != _RENDER_RELATIVE_PATH:
        raise FinalAssemblyPlannerError(
            f"overlay_render artifact {record.artifact_id!r} metadata['source_render_relative_path'] "
            f"must be exactly {_RENDER_RELATIVE_PATH!r}"
        )
    if record.metadata.get("manifest_fingerprint") != manifest.source_fingerprint:
        raise FinalAssemblyPlannerError(
            f"overlay_render artifact {record.artifact_id!r} metadata['manifest_fingerprint'] "
            "does not match the supplied manifest's source_fingerprint"
        )
    if record.metadata.get("overlay_count") != overlay_count:
        raise FinalAssemblyPlannerError(
            f"overlay_render artifact {record.artifact_id!r} metadata['overlay_count'] does not "
            f"match the supplied manifest's computed overlay count ({overlay_count})"
        )


def _count_explicit_overlays(manifest: VideoManifest) -> int:
    """Deterministic order: manifest scene order, then each scene's own
    text_overlays tuple order — same ordering
    src.core.overlay_derivation._flatten_overlays() already establishes,
    reused here for consistency. Only the count is needed by this
    planner."""
    return sum(len(scene.text_overlays) for scene in manifest.scene_plan.scenes)


def build_final_assembly_plan(
    project: ProjectRecord,
    manifest: VideoManifest,
    artifacts: Sequence[ArtifactRecord],
) -> FinalAssemblyPlan:
    """Build a read-only readiness/next-step report for the render ->
    overlay-derivation -> overlay-rendering tail of the pipeline. Raises
    FinalAssemblyPlannerError, with nothing returned, for an invalid
    caller input or an impossible internal contract — never merely
    because a prerequisite is absent (that is reported via
    blocked_reasons/next_safe_local_command instead). Never mutates any
    argument; never performs any I/O."""
    if manifest.project_id != project.project_id:
        raise FinalAssemblyPlannerError(
            f"manifest project_id {manifest.project_id!r} does not match project "
            f"{project.project_id!r}"
        )
    if manifest.source_fingerprint != project.manifest_fingerprint:
        raise FinalAssemblyPlannerError(
            "manifest source_fingerprint does not match the project registry's recorded "
            "fingerprint"
        )

    render_artifacts = [a for a in artifacts if a.kind == "render"]
    overlay_render_artifacts = [a for a in artifacts if a.kind == "overlay_render"]
    overlay_count = _count_explicit_overlays(manifest)
    manifest_has_explicit_overlays = overlay_count > 0

    if len(render_artifacts) > 1:
        raise FinalAssemblyPlannerError(
            f"project {project.project_id!r} has {len(render_artifacts)} 'render' artifacts "
            "(expected at most 1)"
        )

    if len(render_artifacts) == 0:
        return FinalAssemblyPlan(
            project_id=project.project_id,
            manifest_project_id_matches=True,
            manifest_fingerprint_matches=True,
            render_artifact_id=None,
            render_ready=False,
            render_duration_seconds=None,
            overlay_count=overlay_count,
            manifest_has_explicit_overlays=manifest_has_explicit_overlays,
            overlay_render_artifact_id=None,
            overlay_render_ready=False,
            final_viewer_output_kind=None,
            final_viewer_output_relative_path=None,
            next_safe_local_command=(
                f"assemble-final-video {project.project_id} --manifest PATH --output PATH"
            ),
            next_command_requires_explicit_paths=True,
            blocked_reasons=("no 'render' artifact is registered for this project yet",),
            notes=(),
        )

    render_artifact = render_artifacts[0]
    render_duration = _validate_render_artifact(render_artifact, project)

    if len(overlay_render_artifacts) > 1:
        raise FinalAssemblyPlannerError(
            f"project {project.project_id!r} has {len(overlay_render_artifacts)} 'overlay_render' "
            "artifacts (expected at most 1)"
        )

    if len(overlay_render_artifacts) == 1:
        overlay_render_artifact = overlay_render_artifacts[0]
        _validate_overlay_render_artifact(
            overlay_render_artifact, project, render_artifact, manifest, overlay_count
        )
        return FinalAssemblyPlan(
            project_id=project.project_id,
            manifest_project_id_matches=True,
            manifest_fingerprint_matches=True,
            render_artifact_id=render_artifact.artifact_id,
            render_ready=True,
            render_duration_seconds=render_duration,
            overlay_count=overlay_count,
            manifest_has_explicit_overlays=manifest_has_explicit_overlays,
            overlay_render_artifact_id=overlay_render_artifact.artifact_id,
            overlay_render_ready=True,
            final_viewer_output_kind="overlay_render",
            final_viewer_output_relative_path=_OVERLAY_RENDER_RELATIVE_PATH,
            next_safe_local_command=None,
            next_command_requires_explicit_paths=False,
            blocked_reasons=(),
            notes=(
                "overlay_render/final.mp4 is the enhanced viewer-facing output for this project.",
                _LIFECYCLE_NOTE,
            ),
        )

    # Exactly one valid render, zero overlay_render artifacts.
    if manifest_has_explicit_overlays:
        next_command = f"render-text-overlays {project.project_id} --manifest PATH --output PATH"
        blocked_reasons = ("overlay_render has not been produced from the explicit-overlay manifest",)
        notes = (
            "render/final.mp4 is currently usable as the lifecycle-recognized base output, but "
            "overlay_render/final.mp4 will become the enhanced viewer-facing output after "
            "rendering succeeds.",
            _LIFECYCLE_NOTE,
        )
    else:
        next_command = (
            f"derive-text-overlays {project.project_id} --manifest INPUT_PATH --output OUTPUT_PATH"
        )
        blocked_reasons = ("the supplied manifest has no explicit text overlays",)
        notes = (
            "the planner never guesses the derived-manifest output path; after derivation, rerun "
            "plan-final-assembly with that explicit derived manifest path.",
            _LIFECYCLE_NOTE,
        )

    return FinalAssemblyPlan(
        project_id=project.project_id,
        manifest_project_id_matches=True,
        manifest_fingerprint_matches=True,
        render_artifact_id=render_artifact.artifact_id,
        render_ready=True,
        render_duration_seconds=render_duration,
        overlay_count=overlay_count,
        manifest_has_explicit_overlays=manifest_has_explicit_overlays,
        overlay_render_artifact_id=None,
        overlay_render_ready=False,
        final_viewer_output_kind="render",
        final_viewer_output_relative_path=_RENDER_RELATIVE_PATH,
        next_safe_local_command=next_command,
        next_command_requires_explicit_paths=True,
        blocked_reasons=blocked_reasons,
        notes=notes,
    )
