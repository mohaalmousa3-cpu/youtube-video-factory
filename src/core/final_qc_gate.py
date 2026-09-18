"""FINAL QC GATE V1: local-only, read-only verification of a project's real
on-disk final video output(s) against their registered artifact records.

Given an already-loaded ProjectRecord, VideoManifest, and the project's
registered ArtifactRecords, this module determines whether the project's
render (and, if present, overlay_render) artifact is trustworthy: its file
actually exists, matches its registered byte_size/sha256 (reusing
src.core.artifact_verifier.verify_artifact() — this module never
reimplements checksum or byte-size comparison itself), and — the one
genuinely new capability this phase adds — its real, freshly-probed video/
audio streams and duration match what was recorded at registration time.

This module never writes to SQLite, never registers an artifact, never
writes or mutates a manifest, never transitions a project's lifecycle stage,
and never writes the report file itself (see src/cli.py's
cmd_verify_final_output for that — the same "core module is pure/read-only,
CLI owns all I/O" split every other Phase 2 module in this codebase already
uses). The only side effect anywhere in this module is two local, read-only
ffprobe subprocess calls per artifact being checked.

registering that produced-elsewhere qc_report belongs to the already-merged
src.core.qc_report_artifact_registrar (Phase 2H); reaching qc_passed belongs
to the already-merged src.core.verified_transition_service. Neither is
called, imported for its logic, or reimplemented here — this module only
produces the FinalQcReport whose JSON serialization already satisfies
qc_report_artifact_registrar's own structural requirement (a top-level
object with a boolean "passed" field), so it can be handed to
`register-qc-report-artifact --file` unchanged.

No provider, no paid service, no network call, no LLM, no Qwen/Groq/
TokenRouter/Kokoro/Gemini/Flow/Veo/Rhubarb/Manim call anywhere in this
module.

An ORDINARY QC failure (a missing render, a checksum mismatch, a missing
stream, a duration drift, invalid lineage metadata, ...) is never raised —
it is represented as one or more failed QcCheckResult rows and reflected in
FinalQcReport.passed=False plus FinalQcReport.blocking_reasons. This module
raises only for: an invalid project/manifest pairing
(ProjectManifestMismatchError), an impossible artifact topology such as more
than one registered render or overlay_render artifact
(ArtifactTopologyError), or a local ffprobe call that itself fails to
execute or produces something QC cannot safely parse (FinalQcProbeError) —
in every one of these cases QC is genuinely unable to determine reality, so
there is nothing honest a returned FinalQcReport could report."""
from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from src.core.artifact_verifier import verify_artifact
from src.core.path_safety import resolve_under_project_dir
from src.models.artifact import ArtifactRecord
from src.models.manifest import VideoManifest
from src.models.project_state import ProjectRecord
from src.utils.config import get_settings

_RENDER_RELATIVE_PATH = "render/final.mp4"
_OVERLAY_RENDER_RELATIVE_PATH = "overlay_render/final.mp4"
_OVERLAY_SOURCE = "text-overlay-renderer-v1"
_DURATION_TOLERANCE_SECONDS = 0.25
_PROBE_TIMEOUT_SECONDS = 30.0


class FinalQcGateError(Exception):
    """Base class for every reason verify_final_output() cannot produce a
    FinalQcReport at all — never raised for an ordinary failed QC check
    (missing file, checksum mismatch, bad stream, duration drift, invalid
    lineage, ...), which is always represented in a returned
    FinalQcReport(passed=False, ...) instead. Message text is always a
    short, sanitized string — never raw ffprobe/SQLite output."""


class ProjectManifestMismatchError(FinalQcGateError):
    """The supplied manifest's project_id or source_fingerprint does not
    match the supplied ProjectRecord — a caller bug, not a QC finding."""


class ArtifactTopologyError(FinalQcGateError):
    """More than one registered 'render' or 'overlay_render' artifact
    exists for this project — an impossible state QC cannot reason about
    (which one is authoritative?), not an ordinary QC finding."""


class FinalQcProbeError(FinalQcGateError):
    """A local ffprobe stream-type or duration probe could not be
    completed and safely parsed: a missing/unusable ffprobe binary, a
    timeout, a non-zero exit, empty output, malformed JSON, or a JSON
    structure that isn't the expected shape. Never raised merely because a
    measured value fails an ordinary requirement (e.g. a genuinely
    zero-duration or non-finite-but-parseable value) — that is instead a
    failed QcCheckResult; see this module's own docstring."""


class ReportOutputConflictError(FinalQcGateError):
    """Raised only by the CLI layer (src/cli.py's cmd_verify_final_output)
    for a --report-output path collision detected before any I/O — this
    core module never writes a report file and never raises this itself.
    Defined here so the CLI can import one small, coherent error hierarchy
    from this module rather than inventing a second one."""


@dataclass(frozen=True)
class QcCheckResult:
    """One named, immutable QC check outcome."""

    check_id: str
    passed: bool
    subject: str
    message: str


@dataclass(frozen=True)
class FinalQcReport:
    """The one typed, immutable result verify_final_output() always
    returns on success (no FinalQcGateError raised). JSON-safe scalar
    fields only; check collections are tuples. Serializing this directly
    to JSON (see cmd_verify_final_output) always yields a top-level object
    with a boolean top-level "passed" field, satisfying
    qc_report_artifact_registrar's own structural requirement unchanged."""

    project_id: str
    passed: bool
    require_overlays: bool
    render_checks: tuple[QcCheckResult, ...]
    overlay_render_checks: tuple[QcCheckResult, ...]
    overlay_render_present: bool
    viewer_facing_output_kind: str | None
    viewer_facing_output_relative_path: str | None
    blocking_reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    generated_at: str


def _check(check_id: str, passed: bool, subject: str, message: str) -> QcCheckResult:
    return QcCheckResult(check_id=check_id, passed=passed, subject=subject, message=message)


def _duration_metadata_ok(metadata: Mapping[str, object]) -> bool:
    value = metadata.get("duration_seconds")
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and value > 0


def _ffprobe_path() -> str:
    ffmpeg_path = Path(get_settings().ffmpeg_path)
    ffprobe_name = "ffprobe.exe" if ffmpeg_path.suffix == ".exe" else "ffprobe"
    return str(ffmpeg_path.with_name(ffprobe_name))


def _probe_stream_types(media_path: Path) -> frozenset[str]:
    """Strict, fail-closed local ffprobe stream-type probe — same shape as
    the small private copies already established in
    src.core.final_video_assembly/src.core.text_overlay_render/
    src.core.render_artifact_registrar (deliberately duplicated here too,
    matching this codebase's own stated convention for these small ffprobe
    helpers, rather than importing a private helper from another module or
    adding one to src/render/ffmpeg_render.py)."""
    ffprobe = _ffprobe_path()
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "stream=codec_type", "-of", "json", str(media_path)],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FinalQcProbeError(f"could not run ffprobe to inspect media streams for {media_path.name!r}") from exc

    if result.returncode != 0:
        raise FinalQcProbeError(f"ffprobe exited with an error while inspecting {media_path.name!r}")
    if not result.stdout or not result.stdout.strip():
        raise FinalQcProbeError(f"ffprobe produced no output while inspecting {media_path.name!r}")

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise FinalQcProbeError(f"ffprobe produced malformed JSON while inspecting {media_path.name!r}") from exc

    if not isinstance(payload, dict):
        raise FinalQcProbeError(f"ffprobe produced an unexpected JSON structure while inspecting {media_path.name!r}")

    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        raise FinalQcProbeError(
            f"ffprobe produced no usable stream information while inspecting {media_path.name!r}"
        )

    stream_types: set[str] = set()
    for stream in streams:
        if not isinstance(stream, dict):
            raise FinalQcProbeError(
                f"ffprobe produced an unexpected stream entry while inspecting {media_path.name!r}"
            )
        if "codec_type" not in stream:
            raise FinalQcProbeError(
                f"ffprobe produced a stream entry with no codec_type while inspecting {media_path.name!r}"
            )
        codec_type = stream["codec_type"]
        if not isinstance(codec_type, str) or not codec_type:
            raise FinalQcProbeError(
                f"ffprobe produced a stream entry with an invalid codec_type while inspecting {media_path.name!r}"
            )
        stream_types.add(codec_type)
    return frozenset(stream_types)


def _probe_duration_seconds(media_path: Path) -> float:
    """Strict, fail-closed local ffprobe duration probe. Raises
    FinalQcProbeError only for a structural failure (ffprobe itself
    unusable, or its output not even parseable as a duration number) —
    a parsed-but-non-finite or parsed-but-non-positive value is returned
    normally, since "duration finite"/"duration > 0" are ordinary QC
    requirements the caller evaluates, not probe-level failures; see this
    module's own docstring."""
    ffprobe = _ffprobe_path()
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(media_path)],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FinalQcProbeError(f"could not run ffprobe to measure duration for {media_path.name!r}") from exc

    if result.returncode != 0:
        raise FinalQcProbeError(f"ffprobe exited with an error while measuring {media_path.name!r}")
    if not result.stdout or not result.stdout.strip():
        raise FinalQcProbeError(f"ffprobe produced no output while measuring {media_path.name!r}")

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise FinalQcProbeError(f"ffprobe produced malformed JSON while measuring {media_path.name!r}") from exc

    if not isinstance(payload, dict):
        raise FinalQcProbeError(f"ffprobe produced an unexpected JSON structure while measuring {media_path.name!r}")

    fmt = payload.get("format")
    if not isinstance(fmt, dict) or "duration" not in fmt:
        raise FinalQcProbeError(f"ffprobe produced no usable duration information while measuring {media_path.name!r}")

    raw = fmt["duration"]
    if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
        raise FinalQcProbeError(f"ffprobe produced an invalid duration value while measuring {media_path.name!r}")
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise FinalQcProbeError(f"ffprobe produced an unparseable duration value while measuring {media_path.name!r}") from exc


def _media_checks(media_path: Path, recorded_duration: float, subject: str) -> tuple[list[QcCheckResult], float | None]:
    """Run the fresh stream+duration checks (STEP 6) for one already
    file-integrity-verified artifact. May raise FinalQcProbeError, which
    propagates uncaught — QC is genuinely unable to determine reality in
    that case. `recorded_duration` is assumed already validated numeric/
    finite/positive by the caller's topology gate."""
    stream_types = _probe_stream_types(media_path)
    has_video = "video" in stream_types
    has_audio = "audio" in stream_types

    duration = _probe_duration_seconds(media_path)
    finite_ok = math.isfinite(duration)
    positive_ok = finite_ok and duration > 0
    duration_match = finite_ok and abs(duration - recorded_duration) <= _DURATION_TOLERANCE_SECONDS

    checks = [
        _check(
            f"{subject}_video_stream_present", has_video, subject,
            f"{subject} has a video stream" if has_video else f"{subject} is missing a video stream",
        ),
        _check(
            f"{subject}_audio_stream_present", has_audio, subject,
            f"{subject} has an audio stream" if has_audio else f"{subject} is missing an audio stream",
        ),
        _check(
            f"{subject}_duration_finite", finite_ok, subject,
            f"{subject} measured duration is finite" if finite_ok else f"{subject} measured duration is not finite",
        ),
        _check(
            f"{subject}_duration_positive", positive_ok, subject,
            f"{subject} measured duration is positive" if positive_ok
            else f"{subject} measured duration is not positive",
        ),
        _check(
            f"{subject}_duration_matches_metadata", duration_match, subject,
            (
                f"{subject} measured duration {duration:.3f}s matches recorded {recorded_duration:.3f}s "
                f"within {_DURATION_TOLERANCE_SECONDS}s"
            ) if duration_match else (
                f"{subject} measured duration {duration:.3f}s does not match recorded "
                f"{recorded_duration:.3f}s within {_DURATION_TOLERANCE_SECONDS}s"
            ),
        ),
    ]
    return checks, (duration if finite_ok else None)


def _render_topology_checks(artifact: ArtifactRecord, project: ProjectRecord) -> list[QcCheckResult]:
    scene_id_ok = artifact.scene_id is None
    project_id_ok = artifact.project_id == project.project_id
    path_ok = artifact.relative_path == _RENDER_RELATIVE_PATH
    duration_ok = _duration_metadata_ok(artifact.metadata)
    return [
        _check(
            "render_scene_id_none", scene_id_ok, "render",
            "render artifact scene_id is None" if scene_id_ok
            else f"render artifact scene_id must be None, got {artifact.scene_id!r}",
        ),
        _check(
            "render_project_id_matches", project_id_ok, "render",
            "render artifact project_id matches the project" if project_id_ok
            else f"render artifact project_id {artifact.project_id!r} does not match project {project.project_id!r}",
        ),
        _check(
            "render_relative_path", path_ok, "render",
            f"render artifact relative_path is {_RENDER_RELATIVE_PATH!r}" if path_ok
            else f"render artifact relative_path must be {_RENDER_RELATIVE_PATH!r}, got {artifact.relative_path!r}",
        ),
        _check(
            "render_metadata_duration", duration_ok, "render",
            "render artifact metadata duration_seconds is valid" if duration_ok
            else "render artifact metadata duration_seconds is missing or invalid",
        ),
    ]


def _overlay_topology_checks(
    artifact: ArtifactRecord,
    project: ProjectRecord,
    render_artifact: ArtifactRecord,
    manifest: VideoManifest,
    overlay_count: int,
) -> list[QcCheckResult]:
    kind_ok = artifact.kind == "overlay_render"
    scene_id_ok = artifact.scene_id is None
    project_id_ok = artifact.project_id == project.project_id
    path_ok = artifact.relative_path == _OVERLAY_RENDER_RELATIVE_PATH
    duration_ok = _duration_metadata_ok(artifact.metadata)
    source_ok = artifact.metadata.get("source") == _OVERLAY_SOURCE
    artifact_id_ok = artifact.metadata.get("source_render_artifact_id") == render_artifact.artifact_id
    sha_ok = artifact.metadata.get("source_render_sha256") == render_artifact.sha256_checksum
    relpath_ok = artifact.metadata.get("source_render_relative_path") == _RENDER_RELATIVE_PATH
    fingerprint_ok = artifact.metadata.get("manifest_fingerprint") == manifest.source_fingerprint
    count_ok = artifact.metadata.get("overlay_count") == overlay_count
    return [
        _check(
            "overlay_kind", kind_ok, "overlay_render",
            "overlay_render artifact kind is 'overlay_render'" if kind_ok
            else f"overlay_render artifact kind must be 'overlay_render', got {artifact.kind!r}",
        ),
        _check(
            "overlay_scene_id_none", scene_id_ok, "overlay_render",
            "overlay_render artifact scene_id is None" if scene_id_ok
            else f"overlay_render artifact scene_id must be None, got {artifact.scene_id!r}",
        ),
        _check(
            "overlay_project_id_matches", project_id_ok, "overlay_render",
            "overlay_render artifact project_id matches the project" if project_id_ok
            else f"overlay_render artifact project_id {artifact.project_id!r} does not match project "
            f"{project.project_id!r}",
        ),
        _check(
            "overlay_relative_path", path_ok, "overlay_render",
            f"overlay_render artifact relative_path is {_OVERLAY_RENDER_RELATIVE_PATH!r}" if path_ok
            else f"overlay_render artifact relative_path must be {_OVERLAY_RENDER_RELATIVE_PATH!r}, got "
            f"{artifact.relative_path!r}",
        ),
        _check(
            "overlay_metadata_duration", duration_ok, "overlay_render",
            "overlay_render artifact metadata duration_seconds is valid" if duration_ok
            else "overlay_render artifact metadata duration_seconds is missing or invalid",
        ),
        _check(
            "overlay_source", source_ok, "overlay_render",
            f"overlay_render artifact metadata source is {_OVERLAY_SOURCE!r}" if source_ok
            else f"overlay_render artifact metadata source must be {_OVERLAY_SOURCE!r}, got "
            f"{artifact.metadata.get('source')!r}",
        ),
        _check(
            "overlay_source_render_artifact_id", artifact_id_ok, "overlay_render",
            "overlay_render artifact metadata source_render_artifact_id matches the render artifact"
            if artifact_id_ok else
            "overlay_render artifact metadata source_render_artifact_id does not match the render artifact",
        ),
        _check(
            "overlay_source_render_sha256", sha_ok, "overlay_render",
            "overlay_render artifact metadata source_render_sha256 matches the render artifact's checksum"
            if sha_ok else
            "overlay_render artifact metadata source_render_sha256 does not match the render artifact's checksum",
        ),
        _check(
            "overlay_source_render_relative_path", relpath_ok, "overlay_render",
            f"overlay_render artifact metadata source_render_relative_path is {_RENDER_RELATIVE_PATH!r}"
            if relpath_ok else
            f"overlay_render artifact metadata source_render_relative_path must be {_RENDER_RELATIVE_PATH!r}",
        ),
        _check(
            "overlay_manifest_fingerprint", fingerprint_ok, "overlay_render",
            "overlay_render artifact metadata manifest_fingerprint matches the supplied manifest"
            if fingerprint_ok else
            "overlay_render artifact metadata manifest_fingerprint does not match the supplied manifest",
        ),
        _check(
            "overlay_count_matches", count_ok, "overlay_render",
            f"overlay_render artifact metadata overlay_count matches the supplied manifest's explicit "
            f"overlay count ({overlay_count})" if count_ok else
            f"overlay_render artifact metadata overlay_count does not match the supplied manifest's explicit "
            f"overlay count ({overlay_count})",
        ),
    ]


def _resolve_artifact_path(project_dir_resolved: Path, artifact: ArtifactRecord) -> Path:
    resolved = resolve_under_project_dir(project_dir_resolved, artifact.relative_path)
    if resolved is None:
        # Unreachable in practice: artifact_verifier.verify_artifact() already
        # performed this same resolution and would have failed first (see the
        # render_ok/overlay_ok gating in verify_final_output) — handled
        # defensively anyway, matching this codebase's habit of re-checking an
        # invariant a caller already guarantees rather than trusting it silently.
        raise ArtifactTopologyError(
            f"artifact {artifact.artifact_id!r} relative_path {artifact.relative_path!r} is unsafe"
        )
    return resolved


def verify_final_output(
    *,
    project: ProjectRecord,
    manifest: VideoManifest,
    artifacts: Sequence[ArtifactRecord],
    project_dir: Path,
    require_overlays: bool,
) -> FinalQcReport:
    """Build a read-only FinalQcReport for `project`'s real on-disk render
    (and, if present, overlay_render) output. No database I/O, no provider
    import, no network call, no artifact registration, no lifecycle
    mutation, no manifest write, no report file write — the only I/O this
    function performs is reading local artifact files (via
    artifact_verifier.verify_artifact()) and two local ffprobe process
    calls per artifact whose file-integrity check already passed.
    Deterministic for the same DB records, manifest, files, and FFmpeg
    output (generated_at aside, which is always the real current time).
    Raises FinalQcGateError subclasses only for the caller-input/topology/
    probe failures documented on this module; every ordinary QC failure is
    represented in the returned FinalQcReport instead."""
    if manifest.project_id != project.project_id:
        raise ProjectManifestMismatchError(
            f"manifest project_id {manifest.project_id!r} does not match project {project.project_id!r}"
        )
    if manifest.source_fingerprint != project.manifest_fingerprint:
        raise ProjectManifestMismatchError(
            "manifest source_fingerprint does not match the project registry's recorded fingerprint"
        )

    project_dir_resolved = Path(project_dir).resolve()

    render_artifacts = [a for a in artifacts if a.kind == "render"]
    overlay_artifacts = [a for a in artifacts if a.kind == "overlay_render"]

    if len(render_artifacts) > 1:
        raise ArtifactTopologyError(
            f"project {project.project_id!r} has {len(render_artifacts)} 'render' artifacts (expected at most 1)"
        )
    if len(overlay_artifacts) > 1:
        raise ArtifactTopologyError(
            f"project {project.project_id!r} has {len(overlay_artifacts)} 'overlay_render' artifacts "
            "(expected at most 1)"
        )

    render_artifact = render_artifacts[0] if render_artifacts else None
    overlay_artifact = overlay_artifacts[0] if overlay_artifacts else None
    overlay_count = sum(len(scene.text_overlays) for scene in manifest.scene_plan.scenes)

    render_checks: list[QcCheckResult] = []
    overlay_render_checks: list[QcCheckResult] = []
    warnings: list[str] = []

    render_ok = False
    render_fresh_duration: float | None = None

    if render_artifact is None:
        render_checks.append(
            _check("render_registered", False, "render", "no render artifact is registered for this project")
        )
    else:
        render_checks.append(_check("render_registered", True, "render", "a render artifact is registered"))
        topology_checks = _render_topology_checks(render_artifact, project)
        render_checks.extend(topology_checks)
        if all(c.passed for c in topology_checks):
            verification = verify_artifact(project_dir_resolved, render_artifact, manifest)
            render_checks.append(
                _check(
                    "render_file_integrity", verification.passed, "render",
                    "render file exists, is a regular file, and matches its registered byte_size/sha256"
                    if verification.passed else "; ".join(verification.reasons),
                )
            )
            if verification.passed:
                render_path = _resolve_artifact_path(project_dir_resolved, render_artifact)
                media_checks, render_fresh_duration = _media_checks(
                    render_path, render_artifact.metadata["duration_seconds"], "render"
                )
                render_checks.extend(media_checks)
                render_ok = all(c.passed for c in media_checks)

    overlay_render_present = overlay_artifact is not None

    if not render_ok:
        if overlay_artifact is not None:
            reason = (
                "overlay_render cannot be validated because no render artifact is registered for this project"
                if render_artifact is None else
                "overlay_render cannot be validated because the render artifact failed verification"
            )
            overlay_render_checks.append(_check("overlay_skipped", False, "overlay_render", reason))
        viewer_facing_kind: str | None = None
        viewer_facing_path: str | None = None
        passed = False
    elif overlay_artifact is None:
        if require_overlays:
            overlay_render_checks.append(
                _check("overlay_render_required", False, "overlay_render", "overlay_render is required but missing")
            )
            viewer_facing_kind = None
            viewer_facing_path = None
            passed = False
        else:
            warnings.append("overlay_render is absent; render/final.mp4 is the base viewer-facing output")
            viewer_facing_kind = "render"
            viewer_facing_path = _RENDER_RELATIVE_PATH
            passed = True
    else:
        topology_checks = _overlay_topology_checks(overlay_artifact, project, render_artifact, manifest, overlay_count)
        overlay_render_checks.extend(topology_checks)
        overlay_ok = all(c.passed for c in topology_checks)
        if overlay_ok:
            verification = verify_artifact(project_dir_resolved, overlay_artifact, manifest)
            overlay_render_checks.append(
                _check(
                    "overlay_file_integrity", verification.passed, "overlay_render",
                    "overlay_render file exists, is a regular file, and matches its registered byte_size/sha256"
                    if verification.passed else "; ".join(verification.reasons),
                )
            )
            overlay_ok = verification.passed
            if overlay_ok:
                overlay_path = _resolve_artifact_path(project_dir_resolved, overlay_artifact)
                media_checks, overlay_fresh_duration = _media_checks(
                    overlay_path, overlay_artifact.metadata["duration_seconds"], "overlay_render"
                )
                overlay_render_checks.extend(media_checks)
                overlay_ok = all(c.passed for c in media_checks)
                if overlay_ok and render_fresh_duration is not None and overlay_fresh_duration is not None:
                    consistent = abs(render_fresh_duration - overlay_fresh_duration) <= _DURATION_TOLERANCE_SECONDS
                    overlay_render_checks.append(
                        _check(
                            "render_overlay_duration_consistency", consistent, "overlay_render",
                            (
                                f"render duration {render_fresh_duration:.3f}s and overlay_render duration "
                                f"{overlay_fresh_duration:.3f}s are consistent within {_DURATION_TOLERANCE_SECONDS}s"
                            ) if consistent else (
                                f"render duration {render_fresh_duration:.3f}s and overlay_render duration "
                                f"{overlay_fresh_duration:.3f}s differ by more than {_DURATION_TOLERANCE_SECONDS}s"
                            ),
                        )
                    )
                    overlay_ok = consistent

        if overlay_ok:
            viewer_facing_kind = "overlay_render"
            viewer_facing_path = _OVERLAY_RENDER_RELATIVE_PATH
            passed = True
        else:
            viewer_facing_kind = None
            viewer_facing_path = None
            passed = False

    blocking_reasons = tuple(
        c.message for c in (*render_checks, *overlay_render_checks) if not c.passed
    )

    return FinalQcReport(
        project_id=project.project_id,
        passed=passed,
        require_overlays=require_overlays,
        render_checks=tuple(render_checks),
        overlay_render_checks=tuple(overlay_render_checks),
        overlay_render_present=overlay_render_present,
        viewer_facing_output_kind=viewer_facing_kind,
        viewer_facing_output_relative_path=viewer_facing_path,
        blocking_reasons=blocking_reasons,
        warnings=tuple(warnings),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
