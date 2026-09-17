"""FINAL VIDEO ASSEMBLY V1: local-only, generation-and-registration final
assembly. See src/cli.py's cmd_assemble_final_video for the CLI command.

Given a project_id and an already-fully-produced manifest (every scene
already has exactly one registered "audio" and one registered "animation"
artifact), this module resolves those per-scene artifacts in manifest
order, mux's each scene's animation clip with its registered audio (only
when the animation clip has no audio stream of its own — see below),
concatenates all prepared scene clips into one final MP4, validates it,
atomically places it at --output, and registers it as this project's
canonical, project-level "render" artifact via the existing, unmodified
src.core.render_artifact_registrar.register_render_artifact() — reusing
that exact artifact type/pattern rather than inventing a new one, since it
already IS the "one final video per project" contract this phase needs.

This phase does NOT implement story/scene generation, image generation,
text-overlay derivation/rendering, lip-sync, mouth animation, limb
animation, general end-to-end orchestration, background music, publishing,
or Manual Flow automation. Local FFmpeg/ffprobe only — no provider, no
network call anywhere in this module.

Deliberate convention departure (see the inspection note in this phase's
approval message): every other src/core/ module is 100% database-free,
with the CLI doing all SQLite access before/after calling it. That
convention does not cleanly support this phase's required three-part
lifecycle (a short DB phase, then a long-running FFmpeg phase with NO
connection open, then a second short DB phase) around one long-running
external process, so assemble_final_video() opens and closes its own two
short-lived connections internally, via the same parameterless
get_readonly_connection()/get_connection() every CLI command already uses
— never a connection injected by the caller, and never held open across
any FFmpeg/ffprobe subprocess call.

No function in src/render/ffmpeg_render.py accepts an ffmpeg_path/
ffprobe_path parameter anywhere in this codebase — every existing
generation module resolves the binary via src.utils.config.get_settings()
internally, with zero exceptions. This module follows that same
convention rather than threading an override parameter through.

Three-part lifecycle:
  A. Open a short-lived read-only connection: verify the project, load and
     validate the manifest, resolve exactly one audio + one animation
     artifact per scene (manifest order, never sorted), reject missing/
     ambiguous artifacts, confirm every resolved artifact file exists,
     confirm --output does not conflict with any source path or an
     existing file, confirm no "render" artifact is already registered
     for this project. Capture every immutable value FFmpeg work needs.
     Close the connection.
  B. FFmpeg-only work, no database connection open at all: per scene,
     inspect the animation clip's audio-stream presence (never mux twice),
     mux where needed, concatenate every prepared clip in manifest order,
     validate the concatenated result, atomically replace --output only
     after that validation passes.
  C. Open a short-lived read-write connection: reconfirm nothing else
     registered a "render" artifact for this project in the meantime,
     register the successful --output file via the existing
     register_render_artifact(), close the connection. A registration
     failure at this point removes the just-placed --output file (the
     registrar's own internal project-relative copy already cleans up
     after itself on failure) — this call's own output is the only file
     THIS module is responsible for on that path.

Cleanup proof: check "output_path does not already exist" in phase A
happens before this module creates anything at all, so any file found at
output_path from that point forward was necessarily written by THIS SAME
call — removing it on failure can never delete a pre-existing caller
file. Every temporary scene clip and the temporary final output live under
one tempfile.mkdtemp() directory created on the same filesystem as
output_path (for a true atomic os.replace()), removed via shutil.rmtree()
in a `finally` regardless of outcome. Source artifacts (the registered
audio/animation files) are opened read-only and never written, moved, or
deleted anywhere in this module.

Domain errors: every ordinary rejection this module can produce is one of
the FinalVideoAssemblyError subclasses below — the CLI only ever needs to
catch that one base type. Message text never includes raw ffmpeg/ffprobe
subprocess output; wrapped ProviderError/ManifestStoreError/OSError
instances are always chained via `from exc` (preserved as __cause__), not
echoed verbatim."""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from src.core.manifest_store import ManifestStoreError, load_manifest
from src.core.path_safety import resolve_under_project_dir
from src.core.render_artifact_registrar import (
    PrevalidatedRenderSource,
    RenderArtifactCleanupError,
    RenderSourceMismatchError,
    RenderSourceValidationError,
    prevalidate_render_source,
    register_render_artifact,
)
from src.database.artifact_repository import list_artifacts_by_project
from src.database.db import get_connection, get_readonly_connection
from src.database.project_repository import get_project
from src.models.artifact import ArtifactRecord
from src.models.manifest import VideoManifest
from src.providers.base import ProviderError
from src.render.ffmpeg_render import concat_videos, get_duration_seconds, mux_audio_video
from src.utils.config import get_settings

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_-]")


class FinalVideoAssemblyError(Exception):
    """Base class for every reason assemble_final_video() cannot produce
    --output. The CLI catches only this one type. Message text is always
    a short, sanitized string naming project_id/scene_id/artifact_id/
    counts — never raw subprocess stderr, a manifest's content, or a
    resolved filesystem path beyond what the message needs to be
    actionable."""


class ProjectNotFoundError(FinalVideoAssemblyError):
    """No project is registered with the given project_id."""


class ManifestNotFoundError(FinalVideoAssemblyError):
    """--manifest does not exist or is not a regular file."""


class InvalidManifestError(FinalVideoAssemblyError):
    """--manifest could not be parsed/validated as a VideoManifest, or its
    project_id/source_fingerprint does not match the project registry."""


class EmptyManifestError(FinalVideoAssemblyError):
    """The manifest's scene_plan contains zero scenes."""


class InvalidSceneOrderError(FinalVideoAssemblyError):
    """Defensive only: ScenePlan's own validator already guarantees unique,
    contiguously-sequenced scenes for any manifest that reaches this
    module via load_manifest() — this is unreachable through that path,
    reachable only by calling the internal scene-resolution helper
    directly with a hand-crafted, invariant-violating scene list, the same
    kind of direct-unit-test-only defensive check this codebase already
    uses elsewhere (see src.core.mouth_animation_generation._derive_durations)."""


class MissingSceneArtifactError(FinalVideoAssemblyError):
    """A scene has zero eligible registered audio or animation artifacts."""


class AmbiguousSceneArtifactError(FinalVideoAssemblyError):
    """A scene has more than one eligible registered audio or animation
    artifact — this module never silently picks one."""


class ArtifactFileNotFoundError(FinalVideoAssemblyError):
    """A resolved artifact's recorded relative_path does not exist on disk
    as a regular file."""


class OutputPathConflictError(FinalVideoAssemblyError):
    """--output already exists, is a directory, or resolves to one of the
    scene source artifact paths this call would otherwise read from."""


class ExistingFinalVideoArtifactError(FinalVideoAssemblyError):
    """A "render" artifact is already registered for this project. This
    module never overwrites or replaces an existing registration — reuses
    register_render_artifact()'s own established idempotency policy
    (checksum-match-only) rather than inventing a new one; a rerun that
    would produce different content is rejected here, before any FFmpeg
    work, rather than discovered only after."""


class MediaProbeError(FinalVideoAssemblyError):
    """A local ffprobe stream-type probe could not be completed and
    parsed — the ffprobe binary is missing/unusable, it exited non-zero,
    produced empty or malformed output, or the JSON structure was not the
    expected shape. Never raised to mean "no such stream" — that is a
    fail-open interpretation this module deliberately refuses to make;
    see _probe_stream_types()'s own docstring."""


class FinalVideoCleanupError(FinalVideoAssemblyError):
    """A temporary-directory or orphaned-output cleanup step failed after
    a primary assembly/registration failure. Raised (never silently
    swallowed) so a corrupted or incompletely-cleaned-up state is always
    discoverable — see assemble_final_video()'s own docstring for the
    cleanup contract this protects."""


class SceneMuxError(FinalVideoAssemblyError):
    """Preparing one scene's clip (audio-stream inspection or muxing)
    failed."""


class FinalConcatError(FinalVideoAssemblyError):
    """Concatenating the prepared scene clips failed."""


class InvalidFinalOutputError(FinalVideoAssemblyError):
    """The rendered final file is missing, empty, has an unmeasurable/
    non-finite/non-positive duration, or its duration does not match the
    sum of prepared scene durations within tolerance."""


class FinalArtifactRegistrationError(FinalVideoAssemblyError):
    """Registering the successful --output file as this project's "render"
    artifact failed, after --output was already atomically placed. This
    module removes the just-placed --output file in that case."""


class FinalArtifactCleanupError(FinalVideoAssemblyError):
    """Registering the successful --output file failed AND
    register_render_artifact()'s own strict cleanup of its
    freshly-copied canonical render/final.mp4 copy also failed (this
    module always passes strict_cleanup=True — see the registration call
    site below) — a second, independent orphan on top of the
    registration failure itself, wrapped from the registrar's own
    RenderArtifactCleanupError with its cause preserved. This module's
    own outer cleanup still attempts to remove --output regardless of
    which registration-path exception is in flight; if that removal also
    fails, both orphan paths remain discoverable via exception
    chaining/notes rather than either one going undisclosed — see
    assemble_final_video()'s own docstring."""


@dataclass(frozen=True)
class FinalVideoAssemblyResult:
    """The one typed result assemble_final_video() returns on success."""

    project_id: str
    output_path: Path
    scene_count: int
    measured_duration_seconds: float
    artifact_id: str


@dataclass(frozen=True)
class _SceneSourcePaths:
    scene_id: str
    animation_path: Path
    audio_path: Path


def _sanitize_for_filename(scene_id: str) -> str:
    """scene_id is already constrained to '^scene-[0-9]{2,3}$' by
    ScenePlanItem's own field pattern, so this is defensive rather than
    load-bearing — matches this codebase's habit of re-checking an
    invariant a caller already guarantees rather than trusting it silently."""
    return _SAFE_FILENAME_RE.sub("_", scene_id)


def _duration_tolerance_seconds(scene_count: int) -> float:
    """Documented, evidence-based tolerance for container/timestamp
    rounding only — never large enough to hide a missing scene. 0.25s
    fixed floor (a single dropped/duplicated frame at low fps) plus 0.05s
    per scene (concat-demuxer boundary rounding accumulates per join, the
    same accumulation CLAUDE.md's "known bugs already fixed" section
    documents for mux_audio_video's own tpad fix)."""
    return max(0.25, 0.05 * scene_count)


def _probe_stream_types(media_path: Path) -> frozenset[str]:
    """Strict, fail-closed local ffprobe stream-type probe. Returns the
    set of codec_type values (e.g. {"video"}, {"video", "audio"})
    actually present in `media_path`, proven by a successfully parsed
    ffprobe run. Raises MediaProbeError, chaining the original exception
    as __cause__ where one exists, for EVERY failure mode: an unreadable/
    missing ffprobe binary (OSError/FileNotFoundError), a timeout, a
    non-zero exit, empty output, malformed JSON, or a JSON document that
    is not the expected {"streams": [{"codec_type": ...}, ...]} shape.

    Deliberately the opposite of animation_artifact_registrar.py's own
    _has_video_stream()/render_artifact_registrar.py's own
    _has_video_stream(), both of which return False on any probe
    failure (a defensible fail-open choice there, since both already
    require a prior successful get_duration_seconds() call on the same
    file before ever being invoked). This module cannot make that same
    assumption — an animation artifact's audio-stream presence is a
    yes/no safety gate that decides whether narration gets muxed in or
    skipped, so an unknown probe result must never be silently read as
    "no audio"; it must fail the whole call instead.

    The parse itself is equally strict about STRUCTURE, not just
    transport/JSON-syntax failures: a missing/non-list "streams" key, an
    EMPTY streams list, a non-dict stream entry, a stream entry with no
    "codec_type" key, or a codec_type that is not a non-empty string ALL
    raise MediaProbeError — none of these are silently treated as "this
    stream/file has no such stream type" by returning an empty or partial
    frozenset. Every stream entry must be fully well-formed for this
    function to return normally at all; one malformed entry among
    otherwise-valid ones still raises, never returns a partial result."""
    ffmpeg_path = Path(get_settings().ffmpeg_path)
    ffprobe_name = "ffprobe.exe" if ffmpeg_path.suffix == ".exe" else "ffprobe"
    ffprobe = str(ffmpeg_path.with_name(ffprobe_name))
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "stream=codec_type", "-of", "json", str(media_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MediaProbeError(f"could not run ffprobe to inspect media streams for {media_path.name!r}") from exc

    if result.returncode != 0:
        raise MediaProbeError(f"ffprobe exited with an error while inspecting {media_path.name!r}")
    if not result.stdout or not result.stdout.strip():
        raise MediaProbeError(f"ffprobe produced no output while inspecting {media_path.name!r}")

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MediaProbeError(f"ffprobe produced malformed JSON while inspecting {media_path.name!r}") from exc

    if not isinstance(payload, dict):
        raise MediaProbeError(f"ffprobe produced an unexpected JSON structure while inspecting {media_path.name!r}")

    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        raise MediaProbeError(
            f"ffprobe produced no usable stream information while inspecting {media_path.name!r}"
        )

    stream_types: set[str] = set()
    for stream in streams:
        if not isinstance(stream, dict):
            raise MediaProbeError(
                f"ffprobe produced an unexpected stream entry while inspecting {media_path.name!r}"
            )
        if "codec_type" not in stream:
            raise MediaProbeError(
                f"ffprobe produced a stream entry with no codec_type while inspecting {media_path.name!r}"
            )
        codec_type = stream["codec_type"]
        if not isinstance(codec_type, str) or not codec_type:
            raise MediaProbeError(
                f"ffprobe produced a stream entry with an invalid codec_type while inspecting {media_path.name!r}"
            )
        stream_types.add(codec_type)
    return frozenset(stream_types)


def _resolve_scene_sources(
    manifest: VideoManifest,
    audio_artifacts: Sequence[ArtifactRecord],
    animation_artifacts: Sequence[ArtifactRecord],
    project_dir_resolved: Path,
) -> list[_SceneSourcePaths]:
    """Pure, no I/O beyond the artifact-list arguments already supplied:
    for every scene in manifest.scene_plan.scenes (exact stored order,
    never sorted), resolve exactly one eligible audio and one eligible
    animation artifact and turn each into an absolute path. Mirrors
    src.core.scene_timing_finalizer.finalize_scene_timing()'s own
    by-scene grouping pattern exactly, generalized to two artifact kinds
    instead of one."""
    seen_scene_ids: set[str] = set()
    for scene in manifest.scene_plan.scenes:
        if scene.scene_id in seen_scene_ids:
            # Unreachable via load_manifest() — ScenePlan's own validator
            # already forbids duplicate scene_ids. Defensive only; see
            # InvalidSceneOrderError's own docstring.
            raise InvalidSceneOrderError(f"duplicate scene_id {scene.scene_id!r} in manifest scene order")
        seen_scene_ids.add(scene.scene_id)

    def _by_scene(artifacts: Sequence[ArtifactRecord], kind: str) -> dict[str, list[ArtifactRecord]]:
        grouped: dict[str, list[ArtifactRecord]] = {}
        for artifact in artifacts:
            if artifact.kind != kind:
                continue
            if artifact.project_id != manifest.project_id:
                continue
            grouped.setdefault(artifact.scene_id, []).append(artifact)
        return grouped

    audio_by_scene = _by_scene(audio_artifacts, "audio")
    animation_by_scene = _by_scene(animation_artifacts, "animation")

    def _resolve_one(grouped: dict[str, list[ArtifactRecord]], scene_id: str, kind: str) -> ArtifactRecord:
        matches = grouped.get(scene_id, [])
        if len(matches) == 0:
            raise MissingSceneArtifactError(
                f"scene {scene_id!r} has no eligible registered {kind!r} artifact"
            )
        if len(matches) > 1:
            raise AmbiguousSceneArtifactError(
                f"scene {scene_id!r} has {len(matches)} eligible registered {kind!r} artifacts "
                "(expected exactly 1)"
            )
        return matches[0]

    resolved: list[_SceneSourcePaths] = []
    for scene in manifest.scene_plan.scenes:
        animation_artifact = _resolve_one(animation_by_scene, scene.scene_id, "animation")
        audio_artifact = _resolve_one(audio_by_scene, scene.scene_id, "audio")

        animation_path = resolve_under_project_dir(project_dir_resolved, animation_artifact.relative_path)
        if animation_path is None:
            raise ArtifactFileNotFoundError(
                f"registered animation artifact path for scene {scene.scene_id!r} is unsafe"
            )
        audio_path = resolve_under_project_dir(project_dir_resolved, audio_artifact.relative_path)
        if audio_path is None:
            raise ArtifactFileNotFoundError(
                f"registered audio artifact path for scene {scene.scene_id!r} is unsafe"
            )

        for label, path in (("animation", animation_path), ("audio", audio_path)):
            if not path.exists():
                raise ArtifactFileNotFoundError(
                    f"registered {label} artifact file is missing on disk for scene {scene.scene_id!r}"
                )
            if not path.is_file():
                raise ArtifactFileNotFoundError(
                    f"registered {label} artifact path for scene {scene.scene_id!r} is not a regular file"
                )

        resolved.append(_SceneSourcePaths(scene_id=scene.scene_id, animation_path=animation_path, audio_path=audio_path))

    return resolved


@dataclass(frozen=True)
class _AssemblyPlan:
    project_id: str
    manifest: VideoManifest
    output_path: Path
    scene_sources: tuple[_SceneSourcePaths, ...]


def _load_assembly_plan(project_id: str, manifest_path: Path, output_path: Path) -> _AssemblyPlan:
    """Phase A: the only part of this module that opens SQLite, and only
    for the short duration of this function. Raises a
    FinalVideoAssemblyError subclass for every rejection; creates nothing
    on disk and leaves no partial database state on any exit."""
    if not project_id:
        raise FinalVideoAssemblyError("project_id must not be empty")

    try:
        conn = get_readonly_connection()
    except sqlite3.Error as exc:
        raise ProjectNotFoundError("no local project database found") from exc

    try:
        project = get_project(conn, project_id)
        if project is None:
            raise ProjectNotFoundError(f"unknown project_id {project_id!r}")

        if not manifest_path.exists():
            raise ManifestNotFoundError("--manifest does not exist")
        if not manifest_path.is_file():
            raise ManifestNotFoundError("--manifest is not a regular file")

        try:
            manifest = load_manifest(manifest_path)
        except ManifestStoreError as exc:
            raise InvalidManifestError("could not load manifest at --manifest (unreadable or invalid)") from exc

        if manifest.project_id != project_id:
            raise InvalidManifestError(f"--manifest project_id does not match project_id {project_id!r}")
        if manifest.source_fingerprint != project.manifest_fingerprint:
            raise InvalidManifestError(
                "--manifest source_fingerprint does not match the project registry's recorded fingerprint"
            )

        if len(manifest.scene_plan.scenes) == 0:
            raise EmptyManifestError("manifest scene_plan contains no scenes")

        project_dir_resolved = Path(project.manifest_path).resolve().parent

        audio_artifacts = list_artifacts_by_project(conn, project_id, kind="audio")
        animation_artifacts = list_artifacts_by_project(conn, project_id, kind="animation")
        scene_sources = _resolve_scene_sources(manifest, audio_artifacts, animation_artifacts, project_dir_resolved)

        output_resolved = Path(output_path).resolve()
        if output_resolved.exists():
            raise OutputPathConflictError("--output already exists")
        if output_resolved.is_dir():
            raise OutputPathConflictError("--output is a directory")
        source_paths_resolved = {
            os.path.normcase(str(s.animation_path)) for s in scene_sources
        } | {os.path.normcase(str(s.audio_path)) for s in scene_sources}
        if os.path.normcase(str(output_resolved)) in source_paths_resolved:
            raise OutputPathConflictError("--output must not be the same path as a scene source artifact")

        existing_render_artifacts = list_artifacts_by_project(conn, project_id, kind="render")
        if existing_render_artifacts:
            raise ExistingFinalVideoArtifactError(
                f"a render artifact is already registered for project {project_id!r} "
                f"({existing_render_artifacts[0].artifact_id!r}) — this command never replaces an "
                "existing final-video registration"
            )
    finally:
        conn.close()

    return _AssemblyPlan(
        project_id=project_id,
        manifest=manifest,
        output_path=output_resolved,
        scene_sources=tuple(scene_sources),
    )


def _prepare_scene_clip(source: _SceneSourcePaths, out_path: Path) -> Path:
    """Phase B, per scene: mux the registered audio into the animation
    clip only when it has no audio stream of its own; if it already has
    one, fail rather than guess whether it is the approved scene audio
    (see this module's docstring, point 3 of the inspection note). Every
    stream-presence check here is strict (_probe_stream_types(), never
    the old fail-open helper) — a probe failure raises MediaProbeError
    and is never read as "no such stream"."""
    animation_stream_types = _probe_stream_types(source.animation_path)
    if "video" not in animation_stream_types:
        raise SceneMuxError(
            f"scene {source.scene_id!r}'s registered animation artifact has no video stream"
        )
    if "audio" in animation_stream_types:
        raise SceneMuxError(
            f"scene {source.scene_id!r}'s registered animation artifact already contains an audio "
            "stream — this command cannot safely verify it is the approved scene audio, so it "
            "refuses to guess rather than risk muxing audio twice or using the wrong track"
        )

    try:
        mux_audio_video(source.animation_path, source.audio_path, out_path)
    except ProviderError as exc:
        raise SceneMuxError(f"failed to prepare clip for scene {source.scene_id!r}") from exc

    prepared_stream_types = _probe_stream_types(out_path)
    if "video" not in prepared_stream_types or "audio" not in prepared_stream_types:
        raise SceneMuxError(
            f"prepared clip for scene {source.scene_id!r} is missing a required video or audio "
            "stream after muxing"
        )
    return out_path


def _assemble_media(plan: _AssemblyPlan, tempdir: Path) -> tuple[Path, float, float]:
    """Phase B: pure FFmpeg work, no database connection open anywhere in
    this function. Returns (final_temp_path, measured_duration_seconds,
    sum_of_prepared_scene_durations) for phase C's registration step and
    the caller's atomic-replace step. `tempdir` is created and cleaned up
    by the caller (assemble_final_video), never here — this function may
    raise at any point partway through, and ownership must stay with a
    `finally` that is guaranteed to run regardless of where that happens,
    matching src.core.ken_burns_upscale_pipeline.py's own established
    tempdir-at-the-outermost-level pattern exactly."""
    scene_clip_paths: list[Path] = []
    scene_durations: list[float] = []
    for index, source in enumerate(plan.scene_sources, start=1):
        clip_path = tempdir / f"{index:04d}_{_sanitize_for_filename(source.scene_id)}.mp4"
        _prepare_scene_clip(source, clip_path)
        try:
            duration = get_duration_seconds(clip_path)
        except ProviderError as exc:
            raise SceneMuxError(f"could not measure prepared clip duration for scene {source.scene_id!r}") from exc
        if not math.isfinite(duration) or duration <= 0:
            raise SceneMuxError(f"prepared clip for scene {source.scene_id!r} has an invalid duration")
        scene_clip_paths.append(clip_path)
        scene_durations.append(duration)

    final_temp_path = tempdir / "final.mp4"
    try:
        concat_videos(scene_clip_paths, final_temp_path)
    except ProviderError as exc:
        raise FinalConcatError("final concatenation failed") from exc

    if not final_temp_path.exists():
        raise InvalidFinalOutputError("concatenation did not produce an output file")
    if not final_temp_path.is_file():
        raise InvalidFinalOutputError("concatenation output is not a regular file")
    if final_temp_path.stat().st_size == 0:
        raise InvalidFinalOutputError("concatenation output is empty")

    try:
        final_duration = get_duration_seconds(final_temp_path)
    except ProviderError as exc:
        raise InvalidFinalOutputError("could not measure final output duration") from exc
    if not math.isfinite(final_duration) or final_duration <= 0:
        raise InvalidFinalOutputError("final output has an invalid duration")

    # Duration alone does not prove a valid final audiovisual output — a
    # video-only or audio-only concatenation could still report a
    # plausible duration. Strict probing closes that gap.
    final_stream_types = _probe_stream_types(final_temp_path)
    if "video" not in final_stream_types:
        raise InvalidFinalOutputError("final output has no video stream")
    if "audio" not in final_stream_types:
        raise InvalidFinalOutputError("final output has no audio stream")

    expected_total = sum(scene_durations)
    tolerance = _duration_tolerance_seconds(len(plan.scene_sources))
    if abs(final_duration - expected_total) > tolerance:
        raise InvalidFinalOutputError(
            f"final duration {final_duration:.3f}s does not match the sum of prepared scene "
            f"durations {expected_total:.3f}s within tolerance {tolerance:.3f}s"
        )

    return final_temp_path, final_duration, expected_total


def assemble_final_video(project_id: str, manifest_path: Path, output_path: Path) -> FinalVideoAssemblyResult:
    """Generate one final MP4 for `project_id` at `output_path` by
    concatenating every scene's already-registered animation+audio
    artifacts, in manifest order, and register the result as this
    project's canonical "render" artifact. Raises a
    FinalVideoAssemblyError subclass and leaves no new file or database
    row behind on any rejection or failure — see this module's docstring
    for the full three-part lifecycle and cleanup proof.

    Phase C never runs ffprobe while its SQLite connection is open:
    prevalidate_render_source() (pure, local, no SQLite) runs on the
    already-placed --output file BEFORE get_connection() is ever called,
    and its result is handed to register_render_artifact() via the
    `prevalidated` parameter, which itself only re-confirms size/checksum
    (plain file reads, not a subprocess) rather than re-running ffprobe.

    Cleanup failures are never silently swallowed. If temp-directory or
    orphaned-output removal fails while a primary assembly/registration
    exception is already propagating, that failure is attached to the
    primary exception via add_note() (Python 3.11+) rather than replacing
    it — the original, actionable failure stays the one callers see and
    catch. If cleanup fails with no primary exception in flight (a
    cleanup-only failure after an otherwise-complete run), it is raised
    directly as FinalVideoCleanupError, since there is nothing else to
    preserve."""
    plan = _load_assembly_plan(project_id, Path(manifest_path), Path(output_path))

    plan.output_path.parent.mkdir(parents=True, exist_ok=True)
    tempdir = Path(tempfile.mkdtemp(prefix="final-video-assembly-", dir=plan.output_path.parent))

    output_created = False
    registration_succeeded = False
    primary_exc: Exception | None = None
    result = None
    final_duration = None
    try:
        try:
            final_temp_path, final_duration, _expected_total = _assemble_media(plan, tempdir)

            os.replace(final_temp_path, plan.output_path)
            output_created = True

            # Pure, local, no SQLite connection open anywhere in this
            # call — the only ffprobe work phase C needs, done BEFORE
            # get_connection() below.
            try:
                prevalidated = prevalidate_render_source(plan.output_path)
            except RenderSourceValidationError as exc:
                raise InvalidFinalOutputError("final output failed pre-registration validation") from exc

            try:
                conn = get_connection()
            except sqlite3.Error as exc:
                raise FinalArtifactRegistrationError("no local project database found") from exc

            try:
                existing = list_artifacts_by_project(conn, plan.project_id, kind="render")
                if existing:
                    raise ExistingFinalVideoArtifactError(
                        f"a render artifact was registered for project {plan.project_id!r} "
                        f"({existing[0].artifact_id!r}) while this command was running"
                    )

                project = get_project(conn, plan.project_id)
                if project is None:
                    raise ProjectNotFoundError(f"unknown project_id {plan.project_id!r}")

                metadata_overrides = {
                    "source": "final-video-assembly-v1",
                    "scene_count": len(plan.scene_sources),
                    "manifest_fingerprint": plan.manifest.source_fingerprint,
                }
                try:
                    result = register_render_artifact(
                        conn,
                        project,
                        plan.manifest,
                        plan.output_path,
                        now=datetime.now(timezone.utc),
                        prevalidated=prevalidated,
                        metadata_overrides=metadata_overrides,
                        strict_cleanup=True,
                    )
                except RenderSourceMismatchError as exc:
                    raise FinalArtifactRegistrationError(
                        "final output changed unexpectedly between pre-validation and registration"
                    ) from exc
                except RenderArtifactCleanupError as exc:
                    raise FinalArtifactCleanupError(
                        "final artifact registration failed and its own canonical-copy cleanup also failed"
                    ) from exc
                except Exception as exc:
                    raise FinalArtifactRegistrationError("final artifact registration failed") from exc

                if not result.ok:
                    raise FinalArtifactRegistrationError(
                        "final artifact registration was rejected: " + "; ".join(result.reasons)
                    )
            finally:
                conn.close()

            registration_succeeded = True
        except Exception as exc:
            primary_exc = exc
            raise
    finally:
        try:
            shutil.rmtree(tempdir)
        except OSError as cleanup_exc:
            if primary_exc is not None:
                primary_exc.add_note(
                    f"additionally, failed to remove temporary assembly directory {tempdir}: "
                    f"{type(cleanup_exc).__name__}"
                )
            else:
                raise FinalVideoCleanupError(
                    f"failed to remove temporary assembly directory {tempdir}"
                ) from cleanup_exc

        if output_created and not registration_succeeded:
            try:
                plan.output_path.unlink()
            except OSError as cleanup_exc:
                if primary_exc is not None:
                    primary_exc.add_note(
                        f"additionally, failed to remove the orphaned output at {plan.output_path}: "
                        f"{type(cleanup_exc).__name__}"
                    )
                else:
                    raise FinalVideoCleanupError(
                        f"registration failed and the orphaned output at {plan.output_path} "
                        "could not be removed"
                    ) from cleanup_exc

    return FinalVideoAssemblyResult(
        project_id=plan.project_id,
        output_path=plan.output_path,
        scene_count=len(plan.scene_sources),
        measured_duration_seconds=final_duration,
        artifact_id=result.artifact_id,
    )
