"""TEXT OVERLAY RENDERER V1: registration-first, project-level Overlay
Render Artifact Registration.

Registers a single, already-produced local MP4 (a text-overlay-burned-in
copy of a project's registered "render" artifact) as THE project's
canonical "overlay_render" artifact. This is a DELIBERATELY SEPARATE kind
from "render" — src/core/render_artifact_registrar.py's own
register_render_artifact() enforces that at most one "render" artifact
may ever exist per project (src/core/final_video_assembly.py's own
ExistingFinalVideoArtifactError), so an overlay-burned copy of that render
cannot itself be registered as a second "render" without weakening that
existing, already-tested policy. "overlay_render" is therefore its own
ArtifactKind (src/models/enums.py), also project-level
(src/models/artifact.py's PROJECT_LEVEL_ARTIFACT_KINDS), with exactly one
row per project, fixed and deterministic:

    artifact_id:    overlay-render-final
    kind:           overlay_render
    scene_id:       None
    relative_path:  overlay/final.mp4

This module never renders, never invokes ffmpeg's *encode* path itself
(src/core/text_overlay_render.py does that), never mutates the manifest or
ProjectRecord.manifest_fingerprint, never advances a project's lifecycle
stage, and never touches the "render" artifact's own row or file
(render/final.mp4) — it only reads that row to confirm the overlay source
this call is registering was in fact produced from it (see
prevalidate_overlay_source()'s docstring). Local filesystem and local
SQLite only, plus local, read-only ffprobe subprocess calls to validate
the source file before ever copying it — same shape as
src/core/render_artifact_registrar.py, deliberately duplicated rather than
imported (this codebase's own established convention: see that module's
own docstring, "Deliberately NOT shared... intentional, small,
self-contained duplicates").

Same three write paths, same ordering, same zero-writes-on-rejection
guarantee, and same "no SQLite connection open during any ffprobe/ffmpeg
subprocess call" discipline as register_render_artifact() — achieved the
identical way: prevalidate_overlay_source() runs all file/media validation
with no connection open; register_overlay_render_artifact() accepts that
result via its own `prevalidated` parameter and only re-confirms size/
checksum (pure, local, non-subprocess) before trusting it."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

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
from src.utils.config import get_settings

_CHUNK_SIZE = 1024 * 1024

_ARTIFACT_ID = "overlay-render-final"
_RELATIVE_PATH = "overlay/final.mp4"


class OverlayArtifactRegistrationError(Exception):
    """Raised only for a project/manifest identity mismatch — a caller
    bug, not a normal registration outcome. Ordinary rejections (invalid
    source file, a conflicting existing record/file, ...) are reported via
    a returned OverlayArtifactRegistrationResult instead."""


class OverlaySourceValidationError(Exception):
    """Raised by prevalidate_overlay_source() when `source_file` cannot be
    validated as a playable video with the required stream layout."""


class OverlaySourceMismatchError(Exception):
    """Raised by register_overlay_render_artifact() when a supplied
    `prevalidated` result does not identify the same file as
    `source_file`, or no longer matches its current content — same
    binding contract as src.core.render_artifact_registrar's
    RenderSourceMismatchError, applied here for the same reason."""


class OverlayArtifactCleanupError(Exception):
    """Raised by register_overlay_render_artifact() only when
    `strict_cleanup=True` AND this call's own freshly-copied canonical
    overlay/final.mp4 copy could not be removed after a registration
    failure. Default (`strict_cleanup=False`) callers keep the original
    best-effort, silently-swallowed cleanup behavior."""


@dataclass(frozen=True)
class PrevalidatedOverlaySource:
    """The result of prevalidate_overlay_source(): local file/media facts
    about one overlay-render source file, computed WITHOUT any open SQLite
    connection."""

    source_path: Path
    byte_size: int
    sha256_checksum: str
    duration_seconds: float


@dataclass(frozen=True)
class OverlayArtifactRegistrationResult:
    """The one typed result register_overlay_render_artifact() always
    returns, success or failure. Never itself mutates the manifest, a
    ProjectRecord's lifecycle state, or the "render" artifact."""

    project_id: str
    artifact_id: str
    relative_path: str
    ok: bool
    idempotent: bool
    copied: bool
    duration_seconds: float | None
    artifact: ArtifactRecord | None
    reasons: tuple[str, ...] = ()


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe_stream_types(media_path: Path) -> frozenset[str]:
    """Strict, fail-closed local ffprobe stream-type probe — same
    contract as src.core.final_video_assembly's own _probe_stream_types()
    (deliberately duplicated, not imported, matching that module's own
    stated convention for these small ffprobe helpers): raises
    OverlaySourceValidationError, chaining the original exception as
    __cause__, for every failure mode (missing/unusable ffprobe binary,
    non-zero exit, empty output, malformed JSON, an unexpected JSON
    structure, an empty/malformed streams list, or a stream entry with a
    missing/empty/non-string codec_type) — never silently interpreted as
    "no such stream"."""
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
        raise OverlaySourceValidationError(
            f"could not run ffprobe to inspect media streams for {media_path.name!r}"
        ) from exc

    if result.returncode != 0:
        raise OverlaySourceValidationError(f"ffprobe exited with an error while inspecting {media_path.name!r}")
    if not result.stdout or not result.stdout.strip():
        raise OverlaySourceValidationError(f"ffprobe produced no output while inspecting {media_path.name!r}")

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise OverlaySourceValidationError(
            f"ffprobe produced malformed JSON while inspecting {media_path.name!r}"
        ) from exc

    if not isinstance(payload, dict):
        raise OverlaySourceValidationError(
            f"ffprobe produced an unexpected JSON structure while inspecting {media_path.name!r}"
        )
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        raise OverlaySourceValidationError(
            f"ffprobe produced no usable stream information while inspecting {media_path.name!r}"
        )

    stream_types: set[str] = set()
    for stream in streams:
        if not isinstance(stream, dict):
            raise OverlaySourceValidationError(
                f"ffprobe produced an unexpected stream entry while inspecting {media_path.name!r}"
            )
        if "codec_type" not in stream:
            raise OverlaySourceValidationError(
                f"ffprobe produced a stream entry with no codec_type while inspecting {media_path.name!r}"
            )
        codec_type = stream["codec_type"]
        if not isinstance(codec_type, str) or not codec_type:
            raise OverlaySourceValidationError(
                f"ffprobe produced a stream entry with an invalid codec_type while inspecting {media_path.name!r}"
            )
        stream_types.add(codec_type)
    return frozenset(stream_types)


def _probe_duration_seconds(media_path: Path) -> float:
    ffmpeg_path = Path(get_settings().ffmpeg_path)
    ffprobe_name = "ffprobe.exe" if ffmpeg_path.suffix == ".exe" else "ffprobe"
    ffprobe = str(ffmpeg_path.with_name(ffprobe_name))
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(media_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise OverlaySourceValidationError(f"could not measure duration for {media_path.name!r}") from exc
    if result.returncode != 0:
        raise OverlaySourceValidationError(f"ffprobe exited with an error while measuring {media_path.name!r}")
    try:
        return float(json.loads(result.stdout)["format"]["duration"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise OverlaySourceValidationError(
            f"ffprobe produced an unexpected duration payload for {media_path.name!r}"
        ) from exc


def _validate_overlay_video(source_file: Path) -> tuple[float | None, str | None]:
    """Validate `source_file` is playable video with both a video and an
    audio stream (an overlay-burned final video must keep the same
    audiovisual shape the source "render" artifact had). Returns
    (duration_seconds, None) on success or (None, error_message) on
    failure."""
    try:
        duration = _probe_duration_seconds(source_file)
    except OverlaySourceValidationError as exc:
        return None, f"could not measure video duration for {source_file}: {exc}"
    try:
        stream_types = _probe_stream_types(source_file)
    except OverlaySourceValidationError as exc:
        return None, f"could not inspect media streams for {source_file}: {exc}"
    if "video" not in stream_types:
        return None, f"source file at {source_file} does not contain a video stream"
    if "audio" not in stream_types:
        return None, f"source file at {source_file} does not contain an audio stream"
    return duration, None


def prevalidate_overlay_source(source_file: Path) -> PrevalidatedOverlaySource:
    """Pure, local, no SQLite connection anywhere in this function —
    identical contract to render_artifact_registrar.prevalidate_render_source(),
    applied to an overlay-burned output file instead."""
    if not source_file.exists():
        raise OverlaySourceValidationError(f"source file does not exist at {source_file}")
    if not source_file.is_file():
        raise OverlaySourceValidationError(f"source file at {source_file} is not a regular file")
    size = source_file.stat().st_size
    if size == 0:
        raise OverlaySourceValidationError(f"source file at {source_file} is empty")
    checksum = _sha256_of_file(source_file)

    duration_seconds, video_error = _validate_overlay_video(source_file)
    if video_error is not None:
        raise OverlaySourceValidationError(video_error)

    return PrevalidatedOverlaySource(
        source_path=source_file.resolve(),
        byte_size=size,
        sha256_checksum=checksum,
        duration_seconds=duration_seconds,
    )


def _cleanup_fresh_copy(destination: Path, created_fresh_copy: bool, *, strict: bool = False) -> None:
    if not created_fresh_copy:
        return
    if strict:
        destination.unlink(missing_ok=True)
        return
    try:
        destination.unlink(missing_ok=True)
    except OSError:
        pass


def register_overlay_render_artifact(
    conn: sqlite3.Connection,
    project: ProjectRecord,
    manifest: VideoManifest,
    source_file: Path,
    *,
    now: datetime | None = None,
    prevalidated: PrevalidatedOverlaySource | None = None,
    metadata_overrides: Mapping[str, str | int | float | bool | None] | None = None,
    strict_cleanup: bool = False,
) -> OverlayArtifactRegistrationResult:
    """Register `source_file` as this project's canonical, project-level
    "overlay_render" artifact. Writes nothing to SQLite or the filesystem
    on any rejection path. Stage-agnostic. Takes no scene_id.

    `prevalidated` (optional, keyword-only): mirrors
    register_render_artifact()'s own contract exactly, including the
    source-path binding check (resolve both sides, compare via
    os.path.normcase) BEFORE trusting any of `prevalidated`'s fields, then
    re-confirming size/checksum. Omitting it (the default) runs full
    ffprobe validation every time.

    `metadata_overrides` / `strict_cleanup`: same contracts as
    register_render_artifact()'s own parameters of the same names."""
    when = now if now is not None else datetime.now(timezone.utc)
    if manifest.project_id != project.project_id:
        raise OverlayArtifactRegistrationError(
            f"manifest project_id {manifest.project_id!r} does not match project {project.project_id!r}"
        )
    if manifest.source_fingerprint != project.manifest_fingerprint:
        raise OverlayArtifactRegistrationError(
            "manifest fingerprint does not match the project registry's recorded fingerprint"
        )

    def _result(
        *,
        ok: bool,
        idempotent: bool = False,
        copied: bool = False,
        duration_seconds: float | None = None,
        artifact: ArtifactRecord | None = None,
        reasons: tuple[str, ...] = (),
    ) -> OverlayArtifactRegistrationResult:
        return OverlayArtifactRegistrationResult(
            project_id=project.project_id,
            artifact_id=_ARTIFACT_ID,
            relative_path=_RELATIVE_PATH,
            ok=ok,
            idempotent=idempotent,
            copied=copied,
            duration_seconds=duration_seconds,
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

    existing_matches = list_artifacts_by_project(conn, project.project_id, kind="overlay_render")
    if existing_matches:
        existing = existing_matches[0]
        if existing.sha256_checksum == source_checksum:
            return _result(ok=True, idempotent=True, artifact=existing)
        return _result(
            ok=False,
            reasons=(
                f"an overlay_render artifact is already registered for this project "
                f"({existing.artifact_id!r}) with different content — this module never "
                "overwrites an existing artifact registration",
            ),
        )

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
                    "this module never overwrites an existing file",
                ),
            )
        final_checksum, final_size = dest_checksum, dest_size
    else:
        final_checksum, final_size = None, None

    if prevalidated is not None:
        resolved_source = source_file.resolve()
        if os.path.normcase(str(resolved_source)) != os.path.normcase(str(prevalidated.source_path)):
            raise OverlaySourceMismatchError(
                f"prevalidated result was computed for {prevalidated.source_path}, which does not "
                f"resolve to the same file as the source_file supplied for registration "
                f"({source_file}) — refusing to trust a prevalidation result computed for a "
                "different path even if its recorded bytes currently match"
            )
        if source_checksum != prevalidated.sha256_checksum or source_size != prevalidated.byte_size:
            raise OverlaySourceMismatchError(
                f"prevalidated result for {prevalidated.source_path} no longer matches the current "
                f"content of {source_file} (size/checksum changed since prevalidation) — refusing to "
                "trust a stale prevalidation result"
            )
        duration_seconds, video_error = prevalidated.duration_seconds, None
    else:
        duration_seconds, video_error = _validate_overlay_video(source_file)
    if video_error is not None:
        return _result(ok=False, reasons=(video_error,))

    if final_checksum is None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, destination)
        created_fresh_copy = True
        final_size = destination.stat().st_size
        final_checksum = _sha256_of_file(destination)

    metadata = {"duration_seconds": duration_seconds, "source": "text-overlay-renderer-v1"}
    if metadata_overrides is not None:
        metadata.update(metadata_overrides)

    record = ArtifactRecord(
        artifact_id=_ARTIFACT_ID,
        project_id=project.project_id,
        kind="overlay_render",
        scene_id=None,
        relative_path=_RELATIVE_PATH,
        byte_size=final_size,
        sha256_checksum=final_checksum,
        created_at=when,
        metadata=metadata,
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
        try:
            _cleanup_fresh_copy(destination, created_fresh_copy, strict=strict_cleanup)
        except OSError as cleanup_exc:
            cleanup_error = OverlayArtifactCleanupError(
                f"registration was rejected ({exc}) and the freshly-copied canonical file at "
                f"{destination} could not be removed"
            )
            cleanup_error.add_note(f"original registration rejection: {type(exc).__name__}: {exc}")
            raise cleanup_error from cleanup_exc
        return _result(ok=False, duration_seconds=duration_seconds, reasons=(str(exc),))
    except Exception as exc:
        try:
            _cleanup_fresh_copy(destination, created_fresh_copy, strict=strict_cleanup)
        except OSError as cleanup_exc:
            cleanup_error = OverlayArtifactCleanupError(
                f"registration failed unexpectedly ({type(exc).__name__}) and the freshly-copied "
                f"canonical file at {destination} could not be removed"
            )
            cleanup_error.add_note(f"original unexpected registration failure: {type(exc).__name__}: {exc}")
            raise cleanup_error from cleanup_exc
        raise

    return _result(
        ok=True,
        copied=created_fresh_copy,
        duration_seconds=duration_seconds,
        artifact=record,
    )
