"""Phase 2D: registration-first Audio Artifact Pipeline.

Registers a single, already-produced local audio file as one scene's
"audio" artifact, so a project can later satisfy Phase 2C's
verify-and-advance(..., to_stage="audio_ready", ...) requirement of one
verified audio artifact per scene. This module never synthesizes audio
(no TTS, no Kokoro, no provider call), never mutates the manifest or
ProjectRecord.manifest_fingerprint, and never advances a project's
lifecycle stage — see src/core/verified_transition_service.py for that.

The canonical identity for a project's scene is fixed and deterministic —
there is no alternate-take support in Phase 2D:

    artifact_id:    audio-<scene_id>
    kind:           audio
    scene_id:       <scene_id>
    relative_path:  audio/<scene_id>.wav

Local filesystem and local SQLite only, plus one read-only subprocess call
to the local ffprobe binary (src/render/ffmpeg_render.get_duration_seconds)
to measure the source file's real duration — required because the
channel's ChannelPolicy locks final_timing_source to "measured_audio"
(src/utils/channel_config.py), so an audio artifact registered without a
measured duration would misrepresent that policy. No LLM, image
generation, Flow, Veo, renderer, FFmpeg *encode*, YouTube, network/API, or
payment call anywhere in this module.

Write paths, exactly three:
  1. mkdir <project_dir>/audio/ (only once every validation has passed)
  2. copy the source file to <project_dir>/audio/<scene_id>.wav (only for a
     genuinely fresh registration — never when the destination already has
     byte-identical content)
  3. register_artifact() — exactly one INSERT, and only after (1) and (2)
     (if needed) have already succeeded

Every ordinary rejection (unknown scene, invalid source file, unmeasurable
duration, an existing record/file with different content, a documented
register_artifact() failure — ArtifactAlreadyExistsError,
DuplicateArtifactRegistrationError, ArtifactProjectNotFoundError,
ArtifactRegistrationError, or any sqlite3.Error, ...) is reported via the
returned AudioArtifactRegistrationResult with ok=False — this module does
not raise for those. It raises AudioArtifactRegistrationError only for a
caller/input inconsistency it will not silently work around (a manifest
that does not belong to the given project), the same split
src/core/verified_transition_service.py uses. An UNEXPECTED exception from
register_artifact() (anything outside that documented set) is a different
case: this call's own freshly-copied file (never a pre-existing one) is
still cleaned up best-effort, but the exception itself is re-raised
unchanged rather than folded into a result — an undocumented failure is a
bug to surface, not an ordinary outcome to report."""
from __future__ import annotations

import hashlib
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
    list_artifacts_by_scene,
    register_artifact,
)
from src.models.artifact import ArtifactRecord
from src.models.manifest import VideoManifest
from src.models.project_state import ProjectRecord
from src.render.ffmpeg_render import get_duration_seconds
from src.providers.base import ProviderError

_CHUNK_SIZE = 1024 * 1024


class AudioArtifactRegistrationError(Exception):
    """Raised only for a project/manifest identity mismatch — a caller
    bug, not a normal registration outcome. Ordinary rejections (unknown
    scene, invalid source file, unmeasurable duration, a conflicting
    existing record/file, ...) are reported via a returned
    AudioArtifactRegistrationResult instead; see this module's
    docstring."""


@dataclass(frozen=True)
class AudioArtifactRegistrationResult:
    """The one typed result register_audio_artifact() always returns,
    success or failure. Never itself mutates the manifest or a
    ProjectRecord's lifecycle state — it only reports what this call
    already did (or refused to do) to the artifact registry and, at most,
    one file."""

    project_id: str
    scene_id: str
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


def _measure_duration(source_file: Path) -> tuple[float | None, str | None]:
    try:
        return get_duration_seconds(source_file), None
    except (ProviderError, OSError, ValueError, KeyError) as exc:
        return None, f"could not measure audio duration for {source_file}: {exc}"


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


def register_audio_artifact(
    conn: sqlite3.Connection,
    project: ProjectRecord,
    manifest: VideoManifest,
    scene_id: str,
    source_file: Path,
    *,
    now: datetime | None = None,
) -> AudioArtifactRegistrationResult:
    """Register `source_file` as project/scene_id's canonical "audio"
    artifact. Writes nothing to SQLite or the filesystem on any rejection
    path — see this module's docstring for the exact three write points a
    successful, non-idempotent call may reach."""
    when = now if now is not None else datetime.now(timezone.utc)
    if manifest.project_id != project.project_id:
        raise AudioArtifactRegistrationError(
            f"manifest project_id {manifest.project_id!r} does not match project {project.project_id!r}"
        )
    if manifest.source_fingerprint != project.manifest_fingerprint:
        raise AudioArtifactRegistrationError(
            "manifest fingerprint does not match the project registry's recorded fingerprint"
        )

    artifact_id = f"audio-{scene_id}"
    relative_path = f"audio/{scene_id}.wav"

    def _result(
        *,
        ok: bool,
        idempotent: bool = False,
        copied: bool = False,
        duration_seconds: float | None = None,
        artifact: ArtifactRecord | None = None,
        reasons: tuple[str, ...] = (),
    ) -> AudioArtifactRegistrationResult:
        return AudioArtifactRegistrationResult(
            project_id=project.project_id,
            scene_id=scene_id,
            artifact_id=artifact_id,
            relative_path=relative_path,
            ok=ok,
            idempotent=idempotent,
            copied=copied,
            duration_seconds=duration_seconds,
            artifact=artifact,
            reasons=reasons,
        )

    known_scene_ids = frozenset(scene.scene_id for scene in manifest.scene_plan.scenes)
    if scene_id not in known_scene_ids:
        return _result(ok=False, reasons=(f"scene_id {scene_id!r} is not present in the project manifest",))

    if not source_file.exists():
        return _result(ok=False, reasons=(f"source file does not exist at {source_file}",))
    if not source_file.is_file():
        return _result(ok=False, reasons=(f"source file at {source_file} is not a regular file",))
    source_size = source_file.stat().st_size
    if source_size == 0:
        return _result(ok=False, reasons=(f"source file at {source_file} is empty",))
    source_checksum = _sha256_of_file(source_file)

    project_dir_resolved = Path(project.manifest_path).resolve().parent
    destination = resolve_under_project_dir(project_dir_resolved, relative_path)
    if destination is None:
        return _result(
            ok=False,
            reasons=(
                f"relative_path {relative_path!r} is unsafe: it resolves outside "
                f"the project directory {project_dir_resolved}",
            ),
        )

    # 1. Already registered? Compare against the INCOMING file's checksum
    # only (never re-derived from whatever currently sits at `destination`
    # on disk) — an identical match is a pure no-op: no file touched, no
    # DB write. A mismatch is rejected outright; Phase 2D never overwrites
    # an existing registration.
    existing_matches = [a for a in list_artifacts_by_scene(conn, project.project_id, scene_id) if a.kind == "audio"]
    if existing_matches:
        existing = existing_matches[0]
        if existing.sha256_checksum == source_checksum:
            return _result(ok=True, idempotent=True, artifact=existing)
        return _result(
            ok=False,
            reasons=(
                f"an audio artifact is already registered for scene {scene_id!r} "
                f"({existing.artifact_id!r}) with different content — Phase 2D never overwrites "
                "an existing artifact registration",
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
                    "Phase 2D never overwrites an existing file",
                ),
            )
        final_checksum, final_size = dest_checksum, dest_size
    else:
        final_checksum, final_size = None, None  # computed after the copy below

    duration_seconds, duration_error = _measure_duration(source_file)
    if duration_error is not None:
        return _result(ok=False, reasons=(duration_error,))

    if final_checksum is None:
        # Fresh path: nothing exists yet at `destination` — create audio/
        # only now, after every validation (including duration
        # measurement) has already passed.
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, destination)
        created_fresh_copy = True
        final_size = destination.stat().st_size
        final_checksum = _sha256_of_file(destination)  # hash the COPY, not the source

    record = ArtifactRecord(
        artifact_id=artifact_id,
        project_id=project.project_id,
        kind="audio",
        scene_id=scene_id,
        relative_path=relative_path,
        byte_size=final_size,
        sha256_checksum=final_checksum,
        created_at=when,
        metadata={"duration_seconds": duration_seconds, "source": "external"},
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
        return _result(ok=False, duration_seconds=duration_seconds, reasons=(str(exc),))
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
        duration_seconds=duration_seconds,
        artifact=record,
    )
