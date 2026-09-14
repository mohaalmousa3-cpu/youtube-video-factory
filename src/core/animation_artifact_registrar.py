"""Phase 2F: registration-first Animation Artifact Registration Pipeline.

Registers a single, already-produced local MP4 video file as one scene's
"animation" artifact, so a project can later satisfy Phase 2C's
verify-and-advance(..., to_stage="animation_ready", ...) requirement of one
verified animation artifact per scene. This module never renders or
generates animation (no Flow, no Veo, no ffmpeg *encode*, no provider
call), never mutates the manifest or ProjectRecord.manifest_fingerprint,
and never advances a project's lifecycle stage — see
src/core/verified_transition_service.py for that. It is also
stage-agnostic: it never checks or cares what current_stage the project is
in.

The canonical identity for a project's scene is fixed and deterministic —
there is no alternate-take support in Phase 2F, matching Phase 2D/2E's own
design:

    artifact_id:    animation-<scene_id>
    kind:           animation
    scene_id:       <scene_id>
    relative_path:  animation/<scene_id>.mp4

(This intentionally supersedes an earlier, sparsely-used
"animation/<scene_id>.json" placeholder seen in a couple of generic Phase
2B model/service tests — that was never a deliberate file-format decision
the way audio's .wav and visual's .png conventions were; Phase 2F is the
first phase to actually define what a real animation artifact is, and it
is a playable MP4 clip, not JSON.)

Local filesystem and local SQLite only, plus two local, read-only ffprobe
subprocess calls (both via the local ffmpeg/ffprobe binary already
required by this project — no new dependency, no network) to validate the
source file before ever copying it:
  1. src/render/ffmpeg_render.get_duration_seconds() — imported and used
     UNMODIFIED, exactly as audio_artifact_registrar.py already does; a
     corrupt/undecodable file fails here.
  2. a video-STREAM check local to this module (not added to
     ffmpeg_render.py) — duration alone does not prove the file contains
     video content at all (a real, valid, audio-only .mp4/.m4a reports a
     perfectly good duration); this module additionally requires at least
     one stream with codec_type "video", so an audio-only file renamed
     .mp4 is rejected just as firmly as a corrupt one. Checking for a
     video stream this way, rather than matching ffprobe's container
     format_name string, sidesteps that string being an unreliable,
     comma-joined list (e.g. "mov,mp4,m4a,3gp,3g2,mj2") rather than a
     single clean value the way Pillow's img.format is for PNG.

No LLM, TTS, image generation, Flow, Veo, renderer *invocation*, YouTube,
network/API, or payment call anywhere in this module.

Write paths, exactly three — identical shape to audio/visual_artifact_registrar.py:
  1. mkdir <project_dir>/animation/ (only once every validation has passed)
  2. copy the source file to <project_dir>/animation/<scene_id>.mp4 (only
     for a genuinely fresh registration — never when the destination
     already has byte-identical content)
  3. register_artifact() — exactly one INSERT, and only after (1) and (2)
     (if needed) have already succeeded

Every ordinary rejection (unknown scene, invalid source file, unmeasurable
or audio-only video, an existing record/file with different content, a
documented register_artifact() failure — ArtifactAlreadyExistsError,
DuplicateArtifactRegistrationError, ArtifactProjectNotFoundError,
ArtifactRegistrationError, or any sqlite3.Error, ...) is reported via the
returned AnimationArtifactRegistrationResult with ok=False — this module
does not raise for those. It raises AnimationArtifactRegistrationError
only for a caller/input inconsistency it will not silently work around (a
manifest that does not belong to the given project), the same split
src/core/verified_transition_service.py and the two prior registrars use.
An UNEXPECTED exception from register_artifact() (anything outside that
documented set) is a different case: this call's own freshly-copied file
(never a pre-existing one) is still cleaned up best-effort, but the
exception itself is re-raised unchanged rather than folded into a result —
an undocumented failure is a bug to surface, not an ordinary outcome to
report.

Deliberately NOT shared with audio_artifact_registrar.py /
visual_artifact_registrar.py in this phase: _sha256_of_file() and the
fresh-copy cleanup helper below are intentional, small, self-contained
duplicates (a third copy now), not imports — a scoped choice, not an
oversight; a shared-helper extraction is left for a later, deliberate
maintenance refactor, same as Phase 2E's own stated choice."""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
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
from src.providers.base import ProviderError
from src.render.ffmpeg_render import get_duration_seconds
from src.utils.config import get_settings

_CHUNK_SIZE = 1024 * 1024


class AnimationArtifactRegistrationError(Exception):
    """Raised only for a project/manifest identity mismatch — a caller
    bug, not a normal registration outcome. Ordinary rejections (unknown
    scene, invalid source file, unmeasurable or audio-only video, a
    conflicting existing record/file, ...) are reported via a returned
    AnimationArtifactRegistrationResult instead; see this module's
    docstring."""


@dataclass(frozen=True)
class AnimationArtifactRegistrationResult:
    """The one typed result register_animation_artifact() always returns,
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


def _has_video_stream(media_path: Path) -> bool:
    """Local, read-only ffprobe query for whether `media_path` contains at
    least one video stream — same ffprobe-path derivation pattern
    get_duration_seconds() uses, duplicated here rather than added to
    ffmpeg_render.py (this module changes no existing Phase 2D/2E/render
    file). Returns False (never raises) on any ffprobe failure — the
    caller has already confirmed the file is valid, probeable media via
    get_duration_seconds() before this is ever called, so a failure here
    would be unexpected, but this function's only job is a yes/no stream
    check, not error reporting."""
    ffmpeg_path = Path(get_settings().ffmpeg_path)
    ffprobe_name = "ffprobe.exe" if ffmpeg_path.suffix == ".exe" else "ffprobe"
    ffprobe = str(ffmpeg_path.with_name(ffprobe_name))
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "stream=codec_type", "-of", "json", str(media_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    try:
        streams = json.loads(result.stdout).get("streams", [])
    except (json.JSONDecodeError, AttributeError):
        return False
    return any(stream.get("codec_type") == "video" for stream in streams)


def _validate_video(source_file: Path) -> tuple[float | None, str | None]:
    """Validate `source_file` is playable video: a real, measurable
    duration (catches corrupt/undecodable files) AND at least one video
    stream (catches a valid-but-audio-only file, which duration alone
    would not catch). Returns (duration_seconds, None) on success or
    (None, error_message) on any failure."""
    try:
        duration = get_duration_seconds(source_file)
    except (ProviderError, OSError, ValueError, KeyError) as exc:
        return None, f"could not measure video duration for {source_file}: {exc}"
    if not _has_video_stream(source_file):
        return None, (
            f"source file at {source_file} does not contain a video stream "
            "(audio-only or no playable video track) — only real video is accepted"
        )
    return duration, None


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


def register_animation_artifact(
    conn: sqlite3.Connection,
    project: ProjectRecord,
    manifest: VideoManifest,
    scene_id: str,
    source_file: Path,
    *,
    now: datetime | None = None,
) -> AnimationArtifactRegistrationResult:
    """Register `source_file` as project/scene_id's canonical "animation"
    artifact. Writes nothing to SQLite or the filesystem on any rejection
    path — see this module's docstring for the exact three write points a
    successful, non-idempotent call may reach. Stage-agnostic: never reads
    or checks project.current_stage."""
    when = now if now is not None else datetime.now(timezone.utc)
    if manifest.project_id != project.project_id:
        raise AnimationArtifactRegistrationError(
            f"manifest project_id {manifest.project_id!r} does not match project {project.project_id!r}"
        )
    if manifest.source_fingerprint != project.manifest_fingerprint:
        raise AnimationArtifactRegistrationError(
            "manifest fingerprint does not match the project registry's recorded fingerprint"
        )

    artifact_id = f"animation-{scene_id}"
    relative_path = f"animation/{scene_id}.mp4"

    def _result(
        *,
        ok: bool,
        idempotent: bool = False,
        copied: bool = False,
        duration_seconds: float | None = None,
        artifact: ArtifactRecord | None = None,
        reasons: tuple[str, ...] = (),
    ) -> AnimationArtifactRegistrationResult:
        return AnimationArtifactRegistrationResult(
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
    # DB write. A mismatch is rejected outright; Phase 2F never overwrites
    # an existing registration.
    existing_matches = [
        a for a in list_artifacts_by_scene(conn, project.project_id, scene_id) if a.kind == "animation"
    ]
    if existing_matches:
        existing = existing_matches[0]
        if existing.sha256_checksum == source_checksum:
            return _result(ok=True, idempotent=True, artifact=existing)
        return _result(
            ok=False,
            reasons=(
                f"an animation artifact is already registered for scene {scene_id!r} "
                f"({existing.artifact_id!r}) with different content — Phase 2F never overwrites "
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
                    "Phase 2F never overwrites an existing file",
                ),
            )
        final_checksum, final_size = dest_checksum, dest_size
    else:
        final_checksum, final_size = None, None  # computed after the copy below

    duration_seconds, video_error = _validate_video(source_file)
    if video_error is not None:
        return _result(ok=False, reasons=(video_error,))

    if final_checksum is None:
        # Fresh path: nothing exists yet at `destination` — create
        # animation/ only now, after every validation (including video
        # decode + stream-type validation) has already passed.
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, destination)
        created_fresh_copy = True
        final_size = destination.stat().st_size
        final_checksum = _sha256_of_file(destination)  # hash the COPY, not the source

    record = ArtifactRecord(
        artifact_id=artifact_id,
        project_id=project.project_id,
        kind="animation",
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
