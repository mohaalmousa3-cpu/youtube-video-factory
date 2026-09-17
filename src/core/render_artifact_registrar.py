"""Phase 2G: registration-first, project-level Render Artifact Registration.

Registers a single, already-produced local MP4 video file as THE project's
canonical "render" artifact, so a project can later satisfy Phase 2C's
verify-and-advance(..., to_stage="rendered", ...) requirement of one
verified render artifact for the whole project. This module never renders
or encodes video (no ffmpeg *encode*, no provider call), never mutates the
manifest or ProjectRecord.manifest_fingerprint, and never advances a
project's lifecycle stage — see src/core/verified_transition_service.py
for that. It is also stage-agnostic: it never checks or cares what
current_stage the project is in.

Unlike Phase 2D/2E/2F's per-scene registrars, "render" is a PROJECT-level
ArtifactKind (src/models/artifact.py's PROJECT_LEVEL_ARTIFACT_KINDS), so
this module takes no scene_id at all — there is exactly one render per
project, fixed and deterministic:

    artifact_id:    render-final
    kind:           render
    scene_id:       None
    relative_path:  render/final.mp4

scene_id MUST be None here — this is not a convention this module chooses,
it is enforced by ArtifactRecord itself (src/models/artifact.py's
_scene_association_matches_kind model validator raises ValueError if a
project-level kind carries a scene_id), matching the existing, deliberate
test precedent in tests/test_artifact_models.py
(test_valid_render_artifact_is_project_level /
test_project_level_kind_forbids_scene_id). Existing-registration lookups
accordingly use src/database/artifact_repository.list_artifacts_by_project()
(filtered to kind="render"), not list_artifacts_by_scene() — there is no
scene to scope by.

Local filesystem and local SQLite only, plus two local, read-only ffprobe
subprocess calls (identical in kind to audio_artifact_registrar.py /
animation_artifact_registrar.py's approach, duplicated again here rather
than shared) to validate the source file before ever copying it:
  1. src/render/ffmpeg_render.get_duration_seconds() — imported and used
     UNMODIFIED; a corrupt/undecodable file fails here.
  2. a video-STREAM check local to this module (not added to
     ffmpeg_render.py) — duration alone does not prove the file contains
     video content at all (a real, valid, audio-only .mp4/.m4a reports a
     perfectly good duration); this module additionally requires at least
     one stream with codec_type "video".
This phase deliberately does NOT compare the measured duration against
manifest.target_duration_seconds or any other estimate — only the real,
measured value is ever recorded, never validated against a target; that
would be a policy check, not a structural safety one, and does not belong
in a registration-first module.

No LLM, TTS, image generation, Flow, Veo, renderer *invocation*, YouTube,
network/API, or payment call anywhere in this module.

Write paths, exactly three — identical shape to the three prior registrars:
  1. mkdir <project_dir>/render/ (only once every validation has passed)
  2. copy the source file to <project_dir>/render/final.mp4 (only for a
     genuinely fresh registration — never when the destination already has
     byte-identical content)
  3. register_artifact() — exactly one INSERT, and only after (1) and (2)
     (if needed) have already succeeded

Every ordinary rejection (invalid source file, unmeasurable or audio-only
video, an existing record/file with different content, a documented
register_artifact() failure — ArtifactAlreadyExistsError,
DuplicateArtifactRegistrationError, ArtifactProjectNotFoundError,
ArtifactRegistrationError, or any sqlite3.Error, ...) is reported via the
returned RenderArtifactRegistrationResult with ok=False — this module does
not raise for those. It raises RenderArtifactRegistrationError only for a
caller/input inconsistency it will not silently work around (a manifest
that does not belong to the given project), the same split
src/core/verified_transition_service.py and the three prior registrars
use. An UNEXPECTED exception from register_artifact() (anything outside
that documented set) is a different case: this call's own freshly-copied
file (never a pre-existing one) is still cleaned up best-effort, but the
exception itself is re-raised unchanged rather than folded into a result —
an undocumented failure is a bug to surface, not an ordinary outcome to
report.

Deliberately NOT shared with audio/visual/animation_artifact_registrar.py
in this phase: _sha256_of_file() and the fresh-copy cleanup helper below
are intentional, small, self-contained duplicates (a fourth copy now), not
imports — a scoped choice, not an oversight; a shared-helper extraction is
left for a later, deliberate maintenance refactor, same as Phase 2E/2F's
own stated choice."""
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
from src.providers.base import ProviderError
from src.render.ffmpeg_render import get_duration_seconds
from src.utils.config import get_settings

_CHUNK_SIZE = 1024 * 1024

_ARTIFACT_ID = "render-final"
_RELATIVE_PATH = "render/final.mp4"


class RenderArtifactRegistrationError(Exception):
    """Raised only for a project/manifest identity mismatch — a caller
    bug, not a normal registration outcome. Ordinary rejections (invalid
    source file, unmeasurable or audio-only video, a conflicting existing
    record/file, ...) are reported via a returned
    RenderArtifactRegistrationResult instead; see this module's
    docstring."""


@dataclass(frozen=True)
class RenderArtifactRegistrationResult:
    """The one typed result register_render_artifact() always returns,
    success or failure. Never itself mutates the manifest or a
    ProjectRecord's lifecycle state — it only reports what this call
    already did (or refused to do) to the artifact registry and, at most,
    one file."""

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


def _has_video_stream(media_path: Path) -> bool:
    """Local, read-only ffprobe query for whether `media_path` contains at
    least one video stream — same ffprobe-path derivation pattern
    get_duration_seconds() uses, duplicated here rather than added to
    ffmpeg_render.py. Returns False (never raises) on any ffprobe
    failure — the caller has already confirmed the file is valid,
    probeable media via get_duration_seconds() before this is ever
    called, so a failure here would be unexpected, but this function's
    only job is a yes/no stream check, not error reporting."""
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
    (None, error_message) on any failure. Deliberately never compares the
    measured duration against manifest.target_duration_seconds or any
    other estimate — see this module's docstring."""
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


class RenderSourceValidationError(Exception):
    """Raised by prevalidate_render_source() when `source_file` cannot be
    validated as playable video with the required stream layout — exactly
    the same checks register_render_artifact() has always performed via
    _validate_video() above, just made callable on their own, before any
    SQLite connection is opened."""


class RenderSourceMismatchError(Exception):
    """Raised by register_render_artifact() when a supplied `prevalidated`
    result does not identify the same file as `source_file`, or no longer
    matches its current content — e.g. `prevalidated` was computed for a
    different file entirely (even one with byte-identical content), or the
    same file was replaced/modified after prevalidate_render_source() ran.
    register_render_artifact() never trusts a caller-supplied prevalidated
    result without first confirming `prevalidated.source_path` and
    `source_file` resolve to the same real path, THEN re-confirming size
    and checksum via a fresh, local, non-subprocess file read — this is
    the "caller/input inconsistency it will not silently work around"
    case, the same class of failure RenderArtifactRegistrationError
    documents for the project/manifest identity check above."""


@dataclass(frozen=True)
class PrevalidatedRenderSource:
    """The result of prevalidate_render_source(): local file/media facts
    about one render source file, computed WITHOUT any open SQLite
    connection. Passing this into register_render_artifact()'s
    `prevalidated` parameter lets a caller (see
    src.core.final_video_assembly) skip re-running ffprobe — a subprocess
    call — while a database connection is open; register_render_artifact()
    still re-confirms the file's current size and checksum (pure, local,
    no subprocess) before trusting any of these values, rather than
    trusting an arbitrary caller-supplied result outright."""

    source_path: Path
    byte_size: int
    sha256_checksum: str
    duration_seconds: float


def prevalidate_render_source(source_file: Path) -> PrevalidatedRenderSource:
    """Pure, local, no SQLite connection anywhere in this function:
    validate `source_file` exactly as register_render_artifact() has
    always validated a render source (existence, regular file, non-empty,
    measurable duration, at least one video stream) and return its
    measured facts. Raises RenderSourceValidationError on any validation
    failure, with the same sanitized, fixed-shape messages
    register_render_artifact()'s own rejections already use — never raw
    ffprobe stderr."""
    if not source_file.exists():
        raise RenderSourceValidationError(f"source file does not exist at {source_file}")
    if not source_file.is_file():
        raise RenderSourceValidationError(f"source file at {source_file} is not a regular file")
    size = source_file.stat().st_size
    if size == 0:
        raise RenderSourceValidationError(f"source file at {source_file} is empty")
    checksum = _sha256_of_file(source_file)

    duration_seconds, video_error = _validate_video(source_file)
    if video_error is not None:
        raise RenderSourceValidationError(video_error)

    return PrevalidatedRenderSource(
        source_path=source_file.resolve(),
        byte_size=size,
        sha256_checksum=checksum,
        duration_seconds=duration_seconds,
    )


class RenderArtifactCleanupError(Exception):
    """Raised by register_render_artifact() only when `strict_cleanup=True`
    (used exclusively by src.core.final_video_assembly's FINAL VIDEO
    ASSEMBLY call path) AND this call's own freshly-copied canonical
    render/final.mp4 copy could not be removed after a registration
    failure. Every other caller (the default `strict_cleanup=False`) keeps
    _cleanup_fresh_copy()'s original best-effort, silently-swallowed
    behavior unchanged. The canonical orphan path is named in this
    error's own message (sanitized — never raw OS error text beyond
    __cause__); the original registration failure that triggered cleanup
    is preserved via an attached add_note() (naming its type and
    message), and the cleanup OSError itself is preserved as __cause__ —
    so a caller (or FINAL VIDEO ASSEMBLY, via its own
    FinalArtifactCleanupError wrapper) never has to choose which of the
    two failures to report; a plain rejected RenderArtifactRegistrationResult
    is never returned when cleanup actually failed."""


def _cleanup_fresh_copy(destination: Path, created_fresh_copy: bool, *, strict: bool = False) -> None:
    """Delete `destination` only if THIS call is the one that just created
    it fresh — never a pre-existing file the call found already in place.

    Default (`strict=False`, every existing caller): best-effort, swallows
    any failure while deleting so cleanup itself can never mask whatever
    original exception/rejection triggered it — unchanged from this
    function's original behavior.

    `strict=True` (used only via register_render_artifact(...,
    strict_cleanup=True)): a deletion failure is never swallowed — it
    propagates as an OSError for the caller to fold into a typed
    RenderArtifactCleanupError, so an orphaned, unregistered canonical
    file left behind by a failed registration can never go undisclosed."""
    if not created_fresh_copy:
        return
    if strict:
        destination.unlink(missing_ok=True)
        return
    try:
        destination.unlink(missing_ok=True)
    except OSError:
        pass


def register_render_artifact(
    conn: sqlite3.Connection,
    project: ProjectRecord,
    manifest: VideoManifest,
    source_file: Path,
    *,
    now: datetime | None = None,
    prevalidated: PrevalidatedRenderSource | None = None,
    metadata_overrides: Mapping[str, str | int | float | bool | None] | None = None,
    strict_cleanup: bool = False,
) -> RenderArtifactRegistrationResult:
    """Register `source_file` as this project's canonical, project-level
    "render" artifact. Writes nothing to SQLite or the filesystem on any
    rejection path — see this module's docstring for the exact three
    write points a successful, non-idempotent call may reach.
    Stage-agnostic: never reads or checks project.current_stage. Takes no
    scene_id — "render" is project-level, not scene-level; see this
    module's docstring.

    `prevalidated` (optional, keyword-only): when supplied, skips
    re-running the ffprobe-based video validation this function would
    otherwise perform, using `prevalidated.duration_seconds` instead —
    for a caller (see src.core.final_video_assembly) that already ran
    prevalidate_render_source() on `source_file` before opening this
    call's SQLite connection, so no ffprobe/ffmpeg subprocess call ever
    happens while that connection is open. Before any of
    `prevalidated`'s fields are trusted, this function first resolves
    both `prevalidated.source_path` and `source_file` (`Path.resolve()`,
    the same symlink-following/normalization convention this module and
    prevalidate_render_source() already use elsewhere) and compares them
    case-insensitively (`os.path.normcase`, matching this codebase's
    established Windows-safe path-comparison convention — see
    src.core.final_video_assembly's own `_load_assembly_plan()`); a
    prevalidation result computed for a different file — even one whose
    bytes currently happen to match — is rejected via
    RenderSourceMismatchError before its `duration_seconds` (or any other
    field) is ever read. Only once path identity is confirmed are the
    size and checksum this function already computes for `source_file`
    compared against `prevalidated.byte_size`/`.sha256_checksum` (a pure,
    local, non-subprocess file read — never a raw boolean trusted
    blindly); a mismatch there raises RenderSourceMismatchError too,
    rather than silently re-validating or falling back. Every existing
    caller that omits this parameter (the default, `None`) gets exactly
    the same behavior as before this parameter existed — full ffprobe
    validation, every time.

    `metadata_overrides` (optional, keyword-only): merged into (and
    taking precedence over) the default `{"duration_seconds": ...,
    "source": "external"}` metadata this function has always recorded —
    for a caller that wants additional scalar-only metadata fields (e.g.
    scene_count, a manifest identifier) without this function adding a
    new database column or changing its own default metadata shape for
    every other caller that omits this parameter.

    `strict_cleanup` (optional, keyword-only, default False): when a
    freshly-copied canonical file this call itself just created needs to
    be removed after a registration failure, the default (`False`)
    behavior is unchanged — best-effort, silently swallowed, exactly as
    before this parameter existed. `strict_cleanup=True` (used only by
    src.core.final_video_assembly) instead raises RenderArtifactCleanupError
    if that removal fails, naming the orphaned canonical path, so a failed
    run can never silently leave an unregistered canonical render/final.mp4
    behind — see RenderArtifactCleanupError's own docstring."""
    when = now if now is not None else datetime.now(timezone.utc)
    if manifest.project_id != project.project_id:
        raise RenderArtifactRegistrationError(
            f"manifest project_id {manifest.project_id!r} does not match project {project.project_id!r}"
        )
    if manifest.source_fingerprint != project.manifest_fingerprint:
        raise RenderArtifactRegistrationError(
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
    ) -> RenderArtifactRegistrationResult:
        return RenderArtifactRegistrationResult(
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

    # 1. Already registered? Compare against the INCOMING file's checksum
    # only (never re-derived from whatever currently sits at `destination`
    # on disk) — an identical match is a pure no-op: no file touched, no
    # DB write. A mismatch is rejected outright; Phase 2G never overwrites
    # an existing registration. Project-level lookup (no scene to scope
    # by) — at most one "render" row can ever exist per project by
    # construction (the artifacts table's own unique index on
    # (project_id, kind, COALESCE(scene_id,''), relative_path), with both
    # scene_id and relative_path fixed constants here).
    existing_matches = [a for a in list_artifacts_by_project(conn, project.project_id, kind="render")]
    if existing_matches:
        existing = existing_matches[0]
        if existing.sha256_checksum == source_checksum:
            return _result(ok=True, idempotent=True, artifact=existing)
        return _result(
            ok=False,
            reasons=(
                f"a render artifact is already registered for this project ({existing.artifact_id!r}) "
                "with different content — Phase 2G never overwrites an existing artifact registration",
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
                    "Phase 2G never overwrites an existing file",
                ),
            )
        final_checksum, final_size = dest_checksum, dest_size
    else:
        final_checksum, final_size = None, None  # computed after the copy below

    if prevalidated is not None:
        # Path identity FIRST, before any of prevalidated's other fields
        # are trusted — a prevalidation result computed for a different
        # file must never be accepted merely because its recorded bytes
        # currently happen to match. Pure, local, non-subprocess: both
        # sides resolved (symlink-following, matching this module's own
        # resolve() convention) and compared case-insensitively (Windows
        # path-case safety).
        resolved_source = source_file.resolve()
        if os.path.normcase(str(resolved_source)) != os.path.normcase(str(prevalidated.source_path)):
            raise RenderSourceMismatchError(
                f"prevalidated result was computed for {prevalidated.source_path}, which does not "
                f"resolve to the same file as the source_file supplied for registration "
                f"({source_file}) — refusing to trust a prevalidation result computed for a "
                "different path even if its recorded bytes currently match"
            )
        # Pure, local, non-subprocess reconfirmation — never trust a
        # caller-supplied prevalidated result outright. source_checksum/
        # source_size above were already computed unconditionally, so
        # this is just a comparison, not additional file I/O.
        if source_checksum != prevalidated.sha256_checksum or source_size != prevalidated.byte_size:
            raise RenderSourceMismatchError(
                f"prevalidated result for {prevalidated.source_path} no longer matches the current "
                f"content of {source_file} (size/checksum changed since prevalidation) — refusing to "
                "trust a stale prevalidation result"
            )
        duration_seconds, video_error = prevalidated.duration_seconds, None
    else:
        duration_seconds, video_error = _validate_video(source_file)
    if video_error is not None:
        return _result(ok=False, reasons=(video_error,))

    if final_checksum is None:
        # Fresh path: nothing exists yet at `destination` — create
        # render/ only now, after every validation (including video
        # decode + stream-type validation) has already passed.
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, destination)
        created_fresh_copy = True
        final_size = destination.stat().st_size
        final_checksum = _sha256_of_file(destination)  # hash the COPY, not the source

    metadata = {"duration_seconds": duration_seconds, "source": "external"}
    if metadata_overrides is not None:
        metadata.update(metadata_overrides)

    record = ArtifactRecord(
        artifact_id=_ARTIFACT_ID,
        project_id=project.project_id,
        kind="render",
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
        # A documented, expected failure mode of register_artifact() — an
        # ordinary rejection, reported via the result, never raised...
        # UNLESS strict_cleanup=True and this call's own freshly-copied
        # canonical file could not then be removed, in which case that
        # combination must never be reported as a normal clean rejection.
        try:
            _cleanup_fresh_copy(destination, created_fresh_copy, strict=strict_cleanup)
        except OSError as cleanup_exc:
            cleanup_error = RenderArtifactCleanupError(
                f"registration was rejected ({exc}) and the freshly-copied canonical file at "
                f"{destination} could not be removed"
            )
            cleanup_error.add_note(f"original registration rejection: {type(exc).__name__}: {exc}")
            raise cleanup_error from cleanup_exc
        return _result(ok=False, duration_seconds=duration_seconds, reasons=(str(exc),))
    except Exception as exc:
        # Anything else is undocumented/unexpected: still clean up a
        # freshly-copied file (never a pre-existing one), but this is a
        # bug to surface, not an ordinary outcome — re-raise unchanged
        # rather than silently downgrading it into a rejected result...
        # unless strict cleanup itself then fails, in which case THAT
        # failure must not be lost either.
        try:
            _cleanup_fresh_copy(destination, created_fresh_copy, strict=strict_cleanup)
        except OSError as cleanup_exc:
            cleanup_error = RenderArtifactCleanupError(
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
