"""TEXT OVERLAY RENDERER V1: local-only, generation-and-registration
deterministic text-overlay burn-in. See src/cli.py's
cmd_render_text_overlays for the CLI command.

Given a project_id and its enriched manifest, this module reads the
project's already-registered "render" artifact (never an arbitrary
caller-supplied MP4 — src.core.render_artifact_registrar's
register_render_artifact() is the only thing that ever produces one),
flattens every scene's manifest.scene_plan.scenes[*].text_overlays (in
manifest scene order, then per-scene tuple order — the same deterministic
ordering the manifest itself already guarantees), burns each one into the
render as a `drawtext` filter gated by its own timing window, and
registers the result as this project's canonical, project-level
"overlay_render" artifact via src.core.overlay_artifact_registrar — a
DISTINCT artifact from "render" (see that module's docstring for why: the
existing "render" artifact policy forbids more than one "render" per
project, so an overlay-burned copy of it cannot itself BE a second
"render").

V1 SCOPE, DELIBERATE AND NARROW:
  - Consumes ONLY explicit TextOverlay objects already present in
    manifest.scene_plan.scenes[*].text_overlays. Never derives overlay
    text from narration_text, title, topic, scene_type, or any other
    field — that is a DIFFERENT, deliberately deferred phase (see
    docs/spec-v4/TECHNICAL-SPEC-EN.md section 8's "deterministically
    generated from manifest/scene data" language, which describes a
    capability this module does not yet implement).
  - TextOverlay.start_seconds/end_seconds are interpreted as ABSOLUTE
    seconds on the SOURCE RENDER ARTIFACT's own timeline (not
    scene-relative) — a deliberate v1 simplification, since the model
    itself does not document which convention is intended and computing
    a scene's cumulative offset would require re-deriving per-scene
    durations from the (untouched, per this phase's explicit
    constraints) audio/animation artifacts. None supplied defaults to the
    full [0, source_duration] range.
  - Exactly three fixed (style_id, position) pairs are supported:
    "lower_third_primary"/"lower_third", "center_emphasis"/"center",
    "top_label"/"top" — see _STYLES below. Any other style_id, or a
    style_id paired with a mismatched `position`, is rejected.
  - Font is a single fixed, verified-to-exist local path
    (C:\\Windows\\Fonts\\arial.ttf) — no system-font discovery, no
    fontconfig lookup.
  - Overlay text is NEVER interpolated into an ffmpeg filter expression
    string. Each overlay's text is written to its own local temporary
    UTF-8 text file, referenced only via drawtext's `textfile=` option —
    sidesteps the entire class of apostrophe/colon/quote/backslash/
    Unicode/filter-injection escaping problems for arbitrary text
    content; only the (locally-generated, non-adversarial) file PATH
    itself needs filter-syntax escaping, handled by
    _escape_for_filter_value() below.
  - No shell=True anywhere — every subprocess call passes an argument
    list, matching src.render.ffmpeg_render's own convention.

Three-part lifecycle, identical shape to
src.core.final_video_assembly.assemble_final_video():
  A. Short read-only connection: verify the project, load+validate the
     manifest, flatten overlays, resolve the registered "render"
     artifact's file, confirm no "overlay_render" artifact already
     exists, confirm --output does not conflict. Close the connection.
  B. FFmpeg-only work, no database connection open at all: strict-probe
     the render source, validate every overlay's style/timing/text length
     against it, burn all overlays into one ffmpeg invocation, validate
     the result, atomically replace --output only after validation
     passes.
  C. Short read-write connection: prevalidate the placed --output (pure,
     local, no SQLite) BEFORE opening this connection, reconfirm nothing
     else registered an "overlay_render" artifact in the meantime,
     register it, close the connection.

This module never alters, overwrites, unregisters, or replaces: the
"render" artifact's own row, render/final.mp4 itself, any audio/animation
artifact, the source manifest file, or any other pre-existing project
file — it only ever reads the "render" artifact and writes NEW files
(--output and, on success, overlay/final.mp4 via the registrar)."""
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

from src.core.manifest_store import ManifestStoreError, load_manifest
from src.core.overlay_artifact_registrar import (
    OverlayArtifactCleanupError,
    OverlaySourceMismatchError,
    OverlaySourceValidationError,
    prevalidate_overlay_source,
    register_overlay_render_artifact,
)
from src.core.path_safety import resolve_under_project_dir
from src.database.artifact_repository import list_artifacts_by_project
from src.database.db import get_connection, get_readonly_connection
from src.database.project_repository import get_project
from src.models.manifest import VideoManifest
from src.models.overlay import TextOverlay
from src.utils.config import get_settings

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_-]")

# No system-font discovery: one fixed, verified-to-exist local font.
_FONT_PATH = Path(r"C:\Windows\Fonts\arial.ttf")

_MAX_OVERLAY_TEXT_LENGTH = 80
_DURATION_TOLERANCE_SECONDS = 0.25

# The only three (style_id, position) pairs TEXT OVERLAY RENDERER V1
# supports. font_size/margin_px are deliberately fixed, not configurable,
# per the locked v1 scope.
_STYLES: dict[str, dict] = {
    "lower_third_primary": {"position": "lower_third", "font_size": 42, "margin_px": 60},
    "center_emphasis": {"position": "center", "font_size": 54, "margin_px": 0},
    "top_label": {"position": "top", "font_size": 36, "margin_px": 40},
}


class TextOverlayRenderError(Exception):
    """Base class for every reason render_text_overlays() cannot produce
    --output. The CLI catches only this one type. Message text is always
    a short, sanitized string — never raw subprocess stderr or a resolved
    filesystem path beyond what the message needs to be actionable."""


class ProjectNotFoundError(TextOverlayRenderError):
    """No project is registered with the given project_id."""


class ManifestNotFoundError(TextOverlayRenderError):
    """--manifest does not exist or is not a regular file."""


class InvalidManifestError(TextOverlayRenderError):
    """--manifest could not be parsed/validated, or its project_id/
    source_fingerprint does not match the project registry."""


class NoOverlaysToRenderError(TextOverlayRenderError):
    """manifest.scene_plan.scenes[*].text_overlays is empty across every
    scene — there is nothing for this command to burn in. Raised BEFORE
    any ffmpeg work, per the locked v1 scope."""


class SourceRenderNotFoundError(TextOverlayRenderError):
    """No "render" artifact is registered for this project. TEXT OVERLAY
    RENDERER V1 never accepts an arbitrary caller-supplied source MP4 —
    only the project's own registered render."""


class ArtifactFileNotFoundError(TextOverlayRenderError):
    """The registered "render" artifact's recorded relative_path does not
    exist on disk as a regular file."""


class OutputPathConflictError(TextOverlayRenderError):
    """--output already exists, is a directory, or resolves to the
    registered render artifact's own source path."""


class ExistingOverlayArtifactError(TextOverlayRenderError):
    """An "overlay_render" artifact is already registered for this
    project. This module never overwrites or replaces an existing
    registration."""


class UnsupportedOverlayStyleError(TextOverlayRenderError):
    """A TextOverlay's style_id is not one of the three v1-supported
    styles, or its `position` does not match the position that style_id
    requires."""


class OverlayTextTooLongError(TextOverlayRenderError):
    """A TextOverlay's text exceeds the v1 maximum visible length."""


class OverlayTimingOutOfRangeError(TextOverlayRenderError):
    """A TextOverlay's start/end timing falls outside [0, source
    duration], or start > end after defaulting — text must never be
    scheduled outside the source render's own runtime."""


class OverlayFontNotFoundError(TextOverlayRenderError):
    """The fixed, hardcoded overlay font (C:\\Windows\\Fonts\\arial.ttf)
    does not exist on this machine. No fallback/discovery is attempted —
    see this module's own docstring."""


class MediaProbeError(TextOverlayRenderError):
    """A local ffprobe stream-type/duration probe of the render source or
    the burned-in output could not be completed and parsed. Never
    interpreted as "no such stream" — always raised."""


class OverlayDrawError(TextOverlayRenderError):
    """The ffmpeg drawtext burn-in invocation itself failed."""


class InvalidOverlayOutputError(TextOverlayRenderError):
    """The burned-in output is missing, empty, has an unmeasurable/
    non-finite/non-positive duration, its duration does not match the
    source render's duration within tolerance, or it is missing a
    required video or audio stream."""


class OverlayRegistrationError(TextOverlayRenderError):
    """Registering the successful --output file as this project's
    "overlay_render" artifact failed, after --output was already
    atomically placed. This module removes the just-placed --output file
    in that case."""


class OverlayRenderCleanupError(TextOverlayRenderError):
    """A temporary-directory or orphaned-output cleanup step failed after
    a primary render/registration failure — raised, never silently
    swallowed, so a corrupted or incompletely-cleaned-up state is always
    discoverable."""


@dataclass(frozen=True)
class TextOverlayRenderResult:
    """The one typed result render_text_overlays() returns on success."""

    project_id: str
    output_path: Path
    overlay_count: int
    measured_duration_seconds: float
    artifact_id: str


@dataclass(frozen=True)
class _FlattenedOverlay:
    scene_id: str
    index: int
    overlay: TextOverlay


@dataclass(frozen=True)
class _RenderPlan:
    project_id: str
    manifest: VideoManifest
    output_path: Path
    render_source_path: Path
    render_artifact_id: str
    overlays: tuple[_FlattenedOverlay, ...]


def _sanitize_for_filename(scene_id: str, index: int) -> str:
    return _SAFE_FILENAME_RE.sub("_", f"{scene_id}-{index}")


def _flatten_overlays(manifest: VideoManifest) -> tuple[_FlattenedOverlay, ...]:
    """Deterministic order: manifest scene order (already guaranteed
    unique/contiguous by ScenePlan's own validators), then each scene's
    own text_overlays tuple order (already insertion-ordered)."""
    flattened: list[_FlattenedOverlay] = []
    for scene in manifest.scene_plan.scenes:
        for index, overlay in enumerate(scene.text_overlays):
            flattened.append(_FlattenedOverlay(scene_id=scene.scene_id, index=index, overlay=overlay))
    return tuple(flattened)


def _escape_for_filter_value(path: Path) -> str:
    """Windows-safe ffmpeg filter-argument escaping for a local,
    non-adversarial (renderer-generated or fixed-constant) path: forward
    slashes instead of backslashes, and the drive-letter colon escaped —
    verified against a real ffmpeg build during the preceding audit
    (STEP 4): a bare 'C:\\...' fails to parse (colon collides with the
    filter option separator), but 'C\\:/...' inside a single-quoted
    filter value parses correctly. This function is never called with
    caller/manifest-supplied text — only with paths this module itself
    creates or a single fixed constant."""
    text = str(path.resolve()).replace("\\", "/")
    text = text.replace(":", "\\:")
    return text


def _probe_stream_types(media_path: Path) -> frozenset[str]:
    """Strict, fail-closed local ffprobe stream-type probe — same
    contract as src.core.final_video_assembly's own _probe_stream_types()
    and src.core.overlay_artifact_registrar's own copy (deliberately
    duplicated, not imported/shared, matching this codebase's established
    convention for these small ffprobe helpers)."""
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
        raise MediaProbeError(f"ffprobe produced no usable stream information while inspecting {media_path.name!r}")

    stream_types: set[str] = set()
    for stream in streams:
        if not isinstance(stream, dict):
            raise MediaProbeError(f"ffprobe produced an unexpected stream entry while inspecting {media_path.name!r}")
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
        raise MediaProbeError(f"could not run ffprobe to measure duration for {media_path.name!r}") from exc
    if result.returncode != 0:
        raise MediaProbeError(f"ffprobe exited with an error while measuring {media_path.name!r}")
    try:
        return float(json.loads(result.stdout)["format"]["duration"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise MediaProbeError(f"ffprobe produced an unexpected duration payload for {media_path.name!r}") from exc


def _load_render_plan(project_id: str, manifest_path: Path, output_path: Path) -> _RenderPlan:
    """Phase A: the only part of this module that opens SQLite, and only
    for the short duration of this function."""
    if not project_id:
        raise TextOverlayRenderError("project_id must not be empty")

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

        overlays = _flatten_overlays(manifest)
        if not overlays:
            raise NoOverlaysToRenderError(
                f"manifest for project {project_id!r} declares zero text_overlays across every scene"
            )

        project_dir_resolved = Path(project.manifest_path).resolve().parent

        render_artifacts = list_artifacts_by_project(conn, project_id, kind="render")
        if not render_artifacts:
            raise SourceRenderNotFoundError(
                f"no 'render' artifact is registered for project {project_id!r} — "
                "TEXT OVERLAY RENDERER V1 only ever reads the project's own registered render"
            )
        render_artifact = render_artifacts[0]

        render_source_path = resolve_under_project_dir(project_dir_resolved, render_artifact.relative_path)
        if render_source_path is None:
            raise ArtifactFileNotFoundError("registered render artifact path is unsafe")
        if not render_source_path.exists():
            raise ArtifactFileNotFoundError("registered render artifact file is missing on disk")
        if not render_source_path.is_file():
            raise ArtifactFileNotFoundError("registered render artifact path is not a regular file")

        existing_overlay_artifacts = list_artifacts_by_project(conn, project_id, kind="overlay_render")
        if existing_overlay_artifacts:
            raise ExistingOverlayArtifactError(
                f"an overlay_render artifact is already registered for project {project_id!r} "
                f"({existing_overlay_artifacts[0].artifact_id!r}) — this command never replaces an "
                "existing overlay-render registration"
            )

        output_resolved = Path(output_path).resolve()
        if output_resolved.exists():
            raise OutputPathConflictError("--output already exists")
        if output_resolved.is_dir():
            raise OutputPathConflictError("--output is a directory")
        if os.path.normcase(str(output_resolved)) == os.path.normcase(str(render_source_path)):
            raise OutputPathConflictError("--output must not be the same path as the source render artifact")
    finally:
        conn.close()

    return _RenderPlan(
        project_id=project_id,
        manifest=manifest,
        output_path=output_resolved,
        render_source_path=render_source_path,
        render_artifact_id=render_artifact.artifact_id,
        overlays=overlays,
    )


def _position_expr(position: str, margin_px: int) -> tuple[str, str]:
    x_expr = "(w-text_w)/2"
    if position == "top":
        y_expr = f"{margin_px}"
    elif position == "center":
        y_expr = "(h-text_h)/2"
    elif position == "lower_third":
        y_expr = f"h-text_h-{margin_px}"
    else:  # pragma: no cover — unreachable: _STYLES only ever contains these three
        raise UnsupportedOverlayStyleError(f"unsupported overlay position {position!r}")
    return x_expr, y_expr


def _validate_and_build_filters(
    overlays: tuple[_FlattenedOverlay, ...], source_duration: float, tempdir: Path
) -> str:
    """Validates every overlay against the fixed v1 style table, the
    source render's own duration, and the max-text-length limit, then
    writes each overlay's raw text to its own local UTF-8 temp file and
    returns the full chained drawtext filtergraph string. Raises the
    first violation found — never returns a partial/best-effort
    filtergraph."""
    if not _FONT_PATH.exists():
        raise OverlayFontNotFoundError(f"required overlay font not found at {_FONT_PATH}")
    font_escaped = _escape_for_filter_value(_FONT_PATH)

    filter_parts: list[str] = []
    for flat in overlays:
        overlay = flat.overlay
        label = f"scene {flat.scene_id!r} overlay #{flat.index}"

        style = _STYLES.get(overlay.style_id)
        if style is None or style["position"] != overlay.position:
            raise UnsupportedOverlayStyleError(
                f"{label} uses unsupported style_id/position combination "
                f"({overlay.style_id!r}/{overlay.position!r}) — TEXT OVERLAY RENDERER V1 only "
                f"supports {sorted((sid, s['position']) for sid, s in _STYLES.items())}"
            )

        if len(overlay.text) > _MAX_OVERLAY_TEXT_LENGTH:
            raise OverlayTextTooLongError(
                f"{label} text is {len(overlay.text)} characters, exceeding the v1 maximum of "
                f"{_MAX_OVERLAY_TEXT_LENGTH}"
            )

        start = 0.0 if overlay.start_seconds is None else overlay.start_seconds
        end = source_duration if overlay.end_seconds is None else overlay.end_seconds
        if start < 0:
            raise OverlayTimingOutOfRangeError(f"{label} start_seconds must be >= 0")
        if end > source_duration + 1e-6:
            raise OverlayTimingOutOfRangeError(
                f"{label} end_seconds ({end:.3f}) exceeds the source render's duration ({source_duration:.3f})"
            )
        if start > end:
            raise OverlayTimingOutOfRangeError(f"{label} start_seconds must be <= end_seconds after defaulting")

        text_file = tempdir / f"overlay-{_sanitize_for_filename(flat.scene_id, flat.index)}.txt"
        text_file.write_text(overlay.text, encoding="utf-8")
        text_escaped = _escape_for_filter_value(text_file)

        x_expr, y_expr = _position_expr(style["position"], style["margin_px"])
        filter_parts.append(
            f"drawtext=fontfile='{font_escaped}':textfile='{text_escaped}':"
            f"fontsize={style['font_size']}:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=10:"
            f"x={x_expr}:y={y_expr}:enable='between(t,{start:.3f},{end:.3f})'"
        )

    return ",".join(filter_parts)


def _burn_overlays(plan: _RenderPlan, tempdir: Path) -> tuple[Path, float]:
    """Phase B: pure FFmpeg work, no database connection open anywhere in
    this function."""
    source_stream_types = _probe_stream_types(plan.render_source_path)
    if "video" not in source_stream_types:
        raise InvalidOverlayOutputError("source render artifact has no video stream")
    if "audio" not in source_stream_types:
        raise InvalidOverlayOutputError("source render artifact has no audio stream")
    source_duration = _probe_duration_seconds(plan.render_source_path)
    if not math.isfinite(source_duration) or source_duration <= 0:
        raise InvalidOverlayOutputError("source render artifact has an invalid duration")

    vf = _validate_and_build_filters(plan.overlays, source_duration, tempdir)

    ffmpeg = get_settings().ffmpeg_path
    temp_output = tempdir / "overlay-final.mp4"
    result = subprocess.run(
        [
            ffmpeg, "-y",
            "-i", str(plan.render_source_path),
            "-vf", vf,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            str(temp_output),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise OverlayDrawError(f"ffmpeg drawtext burn-in failed for project {plan.project_id!r}")

    if not temp_output.exists():
        raise InvalidOverlayOutputError("drawtext burn-in did not produce an output file")
    if not temp_output.is_file():
        raise InvalidOverlayOutputError("drawtext burn-in output is not a regular file")
    if temp_output.stat().st_size == 0:
        raise InvalidOverlayOutputError("drawtext burn-in output is empty")

    output_stream_types = _probe_stream_types(temp_output)
    if "video" not in output_stream_types:
        raise InvalidOverlayOutputError("burned-in output has no video stream")
    if "audio" not in output_stream_types:
        raise InvalidOverlayOutputError("burned-in output has no audio stream")

    output_duration = _probe_duration_seconds(temp_output)
    if not math.isfinite(output_duration) or output_duration <= 0:
        raise InvalidOverlayOutputError("burned-in output has an invalid duration")
    if abs(output_duration - source_duration) > _DURATION_TOLERANCE_SECONDS:
        raise InvalidOverlayOutputError(
            f"burned-in output duration {output_duration:.3f}s does not match the source render's "
            f"duration {source_duration:.3f}s within tolerance {_DURATION_TOLERANCE_SECONDS:.3f}s"
        )

    return temp_output, output_duration


def render_text_overlays(project_id: str, manifest_path: Path, output_path: Path) -> TextOverlayRenderResult:
    """Burn every manifest-declared TextOverlay into `project_id`'s
    already-registered "render" artifact and register the result as this
    project's canonical "overlay_render" artifact. Raises a
    TextOverlayRenderError subclass and leaves no new file or database row
    behind on any rejection or failure."""
    plan = _load_render_plan(project_id, Path(manifest_path), Path(output_path))

    plan.output_path.parent.mkdir(parents=True, exist_ok=True)
    tempdir = Path(tempfile.mkdtemp(prefix="text-overlay-render-", dir=plan.output_path.parent))

    output_created = False
    registration_succeeded = False
    primary_exc: Exception | None = None
    result = None
    measured_duration = None
    try:
        try:
            temp_output, measured_duration = _burn_overlays(plan, tempdir)

            os.replace(temp_output, plan.output_path)
            output_created = True

            try:
                prevalidated = prevalidate_overlay_source(plan.output_path)
            except OverlaySourceValidationError as exc:
                raise InvalidOverlayOutputError("burned-in output failed pre-registration validation") from exc

            try:
                conn = get_connection()
            except sqlite3.Error as exc:
                raise OverlayRegistrationError("no local project database found") from exc

            try:
                existing = list_artifacts_by_project(conn, plan.project_id, kind="overlay_render")
                if existing:
                    raise ExistingOverlayArtifactError(
                        f"an overlay_render artifact was registered for project {plan.project_id!r} "
                        f"({existing[0].artifact_id!r}) while this command was running"
                    )

                project = get_project(conn, plan.project_id)
                if project is None:
                    raise ProjectNotFoundError(f"unknown project_id {plan.project_id!r}")

                metadata_overrides = {
                    "source": "text-overlay-renderer-v1",
                    "overlay_count": len(plan.overlays),
                    "source_render_artifact_id": plan.render_artifact_id,
                    "manifest_fingerprint": plan.manifest.source_fingerprint,
                }
                try:
                    result = register_overlay_render_artifact(
                        conn,
                        project,
                        plan.manifest,
                        plan.output_path,
                        now=datetime.now(timezone.utc),
                        prevalidated=prevalidated,
                        metadata_overrides=metadata_overrides,
                        strict_cleanup=True,
                    )
                except OverlaySourceMismatchError as exc:
                    raise OverlayRegistrationError(
                        "burned-in output changed unexpectedly between pre-validation and registration"
                    ) from exc
                except OverlayArtifactCleanupError as exc:
                    raise OverlayRenderCleanupError(
                        "overlay artifact registration failed and its own canonical-copy cleanup also failed"
                    ) from exc
                except Exception as exc:
                    raise OverlayRegistrationError("overlay artifact registration failed") from exc

                if not result.ok:
                    raise OverlayRegistrationError(
                        "overlay artifact registration was rejected: " + "; ".join(result.reasons)
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
                    f"additionally, failed to remove temporary overlay-render directory {tempdir}: "
                    f"{type(cleanup_exc).__name__}"
                )
            else:
                raise OverlayRenderCleanupError(
                    f"failed to remove temporary overlay-render directory {tempdir}"
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
                    raise OverlayRenderCleanupError(
                        f"registration failed and the orphaned output at {plan.output_path} "
                        "could not be removed"
                    ) from cleanup_exc

    return TextOverlayRenderResult(
        project_id=plan.project_id,
        output_path=plan.output_path,
        overlay_count=len(plan.overlays),
        measured_duration_seconds=measured_duration,
        artifact_id=result.artifact_id,
    )
