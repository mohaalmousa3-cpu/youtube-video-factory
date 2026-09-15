"""Phase 1D vertical slice: upscale-assisted Ken Burns clip generation
(generation-only — see src/cli.py's cmd_build_upscaled_ken_burns for the
CLI command, and docs/spec-v4/IMPLEMENTATION-PLAN.md's Phase 1D for the
scope this slice narrows).

Composes two already-validated, already-unwired local functions —
src.providers.image_upscale.upscale_image() and
src.render.ffmpeg_render.ken_burns_clip() — into one local orchestration
step: an already-registered scene visual PNG -> a private, temporary
upscaled PNG (never persisted or registered) -> ken_burns_clip() -> an
output MP4. Neither underlying function is modified by this module.

This module never touches SQLite, never registers an artifact, never
mutates a manifest, and never performs a lifecycle transition or QC/mux
step. Registering the resulting MP4 is a separate, later, explicit use of
the existing, unmodified `register-animation-artifact` command
(src/core/animation_artifact_registrar.py).

Validation order, deliberately fail-fast, nothing is created until every
check below has passed:
  1. scene.motion_mode != "manual_flow" (Manual Flow is a human/Google-Flow
     step, not something this local ffmpeg path can approximate).
  2. scene.artifacts.measured_audio_duration_seconds is a real, finite,
     strictly-positive number (never a bool, never None, never a table
     guess — see docs/spec-v4/TECHNICAL-SPEC-EN.md section 7).
  3. out_path does not already exist — this command never overwrites a
     caller's file, and this check is also the precondition the cleanup
     proof below relies on.
  4. image_upscale.health_check() is True — never silently falls back to
     the un-upscaled original just because the real upscaler isn't
     available.

Only after all four pass does this function create anything: a private
tempfile.mkdtemp() directory for the intermediate upscaled PNG (always
removed in `finally`, success or failure — the temporary file is a pure
implementation detail of this pipeline, never registered or left behind)
and, via ken_burns_clip(), the final `out_path` MP4.

Cleanup proof: because check 3 already proved `out_path` did not exist
before this function touched anything, any file found at `out_path` once
the ken_burns_clip() call has begun was necessarily written by THIS SAME
call — deleting it there can never remove a pre-existing caller file. This
holds for ANY exception ken_burns_clip() raises, not only ProviderError: a
`finally` around that call (guarded by a success flag, so a completed run
is never touched) unlinks an owned `out_path` on every failure exit —
ProviderError, OSError/FileNotFoundError (e.g. a misconfigured ffmpeg
path), UnicodeDecodeError (malformed subprocess output), or any other
Exception/BaseException — before that exception continues propagating.
`source_visual_path` is never opened for writing anywhere in this module.

ProviderError specifically (from either upscale_image() or ken_burns_clip())
is additionally translated into one domain exception,
KenBurnsUpscalePipelineError, with a short, fixed message per failure
category — never the wrapped ProviderError's own text (which may carry
raw subprocess stderr), never a resolved filesystem path, never
manifest/prompt content, and never a raw traceback. Any other, genuinely
unexpected exception type is NOT wrapped — it propagates as itself (after
the same cleanup above runs), matching this codebase's convention
elsewhere that an undocumented failure is a bug to surface, not an
ordinary outcome to sanitize into a result."""
from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path

from src.models.scene import ScenePlanItem
from src.providers.base import ProviderError
from src.providers.image_upscale import health_check, upscale_image
from src.render.ffmpeg_render import ken_burns_clip


class KenBurnsUpscalePipelineError(Exception):
    """Raised for any reason build_upscaled_ken_burns_clip() cannot produce
    out_path: an unsupported motion_mode, missing/invalid measured timing,
    an --output path that already exists, an unavailable upscaler, or a
    failure from the upscale/render step itself. Message text is always a
    short, fixed, sanitized string — never raw subprocess stderr, a
    resolved filesystem path, or manifest/prompt content."""


def _validate_duration(scene: ScenePlanItem) -> float:
    value = scene.artifacts.measured_audio_duration_seconds
    if value is None:
        raise KenBurnsUpscalePipelineError("scene has no measured_audio_duration_seconds")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise KenBurnsUpscalePipelineError("scene measured_audio_duration_seconds is not a valid number")
    if not math.isfinite(value):
        raise KenBurnsUpscalePipelineError("scene measured_audio_duration_seconds is not finite")
    if value <= 0:
        raise KenBurnsUpscalePipelineError("scene measured_audio_duration_seconds must be positive")
    return float(value)


def build_upscaled_ken_burns_clip(
    scene: ScenePlanItem,
    source_visual_path: Path,
    out_path: Path,
    *,
    width: int = 1664,
    height: int = 928,
    fps: int = 25,
) -> Path:
    """Generate an upscale-assisted Ken Burns MP4 clip at `out_path` from
    `source_visual_path` (read-only, never modified) using `scene`'s own
    motion_mode and measured_audio_duration_seconds. Raises
    KenBurnsUpscalePipelineError and leaves no new file behind on any
    rejection or failure. Never registers an artifact, saves a manifest,
    or touches SQLite — see this module's docstring for the full
    validation order and cleanup proof."""
    if scene.motion_mode == "manual_flow":
        raise KenBurnsUpscalePipelineError(
            "motion_mode 'manual_flow' is not supported by this command (local Ken Burns only)"
        )

    duration = _validate_duration(scene)

    if out_path.exists():
        raise KenBurnsUpscalePipelineError("--output already exists")

    if not health_check():
        raise KenBurnsUpscalePipelineError(
            "Real-ESRGAN upscaler is unavailable (binary or model files not found)"
        )

    tempdir = Path(tempfile.mkdtemp(prefix="ken-burns-upscale-"))
    try:
        upscaled_path = tempdir / "upscaled.png"
        try:
            upscale_image(source_visual_path, upscaled_path)
        except ProviderError as exc:
            raise KenBurnsUpscalePipelineError("image upscale failed") from exc

        ken_burns_succeeded = False
        try:
            ken_burns_clip(
                upscaled_path,
                duration,
                out_path,
                motion=scene.motion_mode,
                width=width,
                height=height,
                fps=fps,
            )
            ken_burns_succeeded = True
        except ProviderError as exc:
            raise KenBurnsUpscalePipelineError("Ken Burns render failed") from exc
        finally:
            # Runs on every exit from the block above except the plain
            # success case (ken_burns_succeeded stays False for a
            # ProviderError re-raised as KenBurnsUpscalePipelineError above,
            # and for any other exception type that isn't caught at all) —
            # `finally` still executes before such an exception propagates,
            # so this covers OSError/FileNotFoundError/UnicodeDecodeError/
            # any other Exception or BaseException, not only ProviderError.
            # Best-effort and never masks the real failure: out_path.exists()
            # is re-checked (it may never have been created at all) and any
            # unlink failure is swallowed, matching every other cleanup
            # helper in this codebase (e.g. the artifact registrars'
            # _cleanup_fresh_copy()).
            if not ken_burns_succeeded and out_path.exists():
                try:
                    out_path.unlink()
                except OSError:
                    pass
    finally:
        shutil.rmtree(tempdir, ignore_errors=True)

    return out_path
