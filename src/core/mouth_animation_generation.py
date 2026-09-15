"""Phase 1D generation-only slice: mouth-animated clip generation.

Composes local, already-validated functions — src.providers.lipsync_rhubarb
.get_mouth_cues(), src.render.character_rig.apply_mouth_animation(), and
src.render.ffmpeg_render.assemble_frame_sequence() — into one orchestration
step: an already-registered scene visual PNG + an already-registered scene
audio WAV + a hand-measured MouthBox -> Rhubarb mouth cues -> one PNG frame
per cue -> one audio-muxed output MP4. None of those three underlying
functions is modified by this module. Reuses (imports, never duplicates)
the strict ASCII-only MouthBox grammar and integer geometry already defined
in src.core.mouth_animation_preflight.

This module never touches SQLite, never registers an artifact, never
mutates a manifest, and never performs a lifecycle transition. Registering
the resulting MP4 is a separate, later, explicit use of the existing,
unmodified `register-animation-artifact` command. No limb sway here.

Required validation order, nothing is created until every check that
doesn't need Rhubarb has passed:
  1. out_path does not already exist.
  2. lipsync_rhubarb.health_check() is True.
  3. scene.artifacts.measured_audio_duration_seconds is a real, finite,
     strictly-positive number (never a bool).
  4. raw_mouth_box parses (parse_mouth_box) and its rectangle fits inside
     source_visual_path (validate_image_and_box) — both imported from
     mouth_animation_preflight, unmodified; any MouthAnimationPreflightError
     from either is re-raised as MouthAnimationGenerationError with the
     same message text, so the CLI only ever needs to catch one exception
     type from this module.
  5. source_audio_path is assumed already validated safe/existing/regular/
     non-empty by the caller (the CLI) — not re-checked here.

Only after 1-4 pass does this function create the private
tempfile.mkdtemp() directory — earlier than cue acquisition now,
specifically so Rhubarb never receives the real, registered
source_audio_path. get_mouth_cues() (src/providers/lipsync_rhubarb.py,
unmodified) always writes a `.lipsync.json` SIBLING file next to whatever
path it's given, and never deletes it — passing the real artifact path
directly would permanently leave that sidecar next to the registered
audio inside the project directory. Instead:
  6. source_audio_path is copied (shutil.copyfile, unmodified stdlib) into
     the private tempdir as `source_audio<original suffix>` — a
     byte-identical LOCAL copy, never the original.
  7. get_mouth_cues() is called on THAT COPY, never on source_audio_path —
     so its `.lipsync.json` sidecar lands inside tempdir too, cleaned up
     with everything else. Must return a non-empty list.
  8. every cue is structurally/type/range-validated, then the whole list's
     ordering is validated (first cue starts at ~0.0, starts strictly
     increasing) — see _validate_cues()'s docstring for the exact algorithm
     and why cue["end"] is never used to derive a frame's hold duration.
  9. per-frame hold durations are derived and must each be finite and
     strictly positive — see _derive_durations().

apply_mouth_animation() writes its PNG frames into the same tempdir (no
name collision with the `source_audio.*` copy/sidecar — frames are named
`frame_NNNN.png`). assemble_frame_sequence() is given the ORIGINAL
source_audio_path (never the temp copy) for the actual audio mux, so the
output MP4's audio track is the real, registered file — the copy exists
solely to keep Rhubarb's sidecar contained. assemble_frame_sequence()
always calls ffmpeg_render.mux_audio_video() internally; there is no
silent-only output path.

Cleanup proof: because check 1 already proved out_path did not exist
before this function touched anything, any file found at out_path once
generation has begun was necessarily written by THIS SAME call's
assemble_frame_sequence() — deleting it there can never remove a
pre-existing caller file. This holds for ANY exception raised anywhere
from tempdir creation onward (the audio copy, cue acquisition/validation,
frame generation, or ffmpeg assembly — ProviderError, OSError, or
anything else), via a success-flag + `finally` (never a bare
`except Exception`), so a genuinely unexpected exception type still
propagates unchanged after cleanup rather than being silently absorbed.
The temp audio copy and any Rhubarb sidecar are always inside tempdir,
so the same unconditional `shutil.rmtree(tempdir, ...)` removes them too
— there is nothing outside tempdir/out_path for any failure path to leak.

Errors are translated into one domain exception,
MouthAnimationGenerationError, with a short, fixed message per failure
category — never a wrapped exception's own raw text (which may carry
subprocess stderr), never a resolved filesystem path, never
manifest/prompt content, and never a raw traceback."""
from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path
from typing import List

from src.core.mouth_animation_preflight import (
    MouthAnimationPreflightError,
    parse_mouth_box,
    validate_image_and_box,
)
from src.models.scene import ScenePlanItem
from src.providers.base import ProviderError
from src.providers.lipsync_rhubarb import MouthCue, get_mouth_cues, health_check
from src.render.character_rig import MouthBox, apply_mouth_animation
from src.render.ffmpeg_render import assemble_frame_sequence

_FIRST_START_TOLERANCE_SECONDS = 1e-6


class MouthAnimationGenerationError(Exception):
    """Raised for any reason build_mouth_animated_clip() cannot produce
    out_path: an already-existing output, an unavailable Rhubarb binary,
    missing/invalid measured timing, an invalid --mouth-box or one that
    doesn't fit the image, invalid/unusable Rhubarb cue data, or a failure
    from cue acquisition/frame generation/assembly itself. Message text is
    always a short, fixed, sanitized string — never raw subprocess stderr,
    a resolved filesystem path, or manifest/prompt content."""


def _validate_measured_duration(scene: ScenePlanItem) -> float:
    value = scene.artifacts.measured_audio_duration_seconds
    if value is None:
        raise MouthAnimationGenerationError("scene has no measured_audio_duration_seconds")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MouthAnimationGenerationError("scene measured_audio_duration_seconds is not a valid number")
    if not math.isfinite(value):
        raise MouthAnimationGenerationError("scene measured_audio_duration_seconds is not finite")
    if value <= 0:
        raise MouthAnimationGenerationError("scene measured_audio_duration_seconds must be positive")
    return float(value)


def _validate_cues(cues: List[MouthCue], measured_audio_duration_seconds: float) -> None:
    """Pass 1: per-cue structural/type/range checks, in list order (the
    first invalid cue wins). Pass 2: whole-list ordering checks, reachable
    only once every cue has individually passed pass 1. cue["end"] is used
    ONLY for the end->=start check here — never for timing derivation, see
    _derive_durations()."""
    for cue in cues:
        if "start" not in cue or "end" not in cue or "value" not in cue:
            raise MouthAnimationGenerationError("mouth cue is missing a required field")
        for value in (cue["start"], cue["end"]):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise MouthAnimationGenerationError("mouth cue start/end must be a valid number")
            if not math.isfinite(value):
                raise MouthAnimationGenerationError("mouth cue start/end must be finite")
        if cue["end"] < cue["start"]:
            raise MouthAnimationGenerationError("mouth cue end must not be before its start")
        if not (0 <= cue["start"] < measured_audio_duration_seconds):
            raise MouthAnimationGenerationError(
                "mouth cue start is outside the scene's measured audio duration"
            )

    if abs(cues[0]["start"] - 0.0) > _FIRST_START_TOLERANCE_SECONDS:
        raise MouthAnimationGenerationError("first mouth cue must start at 0.0")

    for i in range(1, len(cues)):
        if not (cues[i]["start"] > cues[i - 1]["start"]):
            raise MouthAnimationGenerationError("mouth cue start times must be strictly increasing")


def _derive_durations(cues: List[MouthCue], measured_audio_duration_seconds: float) -> List[float]:
    """Each frame holds until the NEXT cue's start (never its own "end") so
    that gaps are absorbed and overlaps are irrelevant; the last frame
    holds until the scene's own measured_audio_duration_seconds, not the
    last cue's own "end" value, so the derived total always matches the
    real audio length by construction rather than by hoping Rhubarb's own
    numbers line up."""
    durations: List[float] = []
    for i, cue in enumerate(cues):
        end = cues[i + 1]["start"] if i + 1 < len(cues) else measured_audio_duration_seconds
        duration = end - cue["start"]
        if not math.isfinite(duration) or duration <= 0:
            raise MouthAnimationGenerationError("derived mouth-frame duration must be positive")
        durations.append(duration)
    return durations


def build_mouth_animated_clip(
    scene: ScenePlanItem,
    source_visual_path: Path,
    source_audio_path: Path,
    raw_mouth_box: str,
    out_path: Path,
) -> Path:
    """Generate one audio-muxed mouth-animated MP4 clip at `out_path` from
    `source_visual_path` and `source_audio_path` (both read-only, never
    modified) using `scene`'s own measured_audio_duration_seconds and a
    raw `--mouth-box` string. Raises MouthAnimationGenerationError and
    leaves no new file behind on any rejection or failure. Never
    registers, saves a manifest, or touches SQLite — see this module's
    docstring for the full validation order and cleanup proof."""
    if out_path.exists():
        raise MouthAnimationGenerationError("--output already exists")

    if not health_check():
        raise MouthAnimationGenerationError("Rhubarb lip-sync tool is unavailable (binary not found)")

    duration = _validate_measured_duration(scene)

    try:
        mouth_box: MouthBox = parse_mouth_box(raw_mouth_box)
        validate_image_and_box(source_visual_path, mouth_box)
    except MouthAnimationPreflightError as exc:
        raise MouthAnimationGenerationError(str(exc)) from exc

    tempdir = Path(tempfile.mkdtemp(prefix="mouth-animation-"))
    generation_succeeded = False
    try:
        # A local copy, never the registered source_audio_path itself:
        # get_mouth_cues() always writes a `.lipsync.json` sibling next to
        # whatever path it's given and never deletes it — giving it the
        # real artifact path would permanently leave that sidecar inside
        # the project's audio/ directory. The copy lives in this same
        # tempdir, so its sidecar is cleaned up by the one rmtree below
        # regardless of what happens afterward.
        temp_audio_path = tempdir / ("source_audio" + source_audio_path.suffix)
        try:
            shutil.copyfile(source_audio_path, temp_audio_path)
        except OSError as exc:
            raise MouthAnimationGenerationError("temporary audio preparation failed") from exc

        try:
            cues = get_mouth_cues(temp_audio_path)
        except ProviderError as exc:
            raise MouthAnimationGenerationError("mouth cue generation failed") from exc

        if not cues:
            raise MouthAnimationGenerationError("Rhubarb returned no mouth cues")

        _validate_cues(cues, duration)
        durations = _derive_durations(cues, duration)

        try:
            frame_paths = apply_mouth_animation(source_visual_path, mouth_box, cues, tempdir)
        except OSError as exc:
            raise MouthAnimationGenerationError("mouth frame generation failed") from exc

        try:
            # The ORIGINAL source_audio_path, not the temp copy — the
            # output MP4's audio track must be the real registered file.
            assemble_frame_sequence(frame_paths, durations, source_audio_path, out_path)
        except ProviderError as exc:
            raise MouthAnimationGenerationError("mouth animation render failed") from exc

        generation_succeeded = True
    finally:
        if not generation_succeeded and out_path.exists():
            try:
                out_path.unlink()
            except OSError:
                pass
        shutil.rmtree(tempdir, ignore_errors=True)

    return out_path
