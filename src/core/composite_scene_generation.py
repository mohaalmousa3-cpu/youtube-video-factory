"""Phase 1D: unified mouth + limb-sway clip generation slice. Combines what
src.core.mouth_animation_generation.build_mouth_animated_clip() and
src.core.limb_sway_generation.build_limb_sway_clip() each do independently
into a single shared-timeline pipeline: an already-registered scene visual
PNG + an already-registered scene audio WAV + a hand-measured MouthBox + a
hand-measured LimbBox -> Rhubarb mouth cues -> one PNG frame per cue,
drawing BOTH the mouth viseme and a time-matched limb-sway angle on each ->
one audio-muxed output MP4. See src/cli.py's cmd_build_composite_scene_clip
for the CLI command.

Shared-timeline design (the reason this module exists rather than just
running the two existing commands back to back): mouth_animation_generation
and limb_sway_generation use two incompatible timing models — mouth-frame
durations come from Rhubarb's own real, irregular detected cue boundaries;
limb-frame durations come from a uniform division of the measured audio
duration by an arbitrary operator-chosen --frame-count, unrelated to any
real event. Rather than resampling one onto the other's grid, this module
makes the Rhubarb cue timeline authoritative for BOTH effects: the same
cues drive the mouth exactly like build-mouth-animated-clip already does,
and limb angle is evaluated as a CONTINUOUS function of time (_angle_at_time(),
amplitude*sin(2*pi*cycles*t/total_duration)) sampled once at each cue's own
start timestamp — not resampled from limb_sway_generation's own index-based,
--frame-count-driven _generate_angles(). There is therefore no --frame-count
option on this command at all: frame count is however many cues Rhubarb
actually returns for the real narration, exactly as in
build-mouth-animated-clip.

Reuses (imports, never duplicates) as much validation as possible:
  - parse_mouth_box / validate_image_and_box, from
    src.core.mouth_animation_preflight (the same functions
    mouth_animation_generation.py itself imports).
  - parse_limb_box / parse_amplitude_degrees / parse_sway_cycles /
    validate_image_and_limb_box, from src.core.limb_sway_generation (the
    same functions build_limb_sway_clip() itself uses) — note
    parse_frame_count is deliberately NOT imported or used, per the
    shared-timeline design above.
  - _validate_cues / _derive_durations, from
    src.core.mouth_animation_generation — these encode real audio-timing
    correctness (cue monotonicity, first-cue-at-zero, duration derivation)
    that must behave identically whether mouth animation runs alone or
    combined with limb sway; duplicating them here would risk the two
    behaviors silently drifting apart.

_validate_measured_duration() is NOT reused from either sibling module —
it is small, stable, and already duplicated identically between
mouth_animation_generation.py and limb_sway_generation.py; this module
keeps its own third copy for the same reason those two already diverged
rather than sharing it.

This module never touches SQLite, never registers an artifact, never
mutates a manifest, and never performs a lifecycle transition. Registering
the resulting MP4 is a separate, later, explicit use of the existing,
unmodified register-animation-artifact command.

Required validation order, nothing is created until every check that
doesn't need Rhubarb or ffmpeg has passed:
  1. out_path does not already exist.
  2. lipsync_rhubarb.health_check() is True — checked early (mirroring
     mouth_animation_generation.py's own position for this same check)
     since Rhubarb cue acquisition is structurally required before any
     frame can be generated.
  3. scene.artifacts.measured_audio_duration_seconds is a real, finite,
     strictly-positive number (never a bool).
  4. raw_mouth_box parses and its rectangle fits source_visual_path.
  5. raw_limb_box / raw_amplitude_degrees / raw_sway_cycles all parse
     under their strict ASCII grammars, and the limb box + pivot fit
     source_visual_path.
  6. ffmpeg_render.health_check() is True — checked last, immediately
     before the temp directory is created, mirroring
     limb_sway_generation.py's own position for this same check (avoids
     leaving frame files stranded if ffmpeg turned out to be missing only
     at assembly time).

Only after 1-6 pass does this function create a private tempfile.mkdtemp()
directory and copy source_audio_path into it (same Rhubarb-sidecar-
containment reasoning as mouth_animation_generation.py — get_mouth_cues()
always writes an uncleaned `.lipsync.json` sibling next to whatever path
it's given). Cue acquisition/validation and per-frame mouth durations are
derived exactly as build_mouth_animated_clip() does; limb angle is then
evaluated once per cue via _angle_at_time().
apply_mouth_and_limb_animation() writes one PNG frame per cue (mouth
viseme + limb rotation both drawn onto the same frame); the ORIGINAL
source_audio_path (never the temp copy) is what gets muxed into the
output.

Cleanup proof: identical to mouth_animation_generation.py's — check 1
already proved out_path did not exist before this function touched
anything, so any file found there once generation has begun was
necessarily written by THIS SAME call; a success-flag + `finally` (never
a bare `except Exception`) removes it on any failure from tempdir creation
onward, and the same unconditional `shutil.rmtree(tempdir, ...)` removes
the temp audio copy and any Rhubarb sidecar regardless of outcome.

Errors are translated into one domain exception,
CompositeSceneGenerationError, with a short, fixed message per failure
category — never a wrapped exception's own raw text, a resolved
filesystem path, manifest/prompt content, or a raw traceback. Any
MouthAnimationPreflightError or LimbSwayGenerationError raised by a reused
parser/validator is re-raised as CompositeSceneGenerationError with the
same message text, so the CLI only ever needs to catch one exception type
from this module."""
from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path
from typing import List

from src.core.limb_sway_generation import (
    LimbSwayGenerationError,
    parse_amplitude_degrees,
    parse_limb_box,
    parse_sway_cycles,
    validate_image_and_limb_box,
)
from src.core.mouth_animation_generation import (
    MouthAnimationGenerationError,
    _derive_durations,
    _validate_cues,
)
from src.core.mouth_animation_preflight import (
    MouthAnimationPreflightError,
    parse_mouth_box,
    validate_image_and_box,
)
from src.models.scene import ScenePlanItem
from src.providers.base import ProviderError
from src.providers.lipsync_rhubarb import get_mouth_cues
from src.providers.lipsync_rhubarb import health_check as rhubarb_health_check
from src.render.character_rig import LimbBox, MouthBox, apply_mouth_and_limb_animation
from src.render.ffmpeg_render import assemble_frame_sequence
from src.render.ffmpeg_render import health_check as ffmpeg_health_check


class CompositeSceneGenerationError(Exception):
    """Raised for any reason build_composite_scene_clip() cannot produce
    out_path: an already-existing output, an unavailable Rhubarb or ffmpeg
    binary, missing/invalid measured timing, an invalid --mouth-box/
    --limb-box/--sway-amplitude-degrees/--sway-cycles or one that doesn't
    fit the image, invalid/unusable Rhubarb cue data, or a failure from cue
    acquisition/frame generation/assembly itself. Message text is always a
    short, fixed, sanitized string — never raw subprocess stderr, a
    resolved filesystem path, or manifest/prompt content."""


def _validate_measured_duration(scene: ScenePlanItem) -> float:
    value = scene.artifacts.measured_audio_duration_seconds
    if value is None:
        raise CompositeSceneGenerationError("scene has no measured_audio_duration_seconds")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CompositeSceneGenerationError("scene measured_audio_duration_seconds is not a valid number")
    if not math.isfinite(value):
        raise CompositeSceneGenerationError("scene measured_audio_duration_seconds is not finite")
    if value <= 0:
        raise CompositeSceneGenerationError("scene measured_audio_duration_seconds must be positive")
    return float(value)


def _angle_at_time(amplitude_degrees: float, cycles: float, total_duration: float, t: float) -> float:
    """Continuous time-based counterpart to limb_sway_generation's
    index-based _generate_angles(): angle(t) = amplitude *
    sin(2*pi*cycles*t/total_duration). Evaluated once per Rhubarb cue
    timestamp (this module's shared frame grid) instead of once per an
    independent --frame-count, so limb motion stays synced to the same
    real, audio-derived timeline the mouth already uses."""
    angle = amplitude_degrees * math.sin(2 * math.pi * cycles * t / total_duration)
    if not math.isfinite(angle):
        raise CompositeSceneGenerationError("generated sway angle is not finite")
    return angle


def build_composite_scene_clip(
    scene: ScenePlanItem,
    source_visual_path: Path,
    source_audio_path: Path,
    raw_mouth_box: str,
    raw_limb_box: str,
    raw_amplitude_degrees: str,
    raw_sway_cycles: str,
    out_path: Path,
) -> Path:
    """Generate one audio-muxed MP4 clip at `out_path` combining mouth
    lip-sync and limb sway, from `source_visual_path` and
    `source_audio_path` (both read-only, never modified) using `scene`'s
    own measured_audio_duration_seconds and the raw --mouth-box/
    --limb-box/--sway-amplitude-degrees/--sway-cycles strings. Raises
    CompositeSceneGenerationError and leaves no new file behind on any
    rejection or failure. Never registers, saves a manifest, or touches
    SQLite — see this module's docstring for the full validation order and
    cleanup proof."""
    if out_path.exists():
        raise CompositeSceneGenerationError("--output already exists")

    if not rhubarb_health_check():
        raise CompositeSceneGenerationError("Rhubarb lip-sync tool is unavailable (binary not found)")

    duration = _validate_measured_duration(scene)

    try:
        mouth_box: MouthBox = parse_mouth_box(raw_mouth_box)
        validate_image_and_box(source_visual_path, mouth_box)
    except MouthAnimationPreflightError as exc:
        raise CompositeSceneGenerationError(str(exc)) from exc

    try:
        limb_box: LimbBox = parse_limb_box(raw_limb_box)
        amplitude_degrees = parse_amplitude_degrees(raw_amplitude_degrees)
        cycles = parse_sway_cycles(raw_sway_cycles)
        validate_image_and_limb_box(source_visual_path, limb_box)
    except LimbSwayGenerationError as exc:
        raise CompositeSceneGenerationError(str(exc)) from exc

    if not ffmpeg_health_check():
        raise CompositeSceneGenerationError("ffmpeg is unavailable (binary not found)")

    tempdir = Path(tempfile.mkdtemp(prefix="composite-scene-"))
    generation_succeeded = False
    try:
        temp_audio_path = tempdir / ("source_audio" + source_audio_path.suffix)
        try:
            shutil.copyfile(source_audio_path, temp_audio_path)
        except OSError as exc:
            raise CompositeSceneGenerationError("temporary audio preparation failed") from exc

        try:
            cues = get_mouth_cues(temp_audio_path)
        except ProviderError as exc:
            raise CompositeSceneGenerationError("mouth cue generation failed") from exc

        if not cues:
            raise CompositeSceneGenerationError("Rhubarb returned no mouth cues")

        try:
            _validate_cues(cues, duration)
            durations = _derive_durations(cues, duration)
        except MouthAnimationGenerationError as exc:
            raise CompositeSceneGenerationError(str(exc)) from exc

        angles: List[float] = [
            _angle_at_time(amplitude_degrees, cycles, duration, cue["start"]) for cue in cues
        ]

        try:
            frame_paths = apply_mouth_and_limb_animation(
                source_visual_path, mouth_box, cues, limb_box, angles, tempdir
            )
        except OSError as exc:
            raise CompositeSceneGenerationError("composite frame generation failed") from exc

        try:
            # The ORIGINAL source_audio_path, not the temp copy — the
            # output MP4's audio track must be the real registered file.
            assemble_frame_sequence(frame_paths, durations, source_audio_path, out_path)
        except ProviderError as exc:
            raise CompositeSceneGenerationError("composite scene render failed") from exc

        generation_succeeded = True
    finally:
        if not generation_succeeded and out_path.exists():
            try:
                out_path.unlink()
            except OSError:
                pass
        shutil.rmtree(tempdir, ignore_errors=True)

    return out_path
