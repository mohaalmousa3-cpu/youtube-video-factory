"""Phase 1D generation-only slice: standalone limb-sway clip generation.

Composes local, already-validated functions —
src.render.character_rig.apply_limb_sway() and
src.render.ffmpeg_render.assemble_frame_sequence() — into one orchestration
step: an already-registered scene visual PNG + an already-registered scene
audio WAV + a hand-measured LimbBox + a formula-generated sway curve ->
one PNG frame per angle -> one audio-muxed output MP4. Neither underlying
function is modified by this module. Entirely independent of
src.core.mouth_animation_generation — no Rhubarb, no mouth cues, no import
of that module anywhere here. No limb+mouth combination in this slice.

This module never touches SQLite, never registers an artifact, never
mutates a manifest, and never performs a lifecycle transition. Registering
the resulting MP4 is a separate, later, explicit use of the existing,
unmodified `register-animation-artifact` command.

Required validation order, nothing is created until every check has passed:
  1. out_path does not already exist.
  2. scene.artifacts.measured_audio_duration_seconds is a real, finite,
     strictly-positive number (never a bool).
  3. raw_limb_box / raw_amplitude_degrees / raw_frame_count /
     raw_sway_cycles all parse under their strict ASCII grammars (see
     _ASCII_INTEGER_TOKEN_RE / _ASCII_DECIMAL_TOKEN_RE below) and satisfy
     their range rules (frame_count >= 2, amplitude > 0, cycles > 0).
  4. the visual image opens/verifies locally with Pillow and the parsed
     LimbBox's rectangle AND pivot point both fit inside it.
  5. source_audio_path is assumed already validated safe/existing/regular/
     non-empty by the caller (the CLI) — not re-checked here. It is never
     copied — unlike mouth-animation generation, nothing here calls
     Rhubarb, so there is no sidecar-file problem to contain; the ORIGINAL
     source_audio_path is what reaches assemble_frame_sequence().
  6. ffmpeg_render.health_check() is True — a pure, side-effect-free probe
     (one `ffmpeg -version` subprocess call, never raises) checked last,
     immediately before the temp directory is created. Closes a real gap:
     without this, an unavailable ffmpeg binary would only be discovered
     inside assemble_frame_sequence()'s own subprocess call, by which
     point a temp directory and every frame PNG had already been created
     — and ffmpeg_render._run() does not catch the resulting
     FileNotFoundError, so it would reach the CLI as a raw, unsanitized
     exception instead of a clean LimbSwayGenerationError.

Only after 1-6 pass does this function create the private
tempfile.mkdtemp() directory (always removed in `finally`). Angles and
per-frame durations are then generated (pure, no I/O):
  duration_per_frame = measured_audio_duration_seconds / frame_count
  angle_i = amplitude_degrees * sin(2*pi*cycles*i / (frame_count - 1))
  for i in 0..frame_count-1 — deliberately using (frame_count - 1) as the
  denominator so the curve spans exactly `cycles` full periods from i=0 to
  i=frame_count-1: for an INTEGER cycles value this makes the first and
  last angles both (approximately, subject to float rounding) zero; for a
  fractional cycles value they generally do not land on zero.

apply_limb_sway() writes one PNG frame per angle into the private tempdir;
assemble_frame_sequence() is given the ORIGINAL source_audio_path (never
a copy) for the actual audio mux — it always calls
ffmpeg_render.mux_audio_video() internally, so there is no silent-only
output path here either.

Cleanup proof: because check 1 already proved out_path did not exist
before this function touched anything, any file found at out_path once
generation has begun was necessarily written by THIS SAME call's
assemble_frame_sequence() — deleting it there can never remove a
pre-existing caller file. This holds for ANY exception raised from
tempdir creation onward (angle generation, frame generation, or ffmpeg
assembly — ProviderError, OSError, or anything else), via a success-flag
+ `finally` (never a bare `except Exception`), so a genuinely unexpected
exception type still propagates unchanged after cleanup rather than being
silently absorbed.

Errors are translated into one domain exception, LimbSwayGenerationError,
with a short, fixed message per failure category — never a wrapped
exception's own raw text (which may carry subprocess stderr), never a
resolved filesystem path, never manifest/prompt content, and never a raw
traceback."""
from __future__ import annotations

import math
import re
import shutil
import tempfile
from pathlib import Path
from typing import List

from PIL import Image, UnidentifiedImageError

from src.models.scene import ScenePlanItem
from src.providers.base import ProviderError
from src.render.character_rig import LimbBox, apply_limb_sway
from src.render.ffmpeg_render import assemble_frame_sequence, health_check

# Same strict-ASCII philosophy already proven for --mouth-box: [0-9] is
# always literal ASCII regardless of Python's default Unicode regex mode,
# and \Z (not $) rejects a trailing newline that $ would otherwise match
# just before. Integers: no sign, no whitespace, no underscore grouping,
# leading zeros allowed. Decimals: an optional '.', but at least one digit
# required on each side of it if present; no sign (amplitude/cycles are
# both required > 0, so a sign is never part of the grammar at all, not
# merely rejected afterward); no exponent notation — deliberately excluded,
# see this module's docstring/the design review for the justification.
_ASCII_INTEGER_TOKEN_RE = re.compile(r"^[0-9]+\Z")
_ASCII_DECIMAL_TOKEN_RE = re.compile(r"^[0-9]+(\.[0-9]+)?\Z")


class LimbSwayGenerationError(Exception):
    """Raised for any reason build_limb_sway_clip() cannot produce
    out_path: an already-existing output, missing/invalid measured timing,
    an invalid --limb-box/--sway-amplitude-degrees/--frame-count/
    --sway-cycles, a box or pivot that doesn't fit the image, an
    unavailable ffmpeg binary, or a failure from frame generation/assembly
    itself. Message text is always a short, fixed, sanitized string —
    never raw subprocess stderr, a resolved filesystem path, or
    manifest/prompt content."""


def _validate_measured_duration(scene: ScenePlanItem) -> float:
    value = scene.artifacts.measured_audio_duration_seconds
    if value is None:
        raise LimbSwayGenerationError("scene has no measured_audio_duration_seconds")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LimbSwayGenerationError("scene measured_audio_duration_seconds is not a valid number")
    if not math.isfinite(value):
        raise LimbSwayGenerationError("scene measured_audio_duration_seconds is not finite")
    if value <= 0:
        raise LimbSwayGenerationError("scene measured_audio_duration_seconds must be positive")
    return float(value)


def parse_limb_box(raw: str) -> LimbBox:
    """Parse '--limb-box LEFT,TOP,RIGHT,BOTTOM,PIVOT_X,PIVOT_Y'. Exactly
    six comma-separated tokens, each matching the strict ASCII integer
    grammar (no sign/whitespace/underscore/decimal/Unicode-digit). Does
    not know about — and cannot check — any image; see
    validate_image_and_limb_box() for the bounds check."""
    parts = raw.split(",")
    if len(parts) != 6:
        raise LimbSwayGenerationError(
            "--limb-box must be exactly six comma-separated integers: "
            "LEFT,TOP,RIGHT,BOTTOM,PIVOT_X,PIVOT_Y"
        )
    for part in parts:
        if not _ASCII_INTEGER_TOKEN_RE.match(part):
            raise LimbSwayGenerationError("--limb-box values must be base-10 integers")
    left, top, right, bottom, pivot_x, pivot_y = (int(part, base=10) for part in parts)
    return LimbBox(left=left, top=top, right=right, bottom=bottom, pivot_x=pivot_x, pivot_y=pivot_y)


def parse_frame_count(raw: str) -> int:
    """Parse '--frame-count'. Strict ASCII integer grammar, must be >= 2
    (a sway curve needs at least a start and end point)."""
    if not _ASCII_INTEGER_TOKEN_RE.match(raw):
        raise LimbSwayGenerationError("--frame-count must be a base-10 integer")
    value = int(raw, base=10)
    if value < 2:
        raise LimbSwayGenerationError("--frame-count must be at least 2")
    return value


def _parse_positive_finite_decimal(raw: str, field_name: str) -> float:
    """Parse a strict ASCII decimal token (see _ASCII_DECIMAL_TOKEN_RE)
    shared by --sway-amplitude-degrees and --sway-cycles: no sign, no
    exponent notation, optional '.', at least one digit on each side of
    it if present. Requires the parsed value to be finite (defensive,
    beyond what the grammar alone guarantees — an absurdly long all-digit
    string could in principle overflow float() to inf) and strictly
    positive."""
    if not _ASCII_DECIMAL_TOKEN_RE.match(raw):
        raise LimbSwayGenerationError(f"--{field_name} must be a valid decimal number")
    value = float(raw)
    if not math.isfinite(value):
        raise LimbSwayGenerationError(f"--{field_name} must be finite")
    if value <= 0:
        raise LimbSwayGenerationError(f"--{field_name} must be positive")
    return value


def parse_amplitude_degrees(raw: str) -> float:
    return _parse_positive_finite_decimal(raw, "sway-amplitude-degrees")


def parse_sway_cycles(raw: str) -> float:
    return _parse_positive_finite_decimal(raw, "sway-cycles")


def validate_image_and_limb_box(source_visual_path: Path, limb_box: LimbBox) -> tuple[int, int]:
    """Open `source_visual_path` locally with Pillow and confirm
    `limb_box`'s rectangle AND pivot point both lie inside it. Returns
    (image_width, image_height) on success; raises LimbSwayGenerationError
    otherwise. Never registers, transforms, or writes anything.

    Two separate Image.open() calls, deliberately: `.verify()` checks file
    integrity without decoding pixel data, but per Pillow's own contract
    the Image object is unusable for anything else once verify() returns —
    so dimensions are read from a second, fresh open() afterward (a cheap
    header read, not a full decode). Same pattern already proven in
    src/core/mouth_animation_preflight.py, reimplemented here locally
    rather than imported since LimbBox's geometry (including the
    pivot-inside-the-box rule) differs from MouthBox's."""
    try:
        with Image.open(source_visual_path) as img:
            img.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise LimbSwayGenerationError(
            "registered visual artifact image could not be opened or is corrupt"
        ) from exc

    try:
        with Image.open(source_visual_path) as img:
            image_width, image_height = img.size
    except (UnidentifiedImageError, OSError) as exc:
        raise LimbSwayGenerationError(
            "registered visual artifact image could not be opened or is corrupt"
        ) from exc

    if image_width <= 0 or image_height <= 0:
        raise LimbSwayGenerationError("registered visual artifact image has non-positive dimensions")

    if not (0 <= limb_box.left < limb_box.right <= image_width):
        raise LimbSwayGenerationError("--limb-box rectangle extends outside the image bounds")
    if not (0 <= limb_box.top < limb_box.bottom <= image_height):
        raise LimbSwayGenerationError("--limb-box rectangle extends outside the image bounds")
    if not (limb_box.left <= limb_box.pivot_x < limb_box.right):
        raise LimbSwayGenerationError("--limb-box pivot is outside the box bounds")
    if not (limb_box.top <= limb_box.pivot_y < limb_box.bottom):
        raise LimbSwayGenerationError("--limb-box pivot is outside the box bounds")

    return image_width, image_height


def _generate_angles(amplitude_degrees: float, frame_count: int, cycles: float) -> List[float]:
    """angle_i = amplitude * sin(2*pi*cycles*i / (frame_count - 1)) for
    i in 0..frame_count-1. Using (frame_count - 1) as the denominator
    makes the curve span exactly `cycles` full periods from the first to
    the last frame — for an integer `cycles`, both the first (i=0) and
    last (i=frame_count-1) angles land on (approximately, subject to
    float rounding) zero; for a fractional `cycles` they generally do
    not."""
    angles: List[float] = []
    for i in range(frame_count):
        angle = amplitude_degrees * math.sin(2 * math.pi * cycles * i / (frame_count - 1))
        if not math.isfinite(angle):
            raise LimbSwayGenerationError("generated sway angle is not finite")
        angles.append(angle)
    return angles


def _generate_durations(measured_audio_duration_seconds: float, frame_count: int) -> List[float]:
    duration_per_frame = measured_audio_duration_seconds / frame_count
    return [duration_per_frame] * frame_count


def build_limb_sway_clip(
    scene: ScenePlanItem,
    source_visual_path: Path,
    source_audio_path: Path,
    raw_limb_box: str,
    raw_amplitude_degrees: str,
    raw_frame_count: str,
    raw_sway_cycles: str,
    out_path: Path,
) -> Path:
    """Generate one audio-muxed limb-sway MP4 clip at `out_path` from
    `source_visual_path` and `source_audio_path` (both read-only, never
    modified) using `scene`'s own measured_audio_duration_seconds and the
    raw --limb-box/--sway-amplitude-degrees/--frame-count/--sway-cycles
    strings. Raises LimbSwayGenerationError and leaves no new file behind
    on any rejection or failure. Never registers, saves a manifest, or
    touches SQLite — see this module's docstring for the full validation
    order and cleanup proof. Entirely independent of mouth-animation
    generation — no Rhubarb, no mouth cues, no shared code path."""
    if out_path.exists():
        raise LimbSwayGenerationError("--output already exists")

    duration = _validate_measured_duration(scene)

    limb_box = parse_limb_box(raw_limb_box)
    amplitude_degrees = parse_amplitude_degrees(raw_amplitude_degrees)
    frame_count = parse_frame_count(raw_frame_count)
    cycles = parse_sway_cycles(raw_sway_cycles)

    validate_image_and_limb_box(source_visual_path, limb_box)

    if not health_check():
        raise LimbSwayGenerationError("ffmpeg is unavailable (binary not found)")

    tempdir = Path(tempfile.mkdtemp(prefix="limb-sway-"))
    generation_succeeded = False
    try:
        angles = _generate_angles(amplitude_degrees, frame_count, cycles)
        durations = _generate_durations(duration, frame_count)

        try:
            frame_paths = apply_limb_sway(source_visual_path, limb_box, angles, tempdir)
        except OSError as exc:
            raise LimbSwayGenerationError("limb sway frame generation failed") from exc

        try:
            # The ORIGINAL source_audio_path, never a copy — no Rhubarb
            # call anywhere in this module, so there is no sidecar file
            # to contain and no reason to copy the audio at all.
            assemble_frame_sequence(frame_paths, durations, source_audio_path, out_path)
        except ProviderError as exc:
            raise LimbSwayGenerationError("limb sway render failed") from exc

        generation_succeeded = True
    finally:
        if not generation_succeeded and out_path.exists():
            try:
                out_path.unlink()
            except OSError:
                pass
        shutil.rmtree(tempdir, ignore_errors=True)

    return out_path
