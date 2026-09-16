"""Phase 1D: generation-only scene-audio creation via
tts_kokoro.KokoroProvider.synthesize() — self-hosted, local, no network.
See src/cli.py's cmd_build_scene_audio for the CLI command.

Writes only --output (via the provider itself); never registers an
artifact, saves a manifest, or touches SQLite — see
src/core/audio_artifact_registrar.py's existing, unmodified
register-audio-artifact command for that separate, later step.

Required validation order, nothing is created and no provider is
constructed until every check below has passed:
  1. scene.narration_text, stripped, is non-empty. Pydantic's own
     ScenePlanItem.narration_text field already guarantees min_length=1
     on the raw string, but that does not rule out a whitespace-only
     value surviving .strip() to become empty.
  2. output_path.parent exists and is a directory.
  3. output_path does not already exist. This is not optional:
     KokoroProvider.synthesize() calls soundfile.write() unconditionally,
     with no existence check of its own, so it would silently overwrite a
     pre-existing file if this module did not refuse first.
  4. require_paid_approval("kokoro", proposals_path, is_paid=False) —
     Kokoro is self-hosted/local (no API key, no network call), genuinely
     free, matching src.core.scene_generator's Groq/TokenRouter treatment
     rather than src.core.scene_image_generation's fixed is_paid=True for
     Qwen-Image. The guard is still kept, deliberately, for the same
     consistency reason every other currently-reachable real provider call
     site in this codebase is wired through it. A PaidApprovalRequiredError
     from this call is never caught anywhere in this module — it
     propagates immediately, matching every other guard call site's
     established convention.

Only after 1-4 pass does KokoroProvider get constructed and called.

KokoroProvider.synthesize()'s own try/except only wraps the underlying
Kokoro.create() call — NOT out_path.parent.mkdir() or soundfile.write(),
both of which run afterward, unguarded. A failure in either surfaces as a
raw OSError, not a ProviderError. This module catches both exception types
around construction and the synthesize() call alike, mapping either to the
same sanitized SceneAudioGenerationError — never the wrapped exception's
own raw text (which could carry a local path or library-internal detail).

Cleanup proof: because check 3 already proved output_path did not exist
before this function touched anything, any file found there once
generation has begun was necessarily written by THIS SAME call's provider
invocation — deleting it can never remove a pre-existing caller file. This
holds for any exception raised from the provider call onward (a
ProviderError, an OSError, an ffprobe decode failure, or any other
exception type), via a success-flag + `finally` (never a bare `except
Exception`), matching src.core.scene_image_generation.py's own cleanup
pattern exactly.

Output verification uses the project's existing ffprobe-based duration
helper (src.render.ffmpeg_render.get_duration_seconds) — the same tool
src.core.audio_artifact_registrar.py already relies on for real audio
validation — not the stdlib `wave` module, which has no precedent as a
production validation tool anywhere in this codebase (it appears only in
one test file, as a synthetic fixture generator)."""
from __future__ import annotations

import math
from pathlib import Path

from src.core.cost_guard import require_paid_approval
from src.models.scene import ScenePlanItem
from src.providers.base import ProviderError
from src.providers.tts_kokoro import (
    DEFAULT_VOICE,
    DOCUMENTARY_CLAUSE_PAUSE,
    DOCUMENTARY_SENTENCE_PAUSE,
    DOCUMENTARY_SPEED,
    KokoroProvider,
)
from src.render.ffmpeg_render import get_duration_seconds


class SceneAudioGenerationError(Exception):
    """Raised for any reason build_scene_audio() cannot produce
    output_path: empty narration, an unreachable or already-existing
    --output, a Kokoro provider failure, or a generated file that is not
    valid, finite-duration audio. Message text is always a short, fixed,
    sanitized string. A PaidApprovalRequiredError from require_paid_approval()
    is NOT one of these — it is never caught here, and propagates unchanged;
    see this module's docstring."""


def build_scene_audio(
    scene: ScenePlanItem,
    output_path: Path,
    proposals_path: Path,
    *,
    voice: str = DEFAULT_VOICE,
    speed: float = DOCUMENTARY_SPEED,
    sentence_pause: float = DOCUMENTARY_SENTENCE_PAUSE,
    clause_pause: float = DOCUMENTARY_CLAUSE_PAUSE,
) -> Path:
    """Generate one scene's narration audio at `output_path` from
    `scene.narration_text`. Raises SceneAudioGenerationError and leaves no
    new file behind on any rejection or failure. Raises
    PaidApprovalRequiredError, uncaught, if the kokoro paid-proposal check
    fails. Never registers, saves a manifest, or touches SQLite."""
    narration = scene.narration_text.strip()
    if not narration:
        raise SceneAudioGenerationError("scene narration is empty")

    if not output_path.parent.exists():
        raise SceneAudioGenerationError("--output parent directory does not exist")
    if not output_path.parent.is_dir():
        raise SceneAudioGenerationError("--output parent is not a directory")
    if output_path.exists():
        raise SceneAudioGenerationError("--output already exists")

    require_paid_approval("kokoro", proposals_path, is_paid=False)

    generation_succeeded = False
    try:
        try:
            provider = KokoroProvider()
            provider.synthesize(
                narration,
                output_path,
                voice=voice,
                speed=speed,
                sentence_pause=sentence_pause,
                clause_pause=clause_pause,
            )
        except (ProviderError, OSError) as exc:
            raise SceneAudioGenerationError("Kokoro TTS synthesis failed") from exc

        if not output_path.exists():
            raise SceneAudioGenerationError("provider did not produce an output file")
        if not output_path.is_file():
            raise SceneAudioGenerationError("provider output path is not a regular file")

        try:
            duration = get_duration_seconds(output_path)
        except (ProviderError, KeyError, ValueError, TypeError) as exc:
            # ffprobe can "succeed" (returncode 0) on a zero-byte or
            # headerless file while reporting no parseable duration at
            # all — get_duration_seconds() would then raise KeyError
            # (missing "duration") or ValueError (non-numeric) from its
            # own float(...) conversion, not ProviderError. Folded into
            # the same sanitized outcome as an explicit ffprobe failure:
            # from this module's perspective, unparseable is
            # indistinguishable from undecodable.
            raise SceneAudioGenerationError("generated audio could not be decoded") from exc

        if not (math.isfinite(duration) and duration > 0):
            raise SceneAudioGenerationError("generated audio has an invalid duration")

        generation_succeeded = True
    finally:
        if not generation_succeeded and output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                pass

    return output_path
