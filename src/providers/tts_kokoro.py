"""TTSProvider: Kokoro (open weights, Apache 2.0, self-hosted) — runs on CPU
via onnxruntime, no GPU/cloud dependency for this step."""
from __future__ import annotations

from pathlib import Path

import soundfile

from src.providers.base import ProviderError
from src.utils.config import get_settings

DEFAULT_VOICE = "af_bella"
SAMPLE_RATE = 24000

# Systemic fix for "estimated scene duration vs actual TTS duration" drift
# (video2: script table guessed 540s total, real narration came out to 249s)
# — rather than hand-estimating seconds per scene and hoping Kokoro matches,
# calibrate the reading itself to a documentary pace and always take
# duration FROM the real audio, never from a guess. speed<1.0 slows the
# words themselves; the pause bumps add breathing room between
# sentences/clauses that a written script implies but bare TTS won't add
# on its own.
DOCUMENTARY_SPEED = 0.92
DOCUMENTARY_SENTENCE_PAUSE = 0.45  # kokoro-onnx default: 0.25
DOCUMENTARY_CLAUSE_PAUSE = 0.18  # kokoro-onnx default: 0.10


class KokoroProvider:
    def __init__(self):
        settings = get_settings()
        if not settings.kokoro_model_path.exists() or not settings.kokoro_voices_path.exists():
            raise ProviderError(
                f"Kokoro model files missing — expected {settings.kokoro_model_path} "
                f"and {settings.kokoro_voices_path}"
            )
        from kokoro_onnx import Kokoro  # deferred: heavy import (onnxruntime)

        self._kokoro = Kokoro(str(settings.kokoro_model_path), str(settings.kokoro_voices_path))

    def health_check(self) -> bool:
        try:
            samples, _ = self._kokoro.create("ok", voice=DEFAULT_VOICE)
            return len(samples) > 0
        except Exception:
            return False

    def synthesize(
        self,
        text: str,
        out_path: Path,
        voice: str = DEFAULT_VOICE,
        speed: float = DOCUMENTARY_SPEED,
        sentence_pause: float = DOCUMENTARY_SENTENCE_PAUSE,
        clause_pause: float = DOCUMENTARY_CLAUSE_PAUSE,
    ) -> Path:
        try:
            samples, sample_rate = self._kokoro.create(
                text,
                voice=voice,
                speed=speed,
                sentence_pause=sentence_pause,
                clause_pause=clause_pause,
            )
        except Exception as exc:
            raise ProviderError(f"Kokoro synthesis failed: {exc}") from exc
        out_path.parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(str(out_path), samples, sample_rate)
        return out_path
