"""LipSyncProvider: Rhubarb Lip Sync (MIT, open source binary) — turns a
narration WAV into mouth-shape timing (the standard Preston Blair viseme
set: A-H plus X for rest) so the character compositor knows which mouth
image to show at each moment."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import List, TypedDict

from src.providers.base import ProviderError
from src.utils.config import get_settings


class MouthCue(TypedDict):
    start: float
    end: float
    value: str  # one of A B C D E F G H X


def health_check() -> bool:
    try:
        result = subprocess.run([get_settings().rhubarb_path, "--version"], capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def get_mouth_cues(audio_path: Path) -> List[MouthCue]:
    out_json = audio_path.with_suffix(".lipsync.json")
    result = subprocess.run(
        [get_settings().rhubarb_path, "-f", "json", "-o", str(out_json), str(audio_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ProviderError(f"Rhubarb failed: {result.stderr[-2000:]}")
    data = json.loads(out_json.read_text(encoding="utf-8"))
    return data["mouthCues"]
