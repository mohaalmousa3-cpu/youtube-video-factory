"""Settings loaded from environment / .env — same pattern as youtube-intelligence-engine."""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _find_ffmpeg() -> str:
    """shutil.which() works once PATH picks up the winget install (new shells
    do); until then, fall back to the known winget install location."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    winget_guess = Path.home() / (
        "AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
        "/ffmpeg-9.0.1-full_build/bin/ffmpeg.exe"
    )
    return str(winget_guess) if winget_guess.exists() else "ffmpeg"


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    tokenrouter_api_key: str
    tokenrouter_base_url: str
    qwen_api_key: str
    ffmpeg_path: str
    rhubarb_path: str
    realesrgan_path: str
    realesrgan_models_dir: Path
    data_dir: Path
    kokoro_model_path: Path
    kokoro_voices_path: Path

    def has_groq_key(self) -> bool:
        return bool(self.groq_api_key)


@lru_cache
def get_settings() -> Settings:
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    models_dir = PROJECT_ROOT / "models"
    return Settings(
        groq_api_key=os.environ.get("GROQ_API_KEY", ""),
        tokenrouter_api_key=os.environ.get("TOKENROUTER_API_KEY", ""),
        tokenrouter_base_url=os.environ.get("TOKENROUTER_BASE_URL", "https://api.tokenrouter.com/v1"),
        qwen_api_key=os.environ.get("QWEN_API_KEY", ""),
        ffmpeg_path=os.environ.get("FFMPEG_PATH") or _find_ffmpeg(),
        rhubarb_path=os.environ.get("RHUBARB_PATH")
        or str(PROJECT_ROOT / "tools/Rhubarb-Lip-Sync-1.14.0-Windows/rhubarb.exe"),
        realesrgan_path=os.environ.get("REALESRGAN_PATH")
        or str(PROJECT_ROOT / "tools/realesrgan-ncnn-vulkan-v0.2.0-windows/realesrgan-ncnn-vulkan.exe"),
        realesrgan_models_dir=PROJECT_ROOT / "tools/realesrgan-ncnn-vulkan-v0.2.0-windows/models",
        data_dir=data_dir,
        kokoro_model_path=models_dir / "kokoro-v1.0.onnx",
        kokoro_voices_path=models_dir / "voices-v1.0.bin",
    )
