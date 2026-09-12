"""UpscaleProvider: Real-ESRGAN ncnn-vulkan (MIT, open source binary), running
on the Intel Iris Xe iGPU via Vulkan — replaces ffmpeg's naive scale-before-zoom
in ken_burns_clip with a real AI upscale. Model is `digital-art-4x` (from the
Upscayl project's bundled models): measured ~6x faster than the generic
`upscayl-standard-4x` on this GPU (30s vs 3min for a 1328x1328 image) with
comparable-or-sharper output on this project's flat watercolor/ink style —
see tools/realesrgan-ncnn-vulkan-v0.2.0-windows/models for both."""
from __future__ import annotations

import subprocess
from pathlib import Path

from src.providers.base import ProviderError
from src.utils.config import get_settings

DEFAULT_MODEL = "digital-art-4x"


def health_check() -> bool:
    # not a return-code check: -h exits 127 even on success (a quirk of this
    # binary, confirmed by running it directly — the help text still prints
    # fine). Binary + model files present is what actually matters.
    settings = get_settings()
    model = settings.realesrgan_models_dir / f"{DEFAULT_MODEL}.bin"
    return Path(settings.realesrgan_path).exists() and model.exists()


def upscale_image(image_path: Path, out_path: Path, model: str = DEFAULT_MODEL, scale: int = 4) -> Path:
    settings = get_settings()
    result = subprocess.run(
        [
            settings.realesrgan_path,
            "-i", str(image_path),
            "-o", str(out_path),
            "-n", model,
            "-s", str(scale),
            "-m", str(settings.realesrgan_models_dir),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ProviderError(f"realesrgan-ncnn-vulkan failed: {result.stderr[-2000:]}")
    return out_path
