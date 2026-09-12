"""AnimationProvider: Manim — code-drawn, not AI-generated, so the same
scene always looks identical and it costs zero GPU budget."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from src.providers.base import ProviderError

SCENES_DIR = Path(__file__).parent / "scenes"


def health_check() -> bool:
    try:
        import manim  # noqa: F401

        return True
    except ImportError:
        return False


def render_scene(scene_file: Path, scene_class: str, out_dir: Path, quality: str = "l") -> Path:
    """quality: 'l' (low/fast, for iterating) or 'h' (high, for final export)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, "-m", "manim", f"-q{quality}", "--media_dir", str(out_dir), str(scene_file), scene_class],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ProviderError(f"Manim render failed: {result.stderr[-2000:]}")
    rendered = list(out_dir.rglob(f"{scene_class}.mp4"))
    if not rendered:
        raise ProviderError(f"Manim reported success but no {scene_class}.mp4 was found under {out_dir}")
    return rendered[0]
