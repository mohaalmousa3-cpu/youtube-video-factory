"""Thin subprocess wrapper around the ffmpeg binary — stateless CLI calls,
no need for a class/Provider here."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from src.providers.base import ProviderError
from src.utils.config import get_settings


def _run(args: list[str]) -> None:
    ffmpeg = get_settings().ffmpeg_path
    result = subprocess.run([ffmpeg, "-y", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise ProviderError(f"ffmpeg failed: {result.stderr[-2000:]}")


def pad_audio_to_duration(audio_path: Path, min_duration: float, out_path: Path) -> Path:
    """Second half of the pacing fix (see tts_kokoro's DOCUMENTARY_* constants
    for the first half): a short punchy line like 'Three dots appear. Then...
    nothing.' will never naturally take 12s to *say* no matter how much you
    slow the TTS down — that estimate always implied visual dwell time beyond
    the spoken words. Give a scene an explicit min_duration and this pads
    trailing silence onto the real audio to reach it, so the held frame
    actually holds for as long as the scene plan intended, deterministically
    — not a guess we hope Kokoro's pace happens to match."""
    current = get_duration_seconds(audio_path)
    if current >= min_duration:
        return audio_path
    pad = min_duration - current
    _run(["-i", str(audio_path), "-af", f"apad=pad_dur={pad:.3f}", str(out_path)])
    return out_path


def health_check() -> bool:
    ffmpeg = get_settings().ffmpeg_path
    try:
        result = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def mux_audio_video(video_path: Path, audio_path: Path, out_path: Path) -> Path:
    """Combine a silent video clip with a narration track. Narration is
    authoritative for length: if the video track is even slightly shorter
    (the concat demuxer rounds each frame's hold time to the output
    framerate, so drift accumulates over many short segments — confirmed:
    an 8.68s video against 9.04s audio, losing the last ~0.36s of speech),
    the last frame is frozen to cover the gap instead of the narration
    getting truncated by -shortest."""
    video_dur = get_duration_seconds(video_path)
    audio_dur = get_duration_seconds(audio_path)
    args = ["-i", str(video_path), "-i", str(audio_path)]
    pad = audio_dur - video_dur
    if pad > 0.02:
        args += ["-vf", f"tpad=stop_mode=clone:stop_duration={pad:.3f}", "-c:v", "libx264"]
    else:
        args += ["-c:v", "copy"]
    args += ["-c:a", "aac", "-shortest", str(out_path)]
    _run(args)
    return out_path


def assemble_frame_sequence(frame_paths: list[Path], durations: list[float], audio_path: Path, out_path: Path) -> Path:
    """Turns a list of (image, how-long-to-hold-it) pairs into a video track
    synced to audio_path — this is how the talking-character frames from
    character_rig become an actual clip, without ever hitting 30fps-worth of
    AI image calls."""
    if len(frame_paths) != len(durations):
        raise ProviderError("frame_paths and durations must be the same length")
    concat_list = out_path.with_suffix(".concat.txt")
    lines = []
    for path, duration in zip(frame_paths, durations):
        lines.append(f"file '{path.resolve().as_posix()}'")
        lines.append(f"duration {duration}")
    lines.append(f"file '{frame_paths[-1].resolve().as_posix()}'")  # ffmpeg concat quirk: last entry needs no duration but must repeat
    concat_list.write_text("\n".join(lines), encoding="utf-8")

    silent_video = out_path.with_suffix(".silent.mp4")
    _run([
        "-f", "concat", "-safe", "0", "-i", str(concat_list),
        "-fps_mode", "vfr", "-pix_fmt", "yuv420p",  # -vsync was renamed to -fps_mode in ffmpeg 6+
        str(silent_video),
    ])
    mux_audio_video(silent_video, audio_path, out_path)
    concat_list.unlink(missing_ok=True)
    silent_video.unlink(missing_ok=True)
    return out_path


MOTIONS = {"in", "out", "pan_lr", "pan_up", "static"}


def ken_burns_clip(
    image_path: Path,
    duration: float,
    out_path: Path,
    motion: str = "in",
    width: int = 1664,
    height: int = 928,
    fps: int = 25,
) -> Path:
    """Camera motion over a still image — the cheap way to put motion on an
    AI-generated scene without a second AI call per scene. width/height set
    the output canvas (e.g. 1664x928 for 16:9 long-form; use 1328x1328 to
    match a square Qwen-Image source 1:1).

    Explicit centering + upscale-before-crop margin: not fixing a proven
    zoompan bug (a black-border case that looked like one turned out to be
    baked into the source image itself — Qwen-Image sometimes renders the
    torn-paper vignette on a black backdrop instead of cream, see
    image_qwen's STYLE prompt) — just defensive practice against zoompan's
    unspecified-x/y default not recentering the crop.

    'static' still applies a barely-there zoom (1.0->1.03): a fully frozen
    frame reads as dead on screen, and we have no rig to animate the
    character in these full-scene-generated shots — this is the honest
    stand-in for the "subtle motion" a couple of scenes asked for."""
    if motion not in MOTIONS:
        raise ProviderError(f"motion must be one of {MOTIONS}, got {motion!r}")
    frames = max(int(duration * fps), 1)
    upscale_w, upscale_h = int(width * 1.3), int(height * 1.3)
    step = 0.6 / frames

    x_center, y_center = "(iw-iw/zoom)/2", "(ih-ih/zoom)/2"
    if motion == "in":
        zoom_expr = f"min(zoom+{step:.6f},1.15)"
        x_expr, y_expr = x_center, y_center
    elif motion == "out":
        zoom_expr = f"if(eq(on,1),1.15,max(zoom-{step:.6f},1.0))"
        x_expr, y_expr = x_center, y_center
    elif motion == "static":
        zoom_expr = f"min(zoom+{0.03/frames:.6f},1.03)"
        x_expr, y_expr = x_center, y_center
    elif motion == "pan_lr":
        zoom_expr = "1.12"
        x_expr = f"(iw-iw/zoom)*on/{frames}"
        y_expr = y_center
    else:  # pan_up: crop window drifts from the lower part of the frame to centered
        zoom_expr = "1.12"
        x_expr = x_center
        y_expr = f"(ih-ih/zoom)*(1-on/{frames})"

    _run([
        "-loop", "1", "-i", str(image_path),
        "-vf", (
            f"scale={upscale_w}:{upscale_h},"
            f"zoompan=z='{zoom_expr}':d={frames}:s={width}x{height}:fps={fps}"
            f":x='{x_expr}':y='{y_expr}'"
        ),
        "-t", f"{duration:.3f}",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ])
    return out_path


def concat_videos(video_paths: list[Path], out_path: Path) -> Path:
    """Concatenate already-encoded clips (same codec/size) with no re-encode."""
    concat_list = out_path.with_suffix(".concat.txt")
    concat_list.write_text(
        "\n".join(f"file '{p.resolve().as_posix()}'" for p in video_paths), encoding="utf-8"
    )
    _run(["-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(out_path)])
    concat_list.unlink(missing_ok=True)
    return out_path


def get_duration_seconds(media_path: Path) -> float:
    ffmpeg_path = Path(get_settings().ffmpeg_path)
    ffprobe_name = "ffprobe.exe" if ffmpeg_path.suffix == ".exe" else "ffprobe"
    ffprobe = str(ffmpeg_path.with_name(ffprobe_name))
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(media_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ProviderError(f"ffprobe failed: {result.stderr[-2000:]}")
    return float(json.loads(result.stdout)["format"]["duration"])
