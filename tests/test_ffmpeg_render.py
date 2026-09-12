"""Regression test for the video/audio duration-drift bug: a concat video
even slightly shorter than its narration must never truncate the audio."""
import subprocess
import wave
from pathlib import Path

import pytest

from src.render import ffmpeg_render


def _make_silence_wav(path: Path, seconds: float) -> None:
    with wave.open(str(path), "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(24000)
        f.writeframes(b"\x00\x00" * int(24000 * seconds))


def _make_solid_frame(path: Path) -> None:
    subprocess.run(
        [ffmpeg_render.get_settings().ffmpeg_path, "-y", "-f", "lavfi", "-i", "color=c=white:s=64x64", "-frames:v", "1", str(path)],
        check=True,
        capture_output=True,
    )


@pytest.mark.skipif(not ffmpeg_render.health_check(), reason="ffmpeg not available")
def test_mux_pads_video_when_shorter_than_audio(tmp_path):
    video = tmp_path / "short.mp4"
    audio = tmp_path / "narration.wav"
    frame = tmp_path / "frame.png"
    _make_solid_frame(frame)
    _make_silence_wav(audio, seconds=3.0)
    subprocess.run(
        [ffmpeg_render.get_settings().ffmpeg_path, "-y", "-loop", "1", "-i", str(frame), "-t", "1.5", str(video)],
        check=True,
        capture_output=True,
    )

    out = ffmpeg_render.mux_audio_video(video, audio, tmp_path / "out.mp4")

    assert ffmpeg_render.get_duration_seconds(out) == pytest.approx(3.0, abs=0.05)


@pytest.mark.skipif(not ffmpeg_render.health_check(), reason="ffmpeg not available")
def test_pad_audio_to_duration_reaches_target(tmp_path):
    audio = tmp_path / "short.wav"
    _make_silence_wav(audio, seconds=4.0)

    out = ffmpeg_render.pad_audio_to_duration(audio, min_duration=12.0, out_path=tmp_path / "padded.wav")

    assert ffmpeg_render.get_duration_seconds(out) == pytest.approx(12.0, abs=0.05)


@pytest.mark.skipif(not ffmpeg_render.health_check(), reason="ffmpeg not available")
def test_pad_audio_to_duration_is_a_noop_when_already_long_enough(tmp_path):
    audio = tmp_path / "long.wav"
    _make_silence_wav(audio, seconds=8.0)

    out = ffmpeg_render.pad_audio_to_duration(audio, min_duration=5.0, out_path=tmp_path / "padded.wav")

    assert out == audio  # untouched — no ffmpeg call, no output file written
