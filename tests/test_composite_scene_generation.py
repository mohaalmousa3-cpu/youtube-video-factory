"""Tests for src/core/composite_scene_generation.py — the pure local
orchestration combining get_mouth_cues() + the time-based sway angle
formula + apply_mouth_and_limb_animation() + assemble_frame_sequence().
Every test mocks rhubarb_health_check/ffmpeg_health_check/get_mouth_cues/
apply_mouth_and_limb_animation/assemble_frame_sequence at their point of
import in this module (same monkeypatch.setattr("src.core.<module>.<name>",
fake) convention every prior core-module test in this repo uses) — no real
Rhubarb binary, no real ffmpeg, no network call anywhere in this file. Cue
structural/ordering validation itself is not re-tested exhaustively here
(that's covered in full in tests/test_mouth_animation_generation.py, since
this module reuses _validate_cues/_derive_durations directly rather than
duplicating them) — just that the reuse actually happens and translates
into this module's own exception type."""
from __future__ import annotations

import math
from pathlib import Path

import pytest
from PIL import Image

from src.core.composite_scene_generation import (
    CompositeSceneGenerationError,
    _angle_at_time,
    build_composite_scene_clip,
)
from src.models.scene import SceneArtifacts, ScenePlanItem
from src.providers.base import ProviderError

MODULE = "src.core.composite_scene_generation"


def _scene(duration=5.0, scene_id: str = "scene-01") -> ScenePlanItem:
    return ScenePlanItem(
        scene_id=scene_id,
        sequence=1,
        narration_text="Narration text.",
        scene_type="narration",
        narrative_beat="setup",
        visual_brief="Visual brief.",
        motion_mode="static",
        approval_state="approved",
        artifacts=SceneArtifacts(measured_audio_duration_seconds=duration),
    )


def _png(path: Path, size=(200, 100)) -> None:
    Image.new("RGB", size, color=(200, 150, 100)).save(path, format="PNG")


def _cue(start, end, value="X"):
    return {"start": start, "end": end, "value": value}


def _mock_ok(monkeypatch, calls: list, cues=None):
    if cues is None:
        cues = [_cue(0.0, 2.0), _cue(2.0, 5.0)]
    monkeypatch.setattr(f"{MODULE}.rhubarb_health_check", lambda: True)
    monkeypatch.setattr(f"{MODULE}.ffmpeg_health_check", lambda: True)

    def _fake_get_mouth_cues(audio_path):
        calls.append(("get_mouth_cues", Path(audio_path)))
        return cues

    def _fake_apply(scene_image_path, mouth_box, mouth_cues, limb_box, angles, out_dir):
        calls.append(
            ("apply_mouth_and_limb_animation", Path(scene_image_path), mouth_box, mouth_cues, limb_box, list(angles), Path(out_dir))
        )
        frames = []
        for i in range(len(mouth_cues)):
            p = Path(out_dir) / f"frame_{i:04d}.png"
            p.write_bytes(b"fake-frame-png")
            frames.append(p)
        return frames

    def _fake_assemble(frame_paths, durations, audio_path, out_path):
        calls.append(("assemble_frame_sequence", list(frame_paths), list(durations), Path(audio_path), Path(out_path)))
        Path(out_path).write_bytes(b"fake-mp4-bytes")
        return out_path

    monkeypatch.setattr(f"{MODULE}.get_mouth_cues", _fake_get_mouth_cues)
    monkeypatch.setattr(f"{MODULE}.apply_mouth_and_limb_animation", _fake_apply)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _fake_assemble)
    return cues


def _forbid_generation(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("must not be called")

    monkeypatch.setattr(f"{MODULE}.rhubarb_health_check", _boom)
    monkeypatch.setattr(f"{MODULE}.ffmpeg_health_check", _boom)
    monkeypatch.setattr(f"{MODULE}.get_mouth_cues", _boom)
    monkeypatch.setattr(f"{MODULE}.apply_mouth_and_limb_animation", _boom)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _boom)


def _spy_mkdtemp(monkeypatch):
    import tempfile

    created: list[str] = []
    real_mkdtemp = tempfile.mkdtemp

    def _spy(*args, **kwargs):
        d = real_mkdtemp(*args, **kwargs)
        created.append(d)
        return d

    monkeypatch.setattr(f"{MODULE}.tempfile.mkdtemp", _spy)
    return created


# ---------------------------------------------------------------------
# time-based angle formula
# ---------------------------------------------------------------------


def test_angle_at_time_matches_continuous_sine_formula():
    angle = _angle_at_time(amplitude_degrees=4.0, cycles=1.0, total_duration=2.0, t=0.5)
    assert angle == pytest.approx(4.0 * math.sin(2 * math.pi * 1.0 * 0.5 / 2.0))


def test_angle_at_time_zero_at_t_zero():
    assert _angle_at_time(amplitude_degrees=4.0, cycles=1.0, total_duration=2.0, t=0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------
# happy path: order, box/parameter reuse, angles derived from real cues
# ---------------------------------------------------------------------


def test_calls_in_order_get_mouth_cues_then_apply_then_assemble(tmp_path, monkeypatch):
    calls: list = []
    _mock_ok(monkeypatch, calls)
    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    build_composite_scene_clip(
        _scene(duration=5.0), source, audio, "100,50,40,20", "10,10,40,90,25,15", "4.0", "1.0", out_path
    )

    assert [c[0] for c in calls] == ["get_mouth_cues", "apply_mouth_and_limb_animation", "assemble_frame_sequence"]


def test_angles_passed_to_apply_match_cue_start_timestamps(tmp_path, monkeypatch):
    calls: list = []
    cues = [_cue(0.0, 1.0), _cue(1.0, 2.5)]
    _mock_ok(monkeypatch, calls, cues=cues)
    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    build_composite_scene_clip(
        _scene(duration=5.0), source, audio, "100,50,40,20", "10,10,40,90,25,15", "4.0", "1.0", out_path
    )

    apply_call = next(c for c in calls if c[0] == "apply_mouth_and_limb_animation")
    angles = apply_call[5]
    expected = [4.0 * math.sin(2 * math.pi * 1.0 * cue["start"] / 5.0) for cue in cues]
    assert angles == pytest.approx(expected)


def test_reuses_mouth_box_grammar_from_preflight(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.rhubarb_health_check", lambda: True)

    def _boom(*a, **k):
        raise AssertionError("must not be called")

    monkeypatch.setattr(f"{MODULE}.ffmpeg_health_check", _boom)
    monkeypatch.setattr(f"{MODULE}.get_mouth_cues", _boom)
    monkeypatch.setattr(f"{MODULE}.apply_mouth_and_limb_animation", _boom)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _boom)

    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(CompositeSceneGenerationError, match="base-10 integers"):
        build_composite_scene_clip(
            _scene(), source, audio, "abc,50,40,20", "10,10,40,90,25,15", "4.0", "1.0", out_path
        )


def test_reuses_limb_box_grammar_from_limb_sway_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.rhubarb_health_check", lambda: True)

    def _boom(*a, **k):
        raise AssertionError("must not be called")

    monkeypatch.setattr(f"{MODULE}.ffmpeg_health_check", _boom)
    monkeypatch.setattr(f"{MODULE}.get_mouth_cues", _boom)
    monkeypatch.setattr(f"{MODULE}.apply_mouth_and_limb_animation", _boom)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _boom)

    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(CompositeSceneGenerationError, match="base-10 integers"):
        build_composite_scene_clip(
            _scene(), source, audio, "100,50,40,20", "xx,10,40,90,25,15", "4.0", "1.0", out_path
        )


def test_reuses_limb_geometry_bounds_check(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.rhubarb_health_check", lambda: True)

    def _boom(*a, **k):
        raise AssertionError("must not be called")

    monkeypatch.setattr(f"{MODULE}.ffmpeg_health_check", _boom)
    monkeypatch.setattr(f"{MODULE}.get_mouth_cues", _boom)
    monkeypatch.setattr(f"{MODULE}.apply_mouth_and_limb_animation", _boom)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _boom)

    source = tmp_path / "source.png"
    _png(source, size=(200, 100))
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(CompositeSceneGenerationError, match="extends outside the image bounds"):
        build_composite_scene_clip(
            _scene(), source, audio, "100,50,40,20", "10,10,999,90,25,15", "4.0", "1.0", out_path
        )


def test_reuses_cue_validation_from_mouth_animation_generation(tmp_path, monkeypatch):
    calls: list = []
    monkeypatch.setattr(f"{MODULE}.rhubarb_health_check", lambda: True)
    monkeypatch.setattr(f"{MODULE}.ffmpeg_health_check", lambda: True)
    monkeypatch.setattr(f"{MODULE}.get_mouth_cues", lambda audio_path: [_cue(0.5, 1.0)])

    def _boom(*a, **k):
        raise AssertionError("must not be called")

    monkeypatch.setattr(f"{MODULE}.apply_mouth_and_limb_animation", _boom)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _boom)

    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(CompositeSceneGenerationError, match="must start at 0.0"):
        build_composite_scene_clip(
            _scene(duration=5.0), source, audio, "100,50,40,20", "10,10,40,90,25,15", "4.0", "1.0", out_path
        )
    assert not out_path.exists()


def test_empty_cues_rejects(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.rhubarb_health_check", lambda: True)
    monkeypatch.setattr(f"{MODULE}.ffmpeg_health_check", lambda: True)
    monkeypatch.setattr(f"{MODULE}.get_mouth_cues", lambda audio_path: [])

    def _boom(*a, **k):
        raise AssertionError("must not be called")

    monkeypatch.setattr(f"{MODULE}.apply_mouth_and_limb_animation", _boom)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _boom)

    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(CompositeSceneGenerationError, match="no mouth cues"):
        build_composite_scene_clip(
            _scene(), source, audio, "100,50,40,20", "10,10,40,90,25,15", "4.0", "1.0", out_path
        )
    assert not out_path.exists()


# ---------------------------------------------------------------------
# output-exists / health-check ordering (Rhubarb early, ffmpeg last)
# ---------------------------------------------------------------------


def test_existing_output_rejects_before_any_call(tmp_path, monkeypatch):
    _forbid_generation(monkeypatch)
    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"
    out_path.write_bytes(b"pre-existing content nobody asked to touch")

    with pytest.raises(CompositeSceneGenerationError, match="already exists"):
        build_composite_scene_clip(
            _scene(), source, audio, "100,50,40,20", "10,10,40,90,25,15", "4.0", "1.0", out_path
        )
    assert out_path.read_bytes() == b"pre-existing content nobody asked to touch"


def test_unavailable_rhubarb_rejects_before_any_call(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("must not be called")

    monkeypatch.setattr(f"{MODULE}.rhubarb_health_check", lambda: False)
    monkeypatch.setattr(f"{MODULE}.ffmpeg_health_check", _boom)
    monkeypatch.setattr(f"{MODULE}.get_mouth_cues", _boom)
    monkeypatch.setattr(f"{MODULE}.apply_mouth_and_limb_animation", _boom)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _boom)

    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(CompositeSceneGenerationError, match="Rhubarb .* unavailable"):
        build_composite_scene_clip(
            _scene(), source, audio, "100,50,40,20", "10,10,40,90,25,15", "4.0", "1.0", out_path
        )
    assert not out_path.exists()


def test_unavailable_ffmpeg_rejects_after_box_parsing_before_any_generation_call(tmp_path, monkeypatch):
    """ffmpeg is checked LAST, right before the temp dir — mouth/limb box
    parsing and geometry validation must already have run and passed."""
    monkeypatch.setattr(f"{MODULE}.rhubarb_health_check", lambda: True)
    monkeypatch.setattr(f"{MODULE}.ffmpeg_health_check", lambda: False)

    def _boom(*a, **k):
        raise AssertionError("must not be called")

    monkeypatch.setattr(f"{MODULE}.get_mouth_cues", _boom)
    monkeypatch.setattr(f"{MODULE}.apply_mouth_and_limb_animation", _boom)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _boom)

    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(CompositeSceneGenerationError, match="ffmpeg is unavailable"):
        build_composite_scene_clip(
            _scene(), source, audio, "100,50,40,20", "10,10,40,90,25,15", "4.0", "1.0", out_path
        )
    assert not out_path.exists()


# ---------------------------------------------------------------------
# temp-dir / output / sidecar cleanup
# ---------------------------------------------------------------------


def test_temp_dir_removed_after_success(tmp_path, monkeypatch):
    calls: list = []
    created = _spy_mkdtemp(monkeypatch)
    _mock_ok(monkeypatch, calls)
    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    build_composite_scene_clip(
        _scene(), source, audio, "100,50,40,20", "10,10,40,90,25,15", "4.0", "1.0", out_path
    )

    assert len(created) == 1
    assert not Path(created[0]).exists()
    assert out_path.exists()


def test_get_mouth_cues_receives_a_temp_copy_not_source_audio_path(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.rhubarb_health_check", lambda: True)
    monkeypatch.setattr(f"{MODULE}.ffmpeg_health_check", lambda: True)
    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"original-audio-bytes")
    out_path = tmp_path / "out.mp4"

    received_paths = []

    def _fake_get_mouth_cues(audio_path):
        received_paths.append(Path(audio_path))
        return [_cue(0.0, 5.0)]

    def _fake_apply(scene_image_path, mouth_box, mouth_cues, limb_box, angles, out_dir):
        p = Path(out_dir) / "frame_0000.png"
        p.write_bytes(b"x")
        return [p]

    def _fake_assemble(frame_paths, durations, audio_path, out_path):
        Path(out_path).write_bytes(b"fake-mp4")
        return out_path

    monkeypatch.setattr(f"{MODULE}.get_mouth_cues", _fake_get_mouth_cues)
    monkeypatch.setattr(f"{MODULE}.apply_mouth_and_limb_animation", _fake_apply)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _fake_assemble)

    build_composite_scene_clip(
        _scene(duration=5.0), source, audio, "100,50,40,20", "10,10,40,90,25,15", "4.0", "1.0", out_path
    )

    assert len(received_paths) == 1
    received = received_paths[0]
    assert received != audio
    assert received.name == "source_audio.wav"
    assert received.parent != audio.parent


def test_assemble_frame_sequence_receives_original_audio_path_not_temp_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.rhubarb_health_check", lambda: True)
    monkeypatch.setattr(f"{MODULE}.ffmpeg_health_check", lambda: True)
    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"original-audio-bytes")
    out_path = tmp_path / "out.mp4"

    monkeypatch.setattr(f"{MODULE}.get_mouth_cues", lambda audio_path: [_cue(0.0, 5.0)])

    def _fake_apply(scene_image_path, mouth_box, mouth_cues, limb_box, angles, out_dir):
        p = Path(out_dir) / "frame_0000.png"
        p.write_bytes(b"x")
        return [p]

    received = {}

    def _fake_assemble(frame_paths, durations, audio_path, out_path):
        received["audio_path"] = Path(audio_path)
        Path(out_path).write_bytes(b"fake-mp4")
        return out_path

    monkeypatch.setattr(f"{MODULE}.apply_mouth_and_limb_animation", _fake_apply)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _fake_assemble)

    build_composite_scene_clip(
        _scene(duration=5.0), source, audio, "100,50,40,20", "10,10,40,90,25,15", "4.0", "1.0", out_path
    )

    assert received["audio_path"] == audio


def test_temp_dir_and_output_removed_after_assemble_provider_error(tmp_path, monkeypatch):
    created = _spy_mkdtemp(monkeypatch)
    monkeypatch.setattr(f"{MODULE}.rhubarb_health_check", lambda: True)
    monkeypatch.setattr(f"{MODULE}.ffmpeg_health_check", lambda: True)
    monkeypatch.setattr(f"{MODULE}.get_mouth_cues", lambda audio_path: [_cue(0.0, 5.0)])

    def _fake_apply(scene_image_path, mouth_box, mouth_cues, limb_box, angles, out_dir):
        p = Path(out_dir) / "frame_0000.png"
        p.write_bytes(b"x")
        return [p]

    def _fail_assemble(frame_paths, durations, audio_path, out_path):
        Path(out_path).write_bytes(b"partial-mp4-from-a-crashed-ffmpeg")
        raise ProviderError("ffmpeg failed: <raw stderr with internal paths>")

    monkeypatch.setattr(f"{MODULE}.apply_mouth_and_limb_animation", _fake_apply)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _fail_assemble)

    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(CompositeSceneGenerationError) as exc_info:
        build_composite_scene_clip(
            _scene(duration=5.0), source, audio, "100,50,40,20", "10,10,40,90,25,15", "4.0", "1.0", out_path
        )

    assert str(exc_info.value) == "composite scene render failed"
    assert "stderr" not in str(exc_info.value)
    assert len(created) == 1
    assert not Path(created[0]).exists()
    assert not out_path.exists()


def test_temp_dir_removed_after_apply_oserror(tmp_path, monkeypatch):
    created = _spy_mkdtemp(monkeypatch)
    monkeypatch.setattr(f"{MODULE}.rhubarb_health_check", lambda: True)
    monkeypatch.setattr(f"{MODULE}.ffmpeg_health_check", lambda: True)
    monkeypatch.setattr(f"{MODULE}.get_mouth_cues", lambda audio_path: [_cue(0.0, 5.0)])

    def _fail_apply(scene_image_path, mouth_box, mouth_cues, limb_box, angles, out_dir):
        raise OSError("disk full: /some/internal/path")

    def _boom(*a, **k):
        raise AssertionError("must not be called after apply fails")

    monkeypatch.setattr(f"{MODULE}.apply_mouth_and_limb_animation", _fail_apply)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _boom)

    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(CompositeSceneGenerationError) as exc_info:
        build_composite_scene_clip(
            _scene(duration=5.0), source, audio, "100,50,40,20", "10,10,40,90,25,15", "4.0", "1.0", out_path
        )

    assert str(exc_info.value) == "composite frame generation failed"
    assert "/some/internal/path" not in str(exc_info.value)
    assert len(created) == 1
    assert not Path(created[0]).exists()
    assert not out_path.exists()


# ---------------------------------------------------------------------
# optional integration test — real health_check() gates on both sides;
# skips cleanly in this checkout (Rhubarb/ffmpeg availability varies)
# ---------------------------------------------------------------------


def test_real_pipeline_integration(tmp_path):
    import subprocess

    from src.providers import lipsync_rhubarb
    from src.render import ffmpeg_render
    from src.utils.config import get_settings

    if not (lipsync_rhubarb.health_check() and ffmpeg_render.health_check()):
        pytest.skip("real Rhubarb and/or ffmpeg not available in this environment")

    source = tmp_path / "source.png"
    _png(source, size=(200, 200))
    audio = tmp_path / "audio.wav"
    ffmpeg = get_settings().ffmpeg_path
    result = subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", "1.0", str(audio)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    out_path = tmp_path / "out.mp4"
    scene = _scene(duration=1.0)

    build_composite_scene_clip(scene, source, audio, "100,50,40,20", "10,10,40,90,25,15", "4.0", "1.0", out_path)

    assert out_path.exists()
    duration = ffmpeg_render.get_duration_seconds(out_path)
    assert duration == pytest.approx(1.0, abs=0.3)
