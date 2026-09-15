"""Tests for src/core/mouth_animation_generation.py — the pure local
orchestration of get_mouth_cues() + apply_mouth_animation() +
assemble_frame_sequence(). Every test mocks health_check/get_mouth_cues/
apply_mouth_animation/assemble_frame_sequence at their point of import in
this module (same monkeypatch.setattr("src.core.<module>.<name>", fake)
convention every prior core-module test in this repo already uses) — no
real Rhubarb binary, no real ffmpeg, no network call anywhere in this
file, except the one optional, doubly-gated integration test at the
bottom."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.core.mouth_animation_generation import (
    MouthAnimationGenerationError,
    _derive_durations,
    build_mouth_animated_clip,
)
from src.models.scene import SceneArtifacts, ScenePlanItem
from src.providers.base import ProviderError

MODULE = "src.core.mouth_animation_generation"


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
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: True)

    def _fake_get_mouth_cues(audio_path):
        calls.append(("get_mouth_cues", Path(audio_path)))
        return cues

    def _fake_apply_mouth_animation(scene_image_path, mouth_box, mouth_cues, out_dir):
        calls.append(("apply_mouth_animation", Path(scene_image_path), mouth_box, mouth_cues, Path(out_dir)))
        frames = []
        for i in range(len(mouth_cues)):
            p = Path(out_dir) / f"frame_{i:04d}.png"
            p.write_bytes(b"fake-frame-png")
            frames.append(p)
        return frames

    def _fake_assemble_frame_sequence(frame_paths, durations, audio_path, out_path):
        calls.append(("assemble_frame_sequence", list(frame_paths), list(durations), Path(audio_path), Path(out_path)))
        Path(out_path).write_bytes(b"fake-mp4-bytes")
        return out_path

    monkeypatch.setattr(f"{MODULE}.get_mouth_cues", _fake_get_mouth_cues)
    monkeypatch.setattr(f"{MODULE}.apply_mouth_animation", _fake_apply_mouth_animation)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _fake_assemble_frame_sequence)
    return cues


def _forbid_generation(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("must not be called")

    monkeypatch.setattr(f"{MODULE}.health_check", _boom)
    monkeypatch.setattr(f"{MODULE}.get_mouth_cues", _boom)
    monkeypatch.setattr(f"{MODULE}.apply_mouth_animation", _boom)
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
# happy path: order, mouth-box grammar/geometry reuse, source untouched,
# timing derivation
# ---------------------------------------------------------------------


def test_calls_in_order_get_mouth_cues_then_apply_then_assemble(tmp_path, monkeypatch):
    calls: list = []
    _mock_ok(monkeypatch, calls)
    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    build_mouth_animated_clip(_scene(duration=5.0), source, audio, "100,50,40,20", out_path)

    assert [c[0] for c in calls] == ["get_mouth_cues", "apply_mouth_animation", "assemble_frame_sequence"]


def test_reuses_strict_mouth_box_grammar_from_preflight(tmp_path, monkeypatch):
    """Validation order is output-exists -> health_check -> duration ->
    mouth-box/image (per the approved spec), so health_check legitimately
    runs before mouth-box grammar is checked here — only cue/frame/ffmpeg
    work must never be reached."""
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: True)

    def _boom(*a, **k):
        raise AssertionError("must not be called")

    monkeypatch.setattr(f"{MODULE}.get_mouth_cues", _boom)
    monkeypatch.setattr(f"{MODULE}.apply_mouth_animation", _boom)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _boom)

    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(MouthAnimationGenerationError, match="base-10 integers"):
        build_mouth_animated_clip(_scene(), source, audio, "abc,50,40,20", out_path)


def test_reuses_geometry_bounds_check_from_preflight(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: True)
    source = tmp_path / "source.png"
    _png(source, size=(200, 100))
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    def _boom(*a, **k):
        raise AssertionError("must not be called when the box is out of bounds")

    monkeypatch.setattr(f"{MODULE}.get_mouth_cues", _boom)
    monkeypatch.setattr(f"{MODULE}.apply_mouth_animation", _boom)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _boom)

    with pytest.raises(MouthAnimationGenerationError, match="extends outside the image bounds"):
        build_mouth_animated_clip(_scene(), source, audio, "195,50,40,20", out_path)


def test_source_image_bytes_unchanged(tmp_path, monkeypatch):
    calls: list = []
    _mock_ok(monkeypatch, calls)
    source = tmp_path / "source.png"
    _png(source)
    original_bytes = source.read_bytes()
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    build_mouth_animated_clip(_scene(), source, audio, "100,50,40,20", out_path)

    assert source.read_bytes() == original_bytes


def test_validated_measured_duration_used_for_last_frame(tmp_path, monkeypatch):
    calls: list = []
    _mock_ok(monkeypatch, calls, cues=[_cue(0.0, 1.0), _cue(1.0, 1.0)])
    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    build_mouth_animated_clip(_scene(duration=7.5), source, audio, "100,50,40,20", out_path)

    assemble_call = next(c for c in calls if c[0] == "assemble_frame_sequence")
    durations = assemble_call[2]
    assert durations[-1] == pytest.approx(7.5 - 1.0)


def test_exact_derived_duration_list_for_multiple_cues(tmp_path, monkeypatch):
    calls: list = []
    cues = [_cue(0.0, 1.0), _cue(1.5, 3.0), _cue(3.0, 3.0)]
    _mock_ok(monkeypatch, calls, cues=cues)
    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    build_mouth_animated_clip(_scene(duration=10.0), source, audio, "100,50,40,20", out_path)

    assemble_call = next(c for c in calls if c[0] == "assemble_frame_sequence")
    durations = assemble_call[2]
    assert durations == [1.5 - 0.0, 3.0 - 1.5, 10.0 - 3.0]


def test_cue_end_values_do_not_affect_durations(tmp_path, monkeypatch):
    """cue["end"] is only used for the end>=start structural check — never
    for timing math. The derived duration list must come out identical to
    what pure start-to-next-start arithmetic gives, regardless of what
    (individually valid) "end" values the cues carry."""
    calls: list = []
    cues_low_end = [_cue(0.0, 0.05), _cue(2.0, 2.05)]
    _mock_ok(monkeypatch, calls, cues=cues_low_end)
    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    build_mouth_animated_clip(_scene(duration=5.0), source, audio, "100,50,40,20", out_path)
    assemble_call = next(c for c in calls if c[0] == "assemble_frame_sequence")
    durations_low_end = assemble_call[2]

    assert durations_low_end == [2.0 - 0.0, 5.0 - 2.0]


def test_frame_paths_and_audio_path_passed_unchanged_to_assemble(tmp_path, monkeypatch):
    calls: list = []
    _mock_ok(monkeypatch, calls, cues=[_cue(0.0, 2.0), _cue(2.0, 5.0)])
    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    build_mouth_animated_clip(_scene(duration=5.0), source, audio, "100,50,40,20", out_path)

    apply_call = next(c for c in calls if c[0] == "apply_mouth_animation")
    assemble_call = next(c for c in calls if c[0] == "assemble_frame_sequence")
    apply_out_dir = apply_call[4]
    frame_paths_passed_to_assemble = assemble_call[1]

    # The frame paths handed to assemble_frame_sequence are exactly what
    # apply_mouth_animation's fake returned: two frames, both under the
    # same private temp dir apply_mouth_animation was given.
    assert len(frame_paths_passed_to_assemble) == 2
    assert all(p.parent == apply_out_dir for p in frame_paths_passed_to_assemble)
    assert assemble_call[3] == audio
    assert assemble_call[4] == out_path


# ---------------------------------------------------------------------
# cue validation — every required rejection category
# ---------------------------------------------------------------------


def test_empty_cues_rejects(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: True)
    monkeypatch.setattr(f"{MODULE}.get_mouth_cues", lambda audio_path: [])

    def _boom(*a, **k):
        raise AssertionError("must not be called")

    monkeypatch.setattr(f"{MODULE}.apply_mouth_animation", _boom)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _boom)

    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(MouthAnimationGenerationError, match="no mouth cues"):
        build_mouth_animated_clip(_scene(), source, audio, "100,50,40,20", out_path)
    assert not out_path.exists()


@pytest.mark.parametrize(
    "bad_cue",
    [
        {"end": 1.0, "value": "X"},
        {"start": 0.0, "value": "X"},
        {"start": 0.0, "end": 1.0},
    ],
)
def test_missing_cue_field_rejects(tmp_path, monkeypatch, bad_cue):
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: True)
    monkeypatch.setattr(f"{MODULE}.get_mouth_cues", lambda audio_path: [bad_cue])

    def _boom(*a, **k):
        raise AssertionError("must not be called")

    monkeypatch.setattr(f"{MODULE}.apply_mouth_animation", _boom)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _boom)

    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(MouthAnimationGenerationError, match="missing a required field"):
        build_mouth_animated_clip(_scene(), source, audio, "100,50,40,20", out_path)


@pytest.mark.parametrize(
    "cues",
    [
        [_cue(True, 1.0)],
        [_cue(0.0, True)],
        [_cue("0.0", 1.0)],
    ],
)
def test_bool_or_non_numeric_timestamp_rejects(tmp_path, monkeypatch, cues):
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: True)
    monkeypatch.setattr(f"{MODULE}.get_mouth_cues", lambda audio_path: cues)

    def _boom(*a, **k):
        raise AssertionError("must not be called")

    monkeypatch.setattr(f"{MODULE}.apply_mouth_animation", _boom)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _boom)

    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(MouthAnimationGenerationError, match="must be a valid number"):
        build_mouth_animated_clip(_scene(), source, audio, "100,50,40,20", out_path)


@pytest.mark.parametrize("cues", [[_cue(float("nan"), 1.0)], [_cue(0.0, float("inf"))]])
def test_non_finite_timestamp_rejects(tmp_path, monkeypatch, cues):
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: True)
    monkeypatch.setattr(f"{MODULE}.get_mouth_cues", lambda audio_path: cues)

    def _boom(*a, **k):
        raise AssertionError("must not be called")

    monkeypatch.setattr(f"{MODULE}.apply_mouth_animation", _boom)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _boom)

    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(MouthAnimationGenerationError, match="must be finite"):
        build_mouth_animated_clip(_scene(), source, audio, "100,50,40,20", out_path)


def test_invalid_cue_end_before_start_rejects(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: True)
    monkeypatch.setattr(f"{MODULE}.get_mouth_cues", lambda audio_path: [_cue(2.0, 1.0)])

    def _boom(*a, **k):
        raise AssertionError("must not be called")

    monkeypatch.setattr(f"{MODULE}.apply_mouth_animation", _boom)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _boom)

    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(MouthAnimationGenerationError, match="must not be before its start"):
        build_mouth_animated_clip(_scene(), source, audio, "100,50,40,20", out_path)


@pytest.mark.parametrize("cues", [[_cue(-1.0, 0.0)], [_cue(5.0, 5.5)]])
def test_start_outside_range_rejects(tmp_path, monkeypatch, cues):
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: True)
    monkeypatch.setattr(f"{MODULE}.get_mouth_cues", lambda audio_path: cues)

    def _boom(*a, **k):
        raise AssertionError("must not be called")

    monkeypatch.setattr(f"{MODULE}.apply_mouth_animation", _boom)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _boom)

    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(MouthAnimationGenerationError, match="outside the scene's measured audio duration"):
        build_mouth_animated_clip(_scene(duration=5.0), source, audio, "100,50,40,20", out_path)


def test_nonzero_first_start_beyond_tolerance_rejects(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: True)
    monkeypatch.setattr(f"{MODULE}.get_mouth_cues", lambda audio_path: [_cue(0.01, 1.0)])

    def _boom(*a, **k):
        raise AssertionError("must not be called")

    monkeypatch.setattr(f"{MODULE}.apply_mouth_animation", _boom)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _boom)

    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(MouthAnimationGenerationError, match="must start at 0.0"):
        build_mouth_animated_clip(_scene(), source, audio, "100,50,40,20", out_path)


def test_first_start_within_tolerance_is_accepted(tmp_path, monkeypatch):
    calls: list = []
    _mock_ok(monkeypatch, calls, cues=[_cue(5e-7, 1.0), _cue(1.0, 1.0)])
    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    build_mouth_animated_clip(_scene(duration=5.0), source, audio, "100,50,40,20", out_path)
    assert any(c[0] == "assemble_frame_sequence" for c in calls)


def test_duplicate_start_rejects(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: True)
    monkeypatch.setattr(f"{MODULE}.get_mouth_cues", lambda audio_path: [_cue(0.0, 1.0), _cue(0.0, 2.0)])

    def _boom(*a, **k):
        raise AssertionError("must not be called")

    monkeypatch.setattr(f"{MODULE}.apply_mouth_animation", _boom)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _boom)

    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(MouthAnimationGenerationError, match="strictly increasing"):
        build_mouth_animated_clip(_scene(duration=5.0), source, audio, "100,50,40,20", out_path)


def test_out_of_order_start_rejects(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: True)
    # starts: 0.0, 2.0, 1.0 -- the third cue's start (1.0) is LESS than
    # the second cue's start (2.0), violating strict monotonic increase.
    monkeypatch.setattr(f"{MODULE}.get_mouth_cues", lambda audio_path: [_cue(0.0, 3.0), _cue(2.0, 4.0), _cue(1.0, 3.0)])

    def _boom(*a, **k):
        raise AssertionError("must not be called")

    monkeypatch.setattr(f"{MODULE}.apply_mouth_animation", _boom)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _boom)

    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(MouthAnimationGenerationError, match="strictly increasing"):
        build_mouth_animated_clip(_scene(duration=5.0), source, audio, "100,50,40,20", out_path)


def test_non_positive_derived_duration_rejects_direct_unit_test():
    """Given the monotonic-start and start<duration checks _validate_cues()
    already enforces, every derived duration is guaranteed positive by
    construction once a cue list has passed _validate_cues() — so this
    defensive check in _derive_durations() is only reachable by calling it
    directly with a hand-crafted cue list that bypasses _validate_cues(),
    exactly as done here."""
    with pytest.raises(MouthAnimationGenerationError, match="must be positive"):
        _derive_durations([_cue(5.0, 5.0)], measured_audio_duration_seconds=5.0)


# ---------------------------------------------------------------------
# output-exists / Rhubarb-unavailable reject before any generation call
# ---------------------------------------------------------------------


def test_existing_output_rejects_before_any_call(tmp_path, monkeypatch):
    _forbid_generation(monkeypatch)
    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"
    out_path.write_bytes(b"pre-existing content nobody asked to touch")

    with pytest.raises(MouthAnimationGenerationError, match="already exists"):
        build_mouth_animated_clip(_scene(), source, audio, "100,50,40,20", out_path)

    assert out_path.read_bytes() == b"pre-existing content nobody asked to touch"


def test_unavailable_rhubarb_rejects_before_any_call(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: False)

    def _boom(*a, **k):
        raise AssertionError("must not be called")

    monkeypatch.setattr(f"{MODULE}.get_mouth_cues", _boom)
    monkeypatch.setattr(f"{MODULE}.apply_mouth_animation", _boom)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _boom)

    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(MouthAnimationGenerationError, match="unavailable"):
        build_mouth_animated_clip(_scene(), source, audio, "100,50,40,20", out_path)
    assert not out_path.exists()


# ---------------------------------------------------------------------
# temp-dir / output cleanup across every failure category
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

    build_mouth_animated_clip(_scene(), source, audio, "100,50,40,20", out_path)

    assert len(created) == 1
    assert not Path(created[0]).exists()
    assert out_path.exists()


def test_temp_dir_removed_after_get_mouth_cues_provider_error(tmp_path, monkeypatch):
    """get_mouth_cues() failing happens BEFORE any temp dir is created at
    all (cue validation/derivation happens before tempfile.mkdtemp()), so
    no temp dir spy is needed here — this documents that ordering."""
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: True)

    def _fail_get_mouth_cues(audio_path):
        raise ProviderError("Rhubarb failed: <raw stderr with internal paths>")

    monkeypatch.setattr(f"{MODULE}.get_mouth_cues", _fail_get_mouth_cues)

    def _boom(*a, **k):
        raise AssertionError("must not be called")

    monkeypatch.setattr(f"{MODULE}.apply_mouth_animation", _boom)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _boom)

    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(MouthAnimationGenerationError) as exc_info:
        build_mouth_animated_clip(_scene(), source, audio, "100,50,40,20", out_path)

    assert str(exc_info.value) == "mouth cue generation failed"
    assert "stderr" not in str(exc_info.value)
    assert not out_path.exists()


def test_temp_dir_removed_after_apply_mouth_animation_oserror(tmp_path, monkeypatch):
    created = _spy_mkdtemp(monkeypatch)
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: True)
    monkeypatch.setattr(f"{MODULE}.get_mouth_cues", lambda audio_path: [_cue(0.0, 5.0)])

    def _fail_apply(scene_image_path, mouth_box, mouth_cues, out_dir):
        raise OSError("disk full: /some/internal/path")

    def _boom(*a, **k):
        raise AssertionError("must not be called after apply_mouth_animation fails")

    monkeypatch.setattr(f"{MODULE}.apply_mouth_animation", _fail_apply)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _boom)

    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(MouthAnimationGenerationError) as exc_info:
        build_mouth_animated_clip(_scene(duration=5.0), source, audio, "100,50,40,20", out_path)

    assert str(exc_info.value) == "mouth frame generation failed"
    assert "/some/internal/path" not in str(exc_info.value)
    assert len(created) == 1
    assert not Path(created[0]).exists()
    assert not out_path.exists()


def test_temp_dir_removed_after_apply_mouth_animation_unexpected_exception(tmp_path, monkeypatch):
    """A non-OSError exception from apply_mouth_animation is NOT wrapped —
    it propagates as itself — but cleanup still runs."""
    created = _spy_mkdtemp(monkeypatch)
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: True)
    monkeypatch.setattr(f"{MODULE}.get_mouth_cues", lambda audio_path: [_cue(0.0, 5.0)])

    sentinel = "SENTINEL-UNEXPECTED-APPLY-4f8a"

    def _fail_apply(scene_image_path, mouth_box, mouth_cues, out_dir):
        raise TypeError(sentinel)

    def _boom(*a, **k):
        raise AssertionError("must not be called")

    monkeypatch.setattr(f"{MODULE}.apply_mouth_animation", _fail_apply)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _boom)

    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(TypeError) as exc_info:
        build_mouth_animated_clip(_scene(duration=5.0), source, audio, "100,50,40,20", out_path)

    assert str(exc_info.value) == sentinel
    assert len(created) == 1
    assert not Path(created[0]).exists()
    assert not out_path.exists()


def test_temp_dir_and_output_removed_after_assemble_provider_error_with_partial_write(tmp_path, monkeypatch):
    created = _spy_mkdtemp(monkeypatch)
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: True)
    monkeypatch.setattr(f"{MODULE}.get_mouth_cues", lambda audio_path: [_cue(0.0, 5.0)])

    def _fake_apply(scene_image_path, mouth_box, mouth_cues, out_dir):
        p = Path(out_dir) / "frame_0000.png"
        p.write_bytes(b"x")
        return [p]

    def _fail_assemble(frame_paths, durations, audio_path, out_path):
        Path(out_path).write_bytes(b"partial-mp4-from-a-crashed-ffmpeg")
        raise ProviderError("ffmpeg failed: <raw stderr with internal paths>")

    monkeypatch.setattr(f"{MODULE}.apply_mouth_animation", _fake_apply)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _fail_assemble)

    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(MouthAnimationGenerationError) as exc_info:
        build_mouth_animated_clip(_scene(duration=5.0), source, audio, "100,50,40,20", out_path)

    assert str(exc_info.value) == "mouth animation render failed"
    assert "stderr" not in str(exc_info.value)
    assert len(created) == 1
    assert not Path(created[0]).exists()
    assert not out_path.exists()


def test_temp_dir_and_output_removed_after_assemble_unexpected_exception(tmp_path, monkeypatch):
    created = _spy_mkdtemp(monkeypatch)
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: True)
    monkeypatch.setattr(f"{MODULE}.get_mouth_cues", lambda audio_path: [_cue(0.0, 5.0)])

    def _fake_apply(scene_image_path, mouth_box, mouth_cues, out_dir):
        p = Path(out_dir) / "frame_0000.png"
        p.write_bytes(b"x")
        return [p]

    sentinel = "SENTINEL-UNEXPECTED-ASSEMBLE-9c2d"

    def _fail_assemble(frame_paths, durations, audio_path, out_path):
        Path(out_path).write_bytes(b"partial-mp4-from-a-missing-binary")
        raise FileNotFoundError(sentinel)

    monkeypatch.setattr(f"{MODULE}.apply_mouth_animation", _fake_apply)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _fail_assemble)

    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(FileNotFoundError) as exc_info:
        build_mouth_animated_clip(_scene(duration=5.0), source, audio, "100,50,40,20", out_path)

    assert str(exc_info.value) == sentinel
    assert len(created) == 1
    assert not Path(created[0]).exists()
    assert not out_path.exists()


# ---------------------------------------------------------------------
# Rhubarb sidecar containment — get_mouth_cues() must receive a private
# temp copy of the audio, never source_audio_path itself, so its
# .lipsync.json sidecar always lands inside tempdir, never next to the
# registered source audio, and is cleaned up with everything else.
# ---------------------------------------------------------------------


def test_get_mouth_cues_receives_a_temp_copy_not_source_audio_path(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: True)
    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"original-audio-bytes")
    out_path = tmp_path / "out.mp4"

    received_paths = []

    def _fake_get_mouth_cues(audio_path):
        received_paths.append(Path(audio_path))
        return [_cue(0.0, 5.0)]

    def _fake_apply(scene_image_path, mouth_box, mouth_cues, out_dir):
        p = Path(out_dir) / "frame_0000.png"
        p.write_bytes(b"x")
        return [p]

    def _fake_assemble(frame_paths, durations, audio_path, out_path):
        Path(out_path).write_bytes(b"fake-mp4")
        return out_path

    monkeypatch.setattr(f"{MODULE}.get_mouth_cues", _fake_get_mouth_cues)
    monkeypatch.setattr(f"{MODULE}.apply_mouth_animation", _fake_apply)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _fake_assemble)

    build_mouth_animated_clip(_scene(duration=5.0), source, audio, "100,50,40,20", out_path)

    assert len(received_paths) == 1
    received = received_paths[0]
    assert received != audio
    assert received.name == "source_audio.wav"
    assert received.parent != audio.parent


def test_temp_audio_copy_and_sidecar_removed_after_success(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: True)
    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    original_audio_bytes = b"original-audio-bytes"
    audio.write_bytes(original_audio_bytes)
    out_path = tmp_path / "out.mp4"

    captured = {}

    def _fake_get_mouth_cues(audio_path):
        audio_path = Path(audio_path)
        assert audio_path.exists()  # the temp copy really is there
        sidecar = audio_path.with_suffix(".lipsync.json")
        sidecar.write_text("{}", encoding="utf-8")
        captured["temp_audio_path"] = audio_path
        captured["sidecar"] = sidecar
        return [_cue(0.0, 5.0)]

    def _fake_apply(scene_image_path, mouth_box, mouth_cues, out_dir):
        p = Path(out_dir) / "frame_0000.png"
        p.write_bytes(b"x")
        return [p]

    def _fake_assemble(frame_paths, durations, audio_path, out_path):
        Path(out_path).write_bytes(b"fake-mp4")
        return out_path

    monkeypatch.setattr(f"{MODULE}.get_mouth_cues", _fake_get_mouth_cues)
    monkeypatch.setattr(f"{MODULE}.apply_mouth_animation", _fake_apply)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _fake_assemble)

    build_mouth_animated_clip(_scene(duration=5.0), source, audio, "100,50,40,20", out_path)

    assert not captured["temp_audio_path"].exists()
    assert not captured["sidecar"].exists()
    assert audio.read_bytes() == original_audio_bytes
    assert not audio.with_suffix(".lipsync.json").exists()


def test_assemble_frame_sequence_receives_original_audio_path_not_temp_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: True)
    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"original-audio-bytes")
    out_path = tmp_path / "out.mp4"

    def _fake_get_mouth_cues(audio_path):
        return [_cue(0.0, 5.0)]

    def _fake_apply(scene_image_path, mouth_box, mouth_cues, out_dir):
        p = Path(out_dir) / "frame_0000.png"
        p.write_bytes(b"x")
        return [p]

    received = {}

    def _fake_assemble(frame_paths, durations, audio_path, out_path):
        received["audio_path"] = Path(audio_path)
        Path(out_path).write_bytes(b"fake-mp4")
        return out_path

    monkeypatch.setattr(f"{MODULE}.get_mouth_cues", _fake_get_mouth_cues)
    monkeypatch.setattr(f"{MODULE}.apply_mouth_animation", _fake_apply)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _fake_assemble)

    build_mouth_animated_clip(_scene(duration=5.0), source, audio, "100,50,40,20", out_path)

    assert received["audio_path"] == audio


def test_temp_audio_copy_oserror_is_sanitized_and_cleans_up(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: True)
    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    original_audio_bytes = b"original-audio-bytes"
    audio.write_bytes(original_audio_bytes)
    out_path = tmp_path / "out.mp4"

    created = _spy_mkdtemp(monkeypatch)

    def _fail_copyfile(src, dst):
        raise OSError("disk full: <raw internal path>")

    monkeypatch.setattr(f"{MODULE}.shutil.copyfile", _fail_copyfile)

    def _boom(*a, **k):
        raise AssertionError("must not be called")

    monkeypatch.setattr(f"{MODULE}.get_mouth_cues", _boom)
    monkeypatch.setattr(f"{MODULE}.apply_mouth_animation", _boom)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _boom)

    with pytest.raises(MouthAnimationGenerationError) as exc_info:
        build_mouth_animated_clip(_scene(duration=5.0), source, audio, "100,50,40,20", out_path)

    assert str(exc_info.value) == "temporary audio preparation failed"
    assert "disk full" not in str(exc_info.value)
    assert len(created) == 1
    assert not Path(created[0]).exists()
    assert audio.read_bytes() == original_audio_bytes
    assert not out_path.exists()


def test_temp_audio_and_sidecar_removed_after_get_mouth_cues_provider_error(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: True)
    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    original_audio_bytes = b"original-audio-bytes"
    audio.write_bytes(original_audio_bytes)
    out_path = tmp_path / "out.mp4"

    captured = {}

    def _fake_get_mouth_cues_fails(audio_path):
        audio_path = Path(audio_path)
        sidecar = audio_path.with_suffix(".lipsync.json")
        sidecar.write_text("{}", encoding="utf-8")
        captured["temp_audio_path"] = audio_path
        captured["sidecar"] = sidecar
        raise ProviderError("Rhubarb failed: <raw internal stderr>")

    def _boom(*a, **k):
        raise AssertionError("must not be called")

    monkeypatch.setattr(f"{MODULE}.get_mouth_cues", _fake_get_mouth_cues_fails)
    monkeypatch.setattr(f"{MODULE}.apply_mouth_animation", _boom)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _boom)

    with pytest.raises(MouthAnimationGenerationError) as exc_info:
        build_mouth_animated_clip(_scene(duration=5.0), source, audio, "100,50,40,20", out_path)

    assert str(exc_info.value) == "mouth cue generation failed"
    assert "stderr" not in str(exc_info.value)
    assert not captured["temp_audio_path"].exists()
    assert not captured["sidecar"].exists()
    assert audio.read_bytes() == original_audio_bytes
    assert not audio.with_suffix(".lipsync.json").exists()
    assert not out_path.exists()


def test_temp_audio_and_sidecar_removed_after_get_mouth_cues_unexpected_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: True)
    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    original_audio_bytes = b"original-audio-bytes"
    audio.write_bytes(original_audio_bytes)
    out_path = tmp_path / "out.mp4"

    sentinel = "SENTINEL-UNEXPECTED-GET-MOUTH-CUES-2b7f"
    captured = {}

    def _fake_get_mouth_cues_fails(audio_path):
        audio_path = Path(audio_path)
        sidecar = audio_path.with_suffix(".lipsync.json")
        sidecar.write_text("{}", encoding="utf-8")
        captured["temp_audio_path"] = audio_path
        captured["sidecar"] = sidecar
        raise TypeError(sentinel)

    def _boom(*a, **k):
        raise AssertionError("must not be called")

    monkeypatch.setattr(f"{MODULE}.get_mouth_cues", _fake_get_mouth_cues_fails)
    monkeypatch.setattr(f"{MODULE}.apply_mouth_animation", _boom)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _boom)

    with pytest.raises(TypeError) as exc_info:
        build_mouth_animated_clip(_scene(duration=5.0), source, audio, "100,50,40,20", out_path)

    assert str(exc_info.value) == sentinel
    assert not captured["temp_audio_path"].exists()
    assert not captured["sidecar"].exists()
    assert audio.read_bytes() == original_audio_bytes
    assert not out_path.exists()


def test_temp_audio_and_sidecar_removed_after_cue_validation_rejection(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: True)
    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    original_audio_bytes = b"original-audio-bytes"
    audio.write_bytes(original_audio_bytes)
    out_path = tmp_path / "out.mp4"

    captured = {}

    def _fake_get_mouth_cues_bad(audio_path):
        audio_path = Path(audio_path)
        sidecar = audio_path.with_suffix(".lipsync.json")
        sidecar.write_text("{}", encoding="utf-8")
        captured["temp_audio_path"] = audio_path
        captured["sidecar"] = sidecar
        return [_cue(0.5, 1.0)]  # nonzero first start -> rejected by _validate_cues

    def _boom(*a, **k):
        raise AssertionError("must not be called")

    monkeypatch.setattr(f"{MODULE}.get_mouth_cues", _fake_get_mouth_cues_bad)
    monkeypatch.setattr(f"{MODULE}.apply_mouth_animation", _boom)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _boom)

    with pytest.raises(MouthAnimationGenerationError, match="must start at 0.0"):
        build_mouth_animated_clip(_scene(duration=5.0), source, audio, "100,50,40,20", out_path)

    assert not captured["temp_audio_path"].exists()
    assert not captured["sidecar"].exists()
    assert audio.read_bytes() == original_audio_bytes
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
    _png(source)
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

    build_mouth_animated_clip(scene, source, audio, "100,50,40,20", out_path)

    assert out_path.exists()
    duration = ffmpeg_render.get_duration_seconds(out_path)
    assert duration == pytest.approx(1.0, abs=0.3)
