"""Tests for src/core/limb_sway_generation.py — the pure local
orchestration of apply_limb_sway() + assemble_frame_sequence(). Every
ordinary test mocks health_check/apply_limb_sway/assemble_frame_sequence
at their point of import in this module — no real ffmpeg, no Rhubarb (this
module never imports lipsync_rhubarb at all), no network call anywhere in
this file, except the one optional, doubly-gated... actually singly-gated
(ffmpeg only) integration test at the bottom."""
from __future__ import annotations

import math
from pathlib import Path

import pytest
from PIL import Image

from src.core.limb_sway_generation import (
    LimbSwayGenerationError,
    _generate_angles,
    _generate_durations,
    build_limb_sway_clip,
    parse_amplitude_degrees,
    parse_frame_count,
    parse_limb_box,
    parse_sway_cycles,
    validate_image_and_limb_box,
)
from src.models.scene import SceneArtifacts, ScenePlanItem
from src.providers.base import ProviderError
from src.render.character_rig import LimbBox

MODULE = "src.core.limb_sway_generation"


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


def _forbid_generation(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("must not be called")

    monkeypatch.setattr(f"{MODULE}.apply_limb_sway", _boom)
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


def _mock_ok(monkeypatch, calls: list):
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: True)

    def _fake_apply(source_visual_path, limb_box, angles, out_dir):
        calls.append(("apply_limb_sway", Path(source_visual_path), limb_box, list(angles), Path(out_dir)))
        frames = []
        for i in range(len(angles)):
            p = Path(out_dir) / f"limb_{i:04d}.png"
            p.write_bytes(b"fake-frame")
            frames.append(p)
        return frames

    def _fake_assemble(frame_paths, durations, audio_path, out_path):
        calls.append(("assemble_frame_sequence", list(frame_paths), list(durations), Path(audio_path), Path(out_path)))
        Path(out_path).write_bytes(b"fake-mp4-bytes")
        return out_path

    monkeypatch.setattr(f"{MODULE}.apply_limb_sway", _fake_apply)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _fake_assemble)


# ---------------------------------------------------------------------
# parse_limb_box
# ---------------------------------------------------------------------


def test_parse_limb_box_valid():
    box = parse_limb_box("60,60,140,140,100,70")
    assert box == LimbBox(left=60, top=60, right=140, bottom=140, pivot_x=100, pivot_y=70)


def test_parse_limb_box_accepts_leading_zeros():
    box = parse_limb_box("060,060,140,140,100,070")
    assert box == LimbBox(left=60, top=60, right=140, bottom=140, pivot_x=100, pivot_y=70)


def test_parse_limb_box_wrong_component_count():
    with pytest.raises(LimbSwayGenerationError, match="exactly six comma-separated integers"):
        parse_limb_box("60,60,140,140,100")


def test_parse_limb_box_too_many_components():
    with pytest.raises(LimbSwayGenerationError, match="exactly six comma-separated integers"):
        parse_limb_box("60,60,140,140,100,70,1")


@pytest.mark.parametrize(
    "raw",
    [
        "-60,60,140,140,100,70",  # sign
        " 60,60,140,140,100,70",  # leading whitespace
        "60,60,140,140,100,70 ",  # trailing whitespace
        "60,60,140,140,100,70\n",  # trailing newline
        "1_00,60,140,140,100,70",  # underscore grouping
        "60.0,60,140,140,100,70",  # decimal
        "6e1,60,140,140,100,70",  # exponent notation
        ",60,140,140,100,70",  # empty token
        "true,60,140,140,100,70",  # boolean-like
        "٤٥,60,140,140,100,70",  # Arabic-Indic digits
    ],
)
def test_parse_limb_box_rejects_invalid_tokens(raw):
    with pytest.raises(LimbSwayGenerationError, match="base-10 integers"):
        parse_limb_box(raw)


# ---------------------------------------------------------------------
# parse_frame_count
# ---------------------------------------------------------------------


def test_parse_frame_count_valid():
    assert parse_frame_count("10") == 10


def test_parse_frame_count_accepts_leading_zeros():
    assert parse_frame_count("002") == 2


@pytest.mark.parametrize("raw", ["-2", " 2", "2 ", "2.0", "2e1", "", "true", "٢"])
def test_parse_frame_count_rejects_invalid_tokens(raw):
    with pytest.raises(LimbSwayGenerationError, match="base-10 integer"):
        parse_frame_count(raw)


def test_parse_frame_count_zero_rejected():
    with pytest.raises(LimbSwayGenerationError, match="at least 2"):
        parse_frame_count("0")


def test_parse_frame_count_one_rejected():
    with pytest.raises(LimbSwayGenerationError, match="at least 2"):
        parse_frame_count("1")


def test_parse_frame_count_two_accepted():
    assert parse_frame_count("2") == 2


# ---------------------------------------------------------------------
# parse_amplitude_degrees / parse_sway_cycles — shared strict decimal grammar
# ---------------------------------------------------------------------


@pytest.mark.parametrize("parser", [parse_amplitude_degrees, parse_sway_cycles])
class TestSharedDecimalGrammar:
    def test_valid_integer(self, parser):
        assert parser("15") == 15.0

    def test_valid_decimal(self, parser):
        assert parser("15.5") == 15.5

    def test_leading_zeros_allowed(self, parser):
        assert parser("015.500") == 15.5

    @pytest.mark.parametrize(
        "raw",
        [
            "-15",  # sign
            "+15",  # sign
            " 15",  # leading whitespace
            "15 ",  # trailing whitespace
            "15\n",  # trailing newline
            "1_5",  # underscore
            "15.",  # trailing dot, no digits after
            ".5",  # leading dot, no digits before
            "1e5",  # exponent
            "",  # empty
            "true",  # boolean-like
            "NaN",  # explicit NaN text
            "Infinity",  # explicit Infinity text
            "١٥",  # Arabic-Indic digits
        ],
    )
    def test_rejects_invalid_token(self, parser, raw):
        with pytest.raises(LimbSwayGenerationError, match="valid decimal number"):
            parser(raw)

    def test_rejects_non_finite_overflow(self, parser):
        huge = "9" * 400  # grammar-valid, but float() overflows this to inf
        with pytest.raises(LimbSwayGenerationError, match="must be finite"):
            parser(huge)

    def test_rejects_zero(self, parser):
        with pytest.raises(LimbSwayGenerationError, match="must be positive"):
            parser("0")

    def test_rejects_zero_decimal(self, parser):
        with pytest.raises(LimbSwayGenerationError, match="must be positive"):
            parser("0.0")


# ---------------------------------------------------------------------
# validate_image_and_limb_box — box and pivot bounds
# ---------------------------------------------------------------------


def test_validate_image_and_limb_box_valid(tmp_path):
    image_path = tmp_path / "scene.png"
    _png(image_path, (200, 100))
    box = LimbBox(left=60, top=60, right=140, bottom=95, pivot_x=100, pivot_y=70)

    width, height = validate_image_and_limb_box(image_path, box)

    assert (width, height) == (200, 100)


@pytest.mark.parametrize(
    "box_kwargs",
    [
        dict(left=-1, top=10, right=50, bottom=50, pivot_x=20, pivot_y=20),  # left < 0
        dict(left=10, top=10, right=250, bottom=50, pivot_x=20, pivot_y=20),  # right > image_width
        dict(left=10, top=-1, right=50, bottom=50, pivot_x=20, pivot_y=20),  # top < 0
        dict(left=10, top=10, right=50, bottom=150, pivot_x=20, pivot_y=20),  # bottom > image_height
    ],
)
def test_validate_image_and_limb_box_rejects_box_out_of_bounds(tmp_path, box_kwargs):
    image_path = tmp_path / "scene.png"
    _png(image_path, (200, 100))
    box = LimbBox(**box_kwargs)

    with pytest.raises(LimbSwayGenerationError, match="extends outside the image bounds"):
        validate_image_and_limb_box(image_path, box)


@pytest.mark.parametrize(
    "box_kwargs",
    [
        dict(left=10, top=10, right=50, bottom=50, pivot_x=5, pivot_y=20),  # pivot_x < left
        dict(left=10, top=10, right=50, bottom=50, pivot_x=50, pivot_y=20),  # pivot_x == right (not < right)
        dict(left=10, top=10, right=50, bottom=50, pivot_x=20, pivot_y=5),  # pivot_y < top
        dict(left=10, top=10, right=50, bottom=50, pivot_x=20, pivot_y=50),  # pivot_y == bottom
    ],
)
def test_validate_image_and_limb_box_rejects_pivot_out_of_bounds(tmp_path, box_kwargs):
    image_path = tmp_path / "scene.png"
    _png(image_path, (200, 100))
    box = LimbBox(**box_kwargs)

    with pytest.raises(LimbSwayGenerationError, match="pivot is outside the box bounds"):
        validate_image_and_limb_box(image_path, box)


def test_validate_image_and_limb_box_rejects_corrupt_file(tmp_path):
    image_path = tmp_path / "not_an_image.png"
    image_path.write_bytes(b"this is not a valid PNG file at all")
    box = LimbBox(left=1, top=1, right=4, bottom=4, pivot_x=2, pivot_y=2)

    with pytest.raises(LimbSwayGenerationError, match="could not be opened or is corrupt"):
        validate_image_and_limb_box(image_path, box)


# ---------------------------------------------------------------------
# angle / duration generation
# ---------------------------------------------------------------------


def test_generate_angles_n2_cycles_1_first_and_last_approximately_zero():
    angles = _generate_angles(amplitude_degrees=20.0, frame_count=2, cycles=1.0)
    assert len(angles) == 2
    assert angles[0] == pytest.approx(0.0, abs=1e-9)
    assert angles[-1] == pytest.approx(0.0, abs=1e-9)


def test_generate_angles_integer_cycles_first_and_last_approximately_zero():
    angles = _generate_angles(amplitude_degrees=15.0, frame_count=10, cycles=2.0)
    assert len(angles) == 10
    assert angles[0] == pytest.approx(0.0, abs=1e-9)
    assert angles[-1] == pytest.approx(0.0, abs=1e-9)


def test_generate_angles_fractional_cycles_last_is_not_approximately_zero():
    angles = _generate_angles(amplitude_degrees=15.0, frame_count=10, cycles=1.3)
    assert angles[0] == pytest.approx(0.0, abs=1e-9)
    assert abs(angles[-1]) > 1.0  # nowhere near zero for a genuinely fractional cycle count


def test_generate_angles_exact_count():
    angles = _generate_angles(amplitude_degrees=10.0, frame_count=7, cycles=1.0)
    assert len(angles) == 7
    assert all(math.isfinite(a) for a in angles)


def test_generate_durations_exact_count_and_equal():
    durations = _generate_durations(measured_audio_duration_seconds=6.0, frame_count=4)
    assert len(durations) == 4
    assert all(d == pytest.approx(1.5) for d in durations)


def test_generate_durations_sum_within_tolerance():
    total = 7.3
    n = 13
    durations = _generate_durations(measured_audio_duration_seconds=total, frame_count=n)
    assert sum(durations) == pytest.approx(total, abs=1e-9)


# ---------------------------------------------------------------------
# happy path (build_limb_sway_clip end-to-end, mocked generation stages)
# ---------------------------------------------------------------------


def test_calls_apply_limb_sway_before_assemble_frame_sequence(tmp_path, monkeypatch):
    calls: list = []
    _mock_ok(monkeypatch, calls)
    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    build_limb_sway_clip(_scene(duration=5.0), source, audio, "60,60,140,95,100,70", "20", "4", "1.0", out_path)

    assert [c[0] for c in calls] == ["apply_limb_sway", "assemble_frame_sequence"]


def test_exactly_n_angles_and_n_durations_passed_through(tmp_path, monkeypatch):
    calls: list = []
    _mock_ok(monkeypatch, calls)
    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    build_limb_sway_clip(_scene(duration=8.0), source, audio, "60,60,140,95,100,70", "20", "5", "1.0", out_path)

    apply_call = next(c for c in calls if c[0] == "apply_limb_sway")
    assemble_call = next(c for c in calls if c[0] == "assemble_frame_sequence")
    assert len(apply_call[3]) == 5  # angles
    assert len(assemble_call[2]) == 5  # durations
    assert all(d == pytest.approx(8.0 / 5) for d in assemble_call[2])


def test_assemble_receives_original_audio_path(tmp_path, monkeypatch):
    calls: list = []
    _mock_ok(monkeypatch, calls)
    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"original-audio-bytes")
    out_path = tmp_path / "out.mp4"

    build_limb_sway_clip(_scene(duration=5.0), source, audio, "60,60,140,95,100,70", "20", "4", "1.0", out_path)

    assemble_call = next(c for c in calls if c[0] == "assemble_frame_sequence")
    assert assemble_call[3] == audio


def test_source_image_and_audio_bytes_unchanged(tmp_path, monkeypatch):
    calls: list = []
    _mock_ok(monkeypatch, calls)
    source = tmp_path / "source.png"
    _png(source)
    original_image_bytes = source.read_bytes()
    audio = tmp_path / "audio.wav"
    original_audio_bytes = b"original-audio-bytes"
    audio.write_bytes(original_audio_bytes)
    out_path = tmp_path / "out.mp4"

    build_limb_sway_clip(_scene(duration=5.0), source, audio, "60,60,140,95,100,70", "20", "4", "1.0", out_path)

    assert source.read_bytes() == original_image_bytes
    assert audio.read_bytes() == original_audio_bytes


def test_temp_dir_removed_after_success(tmp_path, monkeypatch):
    calls: list = []
    created = _spy_mkdtemp(monkeypatch)
    _mock_ok(monkeypatch, calls)
    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    build_limb_sway_clip(_scene(duration=5.0), source, audio, "60,60,140,95,100,70", "20", "4", "1.0", out_path)

    assert len(created) == 1
    assert not Path(created[0]).exists()
    assert out_path.exists()


def test_default_sway_cycles_used_when_supplied_as_default_string(tmp_path, monkeypatch):
    """The CLI always supplies SOME raw string for --sway-cycles, including
    its own '1.0' default — this proves the default value is handled
    identically to an explicit one."""
    calls: list = []
    _mock_ok(monkeypatch, calls)
    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    build_limb_sway_clip(_scene(duration=5.0), source, audio, "60,60,140,95,100,70", "20", "2", "1.0", out_path)

    apply_call = next(c for c in calls if c[0] == "apply_limb_sway")
    angles = apply_call[3]
    assert angles[0] == pytest.approx(0.0, abs=1e-9)
    assert angles[-1] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------
# rejections that must happen before ANY mocked call
# ---------------------------------------------------------------------


def test_existing_output_rejects_before_any_call(tmp_path, monkeypatch):
    _forbid_generation(monkeypatch)
    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"
    out_path.write_bytes(b"pre-existing content nobody asked to touch")

    with pytest.raises(LimbSwayGenerationError, match="already exists"):
        build_limb_sway_clip(_scene(), source, audio, "60,60,140,95,100,70", "20", "4", "1.0", out_path)

    assert out_path.read_bytes() == b"pre-existing content nobody asked to touch"


def test_missing_measured_duration_rejects_before_any_call(tmp_path, monkeypatch):
    _forbid_generation(monkeypatch)
    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(LimbSwayGenerationError, match="measured_audio_duration_seconds"):
        build_limb_sway_clip(_scene(duration=None), source, audio, "60,60,140,95,100,70", "20", "4", "1.0", out_path)

    assert not out_path.exists()


def test_invalid_limb_box_rejects_before_any_call(tmp_path, monkeypatch):
    _forbid_generation(monkeypatch)
    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(LimbSwayGenerationError, match="base-10 integers"):
        build_limb_sway_clip(_scene(), source, audio, "abc,60,140,95,100,70", "20", "4", "1.0", out_path)

    assert not out_path.exists()


def test_invalid_frame_count_rejects_before_any_call(tmp_path, monkeypatch):
    _forbid_generation(monkeypatch)
    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(LimbSwayGenerationError, match="at least 2"):
        build_limb_sway_clip(_scene(), source, audio, "60,60,140,95,100,70", "20", "1", "1.0", out_path)

    assert not out_path.exists()


def test_invalid_amplitude_rejects_before_any_call(tmp_path, monkeypatch):
    _forbid_generation(monkeypatch)
    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(LimbSwayGenerationError, match="sway-amplitude-degrees"):
        build_limb_sway_clip(_scene(), source, audio, "60,60,140,95,100,70", "-5", "4", "1.0", out_path)

    assert not out_path.exists()


def test_invalid_cycles_rejects_before_any_call(tmp_path, monkeypatch):
    _forbid_generation(monkeypatch)
    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(LimbSwayGenerationError, match="sway-cycles"):
        build_limb_sway_clip(_scene(), source, audio, "60,60,140,95,100,70", "20", "4", "0", out_path)

    assert not out_path.exists()


def test_image_box_bounds_rejects_before_any_call(tmp_path, monkeypatch):
    _forbid_generation(monkeypatch)
    source = tmp_path / "source.png"
    _png(source, size=(200, 100))
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(LimbSwayGenerationError, match="extends outside the image bounds"):
        build_limb_sway_clip(_scene(), source, audio, "60,60,250,95,100,70", "20", "4", "1.0", out_path)

    assert not out_path.exists()


# ---------------------------------------------------------------------
# ffmpeg availability gate — checked last, right before tempfile.mkdtemp(),
# i.e. AFTER every argument/image/LimbBox validation above has already
# passed (a valid fixture is used here specifically to prove that
# ordering, distinct from the "before any call" tests above which reject
# on invalid arguments before ever reaching this gate)
# ---------------------------------------------------------------------


def test_unavailable_ffmpeg_rejects_with_exact_sanitized_message(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: False)

    def _boom(*a, **k):
        raise AssertionError("must not be called when ffmpeg is unavailable")

    monkeypatch.setattr(f"{MODULE}.apply_limb_sway", _boom)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _boom)

    source = tmp_path / "source.png"
    _png(source, size=(200, 100))
    original_image_bytes = source.read_bytes()
    audio = tmp_path / "audio.wav"
    original_audio_bytes = b"original-audio-bytes"
    audio.write_bytes(original_audio_bytes)
    out_path = tmp_path / "out.mp4"

    with pytest.raises(LimbSwayGenerationError) as exc_info:
        build_limb_sway_clip(
            _scene(duration=5.0), source, audio, "60,60,140,95,100,70", "20", "4", "1.0", out_path
        )

    assert str(exc_info.value) == "ffmpeg is unavailable (binary not found)"
    assert not out_path.exists()
    assert source.read_bytes() == original_image_bytes
    assert audio.read_bytes() == original_audio_bytes


def test_unavailable_ffmpeg_creates_no_temp_directory(tmp_path, monkeypatch):
    created = _spy_mkdtemp(monkeypatch)
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: False)

    def _boom(*a, **k):
        raise AssertionError("must not be called when ffmpeg is unavailable")

    monkeypatch.setattr(f"{MODULE}.apply_limb_sway", _boom)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _boom)

    source = tmp_path / "source.png"
    _png(source, size=(200, 100))
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(LimbSwayGenerationError, match="ffmpeg is unavailable"):
        build_limb_sway_clip(
            _scene(duration=5.0), source, audio, "60,60,140,95,100,70", "20", "4", "1.0", out_path
        )

    assert created == []  # tempfile.mkdtemp() was never called at all


# ---------------------------------------------------------------------
# temp-dir / output cleanup across every failure category
# ---------------------------------------------------------------------


def test_temp_dir_removed_after_apply_limb_sway_oserror(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: True)
    created = _spy_mkdtemp(monkeypatch)

    def _fail_apply(source_visual_path, limb_box, angles, out_dir):
        raise OSError("disk full: /some/internal/path")

    def _boom(*a, **k):
        raise AssertionError("must not be called after apply_limb_sway fails")

    monkeypatch.setattr(f"{MODULE}.apply_limb_sway", _fail_apply)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _boom)

    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(LimbSwayGenerationError) as exc_info:
        build_limb_sway_clip(_scene(duration=5.0), source, audio, "60,60,140,95,100,70", "20", "4", "1.0", out_path)

    assert str(exc_info.value) == "limb sway frame generation failed"
    assert "/some/internal/path" not in str(exc_info.value)
    assert len(created) == 1
    assert not Path(created[0]).exists()
    assert not out_path.exists()


def test_temp_dir_removed_after_apply_limb_sway_unexpected_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: True)
    created = _spy_mkdtemp(monkeypatch)
    sentinel = "SENTINEL-UNEXPECTED-APPLY-6a1f"

    def _fail_apply(source_visual_path, limb_box, angles, out_dir):
        raise TypeError(sentinel)

    def _boom(*a, **k):
        raise AssertionError("must not be called")

    monkeypatch.setattr(f"{MODULE}.apply_limb_sway", _fail_apply)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _boom)

    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(TypeError) as exc_info:
        build_limb_sway_clip(_scene(duration=5.0), source, audio, "60,60,140,95,100,70", "20", "4", "1.0", out_path)

    assert str(exc_info.value) == sentinel
    assert len(created) == 1
    assert not Path(created[0]).exists()
    assert not out_path.exists()


def test_temp_dir_and_output_removed_after_assemble_provider_error_with_partial_write(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: True)
    created = _spy_mkdtemp(monkeypatch)

    def _fake_apply(source_visual_path, limb_box, angles, out_dir):
        p = Path(out_dir) / "limb_0000.png"
        p.write_bytes(b"x")
        return [p]

    def _fail_assemble(frame_paths, durations, audio_path, out_path):
        Path(out_path).write_bytes(b"partial-mp4-from-a-crashed-ffmpeg")
        raise ProviderError("ffmpeg failed: <raw internal stderr>")

    monkeypatch.setattr(f"{MODULE}.apply_limb_sway", _fake_apply)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _fail_assemble)

    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(LimbSwayGenerationError) as exc_info:
        build_limb_sway_clip(_scene(duration=5.0), source, audio, "60,60,140,95,100,70", "20", "4", "1.0", out_path)

    assert str(exc_info.value) == "limb sway render failed"
    assert "stderr" not in str(exc_info.value)
    assert len(created) == 1
    assert not Path(created[0]).exists()
    assert not out_path.exists()


def test_temp_dir_and_output_removed_after_assemble_unexpected_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: True)
    created = _spy_mkdtemp(monkeypatch)

    def _fake_apply(source_visual_path, limb_box, angles, out_dir):
        p = Path(out_dir) / "limb_0000.png"
        p.write_bytes(b"x")
        return [p]

    sentinel = "SENTINEL-UNEXPECTED-ASSEMBLE-9f3c"

    def _fail_assemble(frame_paths, durations, audio_path, out_path):
        Path(out_path).write_bytes(b"partial-mp4-from-a-missing-binary")
        raise FileNotFoundError(sentinel)

    monkeypatch.setattr(f"{MODULE}.apply_limb_sway", _fake_apply)
    monkeypatch.setattr(f"{MODULE}.assemble_frame_sequence", _fail_assemble)

    source = tmp_path / "source.png"
    _png(source)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(FileNotFoundError) as exc_info:
        build_limb_sway_clip(_scene(duration=5.0), source, audio, "60,60,140,95,100,70", "20", "4", "1.0", out_path)

    assert str(exc_info.value) == sentinel
    assert len(created) == 1
    assert not Path(created[0]).exists()
    assert not out_path.exists()


# ---------------------------------------------------------------------
# optional integration test — real ffmpeg_render.health_check() gate only
# (no Rhubarb dependency in this module at all); skips cleanly here
# ---------------------------------------------------------------------


def test_real_pipeline_integration(tmp_path):
    import subprocess

    from src.render import ffmpeg_render
    from src.utils.config import get_settings

    if not ffmpeg_render.health_check():
        pytest.skip("real ffmpeg not available in this environment")

    source = tmp_path / "source.png"
    _png(source, size=(200, 100))
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

    build_limb_sway_clip(scene, source, audio, "60,60,140,95,100,70", "20", "4", "1.0", out_path)

    assert out_path.exists()
    duration = ffmpeg_render.get_duration_seconds(out_path)
    assert duration == pytest.approx(1.0, abs=0.3)
