"""Tests for src/core/ken_burns_upscale_pipeline.py — the pure local
orchestration of upscale_image() + ken_burns_clip(). Every test mocks
health_check/upscale_image/ken_burns_clip at their point of import in the
pipeline module (same monkeypatch.setattr("src.core.<module>.<name>", fake)
convention every artifact registrar test already uses) — no real
Real-ESRGAN binary, no real ffmpeg, no network call anywhere in this
file."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.core.ken_burns_upscale_pipeline import (
    KenBurnsUpscalePipelineError,
    build_upscaled_ken_burns_clip,
)
from src.models.scene import SceneArtifacts, ScenePlanItem
from src.providers.base import ProviderError

MODULE = "src.core.ken_burns_upscale_pipeline"


def _scene(motion_mode: str = "static", duration=5.0, scene_id: str = "scene-01") -> ScenePlanItem:
    """`SceneArtifacts.measured_audio_duration_seconds` is a strictly-typed
    Pydantic `float | None` field, so a normal constructor call can never
    hold a bool/str (Pydantic itself rejects those at construction). The
    defensive isinstance/bool checks in _validate_duration() exist anyway,
    matching scene_timing_finalizer's own checks on the same category of
    value one layer up (an ArtifactRecord.metadata JSON blob, which IS
    loosely typed) — to test them here, bypass validation with
    model_construct() the same way a corrupted/hand-built record could."""
    scene = ScenePlanItem(
        scene_id=scene_id,
        sequence=1,
        narration_text="Narration text.",
        scene_type="narration",
        narrative_beat="setup",
        visual_brief="Visual brief.",
        motion_mode=motion_mode,
        approval_state="approved",
        artifacts=SceneArtifacts(measured_audio_duration_seconds=5.0),
    )
    if isinstance(duration, bool) or isinstance(duration, str):
        bad_artifacts = SceneArtifacts.model_construct(measured_audio_duration_seconds=duration)
        return scene.model_copy(update={"artifacts": bad_artifacts})
    return scene.model_copy(update={"artifacts": SceneArtifacts(measured_audio_duration_seconds=duration)})


def _mock_ok(monkeypatch, calls: list, *, spy_mkdtemp: bool = False):
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: True)

    def _fake_upscale(src, dst, *args, **kwargs):
        calls.append(("upscale_image", Path(src), Path(dst)))
        Path(dst).write_bytes(b"fake-upscaled-png-bytes")
        return dst

    def _fake_ken_burns(image_path, duration, out_path, **kwargs):
        calls.append(("ken_burns_clip", Path(image_path), duration, Path(out_path), kwargs))
        Path(out_path).write_bytes(b"fake-mp4-bytes")
        return out_path

    monkeypatch.setattr(f"{MODULE}.upscale_image", _fake_upscale)
    monkeypatch.setattr(f"{MODULE}.ken_burns_clip", _fake_ken_burns)

    if spy_mkdtemp:
        import tempfile

        created: list[str] = []
        real_mkdtemp = tempfile.mkdtemp

        def _spy(*args, **kwargs):
            d = real_mkdtemp(*args, **kwargs)
            created.append(d)
            return d

        monkeypatch.setattr(f"{MODULE}.tempfile.mkdtemp", _spy)
        return created


def _forbid_transformations(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("must not be called")

    monkeypatch.setattr(f"{MODULE}.health_check", _boom)
    monkeypatch.setattr(f"{MODULE}.upscale_image", _boom)
    monkeypatch.setattr(f"{MODULE}.ken_burns_clip", _boom)


# ---------------------------------------------------------------------
# happy path: order, path isolation, propagation, source bytes, cleanup
# ---------------------------------------------------------------------


def test_upscale_is_called_before_ken_burns(tmp_path, monkeypatch):
    calls: list = []
    _mock_ok(monkeypatch, calls)
    source = tmp_path / "source.png"
    source.write_bytes(b"original-png-bytes")
    out_path = tmp_path / "out.mp4"

    build_upscaled_ken_burns_clip(_scene(), source, out_path)

    assert [c[0] for c in calls] == ["upscale_image", "ken_burns_clip"]


def test_ken_burns_receives_the_upscaled_path_not_the_original(tmp_path, monkeypatch):
    calls: list = []
    _mock_ok(monkeypatch, calls)
    source = tmp_path / "source.png"
    source.write_bytes(b"original-png-bytes")
    out_path = tmp_path / "out.mp4"

    build_upscaled_ken_burns_clip(_scene(), source, out_path)

    upscale_call = next(c for c in calls if c[0] == "upscale_image")
    ken_burns_call = next(c for c in calls if c[0] == "ken_burns_clip")
    assert upscale_call[1] == source
    assert ken_burns_call[1] == upscale_call[2]  # the image path ken_burns_clip got IS upscale's out_path
    assert ken_burns_call[1] != source


def test_duration_and_motion_mode_are_passed_through_exactly(tmp_path, monkeypatch):
    calls: list = []
    _mock_ok(monkeypatch, calls)
    source = tmp_path / "source.png"
    source.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    build_upscaled_ken_burns_clip(_scene(motion_mode="pan_lr", duration=7.25), source, out_path)

    ken_burns_call = next(c for c in calls if c[0] == "ken_burns_clip")
    assert ken_burns_call[2] == 7.25
    assert ken_burns_call[4]["motion"] == "pan_lr"


def test_source_visual_bytes_remain_unchanged(tmp_path, monkeypatch):
    calls: list = []
    _mock_ok(monkeypatch, calls)
    source = tmp_path / "source.png"
    original_bytes = b"original-png-bytes-must-survive"
    source.write_bytes(original_bytes)
    out_path = tmp_path / "out.mp4"

    build_upscaled_ken_burns_clip(_scene(), source, out_path)

    assert source.read_bytes() == original_bytes


def test_temp_dir_is_removed_after_success(tmp_path, monkeypatch):
    calls: list = []
    created = _mock_ok(monkeypatch, calls, spy_mkdtemp=True)
    source = tmp_path / "source.png"
    source.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    build_upscaled_ken_burns_clip(_scene(), source, out_path)

    assert len(created) == 1
    assert not Path(created[0]).exists()
    assert out_path.exists()


# ---------------------------------------------------------------------
# failures: temp-dir cleanup, output cleanup, sanitized messages
# ---------------------------------------------------------------------


def test_temp_dir_is_removed_after_upscale_failure(tmp_path, monkeypatch):
    import tempfile

    created: list[str] = []
    real_mkdtemp = tempfile.mkdtemp

    def _spy(*args, **kwargs):
        d = real_mkdtemp(*args, **kwargs)
        created.append(d)
        return d

    monkeypatch.setattr(f"{MODULE}.tempfile.mkdtemp", _spy)
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: True)

    def _fake_upscale_fail(src, dst, *args, **kwargs):
        raise ProviderError("realesrgan-ncnn-vulkan failed: <raw stderr with internal paths>")

    monkeypatch.setattr(f"{MODULE}.upscale_image", _fake_upscale_fail)

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("ken_burns_clip must not be called after an upscale failure")

    monkeypatch.setattr(f"{MODULE}.ken_burns_clip", _must_not_be_called)

    source = tmp_path / "source.png"
    source.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(KenBurnsUpscalePipelineError) as exc_info:
        build_upscaled_ken_burns_clip(_scene(), source, out_path)

    assert str(exc_info.value) == "image upscale failed"
    assert "stderr" not in str(exc_info.value)
    assert len(created) == 1
    assert not Path(created[0]).exists()
    assert not out_path.exists()


def test_temp_dir_and_output_are_removed_after_ken_burns_failure_with_partial_write(tmp_path, monkeypatch):
    import tempfile

    created: list[str] = []
    real_mkdtemp = tempfile.mkdtemp

    def _spy(*args, **kwargs):
        d = real_mkdtemp(*args, **kwargs)
        created.append(d)
        return d

    monkeypatch.setattr(f"{MODULE}.tempfile.mkdtemp", _spy)
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: True)

    def _fake_upscale(src, dst, *args, **kwargs):
        Path(dst).write_bytes(b"fake-upscaled")
        return dst

    monkeypatch.setattr(f"{MODULE}.upscale_image", _fake_upscale)

    def _fake_ken_burns_partial_then_fail(image_path, duration, out_path, **kwargs):
        Path(out_path).write_bytes(b"partial-mp4-bytes-from-a-crashed-ffmpeg")
        raise ProviderError("ffmpeg failed: <raw stderr with internal paths>")

    monkeypatch.setattr(f"{MODULE}.ken_burns_clip", _fake_ken_burns_partial_then_fail)

    source = tmp_path / "source.png"
    source.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(KenBurnsUpscalePipelineError) as exc_info:
        build_upscaled_ken_burns_clip(_scene(), source, out_path)

    assert str(exc_info.value) == "Ken Burns render failed"
    assert "stderr" not in str(exc_info.value)
    assert len(created) == 1
    assert not Path(created[0]).exists()
    assert not out_path.exists()


def test_ken_burns_failure_without_any_write_leaves_no_output(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: True)

    def _fake_upscale(src, dst, *args, **kwargs):
        Path(dst).write_bytes(b"fake-upscaled")
        return dst

    monkeypatch.setattr(f"{MODULE}.upscale_image", _fake_upscale)

    def _fake_ken_burns_fail_no_write(image_path, duration, out_path, **kwargs):
        raise ProviderError("ffmpeg failed: <raw stderr>")

    monkeypatch.setattr(f"{MODULE}.ken_burns_clip", _fake_ken_burns_fail_no_write)

    source = tmp_path / "source.png"
    source.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(KenBurnsUpscalePipelineError):
        build_upscaled_ken_burns_clip(_scene(), source, out_path)

    assert not out_path.exists()


def test_ken_burns_file_not_found_error_propagates_unwrapped_and_still_cleans_up(tmp_path, monkeypatch):
    """A non-ProviderError exception (e.g. subprocess.run() itself raising
    FileNotFoundError for a misconfigured ffmpeg path) must NOT be caught
    or sanitized into KenBurnsUpscalePipelineError — it propagates exactly
    as raised — but cleanup (temp dir + an owned, partially-written
    out_path) must still happen first."""
    import tempfile

    created: list[str] = []
    real_mkdtemp = tempfile.mkdtemp

    def _spy(*args, **kwargs):
        d = real_mkdtemp(*args, **kwargs)
        created.append(d)
        return d

    monkeypatch.setattr(f"{MODULE}.tempfile.mkdtemp", _spy)
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: True)

    def _fake_upscale(src, dst, *args, **kwargs):
        Path(dst).write_bytes(b"fake-upscaled")
        return dst

    monkeypatch.setattr(f"{MODULE}.upscale_image", _fake_upscale)

    sentinel = "SENTINEL-FILE-NOT-FOUND-7f3c9a2b"

    def _fake_ken_burns_file_not_found(image_path, duration, out_path, **kwargs):
        Path(out_path).write_bytes(b"partial-mp4-bytes-from-a-missing-binary")
        raise FileNotFoundError(sentinel)

    monkeypatch.setattr(f"{MODULE}.ken_burns_clip", _fake_ken_burns_file_not_found)

    source = tmp_path / "source.png"
    original_bytes = b"original-png-bytes-must-survive"
    source.write_bytes(original_bytes)
    out_path = tmp_path / "out.mp4"

    with pytest.raises(FileNotFoundError) as exc_info:
        build_upscaled_ken_burns_clip(_scene(), source, out_path)

    # Propagates as the ORIGINAL exception type/message — never re-wrapped
    # into KenBurnsUpscalePipelineError, never substituted with an
    # unrelated/misleading domain-error string.
    assert str(exc_info.value) == sentinel
    assert not out_path.exists()
    assert len(created) == 1
    assert not Path(created[0]).exists()
    assert source.read_bytes() == original_bytes


def test_ken_burns_unicode_decode_error_propagates_unwrapped_and_still_cleans_up(tmp_path, monkeypatch):
    """Same guarantee as the FileNotFoundError case, for a different
    non-ProviderError exception type (e.g. subprocess.run(..., text=True)
    failing to decode malformed stderr after ffmpeg already wrote part of
    out_path)."""
    import tempfile

    created: list[str] = []
    real_mkdtemp = tempfile.mkdtemp

    def _spy(*args, **kwargs):
        d = real_mkdtemp(*args, **kwargs)
        created.append(d)
        return d

    monkeypatch.setattr(f"{MODULE}.tempfile.mkdtemp", _spy)
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: True)

    def _fake_upscale(src, dst, *args, **kwargs):
        Path(dst).write_bytes(b"fake-upscaled")
        return dst

    monkeypatch.setattr(f"{MODULE}.upscale_image", _fake_upscale)

    sentinel = "SENTINEL-UNICODE-DECODE-9d4e1c"
    decode_error = UnicodeDecodeError("utf-8", b"\xff\xfe", 0, 1, sentinel)

    def _fake_ken_burns_unicode_decode_error(image_path, duration, out_path, **kwargs):
        Path(out_path).write_bytes(b"partial-mp4-bytes-from-malformed-stderr")
        raise decode_error

    monkeypatch.setattr(f"{MODULE}.ken_burns_clip", _fake_ken_burns_unicode_decode_error)

    source = tmp_path / "source.png"
    original_bytes = b"original-png-bytes-must-survive"
    source.write_bytes(original_bytes)
    out_path = tmp_path / "out.mp4"

    with pytest.raises(UnicodeDecodeError) as exc_info:
        build_upscaled_ken_burns_clip(_scene(), source, out_path)

    assert exc_info.value is decode_error
    assert sentinel in str(exc_info.value)
    assert not out_path.exists()
    assert len(created) == 1
    assert not Path(created[0]).exists()
    assert source.read_bytes() == original_bytes


def test_pre_existing_output_rejected_before_any_upstream_call_even_if_that_call_would_fail(tmp_path, monkeypatch):
    """The existing --output already exists rejection must fire before
    upscale_image()/ken_burns_clip() are ever reached — proved here with
    fakes that would themselves raise if called, not just a generic
    AssertionError trap."""
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: True)

    def _upscale_that_would_raise(src, dst, *args, **kwargs):
        raise AssertionError("upscale_image must not be called when --output already exists")

    def _ken_burns_that_would_raise(image_path, duration, out_path, **kwargs):
        raise AssertionError("ken_burns_clip must not be called when --output already exists")

    monkeypatch.setattr(f"{MODULE}.upscale_image", _upscale_that_would_raise)
    monkeypatch.setattr(f"{MODULE}.ken_burns_clip", _ken_burns_that_would_raise)

    source = tmp_path / "source.png"
    source.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"
    out_path.write_bytes(b"pre-existing content nobody asked to touch")
    bytes_before = out_path.read_bytes()

    with pytest.raises(KenBurnsUpscalePipelineError, match="already exists"):
        build_upscaled_ken_burns_clip(_scene(), source, out_path)

    assert out_path.read_bytes() == bytes_before


# ---------------------------------------------------------------------
# rejections that must happen before ANY mocked call
# ---------------------------------------------------------------------


def test_manual_flow_is_rejected_before_any_transformation_call(tmp_path, monkeypatch):
    _forbid_transformations(monkeypatch)
    source = tmp_path / "source.png"
    source.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(KenBurnsUpscalePipelineError, match="manual_flow"):
        build_upscaled_ken_burns_clip(_scene(motion_mode="manual_flow"), source, out_path)

    assert not out_path.exists()


@pytest.mark.parametrize(
    "duration,expected_message",
    [
        (None, "scene has no measured_audio_duration_seconds"),
        (True, "scene measured_audio_duration_seconds is not a valid number"),
        ("5.0", "scene measured_audio_duration_seconds is not a valid number"),
        (float("nan"), "scene measured_audio_duration_seconds is not finite"),
        (float("inf"), "scene measured_audio_duration_seconds is not finite"),
        (0.0, "scene measured_audio_duration_seconds must be positive"),
        (-1.0, "scene measured_audio_duration_seconds must be positive"),
    ],
)
def test_invalid_duration_is_rejected_before_any_transformation_call(
    tmp_path, monkeypatch, duration, expected_message
):
    _forbid_transformations(monkeypatch)
    source = tmp_path / "source.png"
    source.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(KenBurnsUpscalePipelineError) as exc_info:
        build_upscaled_ken_burns_clip(_scene(duration=duration), source, out_path)

    assert str(exc_info.value) == expected_message
    assert not out_path.exists()


def test_existing_output_is_rejected_before_temp_or_output_creation_or_any_call(tmp_path, monkeypatch):
    _forbid_transformations(monkeypatch)
    source = tmp_path / "source.png"
    source.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"
    out_path.write_bytes(b"pre-existing content nobody asked to touch")

    with pytest.raises(KenBurnsUpscalePipelineError, match="already exists"):
        build_upscaled_ken_burns_clip(_scene(), source, out_path)

    assert out_path.read_bytes() == b"pre-existing content nobody asked to touch"


def test_unavailable_upscaler_is_rejected_before_temp_or_output_creation_or_any_call(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: False)

    def _boom(*args, **kwargs):
        raise AssertionError("must not be called when the upscaler is unavailable")

    monkeypatch.setattr(f"{MODULE}.upscale_image", _boom)
    monkeypatch.setattr(f"{MODULE}.ken_burns_clip", _boom)

    source = tmp_path / "source.png"
    source.write_bytes(b"x")
    out_path = tmp_path / "out.mp4"

    with pytest.raises(KenBurnsUpscalePipelineError, match="unavailable"):
        build_upscaled_ken_burns_clip(_scene(), source, out_path)

    assert not out_path.exists()
