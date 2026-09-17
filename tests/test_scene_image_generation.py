"""Tests for src/core/scene_image_generation.py — the first real (paid,
network) generation call in this pipeline. QwenImageProvider is always
mocked at the module level (src.core.scene_image_generation.QwenImageProvider)
— never constructed for real, never any real network call anywhere in this
file."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

import src.core.scene_image_generation as scene_image_generation
from src.core.cost_guard import PaidApprovalRequiredError
from src.core.scene_image_generation import SceneImageGenerationError, build_scene_image
from src.models.scene import ScenePlanItem
from src.providers.base import ProviderError

MODULE = "src.core.scene_image_generation"


def _scene(visual_brief: str = "A stick figure waits at a bus stop.") -> ScenePlanItem:
    return ScenePlanItem(
        scene_id="scene-01",
        sequence=1,
        narration_text="Narration text.",
        scene_type="narration",
        narrative_beat="setup",
        visual_brief=visual_brief,
        motion_mode="static",
        approval_state="approved",
    )


def _write_real_png(path: Path, size=(4, 4)) -> None:
    Image.new("RGB", size, color=(10, 20, 30)).save(path, format="PNG")


class _WritesValidPngProvider:
    """Fake QwenImageProvider: generate_with_reference_and_download() writes
    a real, valid PNG to out_path — no network, no base64, no requests."""

    def __init__(self):
        pass

    def generate_with_reference_and_download(self, prompt, reference_image_path, out_path, size="1328*1328"):
        _write_real_png(out_path)
        return out_path


class _WritesGarbageProvider:
    def __init__(self):
        pass

    def generate_with_reference_and_download(self, prompt, reference_image_path, out_path, size="1328*1328"):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"not a real png, just arbitrary bytes")
        return out_path


class _WritesJpegProvider:
    def __init__(self):
        pass

    def generate_with_reference_and_download(self, prompt, reference_image_path, out_path, size="1328*1328"):
        Image.new("RGB", (4, 4)).save(out_path, format="JPEG")
        return out_path


class _RaisesProviderErrorOnCall:
    def __init__(self):
        pass

    def generate_with_reference_and_download(self, prompt, reference_image_path, out_path, size="1328*1328"):
        raise ProviderError("simulated network failure with sensitive detail: sk-secret-123")


def _explode_if_constructed(*args, **kwargs):
    raise AssertionError("QwenImageProvider must never be constructed once a preceding check has already failed")


# ---------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------


def test_happy_path_writes_a_valid_png_and_returns_output_path(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.QwenImageProvider", _WritesValidPngProvider)
    monkeypatch.setattr(f"{MODULE}.require_paid_approval", lambda *a, **k: None)

    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"
    proposals_path = tmp_path / "proposals.json"

    result = build_scene_image(
        _scene(), reference_image, output, proposals_path, include_character_anchor=True
    )

    assert result == output
    assert output.exists()
    with Image.open(output) as img:
        img.load()
        assert img.format == "PNG"


# ---------------------------------------------------------------------
# reference-image validation — before any provider construction
# ---------------------------------------------------------------------


def test_missing_reference_image_rejects_before_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.QwenImageProvider", _explode_if_constructed)
    monkeypatch.setattr(f"{MODULE}.require_paid_approval", lambda *a, **k: None)

    reference_image = tmp_path / "does-not-exist.png"
    output = tmp_path / "out.png"
    proposals_path = tmp_path / "proposals.json"

    with pytest.raises(SceneImageGenerationError, match="--reference-image does not exist"):
        build_scene_image(_scene(), reference_image, output, proposals_path, include_character_anchor=True)


def test_directory_reference_image_rejects_before_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.QwenImageProvider", _explode_if_constructed)
    monkeypatch.setattr(f"{MODULE}.require_paid_approval", lambda *a, **k: None)

    reference_image = tmp_path / "a-directory"
    reference_image.mkdir()
    output = tmp_path / "out.png"
    proposals_path = tmp_path / "proposals.json"

    with pytest.raises(SceneImageGenerationError, match="--reference-image is not a regular file"):
        build_scene_image(_scene(), reference_image, output, proposals_path, include_character_anchor=True)


# ---------------------------------------------------------------------
# output validation — before any provider construction
# ---------------------------------------------------------------------


def test_output_parent_missing_rejects_before_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.QwenImageProvider", _explode_if_constructed)
    monkeypatch.setattr(f"{MODULE}.require_paid_approval", lambda *a, **k: None)

    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "no-such-dir" / "out.png"
    proposals_path = tmp_path / "proposals.json"

    with pytest.raises(SceneImageGenerationError, match="--output parent directory does not exist"):
        build_scene_image(_scene(), reference_image, output, proposals_path, include_character_anchor=True)


def test_output_already_exists_rejects_before_provider_and_is_untouched(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.QwenImageProvider", _explode_if_constructed)
    monkeypatch.setattr(f"{MODULE}.require_paid_approval", lambda *a, **k: None)

    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"
    original_bytes = b"already here, must not change"
    output.write_bytes(original_bytes)
    proposals_path = tmp_path / "proposals.json"

    with pytest.raises(SceneImageGenerationError, match="--output already exists"):
        build_scene_image(_scene(), reference_image, output, proposals_path, include_character_anchor=True)

    assert output.read_bytes() == original_bytes


# ---------------------------------------------------------------------
# paid-approval denial — before any provider construction; propagates
# unchanged, never wrapped as SceneImageGenerationError
# ---------------------------------------------------------------------


def test_paid_approval_denial_blocks_before_provider_construction_and_propagates(tmp_path, monkeypatch):
    calls = []

    def _fake_require(service_name, proposals_path, *, is_paid):
        calls.append((service_name, is_paid))
        raise PaidApprovalRequiredError("denied")

    monkeypatch.setattr(f"{MODULE}.require_paid_approval", _fake_require)
    monkeypatch.setattr(f"{MODULE}.QwenImageProvider", _explode_if_constructed)

    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"
    proposals_path = tmp_path / "proposals.json"

    with pytest.raises(PaidApprovalRequiredError):
        build_scene_image(_scene(), reference_image, output, proposals_path, include_character_anchor=True)

    assert calls == [("qwen-image", True)]
    assert not output.exists()


# ---------------------------------------------------------------------
# provider failure — wrapped, sanitized, no output left behind
# ---------------------------------------------------------------------


def test_provider_error_is_wrapped_and_sanitized_and_leaves_no_output(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.QwenImageProvider", _RaisesProviderErrorOnCall)
    monkeypatch.setattr(f"{MODULE}.require_paid_approval", lambda *a, **k: None)

    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"
    proposals_path = tmp_path / "proposals.json"

    with pytest.raises(SceneImageGenerationError) as excinfo:
        build_scene_image(_scene(), reference_image, output, proposals_path, include_character_anchor=True)

    assert str(excinfo.value) == "Qwen-Image generation failed"
    assert "sk-secret-123" not in str(excinfo.value)
    assert not output.exists()


# ---------------------------------------------------------------------
# invalid generated bytes — output deleted, error raised
# ---------------------------------------------------------------------


def test_non_png_bytes_deletes_output_and_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.QwenImageProvider", _WritesGarbageProvider)
    monkeypatch.setattr(f"{MODULE}.require_paid_approval", lambda *a, **k: None)

    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"
    proposals_path = tmp_path / "proposals.json"

    with pytest.raises(SceneImageGenerationError, match="could not be decoded as an image"):
        build_scene_image(_scene(), reference_image, output, proposals_path, include_character_anchor=True)

    assert not output.exists()


def test_non_png_but_valid_image_format_deletes_output_and_raises(tmp_path, monkeypatch):
    """A genuinely decodable image that is NOT a PNG (e.g. JPEG) must still
    be rejected — decodability alone is not sufficient."""
    monkeypatch.setattr(f"{MODULE}.QwenImageProvider", _WritesJpegProvider)
    monkeypatch.setattr(f"{MODULE}.require_paid_approval", lambda *a, **k: None)

    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"
    proposals_path = tmp_path / "proposals.json"

    with pytest.raises(SceneImageGenerationError, match="generated file is not a PNG"):
        build_scene_image(_scene(), reference_image, output, proposals_path, include_character_anchor=True)

    assert not output.exists()


# ---------------------------------------------------------------------
# a genuinely unexpected (non-ProviderError) exception from the provider
# still triggers cleanup and still propagates unchanged, never masked
# ---------------------------------------------------------------------


def test_unexpected_exception_type_still_cleans_up_and_propagates(tmp_path, monkeypatch):
    class _WritesThenRaisesUnexpectedError:
        def __init__(self):
            pass

        def generate_with_reference_and_download(self, prompt, reference_image_path, out_path, size="1328*1328"):
            _write_real_png(out_path)
            raise RuntimeError("totally unexpected bug")

    monkeypatch.setattr(f"{MODULE}.QwenImageProvider", _WritesThenRaisesUnexpectedError)
    monkeypatch.setattr(f"{MODULE}.require_paid_approval", lambda *a, **k: None)

    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"
    proposals_path = tmp_path / "proposals.json"

    with pytest.raises(RuntimeError, match="totally unexpected bug"):
        build_scene_image(_scene(), reference_image, output, proposals_path, include_character_anchor=True)

    assert not output.exists()


# ---------------------------------------------------------------------
# validation order
# ---------------------------------------------------------------------


def test_reference_image_checked_before_output(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.QwenImageProvider", _explode_if_constructed)
    monkeypatch.setattr(f"{MODULE}.require_paid_approval", lambda *a, **k: None)

    reference_image = tmp_path / "does-not-exist.png"
    output = tmp_path / "no-such-dir" / "out.png"
    proposals_path = tmp_path / "proposals.json"

    with pytest.raises(SceneImageGenerationError, match="--reference-image does not exist"):
        build_scene_image(_scene(), reference_image, output, proposals_path, include_character_anchor=True)


def test_output_checked_before_paid_approval(tmp_path, monkeypatch):
    def _fail_if_called(*a, **k):
        raise AssertionError("require_paid_approval must not be called before output validation passes")

    monkeypatch.setattr(f"{MODULE}.require_paid_approval", _fail_if_called)
    monkeypatch.setattr(f"{MODULE}.QwenImageProvider", _explode_if_constructed)

    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"
    output.write_bytes(b"already here")
    proposals_path = tmp_path / "proposals.json"

    with pytest.raises(SceneImageGenerationError, match="--output already exists"):
        build_scene_image(_scene(), reference_image, output, proposals_path, include_character_anchor=True)


# ---------------------------------------------------------------------
# flags flow through unchanged
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "include_character_anchor,include_color_anchor",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_flags_flow_through_to_prompt_construction(
    tmp_path, monkeypatch, include_character_anchor, include_color_anchor
):
    from src.core.scene_image_prompt import CHARACTER_ANCHOR, COLOR_ANCHOR

    captured_prompt = {}

    class _CapturesPromptProvider:
        def __init__(self):
            pass

        def generate_with_reference_and_download(self, prompt, reference_image_path, out_path, size="1328*1328"):
            captured_prompt["value"] = prompt
            _write_real_png(out_path)
            return out_path

    monkeypatch.setattr(f"{MODULE}.QwenImageProvider", _CapturesPromptProvider)
    monkeypatch.setattr(f"{MODULE}.require_paid_approval", lambda *a, **k: None)

    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"
    proposals_path = tmp_path / "proposals.json"

    build_scene_image(
        _scene(),
        reference_image,
        output,
        proposals_path,
        include_character_anchor=include_character_anchor,
        include_color_anchor=include_color_anchor,
    )

    assert (CHARACTER_ANCHOR in captured_prompt["value"]) is include_character_anchor
    assert (COLOR_ANCHOR in captured_prompt["value"]) is include_color_anchor


# ---------------------------------------------------------------------
# reference image is read-only (never modified by this module itself —
# the provider mock never touches it here either)
# ---------------------------------------------------------------------


def test_reference_image_bytes_unchanged_after_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.QwenImageProvider", _WritesValidPngProvider)
    monkeypatch.setattr(f"{MODULE}.require_paid_approval", lambda *a, **k: None)

    reference_image = tmp_path / "reference.png"
    original_bytes = b"original reference bytes"
    reference_image.write_bytes(original_bytes)
    output = tmp_path / "out.png"
    proposals_path = tmp_path / "proposals.json"

    build_scene_image(_scene(), reference_image, output, proposals_path, include_character_anchor=True)

    assert reference_image.read_bytes() == original_bytes
