"""Tests for src/core/scene_audio_generation.py — self-hosted, local Kokoro
TTS generation. KokoroProvider is always mocked at the module level
(src.core.scene_audio_generation.KokoroProvider) — never constructed for
real, never any real synthesis or network call anywhere in this file. No
manifest/SQLite/artifact code or dependency exists anywhere in the module
under test (asserted structurally below)."""
from __future__ import annotations

import ast
import inspect
import wave
from pathlib import Path

import pytest

import src.core.scene_audio_generation as scene_audio_generation
from src.core.cost_guard import PaidApprovalRequiredError
from src.core.scene_audio_generation import SceneAudioGenerationError, build_scene_audio
from src.models.scene import ScenePlanItem
from src.providers.base import ProviderError

MODULE = "src.core.scene_audio_generation"


def _scene(narration_text: str = "A quiet moment of reflection.") -> ScenePlanItem:
    return ScenePlanItem(
        scene_id="scene-01",
        sequence=1,
        narration_text=narration_text,
        scene_type="narration",
        narrative_beat="setup",
        visual_brief="A visual brief.",
        motion_mode="static",
        approval_state="approved",
    )


def _write_real_wav(path: Path, duration_seconds: float = 0.5, sample_rate: int = 24000) -> None:
    """A real, minimal, ffprobe-decodable silent PCM WAV — no Kokoro, no
    network, stdlib `wave` only (same technique already used by
    tests/test_ffmpeg_render.py for its own audio fixtures)."""
    num_frames = int(duration_seconds * sample_rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_frames)


class _WritesValidWavProvider:
    def __init__(self):
        pass

    def synthesize(self, text, out_path, voice="af_bella", speed=0.92, sentence_pause=0.45, clause_pause=0.18):
        _write_real_wav(out_path)
        return out_path


class _WritesGarbageProvider:
    def __init__(self):
        pass

    def synthesize(self, text, out_path, voice="af_bella", speed=0.92, sentence_pause=0.45, clause_pause=0.18):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"not a real wav, just arbitrary bytes")
        return out_path


class _WritesZeroLengthWavProvider:
    def __init__(self):
        pass

    def synthesize(self, text, out_path, voice="af_bella", speed=0.92, sentence_pause=0.45, clause_pause=0.18):
        _write_real_wav(out_path, duration_seconds=0.0)
        return out_path


class _RaisesProviderErrorOnCall:
    def __init__(self):
        pass

    def synthesize(self, text, out_path, voice="af_bella", speed=0.92, sentence_pause=0.45, clause_pause=0.18):
        raise ProviderError("simulated synthesis failure with internal detail: /some/local/path")


class _RaisesOSErrorOnCall:
    def __init__(self):
        pass

    def synthesize(self, text, out_path, voice="af_bella", speed=0.92, sentence_pause=0.45, clause_pause=0.18):
        raise OSError("simulated disk-full failure")


def _explode_if_constructed(*args, **kwargs):
    raise AssertionError("KokoroProvider must never be constructed once a preceding check has already failed")


# ---------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------


def test_happy_path_writes_real_audio_and_returns_output_path(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.KokoroProvider", _WritesValidWavProvider)
    monkeypatch.setattr(f"{MODULE}.require_paid_approval", lambda *a, **k: None)

    output = tmp_path / "out.wav"
    proposals_path = tmp_path / "proposals.json"

    result = build_scene_audio(_scene(), output, proposals_path)

    assert result == output
    assert output.exists()


def test_all_options_flow_through_unchanged(tmp_path, monkeypatch):
    captured = {}

    class _CapturesArgsProvider:
        def __init__(self):
            pass

        def synthesize(self, text, out_path, voice, speed, sentence_pause, clause_pause):
            captured["text"] = text
            captured["voice"] = voice
            captured["speed"] = speed
            captured["sentence_pause"] = sentence_pause
            captured["clause_pause"] = clause_pause
            _write_real_wav(out_path)
            return out_path

    monkeypatch.setattr(f"{MODULE}.KokoroProvider", _CapturesArgsProvider)
    monkeypatch.setattr(f"{MODULE}.require_paid_approval", lambda *a, **k: None)

    output = tmp_path / "out.wav"
    proposals_path = tmp_path / "proposals.json"
    scene = _scene("  Narration with padding.  ")

    build_scene_audio(
        scene, output, proposals_path,
        voice="af_sky", speed=1.1, sentence_pause=0.3, clause_pause=0.1,
    )

    assert captured["text"] == "Narration with padding."
    assert captured["voice"] == "af_sky"
    assert captured["speed"] == 1.1
    assert captured["sentence_pause"] == 0.3
    assert captured["clause_pause"] == 0.1


# ---------------------------------------------------------------------
# narration validation — before any provider construction
# ---------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["   ", "\n\t \n"])
def test_blank_narration_rejects_before_provider(tmp_path, monkeypatch, raw):
    """ScenePlanItem.narration_text itself enforces min_length=1, so a
    literal empty string can never reach this function via a real scene —
    only whitespace-only text (which passes that length check but strips
    to empty) can exercise this specific validation path."""
    monkeypatch.setattr(f"{MODULE}.KokoroProvider", _explode_if_constructed)
    monkeypatch.setattr(f"{MODULE}.require_paid_approval", lambda *a, **k: None)

    output = tmp_path / "out.wav"
    proposals_path = tmp_path / "proposals.json"

    with pytest.raises(SceneAudioGenerationError, match="scene narration is empty"):
        build_scene_audio(_scene(raw), output, proposals_path)


# ---------------------------------------------------------------------
# output validation — before any provider construction
# ---------------------------------------------------------------------


def test_output_parent_missing_rejects_before_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.KokoroProvider", _explode_if_constructed)
    monkeypatch.setattr(f"{MODULE}.require_paid_approval", lambda *a, **k: None)

    output = tmp_path / "no-such-dir" / "out.wav"
    proposals_path = tmp_path / "proposals.json"

    with pytest.raises(SceneAudioGenerationError, match="--output parent directory does not exist"):
        build_scene_audio(_scene(), output, proposals_path)


def test_output_parent_not_a_directory_rejects_before_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.KokoroProvider", _explode_if_constructed)
    monkeypatch.setattr(f"{MODULE}.require_paid_approval", lambda *a, **k: None)

    parent_is_a_file = tmp_path / "not-a-dir"
    parent_is_a_file.write_bytes(b"x")
    output = parent_is_a_file / "out.wav"
    proposals_path = tmp_path / "proposals.json"

    with pytest.raises(SceneAudioGenerationError, match="--output parent is not a directory"):
        build_scene_audio(_scene(), output, proposals_path)


def test_output_already_exists_rejects_before_provider_and_is_untouched(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.KokoroProvider", _explode_if_constructed)
    monkeypatch.setattr(f"{MODULE}.require_paid_approval", lambda *a, **k: None)

    output = tmp_path / "out.wav"
    original_bytes = b"already here, must not change"
    output.write_bytes(original_bytes)
    proposals_path = tmp_path / "proposals.json"

    with pytest.raises(SceneAudioGenerationError, match="--output already exists"):
        build_scene_audio(_scene(), output, proposals_path)

    assert output.read_bytes() == original_bytes


# ---------------------------------------------------------------------
# paid-approval denial — before any provider construction; propagates
# unchanged, never wrapped as SceneAudioGenerationError
# ---------------------------------------------------------------------


def test_paid_approval_denial_blocks_before_provider_construction_and_propagates(tmp_path, monkeypatch):
    calls = []

    def _fake_require(service_name, proposals_path, *, is_paid):
        calls.append((service_name, is_paid))
        raise PaidApprovalRequiredError("denied")

    monkeypatch.setattr(f"{MODULE}.require_paid_approval", _fake_require)
    monkeypatch.setattr(f"{MODULE}.KokoroProvider", _explode_if_constructed)

    output = tmp_path / "out.wav"
    proposals_path = tmp_path / "proposals.json"

    with pytest.raises(PaidApprovalRequiredError):
        build_scene_audio(_scene(), output, proposals_path)

    assert calls == [("kokoro", False)]
    assert not output.exists()


# ---------------------------------------------------------------------
# provider failures — wrapped, sanitized, no output left behind
# ---------------------------------------------------------------------


def test_provider_construction_failure_is_sanitized_and_leaves_no_output(tmp_path, monkeypatch):
    def _explode_provider_error(*a, **k):
        raise ProviderError("Kokoro model files missing — expected /some/local/path/kokoro-v1.0.onnx")

    monkeypatch.setattr(f"{MODULE}.KokoroProvider", _explode_provider_error)
    monkeypatch.setattr(f"{MODULE}.require_paid_approval", lambda *a, **k: None)

    output = tmp_path / "out.wav"
    proposals_path = tmp_path / "proposals.json"

    with pytest.raises(SceneAudioGenerationError) as excinfo:
        build_scene_audio(_scene(), output, proposals_path)

    assert str(excinfo.value) == "Kokoro TTS synthesis failed"
    assert "/some/local/path" not in str(excinfo.value)
    assert not output.exists()


def test_provider_error_on_synthesize_is_sanitized_and_leaves_no_output(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.KokoroProvider", _RaisesProviderErrorOnCall)
    monkeypatch.setattr(f"{MODULE}.require_paid_approval", lambda *a, **k: None)

    output = tmp_path / "out.wav"
    proposals_path = tmp_path / "proposals.json"

    with pytest.raises(SceneAudioGenerationError) as excinfo:
        build_scene_audio(_scene(), output, proposals_path)

    assert str(excinfo.value) == "Kokoro TTS synthesis failed"
    assert "/some/local/path" not in str(excinfo.value)
    assert not output.exists()


def test_raw_oserror_on_synthesize_is_sanitized_and_leaves_no_output(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.KokoroProvider", _RaisesOSErrorOnCall)
    monkeypatch.setattr(f"{MODULE}.require_paid_approval", lambda *a, **k: None)

    output = tmp_path / "out.wav"
    proposals_path = tmp_path / "proposals.json"

    with pytest.raises(SceneAudioGenerationError) as excinfo:
        build_scene_audio(_scene(), output, proposals_path)

    assert str(excinfo.value) == "Kokoro TTS synthesis failed"
    assert "disk-full" not in str(excinfo.value)
    assert not output.exists()


# ---------------------------------------------------------------------
# invalid generated output — deleted, error raised
# ---------------------------------------------------------------------


def test_non_audio_bytes_rejected_and_deleted(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.KokoroProvider", _WritesGarbageProvider)
    monkeypatch.setattr(f"{MODULE}.require_paid_approval", lambda *a, **k: None)

    output = tmp_path / "out.wav"
    proposals_path = tmp_path / "proposals.json"

    with pytest.raises(SceneAudioGenerationError, match="generated audio could not be decoded"):
        build_scene_audio(_scene(), output, proposals_path)

    assert not output.exists()


def test_zero_duration_audio_rejected_and_deleted(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.KokoroProvider", _WritesZeroLengthWavProvider)
    monkeypatch.setattr(f"{MODULE}.require_paid_approval", lambda *a, **k: None)

    output = tmp_path / "out.wav"
    proposals_path = tmp_path / "proposals.json"

    with pytest.raises(SceneAudioGenerationError, match="generated audio (could not be decoded|has an invalid duration)"):
        build_scene_audio(_scene(), output, proposals_path)

    assert not output.exists()


def test_provider_producing_no_file_rejects(tmp_path, monkeypatch):
    class _WritesNothingProvider:
        def __init__(self):
            pass

        def synthesize(self, text, out_path, voice="af_bella", speed=0.92, sentence_pause=0.45, clause_pause=0.18):
            pass  # deliberately never writes out_path

    monkeypatch.setattr(f"{MODULE}.KokoroProvider", _WritesNothingProvider)
    monkeypatch.setattr(f"{MODULE}.require_paid_approval", lambda *a, **k: None)

    output = tmp_path / "out.wav"
    proposals_path = tmp_path / "proposals.json"

    with pytest.raises(SceneAudioGenerationError, match="provider did not produce an output file"):
        build_scene_audio(_scene(), output, proposals_path)

    assert not output.exists()


def test_provider_writing_a_directory_at_output_path_rejects(tmp_path, monkeypatch):
    class _WritesDirectoryProvider:
        def __init__(self):
            pass

        def synthesize(self, text, out_path, voice="af_bella", speed=0.92, sentence_pause=0.45, clause_pause=0.18):
            out_path.mkdir(parents=True)

    monkeypatch.setattr(f"{MODULE}.KokoroProvider", _WritesDirectoryProvider)
    monkeypatch.setattr(f"{MODULE}.require_paid_approval", lambda *a, **k: None)

    output = tmp_path / "out.wav"
    proposals_path = tmp_path / "proposals.json"

    with pytest.raises(SceneAudioGenerationError, match="provider output path is not a regular file"):
        build_scene_audio(_scene(), output, proposals_path)


# ---------------------------------------------------------------------
# a genuinely unexpected exception still triggers cleanup and still
# propagates unchanged, never masked
# ---------------------------------------------------------------------


def test_unexpected_exception_type_still_cleans_up_and_propagates(tmp_path, monkeypatch):
    class _WritesThenRaisesUnexpectedError:
        def __init__(self):
            pass

        def synthesize(self, text, out_path, voice="af_bella", speed=0.92, sentence_pause=0.45, clause_pause=0.18):
            _write_real_wav(out_path)
            raise RuntimeError("totally unexpected bug")

    monkeypatch.setattr(f"{MODULE}.KokoroProvider", _WritesThenRaisesUnexpectedError)
    monkeypatch.setattr(f"{MODULE}.require_paid_approval", lambda *a, **k: None)

    output = tmp_path / "out.wav"
    proposals_path = tmp_path / "proposals.json"

    with pytest.raises(RuntimeError, match="totally unexpected bug"):
        build_scene_audio(_scene(), output, proposals_path)

    assert not output.exists()


# ---------------------------------------------------------------------
# no manifest/SQLite/artifact code anywhere in this module — proven via
# source inspection, not just trust
# ---------------------------------------------------------------------


def test_module_has_no_manifest_database_or_artifact_imports():
    source = inspect.getsource(scene_audio_generation)
    tree = ast.parse(source)
    import_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]

    prohibited_prefixes = ("src.database", "src.core.manifest_store", "src.core.manifest_builder")
    offending = [
        node
        for node in import_nodes
        if node.module is not None and node.module.startswith(prohibited_prefixes)
    ]

    assert offending == [], (
        "src/core/scene_audio_generation.py must not import any manifest/database/artifact module; "
        f"found: {[ast.dump(n) for n in offending]}"
    )
