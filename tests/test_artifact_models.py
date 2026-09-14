"""Tests for src/models/artifact.py: ArtifactRecord / ArtifactVerificationResult
validation. No filesystem, no database, no network."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from types import MappingProxyType

import pytest
from pydantic import ValidationError

from src.models.artifact import ArtifactRecord, ArtifactVerificationResult

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
VALID_SHA256 = "a" * 64


def _record(**overrides) -> ArtifactRecord:
    data = dict(
        artifact_id="audio-scene-01",
        project_id="proj-abc123",
        kind="audio",
        scene_id="scene-01",
        relative_path="audio/scene-01.wav",
        byte_size=100,
        sha256_checksum=VALID_SHA256,
        created_at=NOW,
    )
    data.update(overrides)
    return ArtifactRecord(**data)


# ---------------------------------------------------------------------
# One valid record per kind
# ---------------------------------------------------------------------


def test_valid_audio_artifact():
    record = _record(kind="audio", scene_id="scene-01")
    assert record.kind == "audio"
    assert record.scene_id == "scene-01"


def test_valid_visual_artifact():
    record = _record(
        artifact_id="visual-scene-01", kind="visual", scene_id="scene-01",
        relative_path="visuals/scene-01.png",
    )
    assert record.kind == "visual"


def test_valid_animation_artifact():
    record = _record(
        artifact_id="animation-scene-01", kind="animation", scene_id="scene-01",
        relative_path="animation/scene-01.json",
    )
    assert record.kind == "animation"


def test_valid_render_artifact_is_project_level():
    record = _record(
        artifact_id="render-final", kind="render", scene_id=None, relative_path="render/final.mp4",
    )
    assert record.kind == "render"
    assert record.scene_id is None


def test_valid_qc_report_artifact_is_project_level():
    record = _record(
        artifact_id="qc-report", kind="qc_report", scene_id=None, relative_path="qc/report.json",
    )
    assert record.kind == "qc_report"
    assert record.scene_id is None


# ---------------------------------------------------------------------
# Frozen / strict
# ---------------------------------------------------------------------


def test_artifact_record_is_frozen():
    record = _record()
    with pytest.raises(ValidationError):
        record.byte_size = 999


def test_artifact_record_rejects_unknown_field():
    with pytest.raises(ValidationError):
        _record(unexpected_field="nope")


# ---------------------------------------------------------------------
# Scene-association invariant
# ---------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["audio", "visual", "animation"])
def test_scene_level_kind_requires_scene_id(kind):
    with pytest.raises(ValidationError, match="scene-level and requires scene_id"):
        _record(kind=kind, scene_id=None, relative_path=f"{kind}/x")


@pytest.mark.parametrize("kind", ["render", "qc_report"])
def test_project_level_kind_forbids_scene_id(kind):
    with pytest.raises(ValidationError, match="project-level and must not set scene_id"):
        _record(kind=kind, scene_id="scene-01", relative_path=f"{kind}/x")


def test_scene_id_must_match_pattern():
    with pytest.raises(ValidationError):
        _record(scene_id="not-a-valid-scene-id")


# ---------------------------------------------------------------------
# SHA-256 checksum format
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_checksum",
    [
        "not-hex-at-all",
        "a" * 63,  # too short
        "a" * 65,  # too long
        "A" * 64,  # uppercase not allowed
        "",
    ],
)
def test_invalid_sha256_checksum_rejected(bad_checksum):
    with pytest.raises(ValidationError):
        _record(sha256_checksum=bad_checksum)


def test_valid_sha256_checksum_accepted():
    record = _record(sha256_checksum="0123456789abcdef" * 4)
    assert len(record.sha256_checksum) == 64


# ---------------------------------------------------------------------
# byte_size
# ---------------------------------------------------------------------


def test_negative_byte_size_rejected():
    with pytest.raises(ValidationError):
        _record(byte_size=-1)


def test_zero_byte_size_accepted():
    record = _record(byte_size=0)
    assert record.byte_size == 0


# ---------------------------------------------------------------------
# No absolute paths, no traversal
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../escape.wav",
        "audio/../../escape.wav",
        "/abs/escape.wav",
        "C:/abs/escape.wav",
        "C:\\abs\\escape.wav",
        "audio\\scene-01.wav",  # backslash not allowed even without traversal
        "",
        "audio//scene-01.wav",  # empty segment
        "./audio/scene-01.wav",
    ],
)
def test_unsafe_relative_path_rejected(unsafe_path):
    with pytest.raises(ValidationError):
        _record(relative_path=unsafe_path)


def test_normalized_relative_path_accepted():
    record = _record(relative_path="audio/nested/scene-01.wav")
    assert record.relative_path == "audio/nested/scene-01.wav"


# ---------------------------------------------------------------------
# artifact_id
# ---------------------------------------------------------------------


def test_artifact_id_must_be_non_empty():
    with pytest.raises(ValidationError):
        _record(artifact_id="")


# ---------------------------------------------------------------------
# metadata is minimal JSON-safe
# ---------------------------------------------------------------------


def test_metadata_defaults_to_empty_dict():
    record = _record()
    assert record.metadata == {}


def test_metadata_accepts_json_safe_scalars():
    record = _record(metadata={"voice": "kokoro-af", "duration": 4.2, "retried": False, "note": None})
    assert record.metadata["voice"] == "kokoro-af"


def test_metadata_rejects_nested_containers():
    with pytest.raises(ValidationError):
        _record(metadata={"nested": {"not": "allowed"}})


# ---------------------------------------------------------------------
# metadata is deeply immutable (types.MappingProxyType), not just an
# attribute frozen=True can't reassign
# ---------------------------------------------------------------------


def test_metadata_is_a_mapping_proxy():
    record = _record(metadata={"voice": "kokoro-af"})
    assert isinstance(record.metadata, MappingProxyType)


def test_metadata_rejects_item_assignment():
    record = _record(metadata={"voice": "kokoro-af"})
    with pytest.raises((TypeError, AttributeError)):
        record.metadata["voice"] = "different-voice"
    assert record.metadata["voice"] == "kokoro-af"  # unchanged


def test_metadata_rejects_item_deletion():
    record = _record(metadata={"voice": "kokoro-af"})
    with pytest.raises((TypeError, AttributeError)):
        del record.metadata["voice"]
    assert record.metadata["voice"] == "kokoro-af"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m.update({"voice": "different-voice"}),
        lambda m: m.clear(),
        lambda m: m.pop("voice"),
        lambda m: m.setdefault("voice", "different-voice"),
    ],
    ids=["update", "clear", "pop", "setdefault"],
)
def test_metadata_rejects_in_place_mutating_methods(mutate):
    record = _record(metadata={"voice": "kokoro-af"})
    with pytest.raises((TypeError, AttributeError)):
        mutate(record.metadata)
    assert record.metadata == {"voice": "kokoro-af"}  # unchanged


def test_metadata_still_readable_via_normal_mapping_access():
    record = _record(metadata={"voice": "kokoro-af", "duration": 4.2})
    assert record.metadata["voice"] == "kokoro-af"
    assert dict(record.metadata) == {"voice": "kokoro-af", "duration": 4.2}
    assert set(record.metadata.keys()) == {"voice", "duration"}


def test_metadata_json_dump_is_a_plain_json_compatible_object():
    record = _record(metadata={"voice": "kokoro-af", "duration": 4.2, "retried": False, "note": None})
    dumped = record.model_dump(mode="json")
    assert type(dumped["metadata"]) is dict  # not a mappingproxy — plain dict, JSON round-trips
    assert dumped["metadata"] == {
        "voice": "kokoro-af",
        "duration": 4.2,
        "retried": False,
        "note": None,
    }
    # round-trips through the stdlib json module unmodified.
    assert json.loads(json.dumps(dumped)) == dumped


def test_metadata_empty_dict_json_dump_is_plain_dict():
    record = _record()
    dumped = record.model_dump(mode="json")
    assert dumped["metadata"] == {}
    assert type(dumped["metadata"]) is dict


# ---------------------------------------------------------------------
# ArtifactVerificationResult
# ---------------------------------------------------------------------


def _result(**overrides) -> ArtifactVerificationResult:
    data = dict(
        artifact_id="audio-scene-01",
        project_id="proj-abc123",
        kind="audio",
        scene_id="scene-01",
        relative_path="audio/scene-01.wav",
        passed=True,
        reasons=(),
    )
    data.update(overrides)
    return ArtifactVerificationResult(**data)


def test_verification_result_passed_with_no_reasons():
    result = _result(passed=True, reasons=())
    assert result.passed is True


def test_verification_result_failed_requires_reasons():
    with pytest.raises(ValidationError, match="reasons must be non-empty"):
        _result(passed=False, reasons=())


def test_verification_result_passed_forbids_reasons():
    with pytest.raises(ValidationError, match="reasons must be empty"):
        _result(passed=True, reasons=("should not be here",))


def test_verification_result_is_frozen():
    result = _result(passed=False, reasons=("missing file",))
    with pytest.raises(ValidationError):
        result.passed = True
