"""Tests for src/core/scene_timing_finalizer.py — the pure, local
measured-audio timing bridge. No SQLite, no filesystem, no ffprobe, no
provider/network call anywhere in this file; every ArtifactRecord is
constructed directly in memory."""
from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from src.core.manifest_builder import build_video_manifest
from src.core.scene_timing_finalizer import TimingFinalizationError, finalize_scene_timing
from src.models.artifact import ArtifactRecord
from src.utils.channel_config import get_channel_policy

FIXED_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _story_input_dict() -> dict:
    return dict(
        story_id="why-we-care-what-people-think",
        title="Why We Care So Much What People Think",
        topic="social psychology",
        target_duration_seconds=480.0,
        language="en-US",
        viewer_facing_language="English",
        approval_status="approved",
    )


def _scene_plan_dict(scene_ids: tuple[str, ...] = ("scene-01", "scene-02")) -> dict:
    return dict(
        scenes=tuple(
            dict(
                scene_id=scene_id,
                sequence=i,
                narration_text=f"Narration for {scene_id}.",
                scene_type="narration",
                narrative_beat="setup",
                visual_brief=f"Visual brief for {scene_id}.",
                motion_mode="static",
                approval_state="approved",
            )
            for i, scene_id in enumerate(scene_ids, start=1)
        ),
        role_outfits=(),
    )


def _manifest(scene_ids: tuple[str, ...] = ("scene-01", "scene-02")):
    return build_video_manifest(
        _story_input_dict(), _scene_plan_dict(scene_ids), get_channel_policy(), created_at=FIXED_NOW
    )


def _audio_artifact(project_id: str, scene_id: str, duration, *, artifact_id: str | None = None) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_id=artifact_id or f"audio-{scene_id}",
        project_id=project_id,
        kind="audio",
        scene_id=scene_id,
        relative_path=f"audio/{scene_id}.wav",
        byte_size=1024,
        sha256_checksum="0" * 64,
        created_at=FIXED_NOW,
        metadata={"duration_seconds": duration, "source": "external"},
    )


# ---------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------


def test_happy_path_every_scene_gets_its_exact_registered_duration():
    manifest = _manifest()
    artifacts = [
        _audio_artifact(manifest.project_id, "scene-01", 5.2),
        _audio_artifact(manifest.project_id, "scene-02", 7.75),
    ]

    result = finalize_scene_timing(manifest, artifacts)

    assert result.scene_plan.scenes[0].artifacts.measured_audio_duration_seconds == 5.2
    assert result.scene_plan.scenes[1].artifacts.measured_audio_duration_seconds == 7.75


def test_manifest_level_measured_audio_duration_seconds_stays_none():
    manifest = _manifest()
    assert manifest.measured_audio_duration_seconds is None
    artifacts = [
        _audio_artifact(manifest.project_id, "scene-01", 5.2),
        _audio_artifact(manifest.project_id, "scene-02", 7.75),
    ]

    result = finalize_scene_timing(manifest, artifacts)

    assert result.measured_audio_duration_seconds is None


def test_project_id_and_source_fingerprint_are_preserved_exactly():
    manifest = _manifest()
    artifacts = [
        _audio_artifact(manifest.project_id, "scene-01", 5.2),
        _audio_artifact(manifest.project_id, "scene-02", 7.75),
    ]

    result = finalize_scene_timing(manifest, artifacts)

    assert result.project_id == manifest.project_id
    assert result.source_fingerprint == manifest.source_fingerprint


def test_result_round_trips_through_video_manifest_validation():
    """finalize_scene_timing() itself re-validates via
    VideoManifest.model_validate() before returning — this test confirms
    the RETURNED object is a genuinely valid VideoManifest (not just that
    construction happened to succeed once) by round-tripping it through
    model_dump()/model_validate() again independently."""
    from src.models.manifest import VideoManifest

    manifest = _manifest()
    artifacts = [
        _audio_artifact(manifest.project_id, "scene-01", 5.2),
        _audio_artifact(manifest.project_id, "scene-02", 7.75),
    ]

    result = finalize_scene_timing(manifest, artifacts)
    round_tripped = VideoManifest.model_validate(result.model_dump(mode="json"))

    assert round_tripped == result
    assert round_tripped.scene_plan.scenes[0].artifacts.measured_audio_duration_seconds == 5.2
    assert round_tripped.scene_plan.scenes[1].artifacts.measured_audio_duration_seconds == 7.75


def test_inputs_are_never_mutated():
    manifest = _manifest()
    artifacts = [
        _audio_artifact(manifest.project_id, "scene-01", 5.2),
        _audio_artifact(manifest.project_id, "scene-02", 7.75),
    ]
    before = manifest.model_dump(mode="json")

    finalize_scene_timing(manifest, artifacts)

    assert manifest.model_dump(mode="json") == before
    assert manifest.scene_plan.scenes[0].artifacts.measured_audio_duration_seconds is None


# ---------------------------------------------------------------------
# all-or-nothing: missing / duplicate audio
# ---------------------------------------------------------------------


def test_missing_audio_for_any_scene_rejects_atomically():
    manifest = _manifest()
    artifacts = [_audio_artifact(manifest.project_id, "scene-01", 5.2)]  # scene-02 missing

    with pytest.raises(TimingFinalizationError, match=r"scene-02.*no eligible"):
        finalize_scene_timing(manifest, artifacts)


def test_duplicate_audio_for_one_scene_rejects_atomically():
    manifest = _manifest()
    artifacts = [
        _audio_artifact(manifest.project_id, "scene-01", 5.2, artifact_id="audio-scene-01-a"),
        _audio_artifact(manifest.project_id, "scene-01", 6.0, artifact_id="audio-scene-01-b"),
        _audio_artifact(manifest.project_id, "scene-02", 7.75),
    ]

    with pytest.raises(TimingFinalizationError, match=r"scene-01.*2 eligible"):
        finalize_scene_timing(manifest, artifacts)


def test_empty_artifact_list_rejects_every_scene():
    manifest = _manifest()

    with pytest.raises(TimingFinalizationError, match=r"no eligible"):
        finalize_scene_timing(manifest, [])


# ---------------------------------------------------------------------
# ineligible records: wrong kind, wrong project, wrong scene
# ---------------------------------------------------------------------


def test_wrong_kind_artifact_rejects():
    manifest = _manifest()
    not_audio = ArtifactRecord(
        artifact_id="visual-scene-01",
        project_id=manifest.project_id,
        kind="visual",
        scene_id="scene-01",
        relative_path="visual/scene-01.png",
        byte_size=1024,
        sha256_checksum="0" * 64,
        created_at=FIXED_NOW,
        metadata={},
    )
    artifacts = [not_audio, _audio_artifact(manifest.project_id, "scene-02", 7.75)]

    with pytest.raises(TimingFinalizationError, match=r"not an 'audio' artifact"):
        finalize_scene_timing(manifest, artifacts)


def test_wrong_project_artifact_rejects():
    manifest = _manifest()
    artifacts = [
        _audio_artifact("proj-some-other-project", "scene-01", 5.2),
        _audio_artifact(manifest.project_id, "scene-02", 7.75),
    ]

    with pytest.raises(TimingFinalizationError, match=r"belongs to project"):
        finalize_scene_timing(manifest, artifacts)


def test_wrong_scene_id_artifact_is_simply_not_a_match():
    """An audio artifact whose scene_id doesn't match any manifest scene
    doesn't itself raise a distinct error — it just fails to satisfy
    whichever real scene needed a match, surfacing as that scene's own
    'no eligible' rejection (proving scene_id matching is exact, not
    fuzzy)."""
    manifest = _manifest(scene_ids=("scene-01",))
    artifacts = [_audio_artifact(manifest.project_id, "scene-99", 5.2)]  # no such scene in the manifest

    with pytest.raises(TimingFinalizationError, match=r"scene-01.*no eligible"):
        finalize_scene_timing(manifest, artifacts)


# ---------------------------------------------------------------------
# duration_seconds validation: missing / non-numeric / non-finite / non-positive
# ---------------------------------------------------------------------


def test_missing_duration_seconds_key_rejects():
    manifest = _manifest(scene_ids=("scene-01",))
    artifact = ArtifactRecord(
        artifact_id="audio-scene-01",
        project_id=manifest.project_id,
        kind="audio",
        scene_id="scene-01",
        relative_path="audio/scene-01.wav",
        byte_size=1024,
        sha256_checksum="0" * 64,
        created_at=FIXED_NOW,
        metadata={"source": "external"},  # no duration_seconds key at all
    )

    with pytest.raises(TimingFinalizationError, match=r"missing a 'duration_seconds'"):
        finalize_scene_timing(manifest, [artifact])


@pytest.mark.parametrize("bad_value", ["not-a-number", True, False])
def test_non_numeric_duration_rejects(bad_value):
    manifest = _manifest(scene_ids=("scene-01",))
    artifact = _audio_artifact(manifest.project_id, "scene-01", bad_value)

    with pytest.raises(TimingFinalizationError, match=r"non-numeric duration_seconds"):
        finalize_scene_timing(manifest, [artifact])


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_non_finite_duration_rejects(bad_value):
    manifest = _manifest(scene_ids=("scene-01",))
    artifact = _audio_artifact(manifest.project_id, "scene-01", bad_value)

    with pytest.raises(TimingFinalizationError, match=r"non-finite duration_seconds"):
        finalize_scene_timing(manifest, [artifact])


@pytest.mark.parametrize("bad_value", [0, 0.0, -1, -3.5])
def test_non_positive_duration_rejects(bad_value):
    manifest = _manifest(scene_ids=("scene-01",))
    artifact = _audio_artifact(manifest.project_id, "scene-01", bad_value)

    with pytest.raises(TimingFinalizationError, match=r"non-positive duration_seconds"):
        finalize_scene_timing(manifest, [artifact])


def test_valid_integer_duration_is_accepted():
    """int is an accepted numeric type (only bool, a subtype of int, is
    excluded) — this is a legitimate, valid duration, not an edge case
    that should be rejected."""
    manifest = _manifest(scene_ids=("scene-01",))
    artifact = _audio_artifact(manifest.project_id, "scene-01", 6)

    result = finalize_scene_timing(manifest, [artifact])

    assert result.scene_plan.scenes[0].artifacts.measured_audio_duration_seconds == 6.0


# ---------------------------------------------------------------------
# no I/O of any kind
# ---------------------------------------------------------------------


def test_core_function_never_touches_ffprobe_or_sqlite(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("finalize_scene_timing() must never call this")

    monkeypatch.setattr("src.render.ffmpeg_render.get_duration_seconds", _boom, raising=False)
    monkeypatch.setattr("sqlite3.connect", _boom)

    manifest = _manifest()
    artifacts = [
        _audio_artifact(manifest.project_id, "scene-01", 5.2),
        _audio_artifact(manifest.project_id, "scene-02", 7.75),
    ]

    result = finalize_scene_timing(manifest, artifacts)

    assert result.scene_plan.scenes[0].artifacts.measured_audio_duration_seconds == 5.2
