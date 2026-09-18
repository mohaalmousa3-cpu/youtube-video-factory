"""Tests for src/core/overlay_derivation.py — the pure, local deterministic
text-overlay derivation module. No SQLite, no filesystem, no ffprobe/FFmpeg,
no provider/network call anywhere in this file; every ArtifactRecord and
manifest is constructed directly in memory, reusing
src.core.scene_timing_finalizer.finalize_scene_timing() (already tested) to
build realistic finalized-timing fixtures rather than hand-crafting them."""
from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from src.core.manifest_builder import build_video_manifest
from src.core.overlay_derivation import (
    EmptyDerivedOverlayTextError,
    InvalidMeasuredDurationError,
    MissingMeasuredDurationError,
    OverlayDerivationSummary,
    RenderArtifactValidationError,
    TimelineMismatchError,
    _duration_tolerance_seconds,
    derive_text_overlays,
    summarize_derivation,
)
from src.core.scene_timing_finalizer import finalize_scene_timing
from src.models.artifact import ArtifactRecord
from src.models.manifest import VideoManifest
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


def _scene_plan_dict(
    scene_ids: tuple[str, ...] = ("scene-01", "scene-02"),
    narration_by_scene: dict[str, str] | None = None,
    overlays_by_scene: dict[str, list[dict]] | None = None,
) -> dict:
    narration_by_scene = narration_by_scene or {}
    overlays_by_scene = overlays_by_scene or {}
    return dict(
        scenes=tuple(
            dict(
                scene_id=scene_id,
                sequence=i,
                narration_text=narration_by_scene.get(scene_id, f"Narration for {scene_id}."),
                scene_type="narration",
                narrative_beat="setup",
                visual_brief=f"Visual brief for {scene_id}.",
                motion_mode="static",
                approval_state="approved",
                text_overlays=tuple(overlays_by_scene.get(scene_id, ())),
            )
            for i, scene_id in enumerate(scene_ids, start=1)
        ),
        role_outfits=(),
    )


def _manifest(
    scene_ids: tuple[str, ...] = ("scene-01", "scene-02"),
    narration_by_scene: dict[str, str] | None = None,
    overlays_by_scene: dict[str, list[dict]] | None = None,
) -> VideoManifest:
    return build_video_manifest(
        _story_input_dict(),
        _scene_plan_dict(scene_ids, narration_by_scene, overlays_by_scene),
        get_channel_policy(),
        created_at=FIXED_NOW,
    )


def _audio_artifact(project_id: str, scene_id: str, duration: float) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_id=f"audio-{scene_id}",
        project_id=project_id,
        kind="audio",
        scene_id=scene_id,
        relative_path=f"audio/{scene_id}.wav",
        byte_size=1024,
        sha256_checksum="0" * 64,
        created_at=FIXED_NOW,
        metadata={"duration_seconds": duration, "source": "external"},
    )


def _finalized_manifest(
    scene_ids: tuple[str, ...] = ("scene-01", "scene-02"),
    durations: tuple[float, ...] = (5.0, 7.0),
    narration_by_scene: dict[str, str] | None = None,
    overlays_by_scene: dict[str, list[dict]] | None = None,
) -> VideoManifest:
    manifest = _manifest(scene_ids, narration_by_scene, overlays_by_scene)
    artifacts = [_audio_artifact(manifest.project_id, sid, d) for sid, d in zip(scene_ids, durations)]
    return finalize_scene_timing(manifest, artifacts)


def _render_artifact(project_id: str, duration: float, *, artifact_id: str = "render-final") -> ArtifactRecord:
    return ArtifactRecord(
        artifact_id=artifact_id,
        project_id=project_id,
        kind="render",
        scene_id=None,
        relative_path="render/final.mp4",
        byte_size=2048,
        sha256_checksum="1" * 64,
        created_at=FIXED_NOW,
        metadata={"duration_seconds": duration, "source": "final-video-assembly-v1"},
    )


def _set_measured_duration(manifest: VideoManifest, scene_id: str, value) -> VideoManifest:
    """Bypass normal pydantic validation (model_copy() never re-runs
    validators) to inject an otherwise-unreachable-via-construction raw
    value into one scene's measured_audio_duration_seconds — same
    technique tests/test_text_overlay_render.py already uses for its own
    defensive-check tests."""
    new_scenes = []
    for scene in manifest.scene_plan.scenes:
        if scene.scene_id == scene_id:
            new_artifacts = scene.artifacts.model_copy(update={"measured_audio_duration_seconds": value})
            new_scenes.append(scene.model_copy(update={"artifacts": new_artifacts}))
        else:
            new_scenes.append(scene)
    new_scene_plan = manifest.scene_plan.model_copy(update={"scenes": tuple(new_scenes)})
    return manifest.model_copy(update={"scene_plan": new_scene_plan})


def _valid_overlay_dict(**overrides) -> dict:
    data = dict(
        text="A pre-existing overlay",
        position="top",
        style_id="top_label",
        viewer_facing_language="English",
        deterministic=True,
        start_seconds=0.0,
        end_seconds=1.0,
    )
    data.update(overrides)
    return data


# ---------------------------------------------------------------------
# 1-5: happy path, single/multi scene derivation
# ---------------------------------------------------------------------


def test_single_scene_no_overlays_receives_one_derived_overlay():
    manifest = _finalized_manifest(("scene-01",), (5.0,))
    render_artifact = _render_artifact(manifest.project_id, 5.0)

    result = derive_text_overlays(manifest, render_artifact)

    assert len(result.scene_plan.scenes[0].text_overlays) == 1


def test_multiple_scenes_receive_overlays_in_manifest_order():
    manifest = _finalized_manifest(("scene-01", "scene-02", "scene-03"), (2.0, 3.0, 4.0))
    render_artifact = _render_artifact(manifest.project_id, 9.0)

    result = derive_text_overlays(manifest, render_artifact)

    for scene in result.scene_plan.scenes:
        assert len(scene.text_overlays) == 1
    assert [s.scene_id for s in result.scene_plan.scenes] == ["scene-01", "scene-02", "scene-03"]


def test_derived_style_is_exactly_lower_third_primary_lower_third():
    manifest = _finalized_manifest(("scene-01",), (5.0,))
    render_artifact = _render_artifact(manifest.project_id, 5.0)

    result = derive_text_overlays(manifest, render_artifact)
    overlay = result.scene_plan.scenes[0].text_overlays[0]

    assert overlay.style_id == "lower_third_primary"
    assert overlay.position == "lower_third"


def test_derived_overlay_is_deterministic_and_english():
    manifest = _finalized_manifest(("scene-01",), (5.0,))
    render_artifact = _render_artifact(manifest.project_id, 5.0)

    result = derive_text_overlays(manifest, render_artifact)
    overlay = result.scene_plan.scenes[0].text_overlays[0]

    assert overlay.deterministic is True
    assert overlay.viewer_facing_language == "English"


def test_derived_start_end_use_cumulative_measured_durations():
    manifest = _finalized_manifest(("scene-01", "scene-02"), (5.0, 7.0))
    render_artifact = _render_artifact(manifest.project_id, 12.0)

    result = derive_text_overlays(manifest, render_artifact)
    first, second = result.scene_plan.scenes[0].text_overlays[0], result.scene_plan.scenes[1].text_overlays[0]

    assert first.start_seconds == 0.0
    assert first.end_seconds == 5.0
    assert second.start_seconds == 5.0
    assert second.end_seconds == 12.0


# ---------------------------------------------------------------------
# 6-8: explicit overlay preservation and idempotency
# ---------------------------------------------------------------------


def test_existing_overlay_tuple_preserved_exactly():
    manifest = _finalized_manifest(
        ("scene-01", "scene-02"),
        (5.0, 7.0),
        overlays_by_scene={"scene-01": [_valid_overlay_dict(text="Keep me")]},
    )
    render_artifact = _render_artifact(manifest.project_id, 12.0)
    before = manifest.scene_plan.scenes[0].text_overlays

    result = derive_text_overlays(manifest, render_artifact)

    assert result.scene_plan.scenes[0].text_overlays == before
    assert result.scene_plan.scenes[0].text_overlays[0].text == "Keep me"
    # scene-02 had none, so it gets exactly one derived overlay.
    assert len(result.scene_plan.scenes[1].text_overlays) == 1


def test_existing_overlay_scene_still_advances_cumulative_timeline():
    manifest = _finalized_manifest(
        ("scene-01", "scene-02"),
        (5.0, 7.0),
        overlays_by_scene={"scene-01": [_valid_overlay_dict(text="Keep me")]},
    )
    render_artifact = _render_artifact(manifest.project_id, 12.0)

    result = derive_text_overlays(manifest, render_artifact)
    derived = result.scene_plan.scenes[1].text_overlays[0]

    # scene-01's own 5.0s duration must still count toward scene-02's offset
    # even though scene-01's own overlay was preserved, not derived.
    assert derived.start_seconds == 5.0
    assert derived.end_seconds == 12.0


def test_idempotent_second_call_adds_no_second_overlay():
    manifest = _finalized_manifest(("scene-01", "scene-02"), (5.0, 7.0))
    render_artifact = _render_artifact(manifest.project_id, 12.0)

    once = derive_text_overlays(manifest, render_artifact)
    twice = derive_text_overlays(once, render_artifact)

    assert twice == once
    for scene in twice.scene_plan.scenes:
        assert len(scene.text_overlays) == 1


# ---------------------------------------------------------------------
# 9-12: identity preservation and round-trip validation
# ---------------------------------------------------------------------


def test_project_id_preserved_exactly():
    manifest = _finalized_manifest(("scene-01",), (5.0,))
    render_artifact = _render_artifact(manifest.project_id, 5.0)

    result = derive_text_overlays(manifest, render_artifact)

    assert result.project_id == manifest.project_id


def test_source_fingerprint_preserved_exactly():
    manifest = _finalized_manifest(("scene-01",), (5.0,))
    render_artifact = _render_artifact(manifest.project_id, 5.0)

    result = derive_text_overlays(manifest, render_artifact)

    assert result.source_fingerprint == manifest.source_fingerprint


def test_input_manifest_never_mutated():
    manifest = _finalized_manifest(("scene-01",), (5.0,))
    render_artifact = _render_artifact(manifest.project_id, 5.0)
    before = manifest.model_dump(mode="json")

    derive_text_overlays(manifest, render_artifact)

    assert manifest.model_dump(mode="json") == before
    assert manifest.scene_plan.scenes[0].text_overlays == ()


def test_result_round_trips_through_video_manifest_validation():
    manifest = _finalized_manifest(("scene-01",), (5.0,))
    render_artifact = _render_artifact(manifest.project_id, 5.0)

    result = derive_text_overlays(manifest, render_artifact)
    round_tripped = VideoManifest.model_validate(result.model_dump(mode="json"))

    assert round_tripped == result


# ---------------------------------------------------------------------
# 13-17: deterministic text extraction
# ---------------------------------------------------------------------


def test_narration_at_or_under_70_chars_unchanged_no_ellipsis():
    text = "Have you ever wondered why being left out stings so much?"
    assert len(text) <= 70
    manifest = _finalized_manifest(("scene-01",), (5.0,), narration_by_scene={"scene-01": text})
    render_artifact = _render_artifact(manifest.project_id, 5.0)

    result = derive_text_overlays(manifest, render_artifact)

    assert result.scene_plan.scenes[0].text_overlays[0].text == text
    assert "..." not in result.scene_plan.scenes[0].text_overlays[0].text


def test_narration_over_70_chars_truncates_at_last_whitespace():
    text = "This is a long narration sentence that definitely exceeds seventy characters in total length"
    assert len(text) > 70
    manifest = _finalized_manifest(("scene-01",), (5.0,), narration_by_scene={"scene-01": text})
    render_artifact = _render_artifact(manifest.project_id, 5.0)

    result = derive_text_overlays(manifest, render_artifact)
    derived = result.scene_plan.scenes[0].text_overlays[0].text

    prefix = text[:70]
    last_space = prefix.rfind(" ")
    expected = prefix[:last_space].strip() + "..."
    assert derived == expected
    assert derived.endswith("...")


def test_narration_over_70_chars_no_whitespace_truncates_at_70():
    text = "Supercalifragilisticexpialidocious" * 3  # 105 chars, zero whitespace
    assert len(text) > 70 and " " not in text[:70]
    manifest = _finalized_manifest(("scene-01",), (5.0,), narration_by_scene={"scene-01": text})
    render_artifact = _render_artifact(manifest.project_id, 5.0)

    result = derive_text_overlays(manifest, render_artifact)
    derived = result.scene_plan.scenes[0].text_overlays[0].text

    assert derived == text[:70] + "..."


def test_derived_text_length_always_at_most_80():
    long_text = "word " * 40  # 200 chars
    manifest = _finalized_manifest(("scene-01",), (5.0,), narration_by_scene={"scene-01": long_text})
    render_artifact = _render_artifact(manifest.project_id, 5.0)

    result = derive_text_overlays(manifest, render_artifact)

    assert len(result.scene_plan.scenes[0].text_overlays[0].text) <= 80


def test_leading_trailing_whitespace_trimmed():
    text = "   Padded narration text.   "
    manifest = _finalized_manifest(("scene-01",), (5.0,), narration_by_scene={"scene-01": text})
    render_artifact = _render_artifact(manifest.project_id, 5.0)

    result = derive_text_overlays(manifest, render_artifact)

    assert result.scene_plan.scenes[0].text_overlays[0].text == "Padded narration text."


def test_whitespace_only_narration_raises_empty_derived_overlay_text_error():
    manifest = _finalized_manifest(("scene-01",), (5.0,), narration_by_scene={"scene-01": "   "})
    render_artifact = _render_artifact(manifest.project_id, 5.0)

    with pytest.raises(EmptyDerivedOverlayTextError, match="scene-01"):
        derive_text_overlays(manifest, render_artifact)


# ---------------------------------------------------------------------
# 19-24: measured scene duration validation
# ---------------------------------------------------------------------


def test_missing_measured_duration_raises():
    manifest = _manifest(("scene-01",))  # never finalized — still None
    render_artifact = _render_artifact(manifest.project_id, 5.0)

    with pytest.raises(MissingMeasuredDurationError, match="scene-01"):
        derive_text_overlays(manifest, render_artifact)


@pytest.mark.parametrize("bad_value", [0, 0.0, -1, -3.5])
def test_zero_or_negative_measured_duration_rejected(bad_value):
    manifest = _finalized_manifest(("scene-01",), (5.0,))
    manifest = _set_measured_duration(manifest, "scene-01", bad_value)
    render_artifact = _render_artifact(manifest.project_id, 5.0)

    with pytest.raises(InvalidMeasuredDurationError, match="non-positive"):
        derive_text_overlays(manifest, render_artifact)


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_non_finite_measured_duration_rejected(bad_value):
    manifest = _finalized_manifest(("scene-01",), (5.0,))
    manifest = _set_measured_duration(manifest, "scene-01", bad_value)
    render_artifact = _render_artifact(manifest.project_id, 5.0)

    with pytest.raises(InvalidMeasuredDurationError, match="non-finite"):
        derive_text_overlays(manifest, render_artifact)


def test_bool_measured_duration_rejected():
    manifest = _finalized_manifest(("scene-01",), (5.0,))
    manifest = _set_measured_duration(manifest, "scene-01", True)
    render_artifact = _render_artifact(manifest.project_id, 5.0)

    with pytest.raises(InvalidMeasuredDurationError, match="non-numeric"):
        derive_text_overlays(manifest, render_artifact)


# ---------------------------------------------------------------------
# 25-30: render artifact validation
# ---------------------------------------------------------------------


def test_missing_render_duration_metadata_rejected():
    manifest = _finalized_manifest(("scene-01",), (5.0,))
    render_artifact = ArtifactRecord(
        artifact_id="render-final",
        project_id=manifest.project_id,
        kind="render",
        scene_id=None,
        relative_path="render/final.mp4",
        byte_size=2048,
        sha256_checksum="1" * 64,
        created_at=FIXED_NOW,
        metadata={"source": "final-video-assembly-v1"},  # no duration_seconds
    )

    with pytest.raises(RenderArtifactValidationError, match="duration_seconds"):
        derive_text_overlays(manifest, render_artifact)


def test_bool_render_duration_rejected():
    manifest = _finalized_manifest(("scene-01",), (5.0,))
    render_artifact = _render_artifact(manifest.project_id, 5.0)
    render_artifact = render_artifact.model_copy(update={"metadata": {"duration_seconds": True}})

    with pytest.raises(RenderArtifactValidationError, match="non-numeric"):
        derive_text_overlays(manifest, render_artifact)


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, 0, -1])
def test_invalid_render_duration_rejected(bad_value):
    manifest = _finalized_manifest(("scene-01",), (5.0,))
    render_artifact = _render_artifact(manifest.project_id, bad_value)

    with pytest.raises(RenderArtifactValidationError):
        derive_text_overlays(manifest, render_artifact)


def test_wrong_render_artifact_kind_rejected():
    manifest = _finalized_manifest(("scene-01",), (5.0,))
    wrong_kind = ArtifactRecord(
        artifact_id="qc-final",
        project_id=manifest.project_id,
        kind="qc_report",
        scene_id=None,
        relative_path="qc/report.json",
        byte_size=10,
        sha256_checksum="2" * 64,
        created_at=FIXED_NOW,
        metadata={"duration_seconds": 5.0},
    )

    with pytest.raises(RenderArtifactValidationError, match="kind"):
        derive_text_overlays(manifest, wrong_kind)


def test_render_artifact_with_non_null_scene_id_rejected_defensively():
    """Unreachable via normal construction — ArtifactRecord's own
    _scene_association_matches_kind validator already forbids a
    project-level kind ('render') from carrying a scene_id. Proven
    directly against derive_text_overlays()'s own defensive re-check via
    model_copy() (which never re-runs validators), same technique used
    elsewhere in this codebase for this exact class of unreachable case."""
    manifest = _finalized_manifest(("scene-01",), (5.0,))
    render_artifact = _render_artifact(manifest.project_id, 5.0)
    render_artifact = render_artifact.model_copy(update={"scene_id": "scene-01"})

    with pytest.raises(RenderArtifactValidationError, match="scene_id"):
        derive_text_overlays(manifest, render_artifact)


def test_render_artifact_project_mismatch_rejected():
    manifest = _finalized_manifest(("scene-01",), (5.0,))
    render_artifact = _render_artifact("proj-some-other-project", 5.0)

    with pytest.raises(RenderArtifactValidationError, match="project_id"):
        derive_text_overlays(manifest, render_artifact)


# ---------------------------------------------------------------------
# 31-33: timeline mismatch tolerance
# ---------------------------------------------------------------------


def test_timeline_mismatch_beyond_tolerance_rejected():
    manifest = _finalized_manifest(("scene-01", "scene-02"), (5.0, 7.0))  # total 12.0
    render_artifact = _render_artifact(manifest.project_id, 50.0)  # wildly different

    with pytest.raises(TimelineMismatchError):
        derive_text_overlays(manifest, render_artifact)


def test_timeline_mismatch_within_tolerance_accepted():
    manifest = _finalized_manifest(("scene-01", "scene-02"), (5.0, 7.0))  # total 12.0
    tolerance = _duration_tolerance_seconds(2)
    render_artifact = _render_artifact(manifest.project_id, 12.0 + tolerance * 0.5)

    result = derive_text_overlays(manifest, render_artifact)

    assert len(result.scene_plan.scenes[0].text_overlays) == 1


def test_timeline_mismatch_exact_boundary_accepted():
    manifest = _finalized_manifest(("scene-01", "scene-02"), (5.0, 7.0))  # total 12.0
    tolerance = _duration_tolerance_seconds(2)
    render_artifact = _render_artifact(manifest.project_id, 12.0 + tolerance)  # exactly at boundary

    result = derive_text_overlays(manifest, render_artifact)

    assert len(result.scene_plan.scenes[0].text_overlays) == 1


# ---------------------------------------------------------------------
# 34-36: renderer-compatible timing, non-alteration, no I/O
# ---------------------------------------------------------------------


def test_derived_timing_is_renderer_compatible():
    manifest = _finalized_manifest(("scene-01", "scene-02"), (5.0, 7.0))
    render_artifact = _render_artifact(manifest.project_id, 12.0)

    result = derive_text_overlays(manifest, render_artifact)

    for scene in result.scene_plan.scenes:
        overlay = scene.text_overlays[0]
        assert overlay.start_seconds is not None
        assert overlay.end_seconds is not None
        assert overlay.start_seconds >= 0
        assert overlay.end_seconds > overlay.start_seconds


def test_scenes_with_explicit_overlays_are_not_altered():
    original_overlay = _valid_overlay_dict(text="Do not touch me", start_seconds=10.0, end_seconds=20.0)
    manifest = _finalized_manifest(
        ("scene-01",), (5.0,), overlays_by_scene={"scene-01": [original_overlay]}
    )
    render_artifact = _render_artifact(manifest.project_id, 5.0)

    result = derive_text_overlays(manifest, render_artifact)
    result_overlay = result.scene_plan.scenes[0].text_overlays[0]

    assert result_overlay.text == "Do not touch me"
    assert result_overlay.start_seconds == 10.0
    assert result_overlay.end_seconds == 20.0
    assert result_overlay.position == "top"
    assert result_overlay.style_id == "top_label"


def test_core_function_has_no_filesystem_db_or_subprocess_activity(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("derive_text_overlays() must never call this")

    monkeypatch.setattr("sqlite3.connect", _boom)
    monkeypatch.setattr("subprocess.run", _boom)

    manifest = _finalized_manifest(("scene-01",), (5.0,))
    render_artifact = _render_artifact(manifest.project_id, 5.0)

    result = derive_text_overlays(manifest, render_artifact)

    assert len(result.scene_plan.scenes[0].text_overlays) == 1


# ---------------------------------------------------------------------
# summarize_derivation()
# ---------------------------------------------------------------------


def test_summarize_derivation_counts_new_and_preserved():
    manifest = _finalized_manifest(
        ("scene-01", "scene-02"),
        (5.0, 7.0),
        overlays_by_scene={"scene-01": [_valid_overlay_dict()]},
    )
    render_artifact = _render_artifact(manifest.project_id, 12.0)

    result = derive_text_overlays(manifest, render_artifact)
    summary = summarize_derivation(manifest, result)

    assert summary == OverlayDerivationSummary(
        scenes_with_new_overlays=1,
        scenes_with_existing_overlays=1,
        derived_overlay_count=1,
        preserved_overlay_count=1,
    )
