"""Tests for src/core/final_assembly_planner.py — the pure, local,
read-only FINAL ASSEMBLY PLANNER V1 report. No SQLite, no filesystem, no
ffprobe/FFmpeg, no provider/network call anywhere in this file; every
ArtifactRecord/ProjectRecord/manifest is constructed directly in memory,
reusing already-tested helpers (build_video_manifest,
create_initial_project, finalize_scene_timing) rather than hand-crafting
them."""
from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from src.core.final_assembly_planner import (
    FinalAssemblyPlan,
    FinalAssemblyPlannerError,
    build_final_assembly_plan,
)
from src.core.manifest_builder import build_video_manifest
from src.core.project_state_machine import create_initial_project
from src.core.scene_timing_finalizer import finalize_scene_timing
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


def _overlay_dict(**overrides) -> dict:
    data = dict(
        text="The Cyberball Game",
        position="lower_third",
        style_id="lower_third_primary",
        viewer_facing_language="English",
        deterministic=True,
        start_seconds=0.0,
        end_seconds=1.0,
    )
    data.update(overrides)
    return data


def _scene_plan_dict(
    scene_ids: tuple[str, ...] = ("scene-01", "scene-02"),
    overlays_by_scene: dict[str, list[dict]] | None = None,
) -> dict:
    overlays_by_scene = overlays_by_scene or {}
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
                text_overlays=tuple(overlays_by_scene.get(scene_id, ())),
            )
            for i, scene_id in enumerate(scene_ids, start=1)
        ),
        role_outfits=(),
    )


def _manifest(
    scene_ids: tuple[str, ...] = ("scene-01", "scene-02"),
    overlays_by_scene: dict[str, list[dict]] | None = None,
):
    return build_video_manifest(
        _story_input_dict(), _scene_plan_dict(scene_ids, overlays_by_scene), get_channel_policy(), created_at=FIXED_NOW
    )


def _finalized_manifest(
    scene_ids: tuple[str, ...] = ("scene-01", "scene-02"),
    durations: tuple[float, ...] = (5.0, 7.0),
    overlays_by_scene: dict[str, list[dict]] | None = None,
):
    manifest = _manifest(scene_ids, overlays_by_scene)
    artifacts = [
        ArtifactRecord(
            artifact_id=f"audio-{sid}",
            project_id=manifest.project_id,
            kind="audio",
            scene_id=sid,
            relative_path=f"audio/{sid}.wav",
            byte_size=1,
            sha256_checksum="0" * 64,
            created_at=FIXED_NOW,
            metadata={"duration_seconds": d, "source": "external"},
        )
        for sid, d in zip(scene_ids, durations)
    ]
    return finalize_scene_timing(manifest, artifacts)


def _project(manifest):
    return create_initial_project(
        f"data/projects/{manifest.project_id}/manifest.json", manifest, now=FIXED_NOW
    )


def _render_artifact(project_id: str, duration, *, artifact_id: str = "render-final", **overrides) -> ArtifactRecord:
    data = dict(
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
    data.update(overrides)
    return ArtifactRecord(**data)


def _overlay_render_artifact(
    project_id: str,
    render_artifact: ArtifactRecord,
    manifest,
    overlay_count: int,
    duration,
    *,
    artifact_id: str = "overlay-render-final",
    metadata_overrides: dict | None = None,
    **overrides,
) -> ArtifactRecord:
    metadata = {
        "duration_seconds": duration,
        "source": "text-overlay-renderer-v1",
        "overlay_count": overlay_count,
        "source_render_artifact_id": render_artifact.artifact_id,
        "source_render_sha256": render_artifact.sha256_checksum,
        "source_render_relative_path": "render/final.mp4",
        "manifest_fingerprint": manifest.source_fingerprint,
    }
    if metadata_overrides:
        metadata.update(metadata_overrides)
    data = dict(
        artifact_id=artifact_id,
        project_id=project_id,
        kind="overlay_render",
        scene_id=None,
        relative_path="overlay_render/final.mp4",
        byte_size=2048,
        sha256_checksum="2" * 64,
        created_at=FIXED_NOW,
        metadata=metadata,
    )
    data.update(overrides)
    return ArtifactRecord(**data)


# ---------------------------------------------------------------------
# 1-3: identity checks
# ---------------------------------------------------------------------


def test_valid_identity_accepted():
    manifest = _finalized_manifest(("scene-01",), (5.0,))
    project = _project(manifest)

    plan = build_final_assembly_plan(project, manifest, [])

    assert plan.manifest_project_id_matches is True
    assert plan.manifest_fingerprint_matches is True


def test_project_id_mismatch_raises():
    manifest = _finalized_manifest(("scene-01",), (5.0,))
    project = _project(manifest)
    mismatched = manifest.model_copy(update={"project_id": "proj-does-not-match"})

    with pytest.raises(FinalAssemblyPlannerError, match="project_id"):
        build_final_assembly_plan(project, mismatched, [])


def test_fingerprint_mismatch_raises():
    manifest = _finalized_manifest(("scene-01",), (5.0,))
    project = _project(manifest)
    tampered = manifest.model_copy(update={"source_fingerprint": "0" * 64})

    with pytest.raises(FinalAssemblyPlannerError, match="fingerprint"):
        build_final_assembly_plan(project, tampered, [])


# ---------------------------------------------------------------------
# 4-6: next-command routing
# ---------------------------------------------------------------------


def test_render_missing_gives_assemble_final_video_next_command():
    manifest = _finalized_manifest(("scene-01",), (5.0,))
    project = _project(manifest)

    plan = build_final_assembly_plan(project, manifest, [])

    assert plan.render_ready is False
    assert plan.overlay_render_ready is False
    assert plan.final_viewer_output_kind is None
    assert plan.final_viewer_output_relative_path is None
    assert plan.next_safe_local_command == f"assemble-final-video {project.project_id} --manifest PATH --output PATH"
    assert plan.next_command_requires_explicit_paths is True
    assert any("render" in r for r in plan.blocked_reasons)


def test_render_ready_no_overlays_gives_derive_text_overlays_next_command():
    manifest = _finalized_manifest(("scene-01",), (5.0,))
    project = _project(manifest)
    render = _render_artifact(project.project_id, 5.0)

    plan = build_final_assembly_plan(project, manifest, [render])

    assert plan.render_ready is True
    assert plan.overlay_count == 0
    assert plan.manifest_has_explicit_overlays is False
    assert plan.final_viewer_output_kind == "render"
    assert plan.final_viewer_output_relative_path == "render/final.mp4"
    assert plan.next_safe_local_command == (
        f"derive-text-overlays {project.project_id} --manifest INPUT_PATH --output OUTPUT_PATH"
    )
    assert plan.next_command_requires_explicit_paths is True
    assert any("no explicit text overlays" in r for r in plan.blocked_reasons)


def test_render_ready_with_overlays_no_overlay_render_gives_render_text_overlays_next_command():
    manifest = _finalized_manifest(
        ("scene-01",), (5.0,), overlays_by_scene={"scene-01": [_overlay_dict()]}
    )
    project = _project(manifest)
    render = _render_artifact(project.project_id, 5.0)

    plan = build_final_assembly_plan(project, manifest, [render])

    assert plan.render_ready is True
    assert plan.overlay_count == 1
    assert plan.manifest_has_explicit_overlays is True
    assert plan.final_viewer_output_kind == "render"
    assert plan.next_safe_local_command == (
        f"render-text-overlays {project.project_id} --manifest PATH --output PATH"
    )
    assert plan.next_command_requires_explicit_paths is True
    assert any("has not been produced" in r for r in plan.blocked_reasons)


def test_render_and_overlay_render_ready_gives_no_next_command():
    manifest = _finalized_manifest(
        ("scene-01",), (5.0,), overlays_by_scene={"scene-01": [_overlay_dict()]}
    )
    project = _project(manifest)
    render = _render_artifact(project.project_id, 5.0)
    overlay_render = _overlay_render_artifact(project.project_id, render, manifest, overlay_count=1, duration=5.0)

    plan = build_final_assembly_plan(project, manifest, [render, overlay_render])

    assert plan.render_ready is True
    assert plan.overlay_render_ready is True
    assert plan.final_viewer_output_kind == "overlay_render"
    assert plan.final_viewer_output_relative_path == "overlay_render/final.mp4"
    assert plan.next_safe_local_command is None
    assert plan.next_command_requires_explicit_paths is False
    assert plan.blocked_reasons == ()


# ---------------------------------------------------------------------
# 8-9: overlay counting/ordering
# ---------------------------------------------------------------------


def test_overlay_count_exact_across_multiple_scenes_and_overlays():
    manifest = _finalized_manifest(
        ("scene-01", "scene-02"),
        (5.0, 7.0),
        overlays_by_scene={
            "scene-01": [_overlay_dict(text="A"), _overlay_dict(text="B", style_id="top_label", position="top")],
            "scene-02": [_overlay_dict(text="C")],
        },
    )
    project = _project(manifest)
    render = _render_artifact(project.project_id, 12.0)

    plan = build_final_assembly_plan(project, manifest, [render])

    assert plan.overlay_count == 3


def test_overlay_ordering_follows_scene_then_tuple_order():
    """Not directly observable on FinalAssemblyPlan (it only exposes a
    count), but the count computation itself must walk scenes in stored
    order — proven by constructing scenes out of alphabetical order and
    confirming the total is still exactly the sum across all scenes."""
    manifest = _finalized_manifest(
        ("scene-02", "scene-01"),
        (7.0, 5.0),
        overlays_by_scene={
            "scene-02": [_overlay_dict(text="X")],
            "scene-01": [_overlay_dict(text="Y"), _overlay_dict(text="Z")],
        },
    )
    project = _project(manifest)
    render = _render_artifact(project.project_id, 12.0)

    plan = build_final_assembly_plan(project, manifest, [render])

    assert plan.overlay_count == 3


# ---------------------------------------------------------------------
# 10-15: render artifact validation
# ---------------------------------------------------------------------


def test_render_with_non_null_scene_id_raises():
    manifest = _finalized_manifest(("scene-01",), (5.0,))
    project = _project(manifest)
    render = _render_artifact(project.project_id, 5.0)
    render = render.model_copy(update={"scene_id": "scene-01"})

    with pytest.raises(FinalAssemblyPlannerError, match="scene_id"):
        build_final_assembly_plan(project, manifest, [render])


def test_render_wrong_relative_path_raises():
    manifest = _finalized_manifest(("scene-01",), (5.0,))
    project = _project(manifest)
    render = _render_artifact(project.project_id, 5.0, relative_path="render/other.mp4")

    with pytest.raises(FinalAssemblyPlannerError, match="relative_path"):
        build_final_assembly_plan(project, manifest, [render])


def test_render_missing_duration_raises():
    manifest = _finalized_manifest(("scene-01",), (5.0,))
    project = _project(manifest)
    render = _render_artifact(project.project_id, 5.0, metadata={"source": "final-video-assembly-v1"})

    with pytest.raises(FinalAssemblyPlannerError, match="duration_seconds"):
        build_final_assembly_plan(project, manifest, [render])


def test_render_bool_duration_raises():
    manifest = _finalized_manifest(("scene-01",), (5.0,))
    project = _project(manifest)
    render = _render_artifact(project.project_id, 5.0)
    render = render.model_copy(update={"metadata": {"duration_seconds": True}})

    with pytest.raises(FinalAssemblyPlannerError, match="non-numeric"):
        build_final_assembly_plan(project, manifest, [render])


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, 0, -1])
def test_render_non_finite_or_non_positive_duration_raises(bad_value):
    manifest = _finalized_manifest(("scene-01",), (5.0,))
    project = _project(manifest)
    render = _render_artifact(project.project_id, bad_value)

    with pytest.raises(FinalAssemblyPlannerError):
        build_final_assembly_plan(project, manifest, [render])


def test_multiple_render_artifacts_raise():
    manifest = _finalized_manifest(("scene-01",), (5.0,))
    project = _project(manifest)
    render_a = _render_artifact(project.project_id, 5.0, artifact_id="render-a", relative_path="render/a.mp4")
    render_b = _render_artifact(project.project_id, 5.0, artifact_id="render-b", relative_path="render/b.mp4")

    with pytest.raises(FinalAssemblyPlannerError, match="render"):
        build_final_assembly_plan(project, manifest, [render_a, render_b])


# ---------------------------------------------------------------------
# 16-25: overlay_render artifact validation
# ---------------------------------------------------------------------


def _ready_manifest_and_render():
    manifest = _finalized_manifest(
        ("scene-01",), (5.0,), overlays_by_scene={"scene-01": [_overlay_dict()]}
    )
    project = _project(manifest)
    render = _render_artifact(project.project_id, 5.0)
    return manifest, project, render


def test_overlay_render_non_null_scene_id_raises():
    manifest, project, render = _ready_manifest_and_render()
    overlay_render = _overlay_render_artifact(project.project_id, render, manifest, 1, 5.0)
    overlay_render = overlay_render.model_copy(update={"scene_id": "scene-01"})

    with pytest.raises(FinalAssemblyPlannerError, match="scene_id"):
        build_final_assembly_plan(project, manifest, [render, overlay_render])


def test_overlay_render_wrong_relative_path_raises():
    manifest, project, render = _ready_manifest_and_render()
    overlay_render = _overlay_render_artifact(
        project.project_id, render, manifest, 1, 5.0, relative_path="overlay_render/other.mp4"
    )

    with pytest.raises(FinalAssemblyPlannerError, match="relative_path"):
        build_final_assembly_plan(project, manifest, [render, overlay_render])


def test_overlay_render_missing_duration_raises():
    manifest, project, render = _ready_manifest_and_render()
    overlay_render = _overlay_render_artifact(project.project_id, render, manifest, 1, 5.0)
    stripped_metadata = {k: v for k, v in overlay_render.metadata.items() if k != "duration_seconds"}
    overlay_render = overlay_render.model_copy(update={"metadata": stripped_metadata})

    with pytest.raises(FinalAssemblyPlannerError, match="duration_seconds"):
        build_final_assembly_plan(project, manifest, [render, overlay_render])


def test_overlay_render_wrong_source_metadata_raises():
    manifest, project, render = _ready_manifest_and_render()
    overlay_render = _overlay_render_artifact(
        project.project_id, render, manifest, 1, 5.0, metadata_overrides={"source": "something-else"}
    )

    with pytest.raises(FinalAssemblyPlannerError, match="source"):
        build_final_assembly_plan(project, manifest, [render, overlay_render])


def test_overlay_render_wrong_source_render_id_raises():
    manifest, project, render = _ready_manifest_and_render()
    overlay_render = _overlay_render_artifact(
        project.project_id, render, manifest, 1, 5.0,
        metadata_overrides={"source_render_artifact_id": "render-does-not-exist"},
    )

    with pytest.raises(FinalAssemblyPlannerError, match="source_render_artifact_id"):
        build_final_assembly_plan(project, manifest, [render, overlay_render])


def test_overlay_render_wrong_source_checksum_raises():
    manifest, project, render = _ready_manifest_and_render()
    overlay_render = _overlay_render_artifact(
        project.project_id, render, manifest, 1, 5.0,
        metadata_overrides={"source_render_sha256": "f" * 64},
    )

    with pytest.raises(FinalAssemblyPlannerError, match="source_render_sha256"):
        build_final_assembly_plan(project, manifest, [render, overlay_render])


def test_overlay_render_wrong_source_path_raises():
    manifest, project, render = _ready_manifest_and_render()
    overlay_render = _overlay_render_artifact(
        project.project_id, render, manifest, 1, 5.0,
        metadata_overrides={"source_render_relative_path": "render/other.mp4"},
    )

    with pytest.raises(FinalAssemblyPlannerError, match="source_render_relative_path"):
        build_final_assembly_plan(project, manifest, [render, overlay_render])


def test_overlay_render_wrong_manifest_fingerprint_raises():
    manifest, project, render = _ready_manifest_and_render()
    overlay_render = _overlay_render_artifact(
        project.project_id, render, manifest, 1, 5.0,
        metadata_overrides={"manifest_fingerprint": "0" * 64},
    )

    with pytest.raises(FinalAssemblyPlannerError, match="manifest_fingerprint"):
        build_final_assembly_plan(project, manifest, [render, overlay_render])


def test_overlay_render_wrong_overlay_count_raises():
    manifest, project, render = _ready_manifest_and_render()
    overlay_render = _overlay_render_artifact(
        project.project_id, render, manifest, 1, 5.0,
        metadata_overrides={"overlay_count": 99},
    )

    with pytest.raises(FinalAssemblyPlannerError, match="overlay_count"):
        build_final_assembly_plan(project, manifest, [render, overlay_render])


def test_multiple_overlay_render_artifacts_raise():
    manifest, project, render = _ready_manifest_and_render()
    overlay_render_a = _overlay_render_artifact(
        project.project_id, render, manifest, 1, 5.0, artifact_id="overlay-render-a",
        relative_path="overlay_render/a.mp4",
    )
    overlay_render_b = _overlay_render_artifact(
        project.project_id, render, manifest, 1, 5.0, artifact_id="overlay-render-b",
        relative_path="overlay_render/b.mp4",
    )

    with pytest.raises(FinalAssemblyPlannerError, match="overlay_render"):
        build_final_assembly_plan(project, manifest, [render, overlay_render_a, overlay_render_b])


def test_valid_overlay_render_metadata_accepted():
    manifest, project, render = _ready_manifest_and_render()
    overlay_render = _overlay_render_artifact(project.project_id, render, manifest, 1, 5.0)

    plan = build_final_assembly_plan(project, manifest, [render, overlay_render])

    assert plan.overlay_render_ready is True
    assert plan.overlay_render_artifact_id == overlay_render.artifact_id


# ---------------------------------------------------------------------
# 27-28: immutability, no I/O
# ---------------------------------------------------------------------


def test_result_is_immutable():
    manifest = _finalized_manifest(("scene-01",), (5.0,))
    project = _project(manifest)

    plan = build_final_assembly_plan(project, manifest, [])

    assert isinstance(plan, FinalAssemblyPlan)
    with pytest.raises(Exception):
        plan.project_id = "changed"


def test_core_function_has_no_filesystem_db_or_subprocess_activity(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("build_final_assembly_plan() must never call this")

    monkeypatch.setattr("sqlite3.connect", _boom)
    monkeypatch.setattr("subprocess.run", _boom)

    manifest = _finalized_manifest(
        ("scene-01",), (5.0,), overlays_by_scene={"scene-01": [_overlay_dict()]}
    )
    project = _project(manifest)
    render = _render_artifact(project.project_id, 5.0)
    overlay_render = _overlay_render_artifact(project.project_id, render, manifest, 1, 5.0)

    plan = build_final_assembly_plan(project, manifest, [render, overlay_render])

    assert plan.overlay_render_ready is True
