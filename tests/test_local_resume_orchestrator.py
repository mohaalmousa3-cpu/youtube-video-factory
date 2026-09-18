"""Tests for src/core/local_resume_orchestrator.py — LOCAL RESUME
ORCHESTRATOR V1. Real SQLite (isolated per test via the isolated_db
fixture, same convention as tests/test_cli_plan_final_assembly.py) but
assemble_final_video/derive_text_overlays/render_text_overlays are faked
(monkeypatched) for the decision-level tests below, so no ffmpeg/ffprobe/
provider/network call happens anywhere in this file except the one
doubly-gated real-FFmpeg integration test at the bottom."""
from __future__ import annotations

import hashlib
import inspect
import shutil
import socket
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.core.local_resume_orchestrator import (
    LocalResumeOrchestratorError,
    LocalResumeResult,
    build_final_local,
)
from src.core.final_video_assembly import FinalVideoAssemblyError
from src.core.manifest_builder import build_video_manifest
from src.core.manifest_store import save_manifest
from src.core.overlay_derivation import OverlayDerivationError, derive_text_overlays
from src.core.project_state_machine import create_initial_project, initial_transition_for
from src.core.scene_timing_finalizer import finalize_scene_timing
from src.core.text_overlay_render import TextOverlayRenderError
from src.database.artifact_repository import register_artifact
from src.database.db import get_connection, init_db
from src.database.project_repository import create_project
from src.models.artifact import ArtifactRecord
from src.utils.channel_config import get_channel_policy

FIXED_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
MODULE = "src.core.local_resume_orchestrator"


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    from src.utils import config

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    config.get_settings.cache_clear()
    yield tmp_path
    config.get_settings.cache_clear()


def _story_input_dict(story_id: str = "why-we-care-what-people-think") -> dict:
    return dict(
        story_id=story_id,
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
    scene_ids: tuple[str, ...] = ("scene-01",),
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


def _create_registered_project(
    projects_root: Path,
    scene_ids: tuple[str, ...] = ("scene-01",),
    story_id: str = "why-we-care-what-people-think",
    durations: tuple[float, ...] | None = None,
    overlays_by_scene: dict[str, list[dict]] | None = None,
):
    """Build+register a project whose manifest is already
    finalize-scene-timing'd (every scene's measured_audio_duration_seconds
    populated) and save it at PROJECTS_ROOT/<project_id>/manifest.json —
    this is the file every test passes as --timed-manifest."""
    manifest = build_video_manifest(
        _story_input_dict(story_id), _scene_plan_dict(scene_ids, overlays_by_scene), get_channel_policy(), created_at=FIXED_NOW
    )
    durations = durations or tuple(5.0 for _ in scene_ids)
    audio_artifacts = [
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
    manifest = finalize_scene_timing(manifest, audio_artifacts)

    manifest_path = projects_root / manifest.project_id / "manifest.json"
    save_manifest(manifest, manifest_path)

    project = create_initial_project(manifest_path, manifest, now=FIXED_NOW)
    transition = initial_transition_for(project)

    init_db()
    conn = get_connection()
    try:
        create_project(conn, project, transition)
    finally:
        conn.close()
    return project.project_id, manifest_path.parent, manifest


def _register(record: ArtifactRecord) -> None:
    conn = get_connection()
    try:
        register_artifact(conn, record)
    finally:
        conn.close()


def _render_artifact_record(project_id: str, duration=5.0, *, artifact_id="render-final") -> ArtifactRecord:
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


def _overlay_render_artifact_record(
    project_id: str, render_artifact: ArtifactRecord, manifest, overlay_count: int, duration=5.0, *, artifact_id="overlay-render-final"
) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_id=artifact_id,
        project_id=project_id,
        kind="overlay_render",
        scene_id=None,
        relative_path="overlay_render/final.mp4",
        byte_size=2048,
        sha256_checksum="2" * 64,
        created_at=FIXED_NOW,
        metadata={
            "duration_seconds": duration,
            "source": "text-overlay-renderer-v1",
            "overlay_count": overlay_count,
            "source_render_artifact_id": render_artifact.artifact_id,
            "source_render_sha256": render_artifact.sha256_checksum,
            "source_render_relative_path": "render/final.mp4",
            "manifest_fingerprint": manifest.source_fingerprint,
        },
    )


def _fake_assemble_final_video(*, duration=5.0, calls: list | None = None):
    def _fake(project_id, manifest_path, output_path):
        if calls is not None:
            calls.append((project_id, Path(manifest_path), Path(output_path)))
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-render-bytes")
        _register(_render_artifact_record(project_id, duration))
        return None

    return _fake


def _fake_render_text_overlays(*, calls: list | None = None):
    def _fake(project_id, manifest_path, output_path):
        from src.core.manifest_store import load_manifest as _load

        if calls is not None:
            calls.append((project_id, Path(manifest_path), Path(output_path)))
        manifest = _load(Path(manifest_path))
        overlay_count = sum(len(s.text_overlays) for s in manifest.scene_plan.scenes)
        conn = get_connection()
        try:
            from src.database.artifact_repository import list_artifacts_by_project

            render_artifact = list_artifacts_by_project(conn, project_id, kind="render")[0]
        finally:
            conn.close()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-overlay-bytes")
        _register(_overlay_render_artifact_record(project_id, render_artifact, manifest, overlay_count))
        return None

    return _fake


def _paths(tmp_path, name="run"):
    base = tmp_path / name
    return dict(
        derived_manifest_output_path=base / "derived.json",
        render_output_path=base / "render.mp4",
        overlay_output_path=base / "overlay.mp4",
    )


# ---------------------------------------------------------------------
# 1-3, 14-16: core decision / immutability / no-retry
# ---------------------------------------------------------------------


def test_render_absent_calls_assembly_first(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    calls: list = []
    monkeypatch.setattr(f"{MODULE}.assemble_final_video", _fake_assemble_final_video(calls=calls))
    monkeypatch.setattr(f"{MODULE}.render_text_overlays", _fake_render_text_overlays())

    result = build_final_local(
        project_id=project_id,
        timed_manifest_path=project_dir / "manifest.json",
        **_paths(tmp_path),
    )

    assert len(calls) == 1
    assert result.render_status == "render_created"
    assert result.derived_manifest_status == "derived_manifest_created"
    assert result.overlay_render_status == "overlay_render_created"
    assert result.stopped_at_stage is None


def test_existing_render_skips_assembly(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register(_render_artifact_record(project_id))
    calls: list = []
    monkeypatch.setattr(f"{MODULE}.assemble_final_video", _fake_assemble_final_video(calls=calls))
    monkeypatch.setattr(f"{MODULE}.render_text_overlays", _fake_render_text_overlays())

    result = build_final_local(
        project_id=project_id, timed_manifest_path=project_dir / "manifest.json", **_paths(tmp_path)
    )

    assert calls == []
    assert result.render_status == "render_reused"
    assert result.render_output_path is None


def test_existing_overlay_render_skips_all_stages(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    render = _render_artifact_record(project_id)
    _register(render)
    _register(_overlay_render_artifact_record(project_id, render, manifest, overlay_count=0))
    assemble_calls: list = []
    render_calls: list = []
    monkeypatch.setattr(f"{MODULE}.assemble_final_video", _fake_assemble_final_video(calls=assemble_calls))
    monkeypatch.setattr(f"{MODULE}.render_text_overlays", _fake_render_text_overlays(calls=render_calls))
    monkeypatch.setattr(f"{MODULE}.derive_text_overlays", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run")))
    monkeypatch.setattr(f"{MODULE}.save_manifest", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run")))

    result = build_final_local(
        project_id=project_id, timed_manifest_path=project_dir / "manifest.json", **_paths(tmp_path)
    )

    assert assemble_calls == []
    assert render_calls == []
    assert result.render_status == "render_reused"
    assert result.derived_manifest_status == "not_needed"
    assert result.overlay_render_status == "overlay_render_reused"
    assert result.stopped_at_stage is None
    assert not (tmp_path / "run" / "render.mp4").exists()
    assert not (tmp_path / "run" / "overlay.mp4").exists()
    assert not (tmp_path / "run" / "derived.json").exists()


def test_render_exists_with_explicit_overlays_calls_renderer_only(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(
        tmp_path / "projects", overlays_by_scene={"scene-01": [_overlay_dict()]}
    )
    _register(_render_artifact_record(project_id))
    assemble_calls: list = []
    render_calls: list = []
    monkeypatch.setattr(f"{MODULE}.assemble_final_video", _fake_assemble_final_video(calls=assemble_calls))
    monkeypatch.setattr(f"{MODULE}.render_text_overlays", _fake_render_text_overlays(calls=render_calls))
    monkeypatch.setattr(f"{MODULE}.derive_text_overlays", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run")))

    result = build_final_local(
        project_id=project_id, timed_manifest_path=project_dir / "manifest.json", **_paths(tmp_path)
    )

    assert assemble_calls == []
    assert len(render_calls) == 1
    assert render_calls[0][1] == (project_dir / "manifest.json").resolve()
    assert result.derived_manifest_status == "not_needed"
    assert result.derived_manifest_path == str((project_dir / "manifest.json").resolve())
    assert result.overlay_render_status == "overlay_render_created"


def test_render_exists_no_overlays_derives_then_renders(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register(_render_artifact_record(project_id))
    render_calls: list = []
    monkeypatch.setattr(f"{MODULE}.render_text_overlays", _fake_render_text_overlays(calls=render_calls))

    paths = _paths(tmp_path)
    result = build_final_local(project_id=project_id, timed_manifest_path=project_dir / "manifest.json", **paths)

    assert result.derived_manifest_status == "derived_manifest_created"
    assert Path(result.derived_manifest_path).exists()
    assert len(render_calls) == 1
    assert render_calls[0][1] == paths["derived_manifest_output_path"].resolve()
    assert result.overlay_render_status == "overlay_render_created"


def test_result_is_immutable(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    render = _render_artifact_record(project_id)
    _register(render)
    _register(_overlay_render_artifact_record(project_id, render, manifest, overlay_count=0))

    result = build_final_local(
        project_id=project_id, timed_manifest_path=project_dir / "manifest.json", **_paths(tmp_path)
    )
    assert isinstance(result, LocalResumeResult)
    with pytest.raises(Exception):
        result.render_status = "render_created"  # type: ignore[misc]


def test_no_automatic_retry_loop(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    calls: list = []

    def _boom(project_id, manifest_path, output_path):
        calls.append(1)
        raise FinalVideoAssemblyError("boom")

    monkeypatch.setattr(f"{MODULE}.assemble_final_video", _boom)

    result = build_final_local(
        project_id=project_id, timed_manifest_path=project_dir / "manifest.json", **_paths(tmp_path)
    )

    assert len(calls) == 1
    assert result.render_status == "failed"


# ---------------------------------------------------------------------
# 8-10, 13: failure stopping behavior
# ---------------------------------------------------------------------


def test_assembly_failure_blocks_later_stages(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    derive_calls: list = []
    render_calls: list = []
    monkeypatch.setattr(
        f"{MODULE}.assemble_final_video",
        lambda *a, **k: (_ for _ in ()).throw(FinalVideoAssemblyError("SENTINEL-assembly-secret-path")),
    )
    monkeypatch.setattr(f"{MODULE}.derive_text_overlays", lambda *a, **k: derive_calls.append(1))
    monkeypatch.setattr(f"{MODULE}.render_text_overlays", lambda *a, **k: render_calls.append(1))

    result = build_final_local(
        project_id=project_id, timed_manifest_path=project_dir / "manifest.json", **_paths(tmp_path)
    )

    assert derive_calls == []
    assert render_calls == []
    assert result.render_status == "failed"
    assert result.derived_manifest_status == "not_attempted"
    assert result.overlay_render_status == "not_attempted"
    assert result.stopped_at_stage == "assemble_final_video"
    assert "SENTINEL-assembly-secret-path" in " ".join(result.blocked_reasons)


def test_derivation_failure_blocks_renderer(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register(_render_artifact_record(project_id))
    render_calls: list = []
    monkeypatch.setattr(
        f"{MODULE}.derive_text_overlays",
        lambda *a, **k: (_ for _ in ()).throw(OverlayDerivationError("derivation exploded")),
    )
    monkeypatch.setattr(f"{MODULE}.render_text_overlays", lambda *a, **k: render_calls.append(1))

    result = build_final_local(
        project_id=project_id, timed_manifest_path=project_dir / "manifest.json", **_paths(tmp_path)
    )

    assert render_calls == []
    assert result.derived_manifest_status == "failed"
    assert result.overlay_render_status == "not_attempted"
    assert result.stopped_at_stage == "derive_text_overlays"


def test_overlay_rendering_failure_preserves_derived_manifest(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register(_render_artifact_record(project_id))
    monkeypatch.setattr(
        f"{MODULE}.render_text_overlays",
        lambda *a, **k: (_ for _ in ()).throw(TextOverlayRenderError("burn-in exploded")),
    )

    paths = _paths(tmp_path)
    result = build_final_local(project_id=project_id, timed_manifest_path=project_dir / "manifest.json", **paths)

    assert result.overlay_render_status == "failed"
    assert result.stopped_at_stage == "render_text_overlays"
    assert result.derived_manifest_status == "derived_manifest_created"
    assert Path(result.derived_manifest_path).exists()
    before = Path(result.derived_manifest_path).read_bytes()
    assert paths["derived_manifest_output_path"].read_bytes() == before


def test_underlying_error_surfaced_without_raw_stderr(isolated_db, tmp_path, monkeypatch):
    monkeypatch.setattr(
        f"{MODULE}.assemble_final_video",
        lambda *a, **k: (_ for _ in ()).throw(FinalVideoAssemblyError("scene 'scene-01' has no eligible artifact")),
    )
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")

    result = build_final_local(
        project_id=project_id, timed_manifest_path=project_dir / "manifest.json", **_paths(tmp_path)
    )
    joined = " ".join(result.blocked_reasons)
    assert "Traceback" not in joined
    assert "subprocess" not in joined.lower()


# ---------------------------------------------------------------------
# 11-12: reload-before-reuse
# ---------------------------------------------------------------------


def test_successful_assembly_reloads_artifacts_before_derivation(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    monkeypatch.setattr(f"{MODULE}.assemble_final_video", _fake_assemble_final_video())
    monkeypatch.setattr(f"{MODULE}.render_text_overlays", _fake_render_text_overlays())

    result = build_final_local(
        project_id=project_id, timed_manifest_path=project_dir / "manifest.json", **_paths(tmp_path)
    )
    assert result.render_status == "render_created"
    assert result.render_artifact_id == "render-final"


def test_successful_overlay_render_reloads_and_confirms_lineage(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    monkeypatch.setattr(f"{MODULE}.assemble_final_video", _fake_assemble_final_video())
    monkeypatch.setattr(f"{MODULE}.render_text_overlays", _fake_render_text_overlays())

    result = build_final_local(
        project_id=project_id, timed_manifest_path=project_dir / "manifest.json", **_paths(tmp_path)
    )
    assert result.overlay_render_status == "overlay_render_created"
    assert result.overlay_render_artifact_id == "overlay-render-final"


def test_overlay_render_reload_mismatch_raises(isolated_db, tmp_path, monkeypatch):
    """render_text_overlays 'succeeds' (per the fake) but registers nothing
    — the orchestrator's own post-call re-check must catch this impossible
    state rather than silently reporting success."""
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    monkeypatch.setattr(f"{MODULE}.assemble_final_video", _fake_assemble_final_video())
    monkeypatch.setattr(f"{MODULE}.render_text_overlays", lambda *a, **k: None)

    with pytest.raises(LocalResumeOrchestratorError):
        build_final_local(project_id=project_id, timed_manifest_path=project_dir / "manifest.json", **_paths(tmp_path))


# ---------------------------------------------------------------------
# 17-23: path collision tests
# ---------------------------------------------------------------------


def test_pairwise_collision_timed_and_derived(isolated_db, tmp_path):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    paths = _paths(tmp_path)
    paths["derived_manifest_output_path"] = project_dir / "manifest.json"
    with pytest.raises(LocalResumeOrchestratorError):
        build_final_local(project_id=project_id, timed_manifest_path=project_dir / "manifest.json", **paths)


def test_pairwise_collision_render_and_overlay(isolated_db, tmp_path):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    paths = _paths(tmp_path)
    paths["overlay_output_path"] = paths["render_output_path"]
    with pytest.raises(LocalResumeOrchestratorError):
        build_final_local(project_id=project_id, timed_manifest_path=project_dir / "manifest.json", **paths)


def test_output_aliases_canonical_manifest_path(isolated_db, tmp_path):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    paths = _paths(tmp_path)
    paths["render_output_path"] = project_dir / "manifest.json"
    with pytest.raises(LocalResumeOrchestratorError):
        build_final_local(project_id=project_id, timed_manifest_path=project_dir / "manifest.json", **paths)


def test_output_aliases_render_final_mp4(isolated_db, tmp_path):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    paths = _paths(tmp_path)
    paths["overlay_output_path"] = project_dir / "render" / "final.mp4"
    with pytest.raises(LocalResumeOrchestratorError):
        build_final_local(project_id=project_id, timed_manifest_path=project_dir / "manifest.json", **paths)


def test_output_aliases_overlay_render_final_mp4(isolated_db, tmp_path):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    paths = _paths(tmp_path)
    paths["render_output_path"] = project_dir / "overlay_render" / "final.mp4"
    with pytest.raises(LocalResumeOrchestratorError):
        build_final_local(project_id=project_id, timed_manifest_path=project_dir / "manifest.json", **paths)


def test_output_aliases_registered_audio_artifact(isolated_db, tmp_path):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register(
        ArtifactRecord(
            artifact_id="audio-scene-01",
            project_id=project_id,
            kind="audio",
            scene_id="scene-01",
            relative_path="audio/scene-01.wav",
            byte_size=1,
            sha256_checksum="5" * 64,
            created_at=FIXED_NOW,
            metadata={"duration_seconds": 5.0, "source": "external"},
        )
    )
    paths = _paths(tmp_path)
    paths["render_output_path"] = project_dir / "audio" / "scene-01.wav"
    with pytest.raises(LocalResumeOrchestratorError):
        build_final_local(project_id=project_id, timed_manifest_path=project_dir / "manifest.json", **paths)


def test_output_aliases_registered_visual_artifact(isolated_db, tmp_path):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register(
        ArtifactRecord(
            artifact_id="visual-scene-01",
            project_id=project_id,
            kind="visual",
            scene_id="scene-01",
            relative_path="visual/scene-01.png",
            byte_size=1,
            sha256_checksum="3" * 64,
            created_at=FIXED_NOW,
            metadata={},
        )
    )
    paths = _paths(tmp_path)
    paths["overlay_output_path"] = project_dir / "visual" / "scene-01.png"
    with pytest.raises(LocalResumeOrchestratorError):
        build_final_local(project_id=project_id, timed_manifest_path=project_dir / "manifest.json", **paths)


def test_output_aliases_registered_animation_artifact(isolated_db, tmp_path):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register(
        ArtifactRecord(
            artifact_id="animation-scene-01",
            project_id=project_id,
            kind="animation",
            scene_id="scene-01",
            relative_path="animation/scene-01.mp4",
            byte_size=1,
            sha256_checksum="4" * 64,
            created_at=FIXED_NOW,
            metadata={},
        )
    )
    paths = _paths(tmp_path)
    paths["derived_manifest_output_path"] = project_dir / "animation" / "scene-01.mp4"
    with pytest.raises(LocalResumeOrchestratorError):
        build_final_local(project_id=project_id, timed_manifest_path=project_dir / "manifest.json", **paths)


# ---------------------------------------------------------------------
# 24-28: existence hard-stops / resume exceptions
# ---------------------------------------------------------------------


def test_existing_unrelated_derived_manifest_output_does_not_hard_stop_up_front(isolated_db, tmp_path, monkeypatch):
    """Unlike render/overlay output, an existing derived-manifest path is
    a legitimate resume scenario (case E3) — it must not be a blanket
    up-front hard-stop; it's validated instead."""
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register(_render_artifact_record(project_id))
    paths = _paths(tmp_path)
    paths["derived_manifest_output_path"].parent.mkdir(parents=True, exist_ok=True)
    paths["derived_manifest_output_path"].write_text("not a manifest", encoding="utf-8")
    monkeypatch.setattr(f"{MODULE}.render_text_overlays", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run")))

    result = build_final_local(project_id=project_id, timed_manifest_path=project_dir / "manifest.json", **paths)
    assert result.derived_manifest_status == "blocked"
    assert result.stopped_at_stage == "derived_manifest_validation"


def test_existing_render_output_hard_stops_when_render_missing(isolated_db, tmp_path):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    paths = _paths(tmp_path)
    paths["render_output_path"].parent.mkdir(parents=True, exist_ok=True)
    paths["render_output_path"].write_bytes(b"pre-existing")
    with pytest.raises(LocalResumeOrchestratorError):
        build_final_local(project_id=project_id, timed_manifest_path=project_dir / "manifest.json", **paths)


def test_existing_overlay_output_hard_stops_when_overlay_render_missing(isolated_db, tmp_path):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    paths = _paths(tmp_path)
    paths["overlay_output_path"].parent.mkdir(parents=True, exist_ok=True)
    paths["overlay_output_path"].write_bytes(b"pre-existing")
    with pytest.raises(LocalResumeOrchestratorError):
        build_final_local(project_id=project_id, timed_manifest_path=project_dir / "manifest.json", **paths)


def test_valid_existing_render_ignores_render_output_without_write(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register(_render_artifact_record(project_id))
    monkeypatch.setattr(f"{MODULE}.render_text_overlays", _fake_render_text_overlays())

    paths = _paths(tmp_path)
    result = build_final_local(project_id=project_id, timed_manifest_path=project_dir / "manifest.json", **paths)
    assert result.render_status == "render_reused"
    assert not paths["render_output_path"].exists()


def test_valid_existing_overlay_render_ignores_all_outputs_without_write(isolated_db, tmp_path):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    render = _render_artifact_record(project_id)
    _register(render)
    _register(_overlay_render_artifact_record(project_id, render, manifest, overlay_count=0))

    paths = _paths(tmp_path)
    result = build_final_local(project_id=project_id, timed_manifest_path=project_dir / "manifest.json", **paths)
    assert result.stopped_at_stage is None
    assert not paths["render_output_path"].exists()
    assert not paths["overlay_output_path"].exists()
    assert not paths["derived_manifest_output_path"].exists()


# ---------------------------------------------------------------------
# 29-40: derived-manifest validation
# ---------------------------------------------------------------------


def _write_derived_manifest_at(project_id, project_dir, manifest, render, path: Path, mutate=None):
    derived = derive_text_overlays(manifest, render)
    if mutate is not None:
        derived = mutate(derived)
    save_manifest(derived, path)
    return derived


def test_derived_manifest_project_id_mismatch_blocked(isolated_db, tmp_path):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    render = _render_artifact_record(project_id)
    _register(render)
    paths = _paths(tmp_path)
    _write_derived_manifest_at(
        project_id, project_dir, manifest, render, paths["derived_manifest_output_path"],
        mutate=lambda m: m.model_copy(update={"project_id": "different-project"}),
    )
    result = build_final_local(project_id=project_id, timed_manifest_path=project_dir / "manifest.json", **paths)
    assert result.derived_manifest_status == "blocked"
    assert result.stopped_at_stage == "derived_manifest_validation"


def test_derived_manifest_alters_explicit_overlays_blocked(isolated_db, tmp_path):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    render = _render_artifact_record(project_id)
    _register(render)
    paths = _paths(tmp_path)

    def _mutate(m):
        scene = m.scene_plan.scenes[0]
        overlay = scene.text_overlays[0].model_copy(update={"text": "a completely different caption"})
        new_scene = scene.model_copy(update={"text_overlays": (overlay,)})
        new_plan = m.scene_plan.model_copy(update={"scenes": (new_scene,)})
        return m.model_copy(update={"scene_plan": new_plan})

    _write_derived_manifest_at(
        project_id, project_dir, manifest, render, paths["derived_manifest_output_path"], mutate=_mutate
    )
    result = build_final_local(project_id=project_id, timed_manifest_path=project_dir / "manifest.json", **paths)
    assert result.derived_manifest_status == "blocked"


def test_derived_manifest_exact_equality_is_reused(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    render = _render_artifact_record(project_id)
    _register(render)
    paths = _paths(tmp_path)
    _write_derived_manifest_at(project_id, project_dir, manifest, render, paths["derived_manifest_output_path"])

    derive_calls: list = []
    monkeypatch.setattr(
        f"{MODULE}.derive_text_overlays",
        lambda m, r: (derive_calls.append(1), derive_text_overlays(m, r))[1],
    )
    save_calls: list = []
    monkeypatch.setattr(f"{MODULE}.save_manifest", lambda *a, **k: save_calls.append(1))
    monkeypatch.setattr(f"{MODULE}.render_text_overlays", _fake_render_text_overlays())

    result = build_final_local(project_id=project_id, timed_manifest_path=project_dir / "manifest.json", **paths)

    assert result.derived_manifest_status == "derived_manifest_reused"
    assert save_calls == []  # save_manifest is never called for a reused derived manifest
    assert result.overlay_render_status == "overlay_render_created"


def test_newly_written_derived_manifest_matches_expected(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register(_render_artifact_record(project_id))
    monkeypatch.setattr(f"{MODULE}.render_text_overlays", _fake_render_text_overlays())
    paths = _paths(tmp_path)

    result = build_final_local(project_id=project_id, timed_manifest_path=project_dir / "manifest.json", **paths)
    assert result.derived_manifest_status == "derived_manifest_created"
    from src.core.manifest_store import load_manifest

    written = load_manifest(Path(result.derived_manifest_path))
    render = _render_artifact_record(project_id)
    expected = derive_text_overlays(manifest, render)
    assert written.model_dump(mode="json") == expected.model_dump(mode="json")


def test_timed_manifest_unchanged(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    monkeypatch.setattr(f"{MODULE}.assemble_final_video", _fake_assemble_final_video())
    monkeypatch.setattr(f"{MODULE}.render_text_overlays", _fake_render_text_overlays())
    manifest_path = project_dir / "manifest.json"
    before = manifest_path.read_bytes()

    build_final_local(project_id=project_id, timed_manifest_path=manifest_path, **_paths(tmp_path))
    assert manifest_path.read_bytes() == before


def test_derived_manifest_unchanged_after_downstream_render_failure(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register(_render_artifact_record(project_id))
    monkeypatch.setattr(
        f"{MODULE}.render_text_overlays",
        lambda *a, **k: (_ for _ in ()).throw(TextOverlayRenderError("boom")),
    )
    paths = _paths(tmp_path)
    result = build_final_local(project_id=project_id, timed_manifest_path=project_dir / "manifest.json", **paths)
    before = paths["derived_manifest_output_path"].read_bytes()

    # Call again with the same (now-existing) derived-manifest path and a
    # fresh overlay-output path — still failing render — must not touch it.
    monkeypatch.setattr(
        f"{MODULE}.render_text_overlays",
        lambda *a, **k: (_ for _ in ()).throw(TextOverlayRenderError("boom again")),
    )
    paths2 = dict(paths)
    paths2["overlay_output_path"] = tmp_path / "run2-overlay.mp4"
    build_final_local(project_id=project_id, timed_manifest_path=project_dir / "manifest.json", **paths2)
    assert paths["derived_manifest_output_path"].read_bytes() == before


# ---------------------------------------------------------------------
# 41-52: safety / lifecycle
# ---------------------------------------------------------------------


def _orchestrator_function_source() -> str:
    """Source of every actual function body in the orchestrator module
    (never the module's own top-level docstring, which legitimately
    names these same forbidden words in explanatory prose about what is
    NOT done — same false-positive pitfall already fixed once before in
    this codebase for cmd_plan_final_assembly's own docstring)."""
    import src.core.local_resume_orchestrator as m

    return "\n".join(
        inspect.getsource(obj)
        for name, obj in vars(m).items()
        if inspect.isfunction(obj) and obj.__module__ == m.__name__
    )


def test_no_provider_import():
    source = _orchestrator_function_source()
    for forbidden in ("llm_groq", "llm_tokenrouter", "tts_kokoro", "image_qwen", "lipsync_rhubarb"):
        assert forbidden not in source


def test_no_direct_subprocess_or_ffmpeg_call():
    source = _orchestrator_function_source()
    assert "subprocess" not in source
    assert "ffmpeg" not in source.lower()


def test_no_artifact_verifier_call():
    source = _orchestrator_function_source()
    assert "artifact_verifier" not in source
    assert "verify_artifact" not in source


def test_no_lifecycle_or_job_write():
    source = _orchestrator_function_source()
    assert "save_transition" not in source
    assert "transition_project" not in source
    assert "jobs" not in source.lower()


def test_source_artifacts_never_modified(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    monkeypatch.setattr(f"{MODULE}.assemble_final_video", _fake_assemble_final_video())
    monkeypatch.setattr(f"{MODULE}.render_text_overlays", _fake_render_text_overlays())
    audio_path = project_dir / "audio" / "scene-01.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"original-audio")

    build_final_local(project_id=project_id, timed_manifest_path=project_dir / "manifest.json", **_paths(tmp_path))
    assert audio_path.read_bytes() == b"original-audio"


def test_no_output_parent_directory_created_before_validation_fails(isolated_db, tmp_path):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    paths = _paths(tmp_path)
    paths["derived_manifest_output_path"] = project_dir / "manifest.json"  # forces a collision
    with pytest.raises(LocalResumeOrchestratorError):
        build_final_local(project_id=project_id, timed_manifest_path=project_dir / "manifest.json", **paths)
    assert not paths["render_output_path"].parent.exists()


def test_keyboard_interrupt_not_caught(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    monkeypatch.setattr(f"{MODULE}.assemble_final_video", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        build_final_local(project_id=project_id, timed_manifest_path=project_dir / "manifest.json", **_paths(tmp_path))


def test_empty_project_id_raises():
    with pytest.raises(LocalResumeOrchestratorError):
        build_final_local(
            project_id="",
            timed_manifest_path=Path("x"),
            derived_manifest_output_path=Path("y"),
            render_output_path=Path("z"),
            overlay_output_path=Path("w"),
        )


def test_unknown_project_id_raises(isolated_db, tmp_path):
    init_db()
    with pytest.raises(LocalResumeOrchestratorError):
        build_final_local(
            project_id="does-not-exist",
            timed_manifest_path=tmp_path / "m.json",
            derived_manifest_output_path=tmp_path / "d.json",
            render_output_path=tmp_path / "r.mp4",
            overlay_output_path=tmp_path / "o.mp4",
        )


def test_missing_timed_manifest_raises(isolated_db, tmp_path):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    paths = _paths(tmp_path)
    with pytest.raises(LocalResumeOrchestratorError):
        build_final_local(project_id=project_id, timed_manifest_path=tmp_path / "does-not-exist.json", **paths)


# ---------------------------------------------------------------------
# Real FFmpeg integration (doubly-gated)
# ---------------------------------------------------------------------

_FFMPEG_PATH = None
try:
    from src.utils.config import get_settings as _get_settings

    _FFMPEG_PATH = shutil.which(_get_settings().ffmpeg_path) or (
        _get_settings().ffmpeg_path if Path(_get_settings().ffmpeg_path).exists() else None
    )
except Exception:  # pragma: no cover — settings not loadable in this environment
    _FFMPEG_PATH = None

_ARIAL_PATH = Path(r"C:\Windows\Fonts\arial.ttf")


def _ffmpeg_available() -> bool:
    if not _FFMPEG_PATH:
        return False
    import subprocess

    try:
        result = subprocess.run([_FFMPEG_PATH, "-hide_banner", "-filters"], capture_output=True, text=True, timeout=10)
    except OSError:
        return False
    return result.returncode == 0 and "drawtext" in result.stdout


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg with drawtext is not available")
@pytest.mark.skipif(not _ARIAL_PATH.exists(), reason="C:\\Windows\\Fonts\\arial.ttf is not available")
def test_real_ffmpeg_end_to_end_build_final_local(isolated_db, tmp_path):
    import wave

    from src.render.ffmpeg_render import ken_burns_clip

    project_id, project_dir, manifest = _create_registered_project(
        tmp_path / "projects", scene_ids=("scene-01", "scene-02")
    )

    # Real local audio (silence) + real local animation (Ken Burns over a
    # tiny generated PNG) for both scenes — no provider anywhere.
    from PIL import Image

    image_path = tmp_path / "scene.png"
    Image.new("RGB", (640, 360), color=(200, 200, 200)).save(image_path)

    for scene_id in ("scene-01", "scene-02"):
        audio_path = project_dir / "audio" / f"{scene_id}.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(audio_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 16000 * 5)  # 5s silence

        animation_path = project_dir / "animation" / f"{scene_id}.mp4"
        animation_path.parent.mkdir(parents=True, exist_ok=True)
        ken_burns_clip(image_path, 5.0, animation_path, motion="static")

        _register(
            ArtifactRecord(
                artifact_id=f"audio-{scene_id}",
                project_id=project_id,
                kind="audio",
                scene_id=scene_id,
                relative_path=f"audio/{scene_id}.wav",
                byte_size=audio_path.stat().st_size,
                sha256_checksum=hashlib.sha256(audio_path.read_bytes()).hexdigest(),
                created_at=FIXED_NOW,
                metadata={"duration_seconds": 5.0, "source": "external"},
            )
        )
        _register(
            ArtifactRecord(
                artifact_id=f"animation-{scene_id}",
                project_id=project_id,
                kind="animation",
                scene_id=scene_id,
                relative_path=f"animation/{scene_id}.mp4",
                byte_size=animation_path.stat().st_size,
                sha256_checksum=hashlib.sha256(animation_path.read_bytes()).hexdigest(),
                created_at=FIXED_NOW,
                metadata={"duration_seconds": 5.0, "source": "external"},
            )
        )

    audio_original = {
        sid: (project_dir / "audio" / f"{sid}.wav").read_bytes() for sid in ("scene-01", "scene-02")
    }
    animation_original = {
        sid: (project_dir / "animation" / f"{sid}.mp4").read_bytes() for sid in ("scene-01", "scene-02")
    }

    socket_calls: list = []
    real_socket_init = socket.socket.__init__

    def _tracking_init(self, *args, **kwargs):
        socket_calls.append((args, kwargs))
        return real_socket_init(self, *args, **kwargs)

    socket.socket.__init__ = _tracking_init
    try:
        paths = _paths(tmp_path, "real")
        result = build_final_local(project_id=project_id, timed_manifest_path=project_dir / "manifest.json", **paths)
    finally:
        socket.socket.__init__ = real_socket_init
    assert socket_calls == []

    assert result.stopped_at_stage is None
    assert result.render_status == "render_created"
    assert result.derived_manifest_status == "derived_manifest_created"
    assert result.overlay_render_status == "overlay_render_created"

    assert paths["render_output_path"].exists()
    assert paths["overlay_output_path"].exists()
    assert (project_dir / "render" / "final.mp4").exists()
    assert (project_dir / "overlay_render" / "final.mp4").exists()

    from src.core.manifest_store import load_manifest

    derived = load_manifest(paths["derived_manifest_output_path"])
    assert all(scene.text_overlays for scene in derived.scene_plan.scenes)

    conn = get_connection()
    try:
        from src.database.artifact_repository import list_artifacts_by_project

        render_artifacts = list_artifacts_by_project(conn, project_id, kind="render")
        overlay_artifacts = list_artifacts_by_project(conn, project_id, kind="overlay_render")
    finally:
        conn.close()
    assert len(render_artifacts) == 1
    assert len(overlay_artifacts) == 1
    assert overlay_artifacts[0].metadata["source_render_artifact_id"] == render_artifacts[0].artifact_id
    assert overlay_artifacts[0].metadata["manifest_fingerprint"] == derived.source_fingerprint

    for sid in ("scene-01", "scene-02"):
        assert (project_dir / "audio" / f"{sid}.wav").read_bytes() == audio_original[sid]
        assert (project_dir / "animation" / f"{sid}.mp4").read_bytes() == animation_original[sid]

    for tmp_dir in tmp_path.glob("**/final-video-assembly-*"):
        assert not tmp_dir.exists()
    for tmp_dir in tmp_path.glob("**/text-overlay-render-*"):
        assert not tmp_dir.exists()
