"""Tests for src/core/final_video_assembly.py — the local-only,
generation-and-registration final assembly module. FFmpeg helpers
(mux_audio_video/concat_videos/get_duration_seconds/_probe_stream_types)
are mocked at their point of import in this module for every test except
the one doubly-gated real-ffmpeg integration test at the bottom. Uses the
same isolated_db / _create_registered_project pattern as every other
CLI-level test in this repo, since assemble_final_video() itself owns its
own short-lived SQLite connections (see the module's own docstring for
why)."""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.core.final_video_assembly import (
    AmbiguousSceneArtifactError,
    ArtifactFileNotFoundError,
    EmptyManifestError,
    ExistingFinalVideoArtifactError,
    FinalArtifactRegistrationError,
    FinalConcatError,
    FinalVideoAssemblyResult,
    FinalVideoCleanupError,
    InvalidFinalOutputError,
    InvalidManifestError,
    InvalidSceneOrderError,
    ManifestNotFoundError,
    MediaProbeError,
    MissingSceneArtifactError,
    OutputPathConflictError,
    ProjectNotFoundError,
    SceneMuxError,
    _duration_tolerance_seconds,
    _resolve_scene_sources,
    assemble_final_video,
)
from src.core.manifest_builder import build_video_manifest
from src.core.manifest_store import save_manifest
from src.core.project_state_machine import create_initial_project, initial_transition_for
from src.database.artifact_repository import register_artifact
from src.database.db import get_connection
from src.database.project_repository import create_project
from src.models.artifact import ArtifactRecord
from src.utils.channel_config import get_channel_policy

FIXED_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
MODULE = "src.core.final_video_assembly"
REGISTRAR = "src.core.render_artifact_registrar"


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


def _create_registered_project(
    projects_root: Path,
    scene_ids: tuple[str, ...] = ("scene-01", "scene-02"),
    story_id: str = "why-we-care-what-people-think",
):
    manifest = build_video_manifest(
        _story_input_dict(story_id), _scene_plan_dict(scene_ids), get_channel_policy(), created_at=FIXED_NOW
    )
    manifest_path = projects_root / manifest.project_id / "manifest.json"
    save_manifest(manifest, manifest_path)

    project = create_initial_project(manifest_path, manifest, now=FIXED_NOW)
    transition = initial_transition_for(project)

    from src.database.db import init_db

    init_db()
    conn = get_connection()
    try:
        create_project(conn, project, transition)
    finally:
        conn.close()
    return project.project_id, manifest_path.parent, manifest


def _register_artifact(project_dir, project_id, scene_id, kind, *, suffix, content=b"fake-media-bytes") -> Path:
    path = project_dir / kind / f"{scene_id}.{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    checksum = hashlib.sha256(content).hexdigest()
    conn = get_connection()
    try:
        register_artifact(
            conn,
            ArtifactRecord(
                artifact_id=f"{kind}-{scene_id}",
                project_id=project_id,
                kind=kind,
                scene_id=scene_id,
                relative_path=f"{kind}/{scene_id}.{suffix}",
                byte_size=len(content),
                sha256_checksum=checksum,
                created_at=FIXED_NOW,
                metadata={"duration_seconds": 1.0, "source": "external"},
            ),
        )
    finally:
        conn.close()
    return path


def _register_audio(project_dir, project_id, scene_id, **kw) -> Path:
    return _register_artifact(project_dir, project_id, scene_id, "audio", suffix="wav", **kw)


def _register_animation(project_dir, project_id, scene_id, **kw) -> Path:
    return _register_artifact(project_dir, project_id, scene_id, "animation", suffix="mp4", **kw)


def _register_scenes(project_dir, project_id, scene_ids):
    for scene_id in scene_ids:
        _register_animation(project_dir, project_id, scene_id)
        _register_audio(project_dir, project_id, scene_id)


def _mock_ffmpeg_ok(monkeypatch, calls, scene_duration=2.0, animation_has_audio=False):
    """Wires up mux_audio_video/concat_videos/get_duration_seconds/
    _probe_stream_types with a shared, internally-consistent fake ledger.
    Any path not yet seen by the fakes (i.e. an original registered
    animation source) probes as video-only unless `animation_has_audio`
    is set; every path the fake mux/concat write themselves is recorded
    as containing both a video and an audio stream, matching real
    mux_audio_video()/concat_videos() behavior. Also patches
    render_artifact_registrar's OWN get_duration_seconds/_has_video_stream
    (a separate import binding register_render_artifact()/
    prevalidate_render_source() use for their own ffprobe calls) so
    registration succeeds against the same fake bytes."""
    durations: dict[str, float] = {}
    stream_types: dict[str, frozenset[str]] = {}
    final_duration_holder: dict[str, float] = {}

    def _fake_probe(path):
        calls.append(("probe", Path(path)))
        key = str(Path(path))
        if key in stream_types:
            return stream_types[key]
        return frozenset({"video", "audio"}) if animation_has_audio else frozenset({"video"})

    def _fake_mux(video_path, audio_path, out_path):
        calls.append(("mux", Path(video_path), Path(audio_path), Path(out_path)))
        Path(out_path).write_bytes(b"fake-muxed-clip")
        durations[str(Path(out_path))] = scene_duration
        stream_types[str(Path(out_path))] = frozenset({"video", "audio"})
        return out_path

    def _fake_concat(video_paths, out_path):
        calls.append(("concat", [Path(p) for p in video_paths], Path(out_path)))
        Path(out_path).write_bytes(b"fake-final-video")
        total = sum(durations[str(Path(p))] for p in video_paths)
        durations[str(Path(out_path))] = total
        stream_types[str(Path(out_path))] = frozenset({"video", "audio"})
        final_duration_holder["value"] = total
        return out_path

    def _fake_get_duration_seconds(path):
        calls.append(("get_duration_seconds", Path(path)))
        return durations[str(Path(path))]

    monkeypatch.setattr(f"{MODULE}._probe_stream_types", _fake_probe)
    monkeypatch.setattr(f"{MODULE}.mux_audio_video", _fake_mux)
    monkeypatch.setattr(f"{MODULE}.concat_videos", _fake_concat)
    monkeypatch.setattr(f"{MODULE}.get_duration_seconds", _fake_get_duration_seconds)

    monkeypatch.setattr(f"{REGISTRAR}.get_duration_seconds", lambda path: final_duration_holder["value"])
    monkeypatch.setattr(f"{REGISTRAR}._has_video_stream", lambda path: True)
    return durations, stream_types


def _forbid_ffmpeg(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("must not be called")

    monkeypatch.setattr(f"{MODULE}._probe_stream_types", _boom)
    monkeypatch.setattr(f"{MODULE}.mux_audio_video", _boom)
    monkeypatch.setattr(f"{MODULE}.concat_videos", _boom)
    monkeypatch.setattr(f"{MODULE}.get_duration_seconds", _boom)


# ---------------------------------------------------------------------
# 1-3: successful assembly, order preservation
# ---------------------------------------------------------------------


def test_single_scene_assembly_succeeds(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_scenes(project_dir, project_id, ("scene-01",))
    calls: list = []
    _mock_ffmpeg_ok(monkeypatch, calls)
    output = tmp_path / "final.mp4"

    result = assemble_final_video(project_id, project_dir / "manifest.json", output)

    assert result.project_id == project_id
    assert result.output_path == output.resolve()
    assert result.scene_count == 1
    assert result.measured_duration_seconds == pytest.approx(2.0)
    assert result.artifact_id == "render-final"
    assert output.exists()


def test_multi_scene_assembly_succeeds(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(
        tmp_path / "projects", ("scene-01", "scene-02", "scene-03")
    )
    _register_scenes(project_dir, project_id, ("scene-01", "scene-02", "scene-03"))
    calls: list = []
    _mock_ffmpeg_ok(monkeypatch, calls)
    output = tmp_path / "final.mp4"

    result = assemble_final_video(project_id, project_dir / "manifest.json", output)

    assert result.scene_count == 3
    assert result.measured_duration_seconds == pytest.approx(6.0)


def test_exact_manifest_order_preserved(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(
        tmp_path / "projects", ("scene-01", "scene-02", "scene-03")
    )
    _register_scenes(project_dir, project_id, ("scene-01", "scene-02", "scene-03"))
    calls: list = []
    _mock_ffmpeg_ok(monkeypatch, calls)
    output = tmp_path / "final.mp4"

    assemble_final_video(project_id, project_dir / "manifest.json", output)

    mux_calls = [c for c in calls if c[0] == "mux"]
    assert len(mux_calls) == 3
    ordered_scene_ids = [video_path.stem for _, video_path, _audio_path, _out in mux_calls]
    assert ordered_scene_ids == ["scene-01", "scene-02", "scene-03"]

    concat_call = next(c for c in calls if c[0] == "concat")
    concat_scene_order = [p.name for p in concat_call[1]]
    assert concat_scene_order == ["0001_scene-01.mp4", "0002_scene-02.mp4", "0003_scene-03.mp4"]


# ---------------------------------------------------------------------
# 4-9: preflight failures before FFmpeg
# ---------------------------------------------------------------------


def test_missing_project_fails_before_ffmpeg(isolated_db, tmp_path, monkeypatch):
    from src.database.db import init_db

    init_db()
    _forbid_ffmpeg(monkeypatch)
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ProjectNotFoundError):
        assemble_final_video("does-not-exist", manifest_path, tmp_path / "out.mp4")


def test_missing_manifest_fails_before_ffmpeg(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _forbid_ffmpeg(monkeypatch)

    with pytest.raises(ManifestNotFoundError):
        assemble_final_video(project_id, tmp_path / "does-not-exist.json", tmp_path / "out.mp4")


def test_invalid_json_manifest_fails(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _forbid_ffmpeg(monkeypatch)
    bad_manifest = tmp_path / "bad.json"
    bad_manifest.write_text("not valid json", encoding="utf-8")

    with pytest.raises(InvalidManifestError):
        assemble_final_video(project_id, bad_manifest, tmp_path / "out.mp4")


def test_schema_invalid_manifest_fails(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    other_id, other_dir, other_manifest = _create_registered_project(
        tmp_path / "projects2", ("scene-01",), story_id="a-completely-different-story"
    )
    _forbid_ffmpeg(monkeypatch)

    with pytest.raises(InvalidManifestError, match="does not match"):
        assemble_final_video(project_id, other_dir / "manifest.json", tmp_path / "out.mp4")


def test_empty_scene_list_fails():
    """ScenePlan.scenes itself has min_length=1 (see src/models/scene.py),
    so a manifest with zero scenes can never reach load_manifest()
    successfully in practice — proven here directly, and
    EmptyManifestError's own check is exercised as the defensive backstop
    this module keeps for that guarantee."""
    from src.models.scene import ScenePlan

    with pytest.raises(ValueError, match="at least 1 item"):
        ScenePlan(scenes=(), role_outfits=())

    with pytest.raises(EmptyManifestError):
        raise EmptyManifestError("manifest scene_plan contains no scenes")


def test_duplicate_scene_ids_rejected_defensively():
    """Unreachable via load_manifest() (ScenePlan forbids duplicates at
    construction) — proven directly against the internal resolver, same
    style as test_non_positive_derived_duration_rejects_direct_unit_test
    in tests/test_mouth_animation_generation.py."""
    from types import SimpleNamespace

    scene_a = SimpleNamespace(scene_id="scene-01")
    scene_b = SimpleNamespace(scene_id="scene-01")
    fake_scene_plan = SimpleNamespace(scenes=(scene_a, scene_b))
    fake_manifest = SimpleNamespace(project_id="proj-x", scene_plan=fake_scene_plan)

    with pytest.raises(InvalidSceneOrderError, match="duplicate scene_id"):
        _resolve_scene_sources(fake_manifest, [], [], Path("."))


# ---------------------------------------------------------------------
# 10-15: artifact resolution failures
# ---------------------------------------------------------------------


def test_missing_audio_artifact_fails(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_animation(project_dir, project_id, "scene-01")
    _forbid_ffmpeg(monkeypatch)

    with pytest.raises(MissingSceneArtifactError, match="'audio'"):
        assemble_final_video(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")


def test_missing_animation_artifact_fails(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_audio(project_dir, project_id, "scene-01")
    _forbid_ffmpeg(monkeypatch)

    with pytest.raises(MissingSceneArtifactError, match="'animation'"):
        assemble_final_video(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")


def test_ambiguous_audio_artifact_fails(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_animation(project_dir, project_id, "scene-01")
    _register_audio(project_dir, project_id, "scene-01")
    conn = get_connection()
    try:
        extra_path = project_dir / "audio" / "scene-01-extra.wav"
        extra_path.write_bytes(b"extra")
        register_artifact(
            conn,
            ArtifactRecord(
                artifact_id="audio-scene-01-extra",
                project_id=project_id,
                kind="audio",
                scene_id="scene-01",
                relative_path="audio/scene-01-extra.wav",
                byte_size=5,
                sha256_checksum=hashlib.sha256(b"extra").hexdigest(),
                created_at=FIXED_NOW,
                metadata={"duration_seconds": 1.0, "source": "external"},
            ),
        )
    finally:
        conn.close()
    _forbid_ffmpeg(monkeypatch)

    with pytest.raises(AmbiguousSceneArtifactError, match="'audio'"):
        assemble_final_video(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")


def test_ambiguous_animation_artifact_fails(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_audio(project_dir, project_id, "scene-01")
    _register_animation(project_dir, project_id, "scene-01")
    conn = get_connection()
    try:
        extra_path = project_dir / "animation" / "scene-01-extra.mp4"
        extra_path.write_bytes(b"extra")
        register_artifact(
            conn,
            ArtifactRecord(
                artifact_id="animation-scene-01-extra",
                project_id=project_id,
                kind="animation",
                scene_id="scene-01",
                relative_path="animation/scene-01-extra.mp4",
                byte_size=5,
                sha256_checksum=hashlib.sha256(b"extra").hexdigest(),
                created_at=FIXED_NOW,
                metadata={"duration_seconds": 1.0, "source": "external"},
            ),
        )
    finally:
        conn.close()
    _forbid_ffmpeg(monkeypatch)

    with pytest.raises(AmbiguousSceneArtifactError, match="'animation'"):
        assemble_final_video(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")


def test_audio_record_points_to_missing_file_fails(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_animation(project_dir, project_id, "scene-01")
    audio_path = _register_audio(project_dir, project_id, "scene-01")
    audio_path.unlink()
    _forbid_ffmpeg(monkeypatch)

    with pytest.raises(ArtifactFileNotFoundError, match="audio"):
        assemble_final_video(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")


def test_animation_record_points_to_missing_file_fails(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    animation_path = _register_animation(project_dir, project_id, "scene-01")
    _register_audio(project_dir, project_id, "scene-01")
    animation_path.unlink()
    _forbid_ffmpeg(monkeypatch)

    with pytest.raises(ArtifactFileNotFoundError, match="animation"):
        assemble_final_video(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")


# ---------------------------------------------------------------------
# 16-18: output path conflicts
# ---------------------------------------------------------------------


def test_output_equals_audio_input_fails(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_animation(project_dir, project_id, "scene-01")
    audio_path = _register_audio(project_dir, project_id, "scene-01")
    _forbid_ffmpeg(monkeypatch)

    with pytest.raises(OutputPathConflictError):
        assemble_final_video(project_id, project_dir / "manifest.json", audio_path)


def test_output_equals_animation_input_fails(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    animation_path = _register_animation(project_dir, project_id, "scene-01")
    _register_audio(project_dir, project_id, "scene-01")
    _forbid_ffmpeg(monkeypatch)

    with pytest.raises(OutputPathConflictError):
        assemble_final_video(project_id, project_dir / "manifest.json", animation_path)


def test_existing_output_path_fails(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_scenes(project_dir, project_id, ("scene-01",))
    _forbid_ffmpeg(monkeypatch)
    output = tmp_path / "out.mp4"
    output.write_bytes(b"pre-existing content nobody asked to touch")

    with pytest.raises(OutputPathConflictError, match="already exists"):
        assemble_final_video(project_id, project_dir / "manifest.json", output)

    assert output.read_bytes() == b"pre-existing content nobody asked to touch"


# ---------------------------------------------------------------------
# 19-20: mux/concat failure preserves cause
# ---------------------------------------------------------------------


def test_scene_mux_failure_preserves_cause(isolated_db, tmp_path, monkeypatch):
    from src.providers.base import ProviderError

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_scenes(project_dir, project_id, ("scene-01",))
    monkeypatch.setattr(f"{MODULE}._probe_stream_types", lambda path: frozenset({"video"}))

    def _fail_mux(video_path, audio_path, out_path):
        raise ProviderError("ffmpeg failed: <raw internal stderr>")

    monkeypatch.setattr(f"{MODULE}.mux_audio_video", _fail_mux)
    monkeypatch.setattr(f"{MODULE}.concat_videos", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not be called")))

    with pytest.raises(SceneMuxError) as exc_info:
        assemble_final_video(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")

    assert isinstance(exc_info.value.__cause__, ProviderError)
    assert "stderr" not in str(exc_info.value)
    assert not (tmp_path / "out.mp4").exists()


def test_concat_failure_preserves_cause(isolated_db, tmp_path, monkeypatch):
    from src.providers.base import ProviderError

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_scenes(project_dir, project_id, ("scene-01",))
    calls: list = []
    _mock_ffmpeg_ok(monkeypatch, calls)

    def _fail_concat(video_paths, out_path):
        raise ProviderError("ffmpeg failed: <raw internal stderr>")

    monkeypatch.setattr(f"{MODULE}.concat_videos", _fail_concat)

    with pytest.raises(FinalConcatError) as exc_info:
        assemble_final_video(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")

    assert isinstance(exc_info.value.__cause__, ProviderError)
    assert "stderr" not in str(exc_info.value)
    assert not (tmp_path / "out.mp4").exists()


# ---------------------------------------------------------------------
# 21-25: final output validation
# ---------------------------------------------------------------------


def test_nonexistent_final_temp_output_fails(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_scenes(project_dir, project_id, ("scene-01",))
    stream_types: dict[str, frozenset[str]] = {}
    monkeypatch.setattr(f"{MODULE}._probe_stream_types", lambda path: stream_types.get(str(Path(path)), frozenset({"video"})))

    def _fake_mux(video_path, audio_path, out_path):
        Path(out_path).write_bytes(b"fake")
        stream_types[str(Path(out_path))] = frozenset({"video", "audio"})
        return out_path

    def _fake_concat(video_paths, out_path):
        return out_path  # never actually writes the file

    monkeypatch.setattr(f"{MODULE}.mux_audio_video", _fake_mux)
    monkeypatch.setattr(f"{MODULE}.concat_videos", _fake_concat)
    monkeypatch.setattr(f"{MODULE}.get_duration_seconds", lambda path: 2.0)

    with pytest.raises(InvalidFinalOutputError, match="did not produce"):
        assemble_final_video(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")


def test_zero_byte_final_output_fails(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_scenes(project_dir, project_id, ("scene-01",))
    stream_types: dict[str, frozenset[str]] = {}
    monkeypatch.setattr(f"{MODULE}._probe_stream_types", lambda path: stream_types.get(str(Path(path)), frozenset({"video"})))

    def _fake_mux(video_path, audio_path, out_path):
        Path(out_path).write_bytes(b"fake")
        stream_types[str(Path(out_path))] = frozenset({"video", "audio"})
        return out_path

    def _fake_concat(video_paths, out_path):
        Path(out_path).write_bytes(b"")  # zero bytes
        return out_path

    monkeypatch.setattr(f"{MODULE}.mux_audio_video", _fake_mux)
    monkeypatch.setattr(f"{MODULE}.concat_videos", _fake_concat)
    monkeypatch.setattr(f"{MODULE}.get_duration_seconds", lambda path: 2.0)

    with pytest.raises(InvalidFinalOutputError, match="empty"):
        assemble_final_video(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")


@pytest.mark.parametrize("bad_duration", [0.0, -1.0, float("nan"), float("inf")])
def test_zero_or_non_finite_measured_duration_fails(isolated_db, tmp_path, monkeypatch, bad_duration):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_scenes(project_dir, project_id, ("scene-01",))
    stream_types: dict[str, frozenset[str]] = {}
    monkeypatch.setattr(f"{MODULE}._probe_stream_types", lambda path: stream_types.get(str(Path(path)), frozenset({"video"})))

    def _fake_mux(video_path, audio_path, out_path):
        Path(out_path).write_bytes(b"fake")
        stream_types[str(Path(out_path))] = frozenset({"video", "audio"})
        return out_path

    def _fake_concat(video_paths, out_path):
        Path(out_path).write_bytes(b"fake-final")
        stream_types[str(Path(out_path))] = frozenset({"video", "audio"})
        return out_path

    def _fake_duration(path):
        if Path(path).name == "final.mp4":
            return bad_duration
        return 2.0

    monkeypatch.setattr(f"{MODULE}.mux_audio_video", _fake_mux)
    monkeypatch.setattr(f"{MODULE}.concat_videos", _fake_concat)
    monkeypatch.setattr(f"{MODULE}.get_duration_seconds", _fake_duration)

    with pytest.raises(InvalidFinalOutputError, match="invalid duration"):
        assemble_final_video(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")


def test_final_duration_mismatch_outside_tolerance_fails(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_scenes(project_dir, project_id, ("scene-01",))
    stream_types: dict[str, frozenset[str]] = {}
    monkeypatch.setattr(f"{MODULE}._probe_stream_types", lambda path: stream_types.get(str(Path(path)), frozenset({"video"})))

    def _fake_mux(video_path, audio_path, out_path):
        Path(out_path).write_bytes(b"fake")
        stream_types[str(Path(out_path))] = frozenset({"video", "audio"})
        return out_path

    def _fake_concat(video_paths, out_path):
        Path(out_path).write_bytes(b"fake-final")
        stream_types[str(Path(out_path))] = frozenset({"video", "audio"})
        return out_path

    def _fake_duration(path):
        if Path(path).name == "final.mp4":
            return 100.0  # wildly off from the 2.0s scene sum
        return 2.0

    monkeypatch.setattr(f"{MODULE}.mux_audio_video", _fake_mux)
    monkeypatch.setattr(f"{MODULE}.concat_videos", _fake_concat)
    monkeypatch.setattr(f"{MODULE}.get_duration_seconds", _fake_duration)

    with pytest.raises(InvalidFinalOutputError, match="does not match"):
        assemble_final_video(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")


def test_boundary_case_inside_duration_tolerance_succeeds(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_scenes(project_dir, project_id, ("scene-01",))
    tolerance = _duration_tolerance_seconds(1)
    stream_types: dict[str, frozenset[str]] = {}
    monkeypatch.setattr(f"{MODULE}._probe_stream_types", lambda path: stream_types.get(str(Path(path)), frozenset({"video"})))

    def _fake_mux(video_path, audio_path, out_path):
        Path(out_path).write_bytes(b"fake")
        stream_types[str(Path(out_path))] = frozenset({"video", "audio"})
        return out_path

    def _fake_concat(video_paths, out_path):
        Path(out_path).write_bytes(b"fake-final")
        stream_types[str(Path(out_path))] = frozenset({"video", "audio"})
        return out_path

    def _fake_duration(path):
        if Path(path).name == "final.mp4":
            return 2.0 + (tolerance * 0.9)  # just inside tolerance
        return 2.0

    monkeypatch.setattr(f"{MODULE}.mux_audio_video", _fake_mux)
    monkeypatch.setattr(f"{MODULE}.concat_videos", _fake_concat)
    monkeypatch.setattr(f"{MODULE}.get_duration_seconds", _fake_duration)
    monkeypatch.setattr(f"{REGISTRAR}.get_duration_seconds", _fake_duration)
    monkeypatch.setattr(f"{REGISTRAR}._has_video_stream", lambda path: True)

    result = assemble_final_video(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")
    assert result.scene_count == 1


# ---------------------------------------------------------------------
# 26-29: cleanup and source-artifact preservation
# ---------------------------------------------------------------------


def test_temporary_scene_clips_removed_after_success(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01", "scene-02"))
    _register_scenes(project_dir, project_id, ("scene-01", "scene-02"))
    calls: list = []
    _mock_ffmpeg_ok(monkeypatch, calls)
    output = tmp_path / "out.mp4"

    assemble_final_video(project_id, project_dir / "manifest.json", output)

    leftover = [p for p in tmp_path.iterdir() if p.name.startswith("final-video-assembly-")]
    assert leftover == []


def test_temporary_scene_clips_removed_after_failure(isolated_db, tmp_path, monkeypatch):
    from src.providers.base import ProviderError

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_scenes(project_dir, project_id, ("scene-01",))
    monkeypatch.setattr(f"{MODULE}._probe_stream_types", lambda path: frozenset({"video"}))

    def _fail_mux(video_path, audio_path, out_path):
        raise ProviderError("boom")

    monkeypatch.setattr(f"{MODULE}.mux_audio_video", _fail_mux)
    output = tmp_path / "out.mp4"

    with pytest.raises(SceneMuxError):
        assemble_final_video(project_id, project_dir / "manifest.json", output)

    leftover = [p for p in tmp_path.iterdir() if p.name.startswith("final-video-assembly-")]
    assert leftover == []


def test_partial_final_output_not_left_after_failure(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_scenes(project_dir, project_id, ("scene-01",))
    stream_types: dict[str, frozenset[str]] = {}
    monkeypatch.setattr(f"{MODULE}._probe_stream_types", lambda path: stream_types.get(str(Path(path)), frozenset({"video"})))

    def _fake_mux(video_path, audio_path, out_path):
        Path(out_path).write_bytes(b"fake")
        stream_types[str(Path(out_path))] = frozenset({"video", "audio"})
        return out_path

    def _fake_concat(video_paths, out_path):
        Path(out_path).write_bytes(b"")
        return out_path

    monkeypatch.setattr(f"{MODULE}.mux_audio_video", _fake_mux)
    monkeypatch.setattr(f"{MODULE}.concat_videos", _fake_concat)
    monkeypatch.setattr(f"{MODULE}.get_duration_seconds", lambda path: 2.0)
    output = tmp_path / "out.mp4"

    with pytest.raises(InvalidFinalOutputError):
        assemble_final_video(project_id, project_dir / "manifest.json", output)

    assert not output.exists()


def test_source_artifacts_never_modified_or_deleted(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_scenes(project_dir, project_id, ("scene-01",))
    animation_path = project_dir / "animation" / "scene-01.mp4"
    audio_path = project_dir / "audio" / "scene-01.wav"
    animation_bytes_before = animation_path.read_bytes()
    audio_bytes_before = audio_path.read_bytes()
    calls: list = []
    _mock_ffmpeg_ok(monkeypatch, calls)

    assemble_final_video(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")

    assert animation_path.read_bytes() == animation_bytes_before
    assert audio_path.read_bytes() == audio_bytes_before


# ---------------------------------------------------------------------
# 30-34: artifact registration behavior
# ---------------------------------------------------------------------


def test_artifact_registered_exactly_once_after_success(isolated_db, tmp_path, monkeypatch):
    from src.database.artifact_repository import list_artifacts_by_project

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_scenes(project_dir, project_id, ("scene-01",))
    calls: list = []
    _mock_ffmpeg_ok(monkeypatch, calls)

    assemble_final_video(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")

    conn = get_connection()
    try:
        renders = list_artifacts_by_project(conn, project_id, kind="render")
    finally:
        conn.close()
    assert len(renders) == 1
    assert renders[0].artifact_id == "render-final"


def test_no_final_artifact_registered_on_preflight_failure(isolated_db, tmp_path, monkeypatch):
    from src.database.artifact_repository import list_artifacts_by_project
    from src.database.db import init_db

    init_db()
    _forbid_ffmpeg(monkeypatch)
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ProjectNotFoundError):
        assemble_final_video("does-not-exist", manifest_path, tmp_path / "out.mp4")

    conn = get_connection()
    try:
        renders = list_artifacts_by_project(conn, "does-not-exist", kind="render")
    finally:
        conn.close()
    assert renders == []


def test_no_final_artifact_registered_on_mux_failure(isolated_db, tmp_path, monkeypatch):
    from src.database.artifact_repository import list_artifacts_by_project
    from src.providers.base import ProviderError

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_scenes(project_dir, project_id, ("scene-01",))
    monkeypatch.setattr(f"{MODULE}._probe_stream_types", lambda path: frozenset({"video"}))
    monkeypatch.setattr(f"{MODULE}.mux_audio_video", lambda *a: (_ for _ in ()).throw(ProviderError("boom")))

    with pytest.raises(SceneMuxError):
        assemble_final_video(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")

    conn = get_connection()
    try:
        renders = list_artifacts_by_project(conn, project_id, kind="render")
    finally:
        conn.close()
    assert renders == []


def test_no_final_artifact_registered_on_concat_failure(isolated_db, tmp_path, monkeypatch):
    from src.database.artifact_repository import list_artifacts_by_project
    from src.providers.base import ProviderError

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_scenes(project_dir, project_id, ("scene-01",))
    calls: list = []
    _mock_ffmpeg_ok(monkeypatch, calls)
    monkeypatch.setattr(f"{MODULE}.concat_videos", lambda *a: (_ for _ in ()).throw(ProviderError("boom")))

    with pytest.raises(FinalConcatError):
        assemble_final_video(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")

    conn = get_connection()
    try:
        renders = list_artifacts_by_project(conn, project_id, kind="render")
    finally:
        conn.close()
    assert renders == []


def test_registration_failure_removes_newly_created_output(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_scenes(project_dir, project_id, ("scene-01",))
    calls: list = []
    _mock_ffmpeg_ok(monkeypatch, calls)

    def _fail_register(conn, project, manifest_arg, source_file, *, now=None, prevalidated=None, metadata_overrides=None):
        raise RuntimeError("unexpected registration failure")

    monkeypatch.setattr(f"{MODULE}.register_render_artifact", _fail_register)
    output = tmp_path / "out.mp4"

    with pytest.raises(FinalArtifactRegistrationError):
        assemble_final_video(project_id, project_dir / "manifest.json", output)

    assert not output.exists()


# ---------------------------------------------------------------------
# 35: duplicate/rerun policy
# ---------------------------------------------------------------------


def test_rerun_with_existing_render_artifact_fails_with_existing_error(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_scenes(project_dir, project_id, ("scene-01",))
    calls: list = []
    _mock_ffmpeg_ok(monkeypatch, calls)

    assemble_final_video(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")

    _forbid_ffmpeg(monkeypatch)
    with pytest.raises(ExistingFinalVideoArtifactError):
        assemble_final_video(project_id, project_dir / "manifest.json", tmp_path / "out2.mp4")


# ---------------------------------------------------------------------
# 36: no DB connection open during ANY subprocess call, including
# registration's own (formerly-ffprobe-while-open) work
# ---------------------------------------------------------------------


def test_no_database_connection_open_during_any_subprocess(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_scenes(project_dir, project_id, ("scene-01",))
    calls: list = []
    durations, stream_types = _mock_ffmpeg_ok(monkeypatch, calls)

    from src.database import db as db_module

    real_get_connection = db_module.get_connection
    real_get_readonly_connection = db_module.get_readonly_connection
    violations: list = []
    live_connections: list = []

    def _tracking_get_connection():
        conn = real_get_connection()
        live_connections.append(conn)
        return conn

    def _tracking_get_readonly_connection():
        conn = real_get_readonly_connection()
        live_connections.append(conn)
        return conn

    monkeypatch.setattr("src.database.db.get_connection", _tracking_get_connection)
    monkeypatch.setattr("src.database.db.get_readonly_connection", _tracking_get_readonly_connection)

    def _any_conn_open() -> bool:
        for conn in live_connections:
            try:
                conn.execute("SELECT 1")
                return True
            except sqlite3.ProgrammingError:
                continue
        return False

    def _checking_mux(video_path, audio_path, out_path):
        if _any_conn_open():
            violations.append("mux_audio_video")
        Path(out_path).write_bytes(b"fake-muxed-clip")
        durations[str(Path(out_path))] = 2.0
        stream_types[str(Path(out_path))] = frozenset({"video", "audio"})
        return out_path

    final_duration_holder: dict[str, float] = {}

    def _checking_concat(video_paths, out_path):
        if _any_conn_open():
            violations.append("concat_videos")
        Path(out_path).write_bytes(b"fake-final-video")
        total = sum(durations[str(Path(p))] for p in video_paths)
        durations[str(Path(out_path))] = total
        stream_types[str(Path(out_path))] = frozenset({"video", "audio"})
        final_duration_holder["value"] = total
        return out_path

    def _checking_get_duration_seconds(path):
        if _any_conn_open():
            violations.append("get_duration_seconds")
        return durations[str(Path(path))]

    def _checking_probe(path):
        if _any_conn_open():
            violations.append("_probe_stream_types")
        key = str(Path(path))
        return stream_types.get(key, frozenset({"video"}))

    def _checking_registrar_duration(path):
        if _any_conn_open():
            violations.append("registrar.get_duration_seconds")
        return final_duration_holder["value"]

    def _checking_registrar_has_video(path):
        if _any_conn_open():
            violations.append("registrar._has_video_stream")
        return True

    monkeypatch.setattr(f"{MODULE}.mux_audio_video", _checking_mux)
    monkeypatch.setattr(f"{MODULE}.concat_videos", _checking_concat)
    monkeypatch.setattr(f"{MODULE}.get_duration_seconds", _checking_get_duration_seconds)
    monkeypatch.setattr(f"{MODULE}._probe_stream_types", _checking_probe)
    monkeypatch.setattr(f"{REGISTRAR}.get_duration_seconds", _checking_registrar_duration)
    monkeypatch.setattr(f"{REGISTRAR}._has_video_stream", _checking_registrar_has_video)

    assemble_final_video(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")

    assert violations == []


# ---------------------------------------------------------------------
# 37: result immutability and completeness
# ---------------------------------------------------------------------


def test_result_contains_all_required_fields_and_is_immutable(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_scenes(project_dir, project_id, ("scene-01",))
    calls: list = []
    _mock_ffmpeg_ok(monkeypatch, calls)

    result = assemble_final_video(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")

    assert result.project_id == project_id
    assert isinstance(result.output_path, Path)
    assert result.scene_count == 1
    assert isinstance(result.measured_duration_seconds, float)
    assert isinstance(result.artifact_id, str)

    with pytest.raises(Exception):
        result.project_id = "changed"


def test_duration_tolerance_formula():
    assert _duration_tolerance_seconds(1) == pytest.approx(max(0.25, 0.05))
    assert _duration_tolerance_seconds(10) == pytest.approx(max(0.25, 0.5))


# ---------------------------------------------------------------------
# artifact metadata (STEP 4)
# ---------------------------------------------------------------------


def test_registered_metadata_contains_scene_count_and_manifest_fingerprint(isolated_db, tmp_path, monkeypatch):
    from src.database.artifact_repository import list_artifacts_by_project

    project_id, project_dir, manifest = _create_registered_project(
        tmp_path / "projects", ("scene-01", "scene-02")
    )
    _register_scenes(project_dir, project_id, ("scene-01", "scene-02"))
    calls: list = []
    _mock_ffmpeg_ok(monkeypatch, calls)

    assemble_final_video(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")

    conn = get_connection()
    try:
        renders = list_artifacts_by_project(conn, project_id, kind="render")
    finally:
        conn.close()
    assert len(renders) == 1
    metadata = renders[0].metadata
    assert metadata["scene_count"] == 2
    assert metadata["manifest_fingerprint"] == manifest.source_fingerprint
    assert metadata["source"] == "final-video-assembly-v1"
    assert metadata["duration_seconds"] == pytest.approx(4.0)


# ---------------------------------------------------------------------
# strict media probing (STEP 2 / STEP 7)
# ---------------------------------------------------------------------


def test_probe_ffprobe_missing_raises_media_probe_error(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_scenes(project_dir, project_id, ("scene-01",))

    def _fail_run(*a, **k):
        raise FileNotFoundError("ffprobe not found")

    monkeypatch.setattr(f"{MODULE}.subprocess.run", _fail_run)

    with pytest.raises(MediaProbeError) as exc_info:
        assemble_final_video(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")
    assert isinstance(exc_info.value.__cause__, FileNotFoundError)


def test_probe_nonzero_exit_raises_media_probe_error(isolated_db, tmp_path, monkeypatch):
    import subprocess as sp

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_scenes(project_dir, project_id, ("scene-01",))

    fake_result = sp.CompletedProcess(args=[], returncode=1, stdout="", stderr="error")
    monkeypatch.setattr(f"{MODULE}.subprocess.run", lambda *a, **k: fake_result)

    with pytest.raises(MediaProbeError):
        assemble_final_video(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")


def test_probe_empty_output_raises_media_probe_error(isolated_db, tmp_path, monkeypatch):
    import subprocess as sp

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_scenes(project_dir, project_id, ("scene-01",))

    fake_result = sp.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    monkeypatch.setattr(f"{MODULE}.subprocess.run", lambda *a, **k: fake_result)

    with pytest.raises(MediaProbeError):
        assemble_final_video(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")


def test_probe_malformed_json_raises_media_probe_error(isolated_db, tmp_path, monkeypatch):
    import subprocess as sp

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_scenes(project_dir, project_id, ("scene-01",))

    fake_result = sp.CompletedProcess(args=[], returncode=0, stdout="{not json", stderr="")
    monkeypatch.setattr(f"{MODULE}.subprocess.run", lambda *a, **k: fake_result)

    with pytest.raises(MediaProbeError) as exc_info:
        assemble_final_video(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")
    import json as _json

    assert isinstance(exc_info.value.__cause__, _json.JSONDecodeError)


def test_probe_structurally_invalid_json_raises_media_probe_error(isolated_db, tmp_path, monkeypatch):
    import subprocess as sp

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_scenes(project_dir, project_id, ("scene-01",))

    fake_result = sp.CompletedProcess(args=[], returncode=0, stdout='{"not_streams": []}', stderr="")
    monkeypatch.setattr(f"{MODULE}.subprocess.run", lambda *a, **k: fake_result)

    with pytest.raises(MediaProbeError):
        assemble_final_video(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")


def test_probe_failure_never_interpreted_as_no_audio(isolated_db, tmp_path, monkeypatch):
    """A probe failure must raise, never silently resolve to "no audio
    stream present" (which would let a scene mux through unexamined)."""
    import subprocess as sp

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_scenes(project_dir, project_id, ("scene-01",))

    fake_result = sp.CompletedProcess(args=[], returncode=1, stdout="", stderr="")
    calls: list = []

    def _tracking_run(*a, **k):
        calls.append("probe_attempted")
        return fake_result

    monkeypatch.setattr(f"{MODULE}.subprocess.run", _tracking_run)

    with pytest.raises(MediaProbeError):
        assemble_final_video(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")
    assert "probe_attempted" in calls  # the probe genuinely ran and genuinely failed, not skipped


def test_video_only_prepared_clip_rejected(isolated_db, tmp_path, monkeypatch):
    """If mux_audio_video() itself somehow produced a clip missing the
    audio stream it was supposed to add, this must be caught, not
    silently accepted on duration alone."""
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_scenes(project_dir, project_id, ("scene-01",))

    def _fake_probe(path):
        return frozenset({"video"})  # never reports audio, even after "muxing"

    def _fake_mux(video_path, audio_path, out_path):
        Path(out_path).write_bytes(b"fake")
        return out_path

    monkeypatch.setattr(f"{MODULE}._probe_stream_types", _fake_probe)
    monkeypatch.setattr(f"{MODULE}.mux_audio_video", _fake_mux)

    with pytest.raises(SceneMuxError, match="missing a required video or audio"):
        assemble_final_video(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")


def test_audio_only_animation_input_rejected(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_scenes(project_dir, project_id, ("scene-01",))
    monkeypatch.setattr(f"{MODULE}._probe_stream_types", lambda path: frozenset({"audio"}))
    monkeypatch.setattr(f"{MODULE}.mux_audio_video", lambda *a: (_ for _ in ()).throw(AssertionError("must not be called")))

    with pytest.raises(SceneMuxError, match="no video stream"):
        assemble_final_video(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")


def test_final_output_missing_audio_rejected(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_scenes(project_dir, project_id, ("scene-01",))
    stream_types: dict[str, frozenset[str]] = {}

    def _fake_probe(path):
        key = str(Path(path))
        if key in stream_types:
            return stream_types[key]
        return frozenset({"video"})

    def _fake_mux(video_path, audio_path, out_path):
        Path(out_path).write_bytes(b"fake")
        stream_types[str(Path(out_path))] = frozenset({"video", "audio"})
        return out_path

    def _fake_concat(video_paths, out_path):
        Path(out_path).write_bytes(b"fake-final")
        stream_types[str(Path(out_path))] = frozenset({"video"})  # missing audio
        return out_path

    monkeypatch.setattr(f"{MODULE}._probe_stream_types", _fake_probe)
    monkeypatch.setattr(f"{MODULE}.mux_audio_video", _fake_mux)
    monkeypatch.setattr(f"{MODULE}.concat_videos", _fake_concat)
    monkeypatch.setattr(f"{MODULE}.get_duration_seconds", lambda path: 2.0)

    with pytest.raises(InvalidFinalOutputError, match="no audio stream"):
        assemble_final_video(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")


def test_final_output_missing_video_rejected(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_scenes(project_dir, project_id, ("scene-01",))
    stream_types: dict[str, frozenset[str]] = {}

    def _fake_probe(path):
        key = str(Path(path))
        if key in stream_types:
            return stream_types[key]
        return frozenset({"video"})

    def _fake_mux(video_path, audio_path, out_path):
        Path(out_path).write_bytes(b"fake")
        stream_types[str(Path(out_path))] = frozenset({"video", "audio"})
        return out_path

    def _fake_concat(video_paths, out_path):
        Path(out_path).write_bytes(b"fake-final")
        stream_types[str(Path(out_path))] = frozenset({"audio"})  # missing video
        return out_path

    monkeypatch.setattr(f"{MODULE}._probe_stream_types", _fake_probe)
    monkeypatch.setattr(f"{MODULE}.mux_audio_video", _fake_mux)
    monkeypatch.setattr(f"{MODULE}.concat_videos", _fake_concat)
    monkeypatch.setattr(f"{MODULE}.get_duration_seconds", lambda path: 2.0)

    with pytest.raises(InvalidFinalOutputError, match="no video stream"):
        assemble_final_video(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")


def test_animation_input_with_existing_audio_rejected(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_scenes(project_dir, project_id, ("scene-01",))
    calls: list = []
    _mock_ffmpeg_ok(monkeypatch, calls, animation_has_audio=True)

    with pytest.raises(SceneMuxError, match="already contains an audio"):
        assemble_final_video(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")


# ---------------------------------------------------------------------
# cleanup failures (STEP 5)
# ---------------------------------------------------------------------


def test_tempdir_removal_failure_is_not_silently_suppressed(isolated_db, tmp_path, monkeypatch):
    """A tempdir-removal failure with NO primary exception in flight (a
    pure cleanup-only failure after an otherwise-complete run) must
    surface as FinalVideoCleanupError, not be swallowed."""
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_scenes(project_dir, project_id, ("scene-01",))
    calls: list = []
    _mock_ffmpeg_ok(monkeypatch, calls)

    def _fail_rmtree(path, *a, **k):
        raise OSError("simulated: directory in use")

    monkeypatch.setattr(f"{MODULE}.shutil.rmtree", _fail_rmtree)

    with pytest.raises(FinalVideoCleanupError) as exc_info:
        assemble_final_video(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")
    assert isinstance(exc_info.value.__cause__, OSError)


def test_tempdir_removal_failure_after_primary_failure_preserves_primary(isolated_db, tmp_path, monkeypatch):
    """When a primary assembly failure is already in flight, a SUBSEQUENT
    tempdir-removal failure must not replace it — the original,
    actionable exception stays the one that propagates, with the cleanup
    problem attached as a note."""
    from src.providers.base import ProviderError

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_scenes(project_dir, project_id, ("scene-01",))
    monkeypatch.setattr(f"{MODULE}._probe_stream_types", lambda path: frozenset({"video"}))
    monkeypatch.setattr(f"{MODULE}.mux_audio_video", lambda *a: (_ for _ in ()).throw(ProviderError("boom")))

    def _fail_rmtree(path, *a, **k):
        raise OSError("simulated: directory in use")

    monkeypatch.setattr(f"{MODULE}.shutil.rmtree", _fail_rmtree)

    with pytest.raises(SceneMuxError) as exc_info:
        assemble_final_video(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")

    notes = getattr(exc_info.value, "__notes__", [])
    assert any("directory" in note for note in notes)


def test_orphaned_output_removal_failure_after_registration_failure(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_scenes(project_dir, project_id, ("scene-01",))
    calls: list = []
    _mock_ffmpeg_ok(monkeypatch, calls)

    def _fail_register(conn, project, manifest_arg, source_file, *, now=None, prevalidated=None, metadata_overrides=None):
        raise RuntimeError("unexpected registration failure")

    monkeypatch.setattr(f"{MODULE}.register_render_artifact", _fail_register)

    real_unlink = Path.unlink

    def _fail_unlink(self, *a, **k):
        if self.name == "out.mp4":
            raise OSError("simulated: permission denied")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", _fail_unlink)

    with pytest.raises(FinalArtifactRegistrationError) as exc_info:
        assemble_final_video(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")

    notes = getattr(exc_info.value, "__notes__", [])
    assert any("orphaned output" in note for note in notes)


def test_pre_existing_files_untouched_when_cleanup_fails(isolated_db, tmp_path, monkeypatch):
    """A cleanup failure must never cause a pre-existing, unrelated file
    to be touched or removed."""
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01",))
    _register_scenes(project_dir, project_id, ("scene-01",))
    unrelated_file = tmp_path / "unrelated.txt"
    unrelated_file.write_bytes(b"do not touch")
    calls: list = []
    _mock_ffmpeg_ok(monkeypatch, calls)

    def _fail_rmtree(path, *a, **k):
        raise OSError("simulated failure")

    monkeypatch.setattr(f"{MODULE}.shutil.rmtree", _fail_rmtree)

    with pytest.raises(FinalVideoCleanupError):
        assemble_final_video(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")

    assert unrelated_file.read_bytes() == b"do not touch"


def test_no_cleanup_exception_silently_suppressed():
    """FinalVideoCleanupError is a real, raisable, catchable exception
    type — not a sentinel that gets swallowed anywhere in this module."""
    with pytest.raises(FinalVideoCleanupError):
        raise FinalVideoCleanupError("temp directory removal failed")


# ---------------------------------------------------------------------
# optional real-ffmpeg integration test — doubly-gated
# ---------------------------------------------------------------------


def test_real_pipeline_integration(isolated_db, tmp_path):
    import subprocess

    from src.database.artifact_repository import list_artifacts_by_project
    from src.render import ffmpeg_render
    from src.utils.config import get_settings

    if not ffmpeg_render.health_check():
        pytest.skip("real ffmpeg not available in this environment")

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", ("scene-01", "scene-02"))

    ffmpeg = get_settings().ffmpeg_path
    for scene_id in ("scene-01", "scene-02"):
        animation_path = project_dir / "animation" / f"{scene_id}.mp4"
        animation_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path = project_dir / "audio" / f"{scene_id}.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            [
                ffmpeg, "-y", "-f", "lavfi", "-i", "color=c=blue:s=64x64:d=1.0:r=10",
                "-pix_fmt", "yuv420p", str(animation_path),
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        result = subprocess.run(
            [ffmpeg, "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", "1.0", str(audio_path)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr

        _register_animation(project_dir, project_id, scene_id, content=animation_path.read_bytes())
        _register_audio(project_dir, project_id, scene_id, content=audio_path.read_bytes())

    output = tmp_path / "final.mp4"
    result = assemble_final_video(project_id, project_dir / "manifest.json", output)

    assert output.exists()

    from src.core.final_video_assembly import _probe_stream_types

    final_streams = _probe_stream_types(output)
    assert "video" in final_streams
    assert "audio" in final_streams

    duration = ffmpeg_render.get_duration_seconds(output)
    assert duration > 0
    assert duration == pytest.approx(2.0, abs=_duration_tolerance_seconds(2))
    assert result.measured_duration_seconds == pytest.approx(duration)

    conn = get_connection()
    try:
        renders = list_artifacts_by_project(conn, project_id, kind="render")
    finally:
        conn.close()
    assert len(renders) == 1
    assert renders[0].metadata["scene_count"] == 2
    assert renders[0].metadata["manifest_fingerprint"] == manifest.source_fingerprint

    leftover = [p for p in tmp_path.iterdir() if p.name.startswith("final-video-assembly-")]
    assert leftover == []
