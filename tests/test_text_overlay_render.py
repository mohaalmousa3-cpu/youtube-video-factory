"""Tests for src/core/text_overlay_render.py — TEXT OVERLAY RENDERER V1.
FFmpeg helpers (subprocess.run / _probe_stream_types / _probe_duration_seconds)
are mocked at their point of import in this module for every test except
the one doubly-gated real-ffmpeg integration test at the bottom. Same
isolated_db / _create_registered_project pattern as
tests/test_final_video_assembly.py, since render_text_overlays() itself
owns its own short-lived SQLite connections."""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.core.text_overlay_render import (
    ArtifactFileNotFoundError,
    ExistingOverlayArtifactError,
    InvalidManifestError,
    InvalidOverlayOutputError,
    ManifestNotFoundError,
    NoOverlaysToRenderError,
    OutputPathConflictError,
    OverlayDrawError,
    OverlayFontNotFoundError,
    OverlayRegistrationError,
    OverlayRenderCleanupError,
    OverlayTextTooLongError,
    OverlayTimingOutOfRangeError,
    ProjectNotFoundError,
    SourceRenderNotFoundError,
    TextOverlayRenderResult,
    UnsupportedOverlayStyleError,
    _MAX_OVERLAY_TEXT_LENGTH,
    render_text_overlays,
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
MODULE = "src.core.text_overlay_render"


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


def _create_registered_project(
    projects_root: Path,
    scene_ids: tuple[str, ...] = ("scene-01", "scene-02"),
    story_id: str = "why-we-care-what-people-think",
    overlays_by_scene: dict[str, list[dict]] | None = None,
):
    manifest = build_video_manifest(
        _story_input_dict(story_id),
        _scene_plan_dict(scene_ids, overlays_by_scene),
        get_channel_policy(),
        created_at=FIXED_NOW,
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


def _register_render_artifact(project_dir, project_id, content=b"fake-render-bytes") -> Path:
    path = project_dir / "render" / "final.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    checksum = hashlib.sha256(content).hexdigest()
    conn = get_connection()
    try:
        register_artifact(
            conn,
            ArtifactRecord(
                artifact_id="render-final",
                project_id=project_id,
                kind="render",
                scene_id=None,
                relative_path="render/final.mp4",
                byte_size=len(content),
                sha256_checksum=checksum,
                created_at=FIXED_NOW,
                metadata={"duration_seconds": 4.0, "source": "final-video-assembly-v1"},
            ),
        )
    finally:
        conn.close()
    return path


def _mock_ffmpeg_ok(monkeypatch, source_duration=4.0):
    """Mocks the ffprobe helpers AND the raw subprocess.run() ffmpeg
    drawtext invocation with a shared, internally-consistent fake result:
    the mocked ffmpeg call writes a fake output file whose (mocked) probed
    duration matches source_duration exactly (drawtext never changes
    timing)."""
    import subprocess as sp

    calls: list = []

    def _fake_probe_stream_types(path):
        calls.append(("probe_streams", Path(path)))
        return frozenset({"video", "audio"})

    def _fake_probe_duration(path):
        calls.append(("probe_duration", Path(path)))
        return source_duration

    def _fake_run(args, **kwargs):
        calls.append(("ffmpeg_run", args))
        out_path = Path(args[-1])
        out_path.write_bytes(b"fake-drawtext-output")
        return sp.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(f"{MODULE}._probe_stream_types", _fake_probe_stream_types)
    monkeypatch.setattr(f"{MODULE}._probe_duration_seconds", _fake_probe_duration)
    monkeypatch.setattr(f"{MODULE}.subprocess.run", _fake_run)
    monkeypatch.setattr(f"{MODULE}._FONT_PATH", Path(__file__))  # any real, existing local file

    REGISTRAR = "src.core.overlay_artifact_registrar"
    monkeypatch.setattr(f"{REGISTRAR}._probe_stream_types", lambda path: frozenset({"video", "audio"}))
    monkeypatch.setattr(f"{REGISTRAR}._probe_duration_seconds", lambda path: source_duration)
    return calls


def _forbid_ffmpeg(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("must not be called")

    monkeypatch.setattr(f"{MODULE}._probe_stream_types", _boom)
    monkeypatch.setattr(f"{MODULE}.subprocess.run", _boom)


# ---------------------------------------------------------------------
# success
# ---------------------------------------------------------------------


def test_single_overlay_renders_and_registers(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(
        tmp_path / "projects", overlays_by_scene={"scene-01": [_overlay_dict()]}
    )
    _register_render_artifact(project_dir, project_id)
    _mock_ffmpeg_ok(monkeypatch)
    output = tmp_path / "overlay-final.mp4"

    result = render_text_overlays(project_id, project_dir / "manifest.json", output)

    assert result.project_id == project_id
    assert result.output_path == output.resolve()
    assert result.overlay_count == 1
    assert result.measured_duration_seconds == pytest.approx(4.0)
    assert result.artifact_id == "overlay-render-final"
    assert output.exists()


def test_multiple_overlays_across_scenes_flattened_in_order(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(
        tmp_path / "projects",
        overlays_by_scene={
            "scene-01": [_overlay_dict(text="First"), _overlay_dict(text="Second", style_id="top_label", position="top")],
            "scene-02": [_overlay_dict(text="Third", style_id="center_emphasis", position="center")],
        },
    )
    _register_render_artifact(project_dir, project_id)
    calls = _mock_ffmpeg_ok(monkeypatch)
    output = tmp_path / "overlay-final.mp4"

    result = render_text_overlays(project_id, project_dir / "manifest.json", output)

    assert result.overlay_count == 3
    ffmpeg_call = next(c for c in calls if c[0] == "ffmpeg_run")
    vf_index = ffmpeg_call[1].index("-vf") + 1
    vf = ffmpeg_call[1][vf_index]
    assert vf.count("drawtext=") == 3


# ---------------------------------------------------------------------
# preflight failures before FFmpeg
# ---------------------------------------------------------------------


def test_missing_project_fails_before_ffmpeg(isolated_db, tmp_path, monkeypatch):
    from src.database.db import init_db

    init_db()
    _forbid_ffmpeg(monkeypatch)
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ProjectNotFoundError):
        render_text_overlays("does-not-exist", manifest_path, tmp_path / "out.mp4")


def test_missing_manifest_fails_before_ffmpeg(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _forbid_ffmpeg(monkeypatch)

    with pytest.raises(ManifestNotFoundError):
        render_text_overlays(project_id, tmp_path / "does-not-exist.json", tmp_path / "out.mp4")


def test_invalid_json_manifest_fails(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _forbid_ffmpeg(monkeypatch)
    bad_manifest = tmp_path / "bad.json"
    bad_manifest.write_text("not valid json", encoding="utf-8")

    with pytest.raises(InvalidManifestError):
        render_text_overlays(project_id, bad_manifest, tmp_path / "out.mp4")


def test_no_overlays_fails_before_ffmpeg(isolated_db, tmp_path, monkeypatch):
    """V1 consumes only explicit text_overlays — an empty collection
    across every scene must fail with NoOverlaysToRenderError, before any
    ffmpeg work, never by deriving overlay text from narration_text."""
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_render_artifact(project_dir, project_id)
    _forbid_ffmpeg(monkeypatch)

    with pytest.raises(NoOverlaysToRenderError):
        render_text_overlays(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")


def test_missing_render_artifact_fails_before_ffmpeg(isolated_db, tmp_path, monkeypatch):
    """V1 never accepts an arbitrary source MP4 — it requires a
    registered "render" artifact to already exist."""
    project_id, project_dir, manifest = _create_registered_project(
        tmp_path / "projects", overlays_by_scene={"scene-01": [_overlay_dict()]}
    )
    _forbid_ffmpeg(monkeypatch)

    with pytest.raises(SourceRenderNotFoundError):
        render_text_overlays(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")


def test_render_artifact_file_missing_on_disk_fails(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(
        tmp_path / "projects", overlays_by_scene={"scene-01": [_overlay_dict()]}
    )
    render_path = _register_render_artifact(project_dir, project_id)
    render_path.unlink()
    _forbid_ffmpeg(monkeypatch)

    with pytest.raises(ArtifactFileNotFoundError):
        render_text_overlays(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")


def test_existing_output_path_fails(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(
        tmp_path / "projects", overlays_by_scene={"scene-01": [_overlay_dict()]}
    )
    _register_render_artifact(project_dir, project_id)
    _forbid_ffmpeg(monkeypatch)
    output = tmp_path / "out.mp4"
    output.write_bytes(b"pre-existing content nobody asked to touch")

    with pytest.raises(OutputPathConflictError, match="already exists"):
        render_text_overlays(project_id, project_dir / "manifest.json", output)

    assert output.read_bytes() == b"pre-existing content nobody asked to touch"


def test_output_equals_render_source_fails(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(
        tmp_path / "projects", overlays_by_scene={"scene-01": [_overlay_dict()]}
    )
    render_path = _register_render_artifact(project_dir, project_id)
    _forbid_ffmpeg(monkeypatch)

    with pytest.raises(OutputPathConflictError):
        render_text_overlays(project_id, project_dir / "manifest.json", render_path)


def test_rerun_with_existing_overlay_artifact_fails(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(
        tmp_path / "projects", overlays_by_scene={"scene-01": [_overlay_dict()]}
    )
    _register_render_artifact(project_dir, project_id)
    _mock_ffmpeg_ok(monkeypatch)

    render_text_overlays(project_id, project_dir / "manifest.json", tmp_path / "out1.mp4")

    _forbid_ffmpeg(monkeypatch)
    with pytest.raises(ExistingOverlayArtifactError):
        render_text_overlays(project_id, project_dir / "manifest.json", tmp_path / "out2.mp4")


# ---------------------------------------------------------------------
# overlay-level validation (style, text length, timing)
# ---------------------------------------------------------------------


def test_unsupported_style_id_rejected(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(
        tmp_path / "projects",
        overlays_by_scene={"scene-01": [_overlay_dict(style_id="totally-unknown-style")]},
    )
    _register_render_artifact(project_dir, project_id)
    _mock_ffmpeg_ok(monkeypatch)

    with pytest.raises(UnsupportedOverlayStyleError):
        render_text_overlays(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")


def test_style_id_position_mismatch_rejected(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(
        tmp_path / "projects",
        overlays_by_scene={"scene-01": [_overlay_dict(style_id="lower_third_primary", position="top")]},
    )
    _register_render_artifact(project_dir, project_id)
    _mock_ffmpeg_ok(monkeypatch)

    with pytest.raises(UnsupportedOverlayStyleError):
        render_text_overlays(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")


def test_text_too_long_rejected(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(
        tmp_path / "projects",
        overlays_by_scene={"scene-01": [_overlay_dict(text="x" * (_MAX_OVERLAY_TEXT_LENGTH + 1))]},
    )
    _register_render_artifact(project_dir, project_id)
    _mock_ffmpeg_ok(monkeypatch)

    with pytest.raises(OverlayTextTooLongError):
        render_text_overlays(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")


def test_end_seconds_beyond_source_duration_rejected(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(
        tmp_path / "projects",
        overlays_by_scene={"scene-01": [_overlay_dict(start_seconds=0.0, end_seconds=999.0)]},
    )
    _register_render_artifact(project_dir, project_id)
    _mock_ffmpeg_ok(monkeypatch, source_duration=4.0)

    with pytest.raises(OverlayTimingOutOfRangeError):
        render_text_overlays(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")


def test_negative_start_seconds_rejected_defensively():
    """Unreachable via a real manifest — TextOverlay's own model_validator
    already forbids start_seconds < 0 at construction (see
    tests/test_manifest_models.py::test_text_overlay_rejects_negative_start).
    Proven directly against the internal validator, same defensive-check
    style as test_duplicate_scene_ids_rejected_defensively in
    tests/test_final_video_assembly.py."""
    from src.core.text_overlay_render import _FlattenedOverlay, _validate_and_build_filters
    from types import SimpleNamespace

    fake_overlay = SimpleNamespace(
        text="x", position="lower_third", style_id="lower_third_primary",
        start_seconds=-1.0, end_seconds=None,
    )
    flat = _FlattenedOverlay(scene_id="scene-01", index=0, overlay=fake_overlay)

    with pytest.raises(OverlayTimingOutOfRangeError, match="start_seconds must be >= 0"):
        _validate_and_build_filters((flat,), 4.0, Path("."))


def test_start_after_defaulted_end_rejected(isolated_db, tmp_path, monkeypatch):
    """A reachable timing-range violation: start_seconds set well past the
    source render's actual duration, with end_seconds left None (so it
    defaults to that same duration) — after defaulting, start > end."""
    project_id, project_dir, manifest = _create_registered_project(
        tmp_path / "projects",
        overlays_by_scene={"scene-01": [_overlay_dict(start_seconds=2.0, end_seconds=None)]},
    )
    _register_render_artifact(project_dir, project_id)
    _mock_ffmpeg_ok(monkeypatch, source_duration=1.0)

    with pytest.raises(OverlayTimingOutOfRangeError):
        render_text_overlays(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")


def test_missing_timing_defaults_to_full_duration(isolated_db, tmp_path, monkeypatch):
    """start_seconds/end_seconds both None must default to [0, source
    duration] rather than being rejected."""
    project_id, project_dir, manifest = _create_registered_project(
        tmp_path / "projects",
        overlays_by_scene={"scene-01": [_overlay_dict(start_seconds=None, end_seconds=None)]},
    )
    _register_render_artifact(project_dir, project_id)
    _mock_ffmpeg_ok(monkeypatch, source_duration=4.0)

    result = render_text_overlays(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")
    assert result.overlay_count == 1


def test_missing_font_file_rejected(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(
        tmp_path / "projects", overlays_by_scene={"scene-01": [_overlay_dict()]}
    )
    _register_render_artifact(project_dir, project_id)
    _mock_ffmpeg_ok(monkeypatch)
    monkeypatch.setattr(f"{MODULE}._FONT_PATH", tmp_path / "does-not-exist-font.ttf")

    with pytest.raises(OverlayFontNotFoundError):
        render_text_overlays(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")


# ---------------------------------------------------------------------
# ffmpeg / output validation failures
# ---------------------------------------------------------------------


def test_ffmpeg_nonzero_exit_raises_draw_error(isolated_db, tmp_path, monkeypatch):
    import subprocess as sp

    project_id, project_dir, manifest = _create_registered_project(
        tmp_path / "projects", overlays_by_scene={"scene-01": [_overlay_dict()]}
    )
    _register_render_artifact(project_dir, project_id)
    monkeypatch.setattr(f"{MODULE}._probe_stream_types", lambda path: frozenset({"video", "audio"}))
    monkeypatch.setattr(f"{MODULE}._probe_duration_seconds", lambda path: 4.0)
    monkeypatch.setattr(f"{MODULE}._FONT_PATH", Path(__file__))
    monkeypatch.setattr(
        f"{MODULE}.subprocess.run",
        lambda args, **k: sp.CompletedProcess(args=args, returncode=1, stdout="", stderr="raw ffmpeg stderr"),
    )

    with pytest.raises(OverlayDrawError) as exc_info:
        render_text_overlays(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")
    assert "raw ffmpeg stderr" not in str(exc_info.value)
    assert not (tmp_path / "out.mp4").exists()


def test_output_missing_audio_stream_rejected(isolated_db, tmp_path, monkeypatch):
    import subprocess as sp

    project_id, project_dir, manifest = _create_registered_project(
        tmp_path / "projects", overlays_by_scene={"scene-01": [_overlay_dict()]}
    )
    _register_render_artifact(project_dir, project_id)
    stream_types: dict[str, frozenset[str]] = {}

    def _fake_probe(path):
        key = str(Path(path))
        return stream_types.get(key, frozenset({"video", "audio"}))

    def _fake_run(args, **k):
        out_path = Path(args[-1])
        out_path.write_bytes(b"fake")
        stream_types[str(out_path)] = frozenset({"video"})  # missing audio
        return sp.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(f"{MODULE}._probe_stream_types", _fake_probe)
    monkeypatch.setattr(f"{MODULE}._probe_duration_seconds", lambda path: 4.0)
    monkeypatch.setattr(f"{MODULE}._FONT_PATH", Path(__file__))
    monkeypatch.setattr(f"{MODULE}.subprocess.run", _fake_run)

    with pytest.raises(InvalidOverlayOutputError, match="no audio stream"):
        render_text_overlays(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")


def test_output_duration_mismatch_rejected(isolated_db, tmp_path, monkeypatch):
    import subprocess as sp

    project_id, project_dir, manifest = _create_registered_project(
        tmp_path / "projects", overlays_by_scene={"scene-01": [_overlay_dict()]}
    )
    _register_render_artifact(project_dir, project_id)
    duration_by_path: dict[str, float] = {}

    def _fake_duration(path):
        key = str(Path(path))
        if key in duration_by_path:
            return duration_by_path[key]
        return 4.0

    def _fake_run(args, **k):
        out_path = Path(args[-1])
        out_path.write_bytes(b"fake")
        duration_by_path[str(out_path)] = 999.0  # wildly different from the 4.0s source
        return sp.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(f"{MODULE}._probe_stream_types", lambda path: frozenset({"video", "audio"}))
    monkeypatch.setattr(f"{MODULE}._probe_duration_seconds", _fake_duration)
    monkeypatch.setattr(f"{MODULE}._FONT_PATH", Path(__file__))
    monkeypatch.setattr(f"{MODULE}.subprocess.run", _fake_run)

    with pytest.raises(InvalidOverlayOutputError, match="does not match"):
        render_text_overlays(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")


# ---------------------------------------------------------------------
# cleanup / registration
# ---------------------------------------------------------------------


def test_no_overlay_artifact_registered_on_preflight_failure(isolated_db, tmp_path, monkeypatch):
    from src.database.artifact_repository import list_artifacts_by_project

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_render_artifact(project_dir, project_id)
    _forbid_ffmpeg(monkeypatch)

    with pytest.raises(NoOverlaysToRenderError):
        render_text_overlays(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")

    conn = get_connection()
    try:
        overlays = list_artifacts_by_project(conn, project_id, kind="overlay_render")
    finally:
        conn.close()
    assert overlays == []


def test_registration_failure_removes_newly_created_output(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(
        tmp_path / "projects", overlays_by_scene={"scene-01": [_overlay_dict()]}
    )
    _register_render_artifact(project_dir, project_id)
    _mock_ffmpeg_ok(monkeypatch)

    def _fail_register(conn, project, manifest_arg, source_file, **kwargs):
        raise RuntimeError("unexpected registration failure")

    monkeypatch.setattr(f"{MODULE}.register_overlay_render_artifact", _fail_register)
    output = tmp_path / "out.mp4"

    with pytest.raises(OverlayRegistrationError):
        render_text_overlays(project_id, project_dir / "manifest.json", output)

    assert not output.exists()


def test_temporary_directory_removed_after_success(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(
        tmp_path / "projects", overlays_by_scene={"scene-01": [_overlay_dict()]}
    )
    _register_render_artifact(project_dir, project_id)
    _mock_ffmpeg_ok(monkeypatch)
    output = tmp_path / "out.mp4"

    render_text_overlays(project_id, project_dir / "manifest.json", output)

    leftover = [p for p in tmp_path.iterdir() if p.name.startswith("text-overlay-render-")]
    assert leftover == []


def test_render_artifact_never_modified_or_deleted(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(
        tmp_path / "projects", overlays_by_scene={"scene-01": [_overlay_dict()]}
    )
    render_path = _register_render_artifact(project_dir, project_id)
    bytes_before = render_path.read_bytes()
    _mock_ffmpeg_ok(monkeypatch)

    render_text_overlays(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")

    assert render_path.read_bytes() == bytes_before

    from src.database.artifact_repository import list_artifacts_by_project

    conn = get_connection()
    try:
        renders = list_artifacts_by_project(conn, project_id, kind="render")
    finally:
        conn.close()
    assert len(renders) == 1
    assert renders[0].artifact_id == "render-final"


# ---------------------------------------------------------------------
# DB lifecycle
# ---------------------------------------------------------------------


def test_no_database_connection_open_during_any_subprocess(isolated_db, tmp_path, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(
        tmp_path / "projects", overlays_by_scene={"scene-01": [_overlay_dict()]}
    )
    _register_render_artifact(project_dir, project_id)

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

    import subprocess as sp

    def _checking_probe_streams(path):
        if _any_conn_open():
            violations.append("_probe_stream_types")
        return frozenset({"video", "audio"})

    def _checking_probe_duration(path):
        if _any_conn_open():
            violations.append("_probe_duration_seconds")
        return 4.0

    def _checking_run(args, **k):
        if _any_conn_open():
            violations.append("subprocess.run")
        out_path = Path(args[-1])
        out_path.write_bytes(b"fake")
        return sp.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(f"{MODULE}._probe_stream_types", _checking_probe_streams)
    monkeypatch.setattr(f"{MODULE}._probe_duration_seconds", _checking_probe_duration)
    monkeypatch.setattr(f"{MODULE}.subprocess.run", _checking_run)
    monkeypatch.setattr(f"{MODULE}._FONT_PATH", Path(__file__))

    REGISTRAR = "src.core.overlay_artifact_registrar"

    def _checking_registrar_probe_streams(path):
        if _any_conn_open():
            violations.append("registrar._probe_stream_types")
        return frozenset({"video", "audio"})

    def _checking_registrar_probe_duration(path):
        if _any_conn_open():
            violations.append("registrar._probe_duration_seconds")
        return 4.0

    monkeypatch.setattr(f"{REGISTRAR}._probe_stream_types", _checking_registrar_probe_streams)
    monkeypatch.setattr(f"{REGISTRAR}._probe_duration_seconds", _checking_registrar_probe_duration)

    render_text_overlays(project_id, project_dir / "manifest.json", tmp_path / "out.mp4")

    assert violations == []


# ---------------------------------------------------------------------
# optional real-ffmpeg integration test — doubly-gated
# ---------------------------------------------------------------------


def test_real_pipeline_integration(isolated_db, tmp_path):
    import socket
    import subprocess

    from src.database.artifact_repository import list_artifacts_by_project
    from src.render import ffmpeg_render
    from src.utils.config import get_settings

    if not ffmpeg_render.health_check():
        pytest.skip("real ffmpeg not available in this environment")
    if not Path(r"C:\Windows\Fonts\arial.ttf").exists():
        pytest.skip("required overlay font not available in this environment")

    project_id, project_dir, manifest = _create_registered_project(
        tmp_path / "projects",
        overlays_by_scene={
            "scene-01": [_overlay_dict(text="Real Overlay Test", start_seconds=0.0, end_seconds=1.0)],
        },
    )

    ffmpeg = get_settings().ffmpeg_path
    render_path = project_dir / "render" / "final.mp4"
    render_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            ffmpeg, "-y", "-f", "lavfi", "-i", "color=c=blue:s=320x180:d=2.0:r=10",
            "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", "2.0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(render_path),
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    from src.database.artifact_repository import register_artifact
    from src.models.artifact import ArtifactRecord

    checksum = hashlib.sha256(render_path.read_bytes()).hexdigest()
    conn = get_connection()
    try:
        register_artifact(
            conn,
            ArtifactRecord(
                artifact_id="render-final",
                project_id=project_id,
                kind="render",
                scene_id=None,
                relative_path="render/final.mp4",
                byte_size=render_path.stat().st_size,
                sha256_checksum=checksum,
                created_at=FIXED_NOW,
                metadata={"duration_seconds": 2.0, "source": "final-video-assembly-v1"},
            ),
        )
    finally:
        conn.close()

    output = tmp_path / "overlay-final.mp4"

    def _no_network(*a, **k):
        raise AssertionError("no network socket access expected during text overlay rendering")

    import pytest as _pytest

    mp = _pytest.MonkeyPatch()
    mp.setattr(socket, "socket", _no_network)
    try:
        result = render_text_overlays(project_id, project_dir / "manifest.json", output)
    finally:
        mp.undo()

    assert output.exists()

    from src.core.text_overlay_render import _probe_stream_types

    final_streams = _probe_stream_types(output)
    assert "video" in final_streams
    assert "audio" in final_streams

    duration = ffmpeg_render.get_duration_seconds(output)
    assert duration == pytest.approx(2.0, abs=0.3)
    assert result.measured_duration_seconds == pytest.approx(duration)

    conn = get_connection()
    try:
        overlays = list_artifacts_by_project(conn, project_id, kind="overlay_render")
    finally:
        conn.close()
    assert len(overlays) == 1
    assert overlays[0].metadata["overlay_count"] == 1
    assert overlays[0].metadata["source_render_artifact_id"] == "render-final"

    leftover = [p for p in tmp_path.iterdir() if p.name.startswith("text-overlay-render-")]
    assert leftover == []
