"""Tests for src/core/overlay_artifact_registrar.py: TEXT OVERLAY
RENDERER V1's registration-first, project-level Overlay Render Artifact
Registration. Same isolated_db / build-manifest-then-register-project
pattern as tests/test_render_artifact_registrar.py. Uses the REAL local
ffmpeg/ffprobe binary (no mocking) — no network, no provider."""
from __future__ import annotations

import hashlib
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.core.manifest_builder import build_video_manifest
from src.core.manifest_store import save_manifest
from src.core.overlay_artifact_registrar import (
    OverlayArtifactCleanupError,
    OverlayArtifactRegistrationError,
    OverlaySourceMismatchError,
    OverlaySourceValidationError,
    PrevalidatedOverlaySource,
    prevalidate_overlay_source,
    register_overlay_render_artifact,
)
from src.core.project_state_machine import create_initial_project, initial_transition_for
from src.database.artifact_repository import list_artifacts_by_project
from src.database.db import SCHEMA
from src.utils.channel_config import get_channel_policy

FIXED_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture()
def conn():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA)
    yield connection
    connection.close()


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


def _scene_plan_dict(scene_ids: tuple[str, ...] = ("scene-01",)) -> dict:
    return dict(
        scenes=tuple(
            dict(
                scene_id=scene_id,
                sequence=i,
                narration_text="Why does being left out sting so much?",
                scene_type="establishing",
                narrative_beat="hook",
                visual_brief="Stickman character alone on a quiet street corner at dusk.",
                motion_mode="in",
                approval_state="approved",
            )
            for i, scene_id in enumerate(scene_ids, start=1)
        ),
        role_outfits=(),
    )


def _registered_project(conn, tmp_path: Path, scene_ids: tuple[str, ...] = ("scene-01",)):
    from src.database.project_repository import create_project

    manifest = build_video_manifest(
        _story_input_dict(), _scene_plan_dict(scene_ids), get_channel_policy(), created_at=FIXED_NOW
    )
    project_dir = tmp_path / "projects" / manifest.project_id
    manifest_path = project_dir / "manifest.json"
    save_manifest(manifest, manifest_path)

    project = create_initial_project(manifest_path, manifest, now=FIXED_NOW)
    transition = initial_transition_for(project)
    create_project(conn, project, transition)
    return project, manifest, project_dir


def _real_mp4(path: Path, duration_seconds: float = 1.0, color: str = "red") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from src.utils.config import get_settings

    ffmpeg = get_settings().ffmpeg_path
    result = subprocess.run(
        [
            ffmpeg, "-y",
            "-f", "lavfi", "-i", f"color=c={color}:size=64x64:rate=5:duration={duration_seconds}",
            "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
            "-t", str(duration_seconds),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def _real_video_only_mp4(path: Path, duration_seconds: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from src.utils.config import get_settings

    ffmpeg = get_settings().ffmpeg_path
    result = subprocess.run(
        [
            ffmpeg, "-y", "-f", "lavfi", "-i", f"color=c=red:size=64x64:rate=5:duration={duration_seconds}",
            "-t", str(duration_seconds), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------


def test_valid_overlay_registration(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.mp4"
    _real_mp4(source, duration_seconds=1.0)

    result = register_overlay_render_artifact(conn, project, manifest, source, now=FIXED_NOW)

    assert result.ok is True
    assert result.idempotent is False
    assert result.copied is True
    assert result.artifact_id == "overlay-render-final"
    assert result.relative_path == "overlay/final.mp4"
    assert result.duration_seconds == pytest.approx(1.0, abs=0.2)

    destination = project_dir / "overlay" / "final.mp4"
    assert destination.exists()
    assert destination.read_bytes() == source.read_bytes()

    stored = list_artifacts_by_project(conn, project.project_id, kind="overlay_render")
    assert len(stored) == 1
    assert stored[0].artifact_id == "overlay-render-final"
    assert stored[0].kind == "overlay_render"
    assert stored[0].scene_id is None
    assert stored[0].metadata["source"] == "text-overlay-renderer-v1"


# ---------------------------------------------------------------------
# Idempotency / rejection
# ---------------------------------------------------------------------


def test_identical_repeat_is_a_no_write_idempotent_success(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.mp4"
    _real_mp4(source)

    first = register_overlay_render_artifact(conn, project, manifest, source, now=FIXED_NOW)
    assert first.ok and not first.idempotent

    second = register_overlay_render_artifact(conn, project, manifest, source, now=FIXED_NOW)

    assert second.ok is True
    assert second.idempotent is True
    assert second.copied is False

    stored = list_artifacts_by_project(conn, project.project_id, kind="overlay_render")
    assert len(stored) == 1


def test_different_second_source_is_rejected_and_leaves_original_unchanged(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source_a = tmp_path / "a.mp4"
    _real_mp4(source_a, color="red")
    first = register_overlay_render_artifact(conn, project, manifest, source_a, now=FIXED_NOW)
    assert first.ok

    source_b = tmp_path / "b.mp4"
    _real_mp4(source_b, color="blue")
    second = register_overlay_render_artifact(conn, project, manifest, source_b, now=FIXED_NOW)

    assert second.ok is False
    assert "already registered" in second.reasons[0]
    stored = list_artifacts_by_project(conn, project.project_id, kind="overlay_render")
    assert len(stored) == 1


def test_video_only_source_rejected_before_any_copy(conn, tmp_path):
    """An overlay-render output must keep both video AND audio (the
    source render always has both) — video-only must be rejected even
    though duration measurement alone would succeed."""
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    video_only = tmp_path / "video_only.mp4"
    _real_video_only_mp4(video_only)

    result = register_overlay_render_artifact(conn, project, manifest, video_only, now=FIXED_NOW)

    assert result.ok is False
    assert "audio stream" in result.reasons[0]
    assert not (project_dir / "overlay").exists()
    assert list_artifacts_by_project(conn, project.project_id, kind="overlay_render") == []


def test_missing_source_file_causes_zero_writes(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)

    result = register_overlay_render_artifact(
        conn, project, manifest, tmp_path / "does-not-exist.mp4", now=FIXED_NOW
    )

    assert result.ok is False
    assert "does not exist" in result.reasons[0]
    assert not (project_dir / "overlay").exists()


def test_mismatched_manifest_project_id_raises_and_causes_zero_writes(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.mp4"
    _real_mp4(source)
    mismatched_manifest = manifest.model_copy(update={"project_id": "proj-does-not-match"})

    with pytest.raises(OverlayArtifactRegistrationError):
        register_overlay_render_artifact(conn, project, mismatched_manifest, source, now=FIXED_NOW)

    assert list_artifacts_by_project(conn, project.project_id, kind="overlay_render") == []


# ---------------------------------------------------------------------
# prevalidate_overlay_source() / prevalidated= — same contract as
# render_artifact_registrar's own prevalidation flow
# ---------------------------------------------------------------------


def test_prevalidate_overlay_source_succeeds_on_real_video(tmp_path):
    source = tmp_path / "source.mp4"
    _real_mp4(source, duration_seconds=1.0)

    result = prevalidate_overlay_source(source)

    assert isinstance(result, PrevalidatedOverlaySource)
    assert result.source_path == source.resolve()
    assert result.duration_seconds == pytest.approx(1.0, abs=0.2)


def test_prevalidate_overlay_source_video_only_fails(tmp_path):
    source = tmp_path / "video_only.mp4"
    _real_video_only_mp4(source)

    with pytest.raises(OverlaySourceValidationError, match="audio stream"):
        prevalidate_overlay_source(source)


def test_register_with_prevalidated_result_succeeds_without_reprobing(conn, tmp_path, monkeypatch):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.mp4"
    _real_mp4(source, duration_seconds=1.0)
    prevalidated = prevalidate_overlay_source(source)

    def _boom(*a, **k):
        raise AssertionError("must not re-probe when prevalidated is supplied")

    monkeypatch.setattr("src.core.overlay_artifact_registrar._probe_stream_types", _boom)
    monkeypatch.setattr("src.core.overlay_artifact_registrar._probe_duration_seconds", _boom)

    result = register_overlay_render_artifact(
        conn, project, manifest, source, now=FIXED_NOW, prevalidated=prevalidated
    )

    assert result.ok is True


def test_register_with_prevalidated_from_different_path_rejected(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source_a = tmp_path / "a.mp4"
    _real_mp4(source_a, duration_seconds=1.0)
    source_b = tmp_path / "b.mp4"
    source_b.write_bytes(source_a.read_bytes())

    prevalidated = prevalidate_overlay_source(source_a)

    with pytest.raises(OverlaySourceMismatchError, match="does not resolve to the same file"):
        register_overlay_render_artifact(conn, project, manifest, source_b, now=FIXED_NOW, prevalidated=prevalidated)

    assert list_artifacts_by_project(conn, project.project_id, kind="overlay_render") == []


def test_register_with_stale_prevalidated_checksum_mismatch_raises(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.mp4"
    _real_mp4(source, duration_seconds=1.0, color="red")
    prevalidated = prevalidate_overlay_source(source)

    _real_mp4(source, duration_seconds=1.0, color="blue")

    with pytest.raises(OverlaySourceMismatchError, match="no longer matches"):
        register_overlay_render_artifact(conn, project, manifest, source, now=FIXED_NOW, prevalidated=prevalidated)


def test_register_without_prevalidated_keeps_existing_behavior(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.mp4"
    _real_mp4(source, duration_seconds=1.0)

    result = register_overlay_render_artifact(conn, project, manifest, source, now=FIXED_NOW)

    assert result.ok is True
    assert result.duration_seconds == pytest.approx(1.0, abs=0.2)


# ---------------------------------------------------------------------
# metadata_overrides
# ---------------------------------------------------------------------


def test_metadata_overrides_merge_into_default_metadata(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.mp4"
    _real_mp4(source, duration_seconds=1.0)

    result = register_overlay_render_artifact(
        conn,
        project,
        manifest,
        source,
        now=FIXED_NOW,
        metadata_overrides={"overlay_count": 3, "source_render_artifact_id": "render-final"},
    )

    assert result.ok is True
    stored = list_artifacts_by_project(conn, project.project_id, kind="overlay_render")[0]
    assert stored.metadata["overlay_count"] == 3
    assert stored.metadata["source_render_artifact_id"] == "render-final"
    assert stored.metadata["source"] == "text-overlay-renderer-v1"


# ---------------------------------------------------------------------
# strict_cleanup
# ---------------------------------------------------------------------


def test_strict_cleanup_unlink_failure_raises_cleanup_error(conn, tmp_path, monkeypatch):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.mp4"
    _real_mp4(source, duration_seconds=1.0)

    def _boom(_conn, _record):
        raise sqlite3.OperationalError("simulated database failure")

    monkeypatch.setattr("src.core.overlay_artifact_registrar.register_artifact", _boom)

    destination = project_dir / "overlay" / "final.mp4"
    real_unlink = Path.unlink

    def _fail_unlink(self, *a, **k):
        if self.name == "final.mp4":
            raise OSError("simulated: permission denied")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", _fail_unlink)

    with pytest.raises(OverlayArtifactCleanupError) as exc_info:
        register_overlay_render_artifact(conn, project, manifest, source, now=FIXED_NOW, strict_cleanup=True)

    assert str(destination) in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, OSError)
    assert list_artifacts_by_project(conn, project.project_id, kind="overlay_render") == []


def test_default_strict_cleanup_false_preserves_existing_callers_behavior(conn, tmp_path, monkeypatch):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.mp4"
    _real_mp4(source, duration_seconds=1.0)

    def _boom(_conn, _record):
        raise sqlite3.OperationalError("simulated database failure")

    monkeypatch.setattr("src.core.overlay_artifact_registrar.register_artifact", _boom)

    real_unlink = Path.unlink

    def _fail_unlink(self, *a, **k):
        if self.name == "final.mp4":
            raise OSError("simulated: permission denied")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", _fail_unlink)

    result = register_overlay_render_artifact(conn, project, manifest, source, now=FIXED_NOW)

    assert result.ok is False
    assert "simulated database failure" in result.reasons[0]
