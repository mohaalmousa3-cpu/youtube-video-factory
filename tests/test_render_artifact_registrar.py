"""Tests for src/core/render_artifact_registrar.py: Phase 2G's
registration-first, project-level Render Artifact Registration. Same
isolated_db / build-manifest-then-register-project pattern as
tests/test_animation_artifact_registrar.py. Uses the REAL local
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
from src.core.project_state_machine import create_initial_project, initial_transition_for
from src.core.render_artifact_registrar import (
    PrevalidatedRenderSource,
    RenderArtifactRegistrationError,
    RenderSourceMismatchError,
    RenderSourceValidationError,
    prevalidate_render_source,
    register_render_artifact,
)
from src.core.verified_transition_service import verify_and_advance
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
    """Generate a real, tiny video+audio MP4 via ffmpeg's lavfi sources —
    a real subprocess call, not a mock."""
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


def _real_audio_only_mp4(path: Path, duration_seconds: float = 1.0) -> None:
    """A genuinely valid, playable MP4 container with a real measurable
    duration — but no video stream. Must be rejected even though duration
    measurement alone would succeed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    from src.utils.config import get_settings

    ffmpeg = get_settings().ffmpeg_path
    result = subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", str(duration_seconds),
         "-c:a", "aac", str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------


def test_valid_render_registration(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.mp4"
    _real_mp4(source, duration_seconds=1.0)

    result = register_render_artifact(conn, project, manifest, source, now=FIXED_NOW)

    assert result.ok is True
    assert result.idempotent is False
    assert result.copied is True
    assert result.artifact_id == "render-final"
    assert result.relative_path == "render/final.mp4"
    assert result.duration_seconds == pytest.approx(1.0, abs=0.2)

    destination = project_dir / "render" / "final.mp4"
    assert destination.exists()
    assert destination.read_bytes() == source.read_bytes()

    stored = list_artifacts_by_project(conn, project.project_id, kind="render")
    assert len(stored) == 1
    assert stored[0].artifact_id == "render-final"
    assert stored[0].kind == "render"
    assert stored[0].scene_id is None
    assert stored[0].sha256_checksum == _sha256(destination)
    assert stored[0].byte_size == destination.stat().st_size
    assert stored[0].metadata["source"] == "external"
    assert stored[0].metadata["duration_seconds"] == pytest.approx(1.0, abs=0.2)
    assert stored[0].created_at == FIXED_NOW


# ---------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------


def test_identical_repeat_is_a_no_write_idempotent_success(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.mp4"
    _real_mp4(source)

    first = register_render_artifact(conn, project, manifest, source, now=FIXED_NOW)
    assert first.ok and not first.idempotent

    destination = project_dir / "render" / "final.mp4"
    bytes_before = destination.read_bytes()
    mtime_before = destination.stat().st_mtime_ns
    rows_before = list_artifacts_by_project(conn, project.project_id, kind="render")

    second = register_render_artifact(conn, project, manifest, source, now=FIXED_NOW)

    assert second.ok is True
    assert second.idempotent is True
    assert second.copied is False
    assert second.reasons == ()

    rows_after = list_artifacts_by_project(conn, project.project_id, kind="render")
    assert rows_after == rows_before  # no new/changed DB row
    assert destination.read_bytes() == bytes_before  # file untouched
    assert destination.stat().st_mtime_ns == mtime_before  # not rewritten


def test_different_second_source_is_rejected_and_leaves_original_unchanged(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source_a = tmp_path / "a.mp4"
    _real_mp4(source_a, color="red")
    first = register_render_artifact(conn, project, manifest, source_a, now=FIXED_NOW)
    assert first.ok

    destination = project_dir / "render" / "final.mp4"
    bytes_before = destination.read_bytes()
    row_before = list_artifacts_by_project(conn, project.project_id, kind="render")[0]

    source_b = tmp_path / "b.mp4"
    _real_mp4(source_b, color="blue")  # genuinely different content

    second = register_render_artifact(conn, project, manifest, source_b, now=FIXED_NOW)

    assert second.ok is False
    assert second.reasons
    assert "already registered" in second.reasons[0]

    assert destination.read_bytes() == bytes_before
    row_after = list_artifacts_by_project(conn, project.project_id, kind="render")[0]
    assert row_after == row_before


# ---------------------------------------------------------------------
# Zero-write rejection cases
# ---------------------------------------------------------------------


def test_mismatched_manifest_project_id_raises_and_causes_zero_writes(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.mp4"
    _real_mp4(source)

    mismatched_manifest = manifest.model_copy(update={"project_id": "proj-does-not-match"})

    with pytest.raises(RenderArtifactRegistrationError):
        register_render_artifact(conn, project, mismatched_manifest, source, now=FIXED_NOW)

    assert list_artifacts_by_project(conn, project.project_id, kind="render") == []
    assert not (project_dir / "render").exists()


def test_missing_source_file_causes_zero_writes(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)

    result = register_render_artifact(conn, project, manifest, tmp_path / "does-not-exist.mp4", now=FIXED_NOW)

    assert result.ok is False
    assert "does not exist" in result.reasons[0]
    assert not (project_dir / "render").exists()


def test_directory_source_causes_zero_writes(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    a_dir = tmp_path / "a_directory.mp4"
    a_dir.mkdir()

    result = register_render_artifact(conn, project, manifest, a_dir, now=FIXED_NOW)

    assert result.ok is False
    assert "not a regular file" in result.reasons[0]
    assert not (project_dir / "render").exists()


def test_zero_byte_source_causes_zero_writes(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")

    result = register_render_artifact(conn, project, manifest, empty, now=FIXED_NOW)

    assert result.ok is False
    assert "empty" in result.reasons[0]
    assert not (project_dir / "render").exists()


# ---------------------------------------------------------------------
# Video validation: malformed, truncated, audio-only
# ---------------------------------------------------------------------


def test_completely_invalid_non_video_source_causes_zero_writes(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    garbage = tmp_path / "garbage.mp4"
    garbage.write_bytes(b"this is not a video at all, just junk bytes 1234567890")

    result = register_render_artifact(conn, project, manifest, garbage, now=FIXED_NOW)

    assert result.ok is False
    assert "could not measure video duration" in result.reasons[0]
    assert not (project_dir / "render").exists()
    assert list_artifacts_by_project(conn, project.project_id, kind="render") == []


def test_truncated_mp4_source_causes_zero_writes(conn, tmp_path):
    """A real MP4 with most of its trailing bytes cut off. As established
    in Phase 2F's test suite, MP4 moov-atom placement isn't perfectly
    predictable for tiny lavfi-generated files (a 50%/75% cut can still
    leave duration metadata readable), so this cuts to 20% — empirically
    verified to reliably corrupt the container ('moov atom not found')."""
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    whole = tmp_path / "whole.mp4"
    _real_mp4(whole, duration_seconds=2.0)
    truncated = tmp_path / "truncated.mp4"
    full_bytes = whole.read_bytes()
    truncated.write_bytes(full_bytes[: int(len(full_bytes) * 0.2)])

    result = register_render_artifact(conn, project, manifest, truncated, now=FIXED_NOW)

    assert result.ok is False
    assert "could not measure video duration" in result.reasons[0]
    assert not (project_dir / "render").exists()
    assert list_artifacts_by_project(conn, project.project_id, kind="render") == []


def test_audio_only_mp4_is_rejected_before_any_copy(conn, tmp_path):
    """A perfectly valid, fully-playable MP4 with a real measurable
    duration — just no video stream. Duration alone would accept this;
    the video-stream check must reject it, before any copy."""
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    audio_only = tmp_path / "audio_only.mp4"
    _real_audio_only_mp4(audio_only)

    result = register_render_artifact(conn, project, manifest, audio_only, now=FIXED_NOW)

    assert result.ok is False
    assert "does not contain a video stream" in result.reasons[0]
    assert not (project_dir / "render").exists()
    assert list_artifacts_by_project(conn, project.project_id, kind="render") == []


# ---------------------------------------------------------------------
# Pre-existing destination file, no matching DB record
# ---------------------------------------------------------------------


def test_different_pre_existing_destination_is_never_overwritten(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    destination = project_dir / "render" / "final.mp4"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"pre-existing content nobody asked to touch")
    bytes_before = destination.read_bytes()

    source = tmp_path / "source.mp4"
    _real_mp4(source)

    result = register_render_artifact(conn, project, manifest, source, now=FIXED_NOW)

    assert result.ok is False
    assert "already exists with different content" in result.reasons[0]
    assert destination.read_bytes() == bytes_before
    assert list_artifacts_by_project(conn, project.project_id, kind="render") == []


def test_identical_pre_existing_destination_is_registered_without_rewriting(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.mp4"
    _real_mp4(source)

    destination = project_dir / "render" / "final.mp4"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(source.read_bytes())  # byte-identical, placed by some other step
    mtime_before = destination.stat().st_mtime_ns

    result = register_render_artifact(conn, project, manifest, source, now=FIXED_NOW)

    assert result.ok is True
    assert result.copied is False  # never re-copied
    assert result.idempotent is False  # this IS a fresh DB registration, just without a copy
    assert destination.stat().st_mtime_ns == mtime_before

    stored = list_artifacts_by_project(conn, project.project_id, kind="render")
    assert len(stored) == 1
    assert stored[0].sha256_checksum == _sha256(source)


# ---------------------------------------------------------------------
# Symlink / path-safety escape
# ---------------------------------------------------------------------


def test_symlinked_render_directory_escape_is_rejected(conn, tmp_path):
    """If `render/` itself were a symlink pointing outside the project
    directory, the shared path-safety check must catch it before any
    write — same defense-in-depth artifact_verifier.py already relies on,
    reused here via src.core.path_safety."""
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    outside = tmp_path / "outside_render"
    outside.mkdir()
    try:
        (project_dir / "render").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"cannot create symlinks in this environment: {exc}")

    source = tmp_path / "source.mp4"
    _real_mp4(source)

    result = register_render_artifact(conn, project, manifest, source, now=FIXED_NOW)

    assert result.ok is False
    assert "unsafe" in result.reasons[0]
    assert list(outside.iterdir()) == []  # nothing was written through the symlink
    assert list_artifacts_by_project(conn, project.project_id, kind="render") == []


# ---------------------------------------------------------------------
# Cleanup on register_artifact() failure
# ---------------------------------------------------------------------


def test_register_artifact_failure_cleans_up_only_a_freshly_copied_destination(conn, tmp_path, monkeypatch):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.mp4"
    _real_mp4(source)

    def _boom(_conn, _record):
        raise sqlite3.OperationalError("simulated database failure")

    monkeypatch.setattr("src.core.render_artifact_registrar.register_artifact", _boom)

    result = register_render_artifact(conn, project, manifest, source, now=FIXED_NOW)

    assert result.ok is False
    assert "simulated database failure" in result.reasons[0]
    destination = project_dir / "render" / "final.mp4"
    assert not destination.exists()  # the freshly-copied file was cleaned up
    if destination.parent.exists():
        assert list(destination.parent.iterdir()) == []  # no orphan left behind either
    assert list_artifacts_by_project(conn, project.project_id, kind="render") == []


def test_register_artifact_failure_never_removes_a_pre_existing_destination(conn, tmp_path, monkeypatch):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.mp4"
    _real_mp4(source)

    destination = project_dir / "render" / "final.mp4"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(source.read_bytes())  # pre-existing, byte-identical

    def _boom(_conn, _record):
        raise sqlite3.OperationalError("simulated database failure")

    monkeypatch.setattr("src.core.render_artifact_registrar.register_artifact", _boom)

    result = register_render_artifact(conn, project, manifest, source, now=FIXED_NOW)

    assert result.ok is False
    assert destination.exists()  # never deleted — this call did not create it
    assert destination.read_bytes() == source.read_bytes()


def test_unexpected_register_artifact_failure_propagates_and_cleans_up_fresh_copy(conn, tmp_path, monkeypatch):
    """An UNEXPECTED exception (not one of register_artifact()'s
    documented failure types) must never be silently downgraded into an
    ordinary rejected result — it propagates unchanged. The freshly-copied
    destination is still cleaned up best-effort first."""
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.mp4"
    _real_mp4(source)

    def _boom(_conn, _record):
        raise RuntimeError("totally unexpected bug")

    monkeypatch.setattr("src.core.render_artifact_registrar.register_artifact", _boom)

    with pytest.raises(RuntimeError, match="totally unexpected bug"):
        register_render_artifact(conn, project, manifest, source, now=FIXED_NOW)

    destination = project_dir / "render" / "final.mp4"
    assert not destination.exists()  # the freshly-copied file was still cleaned up
    assert list_artifacts_by_project(conn, project.project_id, kind="render") == []


def test_unexpected_register_artifact_failure_never_removes_a_pre_existing_destination(conn, tmp_path, monkeypatch):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.mp4"
    _real_mp4(source)

    destination = project_dir / "render" / "final.mp4"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(source.read_bytes())  # pre-existing, byte-identical
    bytes_before = destination.read_bytes()

    def _boom(_conn, _record):
        raise RuntimeError("totally unexpected bug")

    monkeypatch.setattr("src.core.render_artifact_registrar.register_artifact", _boom)

    with pytest.raises(RuntimeError, match="totally unexpected bug"):
        register_render_artifact(conn, project, manifest, source, now=FIXED_NOW)

    assert destination.exists()  # never deleted — this call did not create it
    assert destination.read_bytes() == bytes_before  # byte-for-byte unchanged


# ---------------------------------------------------------------------
# End-to-end with verify_and_advance
# ---------------------------------------------------------------------


def test_end_to_end_register_render_then_verify_and_advance_succeeds(conn, tmp_path):
    from src.core.project_state_machine import transition_project
    from src.database.project_repository import save_transition, get_project

    project, manifest, project_dir = _registered_project(conn, tmp_path, scene_ids=("scene-01", "scene-02"))
    # animation_ready -> render_pending is a plain, unguarded forward
    # transition (render_pending is not verification-required) — no CLI
    # command performs it yet (a pre-existing, repo-wide gap, not specific
    # to Phase 2G), so it's done directly here, same as test setup already
    # does for the earlier _pending stages elsewhere in this test suite.
    # verify_and_advance() only checks the CURRENT target stage's own
    # requirement ("render" for "rendered"), not earlier stages' artifacts,
    # so no audio/visual/animation artifacts need to be registered here.
    stages = [
        ("audio_pending", False), ("audio_ready", True),
        ("visuals_pending", False), ("visuals_ready", True),
        ("animation_pending", False), ("animation_ready", True),
        ("render_pending", False),
    ]
    for stage, verified in stages:
        updated, transition = transition_project(project, stage, now=FIXED_NOW, verified=verified)
        save_transition(conn, project.lifecycle_version, updated, transition)
        project = get_project(conn, project.project_id)

    source = tmp_path / "final.mp4"
    _real_mp4(source, duration_seconds=3.0)
    result = register_render_artifact(conn, project, manifest, source, now=FIXED_NOW)
    assert result.ok, result.reasons

    artifacts = list_artifacts_by_project(conn, project.project_id, kind="render")
    outcome = verify_and_advance(conn, project, manifest, artifacts, "rendered", "smoke test")

    assert outcome.approved is True
    assert outcome.db_committed is True

    final = get_project(conn, project.project_id)
    assert final.current_stage == "rendered"


def test_no_render_artifact_prevents_rendered(conn, tmp_path):
    from src.core.project_state_machine import transition_project
    from src.database.project_repository import save_transition, get_project

    project, manifest, project_dir = _registered_project(conn, tmp_path)
    stages = [
        ("audio_pending", False), ("audio_ready", True),
        ("visuals_pending", False), ("visuals_ready", True),
        ("animation_pending", False), ("animation_ready", True),
        ("render_pending", False),
    ]
    for stage, verified in stages:
        updated, transition = transition_project(project, stage, now=FIXED_NOW, verified=verified)
        save_transition(conn, project.lifecycle_version, updated, transition)
        project = get_project(conn, project.project_id)

    artifacts = list_artifacts_by_project(conn, project.project_id, kind="render")
    assert artifacts == []
    outcome = verify_and_advance(conn, project, manifest, artifacts, "rendered", "smoke test")

    assert outcome.approved is False
    assert outcome.db_committed is False
    assert any("missing required" in r for r in outcome.reasons)

    final = get_project(conn, project.project_id)
    assert final.current_stage == "render_pending"  # never advanced


# ---------------------------------------------------------------------
# prevalidate_render_source() — pure, local, no SQLite connection
# ---------------------------------------------------------------------


def test_prevalidate_render_source_succeeds_on_real_video(tmp_path):
    source = tmp_path / "source.mp4"
    _real_mp4(source, duration_seconds=1.0)

    result = prevalidate_render_source(source)

    assert isinstance(result, PrevalidatedRenderSource)
    assert result.source_path == source.resolve()
    assert result.byte_size == source.stat().st_size
    assert result.sha256_checksum == _sha256(source)
    assert result.duration_seconds == pytest.approx(1.0, abs=0.2)


def test_prevalidate_render_source_missing_file_fails(tmp_path):
    with pytest.raises(RenderSourceValidationError, match="does not exist"):
        prevalidate_render_source(tmp_path / "does-not-exist.mp4")


def test_prevalidate_render_source_empty_file_fails(tmp_path):
    source = tmp_path / "empty.mp4"
    source.write_bytes(b"")

    with pytest.raises(RenderSourceValidationError, match="empty"):
        prevalidate_render_source(source)


def test_prevalidate_render_source_audio_only_fails(tmp_path):
    source = tmp_path / "audio_only.mp4"
    _real_audio_only_mp4(source, duration_seconds=1.0)

    with pytest.raises(RenderSourceValidationError, match="video stream"):
        prevalidate_render_source(source)


# ---------------------------------------------------------------------
# register_render_artifact(..., prevalidated=...) — no ffprobe re-run
# ---------------------------------------------------------------------


def test_register_with_prevalidated_result_succeeds_without_reprobing(conn, tmp_path, monkeypatch):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.mp4"
    _real_mp4(source, duration_seconds=1.0)

    prevalidated = prevalidate_render_source(source)

    def _boom(*a, **k):
        raise AssertionError("must not re-run ffprobe when prevalidated is supplied")

    monkeypatch.setattr("src.core.render_artifact_registrar.get_duration_seconds", _boom)
    monkeypatch.setattr("src.core.render_artifact_registrar._has_video_stream", _boom)

    result = register_render_artifact(conn, project, manifest, source, now=FIXED_NOW, prevalidated=prevalidated)

    assert result.ok is True
    assert result.duration_seconds == pytest.approx(prevalidated.duration_seconds)
    stored = list_artifacts_by_project(conn, project.project_id, kind="render")
    assert len(stored) == 1


def test_register_with_stale_prevalidated_checksum_mismatch_raises(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.mp4"
    _real_mp4(source, duration_seconds=1.0, color="red")

    prevalidated = prevalidate_render_source(source)

    # Replace the file's content after prevalidation ran -- a stale result.
    _real_mp4(source, duration_seconds=1.0, color="blue")

    with pytest.raises(RenderSourceMismatchError, match="no longer matches"):
        register_render_artifact(conn, project, manifest, source, now=FIXED_NOW, prevalidated=prevalidated)

    # No partial registration from the rejected attempt.
    stored = list_artifacts_by_project(conn, project.project_id, kind="render")
    assert stored == []


def test_register_with_stale_prevalidated_size_mismatch_raises(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.mp4"
    _real_mp4(source, duration_seconds=1.0)

    prevalidated = prevalidate_render_source(source)
    # Append bytes after prevalidation -- content AND size both change,
    # but this specifically proves the size check alone would catch it.
    with source.open("ab") as f:
        f.write(b"\x00" * 100)

    with pytest.raises(RenderSourceMismatchError):
        register_render_artifact(conn, project, manifest, source, now=FIXED_NOW, prevalidated=prevalidated)


def test_register_without_prevalidated_keeps_existing_behavior(conn, tmp_path):
    """Every existing caller that omits `prevalidated` (the default,
    None) gets exactly the same full-ffprobe-validation behavior as
    before this parameter existed."""
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.mp4"
    _real_mp4(source, duration_seconds=1.0)

    result = register_render_artifact(conn, project, manifest, source, now=FIXED_NOW)

    assert result.ok is True
    assert result.duration_seconds == pytest.approx(1.0, abs=0.2)


# ---------------------------------------------------------------------
# register_render_artifact(..., metadata_overrides=...)
# ---------------------------------------------------------------------


def test_metadata_overrides_merge_into_default_metadata(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.mp4"
    _real_mp4(source, duration_seconds=1.0)

    result = register_render_artifact(
        conn,
        project,
        manifest,
        source,
        now=FIXED_NOW,
        metadata_overrides={"source": "final-video-assembly-v1", "scene_count": 3, "manifest_fingerprint": "abc123"},
    )

    assert result.ok is True
    stored = list_artifacts_by_project(conn, project.project_id, kind="render")[0]
    assert stored.metadata["source"] == "final-video-assembly-v1"
    assert stored.metadata["scene_count"] == 3
    assert stored.metadata["manifest_fingerprint"] == "abc123"
    assert stored.metadata["duration_seconds"] == pytest.approx(1.0, abs=0.2)


def test_no_metadata_overrides_keeps_default_metadata(conn, tmp_path):
    """Every existing caller that omits `metadata_overrides` gets exactly
    the same default {"duration_seconds": ..., "source": "external"}
    metadata shape as before this parameter existed."""
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.mp4"
    _real_mp4(source, duration_seconds=1.0)

    register_render_artifact(conn, project, manifest, source, now=FIXED_NOW)

    stored = list_artifacts_by_project(conn, project.project_id, kind="render")[0]
    assert stored.metadata["source"] == "external"
    assert set(stored.metadata.keys()) == {"duration_seconds", "source"}
