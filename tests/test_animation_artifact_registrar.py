"""Tests for src/core/animation_artifact_registrar.py: Phase 2F's
registration-first Animation Artifact Registration Pipeline. Same
isolated_db / build-manifest-then-register-project pattern as
tests/test_audio_artifact_registrar.py. Uses the REAL local ffmpeg/ffprobe
binary (no mocking) — no network, no provider, no Flow/Veo."""
from __future__ import annotations

import hashlib
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.core.animation_artifact_registrar import (
    AnimationArtifactRegistrationError,
    register_animation_artifact,
)
from src.core.manifest_builder import build_video_manifest
from src.core.manifest_store import save_manifest
from src.core.project_state_machine import create_initial_project, initial_transition_for
from src.core.verified_transition_service import verify_and_advance
from src.database.artifact_repository import list_artifacts_by_scene
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
            "-f", "lavfi", "-i", f"anullsrc=r=24000:cl=mono",
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


def test_valid_one_scene_registration(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.mp4"
    _real_mp4(source, duration_seconds=1.0)

    result = register_animation_artifact(conn, project, manifest, "scene-01", source, now=FIXED_NOW)

    assert result.ok is True
    assert result.idempotent is False
    assert result.copied is True
    assert result.artifact_id == "animation-scene-01"
    assert result.relative_path == "animation/scene-01.mp4"
    assert result.duration_seconds == pytest.approx(1.0, abs=0.2)

    destination = project_dir / "animation" / "scene-01.mp4"
    assert destination.exists()
    assert destination.read_bytes() == source.read_bytes()

    stored = list_artifacts_by_scene(conn, project.project_id, "scene-01")
    assert len(stored) == 1
    assert stored[0].artifact_id == "animation-scene-01"
    assert stored[0].kind == "animation"
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

    first = register_animation_artifact(conn, project, manifest, "scene-01", source, now=FIXED_NOW)
    assert first.ok and not first.idempotent

    destination = project_dir / "animation" / "scene-01.mp4"
    bytes_before = destination.read_bytes()
    mtime_before = destination.stat().st_mtime_ns
    rows_before = list_artifacts_by_scene(conn, project.project_id, "scene-01")

    second = register_animation_artifact(conn, project, manifest, "scene-01", source, now=FIXED_NOW)

    assert second.ok is True
    assert second.idempotent is True
    assert second.copied is False
    assert second.reasons == ()

    rows_after = list_artifacts_by_scene(conn, project.project_id, "scene-01")
    assert rows_after == rows_before  # no new/changed DB row
    assert destination.read_bytes() == bytes_before  # file untouched
    assert destination.stat().st_mtime_ns == mtime_before  # not rewritten


def test_different_second_source_same_scene_is_rejected_and_leaves_original_unchanged(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source_a = tmp_path / "a.mp4"
    _real_mp4(source_a, color="red")
    first = register_animation_artifact(conn, project, manifest, "scene-01", source_a, now=FIXED_NOW)
    assert first.ok

    destination = project_dir / "animation" / "scene-01.mp4"
    bytes_before = destination.read_bytes()
    row_before = list_artifacts_by_scene(conn, project.project_id, "scene-01")[0]

    source_b = tmp_path / "b.mp4"
    _real_mp4(source_b, color="blue")  # genuinely different content

    second = register_animation_artifact(conn, project, manifest, "scene-01", source_b, now=FIXED_NOW)

    assert second.ok is False
    assert second.reasons
    assert "already registered" in second.reasons[0]

    assert destination.read_bytes() == bytes_before
    row_after = list_artifacts_by_scene(conn, project.project_id, "scene-01")[0]
    assert row_after == row_before


# ---------------------------------------------------------------------
# Zero-write rejection cases
# ---------------------------------------------------------------------


def test_mismatched_manifest_project_id_raises_and_causes_zero_writes(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path, scene_ids=("scene-01",))
    source = tmp_path / "source.mp4"
    _real_mp4(source)

    mismatched_manifest = manifest.model_copy(update={"project_id": "proj-does-not-match"})

    with pytest.raises(AnimationArtifactRegistrationError):
        register_animation_artifact(conn, project, mismatched_manifest, "scene-01", source, now=FIXED_NOW)

    assert list_artifacts_by_scene(conn, project.project_id, "scene-01") == []
    assert not (project_dir / "animation").exists()


def test_unknown_scene_id_causes_zero_writes(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path, scene_ids=("scene-01",))
    source = tmp_path / "source.mp4"
    _real_mp4(source)

    result = register_animation_artifact(conn, project, manifest, "scene-99", source, now=FIXED_NOW)

    assert result.ok is False
    assert "not present in the project manifest" in result.reasons[0]
    assert list_artifacts_by_scene(conn, project.project_id, "scene-99") == []
    assert not (project_dir / "animation").exists()


def test_missing_source_file_causes_zero_writes(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)

    result = register_animation_artifact(
        conn, project, manifest, "scene-01", tmp_path / "does-not-exist.mp4", now=FIXED_NOW
    )

    assert result.ok is False
    assert "does not exist" in result.reasons[0]
    assert not (project_dir / "animation").exists()


def test_directory_source_causes_zero_writes(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    a_dir = tmp_path / "a_directory.mp4"
    a_dir.mkdir()

    result = register_animation_artifact(conn, project, manifest, "scene-01", a_dir, now=FIXED_NOW)

    assert result.ok is False
    assert "not a regular file" in result.reasons[0]
    assert not (project_dir / "animation").exists()


def test_zero_byte_source_causes_zero_writes(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")

    result = register_animation_artifact(conn, project, manifest, "scene-01", empty, now=FIXED_NOW)

    assert result.ok is False
    assert "empty" in result.reasons[0]
    assert not (project_dir / "animation").exists()


# ---------------------------------------------------------------------
# Video validation: malformed, truncated, audio-only
# ---------------------------------------------------------------------


def test_completely_invalid_non_video_source_causes_zero_writes(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    garbage = tmp_path / "garbage.mp4"
    garbage.write_bytes(b"this is not a video at all, just junk bytes 1234567890")

    result = register_animation_artifact(conn, project, manifest, "scene-01", garbage, now=FIXED_NOW)

    assert result.ok is False
    assert "could not measure video duration" in result.reasons[0]
    assert not (project_dir / "animation").exists()
    assert list_artifacts_by_scene(conn, project.project_id, "scene-01") == []


def test_truncated_mp4_source_causes_zero_writes(conn, tmp_path):
    """A real MP4 with most of its trailing bytes cut off. MP4 moov-atom
    placement isn't perfectly predictable (verified: for this tiny
    lavfi-generated file, a 50%/75% cut still leaves duration metadata
    readable — ffmpeg placed moov earlier than the trailing mdat data for
    a file this small — but a 25% cut reliably breaks the container:
    'moov atom not found'), so this cuts to 20% to be robustly corrupt
    rather than relying on an unverified "any truncation breaks it"
    assumption."""
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    whole = tmp_path / "whole.mp4"
    _real_mp4(whole, duration_seconds=2.0)
    truncated = tmp_path / "truncated.mp4"
    full_bytes = whole.read_bytes()
    truncated.write_bytes(full_bytes[: int(len(full_bytes) * 0.2)])

    result = register_animation_artifact(conn, project, manifest, "scene-01", truncated, now=FIXED_NOW)

    assert result.ok is False
    assert "could not measure video duration" in result.reasons[0]
    assert not (project_dir / "animation").exists()
    assert list_artifacts_by_scene(conn, project.project_id, "scene-01") == []


def test_audio_only_mp4_is_rejected_before_any_copy(conn, tmp_path):
    """A perfectly valid, fully-playable MP4 with a real measurable
    duration — just no video stream. Duration alone would accept this;
    the video-stream check must reject it, before any copy."""
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    audio_only = tmp_path / "audio_only.mp4"
    _real_audio_only_mp4(audio_only)

    result = register_animation_artifact(conn, project, manifest, "scene-01", audio_only, now=FIXED_NOW)

    assert result.ok is False
    assert "does not contain a video stream" in result.reasons[0]
    assert not (project_dir / "animation").exists()
    assert list_artifacts_by_scene(conn, project.project_id, "scene-01") == []


# ---------------------------------------------------------------------
# Pre-existing destination file, no matching DB record
# ---------------------------------------------------------------------


def test_different_pre_existing_destination_is_never_overwritten(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    destination = project_dir / "animation" / "scene-01.mp4"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"pre-existing content nobody asked to touch")
    bytes_before = destination.read_bytes()

    source = tmp_path / "source.mp4"
    _real_mp4(source)

    result = register_animation_artifact(conn, project, manifest, "scene-01", source, now=FIXED_NOW)

    assert result.ok is False
    assert "already exists with different content" in result.reasons[0]
    assert destination.read_bytes() == bytes_before
    assert list_artifacts_by_scene(conn, project.project_id, "scene-01") == []


def test_identical_pre_existing_destination_is_registered_without_rewriting(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.mp4"
    _real_mp4(source)

    destination = project_dir / "animation" / "scene-01.mp4"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(source.read_bytes())  # byte-identical, placed by some other step
    mtime_before = destination.stat().st_mtime_ns

    result = register_animation_artifact(conn, project, manifest, "scene-01", source, now=FIXED_NOW)

    assert result.ok is True
    assert result.copied is False  # never re-copied
    assert result.idempotent is False  # this IS a fresh DB registration, just without a copy
    assert destination.stat().st_mtime_ns == mtime_before

    stored = list_artifacts_by_scene(conn, project.project_id, "scene-01")
    assert len(stored) == 1
    assert stored[0].sha256_checksum == _sha256(source)


# ---------------------------------------------------------------------
# Symlink / path-safety escape
# ---------------------------------------------------------------------


def test_symlinked_animation_directory_escape_is_rejected(conn, tmp_path):
    """If `animation/` itself were a symlink pointing outside the project
    directory, the shared path-safety check must catch it before any
    write — same defense-in-depth artifact_verifier.py already relies on,
    reused here via src.core.path_safety."""
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    outside = tmp_path / "outside_animation"
    outside.mkdir()
    try:
        (project_dir / "animation").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"cannot create symlinks in this environment: {exc}")

    source = tmp_path / "source.mp4"
    _real_mp4(source)

    result = register_animation_artifact(conn, project, manifest, "scene-01", source, now=FIXED_NOW)

    assert result.ok is False
    assert "unsafe" in result.reasons[0]
    assert list(outside.iterdir()) == []  # nothing was written through the symlink
    assert list_artifacts_by_scene(conn, project.project_id, "scene-01") == []


# ---------------------------------------------------------------------
# Cleanup on register_artifact() failure
# ---------------------------------------------------------------------


def test_register_artifact_failure_cleans_up_only_a_freshly_copied_destination(conn, tmp_path, monkeypatch):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.mp4"
    _real_mp4(source)

    def _boom(_conn, _record):
        raise sqlite3.OperationalError("simulated database failure")

    monkeypatch.setattr("src.core.animation_artifact_registrar.register_artifact", _boom)

    result = register_animation_artifact(conn, project, manifest, "scene-01", source, now=FIXED_NOW)

    assert result.ok is False
    assert "simulated database failure" in result.reasons[0]
    destination = project_dir / "animation" / "scene-01.mp4"
    assert not destination.exists()  # the freshly-copied file was cleaned up
    if destination.parent.exists():
        assert list(destination.parent.iterdir()) == []  # no orphan left behind either
    assert list_artifacts_by_scene(conn, project.project_id, "scene-01") == []


def test_register_artifact_failure_never_removes_a_pre_existing_destination(conn, tmp_path, monkeypatch):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.mp4"
    _real_mp4(source)

    destination = project_dir / "animation" / "scene-01.mp4"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(source.read_bytes())  # pre-existing, byte-identical

    def _boom(_conn, _record):
        raise sqlite3.OperationalError("simulated database failure")

    monkeypatch.setattr("src.core.animation_artifact_registrar.register_artifact", _boom)

    result = register_animation_artifact(conn, project, manifest, "scene-01", source, now=FIXED_NOW)

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

    monkeypatch.setattr("src.core.animation_artifact_registrar.register_artifact", _boom)

    with pytest.raises(RuntimeError, match="totally unexpected bug"):
        register_animation_artifact(conn, project, manifest, "scene-01", source, now=FIXED_NOW)

    destination = project_dir / "animation" / "scene-01.mp4"
    assert not destination.exists()  # the freshly-copied file was still cleaned up
    assert list_artifacts_by_scene(conn, project.project_id, "scene-01") == []


def test_unexpected_register_artifact_failure_never_removes_a_pre_existing_destination(conn, tmp_path, monkeypatch):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.mp4"
    _real_mp4(source)

    destination = project_dir / "animation" / "scene-01.mp4"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(source.read_bytes())  # pre-existing, byte-identical
    bytes_before = destination.read_bytes()

    def _boom(_conn, _record):
        raise RuntimeError("totally unexpected bug")

    monkeypatch.setattr("src.core.animation_artifact_registrar.register_artifact", _boom)

    with pytest.raises(RuntimeError, match="totally unexpected bug"):
        register_animation_artifact(conn, project, manifest, "scene-01", source, now=FIXED_NOW)

    assert destination.exists()  # never deleted — this call did not create it
    assert destination.read_bytes() == bytes_before  # byte-for-byte unchanged


# ---------------------------------------------------------------------
# End-to-end with verify_and_advance
# ---------------------------------------------------------------------


def test_end_to_end_register_all_scenes_then_verify_and_advance_succeeds(conn, tmp_path):
    from src.core.project_state_machine import transition_project
    from src.database.project_repository import save_transition, get_project

    project, manifest, project_dir = _registered_project(conn, tmp_path, scene_ids=("scene-01", "scene-02"))
    # visuals_ready -> animation_pending is a plain, unguarded forward
    # transition (animation_pending is not verification-required) — no
    # CLI command performs it yet (a pre-existing, repo-wide gap, not
    # specific to Phase 2F), so it's done directly here, same as test
    # setup already does for audio_pending/visuals_pending elsewhere in
    # this test suite.
    stages = [
        ("audio_pending", False), ("audio_ready", True),
        ("visuals_pending", False), ("visuals_ready", True),
        ("animation_pending", False),
    ]
    for stage, verified in stages:
        updated, transition = transition_project(project, stage, now=FIXED_NOW, verified=verified)
        save_transition(conn, project.lifecycle_version, updated, transition)
        project = get_project(conn, project.project_id)

    for scene_id in ("scene-01", "scene-02"):
        source = tmp_path / f"{scene_id}.mp4"
        _real_mp4(source)
        result = register_animation_artifact(conn, project, manifest, scene_id, source, now=FIXED_NOW)
        assert result.ok, result.reasons

    artifacts = list(list_artifacts_by_scene(conn, project.project_id, "scene-01")) + list(
        list_artifacts_by_scene(conn, project.project_id, "scene-02")
    )
    outcome = verify_and_advance(conn, project, manifest, artifacts, "animation_ready", "smoke test")

    assert outcome.approved is True
    assert outcome.db_committed is True

    final = get_project(conn, project.project_id)
    assert final.current_stage == "animation_ready"


def test_incomplete_multi_scene_animation_still_prevents_animation_ready(conn, tmp_path):
    from src.core.project_state_machine import transition_project
    from src.database.project_repository import save_transition, get_project

    project, manifest, project_dir = _registered_project(conn, tmp_path, scene_ids=("scene-01", "scene-02"))
    stages = [
        ("audio_pending", False), ("audio_ready", True),
        ("visuals_pending", False), ("visuals_ready", True),
        ("animation_pending", False),
    ]
    for stage, verified in stages:
        updated, transition = transition_project(project, stage, now=FIXED_NOW, verified=verified)
        save_transition(conn, project.lifecycle_version, updated, transition)
        project = get_project(conn, project.project_id)

    source = tmp_path / "scene-01.mp4"
    _real_mp4(source)
    result = register_animation_artifact(conn, project, manifest, "scene-01", source, now=FIXED_NOW)
    assert result.ok

    artifacts = list_artifacts_by_scene(conn, project.project_id, "scene-01")
    outcome = verify_and_advance(conn, project, manifest, artifacts, "animation_ready", "smoke test")

    assert outcome.approved is False
    assert outcome.db_committed is False
    assert any("missing required" in r for r in outcome.reasons)

    final = get_project(conn, project.project_id)
    assert final.current_stage == "animation_pending"  # never advanced
