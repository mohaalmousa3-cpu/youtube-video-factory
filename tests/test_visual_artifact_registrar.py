"""Tests for src/core/visual_artifact_registrar.py: Phase 2E's
registration-first Visual Artifact Registration Pipeline. Same isolated_db /
build-manifest-then-register-project pattern as
tests/test_audio_artifact_registrar.py. Uses the REAL Pillow decode path
(no mocking of image validation) — no network, no provider, no image
generation."""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest
from PIL import Image

from src.core.manifest_builder import build_video_manifest
from src.core.manifest_store import save_manifest
from src.core.project_state_machine import create_initial_project, initial_transition_for
from src.core.verified_transition_service import verify_and_advance
from src.core.visual_artifact_registrar import (
    VisualArtifactRegistrationError,
    register_visual_artifact,
)
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


def _real_png(path: Path, size: tuple[int, int] = (16, 12), color=(255, 0, 0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path, format="PNG")


def _real_jpeg(path: Path, size: tuple[int, int] = (16, 12), color=(0, 255, 0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path, format="JPEG")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------


def test_valid_one_scene_registration(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.png"
    _real_png(source, size=(16, 12))

    result = register_visual_artifact(conn, project, manifest, "scene-01", source, now=FIXED_NOW)

    assert result.ok is True
    assert result.idempotent is False
    assert result.copied is True
    assert result.artifact_id == "visual-scene-01"
    assert result.relative_path == "visuals/scene-01.png"
    assert result.width == 16
    assert result.height == 12
    assert result.image_format == "PNG"

    destination = project_dir / "visuals" / "scene-01.png"
    assert destination.exists()
    assert destination.read_bytes() == source.read_bytes()

    stored = list_artifacts_by_scene(conn, project.project_id, "scene-01")
    assert len(stored) == 1
    assert stored[0].artifact_id == "visual-scene-01"
    assert stored[0].kind == "visual"
    assert stored[0].sha256_checksum == _sha256(destination)
    assert stored[0].byte_size == destination.stat().st_size
    assert stored[0].metadata["source"] == "external"
    assert stored[0].metadata["width"] == 16
    assert stored[0].metadata["height"] == 12
    assert stored[0].metadata["format"] == "PNG"
    assert stored[0].created_at == FIXED_NOW


# ---------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------


def test_identical_repeat_is_a_no_write_idempotent_success(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.png"
    _real_png(source)

    first = register_visual_artifact(conn, project, manifest, "scene-01", source, now=FIXED_NOW)
    assert first.ok and not first.idempotent

    destination = project_dir / "visuals" / "scene-01.png"
    bytes_before = destination.read_bytes()
    mtime_before = destination.stat().st_mtime_ns
    rows_before = list_artifacts_by_scene(conn, project.project_id, "scene-01")

    second = register_visual_artifact(conn, project, manifest, "scene-01", source, now=FIXED_NOW)

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
    source_a = tmp_path / "a.png"
    _real_png(source_a, color=(255, 0, 0))
    first = register_visual_artifact(conn, project, manifest, "scene-01", source_a, now=FIXED_NOW)
    assert first.ok

    destination = project_dir / "visuals" / "scene-01.png"
    bytes_before = destination.read_bytes()
    row_before = list_artifacts_by_scene(conn, project.project_id, "scene-01")[0]

    source_b = tmp_path / "b.png"
    _real_png(source_b, color=(0, 0, 255))  # genuinely different content

    second = register_visual_artifact(conn, project, manifest, "scene-01", source_b, now=FIXED_NOW)

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
    source = tmp_path / "source.png"
    _real_png(source)

    mismatched_manifest = manifest.model_copy(update={"project_id": "proj-does-not-match"})

    with pytest.raises(VisualArtifactRegistrationError):
        register_visual_artifact(conn, project, mismatched_manifest, "scene-01", source, now=FIXED_NOW)

    assert list_artifacts_by_scene(conn, project.project_id, "scene-01") == []
    assert not (project_dir / "visuals").exists()


def test_unknown_scene_id_causes_zero_writes(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path, scene_ids=("scene-01",))
    source = tmp_path / "source.png"
    _real_png(source)

    result = register_visual_artifact(conn, project, manifest, "scene-99", source, now=FIXED_NOW)

    assert result.ok is False
    assert "not present in the project manifest" in result.reasons[0]
    assert list_artifacts_by_scene(conn, project.project_id, "scene-99") == []
    assert not (project_dir / "visuals").exists()


def test_missing_source_file_causes_zero_writes(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)

    result = register_visual_artifact(
        conn, project, manifest, "scene-01", tmp_path / "does-not-exist.png", now=FIXED_NOW
    )

    assert result.ok is False
    assert "does not exist" in result.reasons[0]
    assert not (project_dir / "visuals").exists()


def test_directory_source_causes_zero_writes(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    a_dir = tmp_path / "a_directory.png"
    a_dir.mkdir()

    result = register_visual_artifact(conn, project, manifest, "scene-01", a_dir, now=FIXED_NOW)

    assert result.ok is False
    assert "not a regular file" in result.reasons[0]
    assert not (project_dir / "visuals").exists()


def test_zero_byte_source_causes_zero_writes(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")

    result = register_visual_artifact(conn, project, manifest, "scene-01", empty, now=FIXED_NOW)

    assert result.ok is False
    assert "empty" in result.reasons[0]
    assert not (project_dir / "visuals").exists()


# ---------------------------------------------------------------------
# Image validation: malformed, truncated, wrong-format
# ---------------------------------------------------------------------


def test_completely_invalid_non_image_source_causes_zero_writes(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    garbage = tmp_path / "garbage.png"
    garbage.write_bytes(b"this is not an image at all, just junk bytes 1234567890")

    result = register_visual_artifact(conn, project, manifest, "scene-01", garbage, now=FIXED_NOW)

    assert result.ok is False
    assert "could not decode image file" in result.reasons[0]
    assert not (project_dir / "visuals").exists()
    assert list_artifacts_by_scene(conn, project.project_id, "scene-01") == []


def test_truncated_png_source_causes_zero_writes(conn, tmp_path):
    """A real PNG whose trailing bytes were cut off — a valid header
    (Image.open() succeeds) but an incomplete data stream (Image.load()
    raises) — proving the decode check covers both Pillow failure points,
    not just the header-identification one."""
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    whole = tmp_path / "whole.png"
    _real_png(whole, size=(200, 150))
    truncated = tmp_path / "truncated.png"
    full_bytes = whole.read_bytes()
    truncated.write_bytes(full_bytes[:-100])

    result = register_visual_artifact(conn, project, manifest, "scene-01", truncated, now=FIXED_NOW)

    assert result.ok is False
    assert "could not decode image file" in result.reasons[0]
    assert not (project_dir / "visuals").exists()
    assert list_artifacts_by_scene(conn, project.project_id, "scene-01") == []


def test_valid_jpeg_source_is_rejected_before_any_copy(conn, tmp_path):
    """A perfectly valid, fully-decodable image — just not a PNG — must
    still be rejected, and rejected before any copy."""
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    jpeg = tmp_path / "valid.jpg"
    _real_jpeg(jpeg)

    result = register_visual_artifact(conn, project, manifest, "scene-01", jpeg, now=FIXED_NOW)

    assert result.ok is False
    assert "not a PNG" in result.reasons[0]
    assert "JPEG" in result.reasons[0]
    assert not (project_dir / "visuals").exists()
    assert list_artifacts_by_scene(conn, project.project_id, "scene-01") == []


# ---------------------------------------------------------------------
# Pre-existing destination file, no matching DB record
# ---------------------------------------------------------------------


def test_different_pre_existing_destination_is_never_overwritten(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    destination = project_dir / "visuals" / "scene-01.png"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"pre-existing content nobody asked to touch")
    bytes_before = destination.read_bytes()

    source = tmp_path / "source.png"
    _real_png(source)

    result = register_visual_artifact(conn, project, manifest, "scene-01", source, now=FIXED_NOW)

    assert result.ok is False
    assert "already exists with different content" in result.reasons[0]
    assert destination.read_bytes() == bytes_before
    assert list_artifacts_by_scene(conn, project.project_id, "scene-01") == []


def test_identical_pre_existing_destination_is_registered_without_rewriting(conn, tmp_path):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.png"
    _real_png(source)

    destination = project_dir / "visuals" / "scene-01.png"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(source.read_bytes())  # byte-identical, placed by some other step
    mtime_before = destination.stat().st_mtime_ns

    result = register_visual_artifact(conn, project, manifest, "scene-01", source, now=FIXED_NOW)

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


def test_symlinked_visuals_directory_escape_is_rejected(conn, tmp_path):
    """If `visuals/` itself were a symlink pointing outside the project
    directory, the shared path-safety check must catch it before any
    write — same defense-in-depth artifact_verifier.py already relies on,
    reused here via src.core.path_safety."""
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    outside = tmp_path / "outside_visuals"
    outside.mkdir()
    try:
        (project_dir / "visuals").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"cannot create symlinks in this environment: {exc}")

    source = tmp_path / "source.png"
    _real_png(source)

    result = register_visual_artifact(conn, project, manifest, "scene-01", source, now=FIXED_NOW)

    assert result.ok is False
    assert "unsafe" in result.reasons[0]
    assert list(outside.iterdir()) == []  # nothing was written through the symlink
    assert list_artifacts_by_scene(conn, project.project_id, "scene-01") == []


# ---------------------------------------------------------------------
# Cleanup on register_artifact() failure
# ---------------------------------------------------------------------


def test_register_artifact_failure_cleans_up_only_a_freshly_copied_destination(conn, tmp_path, monkeypatch):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.png"
    _real_png(source)

    def _boom(_conn, _record):
        raise sqlite3.OperationalError("simulated database failure")

    monkeypatch.setattr("src.core.visual_artifact_registrar.register_artifact", _boom)

    result = register_visual_artifact(conn, project, manifest, "scene-01", source, now=FIXED_NOW)

    assert result.ok is False
    assert "simulated database failure" in result.reasons[0]
    destination = project_dir / "visuals" / "scene-01.png"
    assert not destination.exists()  # the freshly-copied file was cleaned up
    if destination.parent.exists():
        assert list(destination.parent.iterdir()) == []  # no orphan left behind either
    assert list_artifacts_by_scene(conn, project.project_id, "scene-01") == []


def test_register_artifact_failure_never_removes_a_pre_existing_destination(conn, tmp_path, monkeypatch):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.png"
    _real_png(source)

    destination = project_dir / "visuals" / "scene-01.png"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(source.read_bytes())  # pre-existing, byte-identical

    def _boom(_conn, _record):
        raise sqlite3.OperationalError("simulated database failure")

    monkeypatch.setattr("src.core.visual_artifact_registrar.register_artifact", _boom)

    result = register_visual_artifact(conn, project, manifest, "scene-01", source, now=FIXED_NOW)

    assert result.ok is False
    assert destination.exists()  # never deleted — this call did not create it
    assert destination.read_bytes() == source.read_bytes()


def test_unexpected_register_artifact_failure_propagates_and_cleans_up_fresh_copy(conn, tmp_path, monkeypatch):
    """An UNEXPECTED exception (not one of register_artifact()'s
    documented failure types) must never be silently downgraded into an
    ordinary rejected result — it propagates unchanged. The freshly-copied
    destination is still cleaned up best-effort first."""
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.png"
    _real_png(source)

    def _boom(_conn, _record):
        raise RuntimeError("totally unexpected bug")

    monkeypatch.setattr("src.core.visual_artifact_registrar.register_artifact", _boom)

    with pytest.raises(RuntimeError, match="totally unexpected bug"):
        register_visual_artifact(conn, project, manifest, "scene-01", source, now=FIXED_NOW)

    destination = project_dir / "visuals" / "scene-01.png"
    assert not destination.exists()  # the freshly-copied file was still cleaned up
    assert list_artifacts_by_scene(conn, project.project_id, "scene-01") == []


def test_unexpected_register_artifact_failure_never_removes_a_pre_existing_destination(conn, tmp_path, monkeypatch):
    project, manifest, project_dir = _registered_project(conn, tmp_path)
    source = tmp_path / "source.png"
    _real_png(source)

    destination = project_dir / "visuals" / "scene-01.png"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(source.read_bytes())  # pre-existing, byte-identical
    bytes_before = destination.read_bytes()

    def _boom(_conn, _record):
        raise RuntimeError("totally unexpected bug")

    monkeypatch.setattr("src.core.visual_artifact_registrar.register_artifact", _boom)

    with pytest.raises(RuntimeError, match="totally unexpected bug"):
        register_visual_artifact(conn, project, manifest, "scene-01", source, now=FIXED_NOW)

    assert destination.exists()  # never deleted — this call did not create it
    assert destination.read_bytes() == bytes_before  # byte-for-byte unchanged


# ---------------------------------------------------------------------
# End-to-end with verify_and_advance
# ---------------------------------------------------------------------


def test_end_to_end_register_all_scenes_then_verify_and_advance_succeeds(conn, tmp_path):
    from src.core.project_state_machine import transition_project
    from src.database.project_repository import save_transition, get_project

    project, manifest, project_dir = _registered_project(conn, tmp_path, scene_ids=("scene-01", "scene-02"))
    # audio_ready -> visuals_pending is a plain, unguarded forward
    # transition (visuals_pending is not verification-required) — no CLI
    # command performs it yet (a pre-existing, repo-wide gap, not specific
    # to Phase 2E), so it's done directly here, same as test setup already
    # does for audio_pending elsewhere in this test suite.
    for stage, verified in [("audio_pending", False), ("audio_ready", True), ("visuals_pending", False)]:
        updated, transition = transition_project(project, stage, now=FIXED_NOW, verified=verified)
        save_transition(conn, project.lifecycle_version, updated, transition)
        project = get_project(conn, project.project_id)

    for scene_id in ("scene-01", "scene-02"):
        source = tmp_path / f"{scene_id}.png"
        _real_png(source)
        result = register_visual_artifact(conn, project, manifest, scene_id, source, now=FIXED_NOW)
        assert result.ok, result.reasons

    artifacts = list(list_artifacts_by_scene(conn, project.project_id, "scene-01")) + list(
        list_artifacts_by_scene(conn, project.project_id, "scene-02")
    )
    outcome = verify_and_advance(conn, project, manifest, artifacts, "visuals_ready", "smoke test")

    assert outcome.approved is True
    assert outcome.db_committed is True

    final = get_project(conn, project.project_id)
    assert final.current_stage == "visuals_ready"


def test_incomplete_multi_scene_visuals_still_prevents_visuals_ready(conn, tmp_path):
    from src.core.project_state_machine import transition_project
    from src.database.project_repository import save_transition, get_project

    project, manifest, project_dir = _registered_project(conn, tmp_path, scene_ids=("scene-01", "scene-02"))
    for stage, verified in [("audio_pending", False), ("audio_ready", True), ("visuals_pending", False)]:
        updated, transition = transition_project(project, stage, now=FIXED_NOW, verified=verified)
        save_transition(conn, project.lifecycle_version, updated, transition)
        project = get_project(conn, project.project_id)

    source = tmp_path / "scene-01.png"
    _real_png(source)
    result = register_visual_artifact(conn, project, manifest, "scene-01", source, now=FIXED_NOW)
    assert result.ok

    artifacts = list_artifacts_by_scene(conn, project.project_id, "scene-01")
    outcome = verify_and_advance(conn, project, manifest, artifacts, "visuals_ready", "smoke test")

    assert outcome.approved is False
    assert outcome.db_committed is False
    assert any("missing required" in r for r in outcome.reasons)

    final = get_project(conn, project.project_id)
    assert final.current_stage == "visuals_pending"  # never advanced
