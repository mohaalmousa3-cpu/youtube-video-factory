"""Tests for src/core/artifact_verifier.py: pure, read-only artifact
verification against a real (tmp_path) filesystem and a real VideoManifest.
No network, no provider, no database."""
from __future__ import annotations

import builtins
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.core.artifact_verifier import verify_artifact, verify_artifacts
from src.core.manifest_builder import build_video_manifest
from src.core.project_state_machine import create_initial_project, transition_project
from src.models.artifact import ArtifactRecord
from src.utils.channel_config import get_channel_policy

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _story_input(**overrides) -> dict:
    data = dict(
        story_id="why-we-care-what-people-think",
        title="Why We Care So Much What People Think",
        topic="social psychology",
        target_duration_seconds=480.0,
        language="en-US",
        viewer_facing_language="English",
        approval_status="approved",
    )
    data.update(overrides)
    return data


def _scene_plan(scene_count: int = 1) -> dict:
    scenes = []
    for i in range(1, scene_count + 1):
        scenes.append(
            dict(
                scene_id=f"scene-{i:02d}",
                sequence=i,
                narration_text=f"Narration for scene {i}.",
                scene_type="establishing",
                narrative_beat="hook",
                visual_brief="Stickman character alone on a quiet street corner at dusk.",
                motion_mode="in",
                approval_state="approved",
            )
        )
    return dict(scenes=tuple(scenes), role_outfits=())


def _manifest(scene_count: int = 1, **story_overrides):
    return build_video_manifest(
        _story_input(**story_overrides), _scene_plan(scene_count), get_channel_policy(), created_at=NOW
    )


def _write_file(path: Path, content: bytes) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return len(content), hashlib.sha256(content).hexdigest()


def _artifact(project_id: str, relative_path: str, byte_size: int, checksum: str, **overrides) -> ArtifactRecord:
    data = dict(
        artifact_id="audio-scene-01",
        project_id=project_id,
        kind="audio",
        scene_id="scene-01",
        relative_path=relative_path,
        byte_size=byte_size,
        sha256_checksum=checksum,
        created_at=NOW,
    )
    data.update(overrides)
    return ArtifactRecord(**data)


# ---------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------


def test_verify_artifact_passes_for_matching_file(tmp_path):
    manifest = _manifest()
    size, checksum = _write_file(tmp_path / "audio" / "scene-01.wav", b"hello world")
    artifact = _artifact(manifest.project_id, "audio/scene-01.wav", size, checksum)

    result = verify_artifact(tmp_path, artifact, manifest)

    assert result.passed is True
    assert result.reasons == ()


# ---------------------------------------------------------------------
# checksum mismatch / missing file / wrong byte size / directory instead of file
# ---------------------------------------------------------------------


def test_verify_artifact_fails_on_checksum_mismatch(tmp_path):
    manifest = _manifest()
    size, _real_checksum = _write_file(tmp_path / "audio" / "scene-01.wav", b"hello world")
    artifact = _artifact(manifest.project_id, "audio/scene-01.wav", size, "b" * 64)

    result = verify_artifact(tmp_path, artifact, manifest)

    assert result.passed is False
    assert any("checksum mismatch" in reason for reason in result.reasons)


def test_verify_artifact_fails_on_missing_file(tmp_path):
    manifest = _manifest()
    artifact = _artifact(manifest.project_id, "audio/does-not-exist.wav", 5, "a" * 64)

    result = verify_artifact(tmp_path, artifact, manifest)

    assert result.passed is False
    assert any("does not exist" in reason for reason in result.reasons)


def test_verify_artifact_fails_on_wrong_byte_size(tmp_path):
    manifest = _manifest()
    _size, checksum = _write_file(tmp_path / "audio" / "scene-01.wav", b"hello world")
    artifact = _artifact(manifest.project_id, "audio/scene-01.wav", 999, checksum)

    result = verify_artifact(tmp_path, artifact, manifest)

    assert result.passed is False
    assert any("byte_size mismatch" in reason for reason in result.reasons)


def test_verify_artifact_fails_when_path_is_a_directory(tmp_path):
    manifest = _manifest()
    (tmp_path / "audio" / "scene-01.wav").mkdir(parents=True)
    artifact = _artifact(manifest.project_id, "audio/scene-01.wav", 0, "a" * 64)

    result = verify_artifact(tmp_path, artifact, manifest)

    assert result.passed is False
    assert any("not a regular file" in reason for reason in result.reasons)


# ---------------------------------------------------------------------
# path traversal / absolute paths / symlink escape
# ---------------------------------------------------------------------


def test_verify_artifact_rejects_traversal_that_bypassed_model_validation(tmp_path):
    """ArtifactRecord's own validator already refuses to construct a
    record with a '..' segment — this proves the verifier's own
    filesystem-aware check is a real, independent second line of defense,
    not just decorative, by constructing the record via model_construct()
    (bypassing validation) the way a corrupted/hand-crafted input would."""
    manifest = _manifest()
    outside = tmp_path.parent / "outside-secret.txt"
    outside.write_bytes(b"secret")
    artifact = ArtifactRecord.model_construct(
        artifact_id="audio-scene-01",
        project_id=manifest.project_id,
        kind="audio",
        scene_id="scene-01",
        relative_path="../outside-secret.txt",
        byte_size=6,
        sha256_checksum=hashlib.sha256(b"secret").hexdigest(),
        created_at=NOW,
        metadata={},
    )

    result = verify_artifact(tmp_path, artifact, manifest)

    assert result.passed is False
    assert any("unsafe" in reason for reason in result.reasons)


def test_verify_artifact_rejects_absolute_path_that_bypassed_model_validation(tmp_path):
    manifest = _manifest()
    outside = tmp_path.parent / "outside-secret.txt"
    outside.write_bytes(b"secret")
    artifact = ArtifactRecord.model_construct(
        artifact_id="audio-scene-01",
        project_id=manifest.project_id,
        kind="audio",
        scene_id="scene-01",
        relative_path=str(outside),
        byte_size=6,
        sha256_checksum=hashlib.sha256(b"secret").hexdigest(),
        created_at=NOW,
        metadata={},
    )

    result = verify_artifact(tmp_path, artifact, manifest)

    assert result.passed is False
    assert any("unsafe" in reason for reason in result.reasons)


def test_verify_artifact_rejects_symlink_escape(tmp_path):
    manifest = _manifest()
    outside = tmp_path.parent / "outside-secret.txt"
    outside.write_bytes(b"secret")
    link_path = tmp_path / "audio" / "scene-01.wav"
    link_path.parent.mkdir(parents=True)
    try:
        os.symlink(outside, link_path)
    except OSError as exc:
        pytest.skip(f"cannot create symlinks in this environment: {exc}")

    artifact = _artifact(
        manifest.project_id, "audio/scene-01.wav", 6, hashlib.sha256(b"secret").hexdigest()
    )

    result = verify_artifact(tmp_path, artifact, manifest)

    assert result.passed is False
    assert any("unsafe" in reason for reason in result.reasons)


# ---------------------------------------------------------------------
# OSError / TOCTOU handling: a permission error or a file that changes/
# disappears mid-check must produce a failed result, never a raw
# exception (UltraReview finding — src/core/artifact_verifier.py's
# stat()/is_file()/hashing chain previously had no OSError handling).
# ---------------------------------------------------------------------


def test_verify_artifact_handles_permission_error_on_stat_or_exists(tmp_path, monkeypatch):
    """Path.exists()/is_file() both call self.stat() internally and
    re-raise a non-ignorable OSError (confirmed: errno=EACCES is not in
    pathlib's _IGNORED_ERRNOS) rather than swallowing it — so patching
    stat() alone is enough to exercise that path without needing to reach
    all the way to the hashing step."""
    manifest = _manifest()
    size, checksum = _write_file(tmp_path / "audio" / "scene-01.wav", b"hello world")
    artifact = _artifact(manifest.project_id, "audio/scene-01.wav", size, checksum)

    original_stat = Path.stat

    def raising_stat(self, *args, **kwargs):
        if self.name == "scene-01.wav":
            raise PermissionError(13, "Permission denied")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", raising_stat)

    result = verify_artifact(tmp_path, artifact, manifest)

    assert result.passed is False
    assert any("could not inspect file" in reason for reason in result.reasons)
    assert any("PermissionError" in reason for reason in result.reasons)
    # Concise and non-sensitive: the raw OS error text is never included.
    assert not any("Permission denied" in reason for reason in result.reasons)


def test_verify_artifact_handles_oserror_during_hashing(tmp_path, monkeypatch):
    """Isolates a failure specifically during the SHA-256 read step (file
    exists, stat succeeds, only opening/reading for hashing fails) — e.g.
    a TOCTOU race where the file is deleted or becomes unreadable between
    the existence check and the hash read."""
    import src.core.artifact_verifier as artifact_verifier_module

    manifest = _manifest()
    size, checksum = _write_file(tmp_path / "audio" / "scene-01.wav", b"hello world")
    artifact = _artifact(manifest.project_id, "audio/scene-01.wav", size, checksum)

    def raising_hash(_path):
        raise OSError("simulated read failure")

    monkeypatch.setattr(artifact_verifier_module, "_sha256_of_file", raising_hash)

    result = verify_artifact(tmp_path, artifact, manifest)

    assert result.passed is False
    assert any("could not inspect file" in reason for reason in result.reasons)
    assert any("OSError" in reason for reason in result.reasons)


# ---------------------------------------------------------------------
# scene mismatch / manifest-project mismatch
# ---------------------------------------------------------------------


def test_verify_artifact_fails_when_scene_not_in_manifest(tmp_path):
    manifest = _manifest(scene_count=1)  # only scene-01 exists
    size, checksum = _write_file(tmp_path / "audio" / "scene-05.wav", b"hello world")
    artifact = _artifact(
        manifest.project_id, "audio/scene-05.wav", size, checksum, scene_id="scene-05"
    )

    result = verify_artifact(tmp_path, artifact, manifest)

    assert result.passed is False
    assert any("not present in the project manifest" in reason for reason in result.reasons)


def test_verify_artifact_fails_when_project_id_does_not_match_manifest(tmp_path):
    manifest = _manifest()
    size, checksum = _write_file(tmp_path / "audio" / "scene-01.wav", b"hello world")
    artifact = _artifact("proj-some-other-project", "audio/scene-01.wav", size, checksum)

    result = verify_artifact(tmp_path, artifact, manifest)

    assert result.passed is False
    assert any("does not match" in reason for reason in result.reasons)


def test_verify_artifact_multiple_failures_all_reported(tmp_path):
    manifest = _manifest(scene_count=1)
    artifact = _artifact(
        "proj-some-other-project", "audio/does-not-exist.wav", 5, "a" * 64, scene_id="scene-99"
    )

    result = verify_artifact(tmp_path, artifact, manifest)

    assert result.passed is False
    assert len(result.reasons) >= 3  # project mismatch, scene mismatch, missing file


# ---------------------------------------------------------------------
# batch helper
# ---------------------------------------------------------------------


def test_verify_artifacts_batch_preserves_order(tmp_path):
    manifest = _manifest(scene_count=2)
    size1, checksum1 = _write_file(tmp_path / "audio" / "scene-01.wav", b"one")
    artifact1 = _artifact(manifest.project_id, "audio/scene-01.wav", size1, checksum1, artifact_id="a1")
    artifact2 = _artifact(
        manifest.project_id, "audio/scene-02.wav", 999, "b" * 64,
        artifact_id="a2", scene_id="scene-02",
    )

    results = verify_artifacts(tmp_path, manifest, (artifact1, artifact2))

    assert [r.artifact_id for r in results] == ["a1", "a2"]
    assert results[0].passed is True
    assert results[1].passed is False


# ---------------------------------------------------------------------
# no-write / read-only guarantees
# ---------------------------------------------------------------------


def test_verify_artifact_writes_nothing_to_the_filesystem(tmp_path):
    manifest = _manifest()
    size, checksum = _write_file(tmp_path / "audio" / "scene-01.wav", b"hello world")
    artifact = _artifact(manifest.project_id, "audio/scene-01.wav", size, checksum)

    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    before_bytes = (tmp_path / "audio" / "scene-01.wav").read_bytes()

    verify_artifact(tmp_path, artifact, manifest)

    after = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    after_bytes = (tmp_path / "audio" / "scene-01.wav").read_bytes()

    assert before == after
    assert before_bytes == after_bytes


def test_verify_artifact_does_not_import_provider_or_render_modules(tmp_path, monkeypatch):
    manifest = _manifest()
    size, checksum = _write_file(tmp_path / "audio" / "scene-01.wav", b"hello world")
    artifact = _artifact(manifest.project_id, "audio/scene-01.wav", size, checksum)

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("src.providers") or name.startswith("src.render"):
            raise AssertionError(f"artifact_verifier must not import {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    result = verify_artifact(tmp_path, artifact, manifest)
    assert result.passed is True


# ---------------------------------------------------------------------
# compatibility with the state-machine verification gate, without
# automatically advancing any stage
# ---------------------------------------------------------------------


def test_verifier_result_composes_with_transition_project_gate(tmp_path):
    """verify_artifact() itself never calls transition_project() — its
    signature doesn't even accept a ProjectRecord — but a passed result
    can be fed into the existing verified=True gate by a future caller,
    and a failed result correctly keeps that gate shut."""
    manifest = _manifest()
    manifest_path = tmp_path / "manifest.json"
    project = create_initial_project(manifest_path, manifest, now=NOW)
    project, _t = transition_project(project, "audio_pending", now=NOW)

    size, checksum = _write_file(tmp_path / "audio" / "scene-01.wav", b"hello world")
    good_artifact = _artifact(manifest.project_id, "audio/scene-01.wav", size, checksum)
    good_result = verify_artifact(tmp_path, good_artifact, manifest)
    assert good_result.passed is True

    # A passed verification result unlocks the verified=True gate.
    advanced, _t = transition_project(project, "audio_ready", now=NOW, verified=good_result.passed)
    assert advanced.current_stage == "audio_ready"

    # A failed verification result does NOT unlock it, and verify_artifact
    # itself never tried to advance `project` at all (it is still at
    # "audio_pending" here, byte-for-byte as returned above).
    bad_artifact = _artifact(manifest.project_id, "audio/does-not-exist.wav", 5, "a" * 64)
    bad_result = verify_artifact(tmp_path, bad_artifact, manifest)
    assert bad_result.passed is False
    with pytest.raises(Exception):
        transition_project(project, "audio_ready", now=NOW, verified=bad_result.passed)
