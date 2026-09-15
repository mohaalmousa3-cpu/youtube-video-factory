"""Tests for the `finalize-scene-timing` CLI command (src/cli.py
cmd_finalize_scene_timing). Same isolated_db / _create_registered_project
pattern as tests/test_cli_register_audio_artifact.py. Uses the REAL local
ffmpeg/ffprobe binary (via the real register_audio_artifact()) to produce
genuinely measured audio artifacts — no provider, no network, no database
write from this command itself."""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

import src.cli as cli
from src.core.manifest_builder import build_video_manifest
from src.core.manifest_store import save_manifest
from src.core.project_state_machine import create_initial_project, initial_transition_for
from src.database.project_repository import create_project
from src.utils.channel_config import get_channel_policy

FIXED_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    from src.utils import config

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    config.get_settings.cache_clear()
    yield tmp_path
    config.get_settings.cache_clear()


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


def _create_registered_project(projects_root: Path, scene_ids: tuple[str, ...] = ("scene-01", "scene-02")):
    from src.database.db import get_connection, init_db

    manifest = build_video_manifest(
        _story_input_dict(), _scene_plan_dict(scene_ids), get_channel_policy(), created_at=FIXED_NOW
    )
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


def _real_wav(path: Path, duration_seconds: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from src.utils.config import get_settings

    ffmpeg = get_settings().ffmpeg_path
    result = subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", str(duration_seconds), str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def _register_real_audio(tmp_path, project_id, manifest, scene_id, duration_seconds) -> None:
    from src.core.audio_artifact_registrar import register_audio_artifact
    from src.database.db import get_connection
    from src.database.project_repository import get_project

    source = tmp_path / f"source-{scene_id}.wav"
    _real_wav(source, duration_seconds=duration_seconds)

    conn = get_connection()
    try:
        project = get_project(conn, project_id)
        result = register_audio_artifact(conn, project, manifest, scene_id, source, now=FIXED_NOW)
        assert result.ok, result.reasons
    finally:
        conn.close()


# ---------------------------------------------------------------------
# parser registration
# ---------------------------------------------------------------------


def test_parser_registration():
    parser = cli.build_parser()
    args = parser.parse_args(["finalize-scene-timing", "proj-123", "--output", "out.json"])
    assert args.func is cli.cmd_finalize_scene_timing
    assert args.project_id == "proj-123"
    assert args.output == "out.json"


def test_parser_requires_output():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["finalize-scene-timing", "proj-123"])


def test_parser_requires_project_id():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["finalize-scene-timing", "--output", "out.json"])


# ---------------------------------------------------------------------
# success
# ---------------------------------------------------------------------


def test_cli_success_populates_every_scene_and_preserves_identity(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_audio(tmp_path, project_id, manifest, "scene-01", 1.0)
    _register_real_audio(tmp_path, project_id, manifest, "scene-02", 2.0)

    output_path = tmp_path / "enriched.json"
    rc = cli.cmd_finalize_scene_timing(argparse.Namespace(project_id=project_id, output=str(output_path)))
    out = capsys.readouterr().out

    assert rc == 0
    assert "finalize-scene-timing: OK" in out
    assert f"project_id: {project_id}" in out
    assert "scene_count: 2" in out
    assert str(output_path) in out
    assert output_path.exists()

    from src.models.manifest import VideoManifest

    saved = VideoManifest.model_validate(json.loads(output_path.read_text(encoding="utf-8")))
    assert saved.project_id == manifest.project_id
    assert saved.source_fingerprint == manifest.source_fingerprint
    assert saved.measured_audio_duration_seconds is None
    durations = {s.scene_id: s.artifacts.measured_audio_duration_seconds for s in saved.scene_plan.scenes}
    assert durations["scene-01"] == pytest.approx(1.0, abs=0.3)
    assert durations["scene-02"] == pytest.approx(2.0, abs=0.3)


def test_cli_does_not_overwrite_the_canonical_manifest_path(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_audio(tmp_path, project_id, manifest, "scene-01", 1.0)
    _register_real_audio(tmp_path, project_id, manifest, "scene-02", 1.0)

    canonical_path = project_dir / "manifest.json"
    bytes_before = canonical_path.read_bytes()

    output_path = tmp_path / "enriched.json"
    cli.cmd_finalize_scene_timing(argparse.Namespace(project_id=project_id, output=str(output_path)))
    capsys.readouterr()

    assert canonical_path.read_bytes() == bytes_before


# ---------------------------------------------------------------------
# --output must not equal the project's own canonical manifest path
# ---------------------------------------------------------------------


def test_cli_rejects_output_equal_to_canonical_manifest_path(isolated_db, tmp_path, capsys, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_audio(tmp_path, project_id, manifest, "scene-01", 1.0)
    _register_real_audio(tmp_path, project_id, manifest, "scene-02", 1.0)

    canonical_path = project_dir / "manifest.json"
    bytes_before = canonical_path.read_bytes()

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("must not be called when --output equals the canonical manifest path")

    monkeypatch.setattr("src.core.scene_timing_finalizer.finalize_scene_timing", _must_not_be_called)
    monkeypatch.setattr("src.core.manifest_store.save_manifest", _must_not_be_called)

    rc = cli.cmd_finalize_scene_timing(
        argparse.Namespace(project_id=project_id, output=str(canonical_path))
    )
    err = capsys.readouterr().err

    assert rc == 1
    assert "finalize-scene-timing: FAILED" in err
    assert "must not be the project's own canonical manifest path" in err
    assert canonical_path.read_bytes() == bytes_before


def test_cli_rejects_output_equal_to_canonical_manifest_path_via_relative_dot_segments(
    isolated_db, tmp_path, capsys, monkeypatch
):
    """The same canonical path, spelled with '..'/'.' segments that
    Path.resolve() normalizes away — proving the comparison is a real
    normalized-path equality check, not a naive string comparison."""
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_audio(tmp_path, project_id, manifest, "scene-01", 1.0)
    _register_real_audio(tmp_path, project_id, manifest, "scene-02", 1.0)

    canonical_path = project_dir / "manifest.json"
    bytes_before = canonical_path.read_bytes()
    disguised_path = project_dir / "." / "manifest.json"

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("must not be called when --output equals the canonical manifest path")

    monkeypatch.setattr("src.core.scene_timing_finalizer.finalize_scene_timing", _must_not_be_called)
    monkeypatch.setattr("src.core.manifest_store.save_manifest", _must_not_be_called)

    rc = cli.cmd_finalize_scene_timing(
        argparse.Namespace(project_id=project_id, output=str(disguised_path))
    )
    err = capsys.readouterr().err

    assert rc == 1
    assert "must not be the project's own canonical manifest path" in err
    assert canonical_path.read_bytes() == bytes_before


def test_cli_distinct_output_path_still_succeeds_after_the_canonical_path_guard(isolated_db, tmp_path, capsys):
    """The new guard must reject only the exact canonical path — a
    genuinely different destination must continue to work exactly as
    before this fix."""
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_audio(tmp_path, project_id, manifest, "scene-01", 1.0)
    _register_real_audio(tmp_path, project_id, manifest, "scene-02", 1.0)

    output_path = tmp_path / "enriched.json"
    rc = cli.cmd_finalize_scene_timing(argparse.Namespace(project_id=project_id, output=str(output_path)))
    out = capsys.readouterr().out

    assert rc == 0
    assert "finalize-scene-timing: OK" in out
    assert output_path.exists()


def test_cli_never_writes_to_the_database(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_audio(tmp_path, project_id, manifest, "scene-01", 1.0)
    _register_real_audio(tmp_path, project_id, manifest, "scene-02", 1.0)

    from src.utils.config import get_settings

    db_path = get_settings().data_dir / "jobs.db"
    bytes_before = db_path.read_bytes()

    output_path = tmp_path / "enriched.json"
    cli.cmd_finalize_scene_timing(argparse.Namespace(project_id=project_id, output=str(output_path)))
    capsys.readouterr()

    assert db_path.read_bytes() == bytes_before


# ---------------------------------------------------------------------
# rejections -> non-zero exit, no output file, no DB touched
# ---------------------------------------------------------------------


def test_cli_missing_audio_for_one_scene_rejects_and_writes_nothing(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_audio(tmp_path, project_id, manifest, "scene-01", 1.0)
    # scene-02 has no registered audio artifact at all

    output_path = tmp_path / "enriched.json"
    rc = cli.cmd_finalize_scene_timing(argparse.Namespace(project_id=project_id, output=str(output_path)))
    err = capsys.readouterr().err

    assert rc == 1
    assert "finalize-scene-timing: FAILED" in err
    assert "scene-02" in err
    assert not output_path.exists()


def test_cli_duplicate_audio_for_one_scene_rejects_and_writes_nothing(isolated_db, tmp_path, capsys):
    from src.database.artifact_repository import register_artifact
    from src.database.db import get_connection
    from src.models.artifact import ArtifactRecord

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_audio(tmp_path, project_id, manifest, "scene-01", 1.0)
    _register_real_audio(tmp_path, project_id, manifest, "scene-02", 1.0)

    # Directly insert a SECOND "audio" record for scene-01, bypassing the
    # registrar's own one-per-scene guard, to construct the ambiguous case.
    conn = get_connection()
    try:
        register_artifact(
            conn,
            ArtifactRecord(
                artifact_id="audio-scene-01-extra",
                project_id=project_id,
                kind="audio",
                scene_id="scene-01",
                relative_path="audio/scene-01-extra.wav",
                byte_size=1,
                sha256_checksum="1" * 64,
                created_at=FIXED_NOW,
                metadata={"duration_seconds": 3.0, "source": "external"},
            ),
        )
    finally:
        conn.close()

    output_path = tmp_path / "enriched.json"
    rc = cli.cmd_finalize_scene_timing(argparse.Namespace(project_id=project_id, output=str(output_path)))
    err = capsys.readouterr().err

    assert rc == 1
    assert "finalize-scene-timing: FAILED" in err
    assert "scene-01" in err
    assert "2 eligible" in err
    assert not output_path.exists()


def test_cli_invalid_duration_metadata_rejects_and_writes_nothing(isolated_db, tmp_path, capsys):
    from src.database.artifact_repository import register_artifact
    from src.database.db import get_connection
    from src.models.artifact import ArtifactRecord

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects", scene_ids=("scene-01",))

    # Hand-built record with an invalid duration_seconds value — the real
    # registrar can never produce this, so it's inserted directly.
    conn = get_connection()
    try:
        register_artifact(
            conn,
            ArtifactRecord(
                artifact_id="audio-scene-01",
                project_id=project_id,
                kind="audio",
                scene_id="scene-01",
                relative_path="audio/scene-01.wav",
                byte_size=1,
                sha256_checksum="1" * 64,
                created_at=FIXED_NOW,
                metadata={"duration_seconds": -5.0, "source": "external"},
            ),
        )
    finally:
        conn.close()

    output_path = tmp_path / "enriched.json"
    rc = cli.cmd_finalize_scene_timing(argparse.Namespace(project_id=project_id, output=str(output_path)))
    err = capsys.readouterr().err

    assert rc == 1
    assert "non-positive duration_seconds" in err
    assert not output_path.exists()


def test_cli_unknown_project_returns_error(isolated_db, tmp_path, capsys):
    from src.database.db import init_db

    init_db()  # DB exists, but this project_id is not registered in it
    output_path = tmp_path / "enriched.json"

    rc = cli.cmd_finalize_scene_timing(argparse.Namespace(project_id="does-not-exist", output=str(output_path)))
    err = capsys.readouterr().err

    assert rc == 1
    assert "no project found" in err
    assert not output_path.exists()


def test_cli_missing_database_has_zero_filesystem_side_effects(isolated_db, tmp_path, capsys):
    assert list(isolated_db.iterdir()) == []
    output_path = tmp_path / "enriched.json"

    rc = cli.cmd_finalize_scene_timing(argparse.Namespace(project_id="proj-x", output=str(output_path)))
    err = capsys.readouterr().err

    assert rc == 1
    assert "finalize-scene-timing: FAILED" in err
    assert "no local project database found" in err
    assert list(isolated_db.iterdir()) == []
    assert not output_path.exists()


def test_cli_corrupt_manifest_file_rejects_and_writes_nothing(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_audio(tmp_path, project_id, manifest, "scene-01", 1.0)
    _register_real_audio(tmp_path, project_id, manifest, "scene-02", 1.0)

    (project_dir / "manifest.json").write_text("not valid json at all", encoding="utf-8")

    output_path = tmp_path / "enriched.json"
    rc = cli.cmd_finalize_scene_timing(argparse.Namespace(project_id=project_id, output=str(output_path)))
    err = capsys.readouterr().err

    assert rc == 1
    assert "finalize-scene-timing: FAILED" in err
    assert not output_path.exists()


def test_cli_stale_fingerprint_rejects_and_writes_nothing(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_audio(tmp_path, project_id, manifest, "scene-01", 1.0)
    _register_real_audio(tmp_path, project_id, manifest, "scene-02", 1.0)

    manifest_path = project_dir / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["source_fingerprint"] = "0" * 64  # tampered — no longer matches the DB row
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    output_path = tmp_path / "enriched.json"
    rc = cli.cmd_finalize_scene_timing(argparse.Namespace(project_id=project_id, output=str(output_path)))
    err = capsys.readouterr().err

    assert rc == 1
    assert "source_fingerprint does not match" in err
    assert not output_path.exists()


def test_cli_pre_existing_output_file_is_untouched_on_failure(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _register_real_audio(tmp_path, project_id, manifest, "scene-01", 1.0)
    # scene-02 missing -> guaranteed failure

    output_path = tmp_path / "enriched.json"
    output_path.write_bytes(b"pre-existing content nobody asked to touch")
    bytes_before = output_path.read_bytes()

    rc = cli.cmd_finalize_scene_timing(argparse.Namespace(project_id=project_id, output=str(output_path)))
    capsys.readouterr()

    assert rc == 1
    assert output_path.read_bytes() == bytes_before
