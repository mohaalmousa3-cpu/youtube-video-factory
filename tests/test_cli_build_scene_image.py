"""Tests for the `build-scene-image` CLI command (src/cli.py
cmd_build_scene_image). Same isolated_db / _create_registered_project
pattern as tests/test_cli_preview_scene_image_prompt.py. QwenImageProvider
is always faked at src.core.scene_image_generation.QwenImageProvider —
never constructed for real, never any real network call anywhere in this
file."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from PIL import Image

import src.cli as cli
from src.core.cost_guard import PaidApprovalRequiredError
from src.core.manifest_builder import build_video_manifest
from src.core.manifest_store import save_manifest
from src.core.project_state_machine import create_initial_project, initial_transition_for
from src.database.project_repository import create_project
from src.utils.channel_config import get_channel_policy

FIXED_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
GENERATION_MODULE = "src.core.scene_image_generation"


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


def _scene_plan_dict(scene_ids: tuple[str, ...] = ("scene-01",)) -> dict:
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
    projects_root: Path, scene_ids: tuple[str, ...] = ("scene-01",), story_id: str = "why-we-care-what-people-think"
):
    from src.database.db import get_connection, init_db

    manifest = build_video_manifest(
        _story_input_dict(story_id), _scene_plan_dict(scene_ids), get_channel_policy(), created_at=FIXED_NOW
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


def _write_real_png(path: Path) -> None:
    Image.new("RGB", (4, 4), color=(1, 2, 3)).save(path, format="PNG")


class _WritesValidPngProvider:
    def __init__(self):
        pass

    def generate_with_reference_and_download(self, prompt, reference_image_path, out_path, size="1328*1328"):
        _write_real_png(out_path)
        return out_path


def _explode_if_constructed(*args, **kwargs):
    raise AssertionError("QwenImageProvider must never be constructed once a preceding check has already failed")


def _args(
    project_id,
    scene_id,
    manifest_path,
    reference_image,
    output,
    include_character="true",
    include_color_anchor="true",
    proposals_path=None,
    out_format="text",
):
    return argparse.Namespace(
        project_id=project_id,
        scene_id=scene_id,
        manifest=str(manifest_path),
        reference_image=str(reference_image),
        output=str(output),
        include_character=include_character,
        include_color_anchor=include_color_anchor,
        proposals_path=proposals_path,
        format=out_format,
    )


# ---------------------------------------------------------------------
# parser registration
# ---------------------------------------------------------------------


def test_parser_registration():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "build-scene-image",
            "proj-123",
            "scene-01",
            "--manifest",
            "m.json",
            "--reference-image",
            "ref.png",
            "--output",
            "out.png",
            "--include-character",
            "true",
        ]
    )
    assert args.func is cli.cmd_build_scene_image
    assert args.project_id == "proj-123"
    assert args.scene_id == "scene-01"
    assert args.manifest == "m.json"
    assert args.reference_image == "ref.png"
    assert args.output == "out.png"
    assert args.include_character == "true"
    assert args.include_color_anchor == "true"
    assert args.proposals_path is None
    assert args.format == "text"


@pytest.mark.parametrize(
    "argv",
    [
        [
            "build-scene-image", "proj-123", "scene-01",
            "--reference-image", "ref.png", "--output", "out.png", "--include-character", "true",
        ],
        [
            "build-scene-image", "proj-123", "scene-01",
            "--manifest", "m.json", "--output", "out.png", "--include-character", "true",
        ],
        [
            "build-scene-image", "proj-123", "scene-01",
            "--manifest", "m.json", "--reference-image", "ref.png", "--include-character", "true",
        ],
        [
            "build-scene-image", "proj-123", "scene-01",
            "--manifest", "m.json", "--reference-image", "ref.png", "--output", "out.png",
        ],
    ],
)
def test_parser_requires_all_required_arguments(argv):
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(argv)


def test_existing_commands_remain_registered():
    parser = cli.build_parser()
    assert parser.parse_args(["preview-scene-image-prompt"] + [
        "p", "s", "--manifest", "m", "--reference-image", "r", "--output", "o", "--include-character", "true",
    ]).func is cli.cmd_preview_scene_image_prompt
    assert parser.parse_args([
        "build-upscaled-ken-burns", "p", "s", "--manifest", "m", "--output", "o",
    ]).func is cli.cmd_build_upscaled_ken_burns
    assert parser.parse_args(["health"]).func is cli.cmd_health


# ---------------------------------------------------------------------
# success
# ---------------------------------------------------------------------


def test_cli_success_text_mode(isolated_db, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(f"{GENERATION_MODULE}.QwenImageProvider", _WritesValidPngProvider)
    monkeypatch.setattr(f"{GENERATION_MODULE}.require_paid_approval", lambda *a, **k: None)

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"

    rc = cli.cmd_build_scene_image(
        _args(project_id, "scene-01", manifest_path, reference_image, output, include_character="true")
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert "build-scene-image: OK" in out
    assert f"project_id: {project_id}" in out
    assert "scene_id: scene-01" in out
    assert f"output: {output}" in out
    assert f"reference_image: {reference_image}" in out
    assert "include_character_anchor: True" in out
    assert "include_color_anchor: True" in out
    assert output.exists()
    with Image.open(output) as img:
        img.load()
        assert img.format == "PNG"


def test_cli_success_json_mode(isolated_db, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(f"{GENERATION_MODULE}.QwenImageProvider", _WritesValidPngProvider)
    monkeypatch.setattr(f"{GENERATION_MODULE}.require_paid_approval", lambda *a, **k: None)

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"

    rc = cli.cmd_build_scene_image(
        _args(
            project_id, "scene-01", manifest_path, reference_image, output,
            include_character="false", include_color_anchor="false", out_format="json",
        )
    )
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert rc == 0
    assert payload == {
        "ok": True,
        "project_id": project_id,
        "scene_id": "scene-01",
        "output": str(output),
        "reference_image": str(reference_image),
        "include_character_anchor": False,
        "include_color_anchor": False,
        "registered": False,
    }


# ---------------------------------------------------------------------
# default --proposals-path
# ---------------------------------------------------------------------


def test_cli_default_proposals_path_matches_data_dir_convention(isolated_db, tmp_path, capsys, monkeypatch):
    from src.utils.config import get_settings

    captured = {}

    def _spy_require(service_name, proposals_path, *, is_paid):
        captured["service_name"] = service_name
        captured["proposals_path"] = proposals_path
        captured["is_paid"] = is_paid

    monkeypatch.setattr(f"{GENERATION_MODULE}.QwenImageProvider", _WritesValidPngProvider)
    monkeypatch.setattr(f"{GENERATION_MODULE}.require_paid_approval", _spy_require)

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"

    rc = cli.cmd_build_scene_image(
        _args(project_id, "scene-01", manifest_path, reference_image, output, proposals_path=None)
    )
    capsys.readouterr()

    assert rc == 0
    assert captured["service_name"] == "qwen-image"
    assert captured["is_paid"] is True
    assert captured["proposals_path"] == get_settings().data_dir / "paid_proposals.json"


def test_cli_explicit_proposals_path_overrides_default(isolated_db, tmp_path, capsys, monkeypatch):
    captured = {}

    def _spy_require(service_name, proposals_path, *, is_paid):
        captured["proposals_path"] = proposals_path

    monkeypatch.setattr(f"{GENERATION_MODULE}.QwenImageProvider", _WritesValidPngProvider)
    monkeypatch.setattr(f"{GENERATION_MODULE}.require_paid_approval", _spy_require)

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"
    custom_proposals = tmp_path / "custom-proposals.json"

    rc = cli.cmd_build_scene_image(
        _args(
            project_id, "scene-01", manifest_path, reference_image, output,
            proposals_path=str(custom_proposals),
        )
    )
    capsys.readouterr()

    assert rc == 0
    assert captured["proposals_path"] == custom_proposals


# ---------------------------------------------------------------------
# paid-approval denial — blocks before provider construction, reported as
# a clean ordinary CLI failure (rc=1, _fail() text/JSON), never propagates
# out of cmd_build_scene_image, no traceback, no output
# ---------------------------------------------------------------------


def test_cli_paid_approval_denial_reports_clean_failure_text_mode(isolated_db, tmp_path, capsys, monkeypatch):
    def _fake_require(service_name, proposals_path, *, is_paid):
        raise PaidApprovalRequiredError("denied")

    monkeypatch.setattr(f"{GENERATION_MODULE}.require_paid_approval", _fake_require)
    monkeypatch.setattr(f"{GENERATION_MODULE}.QwenImageProvider", _explode_if_constructed)

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"

    rc = cli.cmd_build_scene_image(_args(project_id, "scene-01", manifest_path, reference_image, output))
    err = capsys.readouterr().err

    assert rc == 1
    assert "build-scene-image: FAILED — qwen-image is not approved for a paid provider call" in err
    assert "Traceback" not in err
    assert not output.exists()


def test_cli_paid_approval_denial_reports_clean_failure_json_mode(isolated_db, tmp_path, capsys, monkeypatch):
    def _fake_require(service_name, proposals_path, *, is_paid):
        raise PaidApprovalRequiredError("denied")

    monkeypatch.setattr(f"{GENERATION_MODULE}.require_paid_approval", _fake_require)
    monkeypatch.setattr(f"{GENERATION_MODULE}.QwenImageProvider", _explode_if_constructed)

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"

    rc = cli.cmd_build_scene_image(
        _args(project_id, "scene-01", manifest_path, reference_image, output, out_format="json")
    )
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert rc == 1
    assert payload == {
        "ok": False,
        "project_id": project_id,
        "scene_id": "scene-01",
        "reason": "qwen-image is not approved for a paid provider call",
    }
    assert not output.exists()


# ---------------------------------------------------------------------
# rejections that must block before provider construction
# ---------------------------------------------------------------------


def test_cli_unknown_project_returns_error(isolated_db, tmp_path, capsys, monkeypatch):
    from src.database.db import init_db

    monkeypatch.setattr(f"{GENERATION_MODULE}.QwenImageProvider", _explode_if_constructed)
    init_db()
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"

    rc = cli.cmd_build_scene_image(_args("does-not-exist", "scene-01", manifest_path, reference_image, output))
    err = capsys.readouterr().err

    assert rc == 1
    assert "unknown project_id" in err
    assert not output.exists()


def test_cli_missing_database_returns_error(isolated_db, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(f"{GENERATION_MODULE}.QwenImageProvider", _explode_if_constructed)
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"

    rc = cli.cmd_build_scene_image(_args("proj-x", "scene-01", manifest_path, reference_image, output))
    err = capsys.readouterr().err

    assert rc == 1
    assert "no local project database found" in err


def test_cli_malformed_manifest_file_rejects(isolated_db, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(f"{GENERATION_MODULE}.QwenImageProvider", _explode_if_constructed)
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = tmp_path / "bad.json"
    manifest_path.write_text("not valid json", encoding="utf-8")
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"

    rc = cli.cmd_build_scene_image(_args(project_id, "scene-01", manifest_path, reference_image, output))
    err = capsys.readouterr().err

    assert rc == 1
    assert "could not load manifest" in err


def test_cli_wrong_project_manifest_rejects(isolated_db, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(f"{GENERATION_MODULE}.QwenImageProvider", _explode_if_constructed)
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    other_id, other_dir, other_manifest = _create_registered_project(
        tmp_path / "projects2", story_id="a-completely-different-story"
    )
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"

    rc = cli.cmd_build_scene_image(
        _args(project_id, "scene-01", other_dir / "manifest.json", reference_image, output)
    )
    err = capsys.readouterr().err

    assert rc == 1
    assert "does not match project_id" in err


def test_cli_stale_fingerprint_manifest_rejects(isolated_db, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(f"{GENERATION_MODULE}.QwenImageProvider", _explode_if_constructed)
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["source_fingerprint"] = "0" * 64
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(raw), encoding="utf-8")
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"

    rc = cli.cmd_build_scene_image(_args(project_id, "scene-01", tampered_path, reference_image, output))
    err = capsys.readouterr().err

    assert rc == 1
    assert "source_fingerprint does not match" in err


def test_cli_unknown_scene_rejects(isolated_db, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(f"{GENERATION_MODULE}.QwenImageProvider", _explode_if_constructed)
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"

    rc = cli.cmd_build_scene_image(
        _args(project_id, "scene-does-not-exist", manifest_path, reference_image, output)
    )
    err = capsys.readouterr().err

    assert rc == 1
    assert "unknown scene_id" in err


def test_cli_missing_reference_image_rejects(isolated_db, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(f"{GENERATION_MODULE}.QwenImageProvider", _explode_if_constructed)
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    reference_image = tmp_path / "does-not-exist.png"
    output = tmp_path / "out.png"

    rc = cli.cmd_build_scene_image(_args(project_id, "scene-01", manifest_path, reference_image, output))
    err = capsys.readouterr().err

    assert rc == 1
    assert "--reference-image does not exist" in err
    assert not output.exists()


def test_cli_directory_reference_image_rejects(isolated_db, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(f"{GENERATION_MODULE}.QwenImageProvider", _explode_if_constructed)
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    reference_image = tmp_path / "a-directory"
    reference_image.mkdir()
    output = tmp_path / "out.png"

    rc = cli.cmd_build_scene_image(_args(project_id, "scene-01", manifest_path, reference_image, output))
    err = capsys.readouterr().err

    assert rc == 1
    assert "--reference-image is not a regular file" in err


def test_cli_output_parent_missing_rejects(isolated_db, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(f"{GENERATION_MODULE}.QwenImageProvider", _explode_if_constructed)
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "no-such-dir" / "out.png"

    rc = cli.cmd_build_scene_image(_args(project_id, "scene-01", manifest_path, reference_image, output))
    err = capsys.readouterr().err

    assert rc == 1
    assert "--output parent directory does not exist" in err


def test_cli_output_already_exists_rejects_and_is_never_overwritten(isolated_db, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(f"{GENERATION_MODULE}.QwenImageProvider", _explode_if_constructed)
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"
    original_bytes = b"already here, must not change"
    output.write_bytes(original_bytes)

    rc = cli.cmd_build_scene_image(_args(project_id, "scene-01", manifest_path, reference_image, output))
    err = capsys.readouterr().err

    assert rc == 1
    assert "--output already exists" in err
    assert output.read_bytes() == original_bytes


# ---------------------------------------------------------------------
# invalid generated PNG
# ---------------------------------------------------------------------


def test_cli_non_png_provider_output_deletes_file_and_reports_failure(isolated_db, tmp_path, capsys, monkeypatch):
    class _WritesGarbageProvider:
        def __init__(self):
            pass

        def generate_with_reference_and_download(self, prompt, reference_image_path, out_path, size="1328*1328"):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"garbage bytes")
            return out_path

    monkeypatch.setattr(f"{GENERATION_MODULE}.QwenImageProvider", _WritesGarbageProvider)
    monkeypatch.setattr(f"{GENERATION_MODULE}.require_paid_approval", lambda *a, **k: None)

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"

    rc = cli.cmd_build_scene_image(_args(project_id, "scene-01", manifest_path, reference_image, output))
    err = capsys.readouterr().err

    assert rc == 1
    assert "could not be decoded as an image" in err
    assert not output.exists()


def test_cli_invalid_format_rejects(isolated_db, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(f"{GENERATION_MODULE}.QwenImageProvider", _explode_if_constructed)
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"

    rc = cli.cmd_build_scene_image(
        _args("proj-x", "scene-01", manifest_path, reference_image, output, out_format="xml")
    )
    err = capsys.readouterr().err

    assert rc == 1
    assert "invalid --format" in err


def test_cli_json_failure_has_stable_shape(isolated_db, tmp_path, capsys, monkeypatch):
    from src.database.db import init_db

    monkeypatch.setattr(f"{GENERATION_MODULE}.QwenImageProvider", _explode_if_constructed)
    init_db()
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"

    rc = cli.cmd_build_scene_image(
        _args("does-not-exist", "scene-01", manifest_path, reference_image, output, out_format="json")
    )
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert rc == 1
    assert payload["ok"] is False
    assert payload["project_id"] == "does-not-exist"
    assert payload["scene_id"] == "scene-01"
    assert isinstance(payload["reason"], str) and payload["reason"]


# ---------------------------------------------------------------------
# no manifest/SQLite/artifact mutation, success or failure
# ---------------------------------------------------------------------


def test_cli_never_writes_to_the_database_on_success(isolated_db, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(f"{GENERATION_MODULE}.QwenImageProvider", _WritesValidPngProvider)
    monkeypatch.setattr(f"{GENERATION_MODULE}.require_paid_approval", lambda *a, **k: None)

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"

    from src.utils.config import get_settings

    db_path = get_settings().data_dir / "jobs.db"
    bytes_before = db_path.read_bytes()
    manifest_bytes_before = manifest_path.read_bytes()

    rc = cli.cmd_build_scene_image(_args(project_id, "scene-01", manifest_path, reference_image, output))
    capsys.readouterr()

    assert rc == 0
    assert db_path.read_bytes() == bytes_before
    assert manifest_path.read_bytes() == manifest_bytes_before

    from src.database.artifact_repository import list_artifacts_by_scene
    from src.database.db import get_connection

    conn = get_connection()
    try:
        artifacts = list_artifacts_by_scene(conn, project_id, "scene-01")
    finally:
        conn.close()
    assert artifacts == []


def test_cli_never_writes_to_the_database_on_failure(isolated_db, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(f"{GENERATION_MODULE}.QwenImageProvider", _explode_if_constructed)
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"
    output.write_bytes(b"already here")

    from src.utils.config import get_settings

    db_path = get_settings().data_dir / "jobs.db"
    bytes_before = db_path.read_bytes()

    rc = cli.cmd_build_scene_image(_args(project_id, "scene-01", manifest_path, reference_image, output))
    capsys.readouterr()

    assert rc == 1
    assert db_path.read_bytes() == bytes_before
