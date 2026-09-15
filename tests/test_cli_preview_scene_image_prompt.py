"""Tests for the `preview-scene-image-prompt` CLI command (src/cli.py
cmd_preview_scene_image_prompt). Same isolated_db / _create_registered_project
pattern as tests/test_cli_mouth_animation_preflight.py, minus visual/audio
artifact registration — this command never looks up an artifact. No
provider, no image decode, no file creation, no network anywhere in this
file."""
from __future__ import annotations

import argparse
import json
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


def _args(
    project_id,
    scene_id,
    manifest_path,
    reference_image,
    output,
    include_character="true",
    include_color_anchor="true",
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
        format=out_format,
    )


# ---------------------------------------------------------------------
# parser registration
# ---------------------------------------------------------------------


def test_parser_registration():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "preview-scene-image-prompt",
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
    assert args.func is cli.cmd_preview_scene_image_prompt
    assert args.project_id == "proj-123"
    assert args.scene_id == "scene-01"
    assert args.manifest == "m.json"
    assert args.reference_image == "ref.png"
    assert args.output == "out.png"
    assert args.include_character == "true"
    assert args.include_color_anchor == "true"
    assert args.format == "text"


@pytest.mark.parametrize(
    "argv",
    [
        [
            "preview-scene-image-prompt",
            "proj-123",
            "scene-01",
            "--reference-image",
            "ref.png",
            "--output",
            "out.png",
            "--include-character",
            "true",
        ],
        [
            "preview-scene-image-prompt",
            "proj-123",
            "scene-01",
            "--manifest",
            "m.json",
            "--output",
            "out.png",
            "--include-character",
            "true",
        ],
        [
            "preview-scene-image-prompt",
            "proj-123",
            "scene-01",
            "--manifest",
            "m.json",
            "--reference-image",
            "ref.png",
            "--include-character",
            "true",
        ],
        [
            "preview-scene-image-prompt",
            "proj-123",
            "scene-01",
            "--manifest",
            "m.json",
            "--reference-image",
            "ref.png",
            "--output",
            "out.png",
        ],
    ],
)
def test_parser_requires_all_required_arguments(argv):
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(argv)


def test_parser_rejects_invalid_include_character_choice():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "preview-scene-image-prompt",
                "proj-123",
                "scene-01",
                "--manifest",
                "m.json",
                "--reference-image",
                "ref.png",
                "--output",
                "out.png",
                "--include-character",
                "yes",
            ]
        )


# ---------------------------------------------------------------------
# success
# ---------------------------------------------------------------------


def test_cli_success_text_mode(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"

    rc = cli.cmd_preview_scene_image_prompt(
        _args(project_id, "scene-01", manifest_path, reference_image, output, include_character="true")
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert "preview-scene-image-prompt: OK" in out
    assert f"PROJECT_ID: {project_id}" in out
    assert "SCENE_ID: scene-01" in out
    assert "INCLUDE_CHARACTER_ANCHOR: True" in out
    assert "INCLUDE_COLOR_ANCHOR: True" in out
    assert f"REFERENCE_IMAGE: {reference_image}" in out
    assert f"PLANNED_OUTPUT: {output}" in out
    assert "FINAL_PROMPT:" in out
    assert "Visual brief for scene-01." in out
    assert not output.exists()


def test_cli_success_json_mode(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"

    rc = cli.cmd_preview_scene_image_prompt(
        _args(
            project_id,
            "scene-01",
            manifest_path,
            reference_image,
            output,
            include_character="false",
            include_color_anchor="false",
            out_format="json",
        )
    )
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert rc == 0
    from src.core.scene_image_prompt import build_scene_image_prompt

    expected_prompt = build_scene_image_prompt(
        "Visual brief for scene-01.", include_character_anchor=False, include_color_anchor=False
    )
    assert payload == {
        "ok": True,
        "project_id": project_id,
        "scene_id": "scene-01",
        "visual_brief": "Visual brief for scene-01.",
        "final_prompt": expected_prompt,
        "reference_image_path": str(reference_image),
        "planned_output_path": str(output),
        "include_character_anchor": False,
        "include_color_anchor": False,
    }
    assert not output.exists()


# ---------------------------------------------------------------------
# rejections -> rc=1, no DB/file mutation
# ---------------------------------------------------------------------


def test_cli_unknown_project_returns_error(isolated_db, tmp_path, capsys):
    from src.database.db import init_db

    init_db()
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"

    rc = cli.cmd_preview_scene_image_prompt(
        _args("does-not-exist", "scene-01", manifest_path, reference_image, output)
    )
    err = capsys.readouterr().err

    assert rc == 1
    assert "unknown project_id" in err


def test_cli_missing_database_returns_error(isolated_db, tmp_path, capsys):
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"

    rc = cli.cmd_preview_scene_image_prompt(_args("proj-x", "scene-01", manifest_path, reference_image, output))
    err = capsys.readouterr().err

    assert rc == 1
    assert "no local project database found" in err


def test_cli_malformed_manifest_file_rejects(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = tmp_path / "bad.json"
    manifest_path.write_text("not valid json", encoding="utf-8")
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"

    rc = cli.cmd_preview_scene_image_prompt(_args(project_id, "scene-01", manifest_path, reference_image, output))
    err = capsys.readouterr().err

    assert rc == 1
    assert "could not load manifest" in err


def test_cli_wrong_project_manifest_rejects(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    other_id, other_dir, other_manifest = _create_registered_project(
        tmp_path / "projects2", story_id="a-completely-different-story"
    )
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"

    rc = cli.cmd_preview_scene_image_prompt(
        _args(project_id, "scene-01", other_dir / "manifest.json", reference_image, output)
    )
    err = capsys.readouterr().err

    assert rc == 1
    assert "does not match project_id" in err


def test_cli_stale_fingerprint_manifest_rejects(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["source_fingerprint"] = "0" * 64
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(raw), encoding="utf-8")
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"

    rc = cli.cmd_preview_scene_image_prompt(_args(project_id, "scene-01", tampered_path, reference_image, output))
    err = capsys.readouterr().err

    assert rc == 1
    assert "source_fingerprint does not match" in err


def test_cli_unknown_scene_rejects(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"

    rc = cli.cmd_preview_scene_image_prompt(
        _args(project_id, "scene-does-not-exist", manifest_path, reference_image, output)
    )
    err = capsys.readouterr().err

    assert rc == 1
    assert "unknown scene_id" in err


def test_cli_missing_reference_image_rejects(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    reference_image = tmp_path / "does-not-exist.png"
    output = tmp_path / "out.png"

    rc = cli.cmd_preview_scene_image_prompt(_args(project_id, "scene-01", manifest_path, reference_image, output))
    err = capsys.readouterr().err

    assert rc == 1
    assert "--reference-image does not exist" in err


def test_cli_directory_reference_image_rejects(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    reference_image = tmp_path / "a-directory"
    reference_image.mkdir()
    output = tmp_path / "out.png"

    rc = cli.cmd_preview_scene_image_prompt(_args(project_id, "scene-01", manifest_path, reference_image, output))
    err = capsys.readouterr().err

    assert rc == 1
    assert "--reference-image is not a regular file" in err


def test_cli_output_parent_missing_rejects(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "no-such-dir" / "out.png"

    rc = cli.cmd_preview_scene_image_prompt(_args(project_id, "scene-01", manifest_path, reference_image, output))
    err = capsys.readouterr().err

    assert rc == 1
    assert "--output parent directory does not exist" in err


def test_cli_output_already_exists_rejects(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"
    output.write_bytes(b"already here")

    rc = cli.cmd_preview_scene_image_prompt(_args(project_id, "scene-01", manifest_path, reference_image, output))
    err = capsys.readouterr().err

    assert rc == 1
    assert "--output already exists" in err


def test_cli_invalid_format_rejects(isolated_db, tmp_path, capsys):
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"

    rc = cli.cmd_preview_scene_image_prompt(
        _args("proj-x", "scene-01", manifest_path, reference_image, output, out_format="xml")
    )
    err = capsys.readouterr().err

    assert rc == 1
    assert "invalid --format" in err


def test_cli_json_failure_has_stable_shape(isolated_db, tmp_path, capsys):
    from src.database.db import init_db

    init_db()
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"

    rc = cli.cmd_preview_scene_image_prompt(
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
# read-only guarantees
# ---------------------------------------------------------------------


def test_cli_never_writes_to_the_database(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"

    from src.utils.config import get_settings

    db_path = get_settings().data_dir / "jobs.db"
    bytes_before = db_path.read_bytes()

    cli.cmd_preview_scene_image_prompt(_args(project_id, "scene-01", manifest_path, reference_image, output))
    capsys.readouterr()

    assert db_path.read_bytes() == bytes_before


def test_cli_never_creates_the_output_file(isolated_db, tmp_path, capsys):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"

    rc = cli.cmd_preview_scene_image_prompt(_args(project_id, "scene-01", manifest_path, reference_image, output))
    capsys.readouterr()

    assert rc == 0
    assert not output.exists()


def test_cli_closes_readonly_connection_before_preview_work(isolated_db, tmp_path, capsys, monkeypatch):
    """Same closure-proof technique as
    test_cli_mouth_animation_preflight.py's analogous test: a closed
    sqlite3 connection raises ProgrammingError on further use, so a mocked
    build_scene_image_preview() tries to reuse the CLI's own connection
    object and records whether that raised."""
    import sqlite3

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    output = tmp_path / "out.png"

    import src.database.db as db_module

    real_get_readonly_connection = db_module.get_readonly_connection
    holder = {}

    def _spy_get_readonly_connection():
        conn = real_get_readonly_connection()
        holder["conn"] = conn
        return conn

    monkeypatch.setattr("src.database.db.get_readonly_connection", _spy_get_readonly_connection)

    was_closed = {"value": None}

    def _fake_build_scene_image_preview(*a, **k):
        conn = holder["conn"]
        try:
            conn.execute("SELECT 1")
            was_closed["value"] = False
        except sqlite3.ProgrammingError:
            was_closed["value"] = True
        from src.core.scene_image_preview import ScenePreviewResult

        return ScenePreviewResult(
            project_id=project_id,
            scene_id="scene-01",
            visual_brief="x",
            final_prompt="x",
            reference_image_path=reference_image,
            planned_output_path=output,
            include_character_anchor=True,
            include_color_anchor=True,
        )

    monkeypatch.setattr(
        "src.core.scene_image_preview.build_scene_image_preview", _fake_build_scene_image_preview
    )

    rc = cli.cmd_preview_scene_image_prompt(_args(project_id, "scene-01", manifest_path, reference_image, output))
    capsys.readouterr()

    assert rc == 0
    assert was_closed["value"] is True


def test_cli_no_import_of_any_provider_module():
    """Source-inspection proof (not just trust) that cmd_preview_scene_image_prompt
    never imports a provider module anywhere in its body."""
    import ast
    import inspect

    source = inspect.getsource(cli.cmd_preview_scene_image_prompt)
    tree = ast.parse(source)
    import_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]

    offending = [node for node in import_nodes if node.module is not None and node.module.startswith("src.providers")]

    assert offending == []
