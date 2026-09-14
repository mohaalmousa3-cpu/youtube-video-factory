"""Tests for src/cli.py's Phase 1E commands: validate-input, create-project,
import-queue, list-projects, status, resume-plan, validate-manifest — plus
a registration-only check that `health`/`init-db` are unchanged. No
network, no provider call (cmd_health itself is never invoked).

Command handlers are called directly with a plain argparse.Namespace,
never via subprocess, per the "independently testable" requirement. Every
test that touches the database monkeypatches PROJECT_ROOT to tmp_path
(same pattern as tests/test_job.py) so nothing here ever reads or writes a
real developer path."""
import argparse
import json
import sqlite3

import pytest

import src.cli as cli


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    from src.utils import config

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    config.get_settings.cache_clear()
    yield tmp_path
    config.get_settings.cache_clear()


def _story_input_dict(**overrides) -> dict:
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


def _scene_plan_dict(**overrides) -> dict:
    data = dict(
        scenes=[
            dict(
                scene_id="scene-01",
                sequence=1,
                narration_text="Why does being left out sting so much?",
                scene_type="establishing",
                narrative_beat="hook",
                visual_brief="Stickman character alone on a quiet street corner at dusk.",
                motion_mode="in",
                approval_state="approved",
            ),
        ],
        role_outfits=[],
    )
    data.update(overrides)
    return data


def _write_json(path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _write_story_and_scenes(tmp_path, **overrides) -> tuple:
    story_path = tmp_path / "story.json"
    scenes_path = tmp_path / "scenes.json"
    _write_json(story_path, _story_input_dict(**overrides.get("story", {})))
    _write_json(scenes_path, _scene_plan_dict(**overrides.get("scenes", {})))
    return story_path, scenes_path


def _write_scenes_only(tmp_path, **overrides):
    """Writes only scenes.json — for tests that deliberately put invalid
    content in story.json first and must not have it clobbered by a
    subsequent call to _write_story_and_scenes()."""
    scenes_path = tmp_path / "scenes.json"
    _write_json(scenes_path, _scene_plan_dict(**overrides))
    return scenes_path


# ---------------------------------------------------------------------
# health / init-db unchanged
# ---------------------------------------------------------------------


def test_health_and_init_db_remain_registered_without_executing_health():
    parser = cli.build_parser()

    health_args = parser.parse_args(["health"])
    assert health_args.func is cli.cmd_health  # never actually called — no provider contact

    init_db_args = parser.parse_args(["init-db"])
    assert init_db_args.func is cli.cmd_init_db


def test_cmd_init_db_creates_all_tables(isolated_db):
    from src.database.db import get_connection

    assert cli.cmd_init_db(argparse.Namespace()) == 0

    conn = get_connection()
    try:
        tables = {
            row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
    finally:
        conn.close()
    assert {"jobs", "projects", "project_transitions"}.issubset(tables)


# ---------------------------------------------------------------------
# validate-input
# ---------------------------------------------------------------------


def test_cmd_validate_input_succeeds_and_writes_nothing(tmp_path, capsys):
    story_path, scenes_path = _write_story_and_scenes(tmp_path)

    rc = cli.cmd_validate_input(argparse.Namespace(story=str(story_path), scenes=str(scenes_path)))

    assert rc == 0
    out = capsys.readouterr().out
    assert "project_id" in out
    assert "story_id: why-we-care-what-people-think" in out
    # Nothing was written anywhere besides the two input files we made.
    assert set(p.name for p in tmp_path.iterdir()) == {"story.json", "scenes.json"}


def test_cmd_validate_input_rejects_malformed_json(tmp_path, capsys):
    story_path = tmp_path / "story.json"
    story_path.write_text("{not valid json")
    scenes_path = _write_scenes_only(tmp_path)

    rc = cli.cmd_validate_input(argparse.Namespace(story=str(story_path), scenes=str(scenes_path)))

    assert rc == 1
    assert "FAILED" in capsys.readouterr().err


def test_cmd_validate_input_rejects_non_object_json(tmp_path, capsys):
    story_path = tmp_path / "story.json"
    story_path.write_text("[1, 2, 3]")
    scenes_path = _write_scenes_only(tmp_path)

    rc = cli.cmd_validate_input(argparse.Namespace(story=str(story_path), scenes=str(scenes_path)))

    assert rc == 1
    assert "FAILED" in capsys.readouterr().err


def test_cmd_validate_input_rejects_unapproved_story(tmp_path, capsys):
    story_path, scenes_path = _write_story_and_scenes(tmp_path, story={"approval_status": "draft"})

    rc = cli.cmd_validate_input(argparse.Namespace(story=str(story_path), scenes=str(scenes_path)))

    assert rc == 1
    assert "FAILED" in capsys.readouterr().err


def test_cmd_validate_input_rejects_unapproved_scene(tmp_path, capsys):
    story_path, scenes_path = _write_story_and_scenes(
        tmp_path,
        scenes={
            "scenes": [
                dict(
                    scene_id="scene-01", sequence=1, narration_text="Text.", scene_type="establishing",
                    narrative_beat="hook", visual_brief="Brief.", motion_mode="in", approval_state="draft",
                )
            ]
        },
    )

    rc = cli.cmd_validate_input(argparse.Namespace(story=str(story_path), scenes=str(scenes_path)))

    assert rc == 1
    assert "FAILED" in capsys.readouterr().err


def test_cmd_validate_input_rejects_non_english_viewer_facing_field(tmp_path, capsys):
    story_path, scenes_path = _write_story_and_scenes(tmp_path, story={"viewer_facing_language": "Arabic"})

    rc = cli.cmd_validate_input(argparse.Namespace(story=str(story_path), scenes=str(scenes_path)))

    assert rc == 1
    assert "FAILED" in capsys.readouterr().err


def test_cmd_validate_input_rejects_wrong_overlay(tmp_path, capsys):
    story_path, scenes_path = _write_story_and_scenes(
        tmp_path,
        scenes={
            "scenes": [
                dict(
                    scene_id="scene-01", sequence=1, narration_text="Text.", scene_type="establishing",
                    narrative_beat="hook", visual_brief="Brief.", motion_mode="in", approval_state="approved",
                    text_overlays=[
                        dict(text="Bad", position="top", style_id="s1", viewer_facing_language="English", deterministic=False)
                    ],
                )
            ]
        },
    )

    rc = cli.cmd_validate_input(argparse.Namespace(story=str(story_path), scenes=str(scenes_path)))

    assert rc == 1
    assert "FAILED" in capsys.readouterr().err


def test_cmd_validate_input_rejects_invalid_flow_fallback(tmp_path, capsys):
    story_path, scenes_path = _write_story_and_scenes(
        tmp_path,
        scenes={
            "scenes": [
                dict(
                    scene_id="scene-01", sequence=1, narration_text="Text.", scene_type="establishing",
                    narrative_beat="hook", visual_brief="Brief.", motion_mode="manual_flow", approval_state="approved",
                )
            ]
        },
    )

    rc = cli.cmd_validate_input(argparse.Namespace(story=str(story_path), scenes=str(scenes_path)))

    assert rc == 1
    assert "FAILED" in capsys.readouterr().err


# ---------------------------------------------------------------------
# create-project
# ---------------------------------------------------------------------


def test_cmd_create_project_creates_manifest_and_record(isolated_db, tmp_path, capsys):
    story_path, scenes_path = _write_story_and_scenes(tmp_path)
    output_root = tmp_path / "projects"

    rc = cli.cmd_create_project(
        argparse.Namespace(story=str(story_path), scenes=str(scenes_path), output_root=str(output_root))
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "current_stage: planned" in out

    from src.database.db import get_connection
    from src.database.project_repository import get_project

    conn = get_connection()
    try:
        project_id = out.splitlines()[1].split(": ", 1)[1].strip()
        project = get_project(conn, project_id)
    finally:
        conn.close()

    assert project is not None
    assert project.current_stage == "planned"
    assert (output_root / project.project_id / "manifest.json").exists()


def test_cmd_create_project_does_not_auto_advance_past_planned(isolated_db, tmp_path):
    story_path, scenes_path = _write_story_and_scenes(tmp_path)
    output_root = tmp_path / "projects"

    cli.cmd_create_project(
        argparse.Namespace(story=str(story_path), scenes=str(scenes_path), output_root=str(output_root))
    )

    from src.database.db import get_connection
    from src.database.project_repository import list_projects

    conn = get_connection()
    try:
        projects = list_projects(conn)
    finally:
        conn.close()

    assert len(projects) == 1
    assert projects[0].current_stage == "planned"
    assert projects[0].lifecycle_version == 1


def test_cmd_create_project_rejects_existing_manifest_and_duplicate_id(isolated_db, tmp_path, capsys):
    story_path, scenes_path = _write_story_and_scenes(tmp_path)
    output_root = tmp_path / "projects"
    ns = argparse.Namespace(story=str(story_path), scenes=str(scenes_path), output_root=str(output_root))

    assert cli.cmd_create_project(ns) == 0
    capsys.readouterr()

    rc = cli.cmd_create_project(ns)  # identical inputs -> identical project_id -> conflict

    assert rc == 1
    assert "FAILED" in capsys.readouterr().err

    from src.database.db import get_connection
    from src.database.project_repository import list_projects

    conn = get_connection()
    try:
        projects = list_projects(conn)
    finally:
        conn.close()
    assert len(projects) == 1  # the rejected second attempt created nothing


# ---------------------------------------------------------------------
# list-projects / status / resume-plan / validate-manifest
# ---------------------------------------------------------------------


def test_cmd_list_projects_empty_db(isolated_db, capsys):
    rc = cli.cmd_list_projects(argparse.Namespace())
    assert rc == 0
    assert "No projects found." in capsys.readouterr().out


def test_cmd_list_projects_populated_db(isolated_db, tmp_path, capsys):
    story_path, scenes_path = _write_story_and_scenes(tmp_path)
    output_root = tmp_path / "projects"
    cli.cmd_create_project(
        argparse.Namespace(story=str(story_path), scenes=str(scenes_path), output_root=str(output_root))
    )
    capsys.readouterr()

    rc = cli.cmd_list_projects(argparse.Namespace())

    assert rc == 0
    out = capsys.readouterr().out
    assert "stage=planned" in out


def test_cmd_status_includes_transitions(isolated_db, tmp_path, capsys):
    story_path, scenes_path = _write_story_and_scenes(tmp_path)
    output_root = tmp_path / "projects"
    cli.cmd_create_project(
        argparse.Namespace(story=str(story_path), scenes=str(scenes_path), output_root=str(output_root))
    )
    create_out = capsys.readouterr().out
    project_id = create_out.splitlines()[1].split(": ", 1)[1].strip()

    rc = cli.cmd_status(argparse.Namespace(project_id=project_id))

    assert rc == 0
    out = capsys.readouterr().out
    assert f"project_id: {project_id}" in out
    assert "transitions:" in out
    assert "planned -> planned" in out


def test_cmd_status_missing_project_returns_error(isolated_db, capsys):
    rc = cli.cmd_status(argparse.Namespace(project_id="does-not-exist"))
    assert rc == 1
    assert "FAILED" in capsys.readouterr().err


def test_cmd_resume_plan_reports_without_mutating_db(isolated_db, tmp_path, capsys):
    story_path, scenes_path = _write_story_and_scenes(tmp_path)
    output_root = tmp_path / "projects"
    cli.cmd_create_project(
        argparse.Namespace(story=str(story_path), scenes=str(scenes_path), output_root=str(output_root))
    )
    create_out = capsys.readouterr().out
    project_id = create_out.splitlines()[1].split(": ", 1)[1].strip()

    from src.database.db import get_connection
    from src.database.project_repository import get_project

    conn = get_connection()
    try:
        before = get_project(conn, project_id)
    finally:
        conn.close()

    rc = cli.cmd_resume_plan(argparse.Namespace(project_id=project_id))

    assert rc == 0
    out = capsys.readouterr().out
    assert "action: start_stage" in out
    assert "next_stage: audio_pending" in out

    conn = get_connection()
    try:
        after = get_project(conn, project_id)
    finally:
        conn.close()
    assert before == after  # resume-plan never mutates the database


def test_cmd_resume_plan_missing_project_returns_error(isolated_db, capsys):
    rc = cli.cmd_resume_plan(argparse.Namespace(project_id="does-not-exist"))
    assert rc == 1
    assert "FAILED" in capsys.readouterr().err


def test_cmd_validate_manifest_validates_without_writes(isolated_db, tmp_path, capsys):
    story_path, scenes_path = _write_story_and_scenes(tmp_path)
    output_root = tmp_path / "projects"
    cli.cmd_create_project(
        argparse.Namespace(story=str(story_path), scenes=str(scenes_path), output_root=str(output_root))
    )
    create_out = capsys.readouterr().out
    manifest_path = create_out.splitlines()[2].split(": ", 1)[1].strip()
    before_bytes = open(manifest_path, "rb").read()

    rc = cli.cmd_validate_manifest(argparse.Namespace(path=manifest_path))

    assert rc == 0
    out = capsys.readouterr().out
    assert "only validation was performed" in out
    after_bytes = open(manifest_path, "rb").read()
    assert before_bytes == after_bytes


def test_cmd_validate_manifest_missing_file_returns_error(tmp_path, capsys):
    rc = cli.cmd_validate_manifest(argparse.Namespace(path=str(tmp_path / "does-not-exist.json")))
    assert rc == 1
    assert "FAILED" in capsys.readouterr().err


# ---------------------------------------------------------------------
# import-queue
# ---------------------------------------------------------------------


def _write_queue(tmp_path, **overrides) -> "Path":
    from pathlib import Path

    queue_dir = tmp_path / "queue_root"
    queue_dir.mkdir(exist_ok=True)
    _write_json(queue_dir / "story.json", _story_input_dict())
    _write_json(queue_dir / "scenes.json", _scene_plan_dict())

    data = dict(
        queue_version="1.0",
        items=[
            dict(
                queue_item_id="item-1",
                story_input_path="story.json",
                scene_plan_path="scenes.json",
                owner_notes=None,
                output_subdirectory=None,
                enabled=True,
            )
        ],
    )
    data.update(overrides)
    queue_path: Path = queue_dir / "queue.json"
    _write_json(queue_path, data)
    return queue_path


def test_cmd_import_queue_dry_run_writes_nothing(isolated_db, tmp_path, capsys):
    queue_path = _write_queue(tmp_path)
    output_root = tmp_path / "projects"

    rc = cli.cmd_import_queue(
        argparse.Namespace(queue=str(queue_path), output_root=str(output_root), confirm_import=False)
    )

    assert rc == 2
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert not output_root.exists()

    from src.database.db import get_connection

    conn = get_connection()
    try:
        count = conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
    finally:
        conn.close()
    assert count == 0


def test_cmd_import_queue_confirmed_creates_projects(isolated_db, tmp_path, capsys):
    queue_path = _write_queue(tmp_path)
    output_root = tmp_path / "projects"

    rc = cli.cmd_import_queue(
        argparse.Namespace(queue=str(queue_path), output_root=str(output_root), confirm_import=True)
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "created:" in out

    from src.database.db import get_connection
    from src.database.project_repository import list_projects

    conn = get_connection()
    try:
        projects = list_projects(conn)
    finally:
        conn.close()
    assert len(projects) == 1
    assert projects[0].current_stage == "planned"


def test_cmd_import_queue_reports_disabled_entries_as_skipped(isolated_db, tmp_path, capsys):
    queue_path = _write_queue(
        tmp_path,
        items=[
            dict(
                queue_item_id="item-1", story_input_path="story.json", scene_plan_path="scenes.json", enabled=False
            )
        ],
    )
    output_root = tmp_path / "projects"

    rc = cli.cmd_import_queue(
        argparse.Namespace(queue=str(queue_path), output_root=str(output_root), confirm_import=True)
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "skipped (disabled): item-1" in out
    assert not output_root.exists()


def test_cmd_import_queue_bad_queue_file_returns_error(isolated_db, tmp_path, capsys):
    queue_path = tmp_path / "queue.json"
    queue_path.write_text("not json")
    output_root = tmp_path / "projects"

    rc = cli.cmd_import_queue(
        argparse.Namespace(queue=str(queue_path), output_root=str(output_root), confirm_import=False)
    )

    assert rc == 1
    assert "FAILED" in capsys.readouterr().err
