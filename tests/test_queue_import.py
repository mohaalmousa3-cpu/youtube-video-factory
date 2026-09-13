"""Tests for src/core/queue_import.py: local file loading, single-project
creation, and all-or-nothing queue import. No network, no provider — all
inputs are local JSON fixtures written to tmp_path."""
import json
import sqlite3
from pathlib import Path

import pytest

from src.core.queue_import import (
    InputFileError,
    ProjectCreationError,
    QueueImportError,
    _resolve_within,
    create_project_from_inputs,
    import_queue,
    load_queue_import,
    load_scene_plan_file,
    load_story_input_file,
    prepare_queue_import,
)
from src.database.db import SCHEMA
from src.database.project_repository import get_project, list_project_transitions
from src.models.queue import QueueImport
from src.utils.channel_config import get_channel_policy


@pytest.fixture()
def conn():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA)
    yield connection
    connection.close()


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


# ---------------------------------------------------------------------
# File loaders
# ---------------------------------------------------------------------


def test_load_story_input_file_valid(tmp_path):
    path = tmp_path / "story.json"
    _write_json(path, _story_input_dict())
    story = load_story_input_file(path)
    assert story.story_id == "why-we-care-what-people-think"


def test_load_story_input_file_missing_raises(tmp_path):
    with pytest.raises(InputFileError):
        load_story_input_file(tmp_path / "does-not-exist.json")


def test_load_story_input_file_invalid_json_raises(tmp_path):
    path = tmp_path / "story.json"
    path.write_text("{not valid json")
    with pytest.raises(InputFileError):
        load_story_input_file(path)


def test_load_story_input_file_non_object_json_raises(tmp_path):
    path = tmp_path / "story.json"
    path.write_text("[1, 2, 3]")
    with pytest.raises(InputFileError):
        load_story_input_file(path)


def test_load_story_input_file_accepts_unapproved_story_at_the_file_load_layer(tmp_path):
    """StoryInput itself validates fine with approval_status="draft" (it's
    just a Literal value) — the "must be approved" rule is enforced later
    by ManifestBuilder (build_video_manifest), not by this file loader.
    The CLI/queue path rejects an unapproved story via that later check —
    see test_manifest_builder.py's existing approval-gate tests, and
    test_import_queue_rejects_non_english_content_via_manifest_builder
    below for the same "loader succeeds, builder rejects" split applied to
    language instead of approval."""
    path = tmp_path / "story.json"
    _write_json(path, _story_input_dict(approval_status="draft"))
    story = load_story_input_file(path)
    assert story.approval_status == "draft"


def test_load_scene_plan_file_valid(tmp_path):
    path = tmp_path / "scenes.json"
    _write_json(path, _scene_plan_dict())
    plan = load_scene_plan_file(path)
    assert len(plan.scenes) == 1


def test_load_scene_plan_file_invalid_flow_fallback_raises(tmp_path):
    path = tmp_path / "scenes.json"
    _write_json(
        path,
        _scene_plan_dict(
            scenes=[
                dict(
                    scene_id="scene-01",
                    sequence=1,
                    narration_text="Text.",
                    scene_type="establishing",
                    narrative_beat="hook",
                    visual_brief="Brief.",
                    motion_mode="manual_flow",
                    local_fallback_motion_mode="manual_flow",  # invalid: can't fall back to itself
                    approval_state="approved",
                ),
            ],
        ),
    )
    with pytest.raises(InputFileError):
        load_scene_plan_file(path)


# ---------------------------------------------------------------------
# load_queue_import
# ---------------------------------------------------------------------


def _queue_dict(**overrides) -> dict:
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
            ),
        ],
    )
    data.update(overrides)
    return data


def test_load_queue_import_valid(tmp_path):
    path = tmp_path / "queue.json"
    _write_json(path, _queue_dict())
    queue = load_queue_import(path)
    assert isinstance(queue, QueueImport)
    assert len(queue.items) == 1


def test_load_queue_import_rejects_unknown_field(tmp_path):
    path = tmp_path / "queue.json"
    data = _queue_dict()
    data["unexpected"] = True
    _write_json(path, data)
    with pytest.raises(QueueImportError):
        load_queue_import(path)


def test_load_queue_import_rejects_duplicate_queue_item_id(tmp_path):
    path = tmp_path / "queue.json"
    data = _queue_dict(
        items=[
            dict(queue_item_id="dup", story_input_path="a.json", scene_plan_path="b.json", enabled=True),
            dict(queue_item_id="dup", story_input_path="c.json", scene_plan_path="d.json", enabled=False),
        ]
    )
    _write_json(path, data)
    with pytest.raises(QueueImportError):
        load_queue_import(path)


def test_load_queue_import_rejects_missing_enabled_field(tmp_path):
    path = tmp_path / "queue.json"
    data = _queue_dict(
        items=[dict(queue_item_id="item-1", story_input_path="a.json", scene_plan_path="b.json")]
    )
    _write_json(path, data)
    with pytest.raises(QueueImportError):
        load_queue_import(path)


def test_load_queue_import_non_object_json_raises(tmp_path):
    path = tmp_path / "queue.json"
    path.write_text("null")
    with pytest.raises(QueueImportError):
        load_queue_import(path)


# ---------------------------------------------------------------------
# create_project_from_inputs (single project)
# ---------------------------------------------------------------------


def test_create_project_from_inputs_creates_manifest_and_record(tmp_path, conn):
    from src.models.story import StoryInput
    from src.models.scene import ScenePlan

    story = StoryInput.model_validate(_story_input_dict())
    scenes = ScenePlan.model_validate(_scene_plan_dict())
    output_root = tmp_path / "projects"

    project = create_project_from_inputs(conn, story, scenes, get_channel_policy(), output_root)

    assert project.current_stage == "planned"
    manifest_path = output_root / project.project_id / "manifest.json"
    assert manifest_path.exists()
    assert get_project(conn, project.project_id) == project
    assert len(list_project_transitions(conn, project.project_id)) == 1


def test_create_project_from_inputs_rejects_existing_manifest_path(tmp_path, conn):
    from src.models.story import StoryInput
    from src.models.scene import ScenePlan

    story = StoryInput.model_validate(_story_input_dict())
    scenes = ScenePlan.model_validate(_scene_plan_dict())
    output_root = tmp_path / "projects"

    project = create_project_from_inputs(conn, story, scenes, get_channel_policy(), output_root)

    # A second attempt with the SAME (deterministic) inputs produces the
    # SAME project_id/manifest path — must be rejected, not overwritten.
    with pytest.raises(ProjectCreationError):
        create_project_from_inputs(conn, story, scenes, get_channel_policy(), output_root)

    # Confirm nothing was corrupted: still exactly one transition.
    assert len(list_project_transitions(conn, project.project_id)) == 1


def test_create_project_from_inputs_rejects_existing_db_project_id(tmp_path, conn):
    from src.models.story import StoryInput
    from src.models.scene import ScenePlan
    from src.database.project_repository import create_project as repo_create_project
    from src.core.project_state_machine import create_initial_project, initial_transition_for
    from src.core.manifest_builder import build_video_manifest

    story = StoryInput.model_validate(_story_input_dict())
    scenes = ScenePlan.model_validate(_scene_plan_dict())
    output_root = tmp_path / "projects"

    # Pre-create the DB row directly (without a manifest file) so the
    # manifest-path check would pass but the DB check must still catch it.
    manifest = build_video_manifest(story, scenes, get_channel_policy())
    pre_existing = create_initial_project("elsewhere/manifest.json", manifest)
    repo_create_project(conn, pre_existing, initial_transition_for(pre_existing))

    with pytest.raises(ProjectCreationError):
        create_project_from_inputs(conn, story, scenes, get_channel_policy(), output_root)

    # No manifest file was written by the rejected attempt.
    assert not (output_root / manifest.project_id / "manifest.json").exists()


def test_create_project_from_inputs_cleans_up_manifest_on_db_failure(tmp_path, conn, monkeypatch):
    from src.models.story import StoryInput
    from src.models.scene import ScenePlan

    story = StoryInput.model_validate(_story_input_dict())
    scenes = ScenePlan.model_validate(_scene_plan_dict())
    output_root = tmp_path / "projects"

    def _raise(_conn, _record, _transition):
        raise sqlite3.OperationalError("simulated database failure")

    monkeypatch.setattr("src.core.queue_import.create_project", _raise)

    with pytest.raises(ProjectCreationError):
        create_project_from_inputs(conn, story, scenes, get_channel_policy(), output_root)

    from src.core.manifest_builder import build_video_manifest

    manifest = build_video_manifest(story, scenes, get_channel_policy())
    project_dir = output_root / manifest.project_id
    assert not (project_dir / "manifest.json").exists()
    assert not project_dir.exists()  # freshly-created directory was cleaned up too
    assert get_project(conn, manifest.project_id) is None


def test_create_project_from_inputs_with_relative_output_root(tmp_path, conn, monkeypatch):
    """The same _resolve_within() safety check create_project_from_inputs
    now applies to --output-root must work correctly for a RELATIVE
    output_root too, not just an already-absolute tmp_path-based one."""
    from src.models.story import StoryInput
    from src.models.scene import ScenePlan

    monkeypatch.chdir(tmp_path)
    story = StoryInput.model_validate(_story_input_dict())
    scenes = ScenePlan.model_validate(_scene_plan_dict())
    relative_output_root = Path("projects")  # relative to the new cwd (tmp_path)

    project = create_project_from_inputs(conn, story, scenes, get_channel_policy(), relative_output_root)

    resolved_manifest_path = (tmp_path / "projects" / project.project_id / "manifest.json").resolve()
    assert resolved_manifest_path.exists()
    assert resolved_manifest_path.read_text()  # a real, non-empty manifest was written there


def test_create_project_from_inputs_rejects_project_id_that_would_escape_output_root(tmp_path, conn, monkeypatch):
    """Even though build_video_manifest never actually produces a
    project_id containing path-traversal characters (it's always
    "proj-" + a hex fingerprint prefix), create_project_from_inputs must
    still refuse to write outside --output-root if it ever did — this
    proves the unified _resolve_within() check is actually wired into
    this call site, not just present in the module."""
    from src.models.story import StoryInput
    from src.models.scene import ScenePlan
    from src.core.manifest_builder import build_video_manifest as real_build_video_manifest

    story = StoryInput.model_validate(_story_input_dict())
    scenes = ScenePlan.model_validate(_scene_plan_dict())
    output_root = tmp_path / "projects"

    real_manifest = real_build_video_manifest(story, scenes, get_channel_policy())
    malicious_manifest = real_manifest.model_copy(update={"project_id": "../escape"})
    monkeypatch.setattr(
        "src.core.queue_import.build_video_manifest", lambda *args, **kwargs: malicious_manifest
    )

    with pytest.raises(ProjectCreationError):
        create_project_from_inputs(conn, story, scenes, get_channel_policy(), output_root)

    # Nothing was written anywhere, inside or outside output_root.
    assert not output_root.exists()
    assert not (tmp_path / "escape").exists()


# ---------------------------------------------------------------------
# _resolve_within: the one shared path-safety helper, tested directly
# ---------------------------------------------------------------------


def test_resolve_within_accepts_a_safe_relative_path(tmp_path):
    base_dir = tmp_path / "root"
    base_dir.mkdir()

    result = _resolve_within(base_dir, "proj-abc123", what="test path")

    assert result == (base_dir / "proj-abc123").resolve()


def test_resolve_within_rejects_traversal_with_default_error(tmp_path):
    base_dir = tmp_path / "root"
    base_dir.mkdir()

    with pytest.raises(QueueImportError):
        _resolve_within(base_dir, "../escape", what="test path")


def test_resolve_within_rejects_traversal_with_custom_error_cls(tmp_path):
    """Proves the helper's error_cls parameter is real, not decorative —
    create_project_from_inputs relies on this to raise ProjectCreationError
    (not QueueImportError) for its own --output-root check."""
    base_dir = tmp_path / "root"
    base_dir.mkdir()

    with pytest.raises(ProjectCreationError):
        _resolve_within(base_dir, "../escape", what="test path", error_cls=ProjectCreationError)


# ---------------------------------------------------------------------
# prepare_queue_import / import_queue — path resolution and traversal
# ---------------------------------------------------------------------


def _setup_queue_dir(tmp_path, *, story_overrides=None, scene_overrides=None):
    queue_dir = tmp_path / "queue_root"
    queue_dir.mkdir()
    _write_json(queue_dir / "story.json", _story_input_dict(**(story_overrides or {})))
    _write_json(queue_dir / "scenes.json", _scene_plan_dict(**(scene_overrides or {})))
    return queue_dir


def test_prepare_queue_import_resolves_paths_relative_to_queue_dir(tmp_path, conn):
    queue_dir = _setup_queue_dir(tmp_path)
    queue = QueueImport.model_validate(_queue_dict())
    output_root = tmp_path / "projects"

    plan = prepare_queue_import(queue, queue_dir, get_channel_policy(), output_root, conn)

    assert len(plan.prepared) == 1
    assert plan.prepared[0].manifest.story_input.story_id == "why-we-care-what-people-think"


def test_prepare_queue_import_rejects_path_traversal_outside_queue_dir(tmp_path, conn):
    queue_dir = _setup_queue_dir(tmp_path)
    # Escape queue_dir by walking up and into a sibling directory.
    queue = QueueImport.model_validate(
        _queue_dict(
            items=[
                dict(
                    queue_item_id="escape",
                    story_input_path="../outside/story.json",
                    scene_plan_path="scenes.json",
                    enabled=True,
                )
            ]
        )
    )
    output_root = tmp_path / "projects"

    with pytest.raises(QueueImportError):
        prepare_queue_import(queue, queue_dir, get_channel_policy(), output_root, conn)


def test_prepare_queue_import_rejects_absolute_path_escaping_queue_dir(tmp_path, conn):
    queue_dir = _setup_queue_dir(tmp_path)
    queue = QueueImport.model_validate(
        _queue_dict(
            items=[
                dict(
                    queue_item_id="escape",
                    story_input_path="/etc/passwd",
                    scene_plan_path="scenes.json",
                    enabled=True,
                )
            ]
        )
    )
    output_root = tmp_path / "projects"

    with pytest.raises(QueueImportError):
        prepare_queue_import(queue, queue_dir, get_channel_policy(), output_root, conn)


def test_prepare_queue_import_rejects_output_subdirectory_escaping_output_root(tmp_path, conn):
    queue_dir = _setup_queue_dir(tmp_path)
    queue = QueueImport.model_validate(
        _queue_dict(
            items=[
                dict(
                    queue_item_id="item-1",
                    story_input_path="story.json",
                    scene_plan_path="scenes.json",
                    output_subdirectory="../escape",
                    enabled=True,
                )
            ]
        )
    )
    output_root = tmp_path / "projects"
    output_root.mkdir()

    with pytest.raises(QueueImportError):
        prepare_queue_import(queue, queue_dir, get_channel_policy(), output_root, conn)


# ---------------------------------------------------------------------
# prepare_queue_import / import_queue — conflict detection (dry run safe)
# ---------------------------------------------------------------------


def test_prepare_queue_import_is_a_pure_dry_run(tmp_path, conn):
    queue_dir = _setup_queue_dir(tmp_path)
    queue = QueueImport.model_validate(_queue_dict())
    output_root = tmp_path / "projects"

    prepare_queue_import(queue, queue_dir, get_channel_policy(), output_root, conn)

    assert not output_root.exists()  # nothing was ever written
    assert conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0] == 0


def test_prepare_queue_import_rejects_duplicate_project_id_within_batch(tmp_path, conn):
    queue_dir = _setup_queue_dir(tmp_path)
    # Two items pointing at the IDENTICAL story/scene files produce the
    # SAME deterministic project_id.
    queue = QueueImport.model_validate(
        _queue_dict(
            items=[
                dict(queue_item_id="item-1", story_input_path="story.json", scene_plan_path="scenes.json", enabled=True),
                dict(queue_item_id="item-2", story_input_path="story.json", scene_plan_path="scenes.json", enabled=True),
            ]
        )
    )
    output_root = tmp_path / "projects"

    with pytest.raises(QueueImportError):
        prepare_queue_import(queue, queue_dir, get_channel_policy(), output_root, conn)

    assert not output_root.exists()


def test_prepare_queue_import_rejects_existing_db_project_id(tmp_path, conn):
    queue_dir = _setup_queue_dir(tmp_path)
    queue = QueueImport.model_validate(_queue_dict())
    output_root = tmp_path / "projects"

    # First import succeeds and registers the project in the DB.
    import_queue(conn, queue, queue_dir, get_channel_policy(), output_root)

    # A second, independent queue pointing at the same story/scene content
    # (elsewhere on disk, so no manifest-path collision) must still be
    # rejected due to the DB conflict.
    other_queue_dir = tmp_path / "other_queue_root"
    other_queue_dir.mkdir()
    _write_json(other_queue_dir / "story.json", _story_input_dict())
    _write_json(other_queue_dir / "scenes.json", _scene_plan_dict())
    other_queue = QueueImport.model_validate(_queue_dict())

    with pytest.raises(QueueImportError):
        prepare_queue_import(other_queue, other_queue_dir, get_channel_policy(), output_root, conn)


def test_prepare_queue_import_rejects_existing_manifest_file(tmp_path, conn):
    queue_dir = _setup_queue_dir(tmp_path)
    queue = QueueImport.model_validate(_queue_dict())
    output_root = tmp_path / "projects"

    result = import_queue(conn, queue, queue_dir, get_channel_policy(), output_root)
    project_id = result.created[0].project_id

    # Remove the DB row but leave the manifest file behind, to isolate the
    # "manifest file already exists" check from the "DB conflict" check.
    from src.database.project_repository import delete_project

    delete_project(conn, project_id)
    assert (output_root / project_id / "manifest.json").exists()

    with pytest.raises(QueueImportError):
        prepare_queue_import(queue, queue_dir, get_channel_policy(), output_root, conn)


# ---------------------------------------------------------------------
# import_queue — confirmed import behavior
# ---------------------------------------------------------------------


def test_import_queue_creates_projects_for_all_enabled_entries(tmp_path, conn):
    queue_dir = tmp_path / "queue_root"
    queue_dir.mkdir()
    _write_json(queue_dir / "story-a.json", _story_input_dict(story_id="story-a"))
    _write_json(queue_dir / "scenes-a.json", _scene_plan_dict())
    _write_json(queue_dir / "story-b.json", _story_input_dict(story_id="story-b"))
    _write_json(queue_dir / "scenes-b.json", _scene_plan_dict())

    queue = QueueImport.model_validate(
        _queue_dict(
            items=[
                dict(queue_item_id="a", story_input_path="story-a.json", scene_plan_path="scenes-a.json", enabled=True),
                dict(queue_item_id="b", story_input_path="story-b.json", scene_plan_path="scenes-b.json", enabled=False),
            ]
        )
    )
    output_root = tmp_path / "projects"

    result = import_queue(conn, queue, queue_dir, get_channel_policy(), output_root)

    assert len(result.created) == 1
    assert result.skipped_disabled == ("b",)
    created_project = result.created[0]
    assert (output_root / created_project.project_id / "manifest.json").exists()
    assert get_project(conn, created_project.project_id) is not None

    # The disabled entry produced nothing at all.
    assert conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0] == 1


def test_import_queue_never_advances_a_project_past_planned(tmp_path, conn):
    queue_dir = _setup_queue_dir(tmp_path)
    queue = QueueImport.model_validate(_queue_dict())
    output_root = tmp_path / "projects"

    result = import_queue(conn, queue, queue_dir, get_channel_policy(), output_root)

    assert result.created[0].current_stage == "planned"
    assert result.created[0].execution_status == "not_executed"


def test_import_queue_rolls_back_on_partial_failure(tmp_path, conn, monkeypatch):
    queue_dir = tmp_path / "queue_root"
    queue_dir.mkdir()
    _write_json(queue_dir / "story-a.json", _story_input_dict(story_id="story-a"))
    _write_json(queue_dir / "scenes-a.json", _scene_plan_dict())
    _write_json(queue_dir / "story-b.json", _story_input_dict(story_id="story-b"))
    _write_json(queue_dir / "scenes-b.json", _scene_plan_dict())

    queue = QueueImport.model_validate(
        _queue_dict(
            items=[
                dict(queue_item_id="a", story_input_path="story-a.json", scene_plan_path="scenes-a.json", enabled=True),
                dict(queue_item_id="b", story_input_path="story-b.json", scene_plan_path="scenes-b.json", enabled=True),
            ]
        )
    )
    output_root = tmp_path / "projects"

    plan = prepare_queue_import(queue, queue_dir, get_channel_policy(), output_root, conn)
    first_project_id = plan.prepared[0].manifest.project_id
    second_project_id = plan.prepared[1].manifest.project_id

    from src.database.project_repository import create_project as real_create_project

    call_count = {"n": 0}

    def flaky_create_project(conn_arg, record, transition):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise sqlite3.OperationalError("simulated failure on the second entry")
        return real_create_project(conn_arg, record, transition)

    monkeypatch.setattr("src.core.queue_import.create_project", flaky_create_project)

    with pytest.raises(QueueImportError):
        import_queue(conn, queue, queue_dir, get_channel_policy(), output_root)

    # Neither project is left registered in the database...
    assert get_project(conn, first_project_id) is None
    assert get_project(conn, second_project_id) is None
    # ...and the first entry's manifest file (already written before the
    # second entry failed) was cleaned up too.
    assert not (output_root / first_project_id / "manifest.json").exists()
    assert not (output_root / first_project_id).exists()


def test_import_queue_rejects_non_english_content_via_manifest_builder(tmp_path, conn):
    queue_dir = tmp_path / "queue_root"
    queue_dir.mkdir()
    _write_json(queue_dir / "story.json", _story_input_dict(title="عنوان غير إنجليزي"))
    _write_json(queue_dir / "scenes.json", _scene_plan_dict())
    queue = QueueImport.model_validate(_queue_dict())
    output_root = tmp_path / "projects"

    with pytest.raises(QueueImportError):
        prepare_queue_import(queue, queue_dir, get_channel_policy(), output_root, conn)
