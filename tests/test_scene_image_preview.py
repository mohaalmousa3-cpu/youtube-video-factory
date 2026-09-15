"""Tests for src/core/scene_image_preview.py — provider-free, database-free
preview of the scene-image prompt build_scene_image_prompt() would produce.
No provider, no image decode, no file creation anywhere in this file."""
from __future__ import annotations

import ast
import inspect

import pytest

import src.core.scene_image_preview as scene_image_preview
from src.core.scene_image_prompt import CHARACTER_ANCHOR, COLOR_ANCHOR, build_scene_image_prompt
from src.core.scene_image_preview import (
    ScenePreviewError,
    ScenePreviewResult,
    build_scene_image_preview,
)
from src.models.scene import ScenePlanItem


def _scene(scene_id: str = "scene-01", visual_brief: str = "A stick figure waits at a bus stop.") -> ScenePlanItem:
    return ScenePlanItem(
        scene_id=scene_id,
        sequence=1,
        narration_text="Narration text.",
        scene_type="narration",
        narrative_beat="setup",
        visual_brief=visual_brief,
        motion_mode="static",
        approval_state="approved",
    )


# ---------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------


def test_happy_path_returns_exact_expected_data(tmp_path):
    scene = _scene()
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"not-a-real-image-just-existence-matters")
    planned_output = tmp_path / "out.png"

    result = build_scene_image_preview(
        "proj-1", scene, reference_image, planned_output, include_character_anchor=True, include_color_anchor=True
    )

    assert isinstance(result, ScenePreviewResult)
    assert result.project_id == "proj-1"
    assert result.scene_id == "scene-01"
    assert result.visual_brief == scene.visual_brief
    assert result.final_prompt == build_scene_image_prompt(
        scene.visual_brief, include_character_anchor=True, include_color_anchor=True
    )
    assert result.reference_image_path == reference_image
    assert result.planned_output_path == planned_output
    assert result.include_character_anchor is True
    assert result.include_color_anchor is True


# ---------------------------------------------------------------------
# all four flag combinations
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "include_character_anchor,include_color_anchor",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_four_flag_combinations_flow_through_unchanged(tmp_path, include_character_anchor, include_color_anchor):
    scene = _scene()
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    planned_output = tmp_path / "out.png"

    result = build_scene_image_preview(
        "proj-1",
        scene,
        reference_image,
        planned_output,
        include_character_anchor=include_character_anchor,
        include_color_anchor=include_color_anchor,
    )

    assert result.include_character_anchor is include_character_anchor
    assert result.include_color_anchor is include_color_anchor
    assert (CHARACTER_ANCHOR in result.final_prompt) is include_character_anchor
    assert (COLOR_ANCHOR in result.final_prompt) is include_color_anchor


# ---------------------------------------------------------------------
# reference-image validation
# ---------------------------------------------------------------------


def test_missing_reference_image_rejects(tmp_path):
    scene = _scene()
    reference_image = tmp_path / "does-not-exist.png"
    planned_output = tmp_path / "out.png"

    with pytest.raises(ScenePreviewError, match="--reference-image does not exist"):
        build_scene_image_preview(
            "proj-1", scene, reference_image, planned_output, include_character_anchor=True
        )


def test_directory_reference_image_rejects(tmp_path):
    scene = _scene()
    reference_image = tmp_path / "a-directory"
    reference_image.mkdir()
    planned_output = tmp_path / "out.png"

    with pytest.raises(ScenePreviewError, match="--reference-image is not a regular file"):
        build_scene_image_preview(
            "proj-1", scene, reference_image, planned_output, include_character_anchor=True
        )


# ---------------------------------------------------------------------
# planned-output validation
# ---------------------------------------------------------------------


def test_output_parent_missing_rejects(tmp_path):
    scene = _scene()
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    planned_output = tmp_path / "no-such-dir" / "out.png"

    with pytest.raises(ScenePreviewError, match="--output parent directory does not exist"):
        build_scene_image_preview(
            "proj-1", scene, reference_image, planned_output, include_character_anchor=True
        )


def test_output_already_exists_rejects(tmp_path):
    scene = _scene()
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    planned_output = tmp_path / "out.png"
    planned_output.write_bytes(b"already here")

    with pytest.raises(ScenePreviewError, match="--output already exists"):
        build_scene_image_preview(
            "proj-1", scene, reference_image, planned_output, include_character_anchor=True
        )


def test_safe_nonexistent_output_in_existing_parent_succeeds_and_creates_nothing(tmp_path):
    scene = _scene()
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"x")
    planned_output = tmp_path / "out.png"

    build_scene_image_preview("proj-1", scene, reference_image, planned_output, include_character_anchor=False)

    assert not planned_output.exists()


def test_reference_image_checked_before_output(tmp_path):
    """Both invalid at once: the reference-image message wins, proving a
    stable, deterministic check order."""
    scene = _scene()
    reference_image = tmp_path / "does-not-exist.png"
    planned_output = tmp_path / "no-such-dir" / "out.png"

    with pytest.raises(ScenePreviewError, match="--reference-image does not exist"):
        build_scene_image_preview(
            "proj-1", scene, reference_image, planned_output, include_character_anchor=True
        )


# ---------------------------------------------------------------------
# no mutation of any kind, even on success — proven structurally, not
# just by trust
# ---------------------------------------------------------------------


def test_module_has_no_provider_or_database_imports():
    source = inspect.getsource(scene_image_preview)
    tree = ast.parse(source)
    import_nodes = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]

    prohibited_prefixes = ("src.providers", "src.database")
    offending = [
        node
        for node in import_nodes
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith(prohibited_prefixes)
    ]

    assert offending == [], (
        "src/core/scene_image_preview.py must not import any provider or database module; "
        f"found: {[ast.dump(n) for n in offending]}"
    )
