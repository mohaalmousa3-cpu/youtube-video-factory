"""Phase 1D: read-only preview of the exact scene-image generation prompt a
future build-scene-image command would send to
image_qwen.generate_with_reference(). See src/cli.py's
cmd_preview_scene_image_prompt for the CLI command.

This module never calls a provider, never opens/decodes --reference-image,
never creates --output, and never touches SQLite or the manifest — it only
validates the two supplied local paths (existence, not content) and calls
the already-shipped, pure src/core/scene_image_prompt.build_scene_image_prompt().
Project/manifest identity checks and scene lookup are the CLI's job (same
get_readonly_connection()/get_project()/load_manifest() contract as every
other generation-only command) — this module receives an already-resolved
ScenePlanItem, not a project_id/manifest to look up itself."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.core.scene_image_prompt import build_scene_image_prompt
from src.models.scene import ScenePlanItem


class ScenePreviewError(Exception):
    """Raised for any reason this preview cannot be produced: a
    nonexistent or non-file --reference-image, a --output whose parent
    directory does not exist, or a --output that already exists. Message
    text is always a short, fixed, sanitized string."""


@dataclass(frozen=True)
class ScenePreviewResult:
    """The one typed result build_scene_image_preview() returns on
    success. Never itself creates a file, writes SQLite, or mutates a
    manifest — it only reports the prompt that would be used."""

    project_id: str
    scene_id: str
    visual_brief: str
    final_prompt: str
    reference_image_path: Path
    planned_output_path: Path
    include_character_anchor: bool
    include_color_anchor: bool


def build_scene_image_preview(
    project_id: str,
    scene: ScenePlanItem,
    reference_image_path: Path,
    planned_output_path: Path,
    *,
    include_character_anchor: bool,
    include_color_anchor: bool = True,
) -> ScenePreviewResult:
    """Validate `reference_image_path` (must exist, must be a regular
    file — never opened/decoded) and `planned_output_path` (parent must
    exist, path itself must not already exist — never created), then
    assemble the final prompt via build_scene_image_prompt(scene.visual_brief,
    ...). Raises ScenePreviewError for any validation failure; never
    mutates any state, even on success."""
    if not reference_image_path.exists():
        raise ScenePreviewError("--reference-image does not exist")
    if not reference_image_path.is_file():
        raise ScenePreviewError("--reference-image is not a regular file")

    if not planned_output_path.parent.exists():
        raise ScenePreviewError("--output parent directory does not exist")
    if planned_output_path.exists():
        raise ScenePreviewError("--output already exists")

    final_prompt = build_scene_image_prompt(
        scene.visual_brief,
        include_character_anchor=include_character_anchor,
        include_color_anchor=include_color_anchor,
    )

    return ScenePreviewResult(
        project_id=project_id,
        scene_id=scene.scene_id,
        visual_brief=scene.visual_brief,
        final_prompt=final_prompt,
        reference_image_path=reference_image_path,
        planned_output_path=planned_output_path,
        include_character_anchor=include_character_anchor,
        include_color_anchor=include_color_anchor,
    )
