"""Phase 1D: pure scene-image prompt assembly.

Stores the three prompt fragments docs/spec-v4/TECHNICAL-SPEC-EN.md section
9 and docs/spec-v4/prompts/STAGE-PROMPTS.md section 2 require every scene
image prompt sent to image_qwen.generate_with_reference() to append, in
order, after the scene-specific visual description — CHARACTER_ANCHOR (for
character-identity-lock consistency), COLOR_ANCHOR (validated Phase 2 A/B
test), and SAFETY_SUFFIX (always required, never optional — matches
docs/spec-v4/schemas/scene.schema.json's `visual_prompt.uses_safety_suffix:
{"const": true}`) — and build_scene_image_prompt(), the pure function that
assembles them.

This module does not call Qwen-Image, does not create an image-generation
CLI command, and is not wired into any orchestrator — it is the utility a
later scene-image-generation slice will import and call. No file I/O, no
settings/environment access, no provider/network/subprocess/database call,
and no mutation of any kind anywhere in this module — build_scene_image_prompt()
is pure string assembly over its own arguments and the three module-level
constants below."""
from __future__ import annotations

COLOR_ANCHOR = (
    "Consistent warm color grading across the whole scene: soft golden-amber "
    "lighting, gentle warm shadows, calm and emotionally intimate mood, "
    "cohesive muted color palette — avoid harsh contrast or clashing "
    "saturated colors."
)

CHARACTER_ANCHOR = (
    "A recurring adult male stick-figure protagonist with a consistent, "
    "recognizable character design across every scene: a round, pale "
    "gray-blue subtly shaded head with a thick black cartoon outline; "
    "oversized round white eyes with black pupils; thick curved dark "
    "eyebrows; a compact dark-charcoal subtly tapered torso; equal-length "
    "thin dark-charcoal line arms and legs; dark-charcoal rounded mitten-like "
    "hands and flat rounded shoe-like feet. Render the character in thick "
    "black cartoon outlines as a clean 2D cartoon illustration — never 3D "
    "and never photorealistic. Keep the same head-to-body ratio, limb "
    "thickness, height, face placement, color palette, and overall character "
    "design in every scene. The character must remain clearly male and "
    "visually identical from scene to scene.\n\n"
    "Clothing, accessories, held objects, pose, and expression may change "
    "when required by the story, but must preserve the character's core "
    "silhouette, proportions, face design, and dark-charcoal body palette."
)

SAFETY_SUFFIX = (
    "The entire scene, including the background, must be a cohesive clean 2D "
    "cartoon illustration; do not mix a cartoon character with a "
    "photorealistic environment, photographic lighting, or photographic "
    "depth of field. No text, captions, speech bubbles, watermarks, logos, "
    "UI elements, duplicate characters, extra arms, extra legs, malformed "
    "limbs, photorealistic humans, realistic facial details, 3D rendering, "
    "or live-action photography."
)


def build_scene_image_prompt(
    base_description: str,
    *,
    include_character_anchor: bool,
    include_color_anchor: bool = True,
) -> str:
    """Assemble one scene's image-generation prompt: `base_description`
    (leading/trailing-whitespace stripped, otherwise untouched — never
    inspected or sanitized, per docs/spec-v4/prompts/STAGE-PROMPTS.md
    section 2, which treats it as the scene-specific visual description
    that anchors are appended AFTER), then CHARACTER_ANCHOR (only if
    `include_character_anchor`), then COLOR_ANCHOR (only if
    `include_color_anchor`), then SAFETY_SUFFIX (always — there is no
    parameter to omit it, matching scene.schema.json's
    `visual_prompt.uses_safety_suffix: {"const": true}`). Components are
    joined with exactly two newlines ("\\n\\n"). Raises TypeError for a
    non-str `base_description` or a non-bool flag, and ValueError for an
    empty/whitespace-only `base_description` (checked after stripping)."""
    if not isinstance(base_description, str):
        raise TypeError("base_description must be a string")
    if not isinstance(include_character_anchor, bool):
        raise TypeError("include_character_anchor must be a bool")
    if not isinstance(include_color_anchor, bool):
        raise TypeError("include_color_anchor must be a bool")

    stripped = base_description.strip()
    if not stripped:
        raise ValueError("base_description must not be empty")

    parts = [stripped]
    if include_character_anchor:
        parts.append(CHARACTER_ANCHOR)
    if include_color_anchor:
        parts.append(COLOR_ANCHOR)
    parts.append(SAFETY_SUFFIX)

    return "\n\n".join(parts)
