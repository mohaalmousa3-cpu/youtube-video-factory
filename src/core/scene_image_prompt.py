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
constants below.

Constants updated 2026-09-17 to the `stage-scene-image-v2` contract (see
docs/spec-v4/prompts/STAGE-PROMPTS.md section 6) — a cohesive hand-drawn 2D
stick-figure story-animation style. build_scene_image_prompt()'s public
signature, component order, and double-newline separator are unchanged;
only these three constants' literal text changed. The prior
`stage-scene-image-v1` wording is preserved in that same doc as a
historical record."""
from __future__ import annotations

CHARACTER_ANCHOR = (
    "A recurring hand-drawn 2D stick-figure character with a large round pure-white "
    "head, a thin clean charcoal outline, minimal dot or short-line eyes, simple "
    "expressive eyebrows and mouth lines, and a small simple body with thin dark "
    "limbs. Keep the head size, outline weight, limb proportions, clothing palette, "
    "and any fixed accessory identical across every scene where this character "
    "appears. Flat 2D illustration only; no realistic anatomy, no 3D rendering, no "
    "photorealism, no anime, no Pixar-like style."
)

COLOR_ANCHOR = (
    "Warm hand-drawn pastel storybook palette: cream, soft beige, muted peach, "
    "dusty blue, pale sage, and occasional muted salmon accents. Use flat color "
    "fills with subtle paper-like texture only; keep shading minimal and avoid "
    "strong gradients, neon colors, glossy lighting, dramatic cinematic contrast, "
    "or photorealistic materials. For tense or somber scenes, shift mood only "
    "through restrained accent colors, composition, pose, and background tone while "
    "preserving the same flat pastel illustrated style."
)

SAFETY_SUFFIX = (
    "No readable text, captions, speech bubbles, UI panels, logos, watermarks, "
    "brand marks, photorealism, 3D rendering, anime style, glossy CGI, extra "
    "fingers, extra limbs, distorted faces, duplicated characters, cluttered "
    "backgrounds, or unrelated props."
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
