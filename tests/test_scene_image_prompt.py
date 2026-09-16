"""Tests for src/core/scene_image_prompt.py — pure string assembly, no
provider/network/file/database access anywhere. Every test is a plain
function call and string assertion; item at the bottom proves via AST
source inspection (not just trust) that the module imports nothing beyond
`from __future__ import annotations`."""
from __future__ import annotations

import ast
import inspect

import pytest

import src.core.scene_image_prompt as scene_image_prompt
from src.core.scene_image_prompt import (
    CHARACTER_ANCHOR,
    COLOR_ANCHOR,
    SAFETY_SUFFIX,
    build_scene_image_prompt,
)

BASE = "A stick figure waits at a bus stop."


# ---------------------------------------------------------------------
# four include-flag combinations
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "include_character_anchor,include_color_anchor",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_four_flag_combinations(include_character_anchor, include_color_anchor):
    result = build_scene_image_prompt(
        BASE,
        include_character_anchor=include_character_anchor,
        include_color_anchor=include_color_anchor,
    )

    assert BASE in result
    assert SAFETY_SUFFIX in result
    assert (CHARACTER_ANCHOR in result) is include_character_anchor
    assert (COLOR_ANCHOR in result) is include_color_anchor


# ---------------------------------------------------------------------
# exact output string
# ---------------------------------------------------------------------


def test_exact_output_both_anchors_included():
    result = build_scene_image_prompt(
        BASE, include_character_anchor=True, include_color_anchor=True
    )

    expected = "\n\n".join([BASE, CHARACTER_ANCHOR, COLOR_ANCHOR, SAFETY_SUFFIX])
    assert result == expected


def test_exact_output_no_anchors_only_safety_suffix():
    result = build_scene_image_prompt(
        BASE, include_character_anchor=False, include_color_anchor=False
    )

    expected = "\n\n".join([BASE, SAFETY_SUFFIX])
    assert result == expected


# ---------------------------------------------------------------------
# base_description stripping
# ---------------------------------------------------------------------


def test_base_description_leading_trailing_whitespace_stripped():
    result = build_scene_image_prompt(
        "   " + BASE + "  \n", include_character_anchor=False, include_color_anchor=False
    )

    assert result.startswith(BASE)


def test_base_description_internal_whitespace_unchanged():
    raw = "  Line one.\n  Line two with   extra   spaces.  "
    result = build_scene_image_prompt(raw, include_character_anchor=False, include_color_anchor=False)

    base_component = result.split("\n\n", 1)[0]
    assert base_component == "Line one.\n  Line two with   extra   spaces."


# ---------------------------------------------------------------------
# SAFETY_SUFFIX presence/position
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "include_character_anchor,include_color_anchor",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_safety_suffix_appears_exactly_once_as_final_component(
    include_character_anchor, include_color_anchor
):
    result = build_scene_image_prompt(
        BASE,
        include_character_anchor=include_character_anchor,
        include_color_anchor=include_color_anchor,
    )

    assert result.count(SAFETY_SUFFIX) == 1
    assert result.endswith(SAFETY_SUFFIX)


# ---------------------------------------------------------------------
# component order
# ---------------------------------------------------------------------


def test_anchors_appear_in_correct_order_when_both_included():
    result = build_scene_image_prompt(
        BASE, include_character_anchor=True, include_color_anchor=True
    )

    base_index = result.index(BASE)
    character_index = result.index(CHARACTER_ANCHOR)
    color_index = result.index(COLOR_ANCHOR)
    safety_index = result.index(SAFETY_SUFFIX)

    assert base_index < character_index < color_index < safety_index


def test_character_anchor_before_safety_suffix_when_color_excluded():
    result = build_scene_image_prompt(
        BASE, include_character_anchor=True, include_color_anchor=False
    )

    assert result.index(CHARACTER_ANCHOR) < result.index(SAFETY_SUFFIX)


def test_color_anchor_before_safety_suffix_when_character_excluded():
    result = build_scene_image_prompt(
        BASE, include_character_anchor=False, include_color_anchor=True
    )

    assert result.index(COLOR_ANCHOR) < result.index(SAFETY_SUFFIX)


# ---------------------------------------------------------------------
# anchors omitted only when their own flag is False
# ---------------------------------------------------------------------


def test_character_anchor_absent_when_flag_false_even_if_color_true():
    result = build_scene_image_prompt(
        BASE, include_character_anchor=False, include_color_anchor=True
    )
    assert CHARACTER_ANCHOR not in result
    assert COLOR_ANCHOR in result


def test_color_anchor_absent_when_flag_false_even_if_character_true():
    result = build_scene_image_prompt(
        BASE, include_character_anchor=True, include_color_anchor=False
    )
    assert COLOR_ANCHOR not in result
    assert CHARACTER_ANCHOR in result


# ---------------------------------------------------------------------
# empty/whitespace-only base_description rejects
# ---------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["", "   ", "\n\t \n", " "])
def test_empty_or_whitespace_only_base_description_rejects(raw):
    with pytest.raises(ValueError) as exc_info:
        build_scene_image_prompt(raw, include_character_anchor=False, include_color_anchor=False)
    assert str(exc_info.value) == "base_description must not be empty"


# ---------------------------------------------------------------------
# non-string base_description
# ---------------------------------------------------------------------


@pytest.mark.parametrize("raw", [None, b"bytes description", 123, 1.5, [], {}, ("a",)])
def test_non_string_base_description_rejects(raw):
    with pytest.raises(TypeError) as exc_info:
        build_scene_image_prompt(raw, include_character_anchor=False, include_color_anchor=False)
    assert str(exc_info.value) == "base_description must be a string"


# ---------------------------------------------------------------------
# bool flag validation — both flags, independently
# ---------------------------------------------------------------------


class _Truthy:
    def __bool__(self):
        return True


class _Falsy:
    def __bool__(self):
        return False


@pytest.mark.parametrize("bad", [None, 0, 1, "", "true", "false", 1.0, [], _Truthy(), _Falsy()])
def test_include_character_anchor_rejects_non_bool(bad):
    with pytest.raises(TypeError) as exc_info:
        build_scene_image_prompt(BASE, include_character_anchor=bad, include_color_anchor=False)
    assert str(exc_info.value) == "include_character_anchor must be a bool"


@pytest.mark.parametrize("bad", [None, 0, 1, "", "true", "false", 1.0, [], _Truthy(), _Falsy()])
def test_include_color_anchor_rejects_non_bool(bad):
    with pytest.raises(TypeError) as exc_info:
        build_scene_image_prompt(BASE, include_character_anchor=False, include_color_anchor=bad)
    assert str(exc_info.value) == "include_color_anchor must be a bool"


def test_include_character_anchor_type_checked_before_include_color_anchor():
    """Both flags invalid at once -> the character-anchor message wins,
    proving a stable, deterministic check order rather than an
    implementation-dependent one."""
    with pytest.raises(TypeError) as exc_info:
        build_scene_image_prompt(BASE, include_character_anchor=None, include_color_anchor=None)
    assert str(exc_info.value) == "include_character_anchor must be a bool"


# ---------------------------------------------------------------------
# base_description content is never inspected/sanitized/de-duplicated
# ---------------------------------------------------------------------


def test_base_description_may_itself_contain_anchor_like_text_unaltered():
    tricky = "A scene that mentions SAFETY_SUFFIX and CHARACTER_ANCHOR by name."
    result = build_scene_image_prompt(tricky, include_character_anchor=False, include_color_anchor=False)

    base_component = result.split("\n\n", 1)[0]
    assert base_component == tricky


# ---------------------------------------------------------------------
# module has no prohibited imports/calls — proven via source inspection
# ---------------------------------------------------------------------


def test_module_has_no_prohibited_imports():
    source = inspect.getsource(scene_image_prompt)
    tree = ast.parse(source)
    import_nodes = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]

    unexpected = [
        node
        for node in import_nodes
        if not (isinstance(node, ast.ImportFrom) and node.module == "__future__")
    ]

    assert unexpected == [], (
        "src/core/scene_image_prompt.py must import nothing beyond "
        f"'from __future__ import annotations'; found: {[ast.dump(n) for n in unexpected]}"
    )


def test_module_defines_no_function_besides_build_scene_image_prompt():
    """A lightweight structural proxy for 'this module does nothing but
    assemble strings' — no helper that could plausibly wrap I/O."""
    source = inspect.getsource(scene_image_prompt)
    tree = ast.parse(source)
    function_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

    assert function_names == ["build_scene_image_prompt"]


# ---------------------------------------------------------------------
# visual-identity-v2 content guards (2026-09-17) — phrase-level, not
# bare-word: SAFETY_SUFFIX legitimately contains "extra limbs" (an
# anatomical-defect exclusion), so bare "mouth"/"limb" are never banned,
# only rigging-specific phrases from the removed programmatic
# mouth/limb-animation feature (see CLAUDE.md's roadmap decision).
# ---------------------------------------------------------------------

_BANNED_RIGGING_PHRASES = (
    "viseme",
    "lip-sync",
    "lip sync",
    "lipsync",
    "rhubarb",
    "mouthbox",
    "limbbox",
    "mouth_box",
    "limb_box",
    "mouth_cues",
    "mouth cues",
    "mouth animation",
    "limb sway",
    "limb_sway",
    "mouth sway",
    "mouth rig",
    "limb rig",
    "rig the mouth",
    "rig the limb",
)

_SCENE_DIRECTION_WORDS = ("camera", "pan", "zoom", "shot", "frame")

_SAFETY_SUFFIX_REQUIRED_TERMS = (
    "readable text",
    "captions",
    "speech bubbles",
    "logos",
    "watermarks",
    "photorealism",
    "3d rendering",
    "anime",
    "extra fingers",
    "extra limbs",
    "distorted faces",
    "duplicated characters",
    "cluttered backgrounds",
)


@pytest.mark.parametrize("phrase", _BANNED_RIGGING_PHRASES)
def test_character_anchor_contains_no_banned_rigging_phrase(phrase):
    assert phrase not in CHARACTER_ANCHOR.lower()


@pytest.mark.parametrize("phrase", _BANNED_RIGGING_PHRASES)
def test_color_anchor_contains_no_banned_rigging_phrase(phrase):
    assert phrase not in COLOR_ANCHOR.lower()


@pytest.mark.parametrize("phrase", _BANNED_RIGGING_PHRASES)
def test_safety_suffix_contains_no_banned_rigging_phrase(phrase):
    assert phrase not in SAFETY_SUFFIX.lower()


@pytest.mark.parametrize("word", _SCENE_DIRECTION_WORDS)
def test_character_anchor_contains_no_scene_direction_word(word):
    assert word not in CHARACTER_ANCHOR.lower()


@pytest.mark.parametrize("word", _SCENE_DIRECTION_WORDS)
def test_color_anchor_contains_no_scene_direction_word(word):
    assert word not in COLOR_ANCHOR.lower()


@pytest.mark.parametrize("term", _SAFETY_SUFFIX_REQUIRED_TERMS)
def test_safety_suffix_contains_required_exclusion_term(term):
    assert term in SAFETY_SUFFIX.lower()
