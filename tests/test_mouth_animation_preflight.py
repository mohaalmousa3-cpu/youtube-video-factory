"""Tests for src/core/mouth_animation_preflight.py — pure readiness checks,
no SQLite, no generation. Every image fixture is a small, real, local PNG
written directly with Pillow; lipsync_rhubarb.health_check is mocked at its
point of import in this module — no real Rhubarb binary, no network call
anywhere in this file."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.core.mouth_animation_preflight import (
    MouthAnimationPreflightError,
    check_rhubarb_available,
    parse_mouth_box,
    validate_image_and_box,
)
from src.render.character_rig import MouthBox

MODULE = "src.core.mouth_animation_preflight"


def _png(path: Path, size: tuple[int, int]) -> None:
    Image.new("RGB", size, color=(200, 150, 100)).save(path, format="PNG")


# ---------------------------------------------------------------------
# parse_mouth_box
# ---------------------------------------------------------------------


def test_parse_mouth_box_valid():
    box = parse_mouth_box("100,80,60,40")
    assert box == MouthBox(center_x=100, center_y=80, width=60, height=40)


def test_parse_mouth_box_wrong_component_count():
    with pytest.raises(MouthAnimationPreflightError, match="exactly four comma-separated integers"):
        parse_mouth_box("100,80,60")


def test_parse_mouth_box_too_many_components():
    with pytest.raises(MouthAnimationPreflightError, match="exactly four comma-separated integers"):
        parse_mouth_box("100,80,60,40,20")


def test_parse_mouth_box_non_integer_component():
    with pytest.raises(MouthAnimationPreflightError, match="base-10 integers"):
        parse_mouth_box("100,80,60.5,40")


def test_parse_mouth_box_non_numeric_component():
    with pytest.raises(MouthAnimationPreflightError, match="base-10 integers"):
        parse_mouth_box("abc,80,60,40")


def test_parse_mouth_box_non_positive_width():
    with pytest.raises(MouthAnimationPreflightError, match="width and height must be positive"):
        parse_mouth_box("100,80,0,40")


def test_parse_mouth_box_non_positive_height():
    with pytest.raises(MouthAnimationPreflightError, match="width and height must be positive"):
        parse_mouth_box("100,80,60,0")


def test_parse_mouth_box_negative_center_x_is_rejected_as_non_ascii_token():
    """A '-' sign never matches the strict ^[0-9]+$ grammar, so this is
    rejected at the token-grammar stage (base-10 integers), not by the
    (now unreachable) explicit negative-value check."""
    with pytest.raises(MouthAnimationPreflightError, match="base-10 integers"):
        parse_mouth_box("-1,80,60,40")


def test_parse_mouth_box_negative_center_y_is_rejected_as_non_ascii_token():
    with pytest.raises(MouthAnimationPreflightError, match="base-10 integers"):
        parse_mouth_box("100,-1,60,40")


def test_parse_mouth_box_center_at_zero_is_allowed():
    box = parse_mouth_box("0,0,10,10")
    assert box == MouthBox(center_x=0, center_y=0, width=10, height=10)


# ---------------------------------------------------------------------
# parse_mouth_box — strict ASCII-only token grammar ^[0-9]+$
# ---------------------------------------------------------------------


def test_parse_mouth_box_accepts_leading_zeros():
    box = parse_mouth_box("001,002,003,004")
    assert box == MouthBox(center_x=1, center_y=2, width=3, height=4)


@pytest.mark.parametrize(
    "raw",
    [
        " 5,50,40,20",  # leading whitespace
        "5 ,50,40,20",  # trailing whitespace
        "5\t,50,40,20",  # tab
        "5\n,50,40,20",  # newline
        "100,50,40, 20",  # interior/leading whitespace on a later component
    ],
)
def test_parse_mouth_box_rejects_whitespace_token(raw):
    with pytest.raises(MouthAnimationPreflightError, match="base-10 integers"):
        parse_mouth_box(raw)


@pytest.mark.parametrize("raw", ["+5,50,40,20", "-5,50,40,20", "100,50,+40,20", "100,50,40,-20"])
def test_parse_mouth_box_rejects_explicit_sign_token(raw):
    with pytest.raises(MouthAnimationPreflightError, match="base-10 integers"):
        parse_mouth_box(raw)


def test_parse_mouth_box_rejects_underscore_grouped_token():
    with pytest.raises(MouthAnimationPreflightError, match="base-10 integers"):
        parse_mouth_box("1_000,50,40,20")


def test_parse_mouth_box_rejects_decimal_token():
    with pytest.raises(MouthAnimationPreflightError, match="base-10 integers"):
        parse_mouth_box("100,50,60.5,40")


def test_parse_mouth_box_rejects_scientific_notation_token():
    with pytest.raises(MouthAnimationPreflightError, match="base-10 integers"):
        parse_mouth_box("100,50,5e2,20")


@pytest.mark.parametrize("raw", ["true,50,40,20", "True,50,40,20"])
def test_parse_mouth_box_rejects_boolean_like_token(raw):
    with pytest.raises(MouthAnimationPreflightError, match="base-10 integers"):
        parse_mouth_box(raw)


def test_parse_mouth_box_rejects_unicode_arabic_indic_digit_token():
    """'٤٥' (Arabic-Indic digits for 45) is a valid target for Python's own
    int(x, base=10) — this is exactly the accidental permissiveness the
    strict ^[0-9]+$ grammar exists to close."""
    with pytest.raises(MouthAnimationPreflightError, match="base-10 integers"):
        parse_mouth_box("٤٥,50,40,20")


def test_parse_mouth_box_rejects_empty_component():
    with pytest.raises(MouthAnimationPreflightError, match="base-10 integers"):
        parse_mouth_box(",50,40,20")


def test_parse_mouth_box_rejects_other_non_ascii_character():
    with pytest.raises(MouthAnimationPreflightError, match="base-10 integers"):
        parse_mouth_box("100,50,4°40,20")  # embedded degree sign


def test_parse_mouth_box_wrong_component_count_error_is_distinct_from_invalid_token_error():
    """Wrong count is detected and reported BEFORE any token is even
    grammar-checked — a malformed count never produces the token-grammar
    message, and vice versa."""
    with pytest.raises(MouthAnimationPreflightError) as wrong_count:
        parse_mouth_box("1,2,3")
    with pytest.raises(MouthAnimationPreflightError) as bad_token:
        parse_mouth_box("1,2,3,abc")

    assert "exactly four comma-separated integers" in str(wrong_count.value)
    assert "exactly four comma-separated integers" not in str(bad_token.value)
    assert "base-10 integers" in str(bad_token.value)
    assert "base-10 integers" not in str(wrong_count.value)


# ---------------------------------------------------------------------
# validate_image_and_box — geometry (integer floor division, exactly as
# specified: left = center_x - (width // 2), right = left + width, etc.)
# ---------------------------------------------------------------------


def test_validate_image_and_box_valid_box_returns_dimensions(tmp_path):
    image_path = tmp_path / "scene.png"
    _png(image_path, (200, 100))
    box = MouthBox(center_x=100, center_y=50, width=40, height=20)

    width, height = validate_image_and_box(image_path, box)

    assert (width, height) == (200, 100)


def test_validate_image_and_box_valid_odd_sized_box(tmp_path):
    """left = 50 - (41 // 2) = 50 - 20 = 30; right = 30 + 41 = 71 <= 200.
    top = 50 - (23 // 2) = 50 - 11 = 39; bottom = 39 + 23 = 62 <= 100."""
    image_path = tmp_path / "scene.png"
    _png(image_path, (200, 100))
    box = MouthBox(center_x=50, center_y=50, width=41, height=23)

    width, height = validate_image_and_box(image_path, box)

    assert (width, height) == (200, 100)


def test_validate_image_and_box_box_exactly_fills_image_is_valid(tmp_path):
    """left=0, right=0+width=image_width; top=0, bottom=image_height — the
    boundary case '<=' must accept, not just '<'."""
    image_path = tmp_path / "scene.png"
    _png(image_path, (40, 20))
    box = MouthBox(center_x=20, center_y=10, width=40, height=20)

    width, height = validate_image_and_box(image_path, box)

    assert (width, height) == (40, 20)


@pytest.mark.parametrize(
    "center_x,center_y,width,height",
    [
        (5, 50, 40, 20),  # left = 5 - 20 = -15 < 0 -> crosses LEFT edge
        (195, 50, 40, 20),  # right = 175 + 40 = 215 > 200 -> crosses RIGHT edge
        (100, 5, 40, 20),  # top = 5 - 10 = -5 < 0 -> crosses TOP edge
        (100, 95, 40, 20),  # bottom = 85 + 20 = 105 > 100 -> crosses BOTTOM edge
    ],
)
def test_validate_image_and_box_rejects_box_crossing_each_edge(tmp_path, center_x, center_y, width, height):
    image_path = tmp_path / "scene.png"
    _png(image_path, (200, 100))
    box = MouthBox(center_x=center_x, center_y=center_y, width=width, height=height)

    with pytest.raises(MouthAnimationPreflightError, match="extends outside the image bounds"):
        validate_image_and_box(image_path, box)


@pytest.mark.parametrize(
    "center_x,center_y,width,height",
    [
        (5, 50, 41, 23),  # left = 5 - 20 = -15 < 0 -> crosses LEFT edge (odd box)
        (195, 50, 41, 23),  # right = 175 + 41 = 216 > 200 -> crosses RIGHT edge (odd box)
        (100, 5, 41, 23),  # top = 5 - 11 = -6 < 0 -> crosses TOP edge (odd box)
        (100, 95, 41, 23),  # bottom = 84 + 23 = 107 > 100 -> crosses BOTTOM edge (odd box)
    ],
)
def test_validate_image_and_box_rejects_odd_sized_box_crossing_each_edge(
    tmp_path, center_x, center_y, width, height
):
    image_path = tmp_path / "scene.png"
    _png(image_path, (200, 100))
    box = MouthBox(center_x=center_x, center_y=center_y, width=width, height=height)

    with pytest.raises(MouthAnimationPreflightError, match="extends outside the image bounds"):
        validate_image_and_box(image_path, box)


def test_validate_image_and_box_uses_integer_floor_division_not_float(tmp_path):
    """A box whose left edge would be exactly 0 under integer floor
    division (width // 2) but negative under float division (width / 2)
    must be ACCEPTED — proving the implementation uses // as specified,
    not a float approximation. center_x=15, width=31: 31//2=15 ->
    left=0 (valid); 31/2=15.5 -> a float-based left=-0.5 would (incorrectly
    for this contract) also still round to something usable, so the
    sharper proof is the reverse case below."""
    image_path = tmp_path / "scene.png"
    _png(image_path, (200, 100))
    # width=41 (odd): 41 // 2 = 20 (int floor), 41 / 2 = 20.5 (float).
    # center_x=20 -> left = 20 - 20 = 0 (valid under // ); a naive
    # float implementation computing left = center_x - width/2 = -0.5
    # would need to floor/round to decide validity — this box is
    # constructed so the two approaches would only agree if // is used
    # consistently for both left AND right (right = left + width, not
    # center_x + width/2), which is exactly what's specified.
    box = MouthBox(center_x=20, center_y=50, width=41, height=20)

    width_out, height_out = validate_image_and_box(image_path, box)

    assert (width_out, height_out) == (200, 100)


def test_validate_image_and_box_rejects_corrupt_file(tmp_path):
    image_path = tmp_path / "not_an_image.png"
    image_path.write_bytes(b"this is not a valid PNG file at all")
    box = MouthBox(center_x=5, center_y=5, width=4, height=4)

    with pytest.raises(MouthAnimationPreflightError, match="could not be opened or is corrupt"):
        validate_image_and_box(image_path, box)


def test_validate_image_and_box_rejects_missing_file(tmp_path):
    image_path = tmp_path / "does_not_exist.png"
    box = MouthBox(center_x=5, center_y=5, width=4, height=4)

    with pytest.raises(MouthAnimationPreflightError, match="could not be opened or is corrupt"):
        validate_image_and_box(image_path, box)


# ---------------------------------------------------------------------
# check_rhubarb_available
# ---------------------------------------------------------------------


def test_check_rhubarb_available_passes_when_health_check_true(monkeypatch):
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: True)
    check_rhubarb_available()  # must not raise


def test_check_rhubarb_available_raises_when_health_check_false(monkeypatch):
    monkeypatch.setattr(f"{MODULE}.health_check", lambda: False)
    with pytest.raises(MouthAnimationPreflightError, match="Rhubarb lip-sync tool is unavailable"):
        check_rhubarb_available()
