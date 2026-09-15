"""Read-only readiness check for the (not-yet-implemented) mouth-animation
generation slice. See src/cli.py's cmd_mouth_animation_preflight for the CLI
command.

This module performs NO generation and NO write of any kind: no
lipsync_rhubarb.get_mouth_cues(), no character_rig.apply_mouth_animation(),
no ffmpeg, no image transform (Pillow is used only to verify the file opens
and to read its dimensions — see validate_image_and_box()'s docstring), no
SQLite access (that is exclusively src/cli.py's job — see below), no
manifest write, no artifact registration, no lifecycle transition.

Two independent validation phases, deliberately split so the CLI can run
the first (pure, no I/O) before ever opening a database connection:

1. parse_mouth_box() — syntax/type/range validation of the raw --mouth-box
   string only. No filesystem, no image, no database.
2. validate_image_and_box() — needs the already-resolved local visual image
   path (the CLI resolves that via a read-only SQLite artifact lookup
   first, then closes its connection, THEN calls this) — opens the image
   locally with Pillow and checks the parsed MouthBox's rectangle fits
   inside it.

check_rhubarb_available() is a third, independent leg: a local
health_check() subprocess call (Rhubarb's own `--version` probe), not a
real lip-sync invocation.

Geometry (authoritative — later Pillow drawing must use this same
arithmetic, not a float-based center +/- width/2 approximation):

    left   = center_x - (width // 2)
    top    = center_y - (height // 2)
    right  = left + width
    bottom = top + height

Required: 0 <= left < right <= image_width and 0 <= top < bottom <= image_height."""
from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from src.providers.lipsync_rhubarb import health_check
from src.render.character_rig import MouthBox

# Strict, ASCII-only decimal-digit grammar for one --mouth-box component.
# Deliberately NOT str.isdigit() (accepts Unicode digit categories, e.g.
# Arabic-Indic) and NOT a bare `\d` (in Python's default Unicode regex
# mode, `\d` also matches non-ASCII decimal digits) — an explicit `[0-9]`
# character class matches only the literal ASCII code points, regardless
# of Unicode mode, so it rejects whitespace, signs, underscore-grouping,
# decimals, scientific notation, and any non-ASCII digit by construction,
# not by accident. `\Z` (not `$`) anchors the end strictly — plain `$`
# also matches just before a trailing '\n', which would otherwise let
# "5\n" slip through as if it were "5".
_ASCII_DECIMAL_TOKEN_RE = re.compile(r"^[0-9]+\Z")


class MouthAnimationPreflightError(Exception):
    """Raised for any reason this readiness check fails: malformed
    --mouth-box syntax/range, a box that doesn't fit inside the image, an
    unopenable/corrupt image, or an unavailable Rhubarb binary. Message
    text is always a short, fixed, sanitized string — never a resolved
    filesystem path or a raw exception representation."""


def parse_mouth_box(raw: str) -> MouthBox:
    """Parse '--mouth-box CENTER_X,CENTER_Y,WIDTH,HEIGHT'. Validates syntax
    (exactly four comma-separated components), that each component matches
    the strict ASCII-only decimal-digit grammar `^[0-9]+$` (see
    _ASCII_DECIMAL_TOKEN_RE) before any conversion to int, that width/height
    are positive, and that center_x/center_y are not negative. Does not
    know about — and cannot check — any image; see validate_image_and_box()
    for the bounds check against an actual image's dimensions."""
    parts = raw.split(",")
    if len(parts) != 4:
        raise MouthAnimationPreflightError(
            "--mouth-box must be exactly four comma-separated integers: CENTER_X,CENTER_Y,WIDTH,HEIGHT"
        )

    for part in parts:
        if not _ASCII_DECIMAL_TOKEN_RE.match(part):
            raise MouthAnimationPreflightError("--mouth-box values must be base-10 integers")

    center_x, center_y, width, height = (int(part, base=10) for part in parts)

    if width <= 0 or height <= 0:
        raise MouthAnimationPreflightError("--mouth-box width and height must be positive")
    if center_x < 0 or center_y < 0:
        # Unreachable now that every component has already been proven to
        # match ^[0-9]+$ (no '-' can ever match that pattern) — kept as an
        # explicit, self-documenting check rather than silently relying on
        # the regex above to keep guaranteeing this forever.
        raise MouthAnimationPreflightError("--mouth-box center_x and center_y must not be negative")

    return MouthBox(center_x=center_x, center_y=center_y, width=width, height=height)


def validate_image_and_box(source_visual_path: Path, mouth_box: MouthBox) -> tuple[int, int]:
    """Open `source_visual_path` locally with Pillow and confirm
    `mouth_box`'s rectangle lies entirely inside it. Returns
    (image_width, image_height) on success; raises
    MouthAnimationPreflightError otherwise. Never registers, transforms,
    or writes anything.

    Two separate Image.open() calls, deliberately: `.verify()` checks file
    integrity without decoding pixel data, but per Pillow's own contract
    the Image object is unusable for anything else once verify() returns —
    so dimensions are read from a second, fresh open() afterward (a cheap
    header read, not a full decode)."""
    try:
        with Image.open(source_visual_path) as img:
            img.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise MouthAnimationPreflightError(
            "registered visual artifact image could not be opened or is corrupt"
        ) from exc

    try:
        with Image.open(source_visual_path) as img:
            image_width, image_height = img.size
    except (UnidentifiedImageError, OSError) as exc:
        raise MouthAnimationPreflightError(
            "registered visual artifact image could not be opened or is corrupt"
        ) from exc

    if image_width <= 0 or image_height <= 0:
        raise MouthAnimationPreflightError("registered visual artifact image has non-positive dimensions")

    left = mouth_box.center_x - (mouth_box.width // 2)
    top = mouth_box.center_y - (mouth_box.height // 2)
    right = left + mouth_box.width
    bottom = top + mouth_box.height

    if not (0 <= left < right <= image_width):
        raise MouthAnimationPreflightError("--mouth-box rectangle extends outside the image bounds")
    if not (0 <= top < bottom <= image_height):
        raise MouthAnimationPreflightError("--mouth-box rectangle extends outside the image bounds")

    return image_width, image_height


def check_rhubarb_available() -> None:
    """Raises MouthAnimationPreflightError if lipsync_rhubarb.health_check()
    is False. Never calls get_mouth_cues() or any other Rhubarb
    invocation."""
    if not health_check():
        raise MouthAnimationPreflightError("Rhubarb lip-sync tool is unavailable (binary not found)")
