"""Procedural per-frame overlays (mouth shape, limb rotation) on top of a
still image — no AI call per frame, just Pillow. Mouth shapes are simple
filled ellipses, not AI-generated: the art style is plain enough that a
drawn shape matches, and it guarantees pixel-perfect alignment on every
frame, which independent AI generations could never give us.

Two entry points depending on what the base image already is:
- `apply_mouth_animation()` / `apply_limb_sway()` — draw directly onto an
  already-composited full-scene image (character generated in-scene by
  Qwen). This is what the current pipeline actually produces (see
  CLAUDE.md), and is the one to use for new work — Phase 3/4 of the
  pipeline-improvement plan, validated end to end.
- `render_talking_frames()` — the older Phase-1 approach: composites a
  separate fixed transparent character cutout over a separate background.
  Kept for reference, not currently used (see "Character animation" in
  CLAUDE.md for why the pipeline moved away from cutout compositing)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFilter

from src.providers.lipsync_rhubarb import MouthCue

# (width, height) as a fraction of mouth-box size, per Rhubarb viseme.
# X/A: closed. B/C/G/H: slightly open. D: wide open. E: rounded. F: puckered.
_MOUTH_SHAPES: Dict[str, tuple] = {
    "X": (1.0, 0.15),
    "A": (1.0, 0.15),
    "B": (0.8, 0.3),
    "C": (0.9, 0.5),
    "D": (0.9, 0.8),
    "E": (0.6, 0.6),
    "F": (0.4, 0.4),
    "G": (0.7, 0.25),
    "H": (0.85, 0.45),
}


@dataclass
class MouthBox:
    """Pixel rectangle on the base character image where the mouth goes,
    measured once by hand against character_base_transparent.png."""
    center_x: int
    center_y: int
    width: int
    height: int


def draw_mouth(base: Image.Image, box: MouthBox, viseme: str) -> Image.Image:
    """Draws over `base`'s own static mouth line before drawing the new
    viseme — without this, a viseme narrower/shorter than the original
    mouth (e.g. the closed "X"/"A" shapes) leaves the original line
    visible around it, a double-mouth artifact. A first version sized the
    erase patch to exactly the hand-measured MouthBox with a 3-5% margin
    left for the blur's own falloff — the original line still bled through
    near the box edges, because that margin was blurred away *from* full
    opacity rather than being extra coverage *beyond* the box. Fixed by:
    (1) sizing the patch 20% larger than the box in each dimension, so the
    erase region overshoots the box on every side instead of stopping
    exactly at it; (2) keeping the un-blurred solid core out to 99% of
    that already-enlarged patch (not just 97% of the box itself), so the
    blur's falloff happens almost entirely in the new 20%-larger margin,
    away from the box interior; edges still feather into transparency at
    the very outside via a blurred elliptical alpha mask (same technique
    as rotate_limb()), so the patch blends into the surrounding face
    rather than reading as a flat seam."""
    frame = base.copy().convert("RGBA")
    patch_width = round(box.width * 1.2)
    patch_height = round(box.height * 1.2)
    left = box.center_x - patch_width // 2
    top = box.center_y - patch_height // 2

    # Average several points around the enlarged patch instead of trusting
    # one single pixel — a lone sample can land on an anti-aliased edge or
    # a stray fleck of texture and paint the whole patch the wrong color.
    rgb_frame = frame.convert("RGB")
    img_w, img_h = rgb_frame.size

    def _clamp(value, lo, hi):
        return max(lo, min(value, hi))

    sample_points = [
        (_clamp(box.center_x, 0, img_w - 1), _clamp(top - 4, 0, img_h - 1)),
        (_clamp(box.center_x - patch_width // 4, 0, img_w - 1), _clamp(top - 4, 0, img_h - 1)),
        (_clamp(box.center_x + patch_width // 4, 0, img_w - 1), _clamp(top - 4, 0, img_h - 1)),
        (_clamp(left - 4, 0, img_w - 1), _clamp(box.center_y, 0, img_h - 1)),
        (_clamp(left + patch_width + 4, 0, img_w - 1), _clamp(box.center_y, 0, img_h - 1)),
    ]
    samples = [rgb_frame.getpixel(p) for p in sample_points]
    patch_color = tuple(sum(channel) // len(samples) for channel in zip(*samples))

    patch = Image.new("RGBA", (patch_width, patch_height), patch_color + (255,))
    mask = Image.new("L", (patch_width, patch_height), 0)
    ImageDraw.Draw(mask).ellipse(
        [patch_width * 0.01, patch_height * 0.01, patch_width * 0.99, patch_height * 0.99], fill=255
    )
    mask = mask.filter(ImageFilter.GaussianBlur(radius=3))
    patch.putalpha(mask)
    frame.alpha_composite(patch, dest=(left, top))

    draw = ImageDraw.Draw(frame)
    w_frac, h_frac = _MOUTH_SHAPES.get(viseme, _MOUTH_SHAPES["X"])
    w, h = box.width * w_frac, box.height * h_frac
    x0, y0 = box.center_x - w / 2, box.center_y - h / 2
    x1, y1 = box.center_x + w / 2, box.center_y + h / 2
    draw.ellipse([x0, y0, x1, y1], fill=(20, 20, 20, 255))
    return frame


def apply_mouth_animation(
    scene_image_path: Path, mouth_box: MouthBox, mouth_cues: List[MouthCue], out_dir: Path
) -> List[Path]:
    """Phase-3 pipeline improvement: draws the mouth directly onto an
    already-composited full-scene image (character generated in-scene by
    Qwen, not a separate cutout) instead of compositing character+background
    like render_talking_frames does — full-scene generation is what actually
    ships (see CLAUDE.md), this just adds the missing mouth movement on top
    of it without reviving the cutout/rig approach. mouth_box coordinates
    are specific to the one scene image they were measured against — a new
    scene image needs its own box, there's no detector picking this out
    automatically (see draw_mouth's docstring: procedural shapes, not AI)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = Image.open(scene_image_path).convert("RGBA")

    paths = []
    for i, cue in enumerate(mouth_cues):
        frame = draw_mouth(scene, mouth_box, cue["value"])
        path = out_dir / f"frame_{i:04d}.png"
        frame.convert("RGB").save(path)
        paths.append(path)
    return paths


@dataclass
class LimbBox:
    """Pixel rectangle containing one limb plus its pivot (joint) point —
    measured by hand against one specific scene image, same as MouthBox.
    A new scene image needs its own box; nothing detects this automatically."""
    left: int
    top: int
    right: int
    bottom: int
    pivot_x: int
    pivot_y: int


def rotate_limb(base: Image.Image, box: LimbBox, angle_degrees: float) -> Image.Image:
    """Phase-4 pipeline improvement: rigid crop-rotate-paste around the joint
    pivot — not a real bend at the joint, the whole limb (and whatever
    background is inside the box) rotates as one flat piece.

    A first version pasted the rotated rectangle back with a hard edge —
    produced an obvious rotated-square seam even on a plain grey test
    background, unusable. Fixed by feathering the box into an ellipse
    (ImageFilter.GaussianBlur on a solid ellipse alpha mask) so the pasted
    region fades to fully transparent at its edges instead of cutting off
    sharply; only clean on a full illustrated background if the box is
    fitted tight enough that the blurred edge lands on visually uniform
    area (plain skin/fabric), not across a hard outline — still untested on
    a real generated scene, see CLAUDE.md."""
    region = base.crop((box.left, box.top, box.right, box.bottom)).convert("RGBA")
    w, h = region.size
    local_pivot = (box.pivot_x - box.left, box.pivot_y - box.top)

    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).ellipse([w * 0.03, h * 0.02, w * 0.97, h * 0.98], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=3))
    region.putalpha(mask)

    rotated = region.rotate(angle_degrees, resample=Image.BICUBIC, center=local_pivot, fillcolor=(0, 0, 0, 0))
    frame = base.copy().convert("RGBA")
    frame.alpha_composite(rotated, dest=(box.left, box.top))
    return frame


def apply_limb_sway(base_path: Path, box: LimbBox, angles: List[float], out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Image.open(base_path).convert("RGBA")
    paths = []
    for i, angle in enumerate(angles):
        frame = rotate_limb(base, box, angle)
        path = out_dir / f"limb_{i:04d}.png"
        frame.convert("RGB").save(path)
        paths.append(path)
    return paths


def render_talking_frames(
    character_path: Path, background_path: Path, mouth_box: MouthBox, mouth_cues: List[MouthCue], out_dir: Path
) -> List[Path]:
    """One image per mouth cue (held for its duration by the ffmpeg concat
    step) — not per-frame-at-30fps, since the mouth shape is the only thing
    that changes and holding it is indistinguishable on screen."""
    out_dir.mkdir(parents=True, exist_ok=True)
    character = Image.open(character_path).convert("RGBA")
    background = Image.open(background_path).convert("RGBA").resize(character.size)

    paths = []
    for i, cue in enumerate(mouth_cues):
        frame = draw_mouth(character, mouth_box, cue["value"])
        composited = background.copy()
        composited.alpha_composite(frame)
        path = out_dir / f"frame_{i:04d}.png"
        composited.convert("RGB").save(path)
        paths.append(path)
    return paths
