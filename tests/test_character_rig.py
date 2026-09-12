"""Sanity checks for the Phase-3/4 overlay functions — not testing exact
pixel output (that's what the manual visual tests in tools/phase3_test and
tools/phase4_test were for), just that the mechanics don't silently break:
right frame count, visemes actually differ, rotation doesn't crash or wipe
the image out."""
from pathlib import Path

from PIL import Image

from src.render.character_rig import LimbBox, MouthBox, apply_limb_sway, apply_mouth_animation, draw_mouth, rotate_limb


def _solid_image(size=(200, 200), color=(255, 255, 255, 255)) -> Image.Image:
    return Image.new("RGBA", size, color)


def test_draw_mouth_differs_between_closed_and_open_visemes():
    base = _solid_image()
    box = MouthBox(center_x=100, center_y=100, width=60, height=40)

    closed = draw_mouth(base, box, "X")
    wide_open = draw_mouth(base, box, "D")

    assert closed.tobytes() != wide_open.tobytes()
    assert closed.tobytes() != base.tobytes()


def test_apply_mouth_animation_writes_one_frame_per_cue(tmp_path):
    scene_path = tmp_path / "scene.png"
    _solid_image().save(scene_path)
    box = MouthBox(center_x=100, center_y=100, width=60, height=40)
    cues = [{"start": 0.0, "end": 0.2, "value": "X"}, {"start": 0.2, "end": 0.4, "value": "D"}]

    frames = apply_mouth_animation(scene_path, box, cues, tmp_path / "frames")

    assert len(frames) == 2
    assert all(p.exists() for p in frames)


def test_rotate_limb_at_zero_angle_keeps_region_opaque_center_unchanged():
    base = _solid_image(color=(255, 255, 255, 255))
    box = LimbBox(left=60, top=60, right=140, bottom=140, pivot_x=100, pivot_y=100)

    rotated = rotate_limb(base, box, 0.0)

    # center of the box (fully inside the feathered mask's opaque core)
    # should still be the same solid color after a no-op rotation
    assert rotated.getpixel((100, 100))[:3] == (255, 255, 255)


def test_rotate_limb_does_not_crash_on_nonzero_angle():
    base = _solid_image()
    box = LimbBox(left=60, top=60, right=140, bottom=140, pivot_x=100, pivot_y=70)

    rotated = rotate_limb(base, box, 15.0)

    assert rotated.size == base.size


def test_apply_limb_sway_writes_one_frame_per_angle(tmp_path):
    scene_path = tmp_path / "scene.png"
    _solid_image().save(scene_path)
    box = LimbBox(left=60, top=60, right=140, bottom=140, pivot_x=100, pivot_y=70)

    frames = apply_limb_sway(scene_path, box, [-5.0, 0.0, 5.0], tmp_path / "frames")

    assert len(frames) == 3
    assert all(p.exists() for p in frames)
