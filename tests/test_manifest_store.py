"""Tests for src/core/manifest_store.py: atomic save, load, and the error
handling around both. No network, no provider, no database."""
import json
from datetime import datetime, timezone

import pytest

from src.core.manifest_builder import build_video_manifest
from src.core.manifest_store import ManifestStoreError, load_manifest, save_manifest
from src.utils.channel_config import get_channel_policy


FIXED_CREATED_AT = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _valid_story_input(**overrides) -> dict:
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


def _valid_scene_plan() -> dict:
    return dict(
        scenes=(
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
        ),
        role_outfits=(),
    )


def _build_manifest():
    return build_video_manifest(
        _valid_story_input(),
        _valid_scene_plan(),
        get_channel_policy(),
        created_at=FIXED_CREATED_AT,
    )


# ---------------------------------------------------------------------
# Save / load round trip
# ---------------------------------------------------------------------


def test_round_trip_preserves_manifest(tmp_path):
    manifest = _build_manifest()
    path = save_manifest(manifest, tmp_path / "manifest.json")

    loaded = load_manifest(path)

    assert loaded == manifest


def test_save_manifest_returns_the_path(tmp_path):
    manifest = _build_manifest()
    target = tmp_path / "manifest.json"

    result = save_manifest(manifest, target)

    assert result == target
    assert target.exists()


def test_save_creates_parent_directories(tmp_path):
    manifest = _build_manifest()
    target = tmp_path / "projects" / "why-we-care" / "manifest.json"

    save_manifest(manifest, target)

    assert target.exists()


def test_save_does_not_leave_a_temp_file_behind(tmp_path):
    manifest = _build_manifest()
    target = tmp_path / "manifest.json"

    save_manifest(manifest, target)

    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == ["manifest.json"]


def test_save_overwrite_is_atomic_and_leaves_valid_content(tmp_path):
    target = tmp_path / "manifest.json"
    save_manifest(_build_manifest(), target)

    second_manifest = build_video_manifest(
        _valid_story_input(title="A Different Title"),
        _valid_scene_plan(),
        get_channel_policy(),
        created_at=FIXED_CREATED_AT,
    )
    save_manifest(second_manifest, target)

    loaded = load_manifest(target)
    assert loaded.story_input.title == "A Different Title"
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == ["manifest.json"]


def test_saved_file_is_formatted_utf8_json_with_sorted_top_level_keys(tmp_path):
    target = tmp_path / "manifest.json"
    save_manifest(_build_manifest(), target)

    text = target.read_text(encoding="utf-8")
    assert text.startswith("{\n")  # pretty-printed, not a single line

    parsed = json.loads(text)
    assert isinstance(parsed, dict)
    assert list(parsed.keys()) == sorted(parsed.keys())


# ---------------------------------------------------------------------
# Load error handling
# ---------------------------------------------------------------------


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(ManifestStoreError):
        load_manifest(tmp_path / "does-not-exist.json")


def test_load_invalid_json_raises(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("{not valid json")
    with pytest.raises(ManifestStoreError):
        load_manifest(path)


def test_load_non_object_json_raises(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("[1, 2, 3]")
    with pytest.raises(ManifestStoreError):
        load_manifest(path)


def test_load_json_scalar_raises(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('"just a string"')
    with pytest.raises(ManifestStoreError):
        load_manifest(path)


def test_load_invalid_manifest_content_raises(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('{"manifest_version": "1.0"}')  # missing every other required field
    with pytest.raises(ManifestStoreError):
        load_manifest(path)


def test_load_non_utf8_file_raises(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_bytes(b"\xff\xfe\x00\x01invalid-utf8")
    with pytest.raises(ManifestStoreError):
        load_manifest(path)


def test_load_path_is_directory_raises(tmp_path):
    directory = tmp_path / "manifest.json"
    directory.mkdir()
    with pytest.raises(ManifestStoreError):
        load_manifest(directory)
