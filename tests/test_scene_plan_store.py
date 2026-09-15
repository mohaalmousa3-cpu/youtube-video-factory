"""Tests for src/core/scene_plan_store.py — save-only local persistence
for a generated ScenePlan. Mirrors tests/test_manifest_store.py's own
save_manifest() test shape where applicable."""
from __future__ import annotations

import json

import pytest

from src.core.scene_plan_store import ScenePlanStoreError, save_scene_plan
from src.models.scene import ScenePlan, ScenePlanItem


def _scene_plan() -> ScenePlan:
    return ScenePlan(
        scenes=(
            ScenePlanItem(
                scene_id="scene-01",
                sequence=1,
                narration_text="Why does being left out sting so much?",
                scene_type="establishing",
                narrative_beat="hook",
                visual_brief="Stickman character alone on a quiet street corner at dusk.",
                motion_mode="in",
                approval_state="draft",
            ),
        ),
    )


def test_save_writes_formatted_stable_key_order_json(tmp_path):
    path = tmp_path / "out" / "scenes.json"

    result = save_scene_plan(_scene_plan(), path)

    assert result == path
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")

    data = json.loads(text)
    assert data["scenes"][0]["scene_id"] == "scene-01"
    assert data["scenes"][0]["approval_state"] == "draft"
    assert data["role_outfits"] == []

    # stable, sorted key order
    raw_keys = list(data.keys())
    assert raw_keys == sorted(raw_keys)
    scene_keys = list(data["scenes"][0].keys())
    assert scene_keys == sorted(scene_keys)


def test_save_creates_parent_directories(tmp_path):
    path = tmp_path / "a" / "b" / "c" / "scenes.json"
    save_scene_plan(_scene_plan(), path)
    assert path.exists()


def test_save_round_trips_via_scene_plan_model_validate(tmp_path):
    path = tmp_path / "scenes.json"
    save_scene_plan(_scene_plan(), path)

    loaded = ScenePlan.model_validate(json.loads(path.read_text(encoding="utf-8")))
    assert loaded == _scene_plan()


def test_save_is_atomic_no_leftover_tmp_file_on_success(tmp_path):
    path = tmp_path / "scenes.json"
    save_scene_plan(_scene_plan(), path)

    remaining = list(tmp_path.iterdir())
    assert remaining == [path]  # no .scenes.json.<random>.tmp leftover


def test_save_failure_raises_scene_plan_store_error_and_cleans_up_tmp_file(tmp_path, monkeypatch):
    path = tmp_path / "scenes.json"

    import src.core.scene_plan_store as store_module

    def _boom(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(store_module.os, "replace", _boom)

    with pytest.raises(ScenePlanStoreError):
        save_scene_plan(_scene_plan(), path)

    assert not path.exists()
    assert list(tmp_path.iterdir()) == []  # temp file was cleaned up, nothing orphaned
