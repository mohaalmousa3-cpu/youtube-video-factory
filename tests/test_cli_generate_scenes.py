"""Tests for the `generate-scenes` CLI command (src/cli.py
cmd_generate_scenes). Fake providers only via monkeypatch on
src.core.scene_generator.generate_scene_plan — no live network or API
call is ever made from this file. Mirrors the argparse.Namespace-driven
handler-test pattern used by every other tests/test_cli_*.py file."""
from __future__ import annotations

import argparse
import json

import pytest

import src.cli as cli
from src.core.scene_generator import SceneGenerationError
from src.models.scene import ScenePlan, ScenePlanItem


def _story_input_json() -> dict:
    return dict(
        story_id="why-we-care-what-people-think",
        title="Why We Care So Much What People Think",
        topic="social psychology",
        target_duration_seconds=480.0,
        language="en-US",
        viewer_facing_language="English",
        approval_status="approved",
    )


def _write_story_input(tmp_path) -> str:
    path = tmp_path / "story.json"
    path.write_text(json.dumps(_story_input_json()), encoding="utf-8")
    return str(path)


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


def _ns(**kwargs) -> argparse.Namespace:
    defaults = dict(story="story.json", output=None)
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------
# parser registration
# ---------------------------------------------------------------------


def test_parser_registration():
    parser = cli.build_parser()
    args = parser.parse_args(["generate-scenes", "--story", "story.json"])
    assert args.func is cli.cmd_generate_scenes
    assert args.story == "story.json"
    assert args.output is None


def test_parser_accepts_optional_output():
    parser = cli.build_parser()
    args = parser.parse_args(["generate-scenes", "--story", "story.json", "--output", "out.json"])
    assert args.output == "out.json"


def test_parser_requires_story():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["generate-scenes"])


# ---------------------------------------------------------------------
# success: no --output -> summary only, no file written
# ---------------------------------------------------------------------


def test_cli_success_without_output_prints_summary_and_writes_nothing(tmp_path, capsys, monkeypatch):
    story_path = _write_story_input(tmp_path)
    monkeypatch.setattr(
        "src.core.scene_generator.generate_scene_plan", lambda story_input, channel_policy: _scene_plan()
    )

    before = set(tmp_path.iterdir())
    rc = cli.cmd_generate_scenes(_ns(story=story_path, output=None))
    out = capsys.readouterr().out
    after = set(tmp_path.iterdir())

    assert rc == 0
    assert "generate-scenes: OK" in out
    assert "scene_count: 1" in out
    assert "scene-01" in out
    assert after == before  # nothing written to the filesystem


def test_cli_success_without_output_lists_each_scene(tmp_path, capsys, monkeypatch):
    story_path = _write_story_input(tmp_path)

    def _two_scenes(story_input, channel_policy):
        return ScenePlan(
            scenes=(
                ScenePlanItem(
                    scene_id="scene-01", sequence=1, narration_text="First.", scene_type="establishing",
                    narrative_beat="hook", visual_brief="v1", motion_mode="in", approval_state="draft",
                ),
                ScenePlanItem(
                    scene_id="scene-02", sequence=2, narration_text="Second.", scene_type="closing",
                    narrative_beat="payoff", visual_brief="v2", motion_mode="static", approval_state="draft",
                ),
            )
        )

    monkeypatch.setattr("src.core.scene_generator.generate_scene_plan", _two_scenes)

    rc = cli.cmd_generate_scenes(_ns(story=story_path, output=None))
    out = capsys.readouterr().out

    assert rc == 0
    assert "scene_count: 2" in out
    assert "scene-01" in out and "First." in out
    assert "scene-02" in out and "Second." in out


# ---------------------------------------------------------------------
# success: --output -> file written, summary confirms it
# ---------------------------------------------------------------------


def test_cli_success_with_output_saves_file(tmp_path, capsys, monkeypatch):
    story_path = _write_story_input(tmp_path)
    output_path = tmp_path / "scenes.json"
    monkeypatch.setattr(
        "src.core.scene_generator.generate_scene_plan", lambda story_input, channel_policy: _scene_plan()
    )

    rc = cli.cmd_generate_scenes(_ns(story=story_path, output=str(output_path)))
    out = capsys.readouterr().out

    assert rc == 0
    assert "generate-scenes: OK" in out
    assert str(output_path) in out
    assert output_path.exists()

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["scenes"][0]["scene_id"] == "scene-01"
    assert saved["scenes"][0]["approval_state"] == "draft"


def test_cli_never_creates_a_project_manifest_or_database(tmp_path, monkeypatch):
    story_path = _write_story_input(tmp_path)
    output_path = tmp_path / "scenes.json"
    monkeypatch.setattr(
        "src.core.scene_generator.generate_scene_plan", lambda story_input, channel_policy: _scene_plan()
    )

    before = set(tmp_path.iterdir())
    cli.cmd_generate_scenes(_ns(story=story_path, output=str(output_path)))
    after = set(tmp_path.iterdir())

    assert after - before == {output_path}  # exactly one new file: the scene plan itself
    assert not (tmp_path / "data").exists()  # no database directory created


# ---------------------------------------------------------------------
# failures -> non-zero exit, concise stderr, nothing written
# ---------------------------------------------------------------------


def test_cli_generation_failure_returns_error_and_writes_nothing(tmp_path, capsys, monkeypatch):
    story_path = _write_story_input(tmp_path)
    output_path = tmp_path / "scenes.json"

    def _always_fails(story_input, channel_policy):
        raise SceneGenerationError(
            "scene generation failed after attempting GroqProvider (2x) then "
            "TokenRouterProvider (2x); last failure from TokenRouterProvider: provider call failed: down"
        )

    monkeypatch.setattr("src.core.scene_generator.generate_scene_plan", _always_fails)

    rc = cli.cmd_generate_scenes(_ns(story=story_path, output=str(output_path)))
    err = capsys.readouterr().err

    assert rc == 1
    assert "generate-scenes: FAILED" in err
    assert "GroqProvider" in err
    assert not output_path.exists()


def test_cli_missing_story_file_returns_error(tmp_path, capsys):
    rc = cli.cmd_generate_scenes(_ns(story=str(tmp_path / "does-not-exist.json"), output=None))
    err = capsys.readouterr().err

    assert rc == 1
    assert "generate-scenes: FAILED" in err


def test_cli_malformed_story_file_returns_error(tmp_path, capsys):
    bad_path = tmp_path / "story.json"
    bad_path.write_text("not json", encoding="utf-8")

    rc = cli.cmd_generate_scenes(_ns(story=str(bad_path), output=None))
    err = capsys.readouterr().err

    assert rc == 1
    assert "generate-scenes: FAILED" in err


def test_cli_save_failure_after_successful_generation_returns_error(tmp_path, capsys, monkeypatch):
    story_path = _write_story_input(tmp_path)
    monkeypatch.setattr(
        "src.core.scene_generator.generate_scene_plan", lambda story_input, channel_policy: _scene_plan()
    )

    from src.core.scene_plan_store import ScenePlanStoreError

    def _save_boom(scene_plan, path):
        raise ScenePlanStoreError(f"could not save scene plan to {path}: simulated disk failure")

    monkeypatch.setattr("src.core.scene_plan_store.save_scene_plan", _save_boom)

    rc = cli.cmd_generate_scenes(_ns(story=story_path, output=str(tmp_path / "scenes.json")))
    err = capsys.readouterr().err

    assert rc == 1
    assert "generate-scenes: FAILED" in err
