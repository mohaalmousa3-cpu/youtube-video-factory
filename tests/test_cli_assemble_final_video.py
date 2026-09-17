"""Tests for the `assemble-final-video` CLI command (src/cli.py
cmd_assemble_final_video). Same isolated_db / _create_registered_project
pattern as every other CLI-level test in this repo. The core pipeline
(src.core.final_video_assembly.assemble_final_video) is always mocked in
these tests — no real ffmpeg, no real SQLite work beyond project creation,
no network call anywhere in this file."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

import src.cli as cli
from src.core.manifest_builder import build_video_manifest
from src.core.manifest_store import save_manifest
from src.core.project_state_machine import create_initial_project, initial_transition_for
from src.database.project_repository import create_project
from src.utils.channel_config import get_channel_policy

FIXED_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
GENERATION_MODULE = "src.core.final_video_assembly"


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    from src.utils import config

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    config.get_settings.cache_clear()
    yield tmp_path
    config.get_settings.cache_clear()


def _story_input_dict() -> dict:
    return dict(
        story_id="why-we-care-what-people-think",
        title="Why We Care So Much What People Think",
        topic="social psychology",
        target_duration_seconds=480.0,
        language="en-US",
        viewer_facing_language="English",
        approval_status="approved",
    )


def _scene_plan_dict(scene_ids: tuple[str, ...] = ("scene-01",)) -> dict:
    return dict(
        scenes=tuple(
            dict(
                scene_id=scene_id,
                sequence=i,
                narration_text=f"Narration for {scene_id}.",
                scene_type="narration",
                narrative_beat="setup",
                visual_brief=f"Visual brief for {scene_id}.",
                motion_mode="static",
                approval_state="approved",
            )
            for i, scene_id in enumerate(scene_ids, start=1)
        ),
        role_outfits=(),
    )


def _create_registered_project(projects_root: Path, scene_ids: tuple[str, ...] = ("scene-01",)):
    from src.database.db import get_connection, init_db

    manifest = build_video_manifest(
        _story_input_dict(), _scene_plan_dict(scene_ids), get_channel_policy(), created_at=FIXED_NOW
    )
    manifest_path = projects_root / manifest.project_id / "manifest.json"
    save_manifest(manifest, manifest_path)

    project = create_initial_project(manifest_path, manifest, now=FIXED_NOW)
    transition = initial_transition_for(project)

    init_db()
    conn = get_connection()
    try:
        create_project(conn, project, transition)
    finally:
        conn.close()
    return project.project_id, manifest_path.parent, manifest


def _args(project_id, manifest_path, output, out_format="text"):
    return argparse.Namespace(
        project_id=project_id,
        manifest=str(manifest_path),
        output=str(output),
        format=out_format,
    )


def _mock_pipeline_success(monkeypatch):
    from src.core.final_video_assembly import FinalVideoAssemblyResult

    def _fake_assemble(project_id, manifest_path, output_path):
        Path(output_path).write_bytes(b"fake-final-mp4-bytes")
        return FinalVideoAssemblyResult(
            project_id=project_id,
            output_path=Path(output_path),
            scene_count=1,
            measured_duration_seconds=2.5,
            artifact_id="render-final",
        )

    monkeypatch.setattr(f"{GENERATION_MODULE}.assemble_final_video", _fake_assemble)


# ---------------------------------------------------------------------
# parser registration
# ---------------------------------------------------------------------


def test_parser_registers_assemble_final_video():
    parser = cli.build_parser()
    args = parser.parse_args(
        ["assemble-final-video", "proj-123", "--manifest", "m.json", "--output", "out.mp4"]
    )
    assert args.func is cli.cmd_assemble_final_video
    assert args.project_id == "proj-123"
    assert args.manifest == "m.json"
    assert args.output == "out.mp4"
    assert args.format == "text"


def test_project_id_is_positional():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["assemble-final-video", "--manifest", "m.json", "--output", "out.mp4"])


def test_manifest_is_required():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["assemble-final-video", "proj-123", "--output", "out.mp4"])


def test_output_is_required():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["assemble-final-video", "proj-123", "--manifest", "m.json"])


def test_existing_commands_remain_registered():
    parser = cli.build_parser()
    assert parser.parse_args(["health"]).func is cli.cmd_health
    assert parser.parse_args(
        ["build-scene-audio", "p", "s", "--manifest", "m", "--output", "o"]
    ).func is cli.cmd_build_scene_audio
    assert parser.parse_args(
        [
            "build-upscaled-ken-burns", "p", "s", "--manifest", "m", "--output", "o",
        ]
    ).func is cli.cmd_build_upscaled_ken_burns


# ---------------------------------------------------------------------
# success
# ---------------------------------------------------------------------


def test_cli_success_text_mode(isolated_db, tmp_path, capsys, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _mock_pipeline_success(monkeypatch)

    manifest_path = project_dir / "manifest.json"
    output_path = tmp_path / "final.mp4"
    rc = cli.cmd_assemble_final_video(_args(project_id, manifest_path, output_path))
    out = capsys.readouterr().out

    assert rc == 0
    assert "assemble-final-video: OK" in out
    assert f"project_id: {project_id}" in out
    assert f"output: {output_path}" in out
    assert "scene_count: 1" in out
    assert "measured_duration_seconds: 2.5" in out
    assert "artifact_id: render-final" in out


def test_cli_success_json_mode(isolated_db, tmp_path, capsys, monkeypatch):
    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    _mock_pipeline_success(monkeypatch)

    manifest_path = project_dir / "manifest.json"
    output_path = tmp_path / "final.mp4"
    rc = cli.cmd_assemble_final_video(_args(project_id, manifest_path, output_path, out_format="json"))
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert rc == 0
    assert payload == {
        "ok": True,
        "project_id": project_id,
        "output": str(output_path),
        "scene_count": 1,
        "measured_duration_seconds": 2.5,
        "artifact_id": "render-final",
    }


def test_cli_calls_production_assembly_entry_point(isolated_db, tmp_path, capsys, monkeypatch):
    from src.core.final_video_assembly import FinalVideoAssemblyResult

    calls: list = []

    def _fake_assemble(project_id, manifest_path, output_path):
        calls.append((project_id, Path(manifest_path), Path(output_path)))
        return FinalVideoAssemblyResult(
            project_id=project_id,
            output_path=Path(output_path),
            scene_count=1,
            measured_duration_seconds=1.0,
            artifact_id="render-final",
        )

    monkeypatch.setattr(f"{GENERATION_MODULE}.assemble_final_video", _fake_assemble)

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    output_path = tmp_path / "final.mp4"

    cli.cmd_assemble_final_video(_args(project_id, manifest_path, output_path))
    capsys.readouterr()

    assert len(calls) == 1
    assert calls[0] == (project_id, manifest_path, output_path)


# ---------------------------------------------------------------------
# failures
# ---------------------------------------------------------------------


def test_cli_domain_failure_written_to_stderr(isolated_db, tmp_path, capsys, monkeypatch):
    from src.core.final_video_assembly import ProjectNotFoundError

    def _fail(project_id, manifest_path, output_path):
        raise ProjectNotFoundError(f"unknown project_id {project_id!r}")

    monkeypatch.setattr(f"{GENERATION_MODULE}.assemble_final_video", _fail)

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    output_path = tmp_path / "final.mp4"

    rc = cli.cmd_assemble_final_video(_args(project_id, manifest_path, output_path))
    err = capsys.readouterr().err

    assert "assemble-final-video: FAILED" in err
    assert f"unknown project_id {project_id!r}" in err


def test_cli_domain_failure_returns_nonzero(isolated_db, tmp_path, capsys, monkeypatch):
    from src.core.final_video_assembly import ProjectNotFoundError

    monkeypatch.setattr(
        f"{GENERATION_MODULE}.assemble_final_video",
        lambda *a, **k: (_ for _ in ()).throw(ProjectNotFoundError("boom")),
    )

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    output_path = tmp_path / "final.mp4"

    rc = cli.cmd_assemble_final_video(_args(project_id, manifest_path, output_path))
    capsys.readouterr()

    assert rc != 0


def test_cli_unexpected_failure_does_not_expose_secrets(isolated_db, tmp_path, capsys, monkeypatch):
    """An UNDOCUMENTED exception type (not FinalVideoAssemblyError) is not
    caught by this command's own try/except — it propagates unchanged,
    matching this codebase's established "an undocumented failure is a
    bug to surface, not an outcome to sanitize" convention. This test
    proves the command's OWN error-formatting path never echoes secret-
    looking content when it does construct a failure message."""
    from src.core.final_video_assembly import FinalVideoAssemblyError

    sentinel = "SENTINEL-NOT-A-SECRET-9f2a"

    def _fail(project_id, manifest_path, output_path):
        raise FinalVideoAssemblyError(sentinel)

    monkeypatch.setattr(f"{GENERATION_MODULE}.assemble_final_video", _fail)

    project_id, project_dir, manifest = _create_registered_project(tmp_path / "projects")
    manifest_path = project_dir / "manifest.json"
    output_path = tmp_path / "final.mp4"

    rc = cli.cmd_assemble_final_video(_args(project_id, manifest_path, output_path))
    err = capsys.readouterr().err

    assert rc != 0
    assert "QWEN_API_KEY" not in err
    assert "GROQ_API_KEY" not in err
    assert sentinel in err


def test_cli_invalid_format_rejects(isolated_db, tmp_path, capsys):
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text("{}", encoding="utf-8")

    rc = cli.cmd_assemble_final_video(_args("proj-x", manifest_path, tmp_path / "out.mp4", out_format="xml"))
    err = capsys.readouterr().err

    assert rc == 1
    assert "invalid --format" in err


# ---------------------------------------------------------------------
# database lifecycle / provider isolation
# ---------------------------------------------------------------------


def test_cli_opens_no_connection_of_its_own(isolated_db, tmp_path, monkeypatch):
    """cmd_assemble_final_video() itself never imports or calls
    get_connection()/get_readonly_connection() — all SQLite access lives
    inside assemble_final_video() (see that module's docstring for why).
    Proven here by inspecting the CLI function's own source for any such
    call, since mocking assemble_final_video() means no real connection
    is ever opened during this test regardless."""
    import inspect

    source = inspect.getsource(cli.cmd_assemble_final_video)
    assert "get_connection" not in source
    assert "get_readonly_connection" not in source


def test_cli_imports_no_provider_module(monkeypatch):
    """No provider module (Groq/TokenRouter/Kokoro/Qwen/Rhubarb/Flow/Veo)
    is imported anywhere in cmd_assemble_final_video()'s own source —
    only src.core.final_video_assembly, which is itself local-only."""
    import inspect

    source = inspect.getsource(cli.cmd_assemble_final_video)
    for forbidden in ("llm_groq", "llm_tokenrouter", "tts_kokoro", "image_qwen", "lipsync_rhubarb"):
        assert forbidden not in source
