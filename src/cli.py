"""CLI entry point — mirrors youtube-intelligence-engine's argparse style.

Phase 1E adds local-only commands (validate-input, create-project,
import-queue, list-projects, status, resume-plan, validate-manifest) on
top of the pre-existing `health`/`init-db` commands, which are unchanged.
None of the new commands call a provider, an LLM, TTS, image generation,
Flow, Veo, a renderer, FFmpeg, or any remote service — they only read
local JSON files and read/write the local SQLite database. Error handling
convention: every new command function catches its own domain errors and
converts them to a short stderr message + non-zero exit code; none of them
let a raw traceback reach the user, and none of them call sys.exit()
themselves (only `main()`'s `raise SystemExit(main())` does that)."""
import argparse
import sqlite3
import sys
from pathlib import Path

from src.database.db import init_db
from src.utils.logging import setup_logging


def _check(logger, label: str, check) -> bool:
    try:
        if check():
            logger.info("%s: OK", label)
            return True
        logger.error("%s: FAILED", label)
        return False
    except Exception as exc:
        logger.error("%s: FAILED (%s)", label, exc)
        return False


def cmd_health(_args: argparse.Namespace) -> int:
    logger = setup_logging()

    from src.providers.llm_groq import GroqProvider
    from src.providers.llm_tokenrouter import TokenRouterProvider
    from src.providers.tts_kokoro import KokoroProvider
    from src.providers.image_qwen import QwenImageProvider
    from src.providers import lipsync_rhubarb, image_upscale
    from src.render import manim_render, ffmpeg_render

    results = [
        _check(logger, "Groq (LLM)", GroqProvider().health_check),
        _check(logger, "TokenRouter (LLM fallback)", TokenRouterProvider().health_check),
        _check(logger, "Kokoro (TTS)", KokoroProvider().health_check),
        _check(logger, "Qwen-Image (backgrounds/character)", QwenImageProvider().health_check),
        _check(logger, "Rhubarb (lip sync)", lipsync_rhubarb.health_check),
        _check(logger, "Real-ESRGAN (upscale)", image_upscale.health_check),
        _check(logger, "Manim (supporting animation)", manim_render.health_check),
        _check(logger, "FFmpeg (render)", ffmpeg_render.health_check),
    ]

    ok = all(results)
    logger.info("Health check %s", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def cmd_init_db(_args: argparse.Namespace) -> int:
    logger = setup_logging()
    init_db()
    logger.info("Database initialized.")
    return 0


# ---------------------------------------------------------------------
# Phase 1E: local, provider-free commands
# ---------------------------------------------------------------------


def cmd_validate_input(args: argparse.Namespace) -> int:
    from src.core.manifest_builder import ManifestValidationError, build_video_manifest
    from src.core.queue_import import InputFileError, load_scene_plan_file, load_story_input_file
    from src.utils.channel_config import ChannelConfigError, get_channel_policy

    try:
        story_input = load_story_input_file(Path(args.story))
        scene_plan = load_scene_plan_file(Path(args.scenes))
        channel_policy = get_channel_policy()
        manifest = build_video_manifest(story_input, scene_plan, channel_policy)
    except (InputFileError, ChannelConfigError, ManifestValidationError) as exc:
        print(f"validate-input: FAILED — {exc}", file=sys.stderr)
        return 1

    print("validate-input: OK (no provider was called; nothing was saved)")
    print(f"  story_id: {manifest.story_input.story_id}")
    print(f"  project_id: {manifest.project_id}")
    print(f"  scene_count: {len(manifest.scene_plan.scenes)}")
    print(f"  target_duration_seconds: {manifest.target_duration_seconds}")
    return 0


def cmd_create_project(args: argparse.Namespace) -> int:
    from src.core.manifest_builder import ManifestValidationError
    from src.core.manifest_store import ManifestStoreError
    from src.core.queue_import import (
        InputFileError,
        ProjectCreationError,
        create_project_from_inputs,
        load_scene_plan_file,
        load_story_input_file,
    )
    from src.database.db import get_connection
    from src.utils.channel_config import ChannelConfigError, get_channel_policy

    output_root = Path(args.output_root)
    try:
        story_input = load_story_input_file(Path(args.story))
        scene_plan = load_scene_plan_file(Path(args.scenes))
        channel_policy = get_channel_policy()
        init_db()
        conn = get_connection()
        try:
            project = create_project_from_inputs(conn, story_input, scene_plan, channel_policy, output_root)
        finally:
            conn.close()
    except (
        InputFileError,
        ChannelConfigError,
        ManifestValidationError,
        ManifestStoreError,
        ProjectCreationError,
        sqlite3.Error,
    ) as exc:
        print(f"create-project: FAILED — {exc}", file=sys.stderr)
        return 1

    print("create-project: OK (no provider was called)")
    print(f"  project_id: {project.project_id}")
    print(f"  manifest_path: {project.manifest_path}")
    print(f"  current_stage: {project.current_stage}")
    return 0


def cmd_import_queue(args: argparse.Namespace) -> int:
    from src.core.queue_import import QueueImportError, import_queue, load_queue_import, prepare_queue_import
    from src.database.db import get_connection
    from src.utils.channel_config import ChannelConfigError, get_channel_policy

    queue_path = Path(args.queue)
    output_root = Path(args.output_root)

    try:
        queue = load_queue_import(queue_path)
        channel_policy = get_channel_policy()
        init_db()
        conn = get_connection()
    except (QueueImportError, ChannelConfigError, sqlite3.Error) as exc:
        print(f"import-queue: FAILED — {exc}", file=sys.stderr)
        return 1

    try:
        if not args.confirm_import:
            try:
                plan = prepare_queue_import(
                    queue, queue_path.resolve().parent, channel_policy, output_root, conn
                )
            except (QueueImportError, sqlite3.Error) as exc:
                print(f"import-queue: FAILED — {exc}", file=sys.stderr)
                return 1
            print("import-queue: DRY RUN — no manifests or database records were written")
            for line in plan.summary_lines():
                print(f"  {line}")
            print("Re-run with --confirm-import to actually create these projects.")
            return 2  # distinct from 1 (validation failure) and 0 (success)

        try:
            result = import_queue(conn, queue, queue_path.resolve().parent, channel_policy, output_root)
        except (QueueImportError, sqlite3.Error) as exc:
            print(f"import-queue: FAILED — {exc}", file=sys.stderr)
            return 1
    finally:
        conn.close()

    print("import-queue: OK (no provider was called)")
    for project in result.created:
        print(f"  created: {project.project_id}")
    for queue_item_id in result.skipped_disabled:
        print(f"  skipped (disabled): {queue_item_id}")
    return 0


def cmd_list_projects(_args: argparse.Namespace) -> int:
    from src.database.db import get_connection
    from src.database.project_repository import list_projects

    try:
        init_db()
        conn = get_connection()
        try:
            projects = list_projects(conn)
        finally:
            conn.close()
    except sqlite3.Error as exc:
        print(f"list-projects: FAILED — {exc}", file=sys.stderr)
        return 1

    if not projects:
        print("No projects found.")
        return 0

    for project in projects:
        print(
            f"{project.project_id}  stage={project.current_stage}  "
            f"last_success={project.last_successful_stage}  retries={project.retry_count}  "
            f"updated_at={project.updated_at.isoformat()}"
        )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from src.database.db import get_connection
    from src.database.project_repository import get_project, list_project_transitions

    try:
        init_db()
        conn = get_connection()
        try:
            project = get_project(conn, args.project_id)
            if project is None:
                print(f"status: FAILED — no project found with project_id {args.project_id!r}", file=sys.stderr)
                return 1
            transitions = list_project_transitions(conn, args.project_id)
        finally:
            conn.close()
    except sqlite3.Error as exc:
        print(f"status: FAILED — {exc}", file=sys.stderr)
        return 1

    print(f"project_id: {project.project_id}")
    print(f"current_stage: {project.current_stage}")
    print(f"last_successful_stage: {project.last_successful_stage}")
    print(f"failed_stage: {project.failed_stage}")
    print(f"failure_message: {project.failure_message}")
    print(f"lifecycle_version: {project.lifecycle_version}")
    print(f"retry_count: {project.retry_count}")
    print(f"execution_status: {project.execution_status}")
    print(f"manifest_path: {project.manifest_path}")
    print(f"created_at: {project.created_at.isoformat()}")
    print(f"updated_at: {project.updated_at.isoformat()}")
    print(f"completed_at: {project.completed_at.isoformat() if project.completed_at else None}")
    print(f"archived_at: {project.archived_at.isoformat() if project.archived_at else None}")
    print("transitions:")
    for transition in transitions:
        print(
            f"  [v{transition.lifecycle_version}] {transition.from_stage} -> {transition.to_stage} "
            f"at {transition.occurred_at.isoformat()} (retry={transition.is_retry}) "
            f"reason={transition.reason!r}"
        )
    return 0


def cmd_resume_plan(args: argparse.Namespace) -> int:
    from src.core.project_state_machine import build_resume_plan
    from src.database.db import get_connection
    from src.database.project_repository import get_project

    try:
        init_db()
        conn = get_connection()
        try:
            project = get_project(conn, args.project_id)
        finally:
            conn.close()
    except sqlite3.Error as exc:
        print(f"resume-plan: FAILED — {exc}", file=sys.stderr)
        return 1

    if project is None:
        print(f"resume-plan: FAILED — no project found with project_id {args.project_id!r}", file=sys.stderr)
        return 1

    plan = build_resume_plan(project)
    print(f"action: {plan.action}")
    print(f"next_stage: {plan.next_stage}")
    print(f"explanation: {plan.explanation}")
    print(f"last_successful_stage: {plan.last_successful_stage}")
    return 0


def cmd_validate_manifest(args: argparse.Namespace) -> int:
    from src.core.manifest_store import ManifestStoreError, load_manifest

    try:
        manifest = load_manifest(Path(args.path))
    except ManifestStoreError as exc:
        print(f"validate-manifest: FAILED — {exc}", file=sys.stderr)
        return 1

    print("validate-manifest: OK (only validation was performed; nothing was modified)")
    print(f"  project_id: {manifest.project_id}")
    print(f"  source_fingerprint: {manifest.source_fingerprint}")
    print(f"  scene_count: {len(manifest.scene_plan.scenes)}")
    print(f"  lifecycle_state: {manifest.lifecycle_state}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser without running anything — split out from
    main() so tests can inspect subcommand registration (e.g. that `health`
    is still wired to cmd_health) without ever invoking a handler."""
    parser = argparse.ArgumentParser(prog="video-factory")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("health", help="Check all providers are reachable/working").set_defaults(func=cmd_health)
    sub.add_parser("init-db", help="Create the jobs/projects database").set_defaults(func=cmd_init_db)

    validate_input = sub.add_parser(
        "validate-input",
        help="Validate a local StoryInput + ScenePlan pair; writes nothing (no provider is called)",
    )
    validate_input.add_argument("--story", required=True, help="Path to a StoryInput JSON file")
    validate_input.add_argument("--scenes", required=True, help="Path to a ScenePlan JSON file")
    validate_input.set_defaults(func=cmd_validate_input)

    create_project = sub.add_parser(
        "create-project",
        help="Create one project from a local approved StoryInput + ScenePlan (no provider is called)",
    )
    create_project.add_argument("--story", required=True, help="Path to a StoryInput JSON file")
    create_project.add_argument("--scenes", required=True, help="Path to a ScenePlan JSON file")
    create_project.add_argument(
        "--output-root", required=True, help="Directory the new project's manifest is saved under"
    )
    create_project.set_defaults(func=cmd_create_project)

    import_queue = sub.add_parser(
        "import-queue",
        help="Import a local queue of story/scene file pairs as new projects (no provider is called)",
    )
    import_queue.add_argument("--queue", required=True, help="Path to a QueueImport JSON file")
    import_queue.add_argument(
        "--output-root", required=True, help="Directory each new project's manifest is saved under"
    )
    import_queue.add_argument(
        "--confirm-import",
        action="store_true",
        help="Actually write manifests and database records; without this flag, only a dry-run "
        "summary is printed and nothing is written (exit code 2)",
    )
    import_queue.set_defaults(func=cmd_import_queue)

    sub.add_parser("list-projects", help="List all local projects").set_defaults(func=cmd_list_projects)

    status = sub.add_parser("status", help="Show one project's status and transition audit trail")
    status.add_argument("project_id")
    status.set_defaults(func=cmd_status)

    resume_plan = sub.add_parser(
        "resume-plan", help="Show (without executing) what a future orchestrator would do next for a project"
    )
    resume_plan.add_argument("project_id")
    resume_plan.set_defaults(func=cmd_resume_plan)

    validate_manifest = sub.add_parser(
        "validate-manifest", help="Validate a previously saved manifest.json file; writes nothing"
    )
    validate_manifest.add_argument("path")
    validate_manifest.set_defaults(func=cmd_validate_manifest)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
