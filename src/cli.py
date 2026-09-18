"""CLI entry point — mirrors youtube-intelligence-engine's argparse style.

Phase 1E adds local-only commands (validate-input, create-project,
import-queue, list-projects, status, resume-plan, validate-manifest) on
top of the pre-existing `health`/`init-db` commands, which are unchanged.
Phase 2A adds the read-only `dry-run` command. Phase 2B adds the read-only
`verify-artifacts` command (src/core/artifact_verifier.py) — same
strictly-read-only convention as `dry-run`: it uses
src/database/db.py's get_readonly_connection(), never calls init_db(),
and never transitions a project's stage. Phase 2C adds exactly one new
write-capable command, `verify-and-advance`
(src/core/verified_transition_service.py) — it verifies every required
local artifact first and performs at most one guarded transition only if
verification passes. Unlike `create-project`/`import-queue`, it never
calls init_db(): it opens the existing database via
src/database/db.py's get_existing_connection() (mode=rw, same
never-create-anything contract as get_readonly_connection(), just
writable), so a missing data directory or database file fails cleanly
instead of being created. The only write it can ever perform is
save_transition()'s single guarded UPDATE + INSERT, reached only after
every precondition and artifact-verification check has already passed;
it never creates a project or registers an artifact. Phase 2D adds exactly
one new write-capable command, `register-audio-artifact`
(src/core/audio_artifact_registrar.py) — registration-first: it registers
an already-produced local audio file as one scene's canonical "audio"
artifact (fixed identity, no alternate takes), the one remaining local
step verify-and-advance needs to advance a project into "audio_ready". It
also never calls init_db() (same get_existing_connection() contract as
verify-and-advance), never mutates the manifest, and never transitions a
project's lifecycle stage itself. Phase 2E adds the analogous
`register-visual-artifact` command (src/core/visual_artifact_registrar.py)
for one PNG "visual" artifact per scene — same fixed-identity,
registration-first, never-init_db(), never-mutates-the-manifest,
stage-agnostic design as `register-audio-artifact`, validated with Pillow
(already a pinned dependency) instead of ffprobe; PNG only, no Qwen-size
restrictions. Phase 2F adds `register-animation-artifact`
(src/core/animation_artifact_registrar.py) for one MP4 "animation"
artifact per scene — same fixed-identity/registration-first/stage-agnostic
design again, validated with ffprobe like `register-audio-artifact` (real
duration AND at least one actual video stream — an audio-only file is
rejected even though it has a valid duration). Phase 2G adds
`register-render-artifact` (src/core/render_artifact_registrar.py) for
THE project's one MP4 "render" artifact — project-level, not per-scene
(no --scene-id at all: src/models/artifact.py's PROJECT_LEVEL_ARTIFACT_KINDS
forbids a scene_id for kind="render"), same ffprobe duration+video-stream
validation as `register-animation-artifact`, never validated against
manifest.target_duration_seconds — only the measured value is ever
recorded. Phase 2H adds `register-qc-report-artifact`
(src/core/qc_report_artifact_registrar.py) for THE project's one JSON
"qc_report" artifact — project-level like `register-render-artifact`,
requiring only a well-formed JSON object with a boolean `passed` field
(either value registers successfully; a false report is retained audit
data, not an error). Phase 2H also makes one targeted addition to
src/core/verified_transition_service.py: only for to_stage=="qc_passed",
after existing structural verification of both the render and qc_report
artifacts already passed, it additionally re-reads the verified qc_report
FILE (never the mutable ArtifactRecord.metadata) and requires its
`passed` to be strictly True before the transition is allowed — every
other target stage is unaffected. Phase 2I adds `advance-project-stage`
(src/core/stage_advance_service.py) — the non-verification counterpart to
`verify-and-advance`: it performs exactly one guarded transition into one
of the six forward stages that claim NO completed/measured work
(audio_pending, visuals_pending, animation_pending, render_pending,
qc_pending, ready_for_manual_publish), always with verified=False. Every
other forward target — audio_ready, visuals_ready, animation_ready,
rendered, qc_passed, and completed — stays exclusively behind
`verify-and-advance`; `advance-project-stage` rejects all of them (plus
`archived`/`failed`, unclaimed by any command) before ever calling
transition_project(). Same get_existing_connection()/single-guarded-write
contract as `verify-and-advance`. `generate-scenes`
(src/core/scene_generator.py) completes docs/spec-v4/IMPLEMENTATION-PLAN.md's
Phase 1C: given a local StoryInput file, it calls GroqProvider (primary)
then TokenRouterProvider (fallback) — up to two attempts each, four total —
to produce a fully validated ScenePlan. Every LLM-supplied field is
content only (narration_text, scene_type, narrative_beat, visual_brief,
motion_mode, with "manual_flow" excluded); scene_id/sequence/
approval_state ("draft", never auto-approved)/artifacts/role_outfit_id/
text_overlays/sfx_refs/flow_task_id are always assigned programmatically,
never trusted from the model. It never creates a project, manifest,
artifact, or stage transition — with `--output` it saves the ScenePlan as
JSON (src/core/scene_plan_store.py, no database access); without it, only
a summary is printed. `finalize-scene-timing`
(src/core/scene_timing_finalizer.py) completes the other half of
documented Phase 1C: reading a project's manifest and its already-registered
"audio" ArtifactRecords (Phase 2D) strictly read-only
(get_readonly_connection(), never init_db()), it populates every scene's
measured_audio_duration_seconds from that already-ffprobe-measured
metadata — all-or-nothing across every scene, no fallback estimate, and
never touching VideoManifest.measured_audio_duration_seconds (the
whole-video aggregate, deliberately left untouched). project_id and
source_fingerprint are preserved exactly (model_copy(update=...) only,
never build_video_manifest()/_compute_fingerprint()). The result is saved
to a required `--output` path (src/core/manifest_store.save_manifest(),
atomic write) — `--output` is explicitly rejected (before finalize_scene_timing()
or save_manifest() is ever called) if it resolves to the project's own
canonical manifest_path (a normalized, case-folded-on-Windows
Path.resolve() comparison that does not require either path to already
exist), so the project's original manifest can never be silently
overwritten in place. No database write, no provider, no ffprobe call. `preview-scene-image-prompt` (src/core/scene_image_preview.py) is a
read-only preview for a not-yet-implemented future `build-scene-image`
command: given a project/scene and a local --reference-image/--output path
pair, it loads the scene's visual_brief from the manifest (same
get_readonly_connection()/identity-check contract as every other
generation-only command), applies the already-shipped, pure
src/core/scene_image_prompt.build_scene_image_prompt(), and prints the
exact final prompt a future generation call would use. It never opens or
decodes --reference-image, never creates --output, never calls
image_qwen.py or any other provider, and never writes SQLite or the
manifest. `build-scene-image` (src/core/scene_image_generation.py) is that
future command, now implemented: generation-only, it calls
QwenImageProvider.generate_with_reference_and_download() — the first real,
paid, network provider call reachable from this CLI — but only after every
local check (project/manifest identity, scene lookup, --reference-image,
--output) has passed AND require_paid_approval("qwen-image", proposals_path,
is_paid=True) has succeeded; Qwen-Image is fixed as a paid service, never a
CLI-settable option, so no invocation can declare its own way past the
guard. It writes only the --output PNG (verified with Pillow afterward) —
no SQLite write, no artifact registration, no manifest change; register
the result separately with the existing, unmodified `register-visual-artifact`.
`build-scene-audio` (src/core/scene_audio_generation.py) is analogous for
narration audio: generation-only, it calls KokoroProvider.synthesize() —
self-hosted, local TTS, no network call — after every local check
(project/manifest identity, scene lookup, --output) has passed AND
require_paid_approval("kokoro", proposals_path, is_paid=False) has
succeeded; kept even though Kokoro is genuinely free, for the same
consistency reason every other real provider call site in this CLI is
wired through the guard. It writes only the --output WAV (verified with
the existing ffprobe-based duration helper afterward) — no SQLite write,
no artifact registration, no manifest change; register the result
separately with the existing, unmodified `register-audio-artifact`.
None of the other
commands call a provider, an LLM, TTS, image
generation, Flow, Veo, YouTube, or any remote service — `register-audio-artifact`,
`register-animation-artifact`, and `register-render-artifact` use a
renderer module (src/render/ffmpeg_render's local ffprobe wrapper, to
measure a source file's real duration) and `register-visual-artifact` uses
Pillow (to decode-validate a source image); every other command only reads local
JSON files and reads/writes the local SQLite database.
Error handling convention: every new command function catches its own
domain errors and converts them to a short stderr message + non-zero exit
code; none of them let a raw traceback reach the user, and none of them
call sys.exit() themselves (only `main()`'s `raise SystemExit(main())`
does that)."""
import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import get_args

from src.core.stage_advance_service import NON_VERIFICATION_TARGET_STAGES
from src.core.verified_transition_service import SUPPORTED_TARGET_STAGES
from src.database.db import init_db
from src.models.enums import ArtifactKind
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
    """Phase 1E: Groq/TokenRouter/Qwen-Image's health_check() each make a
    real network request (Kokoro/Rhubarb/Real-ESRGAN/Manim/FFmpeg do not —
    self-hosted/local only, no guard needed), so each is preceded by its
    own require_paid_approval(..., is_paid=False) call — same canonical
    names and is_paid=False convention src/core/scene_generator.py already
    uses. Each guard call is a standalone statement, deliberately outside
    _check()'s try/except: _check() catches plain Exception, which would
    otherwise silently fold a PaidApprovalRequiredError into an ordinary
    "FAILED" line for just that one provider. Left unguarded, it propagates
    out of cmd_health() immediately, before that provider is even
    constructed and before any later provider in this list is checked."""
    logger = setup_logging()

    from src.core.cost_guard import require_paid_approval
    from src.providers.llm_groq import GroqProvider
    from src.providers.llm_tokenrouter import TokenRouterProvider
    from src.providers.tts_kokoro import KokoroProvider
    from src.providers.image_qwen import QwenImageProvider
    from src.providers import lipsync_rhubarb, image_upscale
    from src.render import manim_render, ffmpeg_render
    from src.utils.config import get_settings

    # Placeholder only, like scene_generator.py's — never opened/created
    # while is_paid=False (require_paid_approval returns before touching it).
    proposals_path = get_settings().data_dir / "paid_proposals.json"

    results = []

    require_paid_approval("groq", proposals_path, is_paid=False)
    results.append(_check(logger, "Groq (LLM)", GroqProvider().health_check))

    require_paid_approval("tokenrouter", proposals_path, is_paid=False)
    results.append(_check(logger, "TokenRouter (LLM fallback)", TokenRouterProvider().health_check))

    results.append(_check(logger, "Kokoro (TTS)", KokoroProvider().health_check))

    require_paid_approval("qwen-image", proposals_path, is_paid=False)
    results.append(_check(logger, "Qwen-Image (backgrounds/character)", QwenImageProvider().health_check))

    results.append(_check(logger, "Rhubarb (lip sync)", lipsync_rhubarb.health_check))
    results.append(_check(logger, "Real-ESRGAN (upscale)", image_upscale.health_check))
    results.append(_check(logger, "Manim (supporting animation)", manim_render.health_check))
    results.append(_check(logger, "FFmpeg (render)", ffmpeg_render.health_check))

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


def cmd_generate_scenes(args: argparse.Namespace) -> int:
    from src.core.queue_import import InputFileError, load_story_input_file
    from src.core.scene_generator import SceneGenerationError, generate_scene_plan
    from src.utils.channel_config import ChannelConfigError, get_channel_policy

    try:
        story_input = load_story_input_file(Path(args.story))
        channel_policy = get_channel_policy()
        scene_plan = generate_scene_plan(story_input, channel_policy)
    except (InputFileError, ChannelConfigError, SceneGenerationError) as exc:
        print(f"generate-scenes: FAILED — {exc}", file=sys.stderr)
        return 1

    if args.output:
        from src.core.scene_plan_store import ScenePlanStoreError, save_scene_plan

        try:
            save_scene_plan(scene_plan, Path(args.output))
        except ScenePlanStoreError as exc:
            print(f"generate-scenes: FAILED — {exc}", file=sys.stderr)
            return 1
        print(
            "generate-scenes: OK (no project, manifest, artifact, or stage transition was created)"
        )
        print(f"  scene_count: {len(scene_plan.scenes)}")
        print(f"  output: {args.output}")
    else:
        print(
            "generate-scenes: OK (nothing was saved; no project, manifest, artifact, or stage "
            "transition was created)"
        )
        print(f"  scene_count: {len(scene_plan.scenes)}")
        for scene in scene_plan.scenes:
            print(f"  - {scene.scene_id} [{scene.scene_type}/{scene.motion_mode}]: {scene.narration_text}")
    return 0


def cmd_finalize_scene_timing(args: argparse.Namespace) -> int:
    from src.core.manifest_store import ManifestStoreError, load_manifest, save_manifest
    from src.core.scene_timing_finalizer import TimingFinalizationError, finalize_scene_timing
    from src.database.artifact_repository import list_artifacts_by_project
    from src.database.db import get_readonly_connection
    from src.database.project_repository import get_project

    # Strictly read-only: never init_db()/create the data directory, the
    # database file, or its journal/WAL/SHM files — same contract as
    # dry-run's get_readonly_connection(). This command never writes to
    # SQLite under any circumstance.
    try:
        conn = get_readonly_connection()
    except sqlite3.Error as exc:
        print(f"finalize-scene-timing: FAILED — no local project database found: {exc}", file=sys.stderr)
        return 1

    try:
        project = get_project(conn, args.project_id)
        if project is None:
            print(
                f"finalize-scene-timing: FAILED — no project found with project_id {args.project_id!r}",
                file=sys.stderr,
            )
            return 1
        audio_artifacts = list_artifacts_by_project(conn, args.project_id, kind="audio")
    except sqlite3.Error as exc:
        print(f"finalize-scene-timing: FAILED — {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()

    try:
        manifest = load_manifest(Path(project.manifest_path))
    except ManifestStoreError as exc:
        print(f"finalize-scene-timing: FAILED — {exc}", file=sys.stderr)
        return 1

    if manifest.project_id != args.project_id:
        print(
            f"finalize-scene-timing: FAILED — manifest project_id {manifest.project_id!r} "
            f"does not match project {args.project_id!r}",
            file=sys.stderr,
        )
        return 1
    if manifest.source_fingerprint != project.manifest_fingerprint:
        print(
            "finalize-scene-timing: FAILED — manifest source_fingerprint does not match the "
            "project registry's recorded fingerprint",
            file=sys.stderr,
        )
        return 1

    # --output must be a separate, explicit destination — never the
    # project's own canonical manifest_path. Path.resolve() (default
    # strict=False) safely normalizes both sides without requiring either
    # to exist (a fresh --output path, or one whose parent directory
    # doesn't exist yet, is the normal case, not an error); os.path.normcase()
    # additionally folds case on case-insensitive filesystems (Windows),
    # a no-op elsewhere. A symlink/reparse-point that could only be
    # resolved by something not yet on disk (e.g. --output's own
    # not-yet-created parent turning out to be a symlink later) is not,
    # and cannot be, detected here — see this command's own review notes.
    output_resolved = os.path.normcase(str(Path(args.output).resolve()))
    manifest_path_resolved = os.path.normcase(str(Path(project.manifest_path).resolve()))
    if output_resolved == manifest_path_resolved:
        print(
            "finalize-scene-timing: FAILED — --output must not be the project's own canonical "
            "manifest path; choose a separate destination for the enriched copy",
            file=sys.stderr,
        )
        return 1

    try:
        enriched = finalize_scene_timing(manifest, audio_artifacts)
    except TimingFinalizationError as exc:
        print(f"finalize-scene-timing: FAILED — {exc}", file=sys.stderr)
        return 1

    try:
        save_manifest(enriched, Path(args.output))
    except ManifestStoreError as exc:
        print(f"finalize-scene-timing: FAILED — {exc}", file=sys.stderr)
        return 1

    print(
        "finalize-scene-timing: OK (per-scene measured_audio_duration_seconds populated from "
        "already-registered audio artifacts; manifest-level measured_audio_duration_seconds left "
        "unchanged; no database write, no artifact registration, no stage transition)"
    )
    print(f"  project_id: {enriched.project_id}")
    print(f"  scene_count: {len(enriched.scene_plan.scenes)}")
    print(f"  output: {args.output}")
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


def _render_dry_run_text(report) -> str:
    lines = [
        "dry-run: OK (read-only planning/reporting only)",
        f"project_id: {report.project_id}",
        f"manifest_path: {report.manifest_path}",
        f"current_stage: {report.current_stage}",
        f"last_successful_stage: {report.last_successful_stage}",
        f"lifecycle_version: {report.lifecycle_version}",
        f"transition_count: {report.transition_count}",
        f"next_action: {report.execution_plan.next_action}",
        f"next_stage: {report.execution_plan.next_stage}",
        f"explanation: {report.execution_plan.explanation}",
    ]
    lines.append("requirements:")
    for item in report.execution_plan.requirements:
        lines.append(f"  - {item}")
    lines.append("blocks:")
    for item in report.execution_plan.blocks:
        lines.append(f"  - {item}")
    lines.append("warnings:")
    for item in report.execution_plan.warnings:
        lines.append(f"  - {item}")
    lines.append("policy_decisions:")
    for decision in report.policy_decisions:
        lines.append(f"  - {decision.key}: {decision.status} ({decision.detail})")
    lines.append("planned_steps:")
    for step in report.execution_plan.planned_steps:
        lines.append(
            f"  - #{step.order} {step.stage} action={step.action} "
            f"verification_required={step.verification_required}"
        )
    lines.append("non_invoked_integrations:")
    for item in report.non_invoked_integrations:
        lines.append(f"  - {item}")
    return "\n".join(lines)


def cmd_dry_run(args: argparse.Namespace) -> int:
    from src.core.dry_run_orchestrator import DryRunOrchestratorError, build_dry_run_report
    from src.core.manifest_store import ManifestStoreError, load_manifest
    from src.database.db import get_readonly_connection
    from src.database.project_repository import get_project, list_project_transitions
    from src.utils.channel_config import ChannelConfigError, get_channel_policy

    out_format = args.format
    if out_format not in {"text", "json"}:
        print(f"dry-run: FAILED — invalid --format {out_format!r} (expected 'text' or 'json')", file=sys.stderr)
        return 1

    # Strictly read-only: never init_db()/create the data directory, the
    # database file, or its journal/WAL/SHM files — a missing or unreadable
    # database is reported as a clean dry-run error below, not created.
    try:
        conn = get_readonly_connection()
        try:
            project = get_project(conn, args.project_id)
            if project is None:
                print(f"dry-run: FAILED — no project found with project_id {args.project_id!r}", file=sys.stderr)
                return 1
            transitions = list_project_transitions(conn, args.project_id)
        finally:
            conn.close()
    except sqlite3.Error as exc:
        print(f"dry-run: FAILED — could not read project registry: {exc}", file=sys.stderr)
        return 1

    try:
        manifest = load_manifest(Path(project.manifest_path))
    except ManifestStoreError as exc:
        print(f"dry-run: FAILED — {exc}", file=sys.stderr)
        return 1

    try:
        channel_policy = get_channel_policy()
        report = build_dry_run_report(project, transitions, manifest, channel_policy)
    except (ChannelConfigError, DryRunOrchestratorError) as exc:
        print(f"dry-run: FAILED — {exc}", file=sys.stderr)
        return 1

    if out_format == "json":
        print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True, ensure_ascii=True))
    else:
        print(_render_dry_run_text(report))
    return 0


# Derived from the canonical src.models.enums.ArtifactKind Literal — not a
# second, independently-maintained list of kinds. A new kind added there
# is automatically a valid --kind choice here, with no second edit needed.
_ARTIFACT_KINDS = get_args(ArtifactKind)


def _render_verify_artifacts_text(project_id: str, kind: str | None, results) -> str:
    lines = [
        "verify-artifacts: read-only verification only "
        "(no files, database rows, or project state were changed)",
        f"project_id: {project_id}",
        f"kind_filter: {kind}",
        f"artifact_count: {len(results)}",
    ]
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        lines.append(
            f"  [{status}] {result.artifact_id} kind={result.kind} "
            f"scene_id={result.scene_id} path={result.relative_path}"
        )
        for reason in result.reasons:
            lines.append(f"      - {reason}")
    passed_count = sum(1 for r in results if r.passed)
    lines.append(f"passed: {passed_count}/{len(results)}")
    return "\n".join(lines)


def cmd_verify_artifacts(args: argparse.Namespace) -> int:
    from src.core.artifact_verifier import verify_artifacts
    from src.core.manifest_store import ManifestStoreError, load_manifest
    from src.database.artifact_repository import list_artifacts_by_project
    from src.database.db import get_readonly_connection
    from src.database.project_repository import get_project

    out_format = args.format
    if out_format not in {"text", "json"}:
        print(
            f"verify-artifacts: FAILED — invalid --format {out_format!r} (expected 'text' or 'json')",
            file=sys.stderr,
        )
        return 1

    # Strictly read-only, same convention as cmd_dry_run: never
    # init_db()/create the data directory, the database file, or its
    # journal/WAL/SHM files — a missing or unreadable database is reported
    # as a clean error below, not created.
    try:
        conn = get_readonly_connection()
        try:
            project = get_project(conn, args.project_id)
            if project is None:
                print(
                    f"verify-artifacts: FAILED — no project found with project_id {args.project_id!r}",
                    file=sys.stderr,
                )
                return 1
            artifacts = list_artifacts_by_project(conn, args.project_id, kind=args.kind)
        finally:
            conn.close()
    except sqlite3.Error as exc:
        print(f"verify-artifacts: FAILED — could not read artifact registry: {exc}", file=sys.stderr)
        return 1

    try:
        manifest = load_manifest(Path(project.manifest_path))
    except ManifestStoreError as exc:
        print(f"verify-artifacts: FAILED — {exc}", file=sys.stderr)
        return 1

    # The project directory is the directory containing manifest.json —
    # the same layout src/core/queue_import.py's create_project_from_inputs
    # writes (<output_root>/<project_id>/manifest.json) — and every
    # artifact's relative_path is verified as staying safely under it.
    project_dir = Path(project.manifest_path).resolve().parent
    results = verify_artifacts(project_dir, manifest, artifacts)

    if not results:
        kind_note = f" of kind {args.kind!r}" if args.kind else ""
        print(
            f"verify-artifacts: FAILED — no artifacts registered{kind_note} for project "
            f"{args.project_id!r}",
            file=sys.stderr,
        )
        return 1

    ok = all(result.passed for result in results)

    if out_format == "json":
        payload = {
            "read_only": True,
            "project_id": project.project_id,
            "kind_filter": args.kind,
            "artifact_count": len(results),
            "passed": ok,
            "results": [result.model_dump(mode="json") for result in results],
        }
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
    else:
        print(_render_verify_artifacts_text(project.project_id, args.kind, results))

    return 0 if ok else 1


def _render_verify_and_advance_text(result) -> str:
    lines = [
        "verify-and-advance: "
        + ("OK" if result.approved and result.db_committed else "FAILED")
        + " (writes to SQLite only on success; no artifact file, manifest, or queue file is ever touched)",
        f"project_id: {result.project_id}",
        f"from_stage: {result.from_stage}",
        f"to_stage: {result.to_stage}",
        f"approved: {result.approved}",
        f"db_committed: {result.db_committed}",
        f"lifecycle_version_before: {result.lifecycle_version_before}",
        f"lifecycle_version_after: {result.lifecycle_version_after}",
    ]
    lines.append("required_artifacts:")
    for req in result.required_artifacts:
        lines.append(f"  - kind={req.kind} scope={req.scope} scene_id={req.scene_id}")
    lines.append("artifact_verification_results:")
    for artifact_result in result.artifact_verification_results:
        status = "PASS" if artifact_result.passed else "FAIL"
        lines.append(
            f"  [{status}] {artifact_result.artifact_id} kind={artifact_result.kind} "
            f"scene_id={artifact_result.scene_id}"
        )
        for reason in artifact_result.reasons:
            lines.append(f"      - {reason}")
    lines.append("reasons:")
    for reason in result.reasons:
        lines.append(f"  - {reason}")
    return "\n".join(lines)


def cmd_verify_and_advance(args: argparse.Namespace) -> int:
    from src.core.manifest_store import ManifestStoreError, load_manifest
    from src.core.verified_transition_service import VerifiedTransitionServiceError, verify_and_advance
    from src.database.artifact_repository import list_artifacts_by_project
    from src.database.db import get_existing_connection
    from src.database.project_repository import get_project

    out_format = args.format
    if out_format not in {"text", "json"}:
        print(
            f"verify-and-advance: FAILED — invalid --format {out_format!r} (expected 'text' or 'json')",
            file=sys.stderr,
        )
        return 1

    if not args.reason or not args.reason.strip():
        print("verify-and-advance: FAILED — --reason is required and must not be empty", file=sys.stderr)
        return 1

    # Never calls init_db(): a missing data directory or database file
    # must fail cleanly here, not be created — see get_existing_connection()'s
    # docstring. Every precondition below (project exists, manifest loads,
    # artifacts verify, transition is legal) is checked before this
    # command can perform its one write.
    try:
        conn = get_existing_connection()
    except sqlite3.Error as exc:
        print(f"verify-and-advance: FAILED — no local project database found: {exc}", file=sys.stderr)
        return 1

    try:
        project = get_project(conn, args.project_id)
        if project is None:
            print(
                f"verify-and-advance: FAILED — no project found with project_id {args.project_id!r}",
                file=sys.stderr,
            )
            return 1

        try:
            manifest = load_manifest(Path(project.manifest_path))
        except ManifestStoreError as exc:
            print(f"verify-and-advance: FAILED — {exc}", file=sys.stderr)
            return 1

        artifacts = list_artifacts_by_project(conn, args.project_id)

        try:
            result = verify_and_advance(conn, project, manifest, artifacts, args.to, args.reason)
        except VerifiedTransitionServiceError as exc:
            print(f"verify-and-advance: FAILED — {exc}", file=sys.stderr)
            return 1
    except sqlite3.Error as exc:
        print(f"verify-and-advance: FAILED — {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()

    ok = result.approved and result.db_committed

    if out_format == "json":
        print(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True, ensure_ascii=True))
    else:
        print(_render_verify_and_advance_text(result))

    return 0 if ok else 1


def _render_register_audio_artifact_text(result) -> str:
    status = "OK" if result.ok else "FAILED"
    idempotent_note = " (idempotent no-op — nothing was written)" if result.idempotent else ""
    lines = [
        f"register-audio-artifact: {status}{idempotent_note} "
        "(writes exactly one artifact row on a fresh registration; the manifest and "
        "project lifecycle stage are never touched)",
        f"project_id: {result.project_id}",
        f"scene_id: {result.scene_id}",
        f"artifact_id: {result.artifact_id}",
        f"relative_path: {result.relative_path}",
        f"copied: {result.copied}",
        f"duration_seconds: {result.duration_seconds}",
    ]
    for reason in result.reasons:
        lines.append(f"  - {reason}")
    return "\n".join(lines)


def cmd_register_audio_artifact(args: argparse.Namespace) -> int:
    from src.core.audio_artifact_registrar import AudioArtifactRegistrationError, register_audio_artifact
    from src.core.manifest_store import ManifestStoreError, load_manifest
    from src.database.db import get_existing_connection
    from src.database.project_repository import get_project

    out_format = args.format
    if out_format not in {"text", "json"}:
        print(
            f"register-audio-artifact: FAILED — invalid --format {out_format!r} (expected 'text' or 'json')",
            file=sys.stderr,
        )
        return 1

    # Never calls init_db(): a missing data directory or database file
    # must fail cleanly here, not be created — same contract as
    # verify-and-advance's get_existing_connection().
    try:
        conn = get_existing_connection()
    except sqlite3.Error as exc:
        print(f"register-audio-artifact: FAILED — no local project database found: {exc}", file=sys.stderr)
        return 1

    try:
        project = get_project(conn, args.project_id)
        if project is None:
            print(
                f"register-audio-artifact: FAILED — no project found with project_id {args.project_id!r}",
                file=sys.stderr,
            )
            return 1

        try:
            manifest = load_manifest(Path(project.manifest_path))
        except ManifestStoreError as exc:
            print(f"register-audio-artifact: FAILED — {exc}", file=sys.stderr)
            return 1

        try:
            result = register_audio_artifact(conn, project, manifest, args.scene_id, Path(args.file))
        except AudioArtifactRegistrationError as exc:
            print(f"register-audio-artifact: FAILED — {exc}", file=sys.stderr)
            return 1
    except sqlite3.Error as exc:
        print(f"register-audio-artifact: FAILED — {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()

    if out_format == "json":
        payload = {
            "project_id": result.project_id,
            "scene_id": result.scene_id,
            "artifact_id": result.artifact_id,
            "relative_path": result.relative_path,
            "ok": result.ok,
            "idempotent": result.idempotent,
            "copied": result.copied,
            "duration_seconds": result.duration_seconds,
            "reasons": list(result.reasons),
            "artifact": result.artifact.model_dump(mode="json") if result.artifact is not None else None,
        }
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
    else:
        print(_render_register_audio_artifact_text(result))

    return 0 if result.ok else 1


def _render_register_visual_artifact_text(result) -> str:
    status = "OK" if result.ok else "FAILED"
    idempotent_note = " (idempotent no-op — nothing was written)" if result.idempotent else ""
    lines = [
        f"register-visual-artifact: {status}{idempotent_note} "
        "(writes exactly one artifact row on a fresh registration; the manifest and "
        "project lifecycle stage are never touched)",
        f"project_id: {result.project_id}",
        f"scene_id: {result.scene_id}",
        f"artifact_id: {result.artifact_id}",
        f"relative_path: {result.relative_path}",
        f"copied: {result.copied}",
        f"width: {result.width}",
        f"height: {result.height}",
        f"format: {result.image_format}",
    ]
    for reason in result.reasons:
        lines.append(f"  - {reason}")
    return "\n".join(lines)


def cmd_register_visual_artifact(args: argparse.Namespace) -> int:
    from src.core.manifest_store import ManifestStoreError, load_manifest
    from src.core.visual_artifact_registrar import VisualArtifactRegistrationError, register_visual_artifact
    from src.database.db import get_existing_connection
    from src.database.project_repository import get_project

    out_format = args.format
    if out_format not in {"text", "json"}:
        print(
            f"register-visual-artifact: FAILED — invalid --format {out_format!r} (expected 'text' or 'json')",
            file=sys.stderr,
        )
        return 1

    # Never calls init_db(): a missing data directory or database file
    # must fail cleanly here, not be created — same contract as
    # verify-and-advance's/register-audio-artifact's get_existing_connection().
    try:
        conn = get_existing_connection()
    except sqlite3.Error as exc:
        print(f"register-visual-artifact: FAILED — no local project database found: {exc}", file=sys.stderr)
        return 1

    try:
        project = get_project(conn, args.project_id)
        if project is None:
            print(
                f"register-visual-artifact: FAILED — no project found with project_id {args.project_id!r}",
                file=sys.stderr,
            )
            return 1

        try:
            manifest = load_manifest(Path(project.manifest_path))
        except ManifestStoreError as exc:
            print(f"register-visual-artifact: FAILED — {exc}", file=sys.stderr)
            return 1

        try:
            result = register_visual_artifact(conn, project, manifest, args.scene_id, Path(args.file))
        except VisualArtifactRegistrationError as exc:
            print(f"register-visual-artifact: FAILED — {exc}", file=sys.stderr)
            return 1
    except sqlite3.Error as exc:
        print(f"register-visual-artifact: FAILED — {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()

    if out_format == "json":
        payload = {
            "project_id": result.project_id,
            "scene_id": result.scene_id,
            "artifact_id": result.artifact_id,
            "relative_path": result.relative_path,
            "ok": result.ok,
            "idempotent": result.idempotent,
            "copied": result.copied,
            "width": result.width,
            "height": result.height,
            "format": result.image_format,
            "reasons": list(result.reasons),
            "artifact": result.artifact.model_dump(mode="json") if result.artifact is not None else None,
        }
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
    else:
        print(_render_register_visual_artifact_text(result))

    return 0 if result.ok else 1


def _render_register_animation_artifact_text(result) -> str:
    status = "OK" if result.ok else "FAILED"
    idempotent_note = " (idempotent no-op — nothing was written)" if result.idempotent else ""
    lines = [
        f"register-animation-artifact: {status}{idempotent_note} "
        "(writes exactly one artifact row on a fresh registration; the manifest and "
        "project lifecycle stage are never touched)",
        f"project_id: {result.project_id}",
        f"scene_id: {result.scene_id}",
        f"artifact_id: {result.artifact_id}",
        f"relative_path: {result.relative_path}",
        f"copied: {result.copied}",
        f"duration_seconds: {result.duration_seconds}",
    ]
    for reason in result.reasons:
        lines.append(f"  - {reason}")
    return "\n".join(lines)


def cmd_build_upscaled_ken_burns(args: argparse.Namespace) -> int:
    """Phase 1D vertical slice, generation-only: build an upscale-assisted
    Ken Burns MP4 clip for one scene from its already-registered visual
    artifact and the caller-supplied enriched manifest's measured audio
    duration. Writes only the --output MP4 file — no SQLite write, no
    artifact registration, no manifest change, and no lifecycle
    transition. Register the result separately with
    register-animation-artifact (unchanged, run as its own later step).

    Strictly read-only against SQLite: uses get_readonly_connection()
    (never init_db()), same never-create-anything contract as
    finalize-scene-timing/dry-run, and the connection is closed before the
    local upscale/ffmpeg pipeline (src/core/ken_burns_upscale_pipeline.py)
    is ever invoked."""
    from src.core.ken_burns_upscale_pipeline import (
        KenBurnsUpscalePipelineError,
        build_upscaled_ken_burns_clip,
    )
    from src.core.manifest_store import ManifestStoreError, load_manifest
    from src.core.path_safety import resolve_under_project_dir
    from src.database.artifact_repository import list_artifacts_by_scene
    from src.database.db import get_readonly_connection
    from src.database.project_repository import get_project

    out_format = args.format
    if out_format not in {"text", "json"}:
        print(
            f"build-upscaled-ken-burns: FAILED — invalid --format {out_format!r} (expected 'text' or 'json')",
            file=sys.stderr,
        )
        return 1

    def _fail(reason: str) -> int:
        if out_format == "json":
            payload = {
                "ok": False,
                "project_id": args.project_id,
                "scene_id": args.scene_id,
                "reason": reason,
            }
            print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
        else:
            print(f"build-upscaled-ken-burns: FAILED — {reason}", file=sys.stderr)
        return 1

    output_path = Path(args.output)

    try:
        conn = get_readonly_connection()
    except sqlite3.Error:
        return _fail("no local project database found")

    try:
        project = get_project(conn, args.project_id)
        if project is None:
            return _fail(f"unknown project_id {args.project_id!r}")

        try:
            manifest = load_manifest(Path(args.manifest))
        except ManifestStoreError:
            return _fail("could not load manifest at --manifest (unreadable or invalid)")

        if manifest.project_id != args.project_id:
            return _fail(f"--manifest project_id does not match project_id {args.project_id!r}")
        if manifest.source_fingerprint != project.manifest_fingerprint:
            return _fail(
                "--manifest source_fingerprint does not match the project registry's recorded fingerprint"
            )

        scene = next((s for s in manifest.scene_plan.scenes if s.scene_id == args.scene_id), None)
        if scene is None:
            return _fail(f"unknown scene_id {args.scene_id!r} in the provided manifest")

        visual_artifacts = [
            a for a in list_artifacts_by_scene(conn, args.project_id, args.scene_id) if a.kind == "visual"
        ]
        if len(visual_artifacts) == 0:
            return _fail(f"no registered visual artifact found for scene_id {args.scene_id!r}")
        if len(visual_artifacts) > 1:
            return _fail(
                f"multiple registered visual artifacts found for scene_id {args.scene_id!r} "
                "(expected exactly 1)"
            )

        project_dir_resolved = Path(project.manifest_path).resolve().parent
        source_visual_path = resolve_under_project_dir(project_dir_resolved, visual_artifacts[0].relative_path)
        if source_visual_path is None:
            return _fail("registered visual artifact path is unsafe")
        if not source_visual_path.exists():
            return _fail("registered visual artifact file is missing on disk")
        if not source_visual_path.is_file():
            return _fail("registered visual artifact path is not a regular file")
        if source_visual_path.stat().st_size == 0:
            return _fail("registered visual artifact file is empty")
    finally:
        conn.close()

    try:
        build_upscaled_ken_burns_clip(scene, source_visual_path, output_path)
    except KenBurnsUpscalePipelineError as exc:
        return _fail(str(exc))

    if out_format == "json":
        payload = {
            "ok": True,
            "project_id": args.project_id,
            "scene_id": args.scene_id,
            "output": str(output_path),
        }
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
    else:
        print(
            "build-upscaled-ken-burns: OK (clip generated; not registered — "
            "run register-animation-artifact separately)"
        )
        print(f"  project_id: {args.project_id}")
        print(f"  scene_id: {args.scene_id}")
        print(f"  output: {output_path}")
    return 0


def cmd_preview_scene_image_prompt(args: argparse.Namespace) -> int:
    """Read-only: preview the exact image-generation prompt a future
    build-scene-image command would use for one scene, without calling any
    image provider. Loads the scene's visual_brief from the existing
    project/manifest records (same identity checks as every prior
    generation-only command), validates --reference-image and --output as
    plain local paths (existence only — never opened/decoded, never
    created), and assembles the final prompt via the already-shipped,
    provider-free build_scene_image_prompt(). Performs no generation, no
    download, no file write, no SQLite write, no artifact registration, and
    no lifecycle transition.

    Strictly read-only against SQLite: uses get_readonly_connection() (never
    init_db()), same contract as every prior generation-only/read-only
    command in this CLI. The connection is closed before any core preview
    work begins — src/core/scene_image_preview.py imports nothing from
    src.database or any provider module."""
    from src.core.manifest_store import ManifestStoreError, load_manifest
    from src.core.scene_image_preview import ScenePreviewError, build_scene_image_preview
    from src.database.db import get_readonly_connection
    from src.database.project_repository import get_project

    out_format = args.format
    if out_format not in {"text", "json"}:
        print(
            f"preview-scene-image-prompt: FAILED — invalid --format {out_format!r} (expected 'text' or 'json')",
            file=sys.stderr,
        )
        return 1

    def _fail(reason: str) -> int:
        if out_format == "json":
            payload = {
                "ok": False,
                "project_id": args.project_id,
                "scene_id": args.scene_id,
                "reason": reason,
            }
            print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
        else:
            print(f"preview-scene-image-prompt: FAILED — {reason}", file=sys.stderr)
        return 1

    include_character_anchor = args.include_character == "true"
    include_color_anchor = args.include_color_anchor == "true"

    try:
        conn = get_readonly_connection()
    except sqlite3.Error:
        return _fail("no local project database found")

    try:
        project = get_project(conn, args.project_id)
        if project is None:
            return _fail(f"unknown project_id {args.project_id!r}")

        try:
            manifest = load_manifest(Path(args.manifest))
        except ManifestStoreError:
            return _fail("could not load manifest at --manifest (unreadable or invalid)")

        if manifest.project_id != args.project_id:
            return _fail(f"--manifest project_id does not match project_id {args.project_id!r}")
        if manifest.source_fingerprint != project.manifest_fingerprint:
            return _fail(
                "--manifest source_fingerprint does not match the project registry's recorded fingerprint"
            )

        scene = next((s for s in manifest.scene_plan.scenes if s.scene_id == args.scene_id), None)
        if scene is None:
            return _fail(f"unknown scene_id {args.scene_id!r} in the provided manifest")
    except sqlite3.Error:
        # A DB read failure after the connection already opened
        # successfully — distinct from get_readonly_connection() itself
        # failing above, which already has its own specialized message.
        return _fail("local project database could not be read")
    finally:
        conn.close()

    reference_image_path = Path(args.reference_image)
    planned_output_path = Path(args.output)

    try:
        preview = build_scene_image_preview(
            args.project_id,
            scene,
            reference_image_path,
            planned_output_path,
            include_character_anchor=include_character_anchor,
            include_color_anchor=include_color_anchor,
        )
    except ScenePreviewError as exc:
        return _fail(str(exc))

    if out_format == "json":
        payload = {
            "ok": True,
            "project_id": preview.project_id,
            "scene_id": preview.scene_id,
            "visual_brief": preview.visual_brief,
            "final_prompt": preview.final_prompt,
            "reference_image_path": str(preview.reference_image_path),
            "planned_output_path": str(preview.planned_output_path),
            "include_character_anchor": preview.include_character_anchor,
            "include_color_anchor": preview.include_color_anchor,
        }
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
    else:
        print("preview-scene-image-prompt: OK — prompt preview only; no image generated")
        print(f"PROJECT_ID: {preview.project_id}")
        print(f"SCENE_ID: {preview.scene_id}")
        print(f"INCLUDE_CHARACTER_ANCHOR: {preview.include_character_anchor}")
        print(f"INCLUDE_COLOR_ANCHOR: {preview.include_color_anchor}")
        print(f"REFERENCE_IMAGE: {preview.reference_image_path}")
        print(f"PLANNED_OUTPUT: {preview.planned_output_path}")
        print("FINAL_PROMPT:")
        print(preview.final_prompt)
    return 0


def cmd_build_scene_image(args: argparse.Namespace) -> int:
    """Generation-only: build one scene's image via
    QwenImageProvider.generate_with_reference_and_download() — the first
    real (paid, network) generation call in this CLI. Writes only the
    --output PNG file (via the provider) — no SQLite write, no artifact
    registration, no manifest change, no lifecycle transition. Register
    the result separately with register-visual-artifact (unchanged, run as
    its own later step).

    Qwen-Image is treated as a fixed paid service —
    require_paid_approval("qwen-image", proposals_path, is_paid=True) is
    called (inside src.core.scene_image_generation.build_scene_image())
    before any provider construction; is_paid is never a CLI option, so no
    invocation can declare its own way past the guard.
    build_scene_image() itself still lets PaidApprovalRequiredError
    propagate normally (unchanged) — it is this command, at the CLI
    boundary only, that catches exactly that one exception type (never a
    broad Exception) and reports it through the same _fail() text/JSON
    convention as every other rejection below, with a short fixed message
    that never echoes proposals_path, service_name interpolation beyond
    the literal "qwen-image", provider response text, or a traceback.
    Denial never results in constructing, calling, or falling back to any
    provider.

    Strictly read-only against SQLite: uses get_readonly_connection() (never
    init_db()), same contract as every other generation-only command in this
    CLI. The connection is closed before prompt construction, the cost
    guard, or any provider work begins — src/core/scene_image_generation.py
    imports nothing from src.database."""
    from src.core.cost_guard import PaidApprovalRequiredError
    from src.core.manifest_store import ManifestStoreError, load_manifest
    from src.core.scene_image_generation import SceneImageGenerationError, build_scene_image
    from src.database.db import get_readonly_connection
    from src.database.project_repository import get_project
    from src.utils.config import get_settings

    out_format = args.format
    if out_format not in {"text", "json"}:
        print(
            f"build-scene-image: FAILED — invalid --format {out_format!r} (expected 'text' or 'json')",
            file=sys.stderr,
        )
        return 1

    def _fail(reason: str) -> int:
        if out_format == "json":
            payload = {
                "ok": False,
                "project_id": args.project_id,
                "scene_id": args.scene_id,
                "reason": reason,
            }
            print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
        else:
            print(f"build-scene-image: FAILED — {reason}", file=sys.stderr)
        return 1

    include_character_anchor = args.include_character == "true"
    include_color_anchor = args.include_color_anchor == "true"

    try:
        conn = get_readonly_connection()
    except sqlite3.Error:
        return _fail("no local project database found")

    try:
        project = get_project(conn, args.project_id)
        if project is None:
            return _fail(f"unknown project_id {args.project_id!r}")

        try:
            manifest = load_manifest(Path(args.manifest))
        except ManifestStoreError:
            return _fail("could not load manifest at --manifest (unreadable or invalid)")

        if manifest.project_id != args.project_id:
            return _fail(f"--manifest project_id does not match project_id {args.project_id!r}")
        if manifest.source_fingerprint != project.manifest_fingerprint:
            return _fail(
                "--manifest source_fingerprint does not match the project registry's recorded fingerprint"
            )

        scene = next((s for s in manifest.scene_plan.scenes if s.scene_id == args.scene_id), None)
        if scene is None:
            return _fail(f"unknown scene_id {args.scene_id!r} in the provided manifest")
    except sqlite3.Error:
        # A DB read failure after the connection already opened
        # successfully — distinct from get_readonly_connection() itself
        # failing above, which already has its own specialized message.
        return _fail("local project database could not be read")
    finally:
        conn.close()

    reference_image_path = Path(args.reference_image)
    output_path = Path(args.output)
    proposals_path = (
        Path(args.proposals_path) if args.proposals_path is not None else get_settings().data_dir / "paid_proposals.json"
    )

    try:
        build_scene_image(
            scene,
            reference_image_path,
            output_path,
            proposals_path,
            include_character_anchor=include_character_anchor,
            include_color_anchor=include_color_anchor,
        )
    except PaidApprovalRequiredError:
        return _fail("qwen-image is not approved for a paid provider call")
    except SceneImageGenerationError as exc:
        return _fail(str(exc))

    if out_format == "json":
        payload = {
            "ok": True,
            "project_id": args.project_id,
            "scene_id": args.scene_id,
            "output": str(output_path),
            "reference_image": str(reference_image_path),
            "include_character_anchor": include_character_anchor,
            "include_color_anchor": include_color_anchor,
            "registered": False,
        }
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
    else:
        print("build-scene-image: OK (scene image generated; not registered)")
        print(f"  project_id: {args.project_id}")
        print(f"  scene_id: {args.scene_id}")
        print(f"  output: {output_path}")
        print(f"  reference_image: {reference_image_path}")
        print(f"  include_character_anchor: {include_character_anchor}")
        print(f"  include_color_anchor: {include_color_anchor}")
    return 0


def cmd_build_scene_audio(args: argparse.Namespace) -> int:
    """Generation-only: build one scene's narration audio via
    KokoroProvider.synthesize() — self-hosted, local TTS, no network call.
    Writes only the --output WAV file (via the provider) — no SQLite
    write, no artifact registration, no manifest change, no lifecycle
    transition. Register the result separately with
    register-audio-artifact (unchanged, run as its own later step).

    Kokoro is treated as a fixed free service —
    require_paid_approval("kokoro", proposals_path, is_paid=False) is
    called (inside src.core.scene_audio_generation.build_scene_audio())
    before any provider construction, kept for the same consistency reason
    every other currently-reachable real provider call site in this CLI is
    wired through it. A PaidApprovalRequiredError from that call is never
    caught inside the core module — it is caught here, at the CLI
    boundary only, and reported through the same _fail() text/JSON
    convention as every other rejection below (matching
    cmd_build_scene_image's own corrected convention), never left to
    propagate as a raw exception.

    Strictly read-only against SQLite: uses get_readonly_connection() (never
    init_db()), same contract as every other generation-only command in this
    CLI. The connection is closed before any core work begins —
    src/core/scene_audio_generation.py imports nothing from src.database."""
    from src.core.cost_guard import PaidApprovalRequiredError
    from src.core.manifest_store import ManifestStoreError, load_manifest
    from src.core.scene_audio_generation import SceneAudioGenerationError, build_scene_audio
    from src.database.db import get_readonly_connection
    from src.database.project_repository import get_project
    from src.providers.tts_kokoro import (
        DEFAULT_VOICE,
        DOCUMENTARY_CLAUSE_PAUSE,
        DOCUMENTARY_SENTENCE_PAUSE,
        DOCUMENTARY_SPEED,
    )
    from src.utils.config import get_settings

    out_format = args.format
    if out_format not in {"text", "json"}:
        print(
            f"build-scene-audio: FAILED — invalid --format {out_format!r} (expected 'text' or 'json')",
            file=sys.stderr,
        )
        return 1

    def _fail(reason: str) -> int:
        if out_format == "json":
            payload = {
                "ok": False,
                "project_id": args.project_id,
                "scene_id": args.scene_id,
                "reason": reason,
            }
            print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
        else:
            print(f"build-scene-audio: FAILED — {reason}", file=sys.stderr)
        return 1

    voice = args.voice if args.voice is not None else DEFAULT_VOICE
    try:
        speed = float(args.speed) if args.speed is not None else DOCUMENTARY_SPEED
        sentence_pause = float(args.sentence_pause) if args.sentence_pause is not None else DOCUMENTARY_SENTENCE_PAUSE
        clause_pause = float(args.clause_pause) if args.clause_pause is not None else DOCUMENTARY_CLAUSE_PAUSE
    except ValueError:
        return _fail("--speed/--sentence-pause/--clause-pause must be valid decimal numbers")

    try:
        conn = get_readonly_connection()
    except sqlite3.Error:
        return _fail("no local project database found")

    try:
        project = get_project(conn, args.project_id)
        if project is None:
            return _fail(f"unknown project_id {args.project_id!r}")

        try:
            manifest = load_manifest(Path(args.manifest))
        except ManifestStoreError:
            return _fail("could not load manifest at --manifest (unreadable or invalid)")

        if manifest.project_id != args.project_id:
            return _fail(f"--manifest project_id does not match project_id {args.project_id!r}")
        if manifest.source_fingerprint != project.manifest_fingerprint:
            return _fail(
                "--manifest source_fingerprint does not match the project registry's recorded fingerprint"
            )

        scene = next((s for s in manifest.scene_plan.scenes if s.scene_id == args.scene_id), None)
        if scene is None:
            return _fail(f"unknown scene_id {args.scene_id!r} in the provided manifest")
    except sqlite3.Error:
        # A DB read failure after the connection already opened
        # successfully — distinct from get_readonly_connection() itself
        # failing above, which already has its own specialized message.
        return _fail("local project database could not be read")
    finally:
        conn.close()

    output_path = Path(args.output)
    proposals_path = get_settings().data_dir / "paid_proposals.json"

    try:
        build_scene_audio(
            scene,
            output_path,
            proposals_path,
            voice=voice,
            speed=speed,
            sentence_pause=sentence_pause,
            clause_pause=clause_pause,
        )
    except PaidApprovalRequiredError:
        return _fail("kokoro is not approved for a paid provider call")
    except SceneAudioGenerationError as exc:
        return _fail(str(exc))

    if out_format == "json":
        payload = {
            "ok": True,
            "project_id": args.project_id,
            "scene_id": args.scene_id,
            "output": str(output_path),
            "voice": voice,
            "speed": speed,
            "sentence_pause": sentence_pause,
            "clause_pause": clause_pause,
            "registered": False,
        }
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
    else:
        print("build-scene-audio: OK (scene audio generated; not registered)")
        print(f"  project_id: {args.project_id}")
        print(f"  scene_id: {args.scene_id}")
        print(f"  output: {output_path}")
        print(f"  voice: {voice}")
        print(f"  speed: {speed}")
        print(f"  sentence_pause: {sentence_pause}")
        print(f"  clause_pause: {clause_pause}")
    return 0


def cmd_register_animation_artifact(args: argparse.Namespace) -> int:
    from src.core.animation_artifact_registrar import (
        AnimationArtifactRegistrationError,
        register_animation_artifact,
    )
    from src.core.manifest_store import ManifestStoreError, load_manifest
    from src.database.db import get_existing_connection
    from src.database.project_repository import get_project

    out_format = args.format
    if out_format not in {"text", "json"}:
        print(
            f"register-animation-artifact: FAILED — invalid --format {out_format!r} (expected 'text' or 'json')",
            file=sys.stderr,
        )
        return 1

    # Never calls init_db(): a missing data directory or database file
    # must fail cleanly here, not be created — same contract as
    # verify-and-advance's/register-audio-artifact's/register-visual-artifact's
    # get_existing_connection().
    try:
        conn = get_existing_connection()
    except sqlite3.Error as exc:
        print(f"register-animation-artifact: FAILED — no local project database found: {exc}", file=sys.stderr)
        return 1

    try:
        project = get_project(conn, args.project_id)
        if project is None:
            print(
                f"register-animation-artifact: FAILED — no project found with project_id {args.project_id!r}",
                file=sys.stderr,
            )
            return 1

        try:
            manifest = load_manifest(Path(project.manifest_path))
        except ManifestStoreError as exc:
            print(f"register-animation-artifact: FAILED — {exc}", file=sys.stderr)
            return 1

        try:
            result = register_animation_artifact(conn, project, manifest, args.scene_id, Path(args.file))
        except AnimationArtifactRegistrationError as exc:
            print(f"register-animation-artifact: FAILED — {exc}", file=sys.stderr)
            return 1
    except sqlite3.Error as exc:
        print(f"register-animation-artifact: FAILED — {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()

    if out_format == "json":
        payload = {
            "project_id": result.project_id,
            "scene_id": result.scene_id,
            "artifact_id": result.artifact_id,
            "relative_path": result.relative_path,
            "ok": result.ok,
            "idempotent": result.idempotent,
            "copied": result.copied,
            "duration_seconds": result.duration_seconds,
            "reasons": list(result.reasons),
            "artifact": result.artifact.model_dump(mode="json") if result.artifact is not None else None,
        }
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
    else:
        print(_render_register_animation_artifact_text(result))

    return 0 if result.ok else 1


def _render_register_render_artifact_text(result) -> str:
    status = "OK" if result.ok else "FAILED"
    idempotent_note = " (idempotent no-op — nothing was written)" if result.idempotent else ""
    lines = [
        f"register-render-artifact: {status}{idempotent_note} "
        "(writes exactly one artifact row on a fresh registration; the manifest and "
        "project lifecycle stage are never touched)",
        f"project_id: {result.project_id}",
        f"artifact_id: {result.artifact_id}",
        f"relative_path: {result.relative_path}",
        f"copied: {result.copied}",
        f"duration_seconds: {result.duration_seconds}",
    ]
    for reason in result.reasons:
        lines.append(f"  - {reason}")
    return "\n".join(lines)


def cmd_register_render_artifact(args: argparse.Namespace) -> int:
    from src.core.manifest_store import ManifestStoreError, load_manifest
    from src.core.render_artifact_registrar import (
        RenderArtifactRegistrationError,
        register_render_artifact,
    )
    from src.database.db import get_existing_connection
    from src.database.project_repository import get_project

    out_format = args.format
    if out_format not in {"text", "json"}:
        print(
            f"register-render-artifact: FAILED — invalid --format {out_format!r} (expected 'text' or 'json')",
            file=sys.stderr,
        )
        return 1

    # Never calls init_db(): a missing data directory or database file
    # must fail cleanly here, not be created — same contract as
    # verify-and-advance's/the three prior register-*-artifact commands'
    # get_existing_connection().
    try:
        conn = get_existing_connection()
    except sqlite3.Error as exc:
        print(f"register-render-artifact: FAILED — no local project database found: {exc}", file=sys.stderr)
        return 1

    try:
        project = get_project(conn, args.project_id)
        if project is None:
            print(
                f"register-render-artifact: FAILED — no project found with project_id {args.project_id!r}",
                file=sys.stderr,
            )
            return 1

        try:
            manifest = load_manifest(Path(project.manifest_path))
        except ManifestStoreError as exc:
            print(f"register-render-artifact: FAILED — {exc}", file=sys.stderr)
            return 1

        try:
            result = register_render_artifact(conn, project, manifest, Path(args.file))
        except RenderArtifactRegistrationError as exc:
            print(f"register-render-artifact: FAILED — {exc}", file=sys.stderr)
            return 1
    except sqlite3.Error as exc:
        print(f"register-render-artifact: FAILED — {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()

    if out_format == "json":
        payload = {
            "project_id": result.project_id,
            "artifact_id": result.artifact_id,
            "relative_path": result.relative_path,
            "ok": result.ok,
            "idempotent": result.idempotent,
            "copied": result.copied,
            "duration_seconds": result.duration_seconds,
            "reasons": list(result.reasons),
            "artifact": result.artifact.model_dump(mode="json") if result.artifact is not None else None,
        }
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
    else:
        print(_render_register_render_artifact_text(result))

    return 0 if result.ok else 1


def _render_register_qc_report_artifact_text(result) -> str:
    status = "OK" if result.ok else "FAILED"
    idempotent_note = " (idempotent no-op — nothing was written)" if result.idempotent else ""
    lines = [
        f"register-qc-report-artifact: {status}{idempotent_note} "
        "(writes exactly one artifact row on a fresh registration; the manifest and "
        "project lifecycle stage are never touched — passed:false is registered, not rejected; "
        "only verify-and-advance(to=qc_passed) gates on it)",
        f"project_id: {result.project_id}",
        f"artifact_id: {result.artifact_id}",
        f"relative_path: {result.relative_path}",
        f"copied: {result.copied}",
        f"passed: {result.passed}",
    ]
    for reason in result.reasons:
        lines.append(f"  - {reason}")
    return "\n".join(lines)


def cmd_register_qc_report_artifact(args: argparse.Namespace) -> int:
    from src.core.manifest_store import ManifestStoreError, load_manifest
    from src.core.qc_report_artifact_registrar import (
        QcReportArtifactRegistrationError,
        register_qc_report_artifact,
    )
    from src.database.db import get_existing_connection
    from src.database.project_repository import get_project

    out_format = args.format
    if out_format not in {"text", "json"}:
        print(
            f"register-qc-report-artifact: FAILED — invalid --format {out_format!r} (expected 'text' or 'json')",
            file=sys.stderr,
        )
        return 1

    # Never calls init_db(): a missing data directory or database file
    # must fail cleanly here, not be created — same contract as
    # verify-and-advance's/the three prior register-*-artifact commands'
    # get_existing_connection().
    try:
        conn = get_existing_connection()
    except sqlite3.Error as exc:
        print(f"register-qc-report-artifact: FAILED — no local project database found: {exc}", file=sys.stderr)
        return 1

    try:
        project = get_project(conn, args.project_id)
        if project is None:
            print(
                f"register-qc-report-artifact: FAILED — no project found with project_id {args.project_id!r}",
                file=sys.stderr,
            )
            return 1

        try:
            manifest = load_manifest(Path(project.manifest_path))
        except ManifestStoreError as exc:
            print(f"register-qc-report-artifact: FAILED — {exc}", file=sys.stderr)
            return 1

        try:
            result = register_qc_report_artifact(conn, project, manifest, Path(args.file))
        except QcReportArtifactRegistrationError as exc:
            print(f"register-qc-report-artifact: FAILED — {exc}", file=sys.stderr)
            return 1
    except sqlite3.Error as exc:
        print(f"register-qc-report-artifact: FAILED — {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()

    if out_format == "json":
        payload = {
            "project_id": result.project_id,
            "artifact_id": result.artifact_id,
            "relative_path": result.relative_path,
            "ok": result.ok,
            "idempotent": result.idempotent,
            "copied": result.copied,
            "passed": result.passed,
            "reasons": list(result.reasons),
            "artifact": result.artifact.model_dump(mode="json") if result.artifact is not None else None,
        }
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
    else:
        print(_render_register_qc_report_artifact_text(result))

    return 0 if result.ok else 1


def _render_advance_project_stage_text(result) -> str:
    lines = [
        "advance-project-stage: "
        + ("OK" if result.approved and result.db_committed else "FAILED")
        + " (writes to SQLite only on success; never an artifact file, manifest, or queue file; "
        "never a verification-required stage — those stay behind verify-and-advance)",
        f"project_id: {result.project_id}",
        f"from_stage: {result.from_stage}",
        f"to_stage: {result.to_stage}",
        f"approved: {result.approved}",
        f"db_committed: {result.db_committed}",
        f"lifecycle_version_before: {result.lifecycle_version_before}",
        f"lifecycle_version_after: {result.lifecycle_version_after}",
    ]
    for reason in result.reasons:
        lines.append(f"  - {reason}")
    return "\n".join(lines)


def cmd_advance_project_stage(args: argparse.Namespace) -> int:
    from src.core.stage_advance_service import advance_project_stage
    from src.database.db import get_existing_connection
    from src.database.project_repository import get_project

    out_format = args.format
    if out_format not in {"text", "json"}:
        print(
            f"advance-project-stage: FAILED — invalid --format {out_format!r} (expected 'text' or 'json')",
            file=sys.stderr,
        )
        return 1

    # Never calls init_db(): a missing data directory or database file
    # must fail cleanly here, not be created — same contract as
    # verify-and-advance's get_existing_connection().
    try:
        conn = get_existing_connection()
    except sqlite3.Error as exc:
        print(f"advance-project-stage: FAILED — no local project database found: {exc}", file=sys.stderr)
        return 1

    try:
        project = get_project(conn, args.project_id)
        if project is None:
            print(
                f"advance-project-stage: FAILED — no project found with project_id {args.project_id!r}",
                file=sys.stderr,
            )
            return 1

        result = advance_project_stage(conn, project, args.to, reason=args.reason)
    except sqlite3.Error as exc:
        print(f"advance-project-stage: FAILED — {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()

    if out_format == "json":
        payload = {
            "project_id": result.project_id,
            "from_stage": result.from_stage,
            "to_stage": result.to_stage,
            "reason": result.reason,
            "approved": result.approved,
            "db_committed": result.db_committed,
            "reasons": list(result.reasons),
            "lifecycle_version_before": result.lifecycle_version_before,
            "lifecycle_version_after": result.lifecycle_version_after,
        }
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
    else:
        print(_render_advance_project_stage_text(result))

    return 0 if result.approved and result.db_committed else 1


def cmd_assemble_final_video(args: argparse.Namespace) -> int:
    """FINAL VIDEO ASSEMBLY V1: local-only, generation-and-registration
    final assembly for one project. Resolves every scene's already-
    registered animation+audio artifacts (manifest order), concatenates
    them into one MP4 at --output, and registers the result as this
    project's canonical "render" artifact via the existing, unmodified
    register_render_artifact(). Local FFmpeg/ffprobe only — no provider,
    no network call anywhere in this command path.

    All SQLite access lives inside src.core.final_video_assembly.assemble_final_video()
    itself (two short-lived connections, opened and closed internally,
    never held open across the FFmpeg work in between) — a deliberate
    departure from every other command's "CLI owns all SQLite access"
    shape, required by this command's three-part lifecycle; see that
    module's own docstring for why. This command function therefore opens
    no connection of its own.

    JSON-mode failures are printed to stderr here, unlike this codebase's
    other 20+ commands (whose own `_fail()` closures print JSON — success
    or failure alike — to stdout). That is a deliberate, narrow exception
    for this one command, not a silent inconsistency: this command's own
    review explicitly required errors to go to stderr in both output
    modes. The other commands' existing stdout-for-all-JSON behavior is
    untouched.

    An undocumented/unexpected exception (not a FinalVideoAssemblyError
    subclass) is caught at this boundary and reported as one short,
    generic, sanitized message — never the original exception's own text,
    traceback, or any environment/provider credential.
    KeyboardInterrupt/SystemExit are BaseException, not Exception, so
    neither is ever caught here."""
    from src.core.final_video_assembly import FinalVideoAssemblyError, assemble_final_video

    out_format = args.format
    if out_format not in {"text", "json"}:
        print(
            f"assemble-final-video: FAILED — invalid --format {out_format!r} (expected 'text' or 'json')",
            file=sys.stderr,
        )
        return 1

    def _fail(reason: str) -> int:
        if out_format == "json":
            payload = {"ok": False, "project_id": args.project_id, "reason": reason}
            print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True), file=sys.stderr)
        else:
            print(f"assemble-final-video: FAILED — {reason}", file=sys.stderr)
        return 1

    try:
        result = assemble_final_video(args.project_id, Path(args.manifest), Path(args.output))
    except FinalVideoAssemblyError as exc:
        return _fail(str(exc))
    except Exception:
        return _fail("an unexpected internal error occurred")

    if out_format == "json":
        payload = {
            "ok": True,
            "project_id": result.project_id,
            "output": str(result.output_path),
            "scene_count": result.scene_count,
            "measured_duration_seconds": result.measured_duration_seconds,
            "artifact_id": result.artifact_id,
        }
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
    else:
        print("assemble-final-video: OK (final video assembled and registered)")
        print(f"  project_id: {result.project_id}")
        print(f"  output: {result.output_path}")
        print(f"  scene_count: {result.scene_count}")
        print(f"  measured_duration_seconds: {result.measured_duration_seconds}")
        print(f"  artifact_id: {result.artifact_id}")
    return 0


def cmd_render_text_overlays(args: argparse.Namespace) -> int:
    """TEXT OVERLAY RENDERER V1: local-only, deterministic drawtext
    burn-in of every manifest-declared TextOverlay onto one project's
    already-registered "render" artifact, producing a new, distinct
    "overlay_render" artifact at --output. Never accepts an arbitrary
    source MP4 — the render artifact is resolved internally from the
    project's own registry. Local FFmpeg/ffprobe only — no provider, no
    network call anywhere in this command path.

    All SQLite access lives inside
    src.core.text_overlay_render.render_text_overlays() itself (two
    short-lived connections, opened and closed internally, never held
    open across the FFmpeg work in between) — same three-part-lifecycle
    shape as cmd_assemble_final_video, for the same reason.

    JSON-mode failures are printed to stderr here, matching
    cmd_assemble_final_video's own established (narrow, deliberate)
    exception to this codebase's other 20+ commands' stdout-for-all-JSON
    convention.

    An undocumented/unexpected exception (not a TextOverlayRenderError
    subclass) is caught at this boundary and reported as one short,
    generic, sanitized message — never the original exception's own text,
    traceback, or any environment/provider credential.
    KeyboardInterrupt/SystemExit are BaseException, not Exception, so
    neither is ever caught here."""
    from src.core.text_overlay_render import TextOverlayRenderError, render_text_overlays

    out_format = args.format
    if out_format not in {"text", "json"}:
        print(
            f"render-text-overlays: FAILED — invalid --format {out_format!r} (expected 'text' or 'json')",
            file=sys.stderr,
        )
        return 1

    def _fail(reason: str) -> int:
        if out_format == "json":
            payload = {"ok": False, "project_id": args.project_id, "reason": reason}
            print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True), file=sys.stderr)
        else:
            print(f"render-text-overlays: FAILED — {reason}", file=sys.stderr)
        return 1

    try:
        result = render_text_overlays(args.project_id, Path(args.manifest), Path(args.output))
    except TextOverlayRenderError as exc:
        return _fail(str(exc))
    except Exception:
        return _fail("an unexpected internal error occurred")

    if out_format == "json":
        payload = {
            "ok": True,
            "project_id": result.project_id,
            "output": str(result.output_path),
            "overlay_count": result.overlay_count,
            "measured_duration_seconds": result.measured_duration_seconds,
            "artifact_id": result.artifact_id,
        }
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
    else:
        print("render-text-overlays: OK (text overlays burned in and registered)")
        print(f"  project_id: {result.project_id}")
        print(f"  output: {result.output_path}")
        print(f"  overlay_count: {result.overlay_count}")
        print(f"  measured_duration_seconds: {result.measured_duration_seconds}")
        print(f"  artifact_id: {result.artifact_id}")
    return 0


def cmd_derive_text_overlays(args: argparse.Namespace) -> int:
    """DETERMINISTIC OVERLAY DERIVATION V1: local-only, pure manifest
    enrichment. Reads --manifest (an already finalize-scene-timing'd copy,
    every scene's artifacts.measured_audio_duration_seconds populated) and
    the project's single registered project-level "render" artifact, then
    writes a NEW manifest to --output in which every scene with no
    explicit text_overlays receives exactly one deterministically-derived
    TextOverlay — preparing input for the already-merged
    `render-text-overlays` command. No FFmpeg/ffprobe/provider/network
    call anywhere in this command path; no database write; no artifact
    registration; no project/lifecycle-stage change.

    DB access shape follows cmd_finalize_scene_timing exactly: one
    short-lived get_readonly_connection(), used only to fetch the project
    record and resolve exactly one project-level "render" artifact,
    closed via the `finally` block BEFORE src.core.overlay_derivation.
    derive_text_overlays() (a pure function, no I/O of its own) is ever
    called.

    stdout/stderr shape follows cmd_render_text_overlays exactly: success
    (text or JSON) goes to stdout; a domain error (text or JSON) goes to
    stderr — this command's own narrow exception to this codebase's other
    20+ commands' stdout-for-all-JSON convention, chosen for consistency
    with the other already-merged TEXT OVERLAY RENDERER V1 command rather
    than cmd_finalize_scene_timing's own (JSON-less, stderr-only) shape.

    An undocumented/unexpected exception (not an OverlayDerivationError
    subclass, nor a ManifestStoreError, nor a project/argument validation
    failure raised directly here) is caught at this boundary and reported
    as one short, generic, sanitized message — never the original
    exception's own text, traceback, or any environment/provider
    credential. KeyboardInterrupt/SystemExit are BaseException, not
    Exception, so neither is ever caught here."""
    from src.core.manifest_store import ManifestStoreError, load_manifest, save_manifest
    from src.core.overlay_derivation import OverlayDerivationError, derive_text_overlays, summarize_derivation
    from src.database.artifact_repository import list_artifacts_by_project
    from src.database.db import get_readonly_connection
    from src.database.project_repository import get_project

    out_format = args.format
    if out_format not in {"text", "json"}:
        print(
            f"derive-text-overlays: FAILED — invalid --format {out_format!r} (expected 'text' or 'json')",
            file=sys.stderr,
        )
        return 1

    def _fail(reason: str) -> int:
        if out_format == "json":
            payload = {"ok": False, "project_id": args.project_id, "reason": reason}
            print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True), file=sys.stderr)
        else:
            print(f"derive-text-overlays: FAILED — {reason}", file=sys.stderr)
        return 1

    try:
        try:
            conn = get_readonly_connection()
        except sqlite3.Error:
            return _fail("no local project database found")

        try:
            project = get_project(conn, args.project_id)
            if project is None:
                return _fail(f"no project found with project_id {args.project_id!r}")

            render_artifacts = list_artifacts_by_project(conn, args.project_id, kind="render")
        finally:
            conn.close()

        if len(render_artifacts) == 0:
            return _fail(f"no 'render' artifact is registered for project {args.project_id!r}")
        if len(render_artifacts) > 1:
            return _fail(
                f"project {args.project_id!r} has {len(render_artifacts)} 'render' artifacts "
                "(expected exactly 1)"
            )
        render_artifact = render_artifacts[0]

        # --output must be a separate, explicit destination — never the
        # project's own canonical manifest path, and never the same file
        # as --manifest itself. Same Path.resolve()/os.path.normcase()
        # convention cmd_finalize_scene_timing already uses.
        output_path_resolved = Path(args.output).resolve()
        output_resolved = os.path.normcase(str(output_path_resolved))
        manifest_path_resolved = os.path.normcase(str(Path(args.manifest).resolve()))
        canonical_manifest_resolved = os.path.normcase(str(Path(project.manifest_path).resolve()))
        if output_resolved == canonical_manifest_resolved:
            return _fail(
                "--output must not be the project's own canonical manifest path; choose a "
                "separate destination for the derived copy"
            )
        if output_resolved == manifest_path_resolved:
            return _fail("--output must not be the same path as --manifest")
        if output_path_resolved.exists():
            return _fail("--output already exists")

        try:
            manifest = load_manifest(Path(args.manifest))
        except ManifestStoreError as exc:
            return _fail(str(exc))

        if manifest.project_id != args.project_id:
            return _fail(
                f"--manifest project_id {manifest.project_id!r} does not match project_id "
                f"{args.project_id!r}"
            )
        if manifest.source_fingerprint != project.manifest_fingerprint:
            return _fail(
                "--manifest source_fingerprint does not match the project registry's recorded "
                "fingerprint"
            )

        try:
            enriched = derive_text_overlays(manifest, render_artifact)
        except OverlayDerivationError as exc:
            return _fail(str(exc))

        summary = summarize_derivation(manifest, enriched)

        try:
            save_manifest(enriched, Path(args.output))
        except ManifestStoreError as exc:
            return _fail(str(exc))
    except Exception:
        return _fail("an unexpected internal error occurred")

    if out_format == "json":
        payload = {
            "ok": True,
            "project_id": enriched.project_id,
            "input_manifest": str(args.manifest),
            "output_manifest": str(args.output),
            "scenes_with_new_overlays": summary.scenes_with_new_overlays,
            "scenes_with_existing_overlays": summary.scenes_with_existing_overlays,
            "derived_overlay_count": summary.derived_overlay_count,
            "preserved_overlay_count": summary.preserved_overlay_count,
            "source_render_artifact_id": render_artifact.artifact_id,
            "source_render_duration_seconds": render_artifact.metadata["duration_seconds"],
            "source_fingerprint": enriched.source_fingerprint,
        }
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
    else:
        print("derive-text-overlays: OK (derived overlays written to a new manifest copy)")
        print(f"  project_id: {enriched.project_id}")
        print(f"  input_manifest: {args.manifest}")
        print(f"  output_manifest: {args.output}")
        print(f"  scenes_with_new_overlays: {summary.scenes_with_new_overlays}")
        print(f"  scenes_with_existing_overlays: {summary.scenes_with_existing_overlays}")
        print(f"  derived_overlay_count: {summary.derived_overlay_count}")
        print(f"  preserved_overlay_count: {summary.preserved_overlay_count}")
        print(f"  source_render_artifact_id: {render_artifact.artifact_id}")
        print(f"  source_render_duration_seconds: {render_artifact.metadata['duration_seconds']}")
        print(f"  source_fingerprint: {enriched.source_fingerprint}")
    return 0


def cmd_plan_final_assembly(args: argparse.Namespace) -> int:
    """FINAL ASSEMBLY PLANNER V1: local-only, read-only readiness report
    for the render -> overlay-derivation -> overlay-rendering tail of the
    pipeline — the part dry-run/resume-plan cannot see, since neither the
    "overlay_render" ArtifactKind nor the manifest-file-based
    overlay-derivation step is part of the ProjectStage lifecycle model.
    This command complements, and never replaces, dry-run/resume-plan/
    verify-artifacts/verify-and-advance, and never calls
    assemble-final-video/derive-text-overlays/render-text-overlays itself.

    DB access shape follows cmd_derive_text_overlays exactly: one
    short-lived get_readonly_connection(), used only to fetch the project
    record and every registered artifact for it, closed via the `finally`
    block BEFORE src.core.final_assembly_planner.build_final_assembly_plan()
    (a pure function, no I/O of its own) is ever called.

    stdout/stderr shape follows cmd_derive_text_overlays exactly: success
    (text or JSON) goes to stdout; a domain error (text or JSON) goes to
    stderr. An undocumented/unexpected exception is caught at this
    boundary and reported as one short, generic, sanitized message — never
    the original exception's own text, traceback, or any environment/
    provider credential. KeyboardInterrupt/SystemExit are BaseException,
    not Exception, so neither is ever caught here.

    No --output argument exists: this command never writes a manifest,
    never registers an artifact, never opens a write database connection,
    and never shells out to FFmpeg/ffprobe, never imports a provider, and
    never touches the network."""
    from src.core.final_assembly_planner import FinalAssemblyPlannerError, build_final_assembly_plan
    from src.core.manifest_store import ManifestStoreError, load_manifest
    from src.database.artifact_repository import list_artifacts_by_project
    from src.database.db import get_readonly_connection
    from src.database.project_repository import get_project

    out_format = args.format
    if out_format not in {"text", "json"}:
        print(
            f"plan-final-assembly: FAILED — invalid --format {out_format!r} (expected 'text' or 'json')",
            file=sys.stderr,
        )
        return 1

    def _fail(reason: str) -> int:
        if out_format == "json":
            payload = {"ok": False, "project_id": args.project_id, "reason": reason}
            print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True), file=sys.stderr)
        else:
            print(f"plan-final-assembly: FAILED — {reason}", file=sys.stderr)
        return 1

    try:
        try:
            conn = get_readonly_connection()
        except sqlite3.Error:
            return _fail("no local project database found")

        try:
            project = get_project(conn, args.project_id)
            if project is None:
                return _fail(f"no project found with project_id {args.project_id!r}")

            artifacts = list_artifacts_by_project(conn, args.project_id)
        finally:
            conn.close()

        try:
            manifest = load_manifest(Path(args.manifest))
        except ManifestStoreError as exc:
            return _fail(str(exc))

        try:
            plan = build_final_assembly_plan(project, manifest, artifacts)
        except FinalAssemblyPlannerError as exc:
            return _fail(str(exc))
    except Exception:
        return _fail("an unexpected internal error occurred")

    if out_format == "json":
        payload = {
            "ok": True,
            "project_id": plan.project_id,
            "manifest_project_id_matches": plan.manifest_project_id_matches,
            "manifest_fingerprint_matches": plan.manifest_fingerprint_matches,
            "render_artifact_id": plan.render_artifact_id,
            "render_ready": plan.render_ready,
            "render_duration_seconds": plan.render_duration_seconds,
            "overlay_count": plan.overlay_count,
            "manifest_has_explicit_overlays": plan.manifest_has_explicit_overlays,
            "overlay_render_artifact_id": plan.overlay_render_artifact_id,
            "overlay_render_ready": plan.overlay_render_ready,
            "final_viewer_output_kind": plan.final_viewer_output_kind,
            "final_viewer_output_relative_path": plan.final_viewer_output_relative_path,
            "next_safe_local_command": plan.next_safe_local_command,
            "next_command_requires_explicit_paths": plan.next_command_requires_explicit_paths,
            "blocked_reasons": list(plan.blocked_reasons),
            "notes": list(plan.notes),
        }
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
    else:
        print("plan-final-assembly: OK (read-only planning/reporting only)")
        print(f"  project_id: {plan.project_id}")
        print(f"  manifest_project_id_matches: {plan.manifest_project_id_matches}")
        print(f"  manifest_fingerprint_matches: {plan.manifest_fingerprint_matches}")
        print(f"  render_artifact_id: {plan.render_artifact_id}")
        print(f"  render_ready: {plan.render_ready}")
        print(f"  render_duration_seconds: {plan.render_duration_seconds}")
        print(f"  overlay_count: {plan.overlay_count}")
        print(f"  manifest_has_explicit_overlays: {plan.manifest_has_explicit_overlays}")
        print(f"  overlay_render_artifact_id: {plan.overlay_render_artifact_id}")
        print(f"  overlay_render_ready: {plan.overlay_render_ready}")
        print(f"  final_viewer_output_kind: {plan.final_viewer_output_kind}")
        print(f"  final_viewer_output_relative_path: {plan.final_viewer_output_relative_path}")
        print(f"  next_safe_local_command: {plan.next_safe_local_command}")
        print(f"  next_command_requires_explicit_paths: {plan.next_command_requires_explicit_paths}")
        print("  blocked_reasons:")
        for reason in plan.blocked_reasons:
            print(f"    - {reason}")
        print("  notes:")
        for note in plan.notes:
            print(f"    - {note}")
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

    generate_scenes = sub.add_parser(
        "generate-scenes",
        help="Generate a validated ScenePlan from a local StoryInput JSON file via GroqProvider "
        "(TokenRouterProvider fallback) — no project, manifest, artifact, or stage transition is "
        "created; with --output the plan is saved as JSON, otherwise only a summary is printed",
    )
    generate_scenes.add_argument("--story", required=True, help="Path to the local StoryInput JSON file")
    generate_scenes.add_argument(
        "--output",
        default=None,
        help="Optional path to save the generated ScenePlan JSON (default: print a summary only)",
    )
    generate_scenes.set_defaults(func=cmd_generate_scenes)

    finalize_scene_timing = sub.add_parser(
        "finalize-scene-timing",
        help="Populate each scene's measured_audio_duration_seconds from already-registered "
        "'audio' artifacts (Phase 2D) into a copy of the project's manifest, saved to --output; "
        "all-or-nothing across every scene, no database write, no provider/ffprobe call",
    )
    finalize_scene_timing.add_argument("project_id")
    finalize_scene_timing.add_argument(
        "--output", required=True, help="Path to save the enriched VideoManifest JSON"
    )
    finalize_scene_timing.set_defaults(func=cmd_finalize_scene_timing)

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

    dry_run = sub.add_parser(
        "dry-run", help="Read-only execution-plan inspection for an existing project; writes nothing"
    )
    dry_run.add_argument("project_id")
    dry_run.add_argument("--format", default="text", help="Output format: text (default) or json")
    dry_run.set_defaults(func=cmd_dry_run)

    verify_artifacts = sub.add_parser(
        "verify-artifacts",
        help="Read-only verification of an existing project's registered artifacts; writes nothing",
    )
    verify_artifacts.add_argument("project_id")
    verify_artifacts.add_argument(
        "--kind", choices=_ARTIFACT_KINDS, default=None, help="Only verify artifacts of this kind"
    )
    verify_artifacts.add_argument("--format", default="text", help="Output format: text (default) or json")
    verify_artifacts.set_defaults(func=cmd_verify_artifacts)

    verify_and_advance = sub.add_parser(
        "verify-and-advance",
        help="Verify required local artifacts, then perform exactly one guarded transition into a "
        "verification-required stage; writes to SQLite only on success",
    )
    verify_and_advance.add_argument("project_id")
    verify_and_advance.add_argument(
        "--to",
        required=True,
        choices=sorted(SUPPORTED_TARGET_STAGES),
        help="Target lifecycle stage to verify and advance into",
    )
    verify_and_advance.add_argument("--reason", required=True, help="Explicit reason for this transition")
    verify_and_advance.add_argument("--format", default="text", help="Output format: text (default) or json")
    verify_and_advance.set_defaults(func=cmd_verify_and_advance)

    register_audio_artifact = sub.add_parser(
        "register-audio-artifact",
        help="Register a locally produced audio file as one scene's canonical audio artifact; "
        "writes to SQLite (and copies the file) only on a fresh registration",
    )
    register_audio_artifact.add_argument("project_id")
    register_audio_artifact.add_argument("--scene-id", required=True, help="Scene this audio artifact belongs to")
    register_audio_artifact.add_argument("--file", required=True, help="Path to the local audio file to register")
    register_audio_artifact.add_argument("--format", default="text", help="Output format: text (default) or json")
    register_audio_artifact.set_defaults(func=cmd_register_audio_artifact)

    register_visual_artifact = sub.add_parser(
        "register-visual-artifact",
        help="Register a locally produced PNG image as one scene's canonical visual artifact; "
        "writes to SQLite (and copies the file) only on a fresh registration",
    )
    register_visual_artifact.add_argument("project_id")
    register_visual_artifact.add_argument("--scene-id", required=True, help="Scene this visual artifact belongs to")
    register_visual_artifact.add_argument("--file", required=True, help="Path to the local PNG image to register")
    register_visual_artifact.add_argument("--format", default="text", help="Output format: text (default) or json")
    register_visual_artifact.set_defaults(func=cmd_register_visual_artifact)

    register_animation_artifact = sub.add_parser(
        "register-animation-artifact",
        help="Register a locally produced MP4 video as one scene's canonical animation artifact; "
        "writes to SQLite (and copies the file) only on a fresh registration",
    )
    register_animation_artifact.add_argument("project_id")
    register_animation_artifact.add_argument(
        "--scene-id", required=True, help="Scene this animation artifact belongs to"
    )
    register_animation_artifact.add_argument("--file", required=True, help="Path to the local MP4 video to register")
    register_animation_artifact.add_argument("--format", default="text", help="Output format: text (default) or json")
    register_animation_artifact.set_defaults(func=cmd_register_animation_artifact)

    register_render_artifact = sub.add_parser(
        "register-render-artifact",
        help="Register a locally produced MP4 video as this project's canonical render artifact "
        "(project-level, no --scene-id); writes to SQLite (and copies the file) only on a fresh registration",
    )
    register_render_artifact.add_argument("project_id")
    register_render_artifact.add_argument("--file", required=True, help="Path to the local MP4 video to register")
    register_render_artifact.add_argument("--format", default="text", help="Output format: text (default) or json")
    register_render_artifact.set_defaults(func=cmd_register_render_artifact)

    register_qc_report_artifact = sub.add_parser(
        "register-qc-report-artifact",
        help="Register a locally produced JSON QC report as this project's canonical qc_report "
        "artifact (project-level, no --scene-id); passed:true or passed:false are both accepted and "
        "registered — only verify-and-advance(to=qc_passed) gates on the value; writes to SQLite "
        "(and copies the file) only on a fresh registration",
    )
    register_qc_report_artifact.add_argument("project_id")
    register_qc_report_artifact.add_argument(
        "--file", required=True, help="Path to the local QC report JSON file to register"
    )
    register_qc_report_artifact.add_argument(
        "--format", default="text", help="Output format: text (default) or json"
    )
    register_qc_report_artifact.set_defaults(func=cmd_register_qc_report_artifact)

    advance_project_stage = sub.add_parser(
        "advance-project-stage",
        help="Perform exactly one guarded, non-verification transition (no completed/measured-work "
        "claim) — audio_ready, visuals_ready, animation_ready, rendered, qc_passed, completed, "
        "archived, and failed are all rejected here; use verify-and-advance for those",
    )
    advance_project_stage.add_argument("project_id")
    advance_project_stage.add_argument(
        "--to",
        required=True,
        choices=sorted(NON_VERIFICATION_TARGET_STAGES),
        help="Target non-verification lifecycle stage to advance into",
    )
    advance_project_stage.add_argument(
        "--reason", default=None, help="Optional note for this transition's audit trail"
    )
    advance_project_stage.add_argument("--format", default="text", help="Output format: text (default) or json")
    advance_project_stage.set_defaults(func=cmd_advance_project_stage)

    build_upscaled_ken_burns = sub.add_parser(
        "build-upscaled-ken-burns",
        help="Generation-only Phase 1D slice: build a local, upscale-assisted Ken Burns MP4 clip for "
        "one scene from its already-registered visual artifact and the enriched manifest's measured "
        "audio duration; writes only the --output MP4 — no SQLite write, no artifact registration, "
        "no manifest change, no lifecycle transition; register the result separately with "
        "register-animation-artifact",
    )
    build_upscaled_ken_burns.add_argument("project_id")
    build_upscaled_ken_burns.add_argument("scene_id")
    build_upscaled_ken_burns.add_argument(
        "--manifest",
        required=True,
        help="Path to the enriched VideoManifest JSON (e.g. finalize-scene-timing's --output)",
    )
    build_upscaled_ken_burns.add_argument(
        "--output", required=True, help="Path to save the generated MP4 clip; must not already exist"
    )
    build_upscaled_ken_burns.add_argument("--format", default="text", help="Output format: text (default) or json")
    build_upscaled_ken_burns.set_defaults(func=cmd_build_upscaled_ken_burns)

    preview_scene_image_prompt = sub.add_parser(
        "preview-scene-image-prompt",
        help="Read-only: preview the exact scene-image generation prompt for one scene without "
        "calling any image provider; loads visual_brief from the existing project/manifest records "
        "and applies build_scene_image_prompt(); no generation, no download, no file write, no "
        "SQLite write, no artifact registration, no lifecycle transition",
    )
    preview_scene_image_prompt.add_argument("project_id")
    preview_scene_image_prompt.add_argument("scene_id")
    preview_scene_image_prompt.add_argument(
        "--manifest", required=True, help="Path to the manifest JSON identifying this project/scene"
    )
    preview_scene_image_prompt.add_argument(
        "--reference-image",
        required=True,
        help="Local path to the reference image this preview assumes; existence only checked, never opened",
    )
    preview_scene_image_prompt.add_argument(
        "--output",
        required=True,
        help="Planned future output path; must not already exist (never created by this command)",
    )
    preview_scene_image_prompt.add_argument(
        "--include-character",
        required=True,
        choices=["true", "false"],
        help="Whether to append CHARACTER_ANCHOR to the prompt; never inferred automatically",
    )
    preview_scene_image_prompt.add_argument(
        "--include-color-anchor",
        default="true",
        choices=["true", "false"],
        help="Whether to append COLOR_ANCHOR to the prompt (default: true)",
    )
    preview_scene_image_prompt.add_argument(
        "--format", default="text", help="Output format: text (default) or json"
    )
    preview_scene_image_prompt.set_defaults(func=cmd_preview_scene_image_prompt)

    build_scene_image = sub.add_parser(
        "build-scene-image",
        help="Generation-only: build one scene's image via "
        "QwenImageProvider.generate_with_reference_and_download(), guarded by a fixed "
        "is_paid=True require_paid_approval('qwen-image', ...) check; writes only the --output "
        "PNG — no SQLite write, no artifact registration, no manifest change, no lifecycle "
        "transition; register the result separately with register-visual-artifact",
    )
    build_scene_image.add_argument("project_id")
    build_scene_image.add_argument("scene_id")
    build_scene_image.add_argument(
        "--manifest", required=True, help="Path to the manifest JSON identifying this project/scene"
    )
    build_scene_image.add_argument(
        "--reference-image",
        required=True,
        help="Local path to the reference image conditioning generation; must exist and be a regular file",
    )
    build_scene_image.add_argument(
        "--output", required=True, help="Path to save the generated PNG; must not already exist"
    )
    build_scene_image.add_argument(
        "--include-character",
        required=True,
        choices=["true", "false"],
        help="Whether to append CHARACTER_ANCHOR to the prompt; never inferred automatically",
    )
    build_scene_image.add_argument(
        "--include-color-anchor",
        default="true",
        choices=["true", "false"],
        help="Whether to append COLOR_ANCHOR to the prompt (default: true)",
    )
    build_scene_image.add_argument(
        "--proposals-path",
        default=None,
        help="Path to the JSON paid-proposal records; defaults to <data_dir>/paid_proposals.json",
    )
    build_scene_image.add_argument(
        "--format", default="text", help="Output format: text (default) or json"
    )
    build_scene_image.set_defaults(func=cmd_build_scene_image)

    build_scene_audio = sub.add_parser(
        "build-scene-audio",
        help="Generation-only: build one scene's narration audio via KokoroProvider.synthesize() "
        "(self-hosted, local, no network), guarded by a fixed is_paid=False "
        "require_paid_approval('kokoro', ...) check; writes only the --output WAV — no SQLite "
        "write, no artifact registration, no manifest change, no lifecycle transition; register "
        "the result separately with register-audio-artifact",
    )
    build_scene_audio.add_argument("project_id")
    build_scene_audio.add_argument("scene_id")
    build_scene_audio.add_argument(
        "--manifest", required=True, help="Path to the manifest JSON identifying this project/scene"
    )
    build_scene_audio.add_argument(
        "--output", required=True, help="Path to save the generated WAV; must not already exist"
    )
    build_scene_audio.add_argument(
        "--voice", default=None, help="Kokoro voice id; defaults to tts_kokoro.DEFAULT_VOICE"
    )
    build_scene_audio.add_argument(
        "--speed", default=None, help="Speech speed multiplier; defaults to tts_kokoro.DOCUMENTARY_SPEED"
    )
    build_scene_audio.add_argument(
        "--sentence-pause",
        default=None,
        help="Pause duration between sentences; defaults to tts_kokoro.DOCUMENTARY_SENTENCE_PAUSE",
    )
    build_scene_audio.add_argument(
        "--clause-pause",
        default=None,
        help="Pause duration between clauses; defaults to tts_kokoro.DOCUMENTARY_CLAUSE_PAUSE",
    )
    build_scene_audio.add_argument(
        "--format", default="text", help="Output format: text (default) or json"
    )
    build_scene_audio.set_defaults(func=cmd_build_scene_audio)

    assemble_final_video = sub.add_parser(
        "assemble-final-video",
        help="Local-only: assemble one project's already-registered per-scene animation+audio "
        "artifacts, in manifest order, into one final MP4 at --output, and register it as this "
        "project's canonical render artifact; no provider, no network call",
    )
    assemble_final_video.add_argument("project_id")
    assemble_final_video.add_argument(
        "--manifest", required=True, help="Path to the enriched VideoManifest JSON"
    )
    assemble_final_video.add_argument(
        "--output", required=True, help="Path to save the final assembled MP4; must not already exist"
    )
    assemble_final_video.add_argument(
        "--format", default="text", help="Output format: text (default) or json"
    )
    assemble_final_video.set_defaults(func=cmd_assemble_final_video)

    render_text_overlays = sub.add_parser(
        "render-text-overlays",
        help="Local-only: burn every manifest-declared TextOverlay into one project's already-"
        "registered render artifact and register the result as a new, distinct overlay_render "
        "artifact; no provider, no network call",
    )
    render_text_overlays.add_argument("project_id")
    render_text_overlays.add_argument(
        "--manifest", required=True, help="Path to the enriched VideoManifest JSON"
    )
    render_text_overlays.add_argument(
        "--output", required=True, help="Path to save the overlay-burned MP4; must not already exist"
    )
    render_text_overlays.add_argument(
        "--format", default="text", help="Output format: text (default) or json"
    )
    render_text_overlays.set_defaults(func=cmd_render_text_overlays)

    derive_text_overlays = sub.add_parser(
        "derive-text-overlays",
        help="Local-only: write a new manifest copy in which every scene with no explicit "
        "text_overlays receives exactly one deterministically-derived TextOverlay (narration "
        "prefix + cumulative measured timing); requires finalize-scene-timing and an "
        "already-registered render artifact; no provider, no FFmpeg, no network call",
    )
    derive_text_overlays.add_argument("project_id")
    derive_text_overlays.add_argument(
        "--manifest", required=True, help="Path to the finalize-scene-timing'd VideoManifest JSON"
    )
    derive_text_overlays.add_argument(
        "--output", required=True, help="Path to save the derived VideoManifest JSON; must not already exist"
    )
    derive_text_overlays.add_argument(
        "--format", default="text", help="Output format: text (default) or json"
    )
    derive_text_overlays.set_defaults(func=cmd_derive_text_overlays)

    plan_final_assembly = sub.add_parser(
        "plan-final-assembly",
        help="Local-only, read-only readiness report for the render -> overlay-derivation -> "
        "overlay-rendering tail of the pipeline; never calls assemble-final-video/"
        "derive-text-overlays/render-text-overlays itself, no provider, no FFmpeg, no network call",
    )
    plan_final_assembly.add_argument("project_id")
    plan_final_assembly.add_argument(
        "--manifest", required=True, help="Path to the VideoManifest JSON to plan against"
    )
    plan_final_assembly.add_argument(
        "--format", default="text", help="Output format: text (default) or json"
    )
    plan_final_assembly.set_defaults(func=cmd_plan_final_assembly)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
