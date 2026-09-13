"""Phase 1E: local file loaders, single-project creation, and all-or-
nothing Story Queue import.

No provider calls, no network access — every function here only reads
local JSON files and writes local manifest files + local SQLite rows,
via the existing Phase 1C (manifest_builder/manifest_store) and Phase 1D
(project_state_machine/project_repository) building blocks. This module
does not itself decide policy — it just wires already-validated local
input through those existing layers.

Error handling: three exception types, one per concern —
- InputFileError: a single story/scene JSON file failed to load or
  validate (missing, bad encoding, bad JSON, non-object, or a
  StoryInput/ScenePlan validation failure).
- ProjectCreationError: creating ONE project (via create_project_from_inputs)
  failed — an existing manifest path or database project_id conflict, or a
  database write failure after the manifest was already saved.
- QueueImportError: a problem with the queue file itself, or with the
  batch import as a whole. Per-entry InputFileError/ManifestValidationError/
  ManifestStoreError failures are caught and re-raised as QueueImportError
  naming the failing queue_item_id, so a CLI caller only ever needs to
  catch one exception type for the whole "import-queue" command.

All-or-nothing import, honestly: prepare_queue_import() validates every
enabled entry and builds every manifest IN MEMORY, checking for every
conflict (duplicate project_id within the batch, an existing database
project_id, an existing manifest file) before anything is written.
import_queue() only then writes. If a write fails partway through, it
rolls back by deleting the database rows this import attempt already
created (project_repository.delete_project) and best-effort removes their
manifest files/newly-created directories. This is a compensating cleanup,
not a real cross-file transaction — a crash between the DB delete and the
manifest-file removal (or between removing the manifest file and its now-
empty directory) can still leave a partial trace on disk. That residual
risk is accepted and documented rather than pretended away; it only
matters if the process is killed at the exact moment of rollback, and the
leftover in that case is inert (an orphaned manifest.json with no matching
database row, never an inconsistent-but-live project)."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.core.manifest_builder import ManifestValidationError, build_video_manifest
from src.core.manifest_store import ManifestStoreError, save_manifest
from src.core.project_state_machine import create_initial_project, initial_transition_for
from src.database.project_repository import (
    ProjectAlreadyExistsError,
    create_project,
    delete_project,
    get_project,
)
from src.models.manifest import VideoManifest
from src.models.queue import QueueImport, QueueItem
from src.models.scene import ScenePlan
from src.models.story import StoryInput
from src.models.project_state import ProjectRecord
from src.utils.channel_config import ChannelPolicy


class InputFileError(Exception):
    """Raised when a local story or scene input JSON file fails to load
    or validate — missing file, a permission or other OS-level read
    failure, invalid UTF-8, invalid JSON, a non-object document, or a
    StoryInput/ScenePlan validation failure."""


class ProjectCreationError(Exception):
    """Raised by create_project_from_inputs() when creating ONE project
    fails: an existing manifest file (never overwritten), an existing
    database project_id, or a database write failure after the manifest
    was already saved (in which case this exception's message reports
    whether the resulting manifest file/directory could be cleaned up)."""


class QueueImportError(Exception):
    """Raised for any problem with the queue file itself (missing/invalid
    JSON, schema violation, duplicate queue_item_id, a path escaping the
    queue directory) or with a batch import as a whole (a duplicate
    project_id within the batch, a conflict against the database or
    filesystem, or a write failure partway through — see this module's
    docstring for the rollback this triggers)."""


# ---------------------------------------------------------------------
# Local JSON file loading
# ---------------------------------------------------------------------


def _load_json_object(path: Path, *, what: str) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise InputFileError(f"{what} not found at {path}") from exc
    except PermissionError as exc:
        raise InputFileError(f"permission denied reading {what} at {path}") from exc
    except UnicodeDecodeError as exc:
        raise InputFileError(f"{what} at {path} is not valid UTF-8 text: {exc}") from exc
    except OSError as exc:
        raise InputFileError(f"could not read {what} at {path}: {exc}") from exc

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InputFileError(f"invalid JSON in {what} at {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise InputFileError(
            f"{what} at {path} must be a JSON object at the top level, got {type(raw).__name__}"
        )
    return raw


def load_story_input_file(path: Path) -> StoryInput:
    raw = _load_json_object(path, what="story input")
    try:
        return StoryInput.model_validate(raw)
    except ValidationError as exc:
        raise InputFileError(f"invalid story input at {path}: {exc}") from exc


def load_scene_plan_file(path: Path) -> ScenePlan:
    raw = _load_json_object(path, what="scene plan")
    try:
        return ScenePlan.model_validate(raw)
    except ValidationError as exc:
        raise InputFileError(f"invalid scene plan at {path}: {exc}") from exc


def load_queue_import(path: Path) -> QueueImport:
    raw = _load_json_object(path, what="queue import file")
    try:
        return QueueImport.model_validate(raw)
    except ValidationError as exc:
        raise QueueImportError(f"invalid queue import file at {path}: {exc}") from exc


# ---------------------------------------------------------------------
# Path resolution / traversal protection
# ---------------------------------------------------------------------


def _resolve_within(
    base_dir: Path,
    raw_path: str,
    *,
    what: str,
    error_cls: type[Exception] = QueueImportError,
) -> Path:
    """Resolve `raw_path` relative to `base_dir` and reject it if the
    resolved path escapes `base_dir` (via `../`, an absolute path, or a
    symlink) — no override exists in Phase 1E to opt out of this check.

    This is the ONE path-safety helper for every "does this stay inside
    that directory" check in this module: queue item story_input_path/
    scene_plan_path against the queue file's own directory,
    output_subdirectory against --output-root, AND the default
    <output-root>/<project_id> directory a single create-project call
    uses — the latter two both call this with `error_cls=ProjectCreationError`
    or the default `QueueImportError` as appropriate to the caller, so a
    single-project create and a queue import apply exactly the same
    resolve-and-check logic to --output-root, not two independently
    written (and possibly inconsistent) checks."""
    base_resolved = base_dir.resolve()
    candidate = (base_dir / raw_path).resolve()
    try:
        candidate.relative_to(base_resolved)
    except ValueError as exc:
        raise error_cls(
            f"{what} {raw_path!r} resolves to {candidate}, which is outside "
            f"{base_resolved} — this is not allowed"
        ) from exc
    return candidate


# ---------------------------------------------------------------------
# Single-project creation (used by the `create-project` CLI command)
# ---------------------------------------------------------------------


def create_project_from_inputs(
    conn: sqlite3.Connection,
    story_input: StoryInput,
    scene_plan: ScenePlan,
    channel_policy: ChannelPolicy,
    output_root: Path,
    *,
    now: datetime | None = None,
) -> ProjectRecord:
    """Build a deterministic VideoManifest, save it under
    <output_root>/<project_id>/manifest.json, and create the initial
    ProjectRecord + ProjectTransition in SQLite. Raises
    ManifestValidationError if story_input/scene_plan/channel_policy are
    incompatible (nothing is written yet at that point), or
    ProjectCreationError if the manifest path or database project_id
    already exists (never overwritten) or the database write fails after
    the manifest was saved (best-effort cleanup attempted — see this
    module's docstring)."""
    manifest = build_video_manifest(story_input, scene_plan, channel_policy, created_at=now)

    manifest_dir = _resolve_within(
        output_root,
        manifest.project_id,
        what="output_root/project_id directory",
        error_cls=ProjectCreationError,
    )
    manifest_path = manifest_dir / "manifest.json"
    if manifest_path.exists():
        raise ProjectCreationError(
            f"manifest already exists at {manifest_path} — refusing to overwrite"
        )
    if get_project(conn, manifest.project_id) is not None:
        raise ProjectCreationError(
            f"project {manifest.project_id!r} already exists in the database"
        )

    project_dir_existed_before = manifest_path.parent.exists()
    try:
        save_manifest(manifest, manifest_path)
    except ManifestStoreError:
        raise  # nothing to clean up: save_manifest's own atomic write never left a partial file

    project = create_initial_project(manifest_path, manifest, now=now)
    transition = initial_transition_for(project)
    try:
        create_project(conn, project, transition)
    except (ProjectAlreadyExistsError, sqlite3.Error) as exc:
        cleanup_error = _cleanup_manifest(manifest_path, remove_dir=not project_dir_existed_before)
        message = (
            f"database write failed for project {manifest.project_id!r} after saving its "
            f"manifest: {exc}"
        )
        if cleanup_error is not None:
            message += (
                f"; additionally, cleanup of {manifest_path} did not fully complete: "
                f"{cleanup_error} — a manifest file may remain on disk without a matching "
                "database record"
            )
        raise ProjectCreationError(message) from exc

    return project


def _cleanup_manifest(manifest_path: Path, *, remove_dir: bool) -> OSError | None:
    """Best-effort removal of a just-written manifest file (and, if we
    created it fresh this call, its now-presumably-empty parent
    directory). Returns the OSError encountered, if any, rather than
    raising — cleanup failure is reported to the caller, never masks the
    original error that triggered it."""
    try:
        manifest_path.unlink(missing_ok=True)
    except OSError as exc:
        return exc
    if remove_dir:
        try:
            manifest_path.parent.rmdir()
        except OSError:
            pass  # not empty, or already gone — nothing more we can safely do
    return None


# ---------------------------------------------------------------------
# Queue import (used by the `import-queue` CLI command)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class PreparedQueueEntry:
    item: QueueItem
    manifest: VideoManifest
    manifest_path: Path
    # Whether manifest_path's parent directory already existed before this
    # import — so rollback only ever removes a directory THIS import
    # created, never one that happened to already be there (matching
    # create_project_from_inputs's same precaution).
    manifest_dir_existed_before: bool


@dataclass(frozen=True)
class QueueImportPlan:
    prepared: tuple[PreparedQueueEntry, ...]
    skipped_disabled: tuple[str, ...]

    def summary_lines(self) -> list[str]:
        lines = [f"{len(self.prepared)} entries would be created:"]
        for entry in self.prepared:
            lines.append(f"  {entry.item.queue_item_id} -> project_id {entry.manifest.project_id}")
        lines.append(f"{len(self.skipped_disabled)} disabled entries would be skipped:")
        for queue_item_id in self.skipped_disabled:
            lines.append(f"  {queue_item_id}")
        return lines


@dataclass(frozen=True)
class QueueImportResult:
    created: tuple[ProjectRecord, ...]
    skipped_disabled: tuple[str, ...]


def prepare_queue_import(
    queue: QueueImport,
    queue_dir: Path,
    channel_policy: ChannelPolicy,
    output_root: Path,
    conn: sqlite3.Connection,
    *,
    now: datetime | None = None,
) -> QueueImportPlan:
    """Phase 1 of the all-or-nothing import: validate every ENABLED entry
    and build every manifest in memory, checking every conflict (duplicate
    project_id within this batch, an existing database project_id, an
    existing manifest file). Performs NO writes — safe to call for a
    dry-run summary. Raises QueueImportError on the first problem found,
    naming the offending queue_item_id."""
    enabled_items = [item for item in queue.items if item.enabled]
    disabled_ids = tuple(item.queue_item_id for item in queue.items if not item.enabled)

    prepared: list[PreparedQueueEntry] = []
    seen_project_ids: dict[str, str] = {}

    for item in enabled_items:
        story_path = _resolve_within(
            queue_dir, item.story_input_path, what=f"queue item {item.queue_item_id!r} story_input_path"
        )
        scene_path = _resolve_within(
            queue_dir, item.scene_plan_path, what=f"queue item {item.queue_item_id!r} scene_plan_path"
        )

        try:
            story_input = load_story_input_file(story_path)
            scene_plan = load_scene_plan_file(scene_path)
            manifest = build_video_manifest(story_input, scene_plan, channel_policy, created_at=now)
        except (InputFileError, ManifestValidationError) as exc:
            raise QueueImportError(f"queue item {item.queue_item_id!r}: {exc}") from exc

        if manifest.project_id in seen_project_ids:
            raise QueueImportError(
                f"queue item {item.queue_item_id!r} produces project_id "
                f"{manifest.project_id!r}, already produced by queue item "
                f"{seen_project_ids[manifest.project_id]!r} within this same import"
            )
        seen_project_ids[manifest.project_id] = item.queue_item_id

        if get_project(conn, manifest.project_id) is not None:
            raise QueueImportError(
                f"queue item {item.queue_item_id!r} produces project_id "
                f"{manifest.project_id!r}, which already exists in the database"
            )

        # The default (no explicit output_subdirectory) and the explicit
        # override both go through the SAME _resolve_within() check against
        # --output-root — one path-safety mechanism, not two independently
        # written ones. project_id can never actually contain `../` or an
        # absolute path (it's always "proj-" + a hex fingerprint prefix),
        # so the default branch was never exploitable, but it now applies
        # the identical resolve-and-check logic as the explicit-override
        # branch and as create_project_from_inputs's single-project path,
        # for one auditable mechanism instead of two.
        subdirectory = item.output_subdirectory if item.output_subdirectory is not None else manifest.project_id
        manifest_dir = _resolve_within(
            output_root,
            subdirectory,
            what=f"queue item {item.queue_item_id!r} output directory",
        )
        manifest_path = manifest_dir / "manifest.json"

        if manifest_path.exists():
            raise QueueImportError(
                f"queue item {item.queue_item_id!r}: manifest already exists at "
                f"{manifest_path} — refusing to overwrite"
            )

        prepared.append(
            PreparedQueueEntry(
                item=item,
                manifest=manifest,
                manifest_path=manifest_path,
                manifest_dir_existed_before=manifest_path.parent.exists(),
            )
        )

    return QueueImportPlan(prepared=tuple(prepared), skipped_disabled=disabled_ids)


def import_queue(
    conn: sqlite3.Connection,
    queue: QueueImport,
    queue_dir: Path,
    channel_policy: ChannelPolicy,
    output_root: Path,
    *,
    now: datetime | None = None,
) -> QueueImportResult:
    """Phase 2: validate (via prepare_queue_import) then actually write.
    If any write fails partway through, rolls back every database row
    this call already created and best-effort removes their manifest
    files, then raises QueueImportError — see this module's docstring for
    the honest limits of that rollback."""
    plan = prepare_queue_import(queue, queue_dir, channel_policy, output_root, conn, now=now)

    created: list[ProjectRecord] = []
    try:
        for entry in plan.prepared:
            save_manifest(entry.manifest, entry.manifest_path)
            project = create_initial_project(entry.manifest_path, entry.manifest, now=now)
            transition = initial_transition_for(project)
            create_project(conn, project, transition)
            created.append(project)
    except Exception as exc:
        _rollback_partial_import(conn, created, plan.prepared)
        raise QueueImportError(
            f"import failed after creating {len(created)} of {len(plan.prepared)} project(s): {exc}"
        ) from exc

    return QueueImportResult(created=tuple(created), skipped_disabled=plan.skipped_disabled)


def _rollback_partial_import(
    conn: sqlite3.Connection,
    created: list[ProjectRecord],
    prepared: tuple[PreparedQueueEntry, ...],
) -> None:
    entries_by_project_id = {entry.manifest.project_id: entry for entry in prepared}
    for project in created:
        try:
            delete_project(conn, project.project_id)
        except sqlite3.Error:
            pass  # best-effort — nothing more we can safely do from here
        entry = entries_by_project_id.get(project.project_id)
        if entry is not None:
            _cleanup_manifest(entry.manifest_path, remove_dir=not entry.manifest_dir_existed_before)
