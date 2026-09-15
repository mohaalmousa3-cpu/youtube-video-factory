"""Phase 1C: local filesystem persistence for a generated ScenePlan.

JSON only, UTF-8, no database access, no provider calls. Mirrors
src/core/manifest_store.py's save_manifest(): atomic write (a temp file in
the destination's own directory, then os.replace()), so a concurrent
reader never observes a partially-written file. Save-only — nothing in
this Phase 1C change reads a saved scene plan back;
src/core/queue_import.py's existing load_scene_plan_file() already covers
that for a hand-authored or previously-saved scene-plan file, and is
reused as-is, not duplicated here."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from src.models.scene import ScenePlan


class ScenePlanStoreError(Exception):
    """Raised for any problem saving a ScenePlan: a permission or other
    OS-level write failure at the destination path."""


def save_scene_plan(scene_plan: ScenePlan, path: Path) -> Path:
    """Write `scene_plan` to `path` as formatted, stable-key-order UTF-8
    JSON, atomically. Creates `path`'s parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = scene_plan.model_dump(mode="json")
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"

    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_name, path)
    except OSError as exc:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise ScenePlanStoreError(f"could not save scene plan to {path}: {exc}") from exc

    return path
