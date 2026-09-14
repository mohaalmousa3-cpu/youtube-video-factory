"""Phase 1C: local filesystem persistence for a VideoManifest.

JSON only, UTF-8, no database access, no provider calls. Saving is atomic:
write to a temp file in the destination's own directory, then os.replace()
it into place, so a concurrent reader never observes a partially-written
file. No automatic output-directory selection based on the user's home
directory or similar — callers (and tests) always pass an explicit path."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from pydantic import ValidationError

from src.models.manifest import VideoManifest


class ManifestStoreError(Exception):
    """Raised for any problem saving or loading a VideoManifest: a missing
    file, a permission or other OS-level read/write failure, invalid
    UTF-8, invalid JSON, a non-object JSON document, or a document that
    fails VideoManifest validation."""


def save_manifest(manifest: VideoManifest, path: Path) -> Path:
    """Write `manifest` to `path` as formatted, stable-key-order UTF-8
    JSON, atomically. Creates `path`'s parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = manifest.model_dump(mode="json")
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
        raise ManifestStoreError(f"could not save manifest to {path}: {exc}") from exc

    return path


def load_manifest(path: Path) -> VideoManifest:
    """Load and validate a VideoManifest from `path`. Any failure — missing
    file, an OS-level read error, invalid UTF-8, invalid JSON, a
    non-object document, or a document that fails VideoManifest
    validation — raises ManifestStoreError."""
    path = Path(path)

    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ManifestStoreError(f"manifest not found at {path}") from exc
    except PermissionError as exc:
        raise ManifestStoreError(f"permission denied reading manifest at {path}") from exc
    except UnicodeDecodeError as exc:
        raise ManifestStoreError(f"manifest at {path} is not valid UTF-8 text: {exc}") from exc
    except OSError as exc:
        raise ManifestStoreError(f"could not read manifest at {path}: {exc}") from exc

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ManifestStoreError(f"invalid JSON in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ManifestStoreError(
            f"{path} must contain a JSON object at the top level, got {type(raw).__name__}"
        )

    try:
        return VideoManifest.model_validate(raw)
    except ValidationError as exc:
        raise ManifestStoreError(f"invalid manifest at {path}: {exc}") from exc
