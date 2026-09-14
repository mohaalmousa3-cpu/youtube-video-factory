"""Shared, filesystem-aware "does this path stay inside that directory"
check, used by both src/core/artifact_verifier.py (reading an already-
registered artifact) and src/core/audio_artifact_registrar.py (about to
write a freshly registered one). Extracted from artifact_verifier.py
without changing its behavior — see that module's history for the
original implementation this was lifted from verbatim."""
from __future__ import annotations

from pathlib import Path


def resolve_under_project_dir(base_dir_resolved: Path, relative_path: str) -> Path | None:
    """Resolve `relative_path` under `base_dir_resolved` (which the caller
    must already have `.resolve()`d), following symlinks, and return None
    if the resolved path escapes it. A model-level validator (e.g.
    ArtifactRecord's) may already reject an absolute path or '..' segment
    at construction time — that is a string-only check; this is the
    runtime, filesystem-aware check that additionally catches what that
    cannot: a symlink inside the base directory whose target lies outside
    it."""
    candidate = base_dir_resolved / relative_path
    try:
        resolved = candidate.resolve()
    except OSError:
        return None
    try:
        resolved.relative_to(base_dir_resolved)
    except ValueError:
        return None
    return resolved
