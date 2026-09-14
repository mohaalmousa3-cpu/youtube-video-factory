"""Phase 2B: pure, local, read-only artifact verification.

Checks a registered ArtifactRecord against what is actually on disk
(existence, regular-file status, byte size, SHA-256 checksum) and against
the project's own VideoManifest (project_id match, scene_id membership).
Never writes to the database, the filesystem, or a ProjectRecord's
lifecycle state, and never advances a project's stage on its own — it only
reads files (to stat and hash them) and returns structured
ArtifactVerificationResult values. A future orchestrator is expected to
feed a result's `passed` value into src/core/project_state_machine.py's
`transition_project(..., verified=...)`, but nothing in this module calls
that itself.

No provider calls, no network access."""
from __future__ import annotations

import hashlib
from pathlib import Path

from src.models.artifact import ArtifactRecord, ArtifactVerificationResult
from src.models.manifest import VideoManifest

_CHUNK_SIZE = 1024 * 1024


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_under_project_dir(project_dir_resolved: Path, relative_path: str) -> Path | None:
    """Resolve `relative_path` under `project_dir_resolved` (which the
    caller must already have `.resolve()`d — once per batch, not once per
    artifact, is the whole point of the amortization in verify_artifacts()
    below), following symlinks, and return None if the resolved path
    escapes it. ArtifactRecord's own validator already rejects an absolute
    path or '..' segment at construction time (a string-only check); this
    is the runtime, filesystem-aware check that additionally catches what
    that cannot: a symlink inside the project directory whose target lies
    outside it."""
    candidate = project_dir_resolved / relative_path
    try:
        resolved = candidate.resolve()
    except OSError:
        return None
    try:
        resolved.relative_to(project_dir_resolved)
    except ValueError:
        return None
    return resolved


def _verify_one(
    project_dir_resolved: Path,
    manifest_project_id: str,
    known_scene_ids: frozenset[str],
    artifact: ArtifactRecord,
) -> ArtifactVerificationResult:
    """Verify one artifact against pre-computed, batch-shared facts
    (the already-resolved project directory and the manifest's scene-id
    set) — see verify_artifact()/verify_artifacts() below, the two public
    entry points that compute those facts once and call this. Pure
    read-only inspection: at most opens the artifact file to stat and hash
    it — never writes, moves, or deletes anything, and never touches the
    database or project lifecycle state. Any OSError encountered while
    inspecting the file (permission denied, or the file changing/
    disappearing mid-check — a TOCTOU race between the existence check and
    the read) is caught and reported as a failed result, never allowed to
    propagate as a raw exception."""
    reasons: list[str] = []

    if artifact.project_id != manifest_project_id:
        reasons.append(
            f"artifact project_id {artifact.project_id!r} does not match "
            f"manifest project_id {manifest_project_id!r}"
        )

    if artifact.scene_id is not None and artifact.scene_id not in known_scene_ids:
        reasons.append(
            f"scene_id {artifact.scene_id!r} is not present in the project manifest"
        )

    resolved = _resolve_under_project_dir(project_dir_resolved, artifact.relative_path)
    if resolved is None:
        reasons.append(
            f"relative_path {artifact.relative_path!r} is unsafe: it resolves outside "
            f"the project directory {project_dir_resolved} (absolute path, traversal, or "
            "symlink escape)"
        )
    else:
        try:
            if not resolved.exists():
                reasons.append(f"file does not exist at {resolved}")
            elif not resolved.is_file():
                reasons.append(f"{resolved} is not a regular file (e.g. a directory)")
            else:
                actual_size = resolved.stat().st_size
                if actual_size != artifact.byte_size:
                    reasons.append(
                        f"byte_size mismatch: recorded {artifact.byte_size}, actual {actual_size}"
                    )
                actual_checksum = _sha256_of_file(resolved)
                if actual_checksum != artifact.sha256_checksum:
                    reasons.append(
                        f"sha256 checksum mismatch: recorded {artifact.sha256_checksum}, "
                        f"actual {actual_checksum}"
                    )
        except OSError as exc:
            # Covers a permission error on stat/open/read, and a TOCTOU
            # race (deleted/replaced between the exists() check above and
            # the read below) — concise and non-sensitive on purpose: the
            # exception type only, never the raw OS error text.
            reasons.append(
                f"could not inspect file at {resolved}: {exc.__class__.__name__} "
                "(permission denied, or the file changed during verification)"
            )

    return ArtifactVerificationResult(
        artifact_id=artifact.artifact_id,
        project_id=artifact.project_id,
        kind=artifact.kind,
        scene_id=artifact.scene_id,
        relative_path=artifact.relative_path,
        passed=not reasons,
        reasons=tuple(reasons),
    )


def verify_artifact(
    project_dir: Path,
    artifact: ArtifactRecord,
    manifest: VideoManifest,
) -> ArtifactVerificationResult:
    """Verify one artifact. Pure read-only inspection — see _verify_one()'s
    docstring for exactly what "read-only" covers, including OSError
    handling. For verifying more than one artifact against the same
    project_dir/manifest, prefer verify_artifacts() below, which computes
    the shared facts this function recomputes on every call just once."""
    known_scene_ids = frozenset(scene.scene_id for scene in manifest.scene_plan.scenes)
    return _verify_one(project_dir.resolve(), manifest.project_id, known_scene_ids, artifact)


def verify_artifacts(
    project_dir: Path,
    manifest: VideoManifest,
    artifacts: "list[ArtifactRecord] | tuple[ArtifactRecord, ...]",
) -> tuple[ArtifactVerificationResult, ...]:
    """Verify a batch of artifacts against the same project directory and
    manifest. Order-preserving. Resolves project_dir and computes the
    manifest's scene-id set exactly once for the whole batch (not once per
    artifact), then delegates each artifact to _verify_one() — the same
    safety checks and result semantics as calling verify_artifact() once
    per artifact, just without repeating the shared setup work."""
    project_dir_resolved = project_dir.resolve()
    known_scene_ids = frozenset(scene.scene_id for scene in manifest.scene_plan.scenes)
    return tuple(
        _verify_one(project_dir_resolved, manifest.project_id, known_scene_ids, artifact)
        for artifact in artifacts
    )
