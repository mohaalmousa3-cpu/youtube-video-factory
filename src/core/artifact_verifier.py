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


def _resolve_under_project_dir(project_dir: Path, relative_path: str) -> Path | None:
    """Resolve `relative_path` under `project_dir`, following symlinks, and
    return None if the resolved path escapes project_dir. ArtifactRecord's
    own validator already rejects an absolute path or '..' segment at
    construction time (a string-only check); this is the runtime,
    filesystem-aware check that additionally catches what that cannot: a
    symlink inside project_dir whose target lies outside it."""
    project_dir_resolved = project_dir.resolve()
    candidate = project_dir / relative_path
    try:
        resolved = candidate.resolve()
    except OSError:
        return None
    try:
        resolved.relative_to(project_dir_resolved)
    except ValueError:
        return None
    return resolved


def verify_artifact(
    project_dir: Path,
    artifact: ArtifactRecord,
    manifest: VideoManifest,
) -> ArtifactVerificationResult:
    """Verify one artifact. Pure read-only inspection: at most opens the
    artifact file to stat and hash it — never writes, moves, or deletes
    anything, and never touches the database or project lifecycle state."""
    reasons: list[str] = []

    if artifact.project_id != manifest.project_id:
        reasons.append(
            f"artifact project_id {artifact.project_id!r} does not match "
            f"manifest project_id {manifest.project_id!r}"
        )

    if artifact.scene_id is not None:
        known_scene_ids = {scene.scene_id for scene in manifest.scene_plan.scenes}
        if artifact.scene_id not in known_scene_ids:
            reasons.append(
                f"scene_id {artifact.scene_id!r} is not present in the project manifest"
            )

    resolved = _resolve_under_project_dir(project_dir, artifact.relative_path)
    if resolved is None:
        reasons.append(
            f"relative_path {artifact.relative_path!r} is unsafe: it resolves outside "
            f"the project directory {project_dir} (absolute path, traversal, or symlink escape)"
        )
    elif not resolved.exists():
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

    return ArtifactVerificationResult(
        artifact_id=artifact.artifact_id,
        project_id=artifact.project_id,
        kind=artifact.kind,
        scene_id=artifact.scene_id,
        relative_path=artifact.relative_path,
        passed=not reasons,
        reasons=tuple(reasons),
    )


def verify_artifacts(
    project_dir: Path,
    manifest: VideoManifest,
    artifacts: "list[ArtifactRecord] | tuple[ArtifactRecord, ...]",
) -> tuple[ArtifactVerificationResult, ...]:
    """Verify a batch of artifacts against the same project directory and
    manifest. Order-preserving; a pure convenience wrapper around
    verify_artifact()."""
    return tuple(verify_artifact(project_dir, artifact, manifest) for artifact in artifacts)
