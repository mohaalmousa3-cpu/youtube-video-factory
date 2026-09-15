"""Phase 1C: measured-audio timing finalization.

Bridges Phase 2D's already-registered, already-ffprobe-measured "audio"
ArtifactRecord metadata into a VideoManifest's per-scene
artifacts.measured_audio_duration_seconds, closing the "no wpm-estimated
placeholder durations persisted" half of docs/spec-v4/IMPLEMENTATION-PLAN.md's
Phase 1C (see docs/spec-v4/TECHNICAL-SPEC-EN.md section 7:
"Any manifest or scene record carrying a duration field must trace that
value to a measured audio file, not a table lookup or a wpm estimate").

Pure and local only: this module never opens SQLite, never reads a file,
never invokes ffprobe, never calls a provider/LLM/network endpoint, and
never mutates its inputs. It only reasons over a VideoManifest and a
sequence of already-fetched ArtifactRecord objects the caller supplies
(the CLI handler owns all I/O — see src/cli.py's cmd_finalize_scene_timing).

Scope, deliberately narrow: only
`manifest.scene_plan.scenes[*].artifacts.measured_audio_duration_seconds`
is ever populated. `VideoManifest.measured_audio_duration_seconds` (the
whole-video aggregate field) is left exactly as given — this module never
infers, sums, estimates, or otherwise claims a manifest-level duration;
that is explicitly out of scope for this change.

All-or-nothing: finalize_scene_timing() succeeds only when every single
scene in `manifest.scene_plan.scenes` has exactly one eligible audio
ArtifactRecord (kind="audio", matching project_id, matching scene_id) with
metadata["duration_seconds"] a real, finite, strictly-positive int/float
(bool excluded, since bool is an int subclass in Python) — otherwise it
raises TimingFinalizationError and returns nothing; there is no partial
result, no fallback estimate, and no silent skip.

Identity preservation: the returned VideoManifest is built exclusively via
model_copy(update=...) at every nested level (SceneArtifacts ->
ScenePlanItem -> ScenePlan -> VideoManifest) — never via
src/core/manifest_builder.py's build_video_manifest() or
_compute_fingerprint(), which would recompute source_fingerprint (and
therefore project_id) from the full scene_plan payload, including the
very artifacts field this module changes. project_id and
source_fingerprint are consequently never named in any update(...) dict
here, so model_copy leaves them byte-for-byte identical to the input.
Because model_copy() does not re-run pydantic validators, the final
result is re-validated via VideoManifest.model_validate() before being
returned, so an accidental nested-copy mistake cannot silently bypass
ScenePlan's own structural invariants (unique/contiguous scene IDs, valid
role_outfit references)."""
from __future__ import annotations

import math
from typing import Sequence

from src.models.artifact import ArtifactRecord
from src.models.manifest import VideoManifest


class TimingFinalizationError(Exception):
    """Raised for any reason finalize_scene_timing() cannot produce a
    fully, validly enriched VideoManifest: a scene with zero or more than
    one eligible registered audio artifact, an ineligible artifact (wrong
    kind or wrong project), or a duration_seconds metadata value that is
    missing, non-numeric, non-finite, or not strictly positive. Message
    text names only scene_id/artifact_id/kind/project_id/count — never a
    manifest path, generated content, or a raw exception representation."""


def _validate_duration(artifact: ArtifactRecord, scene_id: str) -> float:
    if "duration_seconds" not in artifact.metadata:
        raise TimingFinalizationError(
            f"artifact {artifact.artifact_id!r} for scene {scene_id!r} is missing a "
            "'duration_seconds' metadata value"
        )
    value = artifact.metadata["duration_seconds"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TimingFinalizationError(
            f"artifact {artifact.artifact_id!r} for scene {scene_id!r} has a non-numeric "
            "duration_seconds value"
        )
    if not math.isfinite(value):
        raise TimingFinalizationError(
            f"artifact {artifact.artifact_id!r} for scene {scene_id!r} has a non-finite "
            "duration_seconds value"
        )
    if value <= 0:
        raise TimingFinalizationError(
            f"artifact {artifact.artifact_id!r} for scene {scene_id!r} has a non-positive "
            "duration_seconds value"
        )
    return float(value)


def finalize_scene_timing(
    manifest: VideoManifest,
    audio_artifacts: Sequence[ArtifactRecord],
) -> VideoManifest:
    """Return a new VideoManifest with every scene's
    artifacts.measured_audio_duration_seconds populated from the matching
    entry in `audio_artifacts`. Raises TimingFinalizationError, with
    nothing returned, if any scene lacks exactly one eligible audio
    artifact or that artifact's duration_seconds metadata is invalid.
    Never touches SQLite, the filesystem, ffprobe, or any provider —
    `audio_artifacts` must already be the caller's own query result.
    Never mutates `manifest` or any entry in `audio_artifacts` — every
    input is treated as read-only."""
    by_scene: dict[str, list[ArtifactRecord]] = {}
    for artifact in audio_artifacts:
        if artifact.kind != "audio":
            raise TimingFinalizationError(
                f"artifact {artifact.artifact_id!r} is not an 'audio' artifact (kind={artifact.kind!r})"
            )
        if artifact.project_id != manifest.project_id:
            raise TimingFinalizationError(
                f"artifact {artifact.artifact_id!r} belongs to project {artifact.project_id!r}, "
                f"expected {manifest.project_id!r}"
            )
        by_scene.setdefault(artifact.scene_id, []).append(artifact)

    new_scenes = []
    for scene in manifest.scene_plan.scenes:
        matches = by_scene.get(scene.scene_id, [])
        if len(matches) == 0:
            raise TimingFinalizationError(f"scene {scene.scene_id!r} has no eligible registered audio artifact")
        if len(matches) > 1:
            raise TimingFinalizationError(
                f"scene {scene.scene_id!r} has {len(matches)} eligible registered audio artifacts "
                "(expected exactly 1)"
            )

        duration = _validate_duration(matches[0], scene.scene_id)
        new_artifacts = scene.artifacts.model_copy(update={"measured_audio_duration_seconds": duration})
        new_scenes.append(scene.model_copy(update={"artifacts": new_artifacts}))

    new_scene_plan = manifest.scene_plan.model_copy(update={"scenes": tuple(new_scenes)})
    new_manifest = manifest.model_copy(update={"scene_plan": new_scene_plan})

    # model_copy() does not re-run validators — re-validate explicitly so
    # a nested-copy mistake can never silently bypass ScenePlan's own
    # structural invariants. This also serves as a second, independent
    # proof that project_id/source_fingerprint were never touched: they
    # round-trip through model_validate() unchanged since they were never
    # named in either update(...) call above.
    return VideoManifest.model_validate(new_manifest.model_dump(mode="json"))
