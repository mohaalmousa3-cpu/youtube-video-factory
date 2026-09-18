"""DETERMINISTIC OVERLAY DERIVATION V1: pure, local manifest enrichment.

Given a VideoManifest whose every scene already has
`artifacts.measured_audio_duration_seconds` populated (see
src/core/scene_timing_finalizer.py's finalize_scene_timing(), a required
prior pipeline step) and the project's single registered project-level
"render" ArtifactRecord (see src/core/final_video_assembly.py), this module
returns a NEW VideoManifest in which every scene that currently has NO
explicit `text_overlays` receives exactly one deterministically-derived
TextOverlay — so the already-merged `render-text-overlays` command
(src/core/text_overlay_render.py) has something to consume for scenes a
human/earlier phase never explicitly annotated.

This module never derives overlay text via an LLM, never calls a provider,
never opens a network connection, never opens SQLite, never touches the
filesystem, and never invokes FFmpeg/ffprobe or any other subprocess — it
only reasons over the VideoManifest and ArtifactRecord objects its caller
already fetched (the CLI handler, src/cli.py's cmd_derive_text_overlays,
owns all I/O, exactly mirroring scene_timing_finalizer.py's own split).

Identity preservation follows scene_timing_finalizer.py's finalize_scene_timing()
pattern exactly: the returned VideoManifest is built exclusively via
model_copy(update=...) at every nested level (ScenePlanItem -> ScenePlan ->
VideoManifest) — never via src/core/manifest_builder.py's
build_video_manifest() or _compute_fingerprint(), either of which would
recompute source_fingerprint (and therefore project_id) from the full
scene_plan payload, including the very text_overlays field this module
changes. project_id and source_fingerprint are consequently never named in
any update(...) dict here, so model_copy leaves them byte-for-byte
identical to the input. Because model_copy() does not re-run pydantic
validators, the final result is re-validated via
VideoManifest.model_validate(new_manifest.model_dump(mode="json")) before
being returned, matching finalize_scene_timing()'s own defensive
re-validation.

Idempotent: a scene with any existing text_overlays is preserved exactly,
untouched — never appended to, never replaced. Only a scene whose
text_overlays is empty receives a derived overlay. Running this function
twice in a row therefore produces an equal manifest with no additional
overlays on the second pass."""
from __future__ import annotations

import math
from dataclasses import dataclass

from src.models.artifact import ArtifactRecord
from src.models.manifest import VideoManifest
from src.models.overlay import TextOverlay

# Fixed v1 derivation contract — not configurable. text_overlay_render.py's
# own _STYLES table requires this exact (style_id, position) pairing;
# lower_third is the least visually intrusive of its three supported
# styles, chosen as the single uniform default for every derived overlay.
_STYLE_ID = "lower_third_primary"
_POSITION = "lower_third"
_VIEWER_FACING_LANGUAGE = "English"

# Deterministic text-extraction contract (STEP 6): first N characters of
# narration_text, word-boundary-trimmed, "..." appended only when
# truncation actually occurred. 70 + len("...") = 73, comfortably under
# text_overlay_render.py's own 80-character renderer cap.
_MAX_NARRATION_PREFIX_LENGTH = 70
_ASCII_WHITESPACE_CHARS = " \t\n\r"


class OverlayDerivationError(Exception):
    """Base class for every reason derive_text_overlays() cannot produce a
    fully, validly enriched VideoManifest. Message text names only
    scene_id/artifact_id/project_id/a short reason — never a manifest
    path, generated content, or a raw exception representation."""


class MissingMeasuredDurationError(OverlayDerivationError):
    """A scene's artifacts.measured_audio_duration_seconds is None —
    finalize-scene-timing must run before this module; never estimated or
    guessed here."""


class InvalidMeasuredDurationError(OverlayDerivationError):
    """A scene's measured_audio_duration_seconds is present but not a
    real, finite, strictly-positive number (bool, NaN, infinity, zero, or
    negative)."""


class TimelineMismatchError(OverlayDerivationError):
    """The cumulative sum of every scene's measured_audio_duration_seconds
    does not match the registered render artifact's own measured
    duration_seconds within max(0.25, 0.05 * scene_count) seconds —
    signals the supplied manifest and render artifact do not describe the
    same actual video."""


class EmptyDerivedOverlayTextError(OverlayDerivationError):
    """A scene's narration_text is empty (or entirely whitespace) after
    stripping — there is no content to derive an overlay from."""


class RenderArtifactValidationError(OverlayDerivationError):
    """The supplied render_artifact is not a valid, matching, project-level
    "render" ArtifactRecord for this manifest, or its
    metadata["duration_seconds"] is missing or not a real, finite,
    strictly-positive number."""


@dataclass(frozen=True)
class OverlayDerivationSummary:
    """A typed, immutable summary of one derive_text_overlays() call,
    computed by comparing the original manifest against its derived
    result. Pure — no field here is anything derive_text_overlays()
    itself did not already determine while building that result."""

    scenes_with_new_overlays: int
    scenes_with_existing_overlays: int
    derived_overlay_count: int
    preserved_overlay_count: int


def _duration_tolerance_seconds(scene_count: int) -> float:
    """Same formula src.core.final_video_assembly._duration_tolerance_seconds
    already uses, reused here for consistency rather than inventing a
    second tolerance policy: a 0.25s fixed floor plus 0.05s per scene."""
    return max(0.25, 0.05 * scene_count)


def _validate_render_artifact(render_artifact: ArtifactRecord, manifest: VideoManifest) -> float:
    if render_artifact.kind != "render":
        raise RenderArtifactValidationError(
            f"render_artifact.kind must be 'render', got {render_artifact.kind!r}"
        )
    if render_artifact.scene_id is not None:
        raise RenderArtifactValidationError(
            "render_artifact.scene_id must be None — 'render' is a project-level artifact"
        )
    if render_artifact.project_id != manifest.project_id:
        raise RenderArtifactValidationError(
            f"render_artifact.project_id {render_artifact.project_id!r} does not match "
            f"manifest.project_id {manifest.project_id!r}"
        )
    if "duration_seconds" not in render_artifact.metadata:
        raise RenderArtifactValidationError(
            f"render_artifact {render_artifact.artifact_id!r} metadata is missing a "
            "'duration_seconds' value"
        )
    value = render_artifact.metadata["duration_seconds"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RenderArtifactValidationError(
            f"render_artifact {render_artifact.artifact_id!r} has a non-numeric "
            "duration_seconds value"
        )
    if not math.isfinite(value):
        raise RenderArtifactValidationError(
            f"render_artifact {render_artifact.artifact_id!r} has a non-finite "
            "duration_seconds value"
        )
    if value <= 0:
        raise RenderArtifactValidationError(
            f"render_artifact {render_artifact.artifact_id!r} has a non-positive "
            "duration_seconds value"
        )
    return float(value)


def _validate_scene_duration(scene_id: str, duration: float | None) -> float:
    if duration is None:
        raise MissingMeasuredDurationError(
            f"scene {scene_id!r} has no measured_audio_duration_seconds — run "
            "finalize-scene-timing before deriving overlays"
        )
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise InvalidMeasuredDurationError(
            f"scene {scene_id!r} has a non-numeric measured_audio_duration_seconds value"
        )
    if not math.isfinite(duration):
        raise InvalidMeasuredDurationError(
            f"scene {scene_id!r} has a non-finite measured_audio_duration_seconds value"
        )
    if duration <= 0:
        raise InvalidMeasuredDurationError(
            f"scene {scene_id!r} has a non-positive measured_audio_duration_seconds value"
        )
    return float(duration)


def _last_ascii_whitespace_index(text: str) -> int:
    return max((text.rfind(c) for c in _ASCII_WHITESPACE_CHARS), default=-1)


def _derive_text(narration_text: str, scene_id: str) -> str:
    """STEP 6's exact algorithm: strip, reject if empty, use verbatim if
    <=70 chars, otherwise take a 70-char prefix, trim back to the last
    ASCII whitespace within it (never mid-word) when one exists past
    index 0, and append "..." only because truncation occurred."""
    stripped = narration_text.strip()
    if not stripped:
        raise EmptyDerivedOverlayTextError(
            f"scene {scene_id!r} narration_text is empty after stripping whitespace"
        )
    if len(stripped) <= _MAX_NARRATION_PREFIX_LENGTH:
        return stripped

    prefix = stripped[:_MAX_NARRATION_PREFIX_LENGTH]
    last_ws = _last_ascii_whitespace_index(prefix)
    if last_ws > 0:
        truncated = prefix[:last_ws].strip()
    else:
        truncated = prefix
    return truncated + "..."


def derive_text_overlays(manifest: VideoManifest, render_artifact: ArtifactRecord) -> VideoManifest:
    """Return a new VideoManifest in which every scene with an empty
    text_overlays tuple receives exactly one deterministically-derived
    TextOverlay; every scene with a non-empty text_overlays tuple is
    preserved exactly, untouched. Raises an OverlayDerivationError
    subclass, with nothing returned, on any validation failure — no
    partial derivation. Never mutates `manifest`, `render_artifact`, or
    any nested value — every input is treated as read-only. No
    filesystem, database, subprocess, provider, or network access
    anywhere in this function."""
    render_duration = _validate_render_artifact(render_artifact, manifest)

    scenes = manifest.scene_plan.scenes
    scene_durations = [
        _validate_scene_duration(scene.scene_id, scene.artifacts.measured_audio_duration_seconds)
        for scene in scenes
    ]

    cumulative_total = sum(scene_durations)
    tolerance = _duration_tolerance_seconds(len(scenes))
    if abs(cumulative_total - render_duration) > tolerance:
        raise TimelineMismatchError(
            f"cumulative measured scene duration {cumulative_total:.3f}s does not match the "
            f"render artifact's duration {render_duration:.3f}s within tolerance {tolerance:.3f}s"
        )

    new_scenes = []
    cumulative_start = 0.0
    for scene, duration in zip(scenes, scene_durations):
        if scene.text_overlays:
            # Preserved exactly, byte-for-byte at the model-data level —
            # never appended to, replaced, or re-validated beyond the
            # normal final VideoManifest.model_validate() below.
            new_scenes.append(scene)
        else:
            start_seconds = round(cumulative_start, 3)
            end_seconds = round(cumulative_start + duration, 3)
            derived_overlay = TextOverlay(
                text=_derive_text(scene.narration_text, scene.scene_id),
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                position=_POSITION,
                style_id=_STYLE_ID,
                viewer_facing_language=_VIEWER_FACING_LANGUAGE,
                deterministic=True,
            )
            new_scenes.append(scene.model_copy(update={"text_overlays": (derived_overlay,)}))
        # Advance for EVERY scene, including ones whose overlays were
        # preserved untouched — explicit-overlay preservation never
        # changes the timeline calculation for later scenes.
        cumulative_start += duration

    new_scene_plan = manifest.scene_plan.model_copy(update={"scenes": tuple(new_scenes)})
    new_manifest = manifest.model_copy(update={"scene_plan": new_scene_plan})

    # model_copy() does not re-run validators — re-validate explicitly, so
    # a nested-copy mistake can never silently bypass ScenePlan's own
    # structural invariants. This also independently proves project_id/
    # source_fingerprint were never touched: they round-trip through
    # model_validate() unchanged since they were never named in either
    # update(...) call above.
    return VideoManifest.model_validate(new_manifest.model_dump(mode="json"))


def summarize_derivation(original: VideoManifest, derived: VideoManifest) -> OverlayDerivationSummary:
    """Pure comparison of `original` (the manifest passed into
    derive_text_overlays()) against `derived` (its return value) —
    counts scenes/overlays by comparing which scenes had text_overlays
    before derivation ran. No I/O; safe to call with any two manifests
    that share the same scene order/count."""
    scenes_with_new = 0
    scenes_with_existing = 0
    derived_count = 0
    preserved_count = 0
    for original_scene, derived_scene in zip(original.scene_plan.scenes, derived.scene_plan.scenes):
        if original_scene.text_overlays:
            scenes_with_existing += 1
            preserved_count += len(original_scene.text_overlays)
        else:
            scenes_with_new += 1
            derived_count += len(derived_scene.text_overlays)
    return OverlayDerivationSummary(
        scenes_with_new_overlays=scenes_with_new,
        scenes_with_existing_overlays=scenes_with_existing,
        derived_overlay_count=derived_count,
        preserved_overlay_count=preserved_count,
    )
