"""Phase 1C: pure, local, deterministic Story Input -> Scene Plan -> Video
Manifest assembly.

No filesystem writes, no database access, no provider calls, no
environment-dependent behavior. See docs/spec-v4/IMPLEMENTATION-PLAN.md for
what later phases (not started) add on top of this: Phase 1D wires real
TTS/image/render providers and fills in the artifact placeholders this
builder always leaves None; Phase 1E adds paid-service gating.

Validation is split across three places, each responsible for a different
kind of invariant:

- StoryInput / ScenePlanItem / TextOverlay's own validators enforce
  ABSOLUTE content rules that never depend on anything external (e.g.
  narration_text must be English; a fallback motion mode can't itself be
  "manual_flow").
- ScenePlan's own validators enforce STRUCTURAL invariants intrinsic to
  what a valid scene plan looks like, regardless of pipeline phase or
  policy (unique scene IDs, contiguous ordering, valid role_outfit
  references).
- This module enforces everything else: RELATIONAL checks against a
  specific ChannelPolicy (e.g. "is this story's language the one this
  channel is allowed to publish in"), approval-gate checks, and
  Phase-1C-specific temporal checks ("no scene may claim generated assets
  yet") that don't belong on the reusable ScenePlanItem type itself since a
  later phase reuses that same type once assets are real.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Mapping

from pydantic import ValidationError

from src.models.manifest import PolicySnapshot, VideoManifest
from src.models.scene import ScenePlan
from src.models.story import StoryInput
from src.utils.channel_config import ChannelPolicy


class ManifestValidationError(Exception):
    """Raised for any Phase 1C build-time problem: malformed StoryInput/
    ScenePlan input, an approval gate not met, an incompatibility with the
    active ChannelPolicy, or a cross-scene/policy consistency failure.
    Callers should treat this as fatal — a manifest is never built from
    data that failed validation."""


def _coerce_story_input(value: StoryInput | Mapping[str, Any]) -> StoryInput:
    if isinstance(value, StoryInput):
        return value
    try:
        return StoryInput.model_validate(value)
    except ValidationError as exc:
        raise ManifestValidationError(f"invalid story_input: {exc}") from exc


def _coerce_scene_plan(value: ScenePlan | Mapping[str, Any]) -> ScenePlan:
    if isinstance(value, ScenePlan):
        return value
    try:
        return ScenePlan.model_validate(value)
    except ValidationError as exc:
        raise ManifestValidationError(f"invalid scene_plan: {exc}") from exc


def _build_policy_snapshot(channel_policy: ChannelPolicy) -> PolicySnapshot:
    """Copy the channel_policy VALUES this manifest needs to remember into
    a plain, decoupled PolicySnapshot — never store channel_policy itself
    inside a VideoManifest (see PolicySnapshot's docstring)."""
    return PolicySnapshot(
        language=channel_policy.channel.language,
        viewer_facing_language=channel_policy.channel.viewer_facing_language,
        default_incremental_budget_usd=channel_policy.budget.default_incremental_budget_usd,
        emergency_monthly_ceiling_usd=channel_policy.budget.emergency_monthly_ceiling_usd,
        paid_services_enabled=channel_policy.budget.paid_services_enabled,
        automatic_payment_allowed=channel_policy.budget.automatic_payment_allowed,
        explicit_user_approval_required=channel_policy.budget.explicit_user_approval_required,
        manual_flow_required=channel_policy.motion.manual_flow_required,
        require_local_fallback_for_manual_flow=channel_policy.motion.require_local_fallback_for_manual_flow,
        veo_api_enabled=channel_policy.motion.veo_api_enabled,
        background_music_enabled=channel_policy.audio.background_music_enabled,
        final_timing_source=channel_policy.timing.final_timing_source,
        deterministic_text_overlays_enabled=channel_policy.visual.deterministic_text_overlays_enabled,
        identity_locked_across_channel=channel_policy.character.identity_locked_across_channel,
    )


# The fingerprint (and therefore project_id) must represent a video's
# CONTENT-PRODUCTION identity only — not the channel's current operational
# spending posture. These nine fields describe *what the video is and how
# it may be produced* (language, Flow/Veo/music/timing/overlay/identity
# rules); changing any of them really does describe a different planned
# video, so they participate in the fingerprint.
_FINGERPRINT_POLICY_FIELDS = (
    "language",
    "viewer_facing_language",
    "manual_flow_required",
    "require_local_fallback_for_manual_flow",
    "veo_api_enabled",
    "background_music_enabled",
    "final_timing_source",
    "deterministic_text_overlays_enabled",
    "identity_locked_across_channel",
)

# By contrast, these five fields are governance/spend CONTROLS, not
# content: default_incremental_budget_usd, emergency_monthly_ceiling_usd,
# paid_services_enabled, automatic_payment_allowed, and
# explicit_user_approval_required can all change over time (e.g. a
# paid-proposal being approved — see Phase 1E in
# docs/spec-v4/IMPLEMENTATION-PLAN.md) without the underlying video being a
# different video. They are deliberately EXCLUDED from
# _FINGERPRINT_POLICY_FIELDS/_fingerprint_policy_payload() below, even
# though the full PolicySnapshot (all fourteen fields, governance included)
# is still stored on VideoManifest.policy_snapshot as an audit record of
# what the channel's policy was at build time.


def _canonical_json(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _fingerprint_policy_payload(policy_snapshot: PolicySnapshot) -> dict:
    """The content-production subset of PolicySnapshot used for
    fingerprinting — never the full snapshot. Exposed as its own function
    (rather than inlined into _compute_fingerprint) so it can be tested in
    isolation against hand-built PolicySnapshot values, including ones a
    real, validated ChannelPolicy could never actually produce (e.g.
    automatic_payment_allowed=True) — proving a governance field's
    exclusion this way never requires weakening build_video_manifest()'s
    own safety gate, which continues to reject such a ChannelPolicy
    outright before fingerprinting is ever reached."""
    dumped = policy_snapshot.model_dump(mode="json")
    return {field: dumped[field] for field in _FINGERPRINT_POLICY_FIELDS}


def _compute_fingerprint(
    story_input: StoryInput, scene_plan: ScenePlan, policy_snapshot: PolicySnapshot
) -> str:
    """SHA-256 of the canonical JSON of (story_input, scene_plan,
    fingerprint_policy_payload). Deliberately excludes created_at (two
    builds of identical content at different times must fingerprint
    identically) and the governance/spend subset of policy_snapshot (see
    _fingerprint_policy_payload and _FINGERPRINT_POLICY_FIELDS above) —
    only content-production fields participate in a video's identity."""
    payload = {
        "story_input": story_input.model_dump(mode="json"),
        "scene_plan": scene_plan.model_dump(mode="json"),
        "policy_snapshot": _fingerprint_policy_payload(policy_snapshot),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _check_approval(story_input: StoryInput, scene_plan: ScenePlan) -> None:
    if story_input.approval_status != "approved":
        raise ManifestValidationError(
            f"story_input {story_input.story_id!r} is not approved "
            f"(approval_status={story_input.approval_status!r})"
        )
    for scene in scene_plan.scenes:
        if scene.approval_state != "approved":
            raise ManifestValidationError(
                f"scene {scene.scene_id!r} is not approved "
                f"(approval_state={scene.approval_state!r})"
            )


def _check_policy_compatibility(story_input: StoryInput, channel_policy: ChannelPolicy) -> None:
    if story_input.language != channel_policy.channel.language:
        raise ManifestValidationError(
            f"story_input.language {story_input.language!r} is not compatible with "
            f"channel policy language {channel_policy.channel.language!r}"
        )
    if story_input.viewer_facing_language != channel_policy.channel.viewer_facing_language:
        raise ManifestValidationError(
            f"story_input.viewer_facing_language {story_input.viewer_facing_language!r} is "
            f"not compatible with channel policy viewer_facing_language "
            f"{channel_policy.channel.viewer_facing_language!r}"
        )

    # The remaining checks restate invariants ChannelPolicy itself already
    # guarantees at load time (Phase 1B's Literal-typed fields make the
    # "wrong" branch unreachable through ChannelPolicy's own validated
    # construction path). They stay here anyway, deliberately: this
    # builder should report a policy incompatibility with its own clear
    # ManifestValidationError message rather than silently relying on
    # upstream loading having gone correctly, and a caller could in
    # principle hand this function something ChannelPolicy-shaped that
    # was not obtained via get_channel_policy()/load_channel_policy().
    if channel_policy.budget.automatic_payment_allowed:
        raise ManifestValidationError("channel policy must never allow automatic payment")
    if channel_policy.motion.veo_api_enabled:
        raise ManifestValidationError("channel policy must never enable the Veo API")
    if channel_policy.audio.background_music_enabled:
        raise ManifestValidationError("channel policy must never enable background music")
    if not channel_policy.visual.deterministic_text_overlays_enabled:
        raise ManifestValidationError("channel policy must require deterministic text overlays")
    if channel_policy.timing.final_timing_source != "measured_audio":
        raise ManifestValidationError(
            "channel policy final_timing_source must be 'measured_audio'"
        )
    if not channel_policy.character.identity_locked_across_channel:
        raise ManifestValidationError(
            "channel policy must keep character identity locked across the channel"
        )
    if not channel_policy.motion.require_local_fallback_for_manual_flow:
        raise ManifestValidationError(
            "channel policy must require a local fallback for manual Flow tasks"
        )


def _check_scene_policy_compliance(scene_plan: ScenePlan, channel_policy: ChannelPolicy) -> None:
    for scene in scene_plan.scenes:
        if (
            channel_policy.motion.require_local_fallback_for_manual_flow
            and scene.motion_mode == "manual_flow"
            and scene.local_fallback_motion_mode is None
        ):
            raise ManifestValidationError(
                f"scene {scene.scene_id!r} uses motion_mode='manual_flow' without a "
                "local_fallback_motion_mode (Flow must remain non-blocking)"
            )

        for overlay in scene.text_overlays:
            if not overlay.deterministic:
                raise ManifestValidationError(
                    f"scene {scene.scene_id!r} has a text overlay marked "
                    "deterministic=False, which the channel policy forbids"
                )
            if overlay.viewer_facing_language != channel_policy.channel.viewer_facing_language:
                raise ManifestValidationError(
                    f"scene {scene.scene_id!r} has a text overlay with "
                    f"viewer_facing_language {overlay.viewer_facing_language!r}, expected "
                    f"{channel_policy.channel.viewer_facing_language!r}"
                )

        artifacts = scene.artifacts
        populated = [
            field
            for field in (
                "audio_path",
                "image_path",
                "lipsync_data_path",
                "rendered_clip_path",
                "measured_audio_duration_seconds",
            )
            if getattr(artifacts, field) is not None
        ]
        if populated:
            raise ManifestValidationError(
                f"scene {scene.scene_id!r} declares generated/measured artifact data "
                f"({', '.join(populated)}) — Phase 1C must never claim assets exist or "
                "timing was measured"
            )


def build_video_manifest(
    story_input: StoryInput | Mapping[str, Any],
    scene_plan: ScenePlan | Mapping[str, Any],
    channel_policy: ChannelPolicy,
    *,
    created_at: datetime | None = None,
    output_root: str | None = None,
) -> VideoManifest:
    """Validate `story_input` and `scene_plan` (accepting either an actual
    model instance or a raw dict to be validated), check them against
    `channel_policy`, and return a deterministic VideoManifest.

    Raises ManifestValidationError on any invalid input, unmet approval
    gate, or policy incompatibility. Never touches the filesystem, a
    database, or a remote provider. `created_at` should be passed
    explicitly in tests so results are deterministic and independent of
    the real wall clock; if omitted, the current UTC time is used."""
    story = _coerce_story_input(story_input)
    scenes = _coerce_scene_plan(scene_plan)

    _check_approval(story, scenes)
    _check_policy_compatibility(story, channel_policy)
    _check_scene_policy_compliance(scenes, channel_policy)

    policy_snapshot = _build_policy_snapshot(channel_policy)
    fingerprint = _compute_fingerprint(story, scenes, policy_snapshot)
    project_id = f"proj-{fingerprint[:16]}"

    return VideoManifest(
        project_id=project_id,
        source_fingerprint=fingerprint,
        story_input=story,
        scene_plan=scenes,
        policy_snapshot=policy_snapshot,
        lifecycle_state="planned",
        execution_status="not_executed",
        created_at=created_at if created_at is not None else datetime.now(timezone.utc),
        target_duration_seconds=story.target_duration_seconds,
        measured_audio_duration_seconds=None,
        final_timing_source=policy_snapshot.final_timing_source,
        output_root=output_root,
    )
