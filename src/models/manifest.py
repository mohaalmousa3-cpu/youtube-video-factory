"""VideoManifest — the single deterministic artifact describing one
video's planned build. Phase 1C only ever produces a manifest that is
planning-only: no assets, no measured timing, no provider execution.

PolicySnapshot is a plain, decoupled copy of the ChannelPolicy values that
matter here — never the live ChannelPolicy object itself. This module does
not import src.utils.channel_config at all; the conversion from a real
ChannelPolicy into a PolicySnapshot is src/core/manifest_builder.py's job
(see that module's `_build_policy_snapshot`), keeping this file pure data,
per the "keep data models separate from builder/store logic" instruction."""
from __future__ import annotations

from datetime import datetime

from pydantic import Field

from src.models.common import FrozenStrictModel
from src.models.enums import ExecutionStatus, ManifestLifecycleState
from src.models.scene import ScenePlan
from src.models.story import StoryInput


class PolicySnapshot(FrozenStrictModel):
    """A frozen, self-contained copy of the channel policy values that were
    in effect when this manifest was built — deliberately not the live
    ChannelPolicy object, so a saved manifest never holds a reference to
    (or drifts out of sync with) config/channel-config.yaml after the
    fact.

    This full snapshot (all fields below, including budget/governance) is
    kept purely as an audit record of what the policy was at build time.
    It is NOT what VideoManifest's fingerprint is computed from — the
    fingerprint uses only a content-production subset of these fields
    (excluding the budget/governance ones, which can change over time
    without the underlying video becoming a different video); see
    src/core/manifest_builder.py's _fingerprint_policy_payload()."""

    language: str
    viewer_facing_language: str
    default_incremental_budget_usd: float
    emergency_monthly_ceiling_usd: float
    paid_services_enabled: bool
    automatic_payment_allowed: bool
    explicit_user_approval_required: bool
    manual_flow_required: bool
    require_local_fallback_for_manual_flow: bool
    veo_api_enabled: bool
    background_music_enabled: bool
    final_timing_source: str
    deterministic_text_overlays_enabled: bool
    identity_locked_across_channel: bool


class VideoManifest(FrozenStrictModel):
    manifest_version: str = "1.0"

    # Deterministic identity: project_id is a short, stable label derived
    # from source_fingerprint (the full SHA-256 hex digest of the
    # canonical story_input + scene_plan + policy_snapshot payload) — see
    # manifest_builder.py. Neither is ever a random UUID.
    project_id: str = Field(min_length=1)
    source_fingerprint: str = Field(min_length=1)

    story_input: StoryInput
    scene_plan: ScenePlan
    policy_snapshot: PolicySnapshot

    lifecycle_state: ManifestLifecycleState
    execution_status: ExecutionStatus

    # Injected explicitly by the builder's `created_at` parameter in
    # tests, so tests never depend on the real wall clock.
    created_at: datetime

    target_duration_seconds: float = Field(gt=0)

    # Always None coming out of the Phase 1C builder — populated by a
    # later phase once Kokoro TTS output is actually measured. Never a
    # fabricated/estimated value (see docs/spec-v4/TECHNICAL-SPEC-EN.md
    # section 7 and CLAUDE.md's "known bugs already fixed" on wpm
    # estimates).
    measured_audio_duration_seconds: float | None = None

    # Policy, not a claim: this records that measured_audio IS the
    # required source for the eventual final timing, not that timing has
    # been measured (measured_audio_duration_seconds above is what would
    # actually carry a measured result, once one exists).
    final_timing_source: str

    output_root: str | None = None
