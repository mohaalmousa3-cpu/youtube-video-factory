"""Tests for src/core/manifest_builder.py: determinism, approval gating,
and policy-compatibility rejection. No network, no provider, no LLM — all
inputs are local dicts/fixtures.

Tests that need a ChannelPolicy with a value ChannelPolicy's own
Literal-typed fields make impossible to construct through its normal,
validated API (e.g. veo_api_enabled=True) use a lightweight, duck-typed
`_fake_channel_policy()` (a nested `types.SimpleNamespace`) instead of a
real ChannelPolicy instance. build_video_manifest() only ever accesses
channel_policy's attributes — it never isinstance()-checks it — so a fake
with the right attribute shape exercises the SAME code path a real,
somehow-misconfigured ChannelPolicy would. This does not touch
src/utils/channel_config.py or weaken any of its real invariants."""
import copy
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import yaml

from src.core.manifest_builder import ManifestValidationError, build_video_manifest
from src.utils.channel_config import CHANNEL_CONFIG_PATH, load_channel_policy


FIXED_CREATED_AT = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _real_channel_policy():
    return load_channel_policy(CHANNEL_CONFIG_PATH)


def _fake_channel_policy(**section_overrides) -> SimpleNamespace:
    defaults = {
        "channel": dict(language="en-US", viewer_facing_language="English"),
        "budget": dict(
            default_incremental_budget_usd=0,
            emergency_monthly_ceiling_usd=25,
            paid_services_enabled=False,
            automatic_payment_allowed=False,
            explicit_user_approval_required=True,
        ),
        "motion": dict(
            manual_flow_required=False,
            require_local_fallback_for_manual_flow=True,
            veo_api_enabled=False,
        ),
        "audio": dict(background_music_enabled=False),
        "timing": dict(final_timing_source="measured_audio"),
        "visual": dict(deterministic_text_overlays_enabled=True),
        "character": dict(identity_locked_across_channel=True),
    }
    for section, overrides in section_overrides.items():
        defaults[section] = {**defaults[section], **overrides}
    return SimpleNamespace(**{name: SimpleNamespace(**fields) for name, fields in defaults.items()})


def _valid_story_input(**overrides) -> dict:
    data = dict(
        story_id="why-we-care-what-people-think",
        title="Why We Care So Much What People Think",
        topic="social psychology",
        target_duration_seconds=480.0,
        language="en-US",
        viewer_facing_language="English",
        approval_status="approved",
    )
    data.update(overrides)
    return data


def _valid_scene(**overrides) -> dict:
    data = dict(
        scene_id="scene-01",
        sequence=1,
        narration_text="Why does being left out sting so much?",
        scene_type="establishing",
        narrative_beat="hook",
        visual_brief="Stickman character alone on a quiet street corner at dusk.",
        motion_mode="in",
        approval_state="approved",
    )
    data.update(overrides)
    return data


def _valid_scene_plan(**overrides) -> dict:
    data = dict(scenes=(_valid_scene(),), role_outfits=())
    data.update(overrides)
    return data


# ---------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------


def test_build_video_manifest_with_valid_input_succeeds():
    manifest = build_video_manifest(
        _valid_story_input(), _valid_scene_plan(), _real_channel_policy(), created_at=FIXED_CREATED_AT
    )
    assert manifest.manifest_version == "1.0"
    assert manifest.lifecycle_state == "planned"
    assert manifest.execution_status == "not_executed"
    assert manifest.measured_audio_duration_seconds is None
    assert manifest.created_at == FIXED_CREATED_AT
    assert manifest.target_duration_seconds == 480.0
    assert manifest.project_id.startswith("proj-")
    assert len(manifest.source_fingerprint) == 64  # sha256 hex digest


# ---------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------


def test_identical_inputs_and_fixed_created_at_produce_same_fingerprint():
    policy = _real_channel_policy()
    first = build_video_manifest(
        _valid_story_input(), _valid_scene_plan(), policy, created_at=FIXED_CREATED_AT
    )
    second = build_video_manifest(
        _valid_story_input(), _valid_scene_plan(), policy, created_at=FIXED_CREATED_AT
    )
    assert first.source_fingerprint == second.source_fingerprint
    assert first.project_id == second.project_id


def test_created_at_does_not_affect_fingerprint():
    policy = _real_channel_policy()
    first = build_video_manifest(
        _valid_story_input(), _valid_scene_plan(), policy, created_at=FIXED_CREATED_AT
    )
    second = build_video_manifest(
        _valid_story_input(),
        _valid_scene_plan(),
        policy,
        created_at=datetime(2030, 6, 15, tzinfo=timezone.utc),
    )
    assert first.source_fingerprint == second.source_fingerprint
    assert first.created_at != second.created_at


def test_changed_title_changes_fingerprint():
    policy = _real_channel_policy()
    baseline = build_video_manifest(
        _valid_story_input(), _valid_scene_plan(), policy, created_at=FIXED_CREATED_AT
    )
    changed = build_video_manifest(
        _valid_story_input(title="A Completely Different Title"),
        _valid_scene_plan(),
        policy,
        created_at=FIXED_CREATED_AT,
    )
    assert baseline.source_fingerprint != changed.source_fingerprint
    assert baseline.project_id != changed.project_id


def test_changed_scene_narration_changes_fingerprint():
    policy = _real_channel_policy()
    baseline = build_video_manifest(
        _valid_story_input(), _valid_scene_plan(), policy, created_at=FIXED_CREATED_AT
    )
    changed = build_video_manifest(
        _valid_story_input(),
        _valid_scene_plan(scenes=(_valid_scene(narration_text="A totally different line."),)),
        policy,
        created_at=FIXED_CREATED_AT,
    )
    assert baseline.source_fingerprint != changed.source_fingerprint


def test_changed_policy_snapshot_field_changes_fingerprint(tmp_path):
    """Two REAL, independently valid ChannelPolicy instances that differ
    in exactly one non-locked field (budget.paid_services_enabled) must
    produce different fingerprints for identical story/scene input."""
    with open(CHANNEL_CONFIG_PATH) as f:
        base_config = yaml.safe_load(f)

    policy_a = load_channel_policy(CHANNEL_CONFIG_PATH)

    modified_config = copy.deepcopy(base_config)
    modified_config["budget"]["paid_services_enabled"] = True
    modified_config["budget"]["explicit_user_approval_required"] = True  # keep the cross-field rule satisfied
    modified_path = tmp_path / "channel-config.yaml"
    modified_path.write_text(yaml.safe_dump(modified_config))
    policy_b = load_channel_policy(modified_path)

    assert policy_a.budget.paid_services_enabled != policy_b.budget.paid_services_enabled

    manifest_a = build_video_manifest(
        _valid_story_input(), _valid_scene_plan(), policy_a, created_at=FIXED_CREATED_AT
    )
    manifest_b = build_video_manifest(
        _valid_story_input(), _valid_scene_plan(), policy_b, created_at=FIXED_CREATED_AT
    )
    assert manifest_a.source_fingerprint != manifest_b.source_fingerprint


# ---------------------------------------------------------------------
# Input coercion (raw dict acceptance)
# ---------------------------------------------------------------------


def test_build_video_manifest_accepts_raw_dict_story_and_scene_plan():
    manifest = build_video_manifest(
        _valid_story_input(), _valid_scene_plan(), _real_channel_policy(), created_at=FIXED_CREATED_AT
    )
    assert manifest.story_input.story_id == "why-we-care-what-people-think"


def test_invalid_story_input_dict_is_rejected():
    bad_story = {k: v for k, v in _valid_story_input().items() if k != "story_id"}
    with pytest.raises(ManifestValidationError):
        build_video_manifest(bad_story, _valid_scene_plan(), _real_channel_policy())


def test_invalid_scene_plan_dict_is_rejected():
    with pytest.raises(ManifestValidationError):
        build_video_manifest(_valid_story_input(), {"scenes": ()}, _real_channel_policy())


# ---------------------------------------------------------------------
# Approval gating
# ---------------------------------------------------------------------


def test_unapproved_story_input_is_rejected():
    with pytest.raises(ManifestValidationError):
        build_video_manifest(
            _valid_story_input(approval_status="draft"), _valid_scene_plan(), _real_channel_policy()
        )


def test_unapproved_scene_is_rejected():
    with pytest.raises(ManifestValidationError):
        build_video_manifest(
            _valid_story_input(),
            _valid_scene_plan(scenes=(_valid_scene(approval_state="draft"),)),
            _real_channel_policy(),
        )


# ---------------------------------------------------------------------
# Policy rejection — StoryInput vs. a real ChannelPolicy
# ---------------------------------------------------------------------


def test_non_english_story_language_is_rejected():
    with pytest.raises(ManifestValidationError):
        build_video_manifest(
            _valid_story_input(language="ar-EG"), _valid_scene_plan(), _real_channel_policy()
        )


def test_non_english_story_viewer_facing_language_is_rejected():
    with pytest.raises(ManifestValidationError):
        build_video_manifest(
            _valid_story_input(viewer_facing_language="Arabic"),
            _valid_scene_plan(),
            _real_channel_policy(),
        )


# ---------------------------------------------------------------------
# Policy rejection — scene/overlay content vs. a real ChannelPolicy
# ---------------------------------------------------------------------


def test_overlay_marked_non_deterministic_is_rejected():
    scene = _valid_scene(
        text_overlays=(
            dict(
                text="Some Overlay",
                position="top",
                style_id="style-1",
                viewer_facing_language="English",
                deterministic=False,
            ),
        )
    )
    with pytest.raises(ManifestValidationError):
        build_video_manifest(
            _valid_story_input(), _valid_scene_plan(scenes=(scene,)), _real_channel_policy()
        )


def test_overlay_with_wrong_viewer_facing_language_is_rejected():
    scene = _valid_scene(
        text_overlays=(
            dict(
                text="Some Overlay",
                position="top",
                style_id="style-1",
                viewer_facing_language="Arabic",
                deterministic=True,
            ),
        )
    )
    with pytest.raises(ManifestValidationError):
        build_video_manifest(
            _valid_story_input(), _valid_scene_plan(scenes=(scene,)), _real_channel_policy()
        )


def test_role_outfit_used_outside_allowed_range_is_rejected_via_builder():
    """This scene-plan-level structural invariant is enforced by
    ScenePlan's own validator (see test_manifest_models.py), but since
    build_video_manifest() is given a raw dict here (not a pre-built
    ScenePlan), the same rejection must surface as a ManifestValidationError
    when going through the builder's normal dict-coercion path."""
    scene_plan = _valid_scene_plan(
        scenes=(_valid_scene(scene_id="scene-05", role_outfit_id="narrator-default"),),
        role_outfits=(
            dict(
                role_outfit_id="narrator-default",
                role_name="narrator",
                outfit_description="default outfit",
                allowed_scene_ids=("scene-01",),
                character_identity_ref="narrator-identity-v1",
            ),
        ),
    )
    with pytest.raises(ManifestValidationError):
        build_video_manifest(_valid_story_input(), scene_plan, _real_channel_policy())


def test_scene_claiming_generated_asset_is_rejected():
    scene = _valid_scene(artifacts=dict(audio_path="data/audio/scene-01.wav"))
    with pytest.raises(ManifestValidationError):
        build_video_manifest(
            _valid_story_input(), _valid_scene_plan(scenes=(scene,)), _real_channel_policy()
        )


def test_scene_claiming_measured_timing_is_rejected():
    scene = _valid_scene(artifacts=dict(measured_audio_duration_seconds=5.2))
    with pytest.raises(ManifestValidationError):
        build_video_manifest(
            _valid_story_input(), _valid_scene_plan(scenes=(scene,)), _real_channel_policy()
        )


def test_scene_requiring_flow_without_local_fallback_is_rejected():
    scene = _valid_scene(motion_mode="manual_flow")
    with pytest.raises(ManifestValidationError):
        build_video_manifest(
            _valid_story_input(), _valid_scene_plan(scenes=(scene,)), _real_channel_policy()
        )


def test_scene_requiring_flow_with_local_fallback_is_accepted():
    scene = _valid_scene(motion_mode="manual_flow", local_fallback_motion_mode="static")
    manifest = build_video_manifest(
        _valid_story_input(),
        _valid_scene_plan(scenes=(scene,)),
        _real_channel_policy(),
        created_at=FIXED_CREATED_AT,
    )
    assert manifest.scene_plan.scenes[0].motion_mode == "manual_flow"


# ---------------------------------------------------------------------
# Policy rejection — ChannelPolicy-level invariants (fakes; see module
# docstring for why these use _fake_channel_policy() instead of a real,
# unconstructible-by-design ChannelPolicy).
# ---------------------------------------------------------------------


def test_builder_rejects_automatic_payment_allowed():
    policy = _fake_channel_policy(budget=dict(automatic_payment_allowed=True))
    with pytest.raises(ManifestValidationError):
        build_video_manifest(_valid_story_input(), _valid_scene_plan(), policy)


def test_builder_rejects_veo_enabled():
    policy = _fake_channel_policy(motion=dict(veo_api_enabled=True))
    with pytest.raises(ManifestValidationError):
        build_video_manifest(_valid_story_input(), _valid_scene_plan(), policy)


def test_builder_rejects_background_music_enabled():
    policy = _fake_channel_policy(audio=dict(background_music_enabled=True))
    with pytest.raises(ManifestValidationError):
        build_video_manifest(_valid_story_input(), _valid_scene_plan(), policy)


def test_builder_rejects_non_deterministic_overlay_policy():
    policy = _fake_channel_policy(visual=dict(deterministic_text_overlays_enabled=False))
    with pytest.raises(ManifestValidationError):
        build_video_manifest(_valid_story_input(), _valid_scene_plan(), policy)


def test_builder_rejects_wrong_final_timing_source():
    policy = _fake_channel_policy(timing=dict(final_timing_source="estimated_wpm"))
    with pytest.raises(ManifestValidationError):
        build_video_manifest(_valid_story_input(), _valid_scene_plan(), policy)


def test_builder_rejects_identity_lock_disabled():
    policy = _fake_channel_policy(character=dict(identity_locked_across_channel=False))
    with pytest.raises(ManifestValidationError):
        build_video_manifest(_valid_story_input(), _valid_scene_plan(), policy)


def test_builder_rejects_local_fallback_not_required():
    policy = _fake_channel_policy(motion=dict(require_local_fallback_for_manual_flow=False))
    with pytest.raises(ManifestValidationError):
        build_video_manifest(_valid_story_input(), _valid_scene_plan(), policy)
