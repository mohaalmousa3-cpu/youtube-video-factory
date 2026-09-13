"""Tests for the channel-config.yaml loader/validator (Phase 1B). Every
negative case writes a mutated copy of the real config to tmp_path and
loads that copy directly via load_channel_policy() — the real
config/channel-config.yaml is never modified, and no test depends on
environment variables or secrets."""
import copy

import pytest
import yaml

from src.utils.channel_config import (
    CHANNEL_CONFIG_PATH,
    ChannelConfigError,
    ChannelPolicy,
    get_channel_policy,
    load_channel_policy,
)


@pytest.fixture()
def base_config_dict() -> dict:
    with open(CHANNEL_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _write_and_load(tmp_path, data: dict) -> ChannelPolicy:
    path = tmp_path / "channel-config.yaml"
    path.write_text(yaml.safe_dump(data))
    return load_channel_policy(path)


def test_get_channel_policy_loads_and_validates_default_config():
    policy = get_channel_policy()

    assert isinstance(policy, ChannelPolicy)
    assert policy.channel.language == "en-US"
    assert policy.channel.viewer_facing_language == "English"
    assert policy.budget.default_incremental_budget_usd == 0
    assert policy.budget.emergency_monthly_ceiling_usd == 25
    assert policy.budget.paid_services_enabled is False
    assert policy.budget.automatic_payment_allowed is False
    assert policy.budget.explicit_user_approval_required is True
    assert policy.motion.manual_flow_required is False
    assert policy.motion.require_local_fallback_for_manual_flow is True
    assert policy.motion.veo_api_enabled is False
    assert policy.audio.background_music_enabled is False
    assert policy.timing.final_timing_source == "measured_audio"
    assert policy.visual.deterministic_text_overlays_enabled is True
    assert policy.character.identity_locked_across_channel is True


def test_get_channel_policy_is_cached():
    first = get_channel_policy()
    second = get_channel_policy()
    assert first is second


def test_load_channel_policy_valid_copy_round_trips(base_config_dict, tmp_path):
    policy = _write_and_load(tmp_path, copy.deepcopy(base_config_dict))
    assert isinstance(policy, ChannelPolicy)


def test_load_channel_policy_missing_file_raises(tmp_path):
    with pytest.raises(ChannelConfigError):
        load_channel_policy(tmp_path / "does-not-exist.yaml")


def test_load_channel_policy_invalid_yaml_raises(tmp_path):
    path = tmp_path / "channel-config.yaml"
    path.write_text("channel: [unclosed")
    with pytest.raises(ChannelConfigError):
        load_channel_policy(path)


def test_load_channel_policy_path_is_directory_raises(tmp_path):
    """path.read_text() on a directory raises IsADirectoryError (an OSError
    subclass) — must surface as ChannelConfigError, not propagate raw."""
    directory = tmp_path / "channel-config.yaml"
    directory.mkdir()
    with pytest.raises(ChannelConfigError):
        load_channel_policy(directory)


def test_load_channel_policy_non_mapping_document_raises(tmp_path):
    path = tmp_path / "channel-config.yaml"
    path.write_text("- just\n- a\n- list\n")
    with pytest.raises(ChannelConfigError):
        load_channel_policy(path)


def test_load_channel_policy_missing_required_key_raises(base_config_dict, tmp_path):
    data = copy.deepcopy(base_config_dict)
    del data["budget"]["emergency_monthly_ceiling_usd"]
    with pytest.raises(ChannelConfigError):
        _write_and_load(tmp_path, data)


def test_load_channel_policy_rejects_unknown_key(base_config_dict, tmp_path):
    data = copy.deepcopy(base_config_dict)
    data["budget"]["unexpected_field"] = "oops"
    with pytest.raises(ChannelConfigError):
        _write_and_load(tmp_path, data)


def test_load_channel_policy_rejects_unknown_top_level_section(base_config_dict, tmp_path):
    data = copy.deepcopy(base_config_dict)
    data["publishing"] = {"auto_upload_enabled": True}
    with pytest.raises(ChannelConfigError):
        _write_and_load(tmp_path, data)


def test_budget_automatic_payment_allowed_true_is_rejected(base_config_dict, tmp_path):
    """Critical invariant: no config may set automatic_payment_allowed to
    true — automatic payment must never happen regardless of any other
    field's value."""
    data = copy.deepcopy(base_config_dict)
    data["budget"]["automatic_payment_allowed"] = True
    with pytest.raises(ChannelConfigError):
        _write_and_load(tmp_path, data)


def test_budget_automatic_payment_allowed_false_is_accepted(base_config_dict, tmp_path):
    data = copy.deepcopy(base_config_dict)
    data["budget"]["automatic_payment_allowed"] = False
    policy = _write_and_load(tmp_path, data)
    assert policy.budget.automatic_payment_allowed is False


def test_budget_paid_services_enabled_requires_explicit_approval(base_config_dict, tmp_path):
    data = copy.deepcopy(base_config_dict)
    data["budget"]["paid_services_enabled"] = True
    data["budget"]["explicit_user_approval_required"] = False
    with pytest.raises(ChannelConfigError):
        _write_and_load(tmp_path, data)


def test_budget_paid_services_enabled_with_approval_is_accepted(base_config_dict, tmp_path):
    """The cross-field invariant only forbids enabling paid services
    WITHOUT explicit approval — it doesn't forbid paid services outright.
    Enforcing 'never enable paid services at all' belongs to a later phase
    (see docs/spec-v4/IMPLEMENTATION-PLAN.md, Phase 1E)."""
    data = copy.deepcopy(base_config_dict)
    data["budget"]["paid_services_enabled"] = True
    data["budget"]["explicit_user_approval_required"] = True
    policy = _write_and_load(tmp_path, data)
    assert policy.budget.paid_services_enabled is True


@pytest.mark.parametrize(
    "section, key, bad_value",
    [
        ("channel", "language", "en-GB"),
        ("channel", "viewer_facing_language", "Arabic"),
        ("budget", "default_incremental_budget_usd", 10),
        ("budget", "emergency_monthly_ceiling_usd", 100),
        ("motion", "manual_flow_required", True),
        ("motion", "require_local_fallback_for_manual_flow", False),
        ("motion", "veo_api_enabled", True),
        ("audio", "background_music_enabled", True),
        ("timing", "final_timing_source", "estimated_wpm"),
        ("visual", "deterministic_text_overlays_enabled", False),
        ("character", "identity_locked_across_channel", False),
    ],
)
def test_load_channel_policy_rejects_invariant_violation(base_config_dict, tmp_path, section, key, bad_value):
    data = copy.deepcopy(base_config_dict)
    data[section][key] = bad_value

    with pytest.raises(ChannelConfigError):
        _write_and_load(tmp_path, data)
