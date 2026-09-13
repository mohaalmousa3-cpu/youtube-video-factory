"""Typed loader for config/channel-config.yaml — channel-wide policy
(language, budget, motion/Flow/Veo, audio, timing, visual, character).

Deliberately separate from src/utils/config.py's Settings: Settings covers
paths and provider credentials from .env; ChannelPolicy covers the
channel-wide rules from docs/spec-v4/TECHNICAL-SPEC-EN.md. Phase 1B only
loads and validates this file — nothing in src/providers or src/render
imports this module yet, and this module does not import src/utils/config
(see get_runtime_config() in config.py for the one deferred integration
point, kept one-directional to avoid a circular import and to avoid making
pydantic a hard import-time dependency of config.py)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHANNEL_CONFIG_PATH = PROJECT_ROOT / "config" / "channel-config.yaml"


class ChannelConfigError(Exception):
    """Raised for any problem loading or validating channel-config.yaml —
    missing file, invalid YAML, a non-mapping document, a missing required
    key, an unknown key, or a violated policy invariant. Callers should
    treat this as fatal: channel policy is never guessed at or defaulted
    around a bad file."""


class _StrictModel(BaseModel):
    """Unknown keys are rejected, not silently ignored — a typo'd policy
    key (e.g. 'bakground_music_enabled') must fail loudly, not be dropped.
    frozen=True makes every field read-only after construction (Pydantic
    raises on assignment) — get_channel_policy() hands out one shared,
    lru_cache'd instance to every caller, so mutating a field in place
    would corrupt the cache for everyone else. This is enforced at
    runtime, not just documented: see the mutation-rejection tests in
    tests/test_channel_config.py."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ChannelSettings(_StrictModel):
    language: Literal["en-US"]
    viewer_facing_language: Literal["English"]


class BudgetSettings(_StrictModel):
    default_incremental_budget_usd: Literal[0]
    emergency_monthly_ceiling_usd: Literal[25]
    paid_services_enabled: bool
    # Critical invariant: no code path may ever make an automatic payment —
    # every paid spend requires a human-approved paid-proposal record (see
    # docs/spec-v4/schemas/paid-proposal.schema.json). Locked to False at
    # the type level (not just a default), so a config that sets this to
    # true fails to load at all rather than silently being accepted.
    automatic_payment_allowed: Literal[False]
    explicit_user_approval_required: bool

    @model_validator(mode="after")
    def _paid_services_require_explicit_approval(self) -> "BudgetSettings":
        if self.paid_services_enabled and not self.explicit_user_approval_required:
            raise ValueError(
                "budget.paid_services_enabled=true requires "
                "budget.explicit_user_approval_required=true"
            )
        return self


class MotionSettings(_StrictModel):
    manual_flow_required: Literal[False]
    require_local_fallback_for_manual_flow: Literal[True]
    veo_api_enabled: Literal[False]


class AudioSettings(_StrictModel):
    background_music_enabled: Literal[False]


class TimingSettings(_StrictModel):
    final_timing_source: Literal["measured_audio"]


class VisualSettings(_StrictModel):
    deterministic_text_overlays_enabled: Literal[True]


class CharacterSettings(_StrictModel):
    identity_locked_across_channel: Literal[True]


class ChannelPolicy(_StrictModel):
    """Root policy document, one-to-one with config/channel-config.yaml's
    top-level sections.

    Every model in this hierarchy (including this one, via _StrictModel's
    frozen=True) is actually immutable at runtime, not just documented as
    such: get_channel_policy() hands the same lru_cache'd instance to every
    caller, and attribute assignment on any field — top-level or nested,
    e.g. `policy.motion.veo_api_enabled = True` — raises a pydantic
    ValidationError instead of silently corrupting that shared instance.
    A caller that needs a *different* value should call get_channel_policy()
    fresh, or build a new instance with
    `policy.model_copy(update={...})` (model_copy bypasses frozen checks by
    design — it constructs a new instance rather than mutating this one).
    """

    channel: ChannelSettings
    budget: BudgetSettings
    motion: MotionSettings
    audio: AudioSettings
    timing: TimingSettings
    visual: VisualSettings
    character: CharacterSettings


def load_channel_policy(path: Path) -> ChannelPolicy:
    """Uncached: parse + validate one YAML file. Exists separately from
    get_channel_policy() so tests can point it at a temp-dir copy without
    touching the real config or the module-level cache."""
    try:
        text = path.read_text()
    except FileNotFoundError as exc:
        raise ChannelConfigError(f"channel config not found at {path}") from exc
    except PermissionError as exc:
        raise ChannelConfigError(f"permission denied reading channel config at {path}") from exc
    except UnicodeDecodeError as exc:
        raise ChannelConfigError(f"channel config at {path} is not valid UTF-8 text: {exc}") from exc
    except OSError as exc:
        raise ChannelConfigError(f"could not read channel config at {path}: {exc}") from exc

    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ChannelConfigError(f"invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ChannelConfigError(
            f"{path} must parse to a mapping at the top level, got {type(raw).__name__}"
        )

    try:
        return ChannelPolicy.model_validate(raw)
    except ValidationError as exc:
        raise ChannelConfigError(f"invalid channel config at {path}: {exc}") from exc


@lru_cache
def get_channel_policy() -> ChannelPolicy:
    """Cached accessor for config/channel-config.yaml — parsed/validated
    once per process. The returned ChannelPolicy is shared by every caller
    (lru_cache returns the same instance, not a copy) and is frozen, so
    attribute assignment on it (or any nested model) raises instead of
    corrupting the shared cache — see the ChannelPolicy docstring. Not yet
    called from any provider, renderer, or CLI command (see
    src/utils/config.py's get_runtime_config() for the one Phase 1B
    integration point)."""
    return load_channel_policy(CHANNEL_CONFIG_PATH)
