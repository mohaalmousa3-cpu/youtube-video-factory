"""RoleOutfit — a character-identity/outfit record a ScenePlanItem may
reference.

Loosely inspired by docs/spec-v4/schemas/role-outfit.schema.json, but
extended with allowed_scene_ids and character_identity_ref, which that
Phase 1A JSON Schema does not define — this Phase 1C task's own field list
is the more specific, more recent source of truth where the two diverge
(see the Phase 1C implementation report for the full list of
differences)."""
from __future__ import annotations

from pydantic import Field

from src.models.common import FrozenStrictModel


class RoleOutfit(FrozenStrictModel):
    role_outfit_id: str = Field(min_length=1)
    role_name: str = Field(min_length=1)
    outfit_description: str = Field(min_length=1)

    # Which scenes this outfit variant may be used in. Required and
    # non-empty on purpose — there is no "empty/omitted means all scenes"
    # default, so a caller always states intent explicitly rather than
    # relying on an implicit fallback for a field that gates what a scene
    # is allowed to reference (see ScenePlan's cross-scene validator).
    allowed_scene_ids: tuple[str, ...] = Field(min_length=1)

    # Ties this outfit back to the ONE locked channel character identity
    # (ChannelPolicy.character.identity_locked_across_channel). Every
    # RoleOutfit referenced within a single ScenePlan must share the same
    # value here — enforced by ScenePlan's own validator — so multiple
    # outfits are recognized as variants of one character (e.g. a rain
    # coat over the default outfit), never accidentally a second
    # character.
    character_identity_ref: str = Field(min_length=1)
