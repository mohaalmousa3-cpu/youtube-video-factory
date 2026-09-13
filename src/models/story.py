"""StoryInput — one approved video story request.

Deliberately narrower than docs/spec-v4/schemas/story-input.schema.json
(which also has hook, key_facts, creative_prompt_ref, role_outfit_ref) —
this Phase 1C task's own required-concepts list is the source of truth
where the two diverge; the omitted fields are not needed by this phase's
local, provider-free assembly and can be added back in a later phase if a
real script-generation stage needs them. See the Phase 1C implementation
report for the full list of differences."""
from __future__ import annotations

from pydantic import Field, field_validator

from src.models.common import FrozenStrictModel, is_english_ascii_text
from src.models.enums import ApprovalStatus


class StoryInput(FrozenStrictModel):
    story_id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    title: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    target_duration_seconds: float = Field(gt=0)

    # `language`/`viewer_facing_language` are loosely typed on purpose:
    # whether a given value is actually ALLOWED is a relational question
    # ("compatible with the active ChannelPolicy"), not an intrinsic
    # property of a StoryInput by itself — enforcing that comparison is
    # src/core/manifest_builder.py's job, not this model's. `title`/`topic`
    # above, by contrast, are absolute content rules ("must be English")
    # and so are enforced unconditionally right here.
    language: str = Field(min_length=2, max_length=20)
    viewer_facing_language: str = Field(min_length=2, max_length=40)

    approval_status: ApprovalStatus
    origin: str | None = None

    # Explicitly NOT viewer-facing: may be Arabic or any language, is
    # never read by the render pipeline, and is not subject to the
    # English-content heuristic below. (It IS still part of the
    # deterministic fingerprint payload in manifest_builder.py — changing
    # an owner note is a real content change for identity purposes, even
    # though it never reaches a viewer.)
    owner_notes: str | None = None

    @field_validator("title", "topic")
    @classmethod
    def _must_be_english(cls, value: str, info) -> str:
        if not is_english_ascii_text(value):
            raise ValueError(
                f"StoryInput.{info.field_name} must be English (ASCII-only heuristic)"
            )
        return value
