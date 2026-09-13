"""TextOverlay — deterministic, English-only on-screen text layered over a
scene at render time. It is NEVER baked into an AI-generated image (see
docs/spec-v4/TECHNICAL-SPEC-EN.md section 8) — this model describes the
separate, deterministic compositing step."""
from __future__ import annotations

from pydantic import Field, field_validator, model_validator

from src.models.common import FrozenStrictModel, is_english_ascii_text
from src.models.enums import OverlayPosition


class TextOverlay(FrozenStrictModel):
    # May be empty ("Empty overlays are allowed") but must always be
    # supplied explicitly — no silent default that could mask a caller
    # forgetting to set real text.
    text: str
    start_seconds: float | None = None
    end_seconds: float | None = None
    position: OverlayPosition
    style_id: str = Field(min_length=1)

    # Not hard-locked to "English" at the type level, unlike `text` below —
    # this is a label describing which language the overlay targets, and
    # whether that's the RIGHT language for this channel is a relational
    # question against the active ChannelPolicy, checked by
    # src/core/manifest_builder.py (same reasoning as
    # StoryInput.viewer_facing_language).
    viewer_facing_language: str = Field(min_length=2, max_length=40)

    # Required, no default: every overlay must explicitly assert it is
    # deterministic. This is an absolute content rule (not
    # policy-relational), so it is enforced here unconditionally — a
    # caller that sets this False has described something the channel
    # policy forbids regardless of any other setting.
    deterministic: bool

    @field_validator("text")
    @classmethod
    def _text_must_be_english(cls, value: str) -> str:
        if value and not is_english_ascii_text(value):
            raise ValueError("TextOverlay.text must be English (ASCII-only heuristic)")
        return value

    @model_validator(mode="after")
    def _timing_range_is_sane(self) -> "TextOverlay":
        if self.start_seconds is not None and self.start_seconds < 0:
            raise ValueError("TextOverlay.start_seconds must be >= 0")
        if (
            self.start_seconds is not None
            and self.end_seconds is not None
            and self.end_seconds < self.start_seconds
        ):
            raise ValueError("TextOverlay.end_seconds must be >= start_seconds")
        return self
