"""Shared base class and validation helpers for src/models/*."""
from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict


class FrozenStrictModel(BaseModel):
    """Base for every Phase 1C domain model: unknown fields are rejected
    (extra="forbid") and instances are immutable after construction
    (frozen=True) — the same pattern src/utils/channel_config.py uses for
    ChannelPolicy, for the same reason: these are shared, cached-adjacent
    value objects (a VideoManifest in particular is meant to be built once
    and treated as a fact about a specific project), and accidental
    in-place mutation should fail loudly rather than silently corrupt
    something another part of the pipeline is holding a reference to."""

    model_config = ConfigDict(extra="forbid", frozen=True)


_PRINTABLE_ASCII = re.compile(r"^[\x20-\x7E\r\n\t]*$")


def is_english_ascii_text(text: str) -> bool:
    """Heuristic only — this is NOT real language detection. Returns True
    iff `text` contains nothing but printable ASCII and common whitespace.

    This reliably rejects Arabic, CJK, Cyrillic, and similar non-Latin
    scripts (satisfying the "no Arabic/other viewer-facing language"
    project policy without a new dependency), but it does NOT verify the
    text is grammatically English, and it WILL reject legitimate English
    text that uses non-ASCII punctuation (curly quotes, em dashes,
    accented loanwords like "cliché"). That trade-off is acceptable for
    Phase 1C's local, hand-written fixtures; revisit with a real
    language-detection dependency if that limitation becomes a problem for
    real script content in a later phase."""
    return bool(_PRINTABLE_ASCII.fullmatch(text))
