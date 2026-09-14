"""Phase 1E: the local Story Queue import contract.

QueueItem references story/scene content by FILE PATH
(story_input_path/scene_plan_path), resolved relative to the queue JSON
file's own directory — it does not embed story/scene content inline.

This is a deliberate departure from Phase 1A's draft
docs/spec-v4/examples/queue-import.json, which predates Phase 1C's actual
StoryInput model and inlined full story fields (hook, key_facts,
target_length_seconds, tone, creative_prompt_ref, role_outfit_ref) that
StoryInput does not have. This Phase 1E task's own field list is the more
specific, more recent source of truth; docs/spec-v4/schemas/queue-import.schema.json
and the example file were updated this phase to match (see the Phase 1E
implementation report for the full explanation)."""
from __future__ import annotations

from pydantic import Field, model_validator

from src.models.common import FrozenStrictModel


class QueueItem(FrozenStrictModel):
    queue_item_id: str = Field(min_length=1)
    story_input_path: str = Field(min_length=1)
    scene_plan_path: str = Field(min_length=1)

    # Owner-facing only — never read by the render pipeline, may be
    # Arabic or any language (see project policy on non-viewer-facing text).
    owner_notes: str | None = None

    # Optional per-item override for where this project's manifest is
    # saved: <output-root>/<output_subdirectory or the built project_id>/
    # manifest.json. Resolved and traversal-checked against output-root
    # the same way story_input_path/scene_plan_path are checked against
    # the queue directory — see src/core/queue_import.py.
    output_subdirectory: str | None = None

    # Required, no default: every entry must explicitly state whether it
    # participates in an import, so a queue file can never accidentally
    # import an entry nobody meant to enable.
    enabled: bool


class QueueImport(FrozenStrictModel):
    queue_version: str = Field(min_length=1)
    items: tuple[QueueItem, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _queue_item_ids_are_unique(self) -> "QueueImport":
        ids = [item.queue_item_id for item in self.items]
        if len(ids) != len(set(ids)):
            duplicates = sorted({qid for qid in ids if ids.count(qid) > 1})
            raise ValueError(f"QueueImport has duplicate queue_item_id(s): {duplicates}")
        return self
