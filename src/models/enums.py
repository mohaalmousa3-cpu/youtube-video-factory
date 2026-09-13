"""Finite value sets shared across src/models/*, expressed as Literal type
aliases rather than enum.Enum classes — consistent with
src/utils/channel_config.py's style elsewhere in this codebase."""
from __future__ import annotations

from typing import Literal

# Approval gate used at both the StoryInput and per-scene level. A
# VideoManifest can only be built once every relevant approval_status
# equals "approved" (see src/core/manifest_builder.py).
ApprovalStatus = Literal["draft", "approved", "rejected"]

# Deliberately small and specific to this channel's stickman/watercolor
# psychology-facts format. Extend as real scripts reveal more categories —
# do not read this as an exhaustive taxonomy for every possible channel.
SceneType = Literal["establishing", "narration", "dialogue", "transition", "closing"]

# "manual_flow" is a real, selectable motion mode — Google Flow stays
# optional per channel policy, but a scene CAN request it. There is no
# "veo" value: Veo-based motion is not representable at all here, by
# design, rather than being representable-but-disabled at runtime.
MotionMode = Literal["in", "out", "pan_lr", "pan_up", "static", "manual_flow"]

OverlayPosition = Literal["top", "center", "bottom", "lower_third"]

# Phase 1C's builder only ever produces "planned" — the later values exist
# so VideoManifest doesn't need a breaking type change once later phases
# start advancing a manifest's lifecycle.
ManifestLifecycleState = Literal["planned", "in_production", "complete", "archived"]

# Phase 1C only ever produces "not_executed" — no provider has run.
ExecutionStatus = Literal["not_executed"]
