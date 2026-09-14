"""ScenePlanItem / ScenePlan — the deterministic, locally-supplied list of
planned scenes for one video.

Phase 1C never generates this from an LLM: it only validates scene data
the caller already has (a local JSON/dict fixture, or hand-built objects in
tests). ScenePlan's own validators enforce structural invariants that hold
regardless of pipeline phase (unique IDs, contiguous ordering, valid
role_outfit references); src/core/manifest_builder.py separately enforces
invariants that are specific to being used *as a Phase 1C build* (e.g. "no
scene may claim generated assets yet") or that depend on the active
ChannelPolicy — see that module's docstring for why the split is drawn
there."""
from __future__ import annotations

from pydantic import Field, field_validator, model_validator

from src.models.common import FrozenStrictModel, is_english_ascii_text
from src.models.enums import ApprovalStatus, MotionMode, SceneType
from src.models.overlay import TextOverlay
from src.models.role_outfit import RoleOutfit


class SceneArtifacts(FrozenStrictModel):
    """Placeholders for assets a LATER phase fills in.

    Every field must be None coming out of the Phase 1C builder (see
    manifest_builder.py's "no generated assets yet" check) — but this
    model itself does not forbid non-None values, since a later phase
    reuses this same type once real assets exist. The "must be None right
    now" rule is therefore a Phase-1C-build-time concern, not an intrinsic
    property of SceneArtifacts, so it lives in the builder, not here."""

    audio_path: str | None = None
    image_path: str | None = None
    lipsync_data_path: str | None = None
    rendered_clip_path: str | None = None
    measured_audio_duration_seconds: float | None = None


class ScenePlanItem(FrozenStrictModel):
    scene_id: str = Field(min_length=1, pattern=r"^scene-[0-9]{2,3}$")
    sequence: int = Field(ge=1)
    narration_text: str = Field(min_length=1)
    scene_type: SceneType

    # Free-text production/dramaturgy label (e.g. "hook", "setup",
    # "payoff") — internal planning metadata, not shown to a viewer, so it
    # is not subject to the English-content heuristic below.
    narrative_beat: str = Field(min_length=1)

    role_outfit_id: str | None = None
    text_overlays: tuple[TextOverlay, ...] = ()

    # Internal instruction for a future image-generation stage — not
    # viewer-facing output itself, so (like narrative_beat) not
    # English-heuristic-checked here.
    visual_brief: str = Field(min_length=1)

    motion_mode: MotionMode

    # Optional at the model level on purpose: a scene that never uses
    # manual Flow motion doesn't need one. Whether motion_mode=="manual_flow"
    # WITHOUT this set is actually rejected is a policy-relational
    # question (ChannelPolicy.motion.require_local_fallback_for_manual_flow)
    # and so is enforced by manifest_builder.py, not here. The one thing
    # enforced unconditionally right here is that the fallback itself can
    # never again be "manual_flow" — that would defeat the point of a
    # *local* fallback regardless of any policy value.
    local_fallback_motion_mode: MotionMode | None = None

    flow_task_id: str | None = None
    sfx_refs: tuple[str, ...] = ()
    approval_state: ApprovalStatus
    artifacts: SceneArtifacts = SceneArtifacts()

    @field_validator("narration_text")
    @classmethod
    def _narration_must_be_english(cls, value: str) -> str:
        if not is_english_ascii_text(value):
            raise ValueError(
                "ScenePlanItem.narration_text must be English (ASCII-only heuristic)"
            )
        return value

    @model_validator(mode="after")
    def _fallback_is_never_itself_manual_flow(self) -> "ScenePlanItem":
        if self.local_fallback_motion_mode == "manual_flow":
            raise ValueError(
                "ScenePlanItem.local_fallback_motion_mode cannot itself be 'manual_flow'"
            )
        return self


class ScenePlan(FrozenStrictModel):
    scenes: tuple[ScenePlanItem, ...] = Field(min_length=1)
    role_outfits: tuple[RoleOutfit, ...] = ()

    @model_validator(mode="after")
    def _scenes_are_uniquely_and_contiguously_ordered(self) -> "ScenePlan":
        ids = [s.scene_id for s in self.scenes]
        if len(ids) != len(set(ids)):
            duplicates = sorted({sid for sid in ids if ids.count(sid) > 1})
            raise ValueError(f"ScenePlan has duplicate scene_id(s): {duplicates}")

        sequences = sorted(s.sequence for s in self.scenes)
        expected = list(range(1, len(self.scenes) + 1))
        if sequences != expected:
            raise ValueError(
                "ScenePlan.scenes sequence numbers must be contiguous starting at 1; "
                f"got {sequences}, expected {expected}"
            )
        return self

    @model_validator(mode="after")
    def _role_outfit_references_are_valid(self) -> "ScenePlan":
        catalog = {ro.role_outfit_id: ro for ro in self.role_outfits}

        for scene in self.scenes:
            if scene.role_outfit_id is None:
                continue
            outfit = catalog.get(scene.role_outfit_id)
            if outfit is None:
                raise ValueError(
                    f"scene {scene.scene_id!r} references unknown role_outfit_id "
                    f"{scene.role_outfit_id!r}"
                )
            if scene.scene_id not in outfit.allowed_scene_ids:
                raise ValueError(
                    f"scene {scene.scene_id!r} is not in role_outfit "
                    f"{outfit.role_outfit_id!r}'s allowed_scene_ids"
                )

        identity_refs = {ro.character_identity_ref for ro in self.role_outfits}
        if len(identity_refs) > 1:
            raise ValueError(
                "role_outfits in one ScenePlan must share a single character_identity_ref "
                f"(channel-wide identity lock); found {sorted(identity_refs)}"
            )
        return self
