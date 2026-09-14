"""Tests for the Phase 1C domain models: StoryInput, RoleOutfit,
TextOverlay, ScenePlanItem, ScenePlan. All fixtures are hand-built local
data — no network, no provider, no LLM."""
import pytest
from pydantic import ValidationError

from src.models.role_outfit import RoleOutfit
from src.models.scene import ScenePlan, ScenePlanItem
from src.models.story import StoryInput
from src.models.overlay import TextOverlay


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


def _valid_overlay(**overrides) -> dict:
    data = dict(
        text="The Cyberball Game",
        position="lower_third",
        style_id="lower-third-v1",
        viewer_facing_language="English",
        deterministic=True,
    )
    data.update(overrides)
    return data


def _valid_role_outfit(**overrides) -> dict:
    data = dict(
        role_outfit_id="narrator-default",
        role_name="narrator",
        outfit_description="teal sweater, warm-grey trousers",
        allowed_scene_ids=("scene-01", "scene-02"),
        character_identity_ref="narrator-identity-v1",
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
    data = dict(
        scenes=(_valid_scene(),),
        role_outfits=(),
    )
    data.update(overrides)
    return data


# ---------------------------------------------------------------------
# StoryInput
# ---------------------------------------------------------------------


def test_story_input_valid_round_trips():
    story = StoryInput.model_validate(_valid_story_input())
    assert story.story_id == "why-we-care-what-people-think"
    assert story.language == "en-US"


def test_story_input_rejects_unknown_field():
    with pytest.raises(ValidationError):
        StoryInput.model_validate(_valid_story_input(unexpected_field="oops"))


def test_story_input_rejects_non_english_title():
    with pytest.raises(ValidationError):
        StoryInput.model_validate(_valid_story_input(title="لماذا نهتم"))


def test_story_input_rejects_non_english_topic():
    with pytest.raises(ValidationError):
        StoryInput.model_validate(_valid_story_input(topic="علم النفس"))


def test_story_input_allows_arabic_owner_notes():
    """owner_notes is explicitly not viewer-facing and is not subject to
    the English-content heuristic."""
    story = StoryInput.model_validate(_valid_story_input(owner_notes="ملاحظة للمالك فقط"))
    assert story.owner_notes == "ملاحظة للمالك فقط"


def test_story_input_rejects_non_positive_duration():
    with pytest.raises(ValidationError):
        StoryInput.model_validate(_valid_story_input(target_duration_seconds=0))


def test_story_input_is_frozen():
    story = StoryInput.model_validate(_valid_story_input())
    with pytest.raises(ValidationError):
        story.title = "Something else"


# ---------------------------------------------------------------------
# TextOverlay
# ---------------------------------------------------------------------


def test_text_overlay_empty_text_is_allowed():
    overlay = TextOverlay.model_validate(_valid_overlay(text=""))
    assert overlay.text == ""


def test_text_overlay_rejects_non_english_text():
    with pytest.raises(ValidationError):
        TextOverlay.model_validate(_valid_overlay(text="مرحبا"))


def test_text_overlay_deterministic_field_is_required():
    """No default: a caller must always state the flag explicitly."""
    with pytest.raises(ValidationError):
        TextOverlay.model_validate({k: v for k, v in _valid_overlay().items() if k != "deterministic"})


def test_text_overlay_deterministic_false_is_constructible_at_model_level():
    """The model itself allows constructing deterministic=False — only
    ManifestBuilder rejects that combination, against policy (see
    test_manifest_builder.py). A bare TextOverlay must still be
    constructible so the builder has something to reject."""
    overlay = TextOverlay.model_validate(_valid_overlay(deterministic=False))
    assert overlay.deterministic is False


def test_text_overlay_rejects_end_before_start():
    with pytest.raises(ValidationError):
        TextOverlay.model_validate(_valid_overlay(start_seconds=5.0, end_seconds=2.0))


def test_text_overlay_rejects_negative_start():
    with pytest.raises(ValidationError):
        TextOverlay.model_validate(_valid_overlay(start_seconds=-1.0))


def test_text_overlay_is_frozen():
    overlay = TextOverlay.model_validate(_valid_overlay())
    with pytest.raises(ValidationError):
        overlay.deterministic = False


# ---------------------------------------------------------------------
# RoleOutfit
# ---------------------------------------------------------------------


def test_role_outfit_requires_at_least_one_allowed_scene():
    with pytest.raises(ValidationError):
        RoleOutfit.model_validate(_valid_role_outfit(allowed_scene_ids=()))


def test_role_outfit_is_frozen():
    outfit = RoleOutfit.model_validate(_valid_role_outfit())
    with pytest.raises(ValidationError):
        outfit.character_identity_ref = "someone-else"


# ---------------------------------------------------------------------
# ScenePlanItem / ScenePlan
# ---------------------------------------------------------------------


def test_scene_plan_item_valid_round_trips():
    scene = ScenePlanItem.model_validate(_valid_scene())
    assert scene.motion_mode == "in"
    assert scene.artifacts.audio_path is None


def test_scene_plan_item_rejects_non_english_narration():
    with pytest.raises(ValidationError):
        ScenePlanItem.model_validate(_valid_scene(narration_text="لماذا نهتم بما يفكر به الناس"))


def test_scene_plan_item_local_fallback_cannot_itself_be_manual_flow():
    with pytest.raises(ValidationError):
        ScenePlanItem.model_validate(
            _valid_scene(motion_mode="manual_flow", local_fallback_motion_mode="manual_flow")
        )


def test_scene_plan_item_manual_flow_without_fallback_is_constructible():
    """Unlike the invariant above, THIS combination is only rejected at
    the ManifestBuilder level (a policy-relational check), not here — see
    test_manifest_builder.py. A bare ScenePlanItem must still be
    constructible so the builder has something to reject."""
    scene = ScenePlanItem.model_validate(_valid_scene(motion_mode="manual_flow"))
    assert scene.local_fallback_motion_mode is None


def test_scene_plan_item_is_frozen():
    scene = ScenePlanItem.model_validate(_valid_scene())
    with pytest.raises(ValidationError):
        scene.approval_state = "rejected"


def test_scene_plan_valid_round_trips():
    plan = ScenePlan.model_validate(_valid_scene_plan())
    assert len(plan.scenes) == 1


def test_scene_plan_rejects_duplicate_scene_ids():
    with pytest.raises(ValidationError):
        ScenePlan.model_validate(
            _valid_scene_plan(
                scenes=(
                    _valid_scene(scene_id="scene-01", sequence=1),
                    _valid_scene(scene_id="scene-01", sequence=2),
                )
            )
        )


def test_scene_plan_rejects_non_contiguous_sequence():
    with pytest.raises(ValidationError):
        ScenePlan.model_validate(
            _valid_scene_plan(
                scenes=(
                    _valid_scene(scene_id="scene-01", sequence=1),
                    _valid_scene(scene_id="scene-02", sequence=3),
                )
            )
        )


def test_scene_plan_rejects_sequence_not_starting_at_one():
    with pytest.raises(ValidationError):
        ScenePlan.model_validate(
            _valid_scene_plan(
                scenes=(
                    _valid_scene(scene_id="scene-01", sequence=2),
                    _valid_scene(scene_id="scene-02", sequence=3),
                )
            )
        )


def test_scene_plan_rejects_unknown_role_outfit_reference():
    with pytest.raises(ValidationError):
        ScenePlan.model_validate(
            _valid_scene_plan(
                scenes=(_valid_scene(role_outfit_id="does-not-exist"),),
                role_outfits=(),
            )
        )


def test_scene_plan_rejects_role_outfit_used_outside_allowed_scenes():
    with pytest.raises(ValidationError):
        ScenePlan.model_validate(
            _valid_scene_plan(
                scenes=(_valid_scene(scene_id="scene-05", sequence=1, role_outfit_id="narrator-default"),),
                role_outfits=(_valid_role_outfit(allowed_scene_ids=("scene-01",)),),
            )
        )


def test_scene_plan_accepts_role_outfit_used_within_allowed_scenes():
    plan = ScenePlan.model_validate(
        _valid_scene_plan(
            scenes=(_valid_scene(scene_id="scene-01", sequence=1, role_outfit_id="narrator-default"),),
            role_outfits=(_valid_role_outfit(allowed_scene_ids=("scene-01",)),),
        )
    )
    assert plan.scenes[0].role_outfit_id == "narrator-default"


def test_scene_plan_rejects_multiple_character_identities():
    with pytest.raises(ValidationError):
        ScenePlan.model_validate(
            _valid_scene_plan(
                scenes=(
                    _valid_scene(scene_id="scene-01", sequence=1, role_outfit_id="outfit-a"),
                    _valid_scene(scene_id="scene-02", sequence=2, role_outfit_id="outfit-b"),
                ),
                role_outfits=(
                    _valid_role_outfit(
                        role_outfit_id="outfit-a",
                        allowed_scene_ids=("scene-01",),
                        character_identity_ref="identity-1",
                    ),
                    _valid_role_outfit(
                        role_outfit_id="outfit-b",
                        allowed_scene_ids=("scene-02",),
                        character_identity_ref="identity-2",
                    ),
                ),
            )
        )


def test_scene_plan_is_frozen():
    plan = ScenePlan.model_validate(_valid_scene_plan())
    with pytest.raises(ValidationError):
        plan.scenes = ()
