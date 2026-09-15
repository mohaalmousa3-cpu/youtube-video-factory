"""Tests for src/core/scene_generator.py — Phase 1C's LLM-backed
story-input -> scene[] generation. Fake providers only; no live network or
API call is ever made from this file."""
from __future__ import annotations

import json

import pytest

from src.core.scene_generator import SceneGenerationError, generate_scene_plan
from src.models.story import StoryInput
from src.providers.base import ProviderError
from src.utils.channel_config import ChannelPolicy


class _FakeProvider:
    """Returns each entry in `responses` in order (a str to return, or an
    Exception instance/subclass to raise) — one call consumes one entry.
    Raises AssertionError if called more times than `responses` provides,
    so a test can assert exactly how many calls were made."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def generate(self, prompt: str, *, max_tokens: int = 2048) -> str:
        self.calls.append((prompt, max_tokens))
        if not self._responses:
            raise AssertionError("_FakeProvider called more times than responses were provided")
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _story_input(**overrides) -> StoryInput:
    defaults = dict(
        story_id="why-we-care-what-people-think",
        title="Why We Care So Much What People Think",
        topic="social psychology",
        target_duration_seconds=480.0,
        language="en-US",
        viewer_facing_language="English",
        approval_status="approved",
    )
    defaults.update(overrides)
    return StoryInput(**defaults)


def _channel_policy() -> ChannelPolicy:
    from src.utils.channel_config import get_channel_policy

    return get_channel_policy()


def _valid_response(scene_count: int = 2) -> str:
    scenes = []
    for i in range(1, scene_count + 1):
        scenes.append(
            {
                "narration_text": f"Narration for scene {i}.",
                "scene_type": "narration",
                "narrative_beat": "setup",
                "visual_brief": f"A visual brief for scene {i}.",
                "motion_mode": "static",
            }
        )
    return json.dumps({"scenes": scenes})


# ---------------------------------------------------------------------
# success paths
# ---------------------------------------------------------------------


def test_success_on_first_groq_attempt():
    groq = _FakeProvider([_valid_response(2)])
    tokenrouter = _FakeProvider([])

    plan = generate_scene_plan(
        _story_input(), _channel_policy(), groq_provider=groq, tokenrouter_provider=tokenrouter
    )

    assert len(plan.scenes) == 2
    assert [s.scene_id for s in plan.scenes] == ["scene-01", "scene-02"]
    assert [s.sequence for s in plan.scenes] == [1, 2]
    assert all(s.approval_state == "draft" for s in plan.scenes)
    assert len(groq.calls) == 1
    assert len(tokenrouter.calls) == 0


def test_markdown_fenced_json_is_parsed():
    fenced = "```json\n" + _valid_response(1) + "\n```"
    groq = _FakeProvider([fenced])

    plan = generate_scene_plan(_story_input(), _channel_policy(), groq_provider=groq, tokenrouter_provider=_FakeProvider([]))

    assert len(plan.scenes) == 1


def test_success_on_groq_second_attempt_after_provider_error():
    groq = _FakeProvider([ProviderError("network blip"), _valid_response(1)])
    tokenrouter = _FakeProvider([])

    plan = generate_scene_plan(
        _story_input(), _channel_policy(), groq_provider=groq, tokenrouter_provider=tokenrouter
    )

    assert len(plan.scenes) == 1
    assert len(groq.calls) == 2
    assert len(tokenrouter.calls) == 0


def test_falls_back_to_tokenrouter_after_groq_exhausted():
    groq = _FakeProvider([ProviderError("down"), ProviderError("still down")])
    tokenrouter = _FakeProvider([_valid_response(1)])

    plan = generate_scene_plan(
        _story_input(), _channel_policy(), groq_provider=groq, tokenrouter_provider=tokenrouter
    )

    assert len(plan.scenes) == 1
    assert len(groq.calls) == 2
    assert len(tokenrouter.calls) == 1


def test_malformed_output_and_provider_error_share_the_same_retry_path():
    groq = _FakeProvider(["not json at all", "also not json"])
    tokenrouter = _FakeProvider([_valid_response(1)])

    plan = generate_scene_plan(
        _story_input(), _channel_policy(), groq_provider=groq, tokenrouter_provider=tokenrouter
    )

    assert len(plan.scenes) == 1
    assert len(tokenrouter.calls) == 1


def test_scene_id_and_sequence_are_always_assigned_by_position_never_trusted_from_llm():
    response = json.dumps(
        {
            "scenes": [
                {
                    "scene_id": "scene-99",
                    "sequence": 42,
                    "approval_state": "approved",
                    "narration_text": "Hello there.",
                    "scene_type": "narration",
                    "narrative_beat": "setup",
                    "visual_brief": "A visual.",
                    "motion_mode": "static",
                }
            ]
        }
    )
    groq = _FakeProvider([response])

    plan = generate_scene_plan(_story_input(), _channel_policy(), groq_provider=groq, tokenrouter_provider=_FakeProvider([]))

    assert plan.scenes[0].scene_id == "scene-01"
    assert plan.scenes[0].sequence == 1
    assert plan.scenes[0].approval_state == "draft"  # never trusted from the LLM either


def test_role_outfit_text_overlays_sfx_flow_task_artifacts_stay_at_safe_defaults():
    groq = _FakeProvider([_valid_response(1)])

    plan = generate_scene_plan(_story_input(), _channel_policy(), groq_provider=groq, tokenrouter_provider=_FakeProvider([]))

    scene = plan.scenes[0]
    assert scene.role_outfit_id is None
    assert scene.text_overlays == ()
    assert scene.sfx_refs == ()
    assert scene.flow_task_id is None
    assert scene.artifacts.audio_path is None
    assert scene.artifacts.image_path is None
    assert scene.artifacts.lipsync_data_path is None
    assert scene.artifacts.rendered_clip_path is None
    assert scene.artifacts.measured_audio_duration_seconds is None


# ---------------------------------------------------------------------
# every attempt exhausted -> SceneGenerationError, never a raw exception
# ---------------------------------------------------------------------


def test_all_four_attempts_exhausted_raises_scene_generation_error():
    groq = _FakeProvider([ProviderError("down"), ProviderError("down")])
    tokenrouter = _FakeProvider([ProviderError("down too"), ProviderError("still down")])

    with pytest.raises(SceneGenerationError) as excinfo:
        generate_scene_plan(_story_input(), _channel_policy(), groq_provider=groq, tokenrouter_provider=tokenrouter)

    message = str(excinfo.value)
    assert "GroqProvider" in message
    assert "TokenRouterProvider" in message
    assert len(groq.calls) == 2
    assert len(tokenrouter.calls) == 2


def test_max_four_total_provider_calls_never_exceeded():
    groq = _FakeProvider(["bad", "bad"])
    tokenrouter = _FakeProvider(["bad", "bad"])

    with pytest.raises(SceneGenerationError):
        generate_scene_plan(_story_input(), _channel_policy(), groq_provider=groq, tokenrouter_provider=tokenrouter)

    assert len(groq.calls) + len(tokenrouter.calls) == 4


# ---------------------------------------------------------------------
# content-leak regression: a ValidationError must never echo generated
# scene content into the final SceneGenerationError message
# ---------------------------------------------------------------------


def test_all_four_attempts_exhausted_with_per_scene_validation_error_is_sanitized():
    """narration_text fails ScenePlanItem's own English-content validator
    on every one of the four attempts (a real, naturally-reachable
    per-scene ValidationError — no monkeypatching needed). The final
    SceneGenerationError must retain useful structural information
    (which field, what kind of problem) but must never echo the actual
    sentinel text that was generated — pydantic's own default
    ValidationError rendering would otherwise embed it verbatim via
    input_value=."""
    narration_sentinel = "SENTINEL_NARRATION_PERSCENE_ABCDEFG789_مرحبا"
    visual_sentinel = "SENTINEL_VISUALBRIEF_PERSCENE_HIJKLMN012"
    bad_response = json.dumps(
        {
            "scenes": [
                {
                    "narration_text": narration_sentinel,  # non-English -> real field validator failure
                    "scene_type": "narration",
                    "narrative_beat": "setup",
                    "visual_brief": visual_sentinel,
                    "motion_mode": "static",
                }
            ]
        }
    )
    groq = _FakeProvider([bad_response, bad_response])
    tokenrouter = _FakeProvider([bad_response, bad_response])

    with pytest.raises(SceneGenerationError) as excinfo:
        generate_scene_plan(_story_input(), _channel_policy(), groq_provider=groq, tokenrouter_provider=tokenrouter)

    message = str(excinfo.value)
    assert narration_sentinel not in message
    assert visual_sentinel not in message
    assert "narration_text" in message  # useful structural info retained
    assert "GroqProvider" in message
    assert "TokenRouterProvider" in message
    assert len(groq.calls) == 2
    assert len(tokenrouter.calls) == 2


def test_all_four_attempts_exhausted_with_scene_plan_level_validation_error_is_sanitized(monkeypatch):
    """ScenePlan's own validators (duplicate scene_id, non-contiguous
    sequence, invalid role_outfit reference) can never actually fire
    through generate_scene_plan()'s real path: scene_id/sequence are
    always assigned by array position (never trusted from the LLM) and
    role_outfit_id/role_outfits are always None/empty. This test forces
    that ValidationError anyway, via a monkeypatched ScenePlan
    constructor inside the scene_generator module, to prove
    _summarize_validation_error() sanitizes THIS case too — where
    pydantic's default rendering would otherwise echo the entire input
    dict (every scene's full content), not just one field's value."""
    import src.core.scene_generator as scene_generator_module
    from pydantic import ValidationError as RealValidationError
    from src.models.scene import ScenePlan as RealScenePlan, ScenePlanItem

    narration_sentinel = "SENTINEL_NARRATION_PLANLEVEL_QWERTY123"
    visual_sentinel = "SENTINEL_VISUALBRIEF_PLANLEVEL_ZXCVB456"

    item1 = ScenePlanItem(
        scene_id="scene-01", sequence=1, narration_text=narration_sentinel, scene_type="narration",
        narrative_beat="hook", visual_brief=visual_sentinel, motion_mode="static", approval_state="draft",
    )
    item2 = ScenePlanItem(
        scene_id="scene-01", sequence=2, narration_text="other text", scene_type="narration",
        narrative_beat="setup", visual_brief="other visual", motion_mode="static", approval_state="draft",
    )
    try:
        RealScenePlan(scenes=(item1, item2))
        raise AssertionError("expected a real ValidationError from duplicate scene_id")
    except RealValidationError as captured:
        captured_error = captured

    def _always_raise_plan_level_error(*args, **kwargs):
        raise captured_error

    monkeypatch.setattr(scene_generator_module, "ScenePlan", _always_raise_plan_level_error)

    valid_looking_response = json.dumps(
        {
            "scenes": [
                {
                    "narration_text": "Some narration.",
                    "scene_type": "narration",
                    "narrative_beat": "setup",
                    "visual_brief": "Some visual.",
                    "motion_mode": "static",
                }
            ]
        }
    )
    groq = _FakeProvider([valid_looking_response, valid_looking_response])
    tokenrouter = _FakeProvider([valid_looking_response, valid_looking_response])

    with pytest.raises(SceneGenerationError) as excinfo:
        generate_scene_plan(_story_input(), _channel_policy(), groq_provider=groq, tokenrouter_provider=tokenrouter)

    message = str(excinfo.value)
    assert narration_sentinel not in message
    assert visual_sentinel not in message
    assert "duplicate scene_id" in message  # useful structural info retained
    assert len(groq.calls) == 2
    assert len(tokenrouter.calls) == 2


def test_all_four_attempts_exhausted_with_invalid_motion_mode_is_sanitized():
    """A disallowed motion_mode value, on every one of the four attempts.
    The final SceneGenerationError must state the position and that the
    value was disallowed, but must never echo the actual (sentinel)
    value that was received — the LLM-supplied field is untrusted
    generated content."""
    motion_mode_sentinel = "SENTINEL_MOTIONMODE_BADVALUE_EDC333"
    bad_response = json.dumps(
        {
            "scenes": [
                {
                    "narration_text": "Some narration.",
                    "scene_type": "narration",
                    "narrative_beat": "setup",
                    "visual_brief": "Some visual.",
                    "motion_mode": motion_mode_sentinel,
                }
            ]
        }
    )
    groq = _FakeProvider([bad_response, bad_response])
    tokenrouter = _FakeProvider([bad_response, bad_response])

    with pytest.raises(SceneGenerationError) as excinfo:
        generate_scene_plan(_story_input(), _channel_policy(), groq_provider=groq, tokenrouter_provider=tokenrouter)

    message = str(excinfo.value)
    assert motion_mode_sentinel not in message
    assert "disallowed motion_mode" in message  # useful structural info retained
    assert len(groq.calls) == 2
    assert len(tokenrouter.calls) == 2


# ---------------------------------------------------------------------
# provider-construction failures must count as a failed attempt for that
# provider, never escape as a raw exception
# ---------------------------------------------------------------------


def test_all_four_attempts_exhausted_when_groq_construction_always_fails(monkeypatch):
    """GroqProvider's real constructor is monkeypatched to raise an
    arbitrary non-ProviderError exception on every attempt — simulating,
    e.g., a missing/invalid credential raised eagerly by the underlying
    SDK client at construction time, not just at request time.
    TokenRouter (a supplied fake, since only Groq's construction path is
    under test here) also fails both of its attempts via an ordinary
    ProviderError from generate(). The only thing that may escape is
    SceneGenerationError — no raw traceback, and the constructor
    exception's own sentinel text must never appear in it."""
    groq_ctor_sentinel = "SENTINEL_GROQ_CTOR_FAILURE_QAZ111"
    call_count = {"n": 0}

    def _boom(*args, **kwargs):
        call_count["n"] += 1
        raise RuntimeError(groq_ctor_sentinel)

    monkeypatch.setattr("src.providers.llm_groq.GroqProvider", _boom)

    tokenrouter = _FakeProvider([ProviderError("tokenrouter down"), ProviderError("tokenrouter still down")])

    with pytest.raises(SceneGenerationError) as excinfo:
        generate_scene_plan(_story_input(), _channel_policy(), tokenrouter_provider=tokenrouter)

    message = str(excinfo.value)
    assert groq_ctor_sentinel not in message
    assert "GroqProvider" in message
    assert "TokenRouterProvider" in message
    assert call_count["n"] == 2  # exactly two Groq construction attempts, never more
    assert len(tokenrouter.calls) == 2  # exactly two TokenRouter attempts


def test_all_four_attempts_exhausted_when_tokenrouter_construction_always_fails(monkeypatch):
    """Groq (a supplied fake) fails both its attempts via an ordinary
    ProviderError from generate(); TokenRouterProvider's real constructor
    is then monkeypatched to raise an arbitrary non-ProviderError
    exception on both of ITS attempts — proving the SAME construction-
    failure handling applies symmetrically to the fallback provider, not
    just the primary one."""
    tokenrouter_ctor_sentinel = "SENTINEL_TOKENROUTER_CTOR_FAILURE_WSX222"
    call_count = {"n": 0}

    def _boom(*args, **kwargs):
        call_count["n"] += 1
        raise ValueError(tokenrouter_ctor_sentinel)

    monkeypatch.setattr("src.providers.llm_tokenrouter.TokenRouterProvider", _boom)

    groq = _FakeProvider([ProviderError("groq down"), ProviderError("groq still down")])

    with pytest.raises(SceneGenerationError) as excinfo:
        generate_scene_plan(_story_input(), _channel_policy(), groq_provider=groq)

    message = str(excinfo.value)
    assert tokenrouter_ctor_sentinel not in message
    assert "GroqProvider" in message
    assert "TokenRouterProvider" in message
    assert len(groq.calls) == 2  # exactly two Groq attempts
    assert call_count["n"] == 2  # exactly two TokenRouter construction attempts, never more


def test_groq_construction_failure_falls_back_to_tokenrouter_successfully(monkeypatch):
    """A single Groq construction failure (both of its attempts) must not
    prevent a subsequent TokenRouter success — proving construction
    failures are treated as ordinary retryable attempts, not a hard stop,
    and that the real GroqProvider is never actually reached again once
    it starts failing at construction."""
    call_count = {"n": 0}

    def _boom(*args, **kwargs):
        call_count["n"] += 1
        raise RuntimeError("groq unavailable in this environment")

    monkeypatch.setattr("src.providers.llm_groq.GroqProvider", _boom)

    tokenrouter = _FakeProvider([_valid_response(1)])

    plan = generate_scene_plan(_story_input(), _channel_policy(), tokenrouter_provider=tokenrouter)

    assert len(plan.scenes) == 1
    assert call_count["n"] == 2
    assert len(tokenrouter.calls) == 1


# ---------------------------------------------------------------------
# malformed-output failure modes, each individually
# ---------------------------------------------------------------------


def test_non_json_response_is_retryable_not_a_crash():
    groq = _FakeProvider(["this is not json"] * 2)
    tokenrouter = _FakeProvider(["still not json"] * 2)

    with pytest.raises(SceneGenerationError):
        generate_scene_plan(_story_input(), _channel_policy(), groq_provider=groq, tokenrouter_provider=tokenrouter)


def test_missing_scenes_key_is_retryable():
    bad = json.dumps({"not_scenes": []})
    groq = _FakeProvider([bad, bad])
    tokenrouter = _FakeProvider([bad, bad])

    with pytest.raises(SceneGenerationError):
        generate_scene_plan(_story_input(), _channel_policy(), groq_provider=groq, tokenrouter_provider=tokenrouter)


def test_empty_scenes_array_is_retryable():
    bad = json.dumps({"scenes": []})
    groq = _FakeProvider([bad, bad])
    tokenrouter = _FakeProvider([bad, bad])

    with pytest.raises(SceneGenerationError):
        generate_scene_plan(_story_input(), _channel_policy(), groq_provider=groq, tokenrouter_provider=tokenrouter)


def test_missing_required_field_is_retryable():
    bad = json.dumps({"scenes": [{"narration_text": "hi"}]})  # missing scene_type etc.
    groq = _FakeProvider([bad, bad])
    tokenrouter = _FakeProvider([bad, bad])

    with pytest.raises(SceneGenerationError):
        generate_scene_plan(_story_input(), _channel_policy(), groq_provider=groq, tokenrouter_provider=tokenrouter)


def test_invalid_scene_type_is_retryable():
    bad = json.dumps(
        {
            "scenes": [
                {
                    "narration_text": "hi",
                    "scene_type": "not-a-real-type",
                    "narrative_beat": "setup",
                    "visual_brief": "a visual",
                    "motion_mode": "static",
                }
            ]
        }
    )
    groq = _FakeProvider([bad, bad])
    tokenrouter = _FakeProvider([bad, bad])

    with pytest.raises(SceneGenerationError):
        generate_scene_plan(_story_input(), _channel_policy(), groq_provider=groq, tokenrouter_provider=tokenrouter)


def test_manual_flow_motion_mode_is_rejected_as_disallowed():
    """manual_flow is a legal MotionMode on the model itself but is
    explicitly excluded from what this generator will accept — ADR 0001:
    Flow is something a human requests, never something a generator
    autonomously picks."""
    bad = json.dumps(
        {
            "scenes": [
                {
                    "narration_text": "hi",
                    "scene_type": "narration",
                    "narrative_beat": "setup",
                    "visual_brief": "a visual",
                    "motion_mode": "manual_flow",
                }
            ]
        }
    )
    groq = _FakeProvider([bad, bad])
    tokenrouter = _FakeProvider([bad, bad])

    with pytest.raises(SceneGenerationError) as excinfo:
        generate_scene_plan(_story_input(), _channel_policy(), groq_provider=groq, tokenrouter_provider=tokenrouter)
    assert "manual_flow" in str(excinfo.value) or "disallowed" in str(excinfo.value)


def test_non_english_narration_is_retryable():
    bad = json.dumps(
        {
            "scenes": [
                {
                    "narration_text": "مرحبا بكم",  # Arabic — must be rejected
                    "scene_type": "narration",
                    "narrative_beat": "setup",
                    "visual_brief": "a visual",
                    "motion_mode": "static",
                }
            ]
        }
    )
    groq = _FakeProvider([bad, bad])
    tokenrouter = _FakeProvider([bad, bad])

    with pytest.raises(SceneGenerationError):
        generate_scene_plan(_story_input(), _channel_policy(), groq_provider=groq, tokenrouter_provider=tokenrouter)


def test_non_english_visual_brief_is_accepted_not_viewer_facing():
    """visual_brief is deliberately NOT English-checked by ScenePlanItem
    itself (src/models/scene.py: "Internal instruction for a future
    image-generation stage — not viewer-facing... not subject to the
    English-content heuristic"). This generator relies on the EXISTING
    model validation only (per the approved scope) and must not impose a
    stricter rule than the model itself does."""
    response = json.dumps(
        {
            "scenes": [
                {
                    "narration_text": "hi",
                    "scene_type": "narration",
                    "narrative_beat": "setup",
                    "visual_brief": "一个视觉简报",  # Chinese — legitimately accepted
                    "motion_mode": "static",
                }
            ]
        }
    )
    groq = _FakeProvider([response])

    plan = generate_scene_plan(
        _story_input(), _channel_policy(), groq_provider=groq, tokenrouter_provider=_FakeProvider([])
    )
    assert plan.scenes[0].visual_brief == "一个视觉简报"


def test_non_english_narrative_beat_is_accepted_not_viewer_facing():
    """Symmetric with the visual_brief test above: narrative_beat is also
    deliberately NOT English-checked by ScenePlanItem itself
    (src/models/scene.py: "internal planning metadata, not shown to a
    viewer, so it is not subject to the English-content heuristic")."""
    response = json.dumps(
        {
            "scenes": [
                {
                    "narration_text": "hi",
                    "scene_type": "narration",
                    "narrative_beat": "مقدمة",  # Arabic — legitimately accepted
                    "visual_brief": "a visual",
                    "motion_mode": "static",
                }
            ]
        }
    )
    groq = _FakeProvider([response])

    plan = generate_scene_plan(
        _story_input(), _channel_policy(), groq_provider=groq, tokenrouter_provider=_FakeProvider([])
    )
    assert plan.scenes[0].narrative_beat == "مقدمة"


def test_scene_that_is_not_a_json_object_is_retryable():
    bad = json.dumps({"scenes": ["just a string, not an object"]})
    groq = _FakeProvider([bad, bad])
    tokenrouter = _FakeProvider([bad, bad])

    with pytest.raises(SceneGenerationError):
        generate_scene_plan(_story_input(), _channel_policy(), groq_provider=groq, tokenrouter_provider=tokenrouter)


# ---------------------------------------------------------------------
# language/policy compatibility — checked before any provider call
# ---------------------------------------------------------------------


def test_language_incompatible_story_input_rejected_before_any_provider_call():
    incompatible = _story_input(language="ar-SA", viewer_facing_language="Arabic")
    groq = _FakeProvider([])
    tokenrouter = _FakeProvider([])

    with pytest.raises(SceneGenerationError) as excinfo:
        generate_scene_plan(incompatible, _channel_policy(), groq_provider=groq, tokenrouter_provider=tokenrouter)

    assert "language" in str(excinfo.value)
    assert len(groq.calls) == 0
    assert len(tokenrouter.calls) == 0


# ---------------------------------------------------------------------
# no live provider is ever constructed when fakes are supplied; the
# module-level import of the real providers only happens lazily
# ---------------------------------------------------------------------


def test_real_providers_are_never_constructed_when_fakes_are_supplied(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("real GroqProvider/TokenRouterProvider must never be constructed in this test")

    monkeypatch.setattr("src.providers.llm_groq.GroqProvider", _boom, raising=False)
    monkeypatch.setattr("src.providers.llm_tokenrouter.TokenRouterProvider", _boom, raising=False)

    groq = _FakeProvider([_valid_response(1)])
    plan = generate_scene_plan(_story_input(), _channel_policy(), groq_provider=groq, tokenrouter_provider=_FakeProvider([]))
    assert len(plan.scenes) == 1
