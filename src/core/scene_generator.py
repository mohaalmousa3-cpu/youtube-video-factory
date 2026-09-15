"""Phase 1C: LLM-backed Story Input -> Scene Plan generation.

Turns an already-validated, already-approved-for-content StoryInput into a
fully validated ScenePlan, closing the one remaining gap
docs/spec-v4/IMPLEMENTATION-PLAN.md's Phase 1C left open (see
src/core/manifest_builder.py's docstring, and the gap analysis at
docs/spec-v4/audit/CLAUDE-PHASE-0-GAP-ANALYSIS.md section 2: "No
story-input -> scene[] generation step exists in code").

Local reasoning, one remote call per attempt: GroqProvider is primary,
TokenRouterProvider is the documented fallback (see CLAUDE.md's provider
table) — exactly two attempts per provider, four total, never more. No
database access, no filesystem write, no manifest/artifact/stage-transition
side effect of any kind; this module only ever returns a ScenePlan or
raises SceneGenerationError.

NEVER TRUSTED from the LLM: scene_id, sequence, approval_state, artifacts,
role_outfit_id, text_overlays, sfx_refs, flow_task_id. Every one of those
is assigned programmatically (scene_id/sequence from array position,
approval_state hard-coded "draft", the rest left at ScenePlanItem's own
safe defaults) — an LLM-generated plan is never auto-approved and never
claims a role-outfit/overlay/SFX/Flow-task association it did not actually
reason about. The LLM supplies only content fields ScenePlanItem itself
requires: narration_text, scene_type, narrative_beat, visual_brief,
motion_mode.

motion_mode is additionally constrained to exclude "manual_flow" — a
value ScenePlanItem's own Literal type permits, but that ADR 0001 reserves
as something a human requests, never something a generator autonomously
picks (Google Flow is optional/non-blocking, never a hard dependency this
code path may introduce on its own).

Every one of the following is treated as an equally retryable generation
failure, advancing to the next attempt/provider: a ProviderError from
generate() (network/auth/service), a failure lazily CONSTRUCTING
GroqProvider/TokenRouterProvider itself (an SDK client can raise eagerly
on a missing/invalid credential, not just at request time — wrapped as
ProviderError with a static, sanitized message so it counts as one failed
attempt for that provider rather than escaping as a raw exception), non-JSON
output, a JSON object missing/misshaping the top-level "scenes" key, an
empty scenes array, a scene missing a required field, an invalid
scene_type/motion_mode value (including "manual_flow"), non-English
narration_text (caught by ScenePlanItem's own English-content validator —
narrative_beat and visual_brief are deliberately NOT English-checked, by
this module or the model: both are internal planning/instruction fields,
never viewer-facing, matching the model's own existing policy), or any
other ScenePlan/ScenePlanItem pydantic ValidationError. After all four
attempts are exhausted, the public API raises exactly one exception type,
SceneGenerationError, naming which providers were attempted and a concise
final failure reason — never a raw traceback, never the prompt or story
content, never a token count, never a credential or construction-exception
detail, and never the actual value of any field that failed validation or
was rejected (a disallowed motion_mode names the position and the allowed
set, never the received value itself): every ValidationError is rendered
via exc.errors(include_input=False, include_url=False) (see
_summarize_validation_error() below), which keeps only each error's
type/location/message and strips pydantic's own default input_value=
echo — the concrete leak vectors found and closed during this module's own
review (a pydantic ValidationError's default str() embeds the offending
value verbatim; a provider-construction exception can describe credential
state; the original disallowed-motion_mode message echoed the received
value)."""
from __future__ import annotations

import json
from typing import Any, Protocol

from pydantic import ValidationError

from src.models.scene import ScenePlan, ScenePlanItem
from src.models.story import StoryInput
from src.providers.base import ProviderError
from src.utils.channel_config import ChannelPolicy

_MAX_TOKENS = 2048

# Deliberately excludes "manual_flow" — see module docstring.
_ALLOWED_MOTION_MODES = ("in", "out", "pan_lr", "pan_up", "static")


class SceneGenerationError(Exception):
    """Raised only after every provider attempt (GroqProvider x2, then
    TokenRouterProvider x2 — four total) has failed, or if `story_input`'s
    language is not compatible with `channel_policy` (checked before any
    provider call). Message names the provider(s) attempted and a concise,
    sanitized final failure reason. Never includes a raw traceback, the
    prompt, any story content, generated scene content, a token count, or
    a credential — see this module's docstring for the full list of
    failure modes this covers and how ValidationError is sanitized."""


class _TextGeneratingProvider(Protocol):
    def generate(self, prompt: str, *, max_tokens: int = ...) -> str: ...


def _check_language_compatibility(story_input: StoryInput, channel_policy: ChannelPolicy) -> None:
    if story_input.language != channel_policy.channel.language:
        raise SceneGenerationError(
            f"story_input.language {story_input.language!r} is not compatible with "
            f"channel policy language {channel_policy.channel.language!r}"
        )
    if story_input.viewer_facing_language != channel_policy.channel.viewer_facing_language:
        raise SceneGenerationError(
            f"story_input.viewer_facing_language {story_input.viewer_facing_language!r} is "
            f"not compatible with channel policy viewer_facing_language "
            f"{channel_policy.channel.viewer_facing_language!r}"
        )


def _build_prompt(story_input: StoryInput, channel_policy: ChannelPolicy) -> str:
    return (
        "You are planning scenes for a short English-language explainer video.\n\n"
        f"Title: {story_input.title}\n"
        f"Topic: {story_input.topic}\n"
        f"Approximate target duration: {story_input.target_duration_seconds:.0f} seconds "
        "(a rough guide for how many scenes to plan — do not treat this as an exact or "
        "measured value; actual timing is only ever established later from real recorded "
        "narration audio).\n\n"
        "Respond with ONLY a single JSON object, no prose before or after it and no markdown "
        "code fence, shaped exactly like this:\n"
        '{"scenes": [{"narration_text": "...", "scene_type": "...", "narrative_beat": "...", '
        '"visual_brief": "...", "motion_mode": "..."}, ...]}\n\n'
        "Rules for every scene:\n"
        f"- narration_text: the spoken narration for this scene — REQUIRED to be in "
        f"{channel_policy.channel.viewer_facing_language} only; this is the one field a "
        "viewer actually hears.\n"
        "- scene_type: one of establishing, narration, dialogue, transition, closing.\n"
        "- narrative_beat: a short internal production label (e.g. hook, setup, payoff) — "
        f"internal-only, never shown to a viewer; {channel_policy.channel.viewer_facing_language} "
        "is preferred but not required.\n"
        "- visual_brief: a short instruction for a later image-generation step describing "
        f"what the scene should show — internal-only, never shown to a viewer; "
        f"{channel_policy.channel.viewer_facing_language} is preferred but not required.\n"
        f"- motion_mode: one of {', '.join(_ALLOWED_MOTION_MODES)} — never any other value.\n"
        f"Plan at least one scene. narration_text is the only field that MUST be in "
        f"{channel_policy.channel.viewer_facing_language}."
    )


def _summarize_validation_error(exc: ValidationError) -> str:
    """Render `exc` as a value-free, structured summary: each error's
    dotted field location and message only. Deliberately uses
    exc.errors(include_input=False, include_url=False) rather than
    str(exc)/repr(exc)/exc.json() — pydantic's own default rendering
    embeds the offending INPUT VALUE verbatim (e.g. `input_value='...'`
    on a field-level error, or the entire input dict on a model-level
    one), which for a scene-content field could be actual LLM-generated
    text; this strips that (and the errors.pydantic.dev URL) while
    keeping the structural type/location/message information that is
    genuinely useful for a human reading a failure report."""
    parts = []
    for error in exc.errors(include_input=False, include_url=False):
        loc = ".".join(str(p) for p in error["loc"]) or "<root>"
        parts.append(f"{loc}: {error['msg']}")
    return "; ".join(parts)


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines:
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_and_validate(raw_text: str) -> ScenePlan:
    """Parse `raw_text` as this module's LLM output contract and build a
    fully validated ScenePlan. Raises ValueError, wrapping the underlying
    problem, on any failure — the caller treats this uniformly alongside
    ProviderError as one retryable failure class."""
    text = _strip_code_fence(raw_text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"response is not valid JSON: {exc}") from exc

    if not isinstance(data, dict) or "scenes" not in data:
        raise ValueError("response must be a JSON object with a top-level 'scenes' key")

    raw_scenes = data["scenes"]
    if not isinstance(raw_scenes, list) or not raw_scenes:
        raise ValueError("'scenes' must be a non-empty JSON array")

    items: list[ScenePlanItem] = []
    for position, raw_scene in enumerate(raw_scenes, start=1):
        if not isinstance(raw_scene, dict):
            raise ValueError(f"scene at position {position} must be a JSON object")

        motion_mode = raw_scene.get("motion_mode")
        if motion_mode not in _ALLOWED_MOTION_MODES:
            raise ValueError(
                f"scene at position {position} has a disallowed motion_mode "
                f"(expected one of {_ALLOWED_MOTION_MODES}) — the received value is not "
                "included here, since it is untrusted generated content"
            )

        try:
            item = ScenePlanItem(
                scene_id=f"scene-{position:02d}",
                sequence=position,
                narration_text=raw_scene["narration_text"],
                scene_type=raw_scene["scene_type"],
                narrative_beat=raw_scene["narrative_beat"],
                visual_brief=raw_scene["visual_brief"],
                motion_mode=motion_mode,
                approval_state="draft",
            )
        except KeyError as exc:
            raise ValueError(f"scene at position {position} is missing required field {exc}") from exc
        except ValidationError as exc:
            raise ValueError(
                f"scene at position {position} failed validation: {_summarize_validation_error(exc)}"
            ) from exc
        items.append(item)

    try:
        return ScenePlan(scenes=tuple(items))
    except ValidationError as exc:
        raise ValueError(f"generated scene plan failed validation: {_summarize_validation_error(exc)}") from exc


def generate_scene_plan(
    story_input: StoryInput,
    channel_policy: ChannelPolicy,
    *,
    groq_provider: _TextGeneratingProvider | None = None,
    tokenrouter_provider: _TextGeneratingProvider | None = None,
) -> ScenePlan:
    """Generate a fully validated ScenePlan for `story_input`, honoring
    `channel_policy`'s language requirement. `groq_provider`/
    `tokenrouter_provider` are injectable (any object with a matching
    `.generate(prompt, *, max_tokens=...) -> str` method) so callers —
    tests especially — never need the real network-calling providers;
    when omitted, the real GroqProvider()/TokenRouterProvider() are
    constructed lazily, only if actually needed.

    Raises SceneGenerationError if `story_input` is not language-compatible
    with `channel_policy`, or if all four provider attempts (GroqProvider
    x2, TokenRouterProvider x2) fail — see this module's docstring for the
    complete list of what counts as a retryable failure."""
    _check_language_compatibility(story_input, channel_policy)

    prompt = _build_prompt(story_input, channel_policy)

    def _get_groq() -> _TextGeneratingProvider:
        nonlocal groq_provider
        if groq_provider is None:
            from src.providers.llm_groq import GroqProvider

            try:
                groq_provider = GroqProvider()
            except Exception as exc:
                # Construction can fail synchronously (e.g. a missing/invalid
                # credential raised eagerly by the underlying SDK client,
                # not just at request time) — folded into the SAME
                # ProviderError vocabulary the retry loop already handles,
                # so it counts as one failed Groq attempt rather than
                # escaping as a raw, unsanitized exception. The original
                # exception is deliberately NOT interpolated: a client
                # constructor's error can describe credential state, which
                # must never reach the public SceneGenerationError.
                raise ProviderError("GroqProvider initialization failed") from exc
        return groq_provider

    def _get_tokenrouter() -> _TextGeneratingProvider:
        nonlocal tokenrouter_provider
        if tokenrouter_provider is None:
            from src.providers.llm_tokenrouter import TokenRouterProvider

            try:
                tokenrouter_provider = TokenRouterProvider()
            except Exception as exc:
                # Same reasoning as _get_groq() above.
                raise ProviderError("TokenRouterProvider initialization failed") from exc
        return tokenrouter_provider

    attempts: tuple[tuple[str, Any], ...] = (
        ("GroqProvider", _get_groq),
        ("GroqProvider", _get_groq),
        ("TokenRouterProvider", _get_tokenrouter),
        ("TokenRouterProvider", _get_tokenrouter),
    )

    last_provider_name = ""
    last_reason = ""
    for provider_name, get_provider in attempts:
        last_provider_name = provider_name
        try:
            provider = get_provider()
            raw_text = provider.generate(prompt, max_tokens=_MAX_TOKENS)
        except ProviderError as exc:
            last_reason = f"provider call failed: {exc}"
            continue

        try:
            return _parse_and_validate(raw_text)
        except ValueError as exc:
            last_reason = f"invalid generated output: {exc}"
            continue

    raise SceneGenerationError(
        "scene generation failed after attempting GroqProvider (2x) then "
        f"TokenRouterProvider (2x); last failure from {last_provider_name}: {last_reason}"
    )
