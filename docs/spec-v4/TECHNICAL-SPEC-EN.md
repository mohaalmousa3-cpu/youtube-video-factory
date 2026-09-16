# Technical Specification v4 — YouTube Video Factory

Status: Draft (Phase 1A). Governs the refactor per `docs/adr/0001-spec-v4-authority.md`.
Language: English (this document is the technical source of truth; see
`PRODUCT-GUIDE-AR.md` for the Arabic owner-facing explanation, which is
non-authoritative).

## 1. Purpose

Generalize the current one-off `data/videoN_build.py`-per-video pattern into a
manifest-driven pipeline, without discarding the working providers/renderers
already in `src/`. v4 adds a configuration + schema + manifest layer on top
of the existing code; it does not replace `src/providers/` or `src/render/`.

## 2. High-level architecture

```
config/channel-config.yaml   — global runtime policy (language, budget, motion, audio, timing, visual, character)
        │
        ▼
docs/spec-v4/schemas/*.json  — structural contracts for every artifact below
        │
        ▼
story-input (per idea)  ──▶  scene[] (per video)  ──▶  video-manifest (per video)
        │                                                     │
        ▼                                                     ▼
prompts/CORE-CREATIVE-PROMPT.md + STAGE-PROMPTS.md   src/providers/* + src/render/*
 (versioned prompt templates, referenced by name         (unchanged: Groq, TokenRouter,
  from scene/manifest entries — not hand-copied              Kokoro, Qwen-Image, Rhubarb,
  per script)                                                 Manim, FFmpeg, Real-ESRGAN)
```

A `video-manifest` is the single generated artifact that fully describes one
video's build (scenes, timing, prompts used, character identity reference,
budget declaration). It is what a `videoN_build.py`-equivalent orchestrator
reads and executes; it replaces hand-maintained per-scene duration tables
(see `CLAUDE.md`'s "known bugs already fixed" — a hand-guessed 540s table
that measured out to 249s real).

`role-outfit` records are separate from `scene`/`manifest` because character
identity must stay stable **across** videos, not just within one (see §7).

`manual-flow-task` and `paid-proposal` are out-of-band artifacts: neither is
produced or consumed automatically by the pipeline. They exist so a human
step (Google Flow) or a spending decision (a new paid API) has a structured,
reviewable record instead of an ad hoc chat message.

## 3. Source-of-truth precedence

When two sources disagree, resolve in this order (highest first):

1. **`config/channel-config.yaml`** — global policy. Nothing below this layer
   may override budget, language, or Manual Flow/Veo switches.
2. **`docs/spec-v4/schemas/*.json`** — structural validity. An artifact that
   does not validate against its schema is rejected before it is used,
   regardless of what generated it.
3. **`video-manifest.json`** (per video, itself schema-validated) — the
   concrete build plan for one video. It may only set values the config
   layer leaves open (e.g. per-video scene count, per-video motion choice
   per scene from the allowed `{in, out, pan_lr, pan_up, static}` set).
4. **`docs/spec-v4/prompts/*.md`** — prompt templates. Lowest precedence:
   swapping prompt wording must never be the mechanism for changing budget,
   language, or timing policy. A prompt referenced by a manifest is an
   input to content generation, not a policy override.

Config wins on policy; schemas win on shape; a manifest wins on this video's
concrete plan within that shape; prompts are the most freely editable layer.

## 4. Viewer-facing language policy

All viewer-facing content — narration text, on-screen text overlays, video
titles, descriptions, thumbnails — **must be English** for a global
English-speaking audience. This is enforced structurally: `scene.schema.json`
and `video-manifest.schema.json` require narration/overlay text fields to be
English (see §9 for validation approach). Arabic is permitted only in
owner-facing explanation documents (`PRODUCT-GUIDE-AR.md` and equivalent),
which are never read by the render pipeline and never reach a viewer.

## 5. Budget and cost governance

- `budget.default_incremental_budget_usd: 0` — the pipeline runs at $0/month
  incremental cost by default. Every provider currently wired
  (Groq, TokenRouter, Kokoro, Qwen-Image, Rhubarb, Manim, FFmpeg,
  Real-ESRGAN) is free-tier or self-hosted; see `CLAUDE.md`'s provider table.
- `budget.emergency_monthly_ceiling_usd: 25` — a hard ceiling, **not** a
  budget that is available to spend. Nothing in the pipeline may treat this
  number as headroom to use up automatically.
- `budget.paid_services_enabled: false` and
  `budget.automatic_payment_allowed: false` — no code path may enable
  billing, provision a paid resource, or make a payment autonomously.
- `budget.explicit_user_approval_required: true` — any proposal to spend
  money must be recorded as a `paid-proposal` artifact (see schema) and
  approved by the project owner out of band before anything paid is
  enabled. A `paid-proposal` with `status != "approved"` must never result
  in a paid call being made.
- This governs the whole channel, not just this repo: the sibling
  `youtube-intelligence-engine` and any future automation are bound by the
  same $0-default / $25-ceiling-not-budget rule.

## 6. Manual Flow and Veo constraints

- `motion.veo_api_enabled: false` — the Veo API is disabled by default and
  must stay disabled unless a `paid-proposal` for it is explicitly approved
  (Veo is a paid Google service; see §5).
- `motion.manual_flow_required: false` — Google Flow (manual, human-operated
  motion polish) is **optional**, never required for a video to ship.
- `motion.require_local_fallback_for_manual_flow: true` — every video must
  be fully producible end-to-end using only the local/free pipeline: Ken
  Burns pan/zoom via `ffmpeg_render.ken_burns_clip` (`in`, `out`,
  `pan_lr`, `pan_up`, `static`). Programmatic mouth/limb rigging
  (`character_rig.py`'s `apply_mouth_animation()` / `apply_limb_sway()` /
  `apply_mouth_and_limb_animation()`) is deliberately not part of this
  local fallback — it remains in the repository, fully tested, but
  deferred from the active roadmap (see `IMPLEMENTATION-PLAN.md`). Manual
  Flow is an optional, human-operated handoff to Google Flow — never
  required, never performed locally, and never approximated by local
  animation code; when a human chooses to use it, it is additive polish on
  top of a video that is already complete without it.
- A `manual-flow-task` record (see schema) is how an optional Flow step is
  requested and tracked; it must never carry a hard deadline that blocks
  publication, and the pipeline must produce a valid final video whether or
  not the task is ever picked up.

## 7. Timing policy

`timing.final_timing_source: "measured_audio"` — scene and video duration
are always derived from **measured** TTS output (Kokoro), never estimated
from word count and a target words-per-minute figure. This codifies the bug
already found and fixed once (`CLAUDE.md`: a 9-scene table assumed 540s,
measured 249s). Concretely:

1. Synthesize narration audio for a scene first.
2. Measure its real duration.
3. Derive Ken Burns timing from that measured duration (padding with
   `pad_audio_to_duration()` when the planned screen time exceeds what
   the narration alone would take, per existing `ffmpeg_render.py`
   behavior).

Any manifest or scene record carrying a duration field must trace that
value to a measured audio file, not a table lookup or a wpm estimate.

## 8. Text overlay policy

`visual.deterministic_text_overlays_enabled: true` — on-screen text overlays
are generated **deterministically** from the manifest/scene data (narration
text, timing, a fixed style template), not freely regenerated by an LLM call
per render. The same manifest must always produce the same overlay text and
timing. This also sidesteps the "Qwen-Image bakes in unwanted text/numbers"
bug (`CLAUDE.md`) by keeping overlay text a separate, deterministic
compositing step rather than something baked into a generated image.

## 9. Character identity lock

`character.identity_locked_across_channel: true` — the character's visual
identity (proportions, head shape, palette, default outfit) must stay
consistent across every video on the channel, not just within one video.
Mechanism:

- A canonical `role-outfit` record (see `schemas/role-outfit.schema.json`)
  is the single reference for the character's appearance and any
  role/context-specific outfit variants.
- Every scene image generation call uses `generate_with_reference()`
  (Qwen-Image, per `CLAUDE.md`) seeded with the canonical reference image(s)
  named in the active `role-outfit` record — never plain `generate()`,
  which is known to drift on head shape/proportions/shading scene to scene.
- `CHARACTER_ANCHOR` and `COLOR_ANCHOR` (see
  `prompts/STAGE-PROMPTS.md`) are appended to every scene prompt alongside
  the reference image, for the same consistency reason `COLOR_ANCHOR` was
  validated for in the Phase 2 A/B test.
- Changing the canonical outfit/appearance requires a new `role-outfit`
  record version, not an ad hoc prompt edit on one video.

## 10. Relationship to existing `src/` code

This spec adds a layer; it does not rewrite `src/providers/` or
`src/render/`. Specifically:

- Providers (`llm_groq.py`, `llm_tokenrouter.py`, `tts_kokoro.py`,
  `image_qwen.py`, `lipsync_rhubarb.py`, `image_upscale.py`,
  `manim_render.py`, `ffmpeg_render.py`) keep their existing
  `health_check()` + thin-wrapper contract (`src/providers/base.py`).
- The `jobs` table and `job()` contextmanager (`src/core/job.py`,
  `src/database/db.py`) keep recording one job per provider call; a
  manifest-driven orchestrator is expected to call `job()` per step exactly
  as a hand-written `videoN_build.py` would.
- `image_upscale.py`'s `upscale_image()` is wired into the generation-only
  upscale-assisted Ken Burns path (`src/core/ken_burns_upscale_pipeline.py`,
  the `build-upscaled-ken-burns` CLI command, committed as `bf94d23`).
  Registering the resulting clip as an artifact remains a separate,
  explicit, later step (`register-animation-artifact`), not part of that
  generation call. `character_rig.py`'s `apply_mouth_animation()` /
  `apply_limb_sway()` / `apply_mouth_and_limb_animation()` are also fully
  tested, but are explicitly deferred from the active roadmap (see
  `IMPLEMENTATION-PLAN.md` Phase 1D) — they remain in the repository
  untouched, not scheduled for wiring.

## 11. Non-goals for this document

This spec does not itself change `src/cli.py`, `src/database/db.py`, any
provider, or any test. It does not wire `config/channel-config.yaml` into
the runtime. Those are explicitly deferred; see `IMPLEMENTATION-PLAN.md`.
