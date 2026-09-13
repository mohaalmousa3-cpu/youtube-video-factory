# ADR 0001: spec v4 governs the video-factory refactor

## Status

Accepted.

## Context

The repository grew from a series of hand-written, per-video build scripts
(`data/videoN_build.py`) with policy decisions (budget, language, timing,
character consistency) scattered across `CLAUDE.md` prose, ad hoc constants
inside scrapped scripts (e.g. `SAFETY_SUFFIX`, `CHARACTER_ANCHOR` in the
archived `video2_scenes.py`), and tribal knowledge. There was no single,
versioned specification a new script or a new contributor could read to
know what the pipeline is required to do versus what one script happened to
do. A v4 specification (`docs/spec-v4/`) was drafted to fix that.

## Decision

`docs/spec-v4/` (specifically `TECHNICAL-SPEC-EN.md`) governs the refactor
from this point forward. Where it conflicts with an out-of-date statement
in `CLAUDE.md`, the spec wins for future work — but `CLAUDE.md` is not
edited or overridden retroactively as a history record; see
"Incremental migration strategy" below.

### Source-of-truth order

1. `config/channel-config.yaml` — global policy (budget, language, motion,
   audio, timing, visual, character switches). Highest precedence; nothing
   below may override it.
2. `docs/spec-v4/schemas/*.json` — structural contracts. An artifact that
   fails validation is rejected regardless of source.
3. A video's `video-manifest.json` (itself schema-valid) — the concrete
   build plan for that one video, within the shape the schema allows and
   the policy the config sets.
4. `docs/spec-v4/prompts/*.md` — prompt templates. Lowest precedence; a
   prompt is an input to content generation, never a mechanism for
   overriding budget, language, or timing policy.

### English viewer-facing content

All content a viewer sees or hears — narration, on-screen text overlays,
titles, descriptions, thumbnails — must be English, for a global
English-speaking audience. This is a hard, structural requirement, not a
style preference.

### Arabic owner-facing documentation only

Arabic is permitted exclusively in documents addressed to the project
owner for explanation purposes (e.g. `docs/spec-v4/PRODUCT-GUIDE-AR.md`).
Such documents are explanation-only: never read by the pipeline at runtime,
never a source of truth, never a prompt.

### Budget and paid-proposal rules

- Default incremental budget is $0/month. Every currently-wired provider is
  free-tier or self-hosted.
- $25/month is an emergency **ceiling**, not a budget available to spend.
  Reaching for it requires an explicit, human-approved `paid-proposal`
  record — never an automatic decision by the pipeline.
- No code path may enable billing, provision a paid resource, or make a
  payment autonomously. `budget.paid_services_enabled` and
  `budget.automatic_payment_allowed` default to `false` and stay `false`
  until a human flips them, out of band, after reviewing a proposal.

### Manual Flow and Veo constraints

- Google Flow (manual motion polish) is optional and non-blocking: a video
  must be fully producible without it via the local/free pipeline. Flow
  requests are tracked as `manual-flow-task` records, never as hard
  dependencies.
- The Veo API stays disabled (`motion.veo_api_enabled: false`) unless a
  `paid-proposal` for it is explicitly approved (Veo is a paid service).

### Background music disabled by default

`audio.background_music_enabled: false`. No phase in the current
implementation plan proposes turning this on; doing so would itself need
an explicit decision, since typical background-music sourcing has
licensing and (for stock-music APIs) cost implications.

### Incremental migration strategy

- `CLAUDE.md` remains the authoritative *history* of the project (what was
  tried, what broke, what was fixed, why decisions were made) and is not
  rewritten to match v4. Where `CLAUDE.md` describes something the current
  repository state contradicts, that is tracked as documentation drift in
  `docs/spec-v4/audit/CLAUDE-PHASE-0-GAP-ANALYSIS.md`, not silently
  corrected in place.
- Existing `src/providers/*`, `src/render/*`, the `jobs` table, and all
  current tests are preserved as-is. v4 adds a config/schema/manifest layer
  on top; it does not require rewriting working code to adopt.
- Migration proceeds in phases (`docs/spec-v4/IMPLEMENTATION-PLAN.md`);
  each phase has its own acceptance criteria
  (`docs/spec-v4/ACCEPTANCE-CHECKLIST.md`) and is expected to land as its
  own reviewed change, not as one large rewrite.

### Phase 1A scope

Phase 1A (this change) is documentation and configuration only:
`docs/spec-v4/`, `docs/adr/0001-spec-v4-authority.md`, and
`config/channel-config.yaml`. It does not modify `src/` or `tests/`, does
not wire the new config into the runtime, does not call any remote API or
paid service, and does not merge to `master`.

## Consequences

- Future changes to budget, language, motion, timing, overlay, or character
  policy should be proposed as edits to `config/channel-config.yaml` and
  `TECHNICAL-SPEC-EN.md` together, with a new ADR if the decision reverses
  something recorded here.
- Anyone implementing Phase 1B+ can point to this ADR and the spec instead
  of re-deriving policy from scattered `CLAUDE.md` prose and script
  constants.
- The gap between what `CLAUDE.md` describes and what the repository
  actually contains (see the gap analysis) is now a tracked, visible list
  rather than an implicit assumption.
