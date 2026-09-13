# spec-v4 — Video Factory Specification (Phase 1A)

This directory is the v4 specification for the `youtube-video-factory`
refactor. It is documentation and configuration only as of Phase 1A: nothing
here is wired into `src/` yet. See `docs/adr/0001-spec-v4-authority.md` for
why this spec exists and what governs precedence.

## Contents

| File | Purpose |
|---|---|
| `TECHNICAL-SPEC-EN.md` | Authoritative technical spec (English). Architecture, source-of-truth order, budget/language/motion/timing/overlay/character policy. |
| `PRODUCT-GUIDE-AR.md` | Arabic explanation for the project owner. **Explanation only — never a runtime prompt or source of truth.** |
| `IMPLEMENTATION-PLAN.md` | Phased plan from this doc set to a wired pipeline (Phase 1A done here; Phase 1B+ scoped, not started). |
| `ACCEPTANCE-CHECKLIST.md` | Concrete, checkable criteria for each phase to be considered done. |
| `LEGACY-MIGRATION.md` | How the existing repo (CLAUDE.md history, `data/videoN_*.py` pattern, archived video2 attempt) maps onto this spec. |
| `audit/FINAL-AUDIT.md` | Phase 1A close-out: what was verified, what was not touched, confirmation of scope. |
| `audit/CLAUDE-PHASE-0-GAP-ANALYSIS.md` | Gap analysis: current repo vs. v4 spec. |
| `prompts/CORE-CREATIVE-PROMPT.md` | The channel's core creative/tone prompt template. |
| `prompts/STAGE-PROMPTS.md` | Per-stage prompt templates (script, scene breakdown, image prompt anchors, TTS prep). |
| `schemas/*.json` | JSON Schema (2020-12) contracts for every structured artifact in the pipeline. |
| `examples/*.json` | Example documents, each validating against its corresponding schema. |

## Reading order

1. `docs/adr/0001-spec-v4-authority.md` (why, and precedence rules)
2. `TECHNICAL-SPEC-EN.md` (what)
3. `IMPLEMENTATION-PLAN.md` + `ACCEPTANCE-CHECKLIST.md` (how, and how to know it worked)
4. `audit/CLAUDE-PHASE-0-GAP-ANALYSIS.md` (what's missing today)
5. `schemas/` + `examples/` + `prompts/` (the concrete contracts)

## Scope of Phase 1A

Phase 1A produced this directory, `config/channel-config.yaml`, and the ADR.
It changed nothing under `src/` or `tests/`, called no remote API, and did
not enable any paid service. Phase 1B (wiring config into runtime, building
a manifest-driven orchestrator, generating a real scene file) has not
started.
