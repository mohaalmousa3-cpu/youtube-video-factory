# Implementation Plan — spec v4

This plan sequences the v4 refactor into phases. Only **Phase 1A** is in
scope for this change; everything else is scoped but not started.

## Phase 1A — Documentation, config, and gap audit (this change)

- Write `docs/spec-v4/*` and `docs/adr/0001-spec-v4-authority.md`.
- Add `config/channel-config.yaml` (not wired into runtime).
- Run safe local baseline checks (syntax, pytest, CLI `--help`) and record
  results, including any missing-dependency gaps, without installing
  anything or calling a remote API.
- Produce a gap analysis comparing the current repo to this spec.
- No file under `src/` or `tests/` changes. No remote provider is called.

**Exit criteria:** this document set exists, baseline checks are recorded,
gap analysis is written, and `git status`/`git diff --stat` show only
`docs/` and `config/` changes.

## Phase 1B — Config wiring and schema validation (not started)

- Load `config/channel-config.yaml` in `src/utils/config.py` (or a new
  `src/utils/channel_config.py`) as an explicit, typed settings object —
  additive to the existing `.env`-based `Settings`, not a replacement.
- Add a schema-validation helper (`jsonschema`/`fastjsonschema`, both
  already in `requirements.txt`) that validates a `story-input`, `scene`,
  or `video-manifest` document against `docs/spec-v4/schemas/*.json` before
  it is used by any build step.
- No behavior change to existing providers/renderers yet — this phase only
  makes the config and schemas *readable and enforceable*, it does not yet
  make any script use them to produce a video.

## Phase 1C — Manifest-driven scene generation (not started)

- Define the process that turns a `story-input` into `scene[]` (LLM-backed,
  via `llm_groq.py`/`llm_tokenrouter.py`) and assembles a `video-manifest`.
- Enforce the English-only viewer-content policy at generation time (reject
  or re-generate any non-English narration/overlay text).
- Enforce `timing.final_timing_source: measured_audio` by requiring a
  measured Kokoro duration before a manifest's scene durations are
  considered final (no wpm-estimated placeholder durations persisted).

## Phase 1D — Wire validated-but-unused render capabilities (not started)

- Wire `image_upscale.upscale_image()` into the Ken Burns path (per
  `CLAUDE.md`: "not wired into any `videoN_build.py` yet").
- **Deferred from the active roadmap** (decision recorded 2026-09-17):
  programmatic mouth/limb rigging via `character_rig.py`'s
  `apply_mouth_animation()` and `apply_limb_sway()` remains in the
  repository, fully tested, but is deferred from the active roadmap and
  is not scheduled to be wired into any build. Local motion stays Ken
  Burns only (`in`, `out`, `pan_lr`, `pan_up`, `static`).
- Apply `COLOR_ANCHOR` + `CHARACTER_ANCHOR` to every scene prompt per the
  character-identity-lock policy (§9 of `TECHNICAL-SPEC-EN.md`).

## Phase 1E — Cost governance enforcement (not started)

- Add a runtime guard that refuses to call a provider marked as paid unless
  a corresponding `paid-proposal` record exists with `status: "approved"`.
- Add a `paid-proposal` review workflow (even a manual one — e.g. a
  human reads and edits the JSON file's `status` field) before Phase 1E is
  considered done; no automatic approval path. **Done**: see
  `data/paid_proposals.json.example` — a human copies it to
  `data/paid_proposals.json` (gitignored, never created automatically) and
  edits `status` to `"approved"` there; `src/core/cost_guard.py` enforces
  that no automatic approval path exists.

## Explicit non-goals for all of the above

None of these phases enable Veo, background music, automatic publishing/
uploads, or any paid service by default. Manual Flow stays optional and
non-blocking in every phase. These constraints are config-level (§5, §6 of
`TECHNICAL-SPEC-EN.md`), not phase-level, and apply throughout.
