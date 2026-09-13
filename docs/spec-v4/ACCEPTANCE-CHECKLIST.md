# Acceptance Checklist — spec v4

Concrete, checkable criteria. Use this to verify a phase before calling it
done — do not mark a box checked from intent alone.

## Phase 1A (this change)

- [x] `docs/spec-v4/` exists with the full required layout (README,
      PRODUCT-GUIDE-AR, TECHNICAL-SPEC-EN, IMPLEMENTATION-PLAN,
      ACCEPTANCE-CHECKLIST, LEGACY-MIGRATION, `audit/`, `prompts/`,
      `schemas/`, `examples/`).
- [x] `docs/adr/0001-spec-v4-authority.md` exists and records Accepted
      status, source-of-truth order, and the budget/Manual-Flow/Veo/
      background-music/language constraints.
- [x] `config/channel-config.yaml` exists with every key listed in the task
      spec, and is **not** imported/read by any file under `src/`.
- [x] `python3 -m py_compile` succeeds on every tracked file under `src/`
      and `tests/`.
- [ ] `pytest` runs and its result (pass/fail/blocked) is recorded —
      **blocked in this environment**: `pytest` is not installed and no
      venv exists; it is already listed in `requirements.txt`
      (`pytest==9.1.1`). Not fixed in this phase (would require installing
      a dependency, out of scope for Phase 1A).
- [ ] `python3 -m src.cli --help` runs and its result is recorded —
      **blocked in this environment**: `ModuleNotFoundError: No module
      named 'dotenv'`; `python-dotenv==1.2.3` is already listed in
      `requirements.txt`. Not fixed in this phase.
- [x] `docs/spec-v4/audit/CLAUDE-PHASE-0-GAP-ANALYSIS.md` written, covering
      existing capabilities, missing foundations, outdated docs, job-model/
      CLI risks, provider health-check risks, cost-governance gaps, missing
      tests, deferred items, and Phase 1B blockers.
- [x] `git status --short` / `git diff --stat` show changes only under
      `docs/` and `config/` (no `src/`, no `tests/`, no `.env`).
- [x] No remote API call was made (no network call to Groq, TokenRouter,
      DashScope/Qwen, Kaggle, or any other provider).
- [x] No secret was printed, moved, or committed; `.env` untouched.
- [x] Work happened on a git branch, committed with a scoped message, and
      pushed; a PR was opened against `master` and **not merged**.

## Phase 1B (future — not started, listed for forward reference)

- [ ] `config/channel-config.yaml` is loaded by a settings module under
      `src/` and covered by a unit test.
- [ ] A schema-validation helper rejects a malformed `scene`/`video-manifest`
      document and is covered by a unit test (valid + invalid fixture).
- [ ] No existing test in `tests/` regresses.

## Phase 1C/1D/1E (future — not started, listed for forward reference)

- [ ] A generated `video-manifest.json` validates against
      `schemas/video-manifest.schema.json` for a real (test) story input.
- [ ] Every narration/overlay string in a generated manifest is verified
      English (automated check, not eyeballing).
- [ ] Every scene duration in a generated manifest traces to a measured
      Kokoro audio file, not an estimate.
- [ ] `upscale_image()` runs in the Ken Burns path for at least one test
      scene and produces a measurably sharper frame.
- [ ] `apply_mouth_animation()`/`apply_limb_sway()` run against a real
      illustrated scene background (not just the plain-grey test image) and
      are visually spot-checked for the feathered-edge risk `CLAUDE.md`
      already flags as untested on busy backgrounds.
- [ ] A paid-provider call path is demonstrably blocked when no approved
      `paid-proposal` record exists for it (negative test).
