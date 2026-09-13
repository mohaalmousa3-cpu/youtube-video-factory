# Legacy Migration Notes — spec v4

How the repository's existing history (documented in `CLAUDE.md`) maps onto
this specification. Nothing described here has been executed — this is a
map for Phase 1B+, written during Phase 1A.

## What stays exactly as-is

- All provider modules (`src/providers/*.py`) and their `health_check()`
  contract (`src/providers/base.py`).
- All render modules (`src/render/*.py`), including the validated-but-
  unwired `character_rig.py` functions (`apply_mouth_animation`,
  `apply_limb_sway`, `LimbBox`, `MouthBox`, `rotate_limb`) and
  `image_upscale.py`'s `upscale_image()`.
- The `jobs` table schema and `job()` contextmanager
  (`src/database/db.py`, `src/core/job.py`).
- `tests/test_character_rig.py`, `tests/test_ffmpeg_render.py`,
  `tests/test_job.py` — all three keep passing as-is; v4 adds tests, it
  does not replace these.
- The archived `data/_archived_video2_test/` history (both scrapped video2
  script attempts) — per `CLAUDE.md`, this is moved-not-deleted and
  recoverable; v4 does not touch it and does not reinterpret it as a
  template for the next script.
- `reference/generative-media-skills/` (the vendored OSS skill references)
  — untouched, informational only.

## What v4 formalizes but does not yet build

- `data/videoN_build.py` (per-video, hand-written orchestration) becomes,
  in the target state, a generic orchestrator driven by a
  `video-manifest.json` validated against `schemas/video-manifest.schema.json`.
  Until Phase 1C exists, writing a new `videoN_build.py` by hand (as
  `CLAUDE.md`'s Architecture section says is still expected "for now") is
  not blocked or discouraged — v4 does not remove that option, it adds an
  alternative path.
- `CHARACTER_ANCHOR` / `SAFETY_SUFFIX` (from the scrapped `video2_scenes.py`)
  and the validated-but-unused `COLOR_ANCHOR` (from the Phase 2 A/B test,
  `tools/color_anchor_test/`) become the documented content of
  `prompts/STAGE-PROMPTS.md` in this spec, so the next scene file can pull
  them from one place instead of re-deriving them.
- The "Character animation: current reality vs. Phase-1 plan" note in
  `CLAUDE.md` — that no produced video currently has real mouth/limb
  movement — is the reason Phase 1D exists as its own phase rather than
  being assumed already done.

## What is explicitly out of scope for migration

- `OpenMontage`'s `ink-theater/mocap/` (reviewed, not adopted, per
  `CLAUDE.md`) stays not-adopted. If a future phase revisits character
  movement using motion-capture retargeting, that is a new decision, not
  something this migration plan carries forward.
- The Kaggle GPU credential (`KAGGLE_API_TOKEN`) stays configured-but-unused
  ("not yet used by any actual job" per `CLAUDE.md`) — no phase in this
  plan currently proposes a job for it.

## Known documentation drift to resolve in Phase 1B

See `audit/CLAUDE-PHASE-0-GAP-ANALYSIS.md` for the full list; the short
version: `CLAUDE.md` describes a Windows/local-laptop environment
(`.venv/Scripts/pip`, `winget`, hand-measured GPU checks) that does not
match this container (Linux, no venv present, dependencies from
`requirements.txt` not installed). `CLAUDE.md` is not being edited in
Phase 1A — it remains the authoritative *history* of the project — but
Phase 1B should not assume its environment-setup instructions apply
verbatim to every environment this repo runs in.
