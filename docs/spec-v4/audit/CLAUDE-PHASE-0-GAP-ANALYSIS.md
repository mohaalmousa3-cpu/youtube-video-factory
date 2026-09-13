# Gap Analysis — Current Repository vs. spec v4 (Phase 1A)

Compares the actual repository state (as of commit `9ee69bf8`, branch
`claude/keen-johnson-bqq5p9`) — `src/`, `tests/`, `CLAUDE.md`, the database
schema, and the CLI — against `docs/spec-v4/TECHNICAL-SPEC-EN.md`. No
source code was changed to address anything found here; this is a record
for Phase 1B+ planning.

## 1. Existing capabilities to preserve

- Providers (`src/providers/`): Groq, TokenRouter, Kokoro, Qwen-Image,
  Rhubarb, Real-ESRGAN upscale, Manim, FFmpeg — all thin wrappers with a
  `health_check()`, per `src/providers/base.py`'s deliberately minimal
  `ProviderError` contract. None of this needs to change for v4.
- `src/render/character_rig.py` — `MouthBox`/`LimbBox`, `draw_mouth`,
  `rotate_limb`, `apply_mouth_animation`, `apply_limb_sway`. Validated per
  `CLAUDE.md` (Phase 3/4 of the prior improvement initiative) but not
  wired into any build script.
- `src/render/ffmpeg_render.py` — `ken_burns_clip`, `mux_audio_video`
  (fixed to pad video via `tpad` rather than truncate audio),
  `pad_audio_to_duration`, `get_duration_seconds`.
- `src/core/job.py`'s `job()` contextmanager and the `jobs` table
  (`src/database/db.py`) — queued→running→success/failed/retrying bookkeeping.
- `tests/test_job.py`, `tests/test_ffmpeg_render.py`,
  `tests/test_character_rig.py` — all three are sanity/regression tests
  aimed at real bugs already found (duration drift, viseme differentiation,
  rotation not crashing), not placeholders.

## 2. Missing Phase 1 foundations (relative to spec v4)

- **No config loader.** `config/channel-config.yaml` (added this phase) is
  not read anywhere in `src/`. `src/utils/config.py`'s `Settings` dataclass
  only covers API keys/paths from `.env`, not channel policy (language,
  budget, motion, audio, timing, visual, character switches).
- **No schema validation.** `jsonschema` and `fastjsonschema` are both
  already in `requirements.txt` but nothing in `src/` imports either — there
  is no code path that validates a `scene`/`video-manifest`-shaped document
  today, because no such document type exists yet in the runtime.
- **No manifest concept at all.** The current architecture
  (`data/videoN_*.py` per video, per `CLAUDE.md`) has no equivalent of a
  `video-manifest.json`; scene data lives inline in Python source per video.
- **No `story-input` → `scene[]` generation step exists in code.** This is
  entirely new surface for Phase 1C, not a refactor of something existing.

## 3. Outdated documentation (in `CLAUDE.md`, not corrected in this phase)

- `CLAUDE.md`'s "Fresh environment setup" section assumes Windows
  (`.venv/Scripts/pip`, `winget install`, `tools/Rhubarb-Lip-Sync-...-Windows`,
  `realesrgan-ncnn-vulkan-...-windows`). This container is Linux; those
  exact commands do not apply here. `CLAUDE.md` is being preserved as
  project history per the ADR, not rewritten, but Phase 1B should not
  assume it is a working setup script for every environment.
- `CLAUDE.md`'s Architecture section says `python -m src.cli health` "checks
  all 7 providers live," but reading `src/cli.py` directly shows
  `cmd_health` actually checks **8** providers (Groq, TokenRouter, Kokoro,
  Qwen-Image, Rhubarb, Real-ESRGAN, Manim, FFmpeg). Neither number matches
  the provider table further down the same document, which lists **9**
  rows (those 8 plus Kaggle) — and Kaggle has no `health_check()` wired in
  at all (see section 6). Three different counts (7 claimed, 8 actual, 9
  tabled) for what should be one number. Not corrected in `CLAUDE.md`
  itself per the "don't trust outdated statements" instruction — flagging
  rather than silently editing project history.
- `CLAUDE.md` describes `data/videoN_build.py`, `video2_scenes.py`,
  `final_script_v4.txt`, and `data/_archived_video2_test/` as existing —
  none of these are tracked in git (`data/` is gitignored). This is
  expected (per `CLAUDE.md` itself: "no git repo yet anyway" at the time
  much of that history was written), but it means **this gap analysis
  cannot verify those files' current on-disk state** — only what `CLAUDE.md`
  claims about them. Treat any statement about `data/` contents as
  unverified until someone with local disk access confirms it.
- `CLAUDE.md` says "`git init`: proposed, not done — needs explicit separate
  approval before running." A git repository now exists (this commit is on
  `claude/keen-johnson-bqq5p9`, remote `mohaalmousa3-cpu/youtube-video-factory`),
  so this line is stale. Flagged, not edited, in Phase 1A.

## 4. Risks in the job model and CLI

- `src/database/db.py`'s `SCHEMA` comment says `job_type` is one of
  `'script' | 'tts' | 'animation' | 'render'` and `provider` is one of
  `'groq' | 'kokoro' | 'manim' | 'ffmpeg'`, but neither column has a `CHECK`
  constraint enforcing those values (only `status` does). Any typo'd
  `job_type`/`provider` string is silently accepted. Not a blocker for
  Phase 1A; worth a `CHECK` constraint or application-level enum in 1B.
  since a manifest-driven orchestrator will create many more job rows per
  video than a hand-written script did.
- `src/cli.py`'s `cmd_health` treats a provider health-check exception as a
  same-severity `FAILED` as a returned-`False` health check — reasonable
  for a human running `health` interactively, but if a future automated
  Phase 1B step calls this programmatically to gate a build, it currently
  has no way to distinguish "provider is down" from "provider raised
  because of a config bug" (e.g. a missing path). Worth a typed result if
  `cmd_health` grows a non-interactive caller.
- There is no `list-jobs`/`job-status` CLI command — only `health` and
  `init-db`. A manifest-driven pipeline that creates many jobs per video
  will need some way to inspect job history without opening the sqlite
  file by hand. Not urgent for Phase 1A; noted for 1B/1C scoping.

## 5. Risks in provider health checks

- Per `CLAUDE.md`: `image_upscale.py`'s `health_check()` "doesn't check the
  binary's return code — `-h` exits 127 even on success ... so it just
  checks the binary + model file exist on disk." This means `health`
  reporting Real-ESRGAN "OK" is a weaker guarantee than it looks — it does
  not prove the binary actually runs. Documented already in `CLAUDE.md`;
  restated here because a future cost/paid-service gate (Phase 1E) should
  not treat "health OK" as equivalent to "verified working" for this
  provider specifically.
- `lipsync_rhubarb.py`'s Rhubarb binary and `image_upscale.py`'s
  Real-ESRGAN binary are both external, not-pip-installable, not-committed
  binaries (per `CLAUDE.md`'s re-download instructions). Neither exists in
  this container (this phase did not check for their presence, since doing
  so is unrelated to the documentation-only Phase 1A scope, but their
  absence is consistent with `tools/` being gitignored and no setup script
  having run here). Any Phase 1B code that assumes `health_check()` for
  these can be called in CI/this-type-of-container without the binaries
  present will get a false "provider missing" rather than a real signal.
- No provider health check currently distinguishes "credentials missing"
  from "credentials present but invalid" from "network unreachable" — all
  three collapse to the same `FAILED` in `cmd_health`'s output today. A
  cost-governance gate that wants to alert specifically on "an API key
  looks fine but the account might be billing us" would need finer-grained
  signal than exists now.

## 6. Cost governance gaps

- **No code enforces the $0-default / $25-ceiling-not-a-budget policy
  today.** There is currently no `paid-proposal` concept, no runtime check
  that refuses a paid call absent an approved proposal, and nothing that
  would stop a future contributor from hardcoding a paid API call. This
  gap is exactly what `docs/spec-v4/schemas/paid-proposal.schema.json` and
  Phase 1E (`IMPLEMENTATION-PLAN.md`) are meant to close — they close
  nothing yet, since Phase 1A is documentation-only.
- `.env.example` lists `TOKENROUTER_BASE_URL` and confirms (per
  `CLAUDE.md`) that "other '*-free' IDs on the same platform billed real
  credit when tested, don't trust the label alone." There is no automated
  check anywhere that would catch a future code change accidentally
  switching `llm_tokenrouter.py` off the confirmed-free `z-ai/glm-5.3-free`
  model onto a mislabeled one. Worth a regression test once Phase 1B
  builds config/schema enforcement (e.g. asserting the configured model ID
  against an explicit allowlist).
- Kaggle (`KAGGLE_API_TOKEN`, "not yet used by any actual job" per
  `CLAUDE.md`) has no health check in `cmd_health` at all — it's the one
  row in the provider table with no corresponding file/wrapper. Not a cost
  risk today since nothing calls it, but if Phase 1B+ starts using Kaggle's
  free GPU-hour allowance, that allowance itself becomes something worth
  tracking (a free tier with a cap is a governance-relevant resource even
  at $0 cash cost).

## 7. Missing tests

- No test exercises `src/cli.py` at all (neither `cmd_health` nor
  `cmd_init_db`) — both are currently only reachable by running the CLI
  manually. A CLI smoke test (e.g. `init-db` creates the expected schema
  in a tmp dir) would be a reasonable, low-cost Phase 1B addition.
- No test exercises any provider's `health_check()` in a way that doesn't
  require the real binary/network (e.g. no test asserts `image_upscale`'s
  documented "-h exits 127 even on success" quirk is actually handled the
  way `CLAUDE.md` claims — that behavior is currently only verified by
  prose, not by a test).
- No test exists yet for anything schema-related, obviously — no schema
  validator exists in `src/` yet to test. Flagged here so Phase 1B's
  "add a schema-validation helper" acceptance criterion
  (`ACCEPTANCE-CHECKLIST.md`) includes a test from day one rather than
  being added after the fact.
- No test asserts English-only content, deterministic overlay text, or the
  measured-audio-only timing rule — because none of that logic exists in
  `src/` yet. These become required tests once Phase 1C exists, not before.

## 8. Items deferred to later phases (explicitly, not overlooked)

- Wiring `config/channel-config.yaml` into `src/` (Phase 1B).
- Building the schema-validation helper (Phase 1B).
- `story-input` → `scene[]` generation and `video-manifest` assembly
  (Phase 1C).
- Wiring `upscale_image()`, `apply_mouth_animation()`, `apply_limb_sway()`,
  and `COLOR_ANCHOR`/`CHARACTER_ANCHOR` into an actual build (Phase 1D).
  `CLAUDE.md` already flags that `apply_limb_sway()` is untested against a
  busy illustrated background (only tested on a plain grey test image) —
  that risk carries forward unchanged into Phase 1D, this phase does not
  reduce it.
- `paid-proposal` enforcement at the provider-call boundary (Phase 1E).

## 9. Blockers for Phase 1B

- **Environment gap, this container specifically:** no Python virtual
  environment exists here and none of `requirements.txt` is installed
  (confirmed: `pytest` and `python-dotenv` both `ModuleNotFoundError` when
  invoked directly against system Python 3.11.15; see
  `audit/FINAL-AUDIT.md` for the exact commands/output). Phase 1B's
  acceptance criteria (a config loader + schema validator each covered by
  a passing unit test) cannot be verified in **this** environment until a
  venv is built per `CLAUDE.md`'s "Fresh environment setup" section (or an
  equivalent for whatever OS the work actually happens on) — that step
  needs a decision (which environment Phase 1B work happens in) before it
  can start, since it involves installing packages, which Phase 1A was
  explicitly told not to do.
- **No decision yet on where the config loader lives** — extend
  `src/utils/config.py`'s existing `Settings` dataclass, or add a sibling
  `src/utils/channel_config.py`. `IMPLEMENTATION-PLAN.md` proposes either;
  Phase 1B should pick one before writing code, not mid-implementation.
- **No decision yet on schema-validation library choice** between
  `jsonschema` (feature-complete, pure-Python, already a dependency) and
  `fastjsonschema` (faster, compiles a validator function, also already a
  dependency) — both are in `requirements.txt` already so either is
  available without a new dependency; Phase 1B should pick one rather than
  depending on both.
