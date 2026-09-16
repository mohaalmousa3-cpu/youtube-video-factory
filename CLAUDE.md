# youtube-video-factory

AI-produced video pipeline for a stickman/watercolor-illustration psychology-facts
YouTube channel. Sibling project: `../youtube-intelligence-engine` (the niche-research
engine that picked this niche — see that project's own history for how "psychology
facts / human behavior" was chosen over ~15 other candidates).

**Hard constraint: $0/month.** Every provider below is free-tier or self-hosted.
Laptop = control plane (16GB RAM, no GPU) — nothing GPU-heavy runs locally.

## "Video 2" scrapped — both script attempts archived, not deleted

Two competing scripts existed for the "why we care what people think" video
(`final_script_v4.txt`, hand-built and measured to 8:16, vs. `video2_scenes.py`,
a longer independently-elaborated 15-scene version that actually produced
`video2/final.mp4`). User call: discard both, this whole attempt was a test.
Both scripts, their generated audio/images/clips/lipsync data, and the QA frames
pulled from `final.mp4` are moved to `data/_archived_video2_test/` (not deleted —
recoverable if needed) rather than left loose in `data/`. Next video starts fresh:
no script content should be assumed carried over from this attempt, but everything
in "Architecture" and "Known bugs already fixed" below still applies — those are
lessons about the *pipeline*, not about this specific discarded script.

## In-progress: 4-point pipeline improvement initiative

Separate from any specific video: an approved, phased plan to improve four
things about the pipeline itself (background resolution, body movement, mouth
movement, color/mood consistency), full detail in
`C:\Users\HP\.claude\plans\goofy-frolicking-nova.md` (outside this repo, may
not survive indefinitely — this section is the durable summary).

**Status: all 4 phases done and validated end-to-end** (real audio, real
generated scene images, real output videos — not mocked). None are wired
into a `videoN_build.py` yet since there's no live scene file (see above);
each is a ready-to-use, tested capability waiting for the next script.

- **Phase 0 (gate)**: DashScope account checked by the user directly (2026-09-11)
  — `qwen-image-2.0-pro` confirmed still free, no billing yet. Re-check before
  any large image-generation batch if much time has passed.
- **Phase 1 (background upscaling)**: ✅ — `image_upscale.py`, see provider
  table below.
- **Phase 2 (color/mood consistency)**: ✅ — `COLOR_ANCHOR`, see below.
- **Phase 3 (mouth movement)**: ✅ — `character_rig.py`'s new
  `apply_mouth_animation()` draws the mouth directly onto an already-composited
  full-scene image (NOT the old cutout/rig approach — see "Character animation"
  below). Tested full pipeline: Kokoro narration → `lipsync_rhubarb.get_mouth_cues()`
  → `apply_mouth_animation()` → `ffmpeg_render.assemble_frame_sequence()` →
  a real synced video (`tools/phase3_test/phase3_test.mp4`). Clean result, no
  visible artifacts, on the one scene image it was calibrated against.
  **The mouth box coordinates are scene-specific and hand-measured** — see
  `MouthBox` docstring; there's no automatic mouth detector (checked:
  `requirements.txt` has no landmark-detection library, and one probably
  wouldn't work on this art style anyway, it's trained on real faces).
- **Phase 4 (body movement)**: ✅ (simple case only) — `character_rig.py`'s
  new `LimbBox`/`rotate_limb()`/`apply_limb_sway()`. A single limb, rigid
  crop-rotate-paste around a hand-measured joint pivot. **Two real bugs found
  and fixed during testing, not just designed around in theory:**
  1. A hard rectangular paste produced an obvious rotated-square seam even on
     a plain grey test background — fixed with a feathered elliptical alpha
     mask (`ImageFilter.GaussianBlur` on the mask, not the image).
  2. Even after feathering, the box was too narrow for the limb's actual
     swing at the sway's max angle, so the original static limb peeked out
     next to the rotated one (a ghost/afterimage, not a blend artifact) —
     fixed by widening the box to contain the full range of motion, not just
     the resting pose. **Whoever sizes the next `LimbBox` needs to account
     for max swing angle × distance from pivot, not just the limb's resting
     silhouette.**
  Validated clean on `data/character_test_v1.png` (plain grey background —
  see `tools/phase4_test/phase4_test.mp4`). **Not yet tested on a full
  illustrated scene** — a busy textured background right around the limb is
  likely to show the feathered edge more than the plain test case did, this
  is a real open risk, not a solved problem for production art.
- Sanity tests for both: `tests/test_character_rig.py` (frame counts, viseme
  shapes differ, rotation doesn't crash — not pixel-perfect checks, those were
  done visually, see the `tools/phase{3,4}_test/` videos).
- **`git init`**: proposed, not done — needs explicit separate approval before
  running.

**Roadmap decision (2026-09-17): Phase 3 (mouth movement) and Phase 4
(body movement) are removed from the active roadmap.** Local motion is
Ken Burns only (`in`, `out`, `pan_lr`, `pan_up`, `static`). `manual_flow`
is an optional, human-operated Google Flow handoff — never required,
never performed locally, and never approximated by local animation code.
The mouth/limb modules and CLI commands documented above (Phase 3/4
history preserved as-is) remain in the repository, fully tested, and
untouched — they are simply not wired into any `videoN_build.py`. See
`docs/spec-v4/IMPLEMENTATION-PLAN.md` for the formal record.

**Visual identity v2 (2026-09-17, active policy)**: `CHARACTER_ANCHOR`/
`COLOR_ANCHOR`/`SAFETY_SUFFIX` in `src/core/scene_image_prompt.py` target a
cohesive hand-drawn 2D stick-figure story-animation style (large round
pure-white heads, thin charcoal outlines, warm pastel flat backgrounds) —
see `docs/spec-v4/prompts/STAGE-PROMPTS.md`'s `stage-scene-image-v2`
section for the full text. The prior `stage-scene-image-v1` wording is
preserved there for history.

## Fresh environment setup
`models/` and `tools/` are gitignored (large binaries, no git repo yet anyway) and
won't exist in a fresh checkout/venv. Bring the environment up with:
```
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
# Kokoro weights (~350MB total):
curl -L -o models/kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.0.onnx
curl -L -o models/voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.0.bin
# Rhubarb binary:
gh release download v1.14.0 --repo DanielSWolf/rhubarb-lip-sync --pattern "*Windows*" --dir tools
# then unzip the downloaded file into tools/
# Real-ESRGAN ncnn-vulkan binary + model:
gh release download v0.2.0 --repo xinntao/Real-ESRGAN-ncnn-vulkan --pattern "*windows*" --dir tools
# then unzip, then inside that folder: mkdir models && cd models && curl -L -O https://raw.githubusercontent.com/upscayl/upscayl/main/resources/models/digital-art-4x.bin && curl -L -O https://raw.githubusercontent.com/upscayl/upscayl/main/resources/models/digital-art-4x.param
# FFmpeg (if not already on PATH): winget install --id Gyan.FFmpeg -e
```
Then fill in `.env` (copy `.env.example`) and run `python -m src.cli health` — all 8
providers must report OK before trusting anything else in this repo.

## Architecture

```
src/providers/   — one file per external service, thin wrapper, health_check() + the real call
src/render/      — ffmpeg/manim/character-compositing logic (no external API calls)
src/core/job.py  — job() contextmanager: queued→running→success/failed in the jobs table
src/database/    — sqlite, single `jobs` table
src/cli.py       — `python -m src.cli health` checks all 7 providers live
data/videoN_*.py — per-video orchestration scripts (NOT in src/ — these are one-off
                   build scripts per video, not yet generalized into a reusable
                   pipeline; expect to write a new one per video for now)
```

### Providers wired and confirmed working (as of last check)
| Provider | File | Role | Notes |
|---|---|---|---|
| Groq | `llm_groq.py` | primary LLM | model `openai/gpt-oss-120b` — Groq's free-tier model lineup changes; re-check `client.models.list()` if a model 404s |
| TokenRouter | `llm_tokenrouter.py` | fallback LLM | only `z-ai/glm-5.3-free` confirmed actually free on this account — other "*-free" IDs on the same platform billed real credit when tested, don't trust the label alone |
| Kokoro | `tts_kokoro.py` | TTS, self-hosted (onnxruntime, CPU) | model files in `models/` (not committed, ~350MB) — re-download with `curl -L -o models/kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.0.onnx` and same URL with `voices-v1.0.bin` for the second file, if `models/` is missing. Real measured rate ≈ 158-160 wpm at `DOCUMENTARY_SPEED=0.92`. **Always measure a script's real duration — never estimate from word count and a target wpm.** A 9-scene table once assumed 540s and got 249s. |
| Qwen-Image | `image_qwen.py` | backgrounds + character scenes | DashScope, base_url `.../compatible-mode/v1` for text, native `dashscope` SDK (`base_http_api_url=.../api/v1`) for images — these are different endpoints, don't conflate them. `generate_with_reference()` (model `qwen-image-2.0-pro`, image+text input) is required for any scene with the character in it — plain `generate()` drifts on head shape/proportions/shading scene to scene. Rate limit is tight enough to hit mid-batch (`_call_with_retry` backs off on 429). |
| Rhubarb | `lipsync_rhubarb.py` | mouth-shape timing from audio | binary at `tools/Rhubarb-Lip-Sync-1.14.0-Windows/rhubarb.exe` (not committed, not in pip) — re-download with `gh release download v1.14.0 --repo DanielSWolf/rhubarb-lip-sync --pattern "*Windows*" --dir tools` then unzip, if `tools/` is missing. **Built but not currently consumed by the production pipeline** — see architecture note below. |
| Manim | `manim_render.py` | supporting diagrams only | not used for the main character — see note below |
| FFmpeg | `ffmpeg_render.py` | render/mux | v9.0.1 via winget; `-vsync` was renamed `-fps_mode` in ffmpeg 6+ |
| Kaggle | — | free GPU (~30 GPU-hr/week) | token in `.env`, not yet used by any actual job |
| Real-ESRGAN ncnn-vulkan | `image_upscale.py` | AI upscale before Ken Burns | binary + models at `tools/realesrgan-ncnn-vulkan-v0.2.0-windows/` (not committed, not in pip) — re-download with `gh release download v0.2.0 --repo xinntao/Real-ESRGAN-ncnn-vulkan --pattern "*windows*" --dir tools` then unzip, and pull `digital-art-4x.{bin,param}` from `https://raw.githubusercontent.com/upscayl/upscayl/main/resources/models/` into a `models/` subfolder next to the exe, if `tools/` is missing (the ncnn-vulkan release itself ships no models). Runs on the Intel Iris Xe iGPU via Vulkan (confirmed: `vulkaninfo` reports API 1.4, no CPU fallback needed here). `digital-art-4x` model measured ~6x faster than the generic `upscayl-standard-4x` (30s vs 3min for a 1328x1328 image) with comparable-or-sharper output on this project's flat ink/watercolor style — use it, not `upscayl-standard-4x`. **Not wired into any `videoN_build.py` yet** — call `upscale_image()` on each scene image before `ken_burns_clip`, which still only does a naive non-AI 1.3x `scale=` before its zoompan. `health_check()` doesn't check the binary's return code — `-h` exits 127 even on success (a quirk of this specific binary), so it just checks the binary + model file exist on disk. |

### Character animation: current reality vs. Phase-1 plan
Phase 1 built a cutout-rig approach (`character_rig.py`): one fixed transparent
character PNG + procedurally-drawn mouth shapes (Pillow) synced via Rhubarb, composited
over a separately-generated background. **That is not what `video2_build.py` actually
does.** The pipeline that produced the real `final.mp4` instead generates the character
*already inside* each fully-illustrated scene via `generate_with_reference()`, one
static image per scene, animated only with Ken Burns pan/zoom
(`ffmpeg_render.ken_burns_clip`, `motion` ∈ `{in, out, pan_lr, pan_up, static}`).
**There is no actual mouth/limb movement in any produced video right now** —
`ken_burns_clip` still only pans/zooms a static image. That gap now has a
tested fix ready to wire in (`character_rig.py`'s `apply_mouth_animation()` /
`apply_limb_sway()`, see the pipeline-improvement status above) that works
*with* full-scene generation instead of reviving the cutout approach — but
it's validated on isolated test images, not plugged into any `videoN_build.py`
yet. Don't assume lip-sync or limb movement is active in an actual video
without checking whether that wiring has happened since this was written.

## Known bugs already fixed — don't rediscover these

- **Video shorter than its own narration, audio truncated**: `mux_audio_video` used
  to use `-shortest` blindly. Fixed: it now measures both durations and pads video
  (freeze last frame via `tpad`) whenever audio is longer, `-shortest` only matters
  as a no-op safety net now. Root cause was concat-demuxer duration rounding to the
  output framerate, accumulating drift across many short segments.
- **Scene duration estimates vs. real TTS output**: never trust a hand-guessed
  seconds-per-scene table. `pad_audio_to_duration()` exists for scenes whose planned
  screen time is longer than the words alone would take to say (visual dwell time).
  Always synthesize first, measure, *then* build Ken Burns duration from the real
  audio file.
- **`kokoro-onnx` phonemizer intermittently hangs** (~1 in 10 calls, unrelated to
  text length) **when Popen'd from inside another Python process without its own
  console** on Windows. Fix in `video2_build.py`'s `build_audio()`: run each scene's
  synthesis as its own subprocess with `CREATE_NEW_CONSOLE`, `taskkill /F /T` (not
  a bare `.kill()` — the venv python.exe stub re-launches into a grandchild process
  that survives a plain kill) on timeout, retry up to 3x.
- **Qwen-Image bakes in unwanted text/numbers** even with an explicit "no text"
  instruction in the prompt — confirmed on a fuel-gauge illustration (rendered real
  numbers on the dial) and on a phone-screen close-up (garbled pseudo-text + an
  unrelated ink calligraphy scroll). "No text" alone is not reliable — always
  visually spot-check generated images with text-adjacent objects (screens,
  gauges, dials, signage) before using them.
- **`ffmpeg -vsync vfr`** → renamed `-fps_mode vfr` in ffmpeg 6+; the old flag errors
  out on 9.0.1.
- **Character scale/grounding**: an earlier approach (separately-generated
  background + flatly-composited character cutout) looked pasted-on, wrong scale
  relative to furniture, feet not on the floor plane. Fixed by switching to
  `generate_with_reference()` generating the *whole scene* (character included) in
  one call — perspective/scale come out consistent because the model places the
  character in context itself.
- **video1 style bugs** (see `video2_scenes.py`'s module docstring): pencil
  crosshatch shading and black backgrounds both leaked into video1's outputs from
  underspecified prompts. `SAFETY_SUFFIX` in `video2_scenes.py` now explicitly
  negates both on every prompt.
- **Content-safety note, not a bug**: a script chapter names a real historical hate
  group (Kipling Williams' ostracism research design) for scientific accuracy in
  narration. The corresponding *visual* was deliberately kept abstract (unmarked
  dark silhouette figures, no robes/insignia/identifiable iconography) — never
  render a literal depiction of a real hate group's symbols, regardless of what a
  script's audio track names for historical-accuracy reasons.

## Prompt technique validated, ready for the next script's scene file

**COLOR_ANCHOR** — a mood/color-grading sentence appended to every scene prompt
alongside `CHARACTER_ANCHOR`/`SAFETY_SUFFIX`, same pattern as those two. A/B
tested on the same scene (identical prompt, only this line added) via
`generate_with_reference()` at 1664x928 — real API call, not a mockup:

```
COLOR_ANCHOR = (
    " Consistent warm color grading across the whole scene: soft golden-amber "
    "lighting, gentle warm shadows, calm and emotionally intimate mood, cohesive "
    "muted color palette — avoid harsh contrast or clashing saturated colors."
)
```

Result: the baseline mixed cool charcoal-grey shadows with a warm-lit face pocket;
with `COLOR_ANCHOR` the whole frame reads as one cohesive warm palette — clearly
more consistent scene-to-scene, which was the goal. **Tradeoff worth knowing**:
it also flattens dramatic contrast somewhat (the background shadow in the
with-anchor version reads less moody/atmospheric than the baseline's darker
room). Fine for calm/intimate scenes; reconsider the wording for a scene that
specifically wants tension or gloom. Test images: `tools/color_anchor_test/`.
**Not wired into any scene file yet** — there is no live scene file to put it
in (see "Video 2 scrapped" above). Drop it in alongside `CHARACTER_ANCHOR` when
the next script is written.

## External research/OSS reviewed, not adopted
`OpenMontage` (github.com/calesthio/OpenMontage) was cloned read-only and reviewed —
legitimate, substantial, no malicious patterns found, but AGPL-3.0 and tightly
coupled to its own tool/schema framework. Decision: don't adopt wholesale. Its
`ink-theater/mocap/` submodule (real CMU motion-capture data retargeted onto a
stick figure, genuinely free-for-any-use license) was noted as a stronger approach
to character *movement* than anything built here — not yet implemented. If character
animation becomes a priority again, that's the first thing to look at, but reimplement
the retargeting logic fresh rather than copying `ink-theater.js` itself (AGPL).

## Credentials
All in `.env` (gitignored, no git repo exists yet so this matters less right now but
keep the habit): `GROQ_API_KEY`, `TOKENROUTER_API_KEY`/`TOKENROUTER_BASE_URL`,
`QWEN_API_KEY`/`QWEN_BASE_URL`, `KAGGLE_API_TOKEN`. `.env.example` documents the
shape without values.
