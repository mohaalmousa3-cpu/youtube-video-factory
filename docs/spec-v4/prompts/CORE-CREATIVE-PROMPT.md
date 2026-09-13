# Core Creative Prompt — Channel Identity

This is the channel-level creative brief. It is referenced by name/version
from a `story-input` or `video-manifest` record (see
`schemas/story-input.schema.json`'s `creative_prompt_ref` field) — it is
not copy-pasted per script. Viewer-facing output derived from this prompt
must always be English (see `TECHNICAL-SPEC-EN.md` §4).

```
prompt_id: core-creative-v1
```

## Channel brief

You are writing for a YouTube channel about psychology facts and human
behavior. The visual style is stickman/watercolor illustration — simple
ink-line figures over soft watercolor-style backgrounds, not photorealistic,
not 3D-rendered. Every video features one consistent narrator character
(see the canonical `role-outfit` record for exact appearance) walking the
viewer through a real psychological phenomenon or research finding.

Tone: calm, curious, warm — never clinical or lecture-like, never
sensationalized or clickbait-y in the narration itself (titles/thumbnails
may use stronger hooks, but the narration should earn the claim). Assume an
intelligent, English-speaking adult audience with no psychology background.

Content rules:

- Every factual claim should be attributable to real psychological research
  or a named, real phenomenon — no invented statistics or fabricated
  studies.
- Sensitive historical material (e.g. a named research design that involved
  a real hate group, real trauma, real tragedy) may be referenced in
  narration for accuracy, but the accompanying visual must stay abstract —
  never render real hate-group symbols, identifiable victims, or graphic
  content. See `TECHNICAL-SPEC-EN.md` and `CLAUDE.md`'s content-safety note
  for the precedent (Kipling Williams' ostracism research: unmarked
  silhouette figures only, no robes/insignia).
- No on-screen text, dial, gauge, sign, or screen close-up should be
  assumed "safe" just because the prompt says "no text" — these need a
  visual spot-check per `TECHNICAL-SPEC-EN.md`'s Qwen-Image caveat.

## Output language

English only. This prompt (and everything it produces) is viewer-facing.
