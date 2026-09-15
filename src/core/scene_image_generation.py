"""Phase 1D: generation-only scene-image creation via
image_qwen.QwenImageProvider.generate_with_reference_and_download() — the
first real (paid, network) generation call anywhere in this pipeline. See
src/cli.py's cmd_build_scene_image for the CLI command.

Writes only --output (via the provider itself); never registers an
artifact, saves a manifest, or touches SQLite — see
src/core/visual_artifact_registrar.py's existing, unmodified
register-visual-artifact command for that separate, later step.

Required validation order, nothing is created and no provider is
constructed until every check below has passed:
  1. reference_image_path exists and is a regular file (never opened or
     decoded — the provider itself reads its bytes).
  2. output_path.parent exists.
  3. output_path does not already exist. This is not optional:
     QwenImageProvider._download() writes to out_path unconditionally,
     with no existence check of its own, so it would silently overwrite a
     pre-existing file if this module did not refuse first.
  4. build_scene_image_prompt(scene.visual_brief, ...) — pure, reused
     unmodified from src.core.scene_image_prompt.
  5. require_paid_approval("qwen-image", proposals_path, is_paid=True) —
     Qwen-Image is treated as a fixed paid service. Unlike
     src.core.scene_generator's Groq/TokenRouter calls (genuinely
     documented free-tier, is_paid=False), Qwen-Image's free status is a
     human-reverified, time-bound operational fact (CLAUDE.md: "confirmed
     still free... re-check if much time has passed"), not a code-level
     guarantee — ambiguous pricing fails closed, so is_paid=True is fixed
     here and is never exposed as a CLI/caller-settable option. A
     PaidApprovalRequiredError from this call is never caught anywhere in
     this module or in cmd_build_scene_image — it propagates immediately,
     matching src.core.scene_generator and cmd_health's own guard wiring.

Only after 1-5 pass does QwenImageProvider get constructed and called —
the one real network request in this module.

Cleanup proof: because check 3 already proved output_path did not exist
before this function touched anything, any file found there once
generation has begun was necessarily written by THIS SAME call's provider
invocation — deleting it can never remove a pre-existing caller file. This
holds for any exception raised from the provider call onward (a
ProviderError, a Pillow decode failure, an unexpected format, or any other
exception type), via a success-flag + `finally` (never a bare `except
Exception`), matching src.core.ken_burns_upscale_pipeline.py's and
src.core.limb_sway_generation.py's own cleanup pattern exactly.

Errors from the provider are translated into one domain exception,
SceneImageGenerationError, with a short, fixed message per failure
category — never a wrapped exception's own raw text (which could carry an
HTTP response body or other provider detail), never a resolved filesystem
path, never prompt content, and never an API key."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError

from src.core.cost_guard import require_paid_approval
from src.core.scene_image_prompt import build_scene_image_prompt
from src.models.scene import ScenePlanItem
from src.providers.base import ProviderError
from src.providers.image_qwen import QwenImageProvider

_SIZE = "1328*1328"


class SceneImageGenerationError(Exception):
    """Raised for any reason build_scene_image() cannot produce
    output_path: an invalid/missing --reference-image, an already-existing
    or unreachable --output, a Qwen-Image provider failure, or a generated
    file that is not a valid PNG. Message text is always a short, fixed,
    sanitized string. A PaidApprovalRequiredError from require_paid_approval()
    is NOT one of these — it is never caught here, and propagates unchanged;
    see this module's docstring."""


def build_scene_image(
    scene: ScenePlanItem,
    reference_image_path: Path,
    output_path: Path,
    proposals_path: Path,
    *,
    include_character_anchor: bool,
    include_color_anchor: bool = True,
) -> Path:
    """Generate one scene's image at `output_path` from `reference_image_path`
    (read-only, never modified) using `scene.visual_brief`. Raises
    SceneImageGenerationError and leaves no new file behind on any
    rejection or failure. Raises PaidApprovalRequiredError, uncaught, if
    the qwen-image paid-proposal check fails. Never registers, saves a
    manifest, or touches SQLite."""
    if not reference_image_path.exists():
        raise SceneImageGenerationError("--reference-image does not exist")
    if not reference_image_path.is_file():
        raise SceneImageGenerationError("--reference-image is not a regular file")

    if not output_path.parent.exists():
        raise SceneImageGenerationError("--output parent directory does not exist")
    if output_path.exists():
        raise SceneImageGenerationError("--output already exists")

    prompt = build_scene_image_prompt(
        scene.visual_brief,
        include_character_anchor=include_character_anchor,
        include_color_anchor=include_color_anchor,
    )

    require_paid_approval("qwen-image", proposals_path, is_paid=True)

    generation_succeeded = False
    try:
        try:
            provider = QwenImageProvider()
            provider.generate_with_reference_and_download(
                prompt, reference_image_path, output_path, size=_SIZE
            )
        except ProviderError as exc:
            raise SceneImageGenerationError("Qwen-Image generation failed") from exc

        if not output_path.exists():
            raise SceneImageGenerationError("provider did not produce an output file")
        if not output_path.is_file():
            raise SceneImageGenerationError("provider output path is not a regular file")

        try:
            with Image.open(output_path) as img:
                img.load()  # forces a full decode — a truncated file raises here
                image_format = img.format
        except (UnidentifiedImageError, OSError) as exc:
            raise SceneImageGenerationError("generated file could not be decoded as an image") from exc

        if image_format != "PNG":
            raise SceneImageGenerationError("generated file is not a PNG")

        generation_succeeded = True
    finally:
        if not generation_succeeded and output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                pass

    return output_path
