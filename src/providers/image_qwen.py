"""ImageProvider: Qwen-Image via DashScope (Alibaba) — used for per-scene
backgrounds and one-time character-asset generation.

Qwen-Image only accepts a fixed set of sizes (not arbitrary WxH) — pass one
of ALLOWED_SIZES or the call fails with InvalidParameter.

Text-only "character anchor" prompting (repeating the same description every
call) was NOT enough for consistency — confirmed by comparing the character
across scenes from a real generated video: head shape/proportions and
shading style (flat vs glossy) drifted noticeably scene to scene. Passing
the approved character image itself as a reference (qwen-image-2.0-pro,
image+text input) fixed it — verified head shape/proportions/flat-shading
matched the reference far more closely in a direct test. Use
generate_with_reference() for any scene that needs the character in it;
plain generate() is still fine for character-free shots (pure backgrounds,
inserts)."""
from __future__ import annotations

import base64
import time
from pathlib import Path

import dashscope
import requests
from dashscope import MultiModalConversation

from src.providers.base import ProviderError
from src.utils.config import get_settings

ALLOWED_SIZES = {"1664*928", "1472*1104", "1328*1328", "1104*1472", "928*1664"}
DEFAULT_MODEL = "qwen-image"
EDIT_MODEL = "qwen-image-2.0-pro"  # the model that accepts an image input alongside text
MAX_RETRIES = 4


def _call_with_retry(**kwargs):
    """qwen-image-2.0-pro's rate limit is tight enough to hit mid-batch
    (confirmed: failed on scene 4 of 11 back-to-back calls) — back off and
    retry on 429 instead of losing the whole batch."""
    delay = 5
    for attempt in range(MAX_RETRIES):
        response = MultiModalConversation.call(**kwargs)
        if response.status_code != 429:
            return response
        if attempt == MAX_RETRIES - 1:
            return response
        time.sleep(delay)
        delay *= 2
    return response  # unreachable, satisfies type checkers


class QwenImageProvider:
    def __init__(self):
        settings = get_settings()
        if not settings.qwen_api_key:
            raise ProviderError("QWEN_API_KEY not set")
        self._api_key = settings.qwen_api_key
        dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"

    def health_check(self) -> bool:
        try:
            self.generate("a small red circle on white background", size="1328*1328")
            return True
        except ProviderError:
            return False

    def generate(self, prompt: str, size: str = "1328*1328") -> str:
        """Returns a temporary image URL (expires in ~24h) — download it
        promptly with download_to() rather than storing the URL long-term."""
        if size not in ALLOWED_SIZES:
            raise ProviderError(f"size must be one of {ALLOWED_SIZES}, got {size!r}")
        response = _call_with_retry(
            api_key=self._api_key,
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            result_format="message",
            stream=False,
            watermark=False,
            size=size,
        )
        if response.status_code != 200:
            raise ProviderError(f"Qwen-Image failed ({response.status_code}): {response.message}")
        return response.output.choices[0].message.content[0]["image"]

    def generate_with_reference(self, prompt: str, reference_image_path: Path, size: str = "1328*1328") -> str:
        """Conditions the generation on a local reference image (our approved
        character asset) instead of text description alone — see module
        docstring for why plain generate() drifts on character shape."""
        if size not in ALLOWED_SIZES:
            raise ProviderError(f"size must be one of {ALLOWED_SIZES}, got {size!r}")
        image_bytes = reference_image_path.read_bytes()
        data_uri = f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"
        response = _call_with_retry(
            api_key=self._api_key,
            model=EDIT_MODEL,
            messages=[{"role": "user", "content": [{"image": data_uri}, {"text": prompt}]}],
            result_format="message",
            stream=False,
            watermark=False,
            size=size,
        )
        if response.status_code != 200:
            raise ProviderError(f"Qwen-Image-Edit failed ({response.status_code}): {response.message}")
        return response.output.choices[0].message.content[0]["image"]

    def _download(self, url: str, out_path: Path) -> Path:
        resp = requests.get(url, timeout=60)
        if resp.status_code != 200:
            raise ProviderError(f"downloading generated image failed: HTTP {resp.status_code}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(resp.content)
        return out_path

    def generate_and_download(self, prompt: str, out_path: Path, size: str = "1328*1328") -> Path:
        return self._download(self.generate(prompt, size=size), out_path)

    def generate_with_reference_and_download(
        self, prompt: str, reference_image_path: Path, out_path: Path, size: str = "1328*1328"
    ) -> Path:
        return self._download(self.generate_with_reference(prompt, reference_image_path, size=size), out_path)
