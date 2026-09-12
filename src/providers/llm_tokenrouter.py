"""LLMProvider fallback: TokenRouter (OpenAI-compatible). Only
z-ai/glm-5.3-free is confirmed actually free on this account — the other
IDs advertised as free elsewhere billed real credit when tested."""
from __future__ import annotations

from openai import OpenAI

from src.providers.base import ProviderError
from src.utils.config import get_settings

DEFAULT_MODEL = "z-ai/glm-5.3-free"


class TokenRouterProvider:
    def __init__(self):
        settings = get_settings()
        self._client = OpenAI(api_key=settings.tokenrouter_api_key, base_url=settings.tokenrouter_base_url)

    def health_check(self) -> bool:
        try:
            self.generate("Reply with the single word: ok", max_tokens=5)
            return True
        except ProviderError:
            return False

    def generate(self, prompt: str, model: str = DEFAULT_MODEL, max_tokens: int = 1024) -> str:
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raise ProviderError(f"TokenRouter request failed: {exc}") from exc
        return response.choices[0].message.content
