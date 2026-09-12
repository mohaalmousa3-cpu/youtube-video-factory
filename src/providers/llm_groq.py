"""LLMProvider: Groq — free tier, no card, used for script/idea text generation."""
from __future__ import annotations

from groq import Groq

from src.providers.base import ProviderError
from src.utils.config import get_settings

# ponytail: Groq's free-tier lineup changes over time (this one replaced
# llama-3.3-70b-versatile, which 404'd) — re-check `client.models.list()`
# if this starts failing.
DEFAULT_MODEL = "openai/gpt-oss-120b"


class GroqProvider:
    def __init__(self, api_key: str = ""):
        settings = get_settings()
        self._client = Groq(api_key=api_key or settings.groq_api_key)

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
            raise ProviderError(f"Groq request failed: {exc}") from exc
        return response.choices[0].message.content
