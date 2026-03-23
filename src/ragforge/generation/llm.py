from __future__ import annotations

import json
import os
from typing import Protocol
from urllib import error, request

from ragforge.generation.schemas import LLMResponse


class ChatLLM(Protocol):
    """Protocol for chat-style LLM clients."""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a response for the given prompts."""


class ExtractiveFallbackLLM:
    """Offline-safe fallback that extracts a grounded answer from the context."""

    def __init__(self, model_name: str = "extractive-fallback") -> None:
        self.model_name = model_name

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        del system_prompt, temperature, max_tokens
        lines = [line.strip() for line in user_prompt.splitlines() if line.strip()]
        answer = ""
        for index, line in enumerate(lines):
            if line.startswith("[") and index + 1 < len(lines):
                candidate = lines[index + 1]
                if not candidate.startswith("["):
                    answer = candidate
                    break

        if not answer:
            answer = "I could not find grounded context to answer safely."

        return LLMResponse(model=self.model_name, content=answer)


class OpenAICompatibleLLM:
    """Minimal OpenAI-compatible chat client using the standard library."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAICompatibleLLM requires an API key.")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        payload: dict[str, object] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with request.urlopen(req) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            raise RuntimeError(f"LLM request failed with status {exc.code}.") from exc

        content = raw["choices"][0]["message"]["content"]
        usage = raw.get("usage", {})
        return LLMResponse(
            model=self.model_name,
            content=content,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            metadata={"raw_response": raw},
        )
