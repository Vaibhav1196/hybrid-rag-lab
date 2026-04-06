from __future__ import annotations

import json
import os
import re
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
        query = self._extract_question(user_prompt)
        candidates = self._extract_candidates(user_prompt)
        answer = self._choose_best_candidate(query=query, candidates=candidates)

        if not answer:
            answer = "I could not find grounded context to answer safely."

        return LLMResponse(model=self.model_name, content=answer)

    @staticmethod
    def _extract_question(user_prompt: str) -> str:
        """Extract the question text from the prompt."""
        match = re.search(r"Question:\s*(.+)", user_prompt)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Tokenize text into lowercase alphanumeric terms."""
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split snippet text into short sentence candidates."""
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [part.strip() for part in parts if part.strip()]

    def _extract_candidates(self, user_prompt: str) -> list[str]:
        """Extract snippet sentences from the context section of the prompt."""
        lines = [line.strip() for line in user_prompt.splitlines() if line.strip()]
        candidates: list[str] = []

        for index, line in enumerate(lines):
            if line.startswith("[") and index + 1 < len(lines):
                candidate_line = lines[index + 1]
                if not candidate_line.startswith("["):
                    candidates.extend(self._split_sentences(candidate_line))

        return candidates

    def _choose_best_candidate(self, query: str, candidates: list[str]) -> str:
        """Choose the most query-relevant candidate sentence."""
        if not candidates:
            return ""

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return candidates[0]

        scored_candidates: list[tuple[int, int, str]] = []
        for candidate in candidates:
            candidate_tokens = self._tokenize(candidate)
            overlap = len(query_tokens & candidate_tokens)
            # Prefer shorter, more direct sentences when overlap ties.
            scored_candidates.append((overlap, -len(candidate), candidate))

        scored_candidates.sort(reverse=True)
        best_overlap, _, best_candidate = scored_candidates[0]
        return best_candidate if best_overlap > 0 else candidates[0]


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


class HuggingFaceInferenceLLM(OpenAICompatibleLLM):
    """Hugging Face Inference Providers client using the OpenAI-compatible router."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        resolved_api_key = api_key or os.getenv("HF_TOKEN")
        if not resolved_api_key:
            raise ValueError("HuggingFaceInferenceLLM requires HF_TOKEN or an explicit api_key.")

        super().__init__(
            model_name=model_name,
            base_url=base_url or "https://router.huggingface.co/v1",
            api_key=resolved_api_key,
        )
