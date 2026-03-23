from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ragforge.core.schemas import RetrievalResult
from ragforge.core.telemetry import PipelineTrace


@dataclass(slots=True)
class ContextSnippet:
    """A single prompt-ready evidence snippet."""

    citation_id: str
    doc_id: str
    chunk_id: str
    text: str
    rank: int
    retrieval_score: float
    retrieval_source: str


@dataclass(slots=True)
class ConstructedContext:
    """The final context block passed into generation."""

    query: str
    snippets: list[ContextSnippet]
    prompt_context: str
    total_chars: int


@dataclass(slots=True)
class GenerationRequest:
    """Structured request for a RAG answer."""

    query: str
    retrieval_top_k: int = 5


@dataclass(slots=True)
class LLMResponse:
    """Normalized response returned by an LLM client."""

    model: str
    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationResponse:
    """Full answer generation result for a RAG request."""

    query: str
    answer: str
    context: ConstructedContext
    retrieval_results: list[RetrievalResult]
    llm_response: LLMResponse
    trace: PipelineTrace
