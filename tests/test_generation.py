from __future__ import annotations

from ragforge.core.schemas import Chunk, RetrievalResult
from ragforge.generation import ContextBuilder, ExtractiveFallbackLLM, RAGPipeline
from ragforge.generation.schemas import LLMResponse


class FakeRetrievalPipeline:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        return self.results[:top_k]


class FakeLLM:
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        del system_prompt, user_prompt, temperature, max_tokens
        return LLMResponse(model="fake-llm", content="Grounded answer [1]")


def make_result(text: str) -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(chunk_id="chunk-1", doc_id="doc-1", text=text, metadata={}),
        score=1.0,
        source="test",
    )


def test_rag_pipeline_returns_structured_answer() -> None:
    pipeline = RAGPipeline(
        retrieval_pipeline=FakeRetrievalPipeline([make_result("python backend systems")]),
        context_builder=ContextBuilder(),
        llm=FakeLLM(),
    )

    response = pipeline.answer("What is Python?", retrieval_top_k=1)

    assert response.answer == "Grounded answer [1]"
    assert response.context.snippets[0].doc_id == "doc-1"
    assert response.llm_response.model == "fake-llm"
    assert response.trace.total_duration_ms >= 0


def test_rag_pipeline_uses_safe_fallback_when_no_results() -> None:
    pipeline = RAGPipeline.with_fallback_llm(
        retrieval_pipeline=FakeRetrievalPipeline([]),
        context_builder=ContextBuilder(),
    )

    response = pipeline.answer("What is Python?", retrieval_top_k=1)

    assert "could not find grounded context" in response.answer.lower()
    assert response.llm_response.model == "no-generation"


def test_extractive_fallback_llm_extracts_context_text() -> None:
    llm = ExtractiveFallbackLLM()
    response = llm.generate(
        system_prompt="system",
        user_prompt="Question: q\n\nContext:\n[1] doc_id=doc-1 chunk_id=c1 source=test\nUseful answer sentence.",
    )

    assert response.content == "Useful answer sentence."
