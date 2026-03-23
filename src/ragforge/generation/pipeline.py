from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from ragforge.core.schemas import RetrievalResult
from ragforge.core.telemetry import PipelineTrace
from ragforge.generation.context import ContextBuilder
from ragforge.generation.llm import ChatLLM, ExtractiveFallbackLLM
from ragforge.generation.schemas import ConstructedContext, GenerationResponse, LLMResponse


class RetrievalPipeline(Protocol):
    """Protocol for retrieval pipelines used by generation."""

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Run retrieval for a query."""


def build_answer_prompts(query: str, context: ConstructedContext) -> tuple[str, str]:
    """Build the prompts used for grounded answer generation."""
    system_prompt = (
        "You are a grounded RAG assistant. Answer only from the provided context. "
        "If the context is insufficient, say so clearly. Cite supporting snippets like [1]."
    )
    user_prompt = (
        f"Question: {query}\n\n"
        f"Context:\n{context.prompt_context or 'No relevant context was retrieved.'}\n\n"
        "Write a concise answer grounded in the context."
    )
    return system_prompt, user_prompt


@dataclass(slots=True)
class RAGPipeline:
    """End-to-end retrieval-augmented generation pipeline."""

    retrieval_pipeline: RetrievalPipeline
    context_builder: ContextBuilder
    llm: ChatLLM

    @classmethod
    def with_fallback_llm(
        cls,
        retrieval_pipeline: RetrievalPipeline,
        context_builder: ContextBuilder | None = None,
    ) -> RAGPipeline:
        """Create a pipeline with the offline-safe fallback LLM."""
        return cls(
            retrieval_pipeline=retrieval_pipeline,
            context_builder=context_builder or ContextBuilder(),
            llm=ExtractiveFallbackLLM(),
        )

    def answer(self, query: str, retrieval_top_k: int = 5) -> GenerationResponse:
        """Run retrieval, build context, and generate a grounded answer."""
        query = query.strip()
        if not query:
            raise ValueError("query must not be blank.")
        if retrieval_top_k <= 0:
            raise ValueError("retrieval_top_k must be > 0.")

        trace = PipelineTrace()

        retrieval_start = perf_counter()
        retrieval_results = self.retrieval_pipeline.search(query=query, top_k=retrieval_top_k)
        trace.add_stage("retrieval", (perf_counter() - retrieval_start) * 1000)

        context_start = perf_counter()
        context = self.context_builder.build(query=query, results=retrieval_results)
        trace.add_stage("context_construction", (perf_counter() - context_start) * 1000)

        if not retrieval_results or not context.snippets:
            fallback = "I could not find grounded context to answer safely."
            trace.metadata.update({"retrieved_results": 0, "context_snippets": 0})
            return GenerationResponse(
                query=query,
                answer=fallback,
                context=context,
                retrieval_results=retrieval_results,
                llm_response=LLMResponse(model="no-generation", content=fallback),
                trace=trace,
            )

        prompt_start = perf_counter()
        system_prompt, user_prompt = build_answer_prompts(query=query, context=context)
        llm_response = self.llm.generate(system_prompt=system_prompt, user_prompt=user_prompt)
        trace.add_stage("llm_generation", (perf_counter() - prompt_start) * 1000)
        trace.metadata.update(
            {
                "retrieved_results": len(retrieval_results),
                "context_snippets": len(context.snippets),
                "llm_model": llm_response.model,
            }
        )

        return GenerationResponse(
            query=query,
            answer=llm_response.content.strip(),
            context=context,
            retrieval_results=retrieval_results,
            llm_response=llm_response,
            trace=trace,
        )
