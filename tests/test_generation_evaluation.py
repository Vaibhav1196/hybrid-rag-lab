from __future__ import annotations

import json

from ragforge.core.schemas import Chunk, RetrievalResult
from ragforge.core.telemetry import PipelineTrace
from ragforge.evaluation import evaluate_answer_heuristics, evaluate_answer_with_judge
from ragforge.evaluation.schemas import AnswerEvaluationSample
from ragforge.generation.schemas import ConstructedContext, ContextSnippet, GenerationResponse, LLMResponse


class FakeJudge:
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        del system_prompt, user_prompt, temperature, max_tokens
        return LLMResponse(
            model="fake-judge",
            content=json.dumps(
                {
                    "groundedness": 0.9,
                    "correctness": 0.8,
                    "completeness": 0.85,
                    "overall": 0.85,
                    "reason": "Grounded and mostly correct.",
                }
            ),
        )


def make_generation_response(answer: str) -> GenerationResponse:
    snippet = ContextSnippet(
        citation_id="[1]",
        doc_id="doc-1",
        chunk_id="chunk-1",
        text="Python is a programming language.",
        rank=1,
        retrieval_score=1.0,
        retrieval_source="hybrid_reranked",
    )
    return GenerationResponse(
        query="What is Python?",
        answer=answer,
        context=ConstructedContext(
            query="What is Python?",
            snippets=[snippet],
            prompt_context="[1] doc_id=doc-1 chunk_id=chunk-1 source=hybrid_reranked\nPython is a programming language.",
            total_chars=len(snippet.text),
        ),
        retrieval_results=[
            RetrievalResult(
                chunk=Chunk(chunk_id="chunk-1", doc_id="doc-1", text=snippet.text, metadata={}),
                score=1.0,
                source="hybrid_reranked",
            )
        ],
        llm_response=LLMResponse(model="fake-llm", content=answer),
        trace=PipelineTrace(),
    )


def test_heuristic_answer_evaluation_scores_grounded_answer() -> None:
    sample = AnswerEvaluationSample(
        query_id="q1",
        query="What is Python?",
        reference_answer="Python is a programming language.",
        relevant_doc_ids=["doc-1"],
    )
    response = make_generation_response("Python is a programming language. [1]")

    result = evaluate_answer_heuristics(sample, response)

    assert result.answer_present is True
    assert result.cites_context is True
    assert result.grounded_to_relevant_context is True
    assert result.reference_term_overlap > 0.5


def test_answer_judge_parses_json_scores() -> None:
    sample = AnswerEvaluationSample(
        query_id="q1",
        query="What is Python?",
        reference_answer="Python is a programming language.",
        relevant_doc_ids=["doc-1"],
    )
    response = make_generation_response("Python is a programming language. [1]")

    result = evaluate_answer_with_judge(FakeJudge(), sample, response)

    assert result.query_id == "q1"
    assert result.overall == 0.85
    assert "Grounded" in result.reason
