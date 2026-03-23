from __future__ import annotations

import json
import re
from typing import Protocol

from ragforge.evaluation.schemas import AnswerEvaluationSample, HeuristicAnswerEvaluation, LLMJudgeEvaluation
from ragforge.generation.schemas import GenerationResponse, LLMResponse


class JudgeLLM(Protocol):
    """Protocol for LLM-as-judge backends."""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a judgment response."""


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def evaluate_answer_heuristics(
    sample: AnswerEvaluationSample,
    response: GenerationResponse,
) -> HeuristicAnswerEvaluation:
    """Evaluate a generated answer with lightweight heuristics."""
    answer = response.answer.strip()
    answer_present = bool(answer)
    cites_context = any(snippet.citation_id in answer for snippet in response.context.snippets)
    relevant_doc_ids = set(sample.relevant_doc_ids)
    relevant_chunk_ids = set(sample.relevant_chunk_ids)
    grounded_to_relevant_context = any(
        snippet.doc_id in relevant_doc_ids or snippet.chunk_id in relevant_chunk_ids
        for snippet in response.context.snippets
    )

    reference_tokens = _tokenize(sample.reference_answer)
    answer_tokens = _tokenize(answer)
    reference_term_overlap = (
        0.0 if not reference_tokens else len(reference_tokens & answer_tokens) / len(reference_tokens)
    )

    overall_score = (
        float(answer_present)
        + float(cites_context)
        + float(grounded_to_relevant_context)
        + reference_term_overlap
    ) / 4.0

    return HeuristicAnswerEvaluation(
        query_id=sample.query_id,
        answer_present=answer_present,
        cites_context=cites_context,
        grounded_to_relevant_context=grounded_to_relevant_context,
        reference_term_overlap=reference_term_overlap,
        overall_score=overall_score,
    )


def build_llm_judge_prompts(
    sample: AnswerEvaluationSample,
    response: GenerationResponse,
) -> tuple[str, str]:
    """Build prompts for LLM-as-judge evaluation."""
    system_prompt = (
        "You are an evaluation judge for RAG systems. Score the answer as JSON with keys "
        "groundedness, correctness, completeness, overall, and reason. Scores must be between 0 and 1."
    )
    user_prompt = (
        f"Query: {sample.query}\n\n"
        f"Reference Answer: {sample.reference_answer}\n\n"
        f"Retrieved Context:\n{response.context.prompt_context}\n\n"
        f"Generated Answer:\n{response.answer}\n"
    )
    return system_prompt, user_prompt


def evaluate_answer_with_judge(
    judge_llm: JudgeLLM,
    sample: AnswerEvaluationSample,
    response: GenerationResponse,
) -> LLMJudgeEvaluation:
    """Evaluate a generated answer with an LLM judge."""
    system_prompt, user_prompt = build_llm_judge_prompts(sample, response)
    llm_response = judge_llm.generate(system_prompt=system_prompt, user_prompt=user_prompt)
    parsed = json.loads(llm_response.content)

    return LLMJudgeEvaluation(
        query_id=sample.query_id,
        groundedness=float(parsed["groundedness"]),
        correctness=float(parsed["correctness"]),
        completeness=float(parsed["completeness"]),
        overall=float(parsed["overall"]),
        reason=str(parsed["reason"]),
    )
