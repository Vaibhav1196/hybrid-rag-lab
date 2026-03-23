from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RetrievalSample:
    """A single labeled retrieval query for evaluation."""

    query_id: str
    query: str
    relevant_doc_ids: list[str] = field(default_factory=list)
    relevant_chunk_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalQueryResult:
    """Per-query retrieval evaluation details."""

    query_id: str
    query: str
    matched: bool
    reciprocal_rank: float
    recall: float
    matched_chunk_ids: list[str]
    matched_doc_ids: list[str]
    returned_chunk_ids: list[str]
    returned_doc_ids: list[str]


@dataclass(slots=True)
class RetrievalMetrics:
    """Aggregate retrieval metrics over an evaluation set."""

    queries_evaluated: int
    top_k: int
    hit_rate: float
    mean_reciprocal_rank: float
    recall_at_k: float


@dataclass(slots=True)
class RetrievalEvaluationReport:
    """Full retrieval evaluation report."""

    metrics: RetrievalMetrics
    query_results: list[RetrievalQueryResult]


@dataclass(slots=True)
class AnswerEvaluationSample:
    """A single labeled answer-generation query for evaluation."""

    query_id: str
    query: str
    reference_answer: str
    relevant_doc_ids: list[str] = field(default_factory=list)
    relevant_chunk_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HeuristicAnswerEvaluation:
    """Heuristic answer-evaluation result."""

    query_id: str
    answer_present: bool
    cites_context: bool
    grounded_to_relevant_context: bool
    reference_term_overlap: float
    overall_score: float


@dataclass(slots=True)
class LLMJudgeEvaluation:
    """LLM-as-judge output for a generated answer."""

    query_id: str
    groundedness: float
    correctness: float
    completeness: float
    overall: float
    reason: str
