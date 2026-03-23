from __future__ import annotations

import json
from pathlib import Path

from ragforge.evaluation.schemas import RetrievalSample
from ragforge.evaluation.schemas import AnswerEvaluationSample


def load_retrieval_samples(path: str | Path) -> list[RetrievalSample]:
    """Load retrieval evaluation samples from a JSONL file."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset does not exist: {dataset_path}")

    samples: list[RetrievalSample] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            samples.append(
                RetrievalSample(
                    query_id=str(record["query_id"]),
                    query=str(record["query"]),
                    relevant_doc_ids=[str(item) for item in record.get("relevant_doc_ids", [])],
                    relevant_chunk_ids=[str(item) for item in record.get("relevant_chunk_ids", [])],
                    metadata=dict(record.get("metadata", {})),
                )
            )

    return samples


def load_answer_evaluation_samples(path: str | Path) -> list[AnswerEvaluationSample]:
    """Load answer-evaluation samples from a JSONL file."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset does not exist: {dataset_path}")

    samples: list[AnswerEvaluationSample] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            samples.append(
                AnswerEvaluationSample(
                    query_id=str(record["query_id"]),
                    query=str(record["query"]),
                    reference_answer=str(record["reference_answer"]),
                    relevant_doc_ids=[str(item) for item in record.get("relevant_doc_ids", [])],
                    relevant_chunk_ids=[str(item) for item in record.get("relevant_chunk_ids", [])],
                    metadata=dict(record.get("metadata", {})),
                )
            )

    return samples
