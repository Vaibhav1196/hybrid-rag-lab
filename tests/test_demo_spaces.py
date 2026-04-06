from __future__ import annotations

from pathlib import Path

import pytest

from ragforge.generation.schemas import LLMResponse
from ragforge.demo.spaces import (
    build_run_summary,
    format_documents_table,
    format_results_table,
    normalize_upload_paths,
    run_demo_query,
)


def test_normalize_upload_paths_requires_files() -> None:
    with pytest.raises(ValueError, match="Upload at least one"):
        normalize_upload_paths(None)


def test_normalize_upload_paths_limits_demo_upload_count(tmp_path: Path) -> None:
    files = []
    for index in range(4):
        path = tmp_path / f"doc_{index}.txt"
        path.write_text(f"document {index}", encoding="utf-8")
        files.append(path)

    with pytest.raises(ValueError, match="Upload at most"):
        normalize_upload_paths(files)


def test_run_demo_query_returns_ranked_results_and_answer(tmp_path: Path) -> None:
    python_doc = tmp_path / "python.txt"
    python_doc.write_text(
        "Python is used for automation, APIs, and AI applications.",
        encoding="utf-8",
    )
    travel_doc = tmp_path / "travel.txt"
    travel_doc.write_text(
        "Paris is known for museums, cafes, and historic landmarks.",
        encoding="utf-8",
    )
    sports_doc = tmp_path / "sports.txt"
    sports_doc.write_text(
        "Basketball is played on a court with two hoops and five players per team.",
        encoding="utf-8",
    )

    result = run_demo_query(
        uploaded_files=[python_doc, travel_doc, sports_doc],
        query="Python AI",
        pipeline_key="bm25",
        top_k=2,
    )

    assert result.pipeline_label == "BM25"
    assert len(result.documents) == 3
    assert len(result.generation.retrieval_results) >= 1
    assert "Python" in result.generation.answer

    document_rows = format_documents_table(result.documents)
    retrieval_rows = format_results_table(result.generation.retrieval_results)
    summary = build_run_summary(result)

    assert document_rows[0][0] == "python.txt"
    assert retrieval_rows[0][0] == 1
    assert "Pipeline" in summary


def test_run_demo_query_supports_huggingface_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = tmp_path / "python.txt"
    document.write_text(
        "Python is used for automation, APIs, and AI applications.",
        encoding="utf-8",
    )

    class FakeHFLLM:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def generate(
            self,
            system_prompt: str,
            user_prompt: str,
            temperature: float = 0.0,
            max_tokens: int | None = None,
        ) -> LLMResponse:
            del system_prompt, user_prompt, temperature, max_tokens
            return LLMResponse(model=self.model_name, content="HF generated answer")

    monkeypatch.setattr("ragforge.demo.spaces.HuggingFaceInferenceLLM", FakeHFLLM)

    result = run_demo_query(
        uploaded_files=[document],
        query="Python AI",
        pipeline_key="dense",
        generation_mode="huggingface",
        top_k=1,
    )

    assert result.generation_mode == "huggingface"
    assert result.generation.answer == "HF generated answer"
    assert "huggingface-llm" in build_run_summary(result)
