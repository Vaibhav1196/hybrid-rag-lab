from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, Sequence

from ragforge.core.schemas import Document, RetrievalResult
from ragforge.core.telemetry import PipelineTrace
from ragforge.generation import ContextBuilder, ExtractiveFallbackLLM, HuggingFaceInferenceLLM, RAGPipeline
from ragforge.generation.schemas import GenerationResponse
from ragforge.ingestion.loader import load_documents
from ragforge.retrieval import BM25Pipeline, DensePipeline, HybridPipeline
from ragforge.retrieval.embeddings import SentenceTransformerEmbedder
from ragforge.retrieval.pipeline import RerankedHybridPipeline
from ragforge.retrieval.reranking import CrossEncoderScorer

DemoPipelineKey = Literal["bm25", "dense", "hybrid", "reranked"]
GenerationModeKey = Literal["fallback", "huggingface"]

DEMO_PIPELINE_LABELS: dict[DemoPipelineKey, str] = {
    "bm25": "BM25",
    "dense": "Dense",
    "hybrid": "Hybrid",
    "reranked": "Reranked Hybrid",
}

DEFAULT_DENSE_MODEL = "all-MiniLM-L6-v2"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_HF_GENERATION_MODEL = "CohereLabs/tiny-aya-global:cohere"
MAX_UPLOAD_FILES = 3
PREVIEW_CHARS = 220


@dataclass(slots=True)
class DemoRunResult:
    """Structured result returned by the Hugging Face demo service."""

    pipeline_key: DemoPipelineKey
    pipeline_label: str
    generation_mode: GenerationModeKey
    documents: list[Document]
    generation: GenerationResponse


@lru_cache(maxsize=2)
def _get_embedder(model_name: str) -> SentenceTransformerEmbedder:
    """Reuse the dense embedder across demo runs to reduce warmup time."""
    return SentenceTransformerEmbedder(model_name=model_name)


@lru_cache(maxsize=2)
def _get_reranker(model_name: str) -> CrossEncoderScorer:
    """Reuse the reranker scorer across demo runs to reduce warmup time."""
    return CrossEncoderScorer(model_name=model_name)


def normalize_upload_paths(uploaded_files: Sequence[str | Path] | None) -> list[Path]:
    """Validate uploaded files and return concrete paths."""
    if not uploaded_files:
        raise ValueError("Upload at least one TXT, PDF, or DOCX document.")

    paths = [Path(path) for path in uploaded_files]
    if len(paths) > MAX_UPLOAD_FILES:
        raise ValueError(f"Upload at most {MAX_UPLOAD_FILES} files for demo 1.")

    return paths


def _build_pipeline(
    documents: list[Document],
    pipeline_key: DemoPipelineKey,
    chunk_size: int,
    overlap: int,
    top_k: int,
) -> BM25Pipeline | DensePipeline | HybridPipeline | RerankedHybridPipeline:
    """Build the selected retrieval pipeline from already-loaded documents."""
    common = {
        "documents": documents,
        "chunk_size": chunk_size,
        "overlap": overlap,
    }

    if pipeline_key == "bm25":
        return BM25Pipeline.from_documents(**common)

    embedder = _get_embedder(DEFAULT_DENSE_MODEL)
    if pipeline_key == "dense":
        return DensePipeline.from_documents(**common, embedder=embedder, model_name=DEFAULT_DENSE_MODEL)
    if pipeline_key == "hybrid":
        return HybridPipeline.from_documents(**common, embedder=embedder, model_name=DEFAULT_DENSE_MODEL)

    scorer = _get_reranker(DEFAULT_RERANKER_MODEL)
    return RerankedHybridPipeline.from_documents(
        **common,
        embedder=embedder,
        scorer=scorer,
        model_name=DEFAULT_DENSE_MODEL,
        reranker_model_name=DEFAULT_RERANKER_MODEL,
        candidate_top_k=max(top_k + 2, 6),
    )


def _build_llm(generation_mode: GenerationModeKey):
    """Build the selected generation backend for the demo."""
    if generation_mode == "huggingface":
        return HuggingFaceInferenceLLM(model_name=DEFAULT_HF_GENERATION_MODEL)
    return ExtractiveFallbackLLM()


def run_demo_query(
    uploaded_files: Sequence[str | Path] | None,
    query: str,
    pipeline_key: DemoPipelineKey,
    generation_mode: GenerationModeKey = "fallback",
    top_k: int = 3,
    chunk_size: int = 300,
    overlap: int = 50,
) -> DemoRunResult:
    """Run upload -> retrieval -> generation for the Spaces demo."""
    query = query.strip()
    if not query:
        raise ValueError("Query must not be blank.")
    if top_k <= 0:
        raise ValueError("top_k must be > 0.")

    documents = load_documents(normalize_upload_paths(uploaded_files))
    if not documents:
        raise ValueError("No non-empty supported documents were loaded from the uploaded files.")

    retrieval_pipeline = _build_pipeline(
        documents=documents,
        pipeline_key=pipeline_key,
        chunk_size=chunk_size,
        overlap=overlap,
        top_k=top_k,
    )
    rag_pipeline = RAGPipeline(
        retrieval_pipeline=retrieval_pipeline,
        context_builder=ContextBuilder(max_chunks=min(top_k, 4), max_chars=1_800),
        llm=_build_llm(generation_mode),
    )
    generation = rag_pipeline.answer(query=query, retrieval_top_k=top_k)

    return DemoRunResult(
        pipeline_key=pipeline_key,
        pipeline_label=DEMO_PIPELINE_LABELS[pipeline_key],
        generation_mode=generation_mode,
        documents=documents,
        generation=generation,
    )


def _preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    """Create a compact preview for table display."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def format_documents_table(documents: list[Document]) -> list[list[object]]:
    """Format loaded documents for display in the demo table."""
    rows: list[list[object]] = []
    for document in documents:
        rows.append(
            [
                document.metadata.get("filename", document.doc_id),
                document.metadata.get("file_type", "unknown"),
                document.doc_id,
                len(document.text),
                document.metadata.get("page_count", ""),
            ]
        )
    return rows


def format_results_table(results: list[RetrievalResult]) -> list[list[object]]:
    """Format ranked retrieval results for display in the demo table."""
    rows: list[list[object]] = []
    for rank, result in enumerate(results, start=1):
        rows.append(
            [
                rank,
                result.chunk.metadata.get("filename", result.chunk.doc_id),
                result.chunk.doc_id,
                result.chunk.chunk_id,
                result.source,
                round(result.score, 4),
                _preview(result.chunk.text),
            ]
        )
    return rows


def format_trace_payload(trace: PipelineTrace) -> dict[str, object]:
    """Format pipeline telemetry for JSON display."""
    return {
        "total_duration_ms": round(trace.total_duration_ms, 2),
        "stages": [
            {"stage": item.stage, "duration_ms": round(item.duration_ms, 2)}
            for item in trace.stage_timings
        ],
        "metadata": trace.metadata,
    }


def build_run_summary(result: DemoRunResult) -> str:
    """Create a concise markdown summary for the demo UI."""
    trace = result.generation.trace
    if result.generation_mode == "huggingface":
        generation_label = f"huggingface-llm ({result.generation.llm_response.model})"
    else:
        generation_label = "fallback-grounded"
    return (
        f"### Run Summary\n"
        f"- Pipeline: `{result.pipeline_label}`\n"
        f"- Uploaded documents: `{len(result.documents)}`\n"
        f"- Retrieved chunks: `{len(result.generation.retrieval_results)}`\n"
        f"- Generation mode: `{generation_label}`\n"
        f"- Total latency: `{trace.total_duration_ms:.2f} ms`"
    )
