"""Helpers for runnable demos built on top of ragforge."""

from ragforge.demo.spaces import (
    DEMO_PIPELINE_LABELS,
    DEFAULT_DENSE_MODEL,
    DEFAULT_HF_GENERATION_MODEL,
    DEFAULT_RERANKER_MODEL,
    DemoRunResult,
    build_run_summary,
    format_documents_table,
    format_results_table,
    format_trace_payload,
    run_demo_query,
)

__all__ = [
    "DEFAULT_DENSE_MODEL",
    "DEFAULT_HF_GENERATION_MODEL",
    "DEFAULT_RERANKER_MODEL",
    "DEMO_PIPELINE_LABELS",
    "DemoRunResult",
    "build_run_summary",
    "format_documents_table",
    "format_results_table",
    "format_trace_payload",
    "run_demo_query",
]
