# Execution Guide

This guide explains how to run each implemented stage of the project from the command line.

## 1. Environment Setup

Create the virtual environment and install dependencies:

```bash
uv venv .venv --python 3.11
uv pip install -e ".[dev]"
```

Run the full test suite:

```bash
uv run pytest
```

Clean local caches and the virtual environment:

```bash
find . -type d -name "__pycache__" -exec rm -rf {} +
rm -rf .venv .pytest_cache .ruff_cache
```

## 2. Ingestion and Retrieval Stages

The scripts are grouped by responsibility:

- `scripts/ingestion/` for document-loading entry points
- `scripts/retrieval/` for retrieval-only experiments
- `scripts/generation/` for end-to-end RAG flows
- `scripts/evaluation/` for evaluation runners
- `scripts/docs/` for documentation build utilities

### Local ingestion-powered retrieval

Use this script when you want to test the universal ingestion layer locally with `.txt`, `.pdf`, and `.docx` files.

```bash
uv run python scripts/ingestion/run_local_ingestion_retrieval.py --input data --query "hybrid retrieval" --pipeline hybrid --top-k 3
```

You can also point it at explicit files:

```bash
uv run python scripts/ingestion/run_local_ingestion_retrieval.py --input path/to/file.pdf path/to/file.docx path/to/file.txt --query "your question" --pipeline reranked --top-k 5
```

### BM25 retrieval

```bash
uv run python scripts/retrieval/run_bm25_retrieval.py --query "hybrid retrieval" --top-k 3
```

### Dense retrieval

```bash
uv run python scripts/retrieval/run_dense_retrieval.py --query "hybrid retrieval" --top-k 3
```

Optional model override:

```bash
uv run python scripts/retrieval/run_dense_retrieval.py --query "hybrid retrieval" --top-k 3 --model-name all-MiniLM-L6-v2
```

### Hybrid retrieval

```bash
uv run python scripts/retrieval/run_hybrid_retrieval.py --query "hybrid retrieval" --top-k 3
```

With explicit fusion settings:

```bash
uv run python scripts/retrieval/run_hybrid_retrieval.py --query "hybrid retrieval" --top-k 3 --rrf-k 60
```

### Reranked hybrid retrieval

```bash
uv run python scripts/retrieval/run_reranked_retrieval.py --query "hybrid retrieval" --top-k 3 --candidate-top-k 3
```

With explicit reranker model:

```bash
uv run python scripts/retrieval/run_reranked_retrieval.py --query "hybrid retrieval" --top-k 3 --candidate-top-k 5 --reranker-model-name cross-encoder/ms-marco-MiniLM-L-6-v2
```

## 3. Retrieval Evaluation

Three retrieval evaluation datasets are included:

- `data/evals/retrieval_eval.jsonl`
- `data/evals/retrieval_eval_hard.jsonl`
- `data/evals/retrieval_eval_chunked.jsonl`

### BM25 evaluation

```bash
uv run python scripts/evaluation/run_retrieval_evaluation.py --pipeline bm25 --data-dir data --eval-path data/evals/retrieval_eval.jsonl --top-k 3
```

### Dense evaluation

```bash
uv run python scripts/evaluation/run_retrieval_evaluation.py --pipeline dense --data-dir data --eval-path data/evals/retrieval_eval.jsonl --top-k 3
```

### Hybrid evaluation

```bash
uv run python scripts/evaluation/run_retrieval_evaluation.py --pipeline hybrid --data-dir data --eval-path data/evals/retrieval_eval_chunked.jsonl --top-k 3
```

### Reranked evaluation

```bash
uv run python scripts/evaluation/run_retrieval_evaluation.py --pipeline reranked --data-dir data --eval-path data/evals/retrieval_eval_chunked.jsonl --top-k 3 --candidate-top-k 3
```

## 4. End-to-End RAG Generation

### Offline-safe fallback mode

This mode does not require an external LLM API. It uses the local extractive fallback client.

```bash
uv run python scripts/generation/run_rag_pipeline.py --query "What is hybrid retrieval?" --llm-mode fallback --retrieval-top-k 3 --candidate-top-k 3
```

### OpenAI-compatible mode

This mode requires a compatible API endpoint and key.

Set environment variables:

```bash
export OPENAI_API_KEY="your_key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

Then run:

```bash
uv run python scripts/generation/run_rag_pipeline.py --query "What is hybrid retrieval?" --llm-mode openai --llm-model gpt-4.1-mini --retrieval-top-k 3 --candidate-top-k 3
```

## 5. Answer Evaluation

The answer evaluation dataset is:

- `data/evals/answer_eval.jsonl`

### Heuristic answer evaluation

```bash
uv run python scripts/evaluation/run_answer_evaluation.py --eval-path data/evals/answer_eval.jsonl --llm-mode fallback --retrieval-top-k 3 --candidate-top-k 3
```

### LLM-as-judge evaluation

This path requires an OpenAI-compatible model because the judge expects structured JSON output from an actual model backend.

```bash
export OPENAI_API_KEY="your_key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
uv run python scripts/evaluation/run_answer_evaluation.py --eval-path data/evals/answer_eval.jsonl --llm-mode openai --llm-model gpt-4.1-mini --judge --retrieval-top-k 3 --candidate-top-k 3
```

## 6. Teaching Artifacts

PDF guide:

- `docs/hybrid_rag_masterclass.pdf`

## 7. Recommended Order

If you are learning the system from scratch, run the stages in this order:

1. `uv run pytest`
2. `scripts/ingestion/run_local_ingestion_retrieval.py`
3. `scripts/retrieval/run_bm25_retrieval.py`
4. `scripts/retrieval/run_dense_retrieval.py`
5. `scripts/retrieval/run_hybrid_retrieval.py`
6. `scripts/retrieval/run_reranked_retrieval.py`
7. `scripts/evaluation/run_retrieval_evaluation.py`
8. `scripts/generation/run_rag_pipeline.py`
9. `scripts/evaluation/run_answer_evaluation.py`

That order mirrors how the architecture was built.
