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

### BM25 retrieval

```bash
uv run python scripts/run_bm25.py --query "hybrid retrieval" --top-k 3
```

### Dense retrieval

```bash
uv run python scripts/run_dense.py --query "hybrid retrieval" --top-k 3
```

Optional model override:

```bash
uv run python scripts/run_dense.py --query "hybrid retrieval" --top-k 3 --model-name all-MiniLM-L6-v2
```

### Hybrid retrieval

```bash
uv run python scripts/run_hybrid.py --query "hybrid retrieval" --top-k 3
```

With explicit fusion settings:

```bash
uv run python scripts/run_hybrid.py --query "hybrid retrieval" --top-k 3 --rrf-k 60
```

### Reranked hybrid retrieval

```bash
uv run python scripts/run_rerank.py --query "hybrid retrieval" --top-k 3 --candidate-top-k 3
```

With explicit reranker model:

```bash
uv run python scripts/run_rerank.py --query "hybrid retrieval" --top-k 3 --candidate-top-k 5 --reranker-model-name cross-encoder/ms-marco-MiniLM-L-6-v2
```

## 3. Retrieval Evaluation

Three retrieval evaluation datasets are included:

- `data/evals/retrieval_eval.jsonl`
- `data/evals/retrieval_eval_hard.jsonl`
- `data/evals/retrieval_eval_chunked.jsonl`

### BM25 evaluation

```bash
uv run python scripts/run_retrieval_eval.py --pipeline bm25 --data-dir data --eval-path data/evals/retrieval_eval.jsonl --top-k 3
```

### Dense evaluation

```bash
uv run python scripts/run_retrieval_eval.py --pipeline dense --data-dir data --eval-path data/evals/retrieval_eval.jsonl --top-k 3
```

### Hybrid evaluation

```bash
uv run python scripts/run_retrieval_eval.py --pipeline hybrid --data-dir data --eval-path data/evals/retrieval_eval_chunked.jsonl --top-k 3
```

### Reranked evaluation

```bash
uv run python scripts/run_retrieval_eval.py --pipeline reranked --data-dir data --eval-path data/evals/retrieval_eval_chunked.jsonl --top-k 3 --candidate-top-k 3
```

## 4. End-to-End RAG Generation

### Offline-safe fallback mode

This mode does not require an external LLM API. It uses the local extractive fallback client.

```bash
uv run python scripts/run_rag.py --query "What is hybrid retrieval?" --llm-mode fallback --retrieval-top-k 3 --candidate-top-k 3
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
uv run python scripts/run_rag.py --query "What is hybrid retrieval?" --llm-mode openai --llm-model gpt-4.1-mini --retrieval-top-k 3 --candidate-top-k 3
```

## 5. Answer Evaluation

The answer evaluation dataset is:

- `data/evals/answer_eval.jsonl`

### Heuristic answer evaluation

```bash
uv run python scripts/run_answer_eval.py --eval-path data/evals/answer_eval.jsonl --llm-mode fallback --retrieval-top-k 3 --candidate-top-k 3
```

### LLM-as-judge evaluation

This path requires an OpenAI-compatible model because the judge expects structured JSON output from an actual model backend.

```bash
export OPENAI_API_KEY="your_key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
uv run python scripts/run_answer_eval.py --eval-path data/evals/answer_eval.jsonl --llm-mode openai --llm-model gpt-4.1-mini --judge --retrieval-top-k 3 --candidate-top-k 3
```

## 6. Teaching Artifacts

Markdown guide:

- `docs/hybrid_rag_masterclass.md`

PDF guide:

- `docs/hybrid_rag_masterclass.pdf`

If you update the markdown guide and want to rebuild the PDF:

```bash
uv run python scripts/build_learning_pdf.py --input docs/hybrid_rag_masterclass.md --output docs/hybrid_rag_masterclass.pdf
```

## 7. Recommended Order

If you are learning the system from scratch, run the stages in this order:

1. `uv run pytest`
2. `run_bm25.py`
3. `run_dense.py`
4. `run_hybrid.py`
5. `run_rerank.py`
6. `run_retrieval_eval.py`
7. `run_rag.py`
8. `run_answer_eval.py`

That order mirrors how the architecture was built.
