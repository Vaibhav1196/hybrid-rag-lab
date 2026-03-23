# Hybrid RAG Masterclass

This document teaches the full build-out of the `hybrid-rag-lab` project from scratch. It is written as a single-stop guide for understanding the architecture, the implementation sequence, and the reasoning behind each layer.

## 1. What We Are Building

A production-minded Hybrid RAG system has multiple stages:

1. ingestion
2. retrieval
3. fusion
4. reranking
5. context construction
6. answer generation
7. retrieval evaluation
8. answer evaluation
9. observability

The project implements those stages incrementally so that each layer stays understandable and testable.

## 2. Core Principles

The repository follows a few important engineering rules:

- use structured data objects instead of loose dictionaries
- keep responsibilities separate by module
- make every stage independently testable
- keep retrieval and generation decoupled
- add evaluation only after the corresponding runtime layer exists

That is why `Document`, `Chunk`, and `RetrievalResult` were introduced first. They are the contracts the rest of the system builds on.

## 3. Ingestion

The ingestion layer has two jobs:

1. load raw `.txt` files into `Document`
2. split `Document` objects into `Chunk`

This lives under:

- `src/ragforge/ingestion/loader.py`
- `src/ragforge/ingestion/chunker.py`

Why this matters:

- retrieval should not know about the filesystem
- generation should not know how chunking works
- chunking strategy can evolve later without rewriting the rest of the system

## 4. Sparse Retrieval with BM25

The first retriever added was BM25. It is a lexical retriever, which means it relies on token overlap between the query and the chunk text.

BM25 is valuable because:

- it is fast
- it is simple
- it is strong on exact-match queries
- it provides a clean baseline

It lives in:

- `src/ragforge/retrieval/bm25.py`

The first full pipeline built on top of it was `BM25Pipeline`, which loads documents, chunks them, indexes them, and exposes `search()`.

## 5. Dense Retrieval

The second retriever added was dense retrieval. This stage embeds chunks and the query into vectors, then uses cosine similarity to rank candidates.

Files:

- `src/ragforge/retrieval/embeddings.py`
- `src/ragforge/retrieval/dense.py`

Why it exists:

- BM25 is strong on exact wording
- dense retrieval is stronger on paraphrases and semantic similarity

This gave the project a second retrieval signal with the same output contract: `list[RetrievalResult]`.

## 6. Fusion

After sparse and dense retrieval were in place, the system became truly hybrid by adding Reciprocal Rank Fusion.

File:

- `src/ragforge/retrieval/fusion.py`

Why RRF was used:

- BM25 and dense scores are not directly comparable
- RRF combines ranks instead of raw scores
- it is robust and standard for early hybrid systems

The result was `HybridPipeline`.

## 7. Reranking

Fusion improves recall and ranking stability, but it still returns a shortlist. The next step was reranking.

File:

- `src/ragforge/retrieval/reranking.py`

Reranking uses a cross-encoder-style scoring layer over the fused shortlist. This is more expensive than retrieval, so it should only run on a small candidate set.

The result was `RerankedHybridPipeline`.

## 8. Retrieval Evaluation

Once retrieval existed, evaluation could be added in a meaningful way.

Files:

- `src/ragforge/evaluation/schemas.py`
- `src/ragforge/evaluation/datasets.py`
- `src/ragforge/evaluation/retrieval.py`
- `scripts/run_retrieval_eval.py`

This layer evaluates any retrieval pipeline that exposes:

```python
search(query: str, top_k: int) -> list[RetrievalResult]
```

Metrics implemented:

- Hit Rate@k
- Mean Reciprocal Rank
- Recall@k

Datasets created:

- `data/evals/retrieval_eval.jsonl`
- `data/evals/retrieval_eval_hard.jsonl`
- `data/evals/retrieval_eval_chunked.jsonl`

Important lesson:

retrieval evaluation should happen before answer evaluation. If retrieval is weak, answer quality will be unstable no matter how strong the LLM is.

## 9. Context Construction

After retrieval and reranking, the next missing layer was the bridge into generation.

File:

- `src/ragforge/generation/context.py`

This stage takes the ranked shortlist and turns it into:

- ordered evidence snippets
- citations
- a prompt-ready context block
- a bounded context budget

This stage is important because the LLM should consume a clean, explicit context object, not arbitrary retrieval results.

## 10. LLM Generation

Generation lives in:

- `src/ragforge/generation/llm.py`
- `src/ragforge/generation/pipeline.py`
- `src/ragforge/generation/schemas.py`

There are two generation modes:

1. `ExtractiveFallbackLLM`
2. `OpenAICompatibleLLM`

Why both exist:

- the fallback client makes the project runnable offline and testable
- the OpenAI-compatible client makes the architecture ready for a real model backend

The main runtime object is `RAGPipeline`, which does:

1. retrieval
2. context construction
3. prompt building
4. generation

## 11. Answer Evaluation

After generation exists, answer-level evaluation becomes meaningful.

Files:

- `src/ragforge/evaluation/generation.py`
- `data/evals/answer_eval.jsonl`
- `scripts/run_answer_eval.py`

Two answer-evaluation layers exist:

1. heuristic evaluation
2. LLM-as-judge scaffold

Heuristic evaluation checks:

- answer presence
- citation usage
- grounding to relevant context
- overlap with the reference answer

The judge scaffold builds a rubric-driven prompt and expects structured JSON back from a judge model. This makes it possible to add a real LLM judge without rewriting the surrounding evaluation layer.

## 12. Observability

Observability starts in:

- `src/ragforge/core/telemetry.py`

The current implementation records stage timings and pipeline metadata. This is intentionally lightweight, but it establishes the correct design boundary.

Tracked concepts:

- retrieval latency
- context construction latency
- generation latency
- model name
- number of retrieved results
- number of context snippets

This is the beginning of a real observability layer. Later it can expand into structured logs, traces, dashboards, and experiment tracking.

## 13. How the Full Pipeline Works

At this point, the runtime architecture is:

1. load documents
2. chunk them
3. run BM25 retrieval
4. run dense retrieval
5. fuse ranks with RRF
6. rerank the fused shortlist
7. construct prompt context
8. generate a grounded answer
9. evaluate retrieval quality
10. evaluate answer quality
11. record timings and metadata

## 14. Why the Architecture Is Production-Minded

This project stays production-minded because:

- each layer is independently replaceable
- the same retrieval contract is reused throughout the system
- evaluation is separated from runtime logic
- external-model integration is hidden behind protocols and adapters
- the offline fallback path keeps local development practical

This means the system can grow without collapsing into a script pile.

## 15. What to Focus on to Master Hybrid RAG

If you want to master the architecture, focus on these ideas:

### A. retrieval is a multi-stage ranking problem

It is not one model. It is a sequence:

- sparse retrieval
- dense retrieval
- fusion
- reranking

### B. generation quality depends on context quality

The LLM is only as good as the evidence it receives.

### C. evaluation must exist at multiple levels

- retrieval evaluation
- answer evaluation
- judge-based evaluation
- operational metrics

### D. observability is part of the product, not a later add-on

If you cannot measure latency, ranking quality, and answer quality, you do not really control the system.

## 16. Recommended Next Extensions

After the current implementation, the strongest next improvements are:

1. a real answer-generation benchmark with multiple reference answers
2. richer answer metrics such as citation precision and groundedness checks
3. structured run logging to disk
4. API endpoints for retrieval and generation
5. embedding and reranker caching
6. larger and more difficult retrieval corpora

## 17. Final Mental Model

Use this mental model for the whole stack:

- ingestion prepares search units
- retrieval finds evidence
- fusion stabilizes ranking
- reranking sharpens the shortlist
- context construction prepares the prompt
- generation produces the answer
- evaluation measures quality
- observability keeps the system accountable

That is the full Hybrid RAG architecture implemented step by step in this repository.
