# Hybrid RAG Lab

Hybrid RAG Lab is an experimental repository exploring how to design
**production-grade Retrieval Augmented Generation (RAG) systems**.

Modern AI applications are no longer just about calling a language
model. Reliable systems require careful engineering across : 
- Retrieval
- Orchestration 
- Evaluation
- Observability

This project focuses on understanding and implementing those components step by step.

The core package developed in this repository is **`ragforge`**, a
Python library for building evaluation-driven hybrid RAG pipelines.

The goal of this project is to move beyond simple demos and instead
explore **AI systems engineering principles** applied to real-world LLM
applications.

------------------------------------------------------------------------

# Motivation

Most tutorials demonstrate basic RAG pipelines:

-   Embed documents
-   Retrieve similar chunks
-   Pass them to an LLM

While useful for prototyping, these pipelines often fail in real systems
due to:

-   Weak retrieval recall
-   Poor ranking of relevant documents
-   Hallucinated responses
-   Lack of evaluation and monitoring
-   Difficulty reproducing results

Modern AI systems require a more robust architecture that combines :

- Multiple retrieval strategies
- Structured pipelines 
- Evaluation mechanisms

This repository explores those ideas through practical implementations.

------------------------------------------------------------------------

# Project Goals

This project aims to explore how to design AI systems that are:

### Reliable

Ground model outputs in external knowledge and reduce hallucinations.

### Measurable

Introduce evaluation pipelines that allow us to measure system quality.

### Extensible

Build modular components that can evolve into more advanced
architectures such as multi-agent systems.

### Production-oriented

Structure the codebase as a clean Python package rather than a
collection of scripts.

------------------------------------------------------------------------

# The `ragforge` Package

`ragforge` is the core Python package developed in this repository.

It implements a modular framework for building hybrid RAG systems with
evaluation capabilities.

The package is organized into the following modules:

```
ragforge
│
├── core        → shared data structures and abstractions
├── ingestion   → document loading and chunking
├── retrieval   → sparse, dense, and hybrid retrieval methods
├── generation  → LLM response generation
├── evaluation  → evaluation pipelines and LLM-as-judge scoring
└── api         → API interface for interacting with the system
```
Here we try to organize the repo as a production grade AI system.

------------------------------------------------------------------------

# Planned System Architecture

````
 User Query
    ↓
 Query Processing
    ↓
 Hybrid Retrieval
(BM25 + Vector Search)
    ↓
 Rank Fusion
    ↓
 Reranking
    ↓
 Context Construction
    ↓
 LLM Generation
    ↓
 Evaluation
(LLM-as-Judge)
    ↓
 Observability & Metrics


````

We will implement each component incrementally in the repo. I would love people who visit this repo to learn and implement at the same time.

------------------------------------------------------------------------

# Key Concepts Explored

### Hybrid Retrieval

Combining sparse retrieval (BM25) with dense embedding search to improve
recall.

### Reciprocal Rank Fusion

Merging rankings from different retrieval systems.

### Reranking

Using cross-encoders or LLMs to refine retrieval results.

### Evaluation-Driven Development

Measuring system quality through automated evaluation pipelines.

### LLM-as-Judge

Using language models to evaluate generated answers against reference
data.

### Observability

Tracking metrics such as latency, retrieval success, and model behavior.

------------------------------------------------------------------------

# Repository Structure

````
hybrid-rag-lab
│
├── data/               # sample documents
├── tests/              # unit tests
├── scripts/            # runnable entry points grouped by stage
│
├── src/
│   └── ragforge/
│       ├── core/
│       ├── ingestion/
│       ├── retrieval/
│       ├── generation/
│       ├── evaluation/
│       └── api/
│
├── pyproject.toml
└── README.md

````

The project uses a **src-based layout** to ensure the package behaves
correctly when installed.

The `scripts/` directory is grouped by workflow:

- `scripts/ingestion/`
- `scripts/retrieval/`
- `scripts/generation/`
- `scripts/evaluation/`
- `scripts/docs/`

------------------------------------------------------------------------

# Documentation Strategy

This repository is intended to be both a build log and a teaching
resource.

The documentation will evolve at two levels:

-   **Top-level README**: explains the overall system architecture,
    project goals, learning path, and how the pieces fit together.
-   **Package-level READMEs**: explain the design of each layer in
    `src/ragforge`, including responsibilities, interfaces, tradeoffs,
    and implementation steps.

This structure helps readers move from a **system view** to a
**component view** without needing to read the full codebase first.

For example:

-   `src/ragforge/ingestion/README.md` can explain document loading,
    chunking strategies, and metadata handling.
-   `src/ragforge/retrieval/README.md` can explain sparse retrieval,
    dense retrieval, hybrid fusion, and reranking.
-   `src/ragforge/evaluation/README.md` can explain how we measure
    answer quality and retrieval quality.

The goal is for each folder to document not just **what lives there**,
but also **why it exists**, **what design decisions it contains**, and
**how it will be built incrementally**.

------------------------------------------------------------------------

# Development Environment

This project uses **uv** for dependency management and environment
isolation, with a `Makefile` to simplify common development tasks.

You need to have `uv` installed on your system before using the commands
below.

You can also inspect the [Makefile](./Makefile)
to see the exact commands each `make` target runs.

Direct commands without `make`:

Set up the local virtual environment and install dependencies:

```bash
uv venv .venv --python 3.11
uv pip install -e ".[dev]"
```

Clean caches and the local virtual environment:

```bash
find . -type d -name "__pycache__" -exec rm -rf {} +
rm -rf .venv .pytest_cache .ruff_cache
```

Run the full test suite:

```bash
uv run pytest -v --maxfail=1 --disable-warnings -q
```

If you prefer the Makefile wrappers, the equivalent commands are:

```
make setup
```

Format the codebase:

```
make format
```

Lint the codebase:

```
make lint
```

Auto-fix lint issues where possible:

```
make fixlint
```

Run tests:

```
make test
```

Run tests with debug output:

## Execution Guide

For a stage-by-stage runbook covering retrieval, evaluation, the full RAG pipeline, answer evaluation, and rebuilding the teaching PDF, see:

- [Execution Guide](./docs/execution_guide.md)
- [Hybrid RAG Masterclass (PDF)](./docs/hybrid_rag_masterclass.pdf)

```
make test-debug
```

Clean Python cache folders and rebuild the virtual environment if
needed:

```
make clean
```

------------------------------------------------------------------------

# Project Status

This repository is being built incrementally to explore AI system design
patterns.

Planned milestones include:

-   document ingestion and chunking
-   sparse retrieval with BM25
-   vector retrieval with embeddings
-   hybrid retrieval with rank fusion
-   reranking pipelines
-   LLM generation
-   evaluation pipelines
-   API integration

------------------------------------------------------------------------

# Current Build Plan

This repo is being built incrementally so readers can follow the system
architecture while implementing it.

The current plan is:

1.  **Core abstractions and data models**
2.  **Ingestion and chunking**
3.  **Retrieval: BM25, dense search, then hybrid fusion**
4.  **Generation, evaluation, and API integration**

------------------------------------------------------------------------

# Learning Goals

This project serves as a hands-on exploration of:

-   modern RAG architectures
-   AI system design
-   evaluation strategies for LLM systems
-   Python package engineering
-   building maintainable AI infrastructure

------------------------------------------------------------------------

# Future Directions

Possible extensions of this project include:

-   multi-agent orchestration
-   tool-augmented LLM workflows
-   model routing and cost optimization
-   tracing and observability
-   benchmarking retrieval strategies

------------------------------------------------------------------------

# License

MIT License
