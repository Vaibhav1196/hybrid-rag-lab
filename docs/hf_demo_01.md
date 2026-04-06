# Hugging Face Demo 01

This branch contains the first Hugging Face Spaces demo for the current Hybrid RAG Lab architecture.

## What Demo 01 Includes

- upload support for `.txt`, `.pdf`, and `.docx`
- retrieval technique switcher:
  - BM25
  - Dense
  - Hybrid
  - Reranked Hybrid
- generation mode switcher:
  - local fallback extractive mode
  - Hugging Face LLM mode via `HF_TOKEN`
- ranked retrieval results with metadata and text previews
- grounded generated answer
- pipeline timing and metadata trace

The demo is intentionally lightweight and works best with one short document.

## Main Files

- `app.py`
- `requirements.txt`
- `src/ragforge/demo/spaces.py`

## Run Locally

Install the demo dependency and launch the app:

```bash
uv run --with gradio python app.py
```

Then open the local Gradio URL in your browser.

## Deploy to Hugging Face Spaces

The minimum files needed for the Space are:

- `app.py`
- `requirements.txt`
- `src/`
- `pyproject.toml`

Hugging Face Spaces will install dependencies from `requirements.txt`, which installs the local package and `gradio`.

If you want real answer generation instead of the local fallback mode, add this secret in the Space:

- `HF_TOKEN`

## Demo Constraints

- demo 1 supports up to 3 uploaded files
- best experience is with short documents
- fallback mode works without external access
- Hugging Face mode uses `CohereLabs/tiny-aya-global:cohere` by default
- dense and reranked pipelines may take longer on CPU than BM25
