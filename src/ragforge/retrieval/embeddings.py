from __future__ import annotations

from typing import Protocol

import numpy as np


class TextEmbedder(Protocol):
    """Protocol for embedding text into dense vectors."""

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts into a 2D numpy array."""


class SentenceTransformerEmbedder:
    """Thin adapter around sentence-transformers for dense retrieval."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into normalized float32 vectors."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)
