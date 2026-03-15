'''
This files defines a standard interface for text embedders and 1 implementation using sentence-transformers.
This is written in a way that is model agnostic, so you can swap embedding models easily
'''


# annotations are stored as strings internally and evaluated later. 
# This helps with:
# 1. Avoiding circular imports
# 2. Faster imports
# 3. Avoiding circular dependencies
from __future__ import annotations


# Protocol lets you define structural interfaces
# Any object with the required methods is considered valid.
# Unlike inheritance, the class does not need to explicitly subclass it
from typing import Protocol

import numpy as np


#----------------------------------------------------------------------------------

# This defines an interface for embedding models.
class TextEmbedder(Protocol):
    """Protocol for embedding text into dense vectors."""

    # Required method, Any valid TextEmbedder must implement this function encode
    # Example valid embedders:
    # - SentenceTransformers
    # - OpenAI embeddings
    # - Cohere embeddings
    # - custom BERT models
    # As long as they implement encode() with the same signature.
    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts into a 2D numpy array."""


#----------------------------------------------------------------------------------

# This is a concrete implementation of the TextEmbedder interface.
# This is a wrapper around the sentence-transformers library.
# Its purpose is to match the TextEmbedder protocol.
class SentenceTransformerEmbedder:
    """Thin adapter around sentence-transformers for dense retrieval."""

    # This "all-MiniLM-L6-v2" is a 384-dim embedding model optimized for semantic search.
    # It is fast , small and widely used in RAG
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:

        # Import is inside constructor instead of the top
        # Avoids heavy import unless needed
        # Prevents dependency errors if not installed
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into normalized float32 vectors."""

        # First handle empty text, if empty return empty array
        # This prevents crashes later.
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        # Generate the embeddings
        # Ensures output is a NumPy array instead of PyTorch tensors.
        # This keeps the retriever NumPy-based.
        # Each vector is normalized: length = 1.0
        # Because then dot product = cosine similarity.
        # Retriever does : scores = chunk_embeddings @ query_embedding
        # Because vectors are normalized: dot_product = cosine_similarity
        # This makes similarity calculations faster.
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # some embeddings use float64 but we will use float32
        # it uses half the memory, float32 is standard for embeddings and is faster in vector math
        return np.asarray(embeddings, dtype=np.float32)

#----------------------------------------------------------------------------------