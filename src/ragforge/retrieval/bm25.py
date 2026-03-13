from __future__ import annotations

from rank_bm25 import BM25Okapi

from ragforge.core.schemas import Chunk, RetrievalResult


def tokenize(text: str) -> list[str]:
    """Normalize text into simple lowercase whitespace tokens for BM25."""
    return text.lower().split()


class BM25Retriever:
    """Sparse retriever that ranks chunks with BM25."""

    def __init__(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("BM25Retriever requires at least one chunk.")

        self.chunks = chunks
        self.tokenized_corpus = [tokenize(chunk.text) for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Search the chunk index and return ranked BM25 retrieval results."""
        query = query.strip()
        if not query:
            return []
        if top_k <= 0:
            return []

        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(
            (index for index, score in enumerate(scores) if score > 0),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results: list[RetrievalResult] = []
        for index in ranked_indices:
            results.append(
                RetrievalResult(
                    chunk=self.chunks[index],
                    score=float(scores[index]),
                    source="bm25",
                )
            )

        return results
