"""
BM25 is a smart keyword search algorithm. IT is ogood when
- The query words appear in the text
- Exact phrase or exact vocabulary matters

BM25 is the baseline retriever. Its simple, fast, strong and highly interpretable

In our code below we do the following 
- tokenize the chunks using lowercase + whitespaces
- build bm25 index over all chunks
- score a query against all chunks
- return ranked RetrievalResult objects

[Chunk.text -> tokenize -> BM25 index -> query scoring -> ranked RetrievalResult]

Strengths 
- Exact match
- Technical terms*
- Keyword heavy queries
- No GPU needed and Its fast

Weaknesses
- Weak on paraphrases
- Weak when wording changes

Thus BM25 is better than naive TF-IDF
Start with this before entering embeddings space as this gives a clear baseline for comparison

"""


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

        # Sort the indices based on scores in descending order
        ranked_indices = sorted(
            (index for index, score in enumerate(scores) if score > 0),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        # Initialize the results list
        results: list[RetrievalResult] = []
        # Add the chunks to the results list
        for index in ranked_indices:
            results.append(
                RetrievalResult(
                    chunk=self.chunks[index],
                    score=float(scores[index]),
                    source="bm25",
                )
            )

        return results
