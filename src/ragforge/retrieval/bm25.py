
from __future__ import annotations
from rank_bm25 import BM25Okapi
from ragforge.core.schemas import Chunk, RetrievalResult
from typing import List


def tokanize(text: str) -> List[str]:
    """
    A very simple tookanizer for BM25.
    Lower case text and splits on whitespace.
    Later we can improve this with better normalization.
    """

    return text.lower().split()




class BM25Retriever:
    """
    Sparse retriever using BM25 over chunk text.
    """
    def __init__(self, chunks: List[Chunk]) -> None:
        if not chunks:
            raise ValueError("BM25Retriever requires at least one chunk.")

        self.chunks = chunks
        self.tokanized_corpus = [tokanize(chunk.text) for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokanized_corpus)

    
    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Search the indexd chunnks with a Query string.
        Returns top-k chunks with BM25 scores.
        """
        query = query.strip()
        if not query:
            return []

        if top_k < 0:
            return []
        
        tokenized_query = tokanize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Obserev this function and practice writing such functions
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results: List[RetrievalResult] = []
        for index in ranked_indices:
            results.append(
                RetrievalResult(
                    chunk=self.chunks[index],
                    score=scores[index],
                    source="bm25",
                )
            )
        
        return results

        


