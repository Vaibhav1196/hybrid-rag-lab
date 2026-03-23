from __future__ import annotations

from ragforge.core.schemas import RetrievalResult
from ragforge.generation.schemas import ConstructedContext, ContextSnippet


class ContextBuilder:
    """Build prompt-ready context from ranked retrieval results."""

    def __init__(self, max_chunks: int = 4, max_chars: int = 1_800) -> None:
        if max_chunks <= 0:
            raise ValueError("max_chunks must be > 0.")
        if max_chars <= 0:
            raise ValueError("max_chars must be > 0.")

        self.max_chunks = max_chunks
        self.max_chars = max_chars

    def build(self, query: str, results: list[RetrievalResult]) -> ConstructedContext:
        """Build a context block from retrieved results."""
        snippets: list[ContextSnippet] = []
        total_chars = 0

        for rank, result in enumerate(results, start=1):
            if len(snippets) >= self.max_chunks:
                break

            text = result.chunk.text.strip()
            if not text:
                continue

            remaining_chars = self.max_chars - total_chars
            if remaining_chars <= 0:
                break

            if len(text) > remaining_chars:
                if not snippets:
                    text = text[:remaining_chars].rstrip()
                else:
                    break

            snippet = ContextSnippet(
                citation_id=f"[{len(snippets) + 1}]",
                doc_id=result.chunk.doc_id,
                chunk_id=result.chunk.chunk_id,
                text=text,
                rank=rank,
                retrieval_score=result.score,
                retrieval_source=result.source,
            )
            snippets.append(snippet)
            total_chars += len(text)

        prompt_context = "\n\n".join(
            (
                f"{snippet.citation_id} doc_id={snippet.doc_id} "
                f"chunk_id={snippet.chunk_id} source={snippet.retrieval_source}\n"
                f"{snippet.text}"
            )
            for snippet in snippets
        )

        return ConstructedContext(
            query=query,
            snippets=snippets,
            prompt_context=prompt_context,
            total_chars=total_chars,
        )
