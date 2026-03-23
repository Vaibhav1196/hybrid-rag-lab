from ragforge.core.schemas import Chunk, RetrievalResult
from ragforge.generation.context import ContextBuilder


def make_result(chunk_id: str, doc_id: str, text: str, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=text,
            metadata={},
        ),
        score=score,
        source="test",
    )


def test_context_builder_limits_chunks_and_preserves_citations() -> None:
    builder = ContextBuilder(max_chunks=2, max_chars=500)
    results = [
        make_result("c1", "doc-1", "first chunk"),
        make_result("c2", "doc-2", "second chunk"),
        make_result("c3", "doc-3", "third chunk"),
    ]

    context = builder.build("query", results)

    assert len(context.snippets) == 2
    assert context.snippets[0].citation_id == "[1]"
    assert context.snippets[1].citation_id == "[2]"
    assert "doc_id=doc-1" in context.prompt_context


def test_context_builder_respects_character_budget() -> None:
    builder = ContextBuilder(max_chunks=2, max_chars=10)
    results = [make_result("c1", "doc-1", "abcdefghijklmno")]

    context = builder.build("query", results)

    assert len(context.snippets) == 1
    assert context.total_chars == 10
    assert context.snippets[0].text == "abcdefghij"
