from ragforge.ingestion.chunker import chunk_text


def test_chunk_text_returns_chunks() -> None:
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = chunk_text(text, chunk_size=10, overlap=2)

    assert len(chunks) > 0
    assert chunks[0] == "abcdefghij"


def test_chunk_text_rejects_invalid_overlap() -> None:
    try:
        chunk_text("hello world", chunk_size=10, overlap=10)
        assert False, "Expected ValueError"
    except ValueError:
        assert True