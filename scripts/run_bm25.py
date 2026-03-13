# uv run scripts/run_bm25.py
from ragforge.core.schemas import Chunk
from ragforge.retrieval.bm25 import BM25Retriever


def main() -> None:
    chunks = [
        Chunk(
            chunk_id="c1",
            doc_id="doc1",
            text="Python is a popular programming language for AI and backend systems.",
            metadata={},
        ),
        Chunk(
            chunk_id="c2",
            doc_id="doc1",
            text="BM25 is a sparse retrieval algorithm used in information retrieval.",
            metadata={},
        ),
        Chunk(
            chunk_id="c3",
            doc_id="doc2",
            text="Paris is the capital city of France.",
            metadata={},
        ),
    ]

    retriever = BM25Retriever(chunks)

    query = "retrieval algorithm"
    results = retriever.search(query, top_k=2)

    print(f"Query: {query}\n")
    for rank, result in enumerate(results, start=1):
        print(f"Rank {rank}")
        print(f"Chunk ID: {result.chunk.chunk_id}")
        print(f"Score: {result.score:.4f}")
        print(f"Source: {result.source}")
        print(f"Text: {result.chunk.text}")
        print("-" * 50)


if __name__ == "__main__":
    main()