# python -m app.ingestion.test_ingestion
from __future__ import annotations

from ragforge.ingestion.loader import load_documents
from ragforge.ingestion.chunker import chunk_documents


def main() -> None:
    documents = load_documents("data")
    chunks = chunk_documents(documents, chunk_size=120, overlap=20)

    print(f"Loaded {len(documents)} documents")
    print(f"Created {len(chunks)} chunks")
    print("-" * 80)

    for chunk in chunks:
        print(f"chunk_id={chunk.chunk_id}")
        print(f"doc_id={chunk.doc_id}")
        print(f"text={chunk.text}")
        print("-" * 80)


if __name__ == "__main__":
    main()
