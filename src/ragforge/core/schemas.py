''' 
This file defines 3 most important objects
These objects matter because they are the shared language of the whole project
Everything depends on them :
1. Ingestion creates them
2. Retrieval reads them
3. Evaluation scores them

Thus these are the 3 data structures that make the RAG system understandable.

'''
# Delays evaluation of type hints (cleaner typing & forward references)
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

# @dataclass Automatically generates init/representation methods
# Slots here reduces memory and speeds up attribute access
# Document : A fully loaded source document
@dataclass(slots=True)
class Document:
    doc_id: str
    text: str
    metadata: Dict[str, Any]


# Chunk : A smaller piece of a document
@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any]


# A Ranked result retruned by a retriever
@dataclass(slots=True)
class RetrievalResult:
    chunk: Chunk
    score: float
    source: str



    