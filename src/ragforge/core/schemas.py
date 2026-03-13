# Delays evaluation of type hints (cleaner typing & forward references)
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

# @dataclass Automatically generates init/representation methods
# Slots here reduces memory and speeds up attribute access
@dataclass(slots=True)
class Document:
    doc_id: str
    text: str
    metadata: Dict[str, Any]


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any]


@dataclass(slots=True)
class RetrievalResult:
    chunk: Chunk
    score: float
    source: str

