# Delays evaluation of type hints (cleaner typing & forward references)
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

# @dataclass Automatically generates init/representation methods
# Slots here reduces memory and speeds up attribute access
@dataclass(slots=True)
class Document:
    doc_id: str
    text: str
    metadata: Dict[str, str]


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, str]

