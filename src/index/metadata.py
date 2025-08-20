from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class ComplaintChunkMetadata:
	chunk_id: str
	complaint_id: str
	product: str
	issue: str
	date_received: str