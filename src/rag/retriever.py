from __future__ import annotations
from typing import List, Optional, Dict, Any
import numpy as np
from ..features.embeddings import EmbeddingModel
from ..index.store_faiss import VectorStore

class Retriever:
	def __init__(self, vector_store: VectorStore, embedder: EmbeddingModel, top_k: int = 5):
		self.vector_store = vector_store
		self.embedder = embedder
		self.top_k = top_k

	def retrieve(self, question: str, product: Optional[str] = None) -> List[Dict[str, Any]]:
		q_vec = self.embedder.embed_texts([question])
		results = self.vector_store.search(q_vec, top_k=self.top_k, filter_product=product)[0]
		indices = [idx for idx, _ in results]
		scores = {idx: score for idx, score in results}
		records = self.vector_store.get_records(indices)
		for r in records:
			r["score"] = float(scores.get(r.get("index", -1), scores.get(indices[records.index(r)], 0.0)))
		# Attach score accurately using row index
		for i, rec in enumerate(records):
			rec["score"] = float(results[i][1]) if i < len(results) else 0.0
		return records