from __future__ import annotations
import os
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
	def __init__(self, model_name: str | None = None, normalize: bool = True, batch_size: int = 64):
		self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
		self.normalize = normalize
		self.batch_size = batch_size
		self.model = SentenceTransformer(self.model_name, trust_remote_code=True)

	def embed_texts(self, texts: List[str]) -> np.ndarray:
		vectors = self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=self.normalize)
		return vectors.astype(np.float32)