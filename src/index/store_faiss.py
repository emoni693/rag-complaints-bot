from __future__ import annotations
import os
import json
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd

try:
	import faiss  # type: ignore
	_HAS_FAISS = True
except Exception:
	_HAS_FAISS = False

from sklearn.neighbors import NearestNeighbors
from ..index.metadata import ComplaintChunkMetadata

class VectorStore:
	def __init__(self, dir_path: str):
		self.dir_path = dir_path
		os.makedirs(self.dir_path, exist_ok=True)
		self.index = None
		self.backend = None  # "faiss" or "sklearn"
		self.embeddings = None  # np.ndarray [n, d]
		self.meta_df = None  # pd.DataFrame with: chunk_id, text, product, issue, date_received, complaint_id

	def build(self, embeddings: np.ndarray, meta_df: pd.DataFrame):
		self.embeddings = embeddings.astype(np.float32)
		self.meta_df = meta_df.reset_index(drop=True)
		if _HAS_FAISS:
			index = faiss.IndexFlatIP(self.embeddings.shape[1])
			faiss.normalize_L2(self.embeddings)
			index.add(self.embeddings)
			self.index = index
			self.backend = "faiss"
		else:
			nn = NearestNeighbors(metric="cosine", algorithm="auto")
			nn.fit(self.embeddings)
			self.index = nn
			self.backend = "sklearn"

	def save(self):
		if self.embeddings is None or self.meta_df is None or self.index is None:
			raise RuntimeError("Nothing to save; build or load first.")
		np.save(os.path.join(self.dir_path, "embeddings.npy"), self.embeddings)
		self.meta_df.to_parquet(os.path.join(self.dir_path, "metadata.parquet"), index=False)
		with open(os.path.join(self.dir_path, "backend.json"), "w", encoding="utf-8") as f:
			json.dump({"backend": self.backend}, f)
		if self.backend == "faiss":
			faiss.write_index(self.index, os.path.join(self.dir_path, "index.faiss"))  # type: ignore
		else:
			# sklearn index is derived from embeddings; we can rebuild on load
			pass

	def load(self):
		emb_path = os.path.join(self.dir_path, "embeddings.npy")
		meta_path = os.path.join(self.dir_path, "metadata.parquet")
		if not (os.path.exists(emb_path) and os.path.exists(meta_path)):
			raise FileNotFoundError("Vector store files not found. Build first.")
		self.embeddings = np.load(emb_path).astype(np.float32)
		self.meta_df = pd.read_parquet(meta_path)
		backend_file = os.path.join(self.dir_path, "backend.json")
		if os.path.exists(backend_file):
			with open(backend_file, "r", encoding="utf-8") as f:
				info = json.load(f)
			self.backend = info.get("backend", "sklearn")
		else:
			self.backend = "sklearn"
		if self.backend == "faiss" and _HAS_FAISS:
			self.index = faiss.read_index(os.path.join(self.dir_path, "index.faiss"))  # type: ignore
		else:
			nn = NearestNeighbors(metric="cosine", algorithm="auto")
			nn.fit(self.embeddings)
			self.index = nn
			self.backend = "sklearn"

	def search(
		self,
		query_embeddings: np.ndarray,
		top_k: int = 5,
		filter_product: Optional[str] = None,
	) -> List[List[Tuple[int, float]]]:
		if self.index is None or self.embeddings is None or self.meta_df is None:
			raise RuntimeError("Vector store not initialized.")
		if self.backend == "faiss":
			q = query_embeddings.copy().astype(np.float32)
			faiss.normalize_L2(q)  # cosine via inner product on normalized vectors
			distances, indices = self.index.search(q, top_k)  # type: ignore
			# Convert to cosine similarity (already inner product); map to (idx, score)
			results = []
			for row_ix, row in enumerate(indices):
				row_res = []
				for j, idx in enumerate(row):
					if idx < 0:
						continue
					score = float(distances[row_ix, j])
					row_res.append((int(idx), score))
				results.append(row_res)
		else:
			# sklearn cosine distance → convert to similarity
			distances, indices = self.index.kneighbors(query_embeddings, n_neighbors=top_k)  # type: ignore
			results = []
			for row_ix, row in enumerate(indices):
				row_res = []
				for j, idx in enumerate(row):
					dist = float(distances[row_ix, j])
					score = 1.0 - dist
					row_res.append((int(idx), score))
				results.append(row_res)

		# Optional product filter
		if filter_product:
			filter_product = filter_product.lower()
			filtered_results = []
			for row in results:
				tmp = []
				for idx, score in row:
					if str(self.meta_df.iloc[idx]["product"]).lower() == filter_product:
						tmp.append((idx, score))
				filtered_results.append(tmp[:top_k])
			return filtered_results
		return results

	def get_records(self, indices: List[int]) -> List[Dict[str, Any]]:
		if self.meta_df is None:
			return []
		rows = self.meta_df.iloc[indices]
		return rows.to_dict(orient="records")