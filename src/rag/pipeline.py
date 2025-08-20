from __future__ import annotations
import os
import yaml
from typing import Optional, Dict, Any, List
import pandas as pd
from ..features.embeddings import EmbeddingModel
from ..index.store_faiss import VectorStore
from ..rag.retriever import Retriever
from ..rag.prompts import build_messages

try:
	from openai import OpenAI
	_HAS_OPENAI = True
except Exception:
	_HAS_OPENAI = False

class RAGPipeline:
	def __init__(self, config_path: str = "configs/config.yaml"):
		with open(config_path, "r", encoding="utf-8") as f:
			self.cfg = yaml.safe_load(f)

		self.embedder = EmbeddingModel(
			model_name=self.cfg["embedding"]["model_name"],
			normalize=bool(self.cfg["embedding"].get("normalize", True)),
			batch_size=int(self.cfg["embedding"].get("batch_size", 64)),
		)
		self.vs = VectorStore(self.cfg["paths"]["vector_store_dir"])
		self.vs.load()
		self.retriever = Retriever(self.vs, self.embedder, top_k=int(self.cfg["retrieval"]["top_k"]))
		self.gen_model = self.cfg["generation"]["model"]

	def answer(self, question: str, product: Optional[str] = None) -> Dict[str, Any]:
		snippets = self.retriever.retrieve(question, product=product)
		if _HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
			client = OpenAI()
			messages = build_messages(question, snippets)
			resp = client.chat.completions.create(
				model=self.gen_model,
				messages=messages,
				temperature=0.2,
				max_tokens=int(self.cfg["generation"].get("max_tokens", 500)),
			)
			answer = resp.choices[0].message.content.strip()
		else:
			# Fallback extractive summary from top snippets
			joined = " ".join([s.get("text", "") for s in snippets[:3]])
			answer = (joined[:800] + "...") if len(joined) > 800 else joined
			if not answer:
				answer = "I don't have enough context to answer."
		return {"answer": answer, "sources": snippets}

def build_index_from_filtered(config_path: str = "configs/config.yaml"):
	import numpy as np
	from ..features.chunking import make_chunks

	with open(config_path, "r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f)

	processed = cfg["paths"]["processed_dir"]
	filtered_csv = cfg["paths"]["filtered_csv"]
	chunks_parquet = cfg["paths"]["chunks_parquet"]

	if not os.path.exists(filtered_csv):
		raise FileNotFoundError(f"Expected {filtered_csv}. Run Task 1 first.")

	df = pd.read_csv(filtered_csv, dtype=str)
	df = df.fillna("")
	rows = [
		{
			"complaint_id": r["complaint_id"],
			"product": r["product"],
			"issue": r.get("issue", ""),
			"date_received": r.get("date_received", ""),
			"text": r["consumer_complaint_narrative"],
		}
		for _, r in df.iterrows()
	]

	size = int(cfg["chunking"]["size"])
	overlap = int(cfg["chunking"]["overlap"])
	chunks = list(make_chunks(rows, size=size, overlap=overlap))
	chunks_df = pd.DataFrame(chunks)
	os.makedirs(os.path.dirname(chunks_parquet), exist_ok=True)
	chunks_df.to_parquet(chunks_parquet, index=False)

	embedder = EmbeddingModel(cfg["embedding"]["model_name"])
	emb = embedder.embed_texts(chunks_df["text"].tolist())

	vs = VectorStore(cfg["paths"]["vector_store_dir"])
	vs.build(embeddings=emb, meta_df=chunks_df)
	vs.save()