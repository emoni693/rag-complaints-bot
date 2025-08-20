import numpy as np
import pandas as pd
from src.index.store_faiss import VectorStore
from src.features.embeddings import EmbeddingModel
from src.rag.retriever import Retriever

def test_retriever_basic(tmp_path):
	meta = pd.DataFrame(
		[
			{"chunk_id": "1:0", "text": "buy now pay later late fee issues", "product": "buy now, pay later", "issue": "", "date_received": "", "complaint_id": "1"},
			{"chunk_id": "2:0", "text": "credit card interest rate increase", "product": "credit card", "issue": "", "date_received": "", "complaint_id": "2"},
		]
	)
	emb = EmbeddingModel().embed_texts(meta["text"].tolist())
	vs = VectorStore(str(tmp_path))
	vs.build(emb, meta)
	retriever = Retriever(vs, EmbeddingModel(), top_k=1)
	res = retriever.retrieve("BNPL late fees", product="buy now, pay later")
	assert len(res) == 1
	assert res[0]["product"].lower() == "buy now, pay later"