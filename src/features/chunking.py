from __future__ import annotations
from typing import Iterable, List, Dict

def sliding_window_chunks(text: str, size: int, overlap: int) -> List[str]:
	if not text:
		return []
	if overlap >= size:
		raise ValueError("overlap must be smaller than size")
	chunks = []
	start = 0
	while start < len(text):
		end = min(len(text), start + size)
		chunks.append(text[start:end])
		if end == len(text):
			break
		start = end - overlap
	return chunks

def make_chunks(rows: Iterable[Dict], size: int, overlap: int):
	"""
	rows: iterable of dicts with keys: complaint_id, product, issue, date_received, text
	yield: dict with chunk_id, text, complaint_id, product, issue, date_received
	"""
	for row in rows:
		text = row["text"]
		base_id = row["complaint_id"]
		for idx, chunk in enumerate(sliding_window_chunks(text, size=size, overlap=overlap)):
			yield {
				"chunk_id": f"{base_id}:{idx}",
				"text": chunk,
				"complaint_id": row["complaint_id"],
				"product": row["product"],
				"issue": row.get("issue", ""),
				"date_received": row.get("date_received", ""),
			}