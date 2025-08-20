from __future__ import annotations
from typing import List, Dict

def format_sources(sources: List[Dict]) -> List[str]:
	lines = []
	for s in sources:
		product = s.get("product", "")
		complaint_id = s.get("complaint_id", "")
		snippet = s.get("text", "")[:240].strip()
		lines.append(f"{product} | id={complaint_id} | {snippet}")
	return lines