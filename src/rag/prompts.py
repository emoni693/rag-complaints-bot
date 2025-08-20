SYSTEM_PROMPT = """You are a financial services analyst. Answer using ONLY the provided complaint context.
- Cite 2-3 short snippets.
- If the context is insufficient, say you don't know.
- Keep answers concise and specific to the question.
"""

def build_messages(question: str, snippets: list[dict]) -> list[dict]:
	context_lines = []
	for s in snippets[:6]:
		product = s.get("product", "")
		text = s.get("text", "")[:400]
		context_lines.append(f"[{product}] {text}")
	context = "\n\n".join(context_lines) if context_lines else "No context."
	user_content = f"Question:\n{question}\n\nEvidence:\n{context}"
	return [
		{"role": "system", "content": SYSTEM_PROMPT},
		{"role": "user", "content": user_content},
	]