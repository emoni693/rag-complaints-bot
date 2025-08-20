from __future__ import annotations
import os
import pandas as pd
from typing import Iterable
from .cleaning import clean_text

REQUIRED_COLUMNS = [
	"product",
	"issue",
	"consumer_complaint_narrative",
	"date_received",
	"company",
	"complaint_id",
]

def load_and_prepare(csv_path: str, allowed_products: Iterable[str]) -> pd.DataFrame:
	if not os.path.exists(csv_path):
		raise FileNotFoundError(f"Missing dataset at {csv_path}")
	df = pd.read_csv(csv_path, dtype=str)
	missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns: {missing}")
	df = df.copy()
	df["product"] = df["product"].str.lower().str.strip()
	allowed = {p.lower() for p in allowed_products}
	df = df[df["product"].isin(allowed)]
	df["consumer_complaint_narrative"] = df["consumer_complaint_narrative"].map(clean_text)
	df = df[df["consumer_complaint_narrative"].str.len() > 0]
	df["complaint_id"] = df["complaint_id"].astype(str)
	df = df.reset_index(drop=True)
	return df