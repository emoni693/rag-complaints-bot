# run_task2.py (fixed version)
import os
import yaml
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import re

def clean_text(text):
    """Simple text cleaning"""
    if text is None or pd.isna(text):
        return ""
    
    # Convert to string and normalize whitespace
    normalized = re.sub(r"\s+", " ", str(text)).strip()
    
    # Convert to lowercase
    normalized = normalized.lower()
    
    # Simple text replacements
    boilerplate_phrases = [
        "thank you for your help",
        "please contact me", 
        "i have attached",
        "please help",
        "thank you",
        "best regards",
        "sincerely",
    ]
    
    for phrase in boilerplate_phrases:
        normalized = normalized.replace(phrase, "")
    
    # Clean up extra whitespace
    normalized = re.sub(r"\s+", " ", normalized).strip()
    
    return normalized

def sliding_window_chunks(text, size, overlap):
    """Create overlapping text chunks"""
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

def make_chunks(rows, size, overlap):
    """Create chunks from complaint rows"""
    chunks = []
    
    # rows is already a list, so iterate directly
    for row in rows:
        text = row["text"]
        base_id = row["complaint_id"]
        
        for idx, chunk in enumerate(sliding_window_chunks(text, size=size, overlap=overlap)):
            chunks.append({
                "chunk_id": f"{base_id}:{idx}",
                "text": chunk,
                "complaint_id": row["complaint_id"],
                "product": row["product"],
                "issue": row.get("issue", ""),
                "date_received": row.get("date_received", ""),
            })
    
    return chunks

class SimpleVectorStore:
    """Simple vector store using scikit-learn"""
    
    def __init__(self, dir_path):
        self.dir_path = dir_path
        os.makedirs(dir_path, exist_ok=True)
        self.embeddings = None
        self.meta_df = None
        self.nn = None
    
    def build(self, embeddings, meta_df):
        """Build the vector store"""
        from sklearn.neighbors import NearestNeighbors
        
        self.embeddings = embeddings.astype(np.float32)
        self.meta_df = meta_df.reset_index(drop=True)
        
        # Use scikit-learn for nearest neighbors
        self.nn = NearestNeighbors(metric="cosine", algorithm="auto")
        self.nn.fit(self.embeddings)
        print(f"✅ Built vector store with {len(self.embeddings)} chunks")
    
    def save(self):
        """Save the vector store"""
        if self.embeddings is None or self.meta_df is None:
            raise RuntimeError("Nothing to save; build first.")
        
        # Save embeddings
        np.save(os.path.join(self.dir_path, "embeddings.npy"), self.embeddings)
        
        # Save metadata
        self.meta_df.to_parquet(os.path.join(self.dir_path, "metadata.parquet"), index=False)
        
        # Save info about the store
        import json
        with open(os.path.join(self.dir_path, "store_info.json"), "w", encoding="utf-8") as f:
            json.dump({
                "backend": "sklearn",
                "num_chunks": len(self.embeddings),
                "embedding_dim": self.embeddings.shape[1]
            }, f, indent=2)
        
        print(f"✅ Saved vector store to {self.dir_path}")

def main():
    print("=== RAG Complaints Bot - Task 2: Building Vector Index ===")
    
    # Load config
    config_path = "configs/config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found at {config_path}")
        return
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # Check if filtered data exists
    filtered_csv = cfg["paths"]["filtered_csv"]
    if not os.path.exists(filtered_csv):
        print(f"❌ Filtered data not found at {filtered_csv}")
        print("Please run Task 1 first to create the filtered dataset")
        return
    
    print(f"✅ Found filtered data: {filtered_csv}")
    
    # Load filtered data
    print("📊 Loading filtered complaints...")
    df = pd.read_csv(filtered_csv, dtype=str)
    df = df.fillna("")
    print(f"✅ Loaded {len(df)} complaints")
    
    # Prepare rows for chunking
    print(" Preparing data for chunking...")
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "complaint_id": row["complaint_id"],
            "product": row["product"],
            "issue": row.get("issue", ""),
            "date_received": row.get("date_received", ""),
            "text": row["consumer_complaint_narrative"],
        })
    
    # Get chunking parameters
    size = int(cfg["chunking"]["size"])
    overlap = int(cfg["chunking"]["overlap"])
    
    print(f" Chunking with size={size}, overlap={overlap}")
    
    # Create chunks
    print("✂️  Creating text chunks...")
    chunks = make_chunks(rows, size=size, overlap=overlap)
    chunks_df = pd.DataFrame(chunks)
    
    print(f"✅ Created {len(chunks)} chunks from {len(rows)} complaints")
    
    # Save chunks (optional)
    chunks_parquet = cfg["paths"]["chunks_parquet"]
    os.makedirs(os.path.dirname(chunks_parquet), exist_ok=True)
    chunks_df.to_parquet(chunks_parquet, index=False)
    print(f"💾 Saved chunks to {chunks_parquet}")
    
    # Load embedding model
    print("🤖 Loading embedding model...")
    model_name = cfg["embedding"]["model_name"]
    print(f"Using model: {model_name}")
    
    try:
        embedder = SentenceTransformer(model_name, trust_remote_code=True)
        print("✅ Embedding model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading embedding model: {e}")
        print("Trying alternative model...")
        try:
            embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            print("✅ Alternative model loaded successfully")
        except Exception as e2:
            print(f"❌ Failed to load any embedding model: {e2}")
            return
    
    # Generate embeddings
    print("🔢 Generating embeddings...")
    batch_size = int(cfg["embedding"].get("batch_size", 64))
    
    texts = chunks_df["text"].tolist()
    print(f"Processing {len(texts)} chunks in batches of {batch_size}")
    
    embeddings = embedder.encode(
        texts, 
        batch_size=batch_size, 
        show_progress_bar=True, 
        convert_to_numpy=True, 
        normalize_embeddings=bool(cfg["embedding"].get("normalize", True))
    )
    
    print(f"✅ Generated embeddings: {embeddings.shape}")
    
    # Build vector store
    print("🏗️  Building vector store...")
    vector_store_dir = cfg["paths"]["vector_store_dir"]
    vs = SimpleVectorStore(vector_store_dir)
    
    vs.build(embeddings=embeddings, meta_df=chunks_df)
    vs.save()
    
    # Show summary
    print("\n" + "="*50)
    print("📊 INDEX BUILDING SUMMARY")
    print("="*50)
    print(f"Original complaints: {len(rows)}")
    print(f"Text chunks created: {len(chunks)}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")
    print(f"Vector store location: {vector_store_dir}")
    print(f"Chunks saved to: {chunks_parquet}")
    
    print("\n✅ Task 2 completed successfully!")
    print("You can now proceed to Task 3: Testing the RAG pipeline")

if __name__ == "__main__":
    main()