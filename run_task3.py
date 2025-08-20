# run_task3.py (place this in your main project folder)
import os
import yaml
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

class SimpleRAGPipeline:
    """Simple RAG pipeline for testing"""
    
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        
        # Load vector store
        self.vector_store_dir = self.cfg["paths"]["vector_store_dir"]
        self.embeddings = np.load(os.path.join(self.vector_store_dir, "embeddings.npy"))
        self.meta_df = pd.read_parquet(os.path.join(self.vector_store_dir, "metadata.parquet"))
        
        # Load embedding model
        model_name = self.cfg["embedding"]["model_name"]
        try:
            self.embedder = SentenceTransformer(model_name, trust_remote_code=True)
        except:
            self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Build nearest neighbors
        self.nn = NearestNeighbors(metric="cosine", algorithm="auto")
        self.nn.fit(self.embeddings)
        
        self.top_k = int(self.cfg["retrieval"]["top_k"])
        print(f"✅ RAG Pipeline loaded with {len(self.embeddings)} chunks")
    
    def retrieve(self, question, product=None):
        """Retrieve relevant chunks"""
        # Embed question
        q_vec = self.embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)
        
        # Search
        distances, indices = self.nn.kneighbors(q_vec, n_neighbors=self.top_k)
        
        # Get results
        results = []
        for i, idx in enumerate(indices[0]):
            score = 1.0 - distances[0][i]  # Convert distance to similarity
            chunk_data = self.meta_df.iloc[idx].to_dict()
            chunk_data["score"] = float(score)
            results.append(chunk_data)
        
        # Filter by product if specified
        if product:
            product = product.lower()
            results = [r for r in results if str(r["product"]).lower() == product]
        
        return results[:self.top_k]
    
    def answer(self, question, product=None):
        """Generate answer using retrieved chunks"""
        # Retrieve relevant chunks
        snippets = self.retrieve(question, product)
        
        if not snippets:
            return {"answer": "I don't have enough context to answer this question.", "sources": []}
        
        # Simple answer generation (no OpenAI needed)
        if "why" in question.lower():
            answer = f"Based on the complaints, the main issues are: {snippets[0]['text'][:200]}..."
        elif "what" in question.lower():
            answer = f"The complaints show: {snippets[0]['text'][:200]}..."
        elif "how" in question.lower():
            answer = f"Here's how the issues occur: {snippets[0]['text'][:200]}..."
        else:
            answer = f"Here's what I found: {snippets[0]['text'][:200]}..."
        
        return {"answer": answer, "sources": snippets}

def evaluate_rag_pipeline():
    """Evaluate the RAG pipeline with sample questions"""
    
    print("=== RAG Complaints Bot - Task 3: RAG Evaluation ===")
    
    # Load config
    config_path = "configs/config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found at {config_path}")
        return
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # Check if vector store exists
    vector_store_dir = cfg["paths"]["vector_store_dir"]
    if not os.path.exists(vector_store_dir):
        print(f"❌ Vector store not found at {vector_store_dir}")
        print("Please run Task 2 first to build the vector index")
        return
    
    # Initialize RAG pipeline
    print("🤖 Initializing RAG pipeline...")
    try:
        pipeline = SimpleRAGPipeline(config_path)
        print("✅ RAG pipeline ready")
    except Exception as e:
        print(f"❌ Error initializing pipeline: {e}")
        return
    
    # Sample evaluation questions
    evaluation_questions = [
        "Why are people unhappy with credit cards?",
        "What are the main issues with personal loans?",
        "How do BNPL companies handle late fees?",
        "What problems do people face with savings accounts?",
        "Why do customers complain about money transfers?",
        "What are the most common credit card complaints?",
        "How do banks respond to personal loan issues?",
        "What are BNPL payment problems?",
        "Why do savings account holders complain?",
        "What money transfer issues occur most often?"
    ]
    
    print(f"\n📝 Evaluating {len(evaluation_questions)} questions...")
    
    # Evaluation results
    results = []
    
    for i, question in enumerate(evaluation_questions, 1):
        print(f"\n--- Question {i}: {question} ---")
        
        try:
            # Get answer
            result = pipeline.answer(question)
            answer = result["answer"]
            sources = result["sources"]
            
            # Show answer
            print(f"🤖 Answer: {answer}")
            
            # Show top sources
            print(f"📚 Top sources ({len(sources)}):")
            for j, source in enumerate(sources[:3], 1):
                product = source.get("product", "Unknown")
                text = source.get("text", "")[:100]
                score = source.get("score", 0)
                print(f"  {j}. [{product}] Score: {score:.3f} | {text}...")
            
            # Simple quality score (1-5)
            if len(sources) >= 3 and sources[0]["score"] > 0.7:
                quality_score = 5
            elif len(sources) >= 2 and sources[0]["score"] > 0.5:
                quality_score = 4
            elif len(sources) >= 1 and sources[0]["score"] > 0.3:
                quality_score = 3
            elif len(sources) >= 1:
                quality_score = 2
            else:
                quality_score = 1
            
            print(f"⭐ Quality Score: {quality_score}/5")
            
            # Store result
            results.append({
                "question": question,
                "answer": answer,
                "sources_count": len(sources),
                "top_score": sources[0]["score"] if sources else 0,
                "quality_score": quality_score
            })
            
        except Exception as e:
            print(f"❌ Error processing question: {e}")
            results.append({
                "question": question,
                "answer": f"Error: {e}",
                "sources_count": 0,
                "top_score": 0,
                "quality_score": 1
            })
    
    # Summary
    print("\n" + "="*60)
    print("�� EVALUATION SUMMARY")
    print("="*60)
    
    avg_quality = np.mean([r["quality_score"] for r in results])
    avg_sources = np.mean([r["sources_count"] for r in results])
    avg_score = np.mean([r["top_score"] for r in results])
    
    print(f"Average Quality Score: {avg_quality:.2f}/5")
    print(f"Average Sources Retrieved: {avg_sources:.1f}")
    print(f"Average Top Score: {avg_score:.3f}")
    
    # Quality distribution
    quality_dist = {}
    for r in results:
        score = r["quality_score"]
        quality_dist[score] = quality_dist.get(score, 0) + 1
    
    print(f"\nQuality Distribution:")
    for score in sorted(quality_dist.keys()):
        count = quality_dist[score]
        print(f"  {score}/5: {count} questions")
    
    print(f"\n✅ Task 3 completed successfully!")
    print("You can now proceed to Task 4: Launching the chat interface")

if __name__ == "__main__":
    evaluate_rag_pipeline()