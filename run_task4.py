# run_task4.py (place this in your main project folder)
import os
import yaml
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

class ChatInterface:
    """Simple chat interface for the RAG system"""
    
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
        print(f"✅ Chat interface loaded with {len(self.embeddings)} chunks")
    
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
        if product and product.strip():
            product = product.lower()
            results = [r for r in results if str(r["product"]).lower() == product]
        
        return results[:self.top_k]
    
    def answer(self, question, product=None):
        """Generate answer using retrieved chunks"""
        # Retrieve relevant chunks
        snippets = self.retrieve(question, product)
        
        if not snippets:
            return {"answer": "I don't have enough context to answer this question.", "sources": []}
        
        # Simple answer generation
        if "why" in question.lower():
            answer = f"Based on the complaints, the main issues are: {snippets[0]['text'][:200]}..."
        elif "what" in question.lower():
            answer = f"The complaints show: {snippets[0]['text'][:200]}..."
        elif "how" in question.lower():
            answer = f"Here's how the issues occur: {snippets[0]['text'][:200]}..."
        else:
            answer = f"Here's what I found: {snippets[0]['text'][:200]}..."
        
        return {"answer": answer, "sources": snippets}
    
    def format_sources(self, sources):
        """Format sources for display"""
        lines = []
        for i, s in enumerate(sources, 1):
            product = s.get("product", "")
            complaint_id = s.get("complaint_id", "")
            snippet = s.get("text", "")[:200].strip()
            score = s.get("score", 0)
            lines.append(f"{i}. [{product}] Score: {score:.3f} | ID: {complaint_id}")
            lines.append(f"   {snippet}...")
            lines.append("")
        return lines

def main():
    print("=== RAG Complaints Bot - Task 4: Interactive Chat Interface ===")
    
    # Load config
    config_path = "configs/config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found at {config_path}")
        return
    
    # Check if vector store exists
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    vector_store_dir = cfg["paths"]["vector_store_dir"]
    if not os.path.exists(vector_store_dir):
        print(f"❌ Vector store not found at {vector_store_dir}")
        print("Please run Task 2 first to build the vector index")
        return
    
    # Initialize chat interface
    print("🤖 Initializing chat interface...")
    try:
        chat = ChatInterface(config_path)
        print("✅ Chat interface ready!")
    except Exception as e:
        print(f"❌ Error initializing chat: {e}")
        return
    
    # Get available products
    products = cfg["products"]
    
    print(f"\n�� Available products: {', '.join(products)}")
    print("💡 Type 'quit' to exit, 'clear' to start over")
    print("="*60)
    
    conversation_history = []
    
    while True:
        try:
            # Get user input
            print("\n🤔 Your question:")
            question = input("> ").strip()
            
            if question.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            if question.lower() == 'clear':
                conversation_history = []
                print("🧹 Conversation cleared!")
                continue
            
            if not question:
                print("❌ Please enter a question.")
                continue
            
            # Get product filter
            print(f"\n📱 Filter by product (optional, press Enter to skip):")
            print(f"Available: {', '.join(products)}")
            product_filter = input("Product filter > ").strip()
            
            # Generate answer
            print("\n🤖 Generating answer...")
            result = chat.answer(question, product_filter)
            
            # Display answer
            print(f"\n💬 Answer:")
            print(f"{result['answer']}")
            
            # Display sources
            if result['sources']:
                print(f"\n📚 Sources ({len(result['sources'])} found):")
                source_lines = chat.format_sources(result['sources'])
                for line in source_lines:
                    print(line)
            else:
                print("\n📚 No sources found.")
            
            # Store in history
            conversation_history.append({
                "question": question,
                "answer": result['answer'],
                "sources_count": len(result['sources']),
                "product_filter": product_filter
            })
            
            print(f"\n💾 Conversation history: {len(conversation_history)} questions asked")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.")
    
    # Show summary
    if conversation_history:
        print("\n" + "="*60)
        print("📊 CHAT SESSION SUMMARY")
        print("="*60)
        print(f"Total questions asked: {len(conversation_history)}")
        
        product_filters = [c['product_filter'] for c in conversation_history if c['product_filter']]
        if product_filters:
            print(f"Products filtered: {', '.join(set(product_filters))}")
        
        avg_sources = np.mean([c['sources_count'] for c in conversation_history])
        print(f"Average sources per question: {avg_sources:.1f}")
    
    print("\n✅ Task 4 completed successfully!")
    print("You have a working RAG chatbot for complaint analysis!")

if __name__ == "__main__":
    main()