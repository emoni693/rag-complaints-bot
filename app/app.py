# app_simple.py (updated - no sources display)
import os
import yaml
import pandas as pd
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

class SimpleRAGPipeline:
    """Simple RAG pipeline for the Streamlit app"""
    
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
    
    def retrieve(self, question, product=None):
        """Retrieve relevant chunks"""
        # Embed question
        q_vec = self.embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)
        
        # Search
        distances, indices = self.nn.kneighbors(q_vec, n_neighbors=self.top_k)
        
        # Get results
        results = []
        for i, idx in enumerate(indices[0]):
            score = 1.0 - distances[0][i] # Convert distance to similarity
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
            return {"answer": "I don't have enough context to answer this question."}
        
        # Create a better answer by combining multiple snippets
        if len(snippets) >= 3:
            # Combine top 3 snippets for richer answer
            combined_text = " ".join([
                snippets[0]['text'][:150],
                snippets[1]['text'][:150], 
                snippets[2]['text'][:150]
            ])
        else:
            combined_text = snippets[0]['text'][:300]
        
        # Generate contextual answer based on question type
        if "why" in question.lower():
            answer = f"Based on the complaints, the main issues are: {combined_text[:400]}..."
        elif "what" in question.lower():
            answer = f"The complaints show: {combined_text[:400]}..."
        elif "how" in question.lower():
            answer = f"Here's how the issues occur: {combined_text[:400]}..."
        elif "credit" in question.lower() or "card" in question.lower():
            answer = f"Credit card complaints typically involve: {combined_text[:400]}..."
        elif "loan" in question.lower():
            answer = f"Personal loan issues commonly include: {combined_text[:400]}..."
        elif "bnpl" in question.lower() or "buy now" in question.lower():
            answer = f"Buy Now Pay Later complaints often relate to: {combined_text[:400]}..."
        else:
            answer = f"Here's what I found: {combined_text[:400]}..."
        
        return {"answer": answer}

def main():
    st.set_page_config(
        page_title="RAG Complaint Analysis", 
        layout="wide",
        page_icon="��"
    )
    
    st.title("🤖 Intelligent Complaint Analysis (RAG)")
    st.markdown("Ask questions about customer complaints across financial products")
    
    # Check if vector store exists
    config_path = "configs/config.yaml"
    if not os.path.exists(config_path):
        st.error("❌ Config file not found. Please ensure configs/config.yaml exists.")
        return
    
    # Load config
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        st.error(f"❌ Error loading config: {e}")
        return
    
    vector_store_dir = cfg["paths"]["vector_store_dir"]
    if not os.path.exists(vector_store_dir):
        st.error("❌ Vector store not found. Please run Task 2 first to build the vector index.")
        return
    
    # Initialize RAG pipeline
    try:
        pipeline = SimpleRAGPipeline(config_path)
        st.success("✅ RAG pipeline loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading RAG pipeline: {e}")
        return
    
    # Sidebar for filters
    with st.sidebar:
        st.header("🔍 Filters")
        product_filter = st.selectbox(
            "Product (optional)",
            ["", "credit card", "personal loan", "buy now, pay later", "savings account", "money transfer"],
            index=0,
            help="Filter complaints by specific product"
        )
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("**Available Products:**")
        for product in cfg["products"]:
            st.markdown(f"- {product}")
        
        st.markdown("---")
        st.markdown("**Sample Questions:**")
        st.markdown("- Why are people unhappy with credit cards?")
        st.markdown("- What are the main BNPL complaints?")
        st.markdown("- How do banks handle loan issues?")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about complaints..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                try:
                    result = pipeline.answer(prompt, product_filter)
                    
                    # Display clean answer only
                    st.markdown(result['answer'])
                    
                    # Add assistant message to chat history (just the answer)
                    st.session_state.messages.append({"role": "assistant", "content": result['answer']})
                    
                except Exception as e:
                    error_msg = f"❌ Error generating response: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Footer
    st.markdown("---")

if __name__ == "__main__":
    main()