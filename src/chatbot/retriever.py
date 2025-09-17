import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os
import time
import config

# --- CONFIGURATION ---
FAISS_INDEX_PATH = config.FAISS_INDEX_PATH
CHUNKS_JSON_PATH = config.CHUNKS_JSON_PATH
EMBED_MODEL_NAME = config.EMBED_MODEL_NAME

class VectorSearch:
    """
    Handles loading the vector index and content store for efficient semantic search.
    """
    def __init__(self, model_name, index_path, chunks_path):
        print("Initializing VectorSearch...")
        
        # 1. Load the Sentence Transformer model
        print(f"Loading model: '{model_name}'...")
        self.model = SentenceTransformer(model_name)
        
        # 2. Load the FAISS index
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index file not found at: {index_path}")
        print(f"Loading FAISS index from: '{index_path}'...")
        self.index = faiss.read_index(index_path)
        
        # 3. Load the content store (our chunked JSON)
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"Chunks JSON file not found at: {chunks_path}")
        print(f"Loading content store from: '{chunks_path}'...")
        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks_data = json.load(f)
            
        print("Initialization complete.")

    def search(self, query: str, n: int = 5):
        """
        Performs a semantic search.
        
        Workflow:
        1. Encode query to vector.
        2. Search FAISS index to get the *indices* of top N results.
        3. Use these indices to perform a fast lookup in our loaded `chunks_data` list.
        """
        start_time = time.time()
        
        query_vector = self.model.encode([query], convert_to_numpy=True).astype('float32')
        distances, indices = self.index.search(query_vector, n)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            retrieved_chunk = self.chunks_data[idx] # Fast O(1) lookup
            
            results.append({
                'score': distances[0][i],
                'content': retrieved_chunk['content'],
                'source': retrieved_chunk['source_filename']
            })
            
        end_time = time.time()
        print(f"Search completed in {end_time - start_time:.4f} seconds.")
        return results

# --- Example Usage ---
if __name__ == "__main__":
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(CHUNKS_JSON_PATH):
        print("Error: Index or chunks files not found.")
        print("Please run 'create_vectorbase_v2.py' first to generate them.")
    else:
        search_engine = VectorSearch(
            model_name=EMBED_MODEL_NAME,
            index_path=FAISS_INDEX_PATH,
            chunks_path=CHUNKS_JSON_PATH
        )

        user_query = "What are the risks of untreated hypertension in elderly patients?"
        print(f"\nQuery: '{user_query}'")
        search_results = search_engine.search(user_query, n=3)

        for i, res in enumerate(search_results):
            print(f"\n--- Result {i+1} ---")
            print(f"Score (L2 Distance): {res['score']:.4f}")
            print(f"Source: {res['source']}")
            print(f"Content: \n{res['content']}")