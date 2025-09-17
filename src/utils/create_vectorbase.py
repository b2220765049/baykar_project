import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import os
import torch
import config

# --- CONFIGURATION ---
# Input and output paths (updated for clarity)
INPUT_JSON_PATH = config.INPUT_JSON_PATH
CHUNKS_JSON_PATH = config.CHUNKS_JSON_PATH
FAISS_INDEX_PATH = config.FAISS_INDEX_PATH
MODEL_NAME = config.EMBED_MODEL_NAME
CHUNK_SIZE = config.CHUNK_SIZE
CHUNK_OVERLAP = config.CHUNK_OVERLAP

def load_documents(file_path):
    """Loads documents from the specified JSON file."""
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file '{file_path}'.")
        return []

def create_and_save_artifacts():
    """
    Processes documents, creates a vector index and a content store using BioBERT, and saves them.
    """
    # 1. Load the extracted text data
    documents = load_documents(INPUT_JSON_PATH)
    if not documents:
        print("No documents to process. Exiting.")
        return

    # 2. Chunk the documents
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    
    chunks_with_metadata = []
    for doc in tqdm(documents, desc="Processing Documents"):
        chunks = text_splitter.split_text(doc['content'])
        for chunk in chunks:
            chunks_with_metadata.append({
                'source_filename': doc['filename'],
                'content': chunk
            })
            
    print(f"Total chunks created: {len(chunks_with_metadata)}")

    # 3. Save the chunked content store
    print(f"Saving chunked content to '{CHUNKS_JSON_PATH}'...")
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(CHUNKS_JSON_PATH), exist_ok=True)
    with open(CHUNKS_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(chunks_with_metadata, f, indent=2, ensure_ascii=False)

    # 4. Prepare text chunks for embedding
    chunks_for_embedding = [item['content'] for item in chunks_with_metadata]

    # --- KEY CHANGE: MODEL LOADING AND DEVICE PLACEMENT ---
    # 5. Load the embedding model and set device
    # Check if a CUDA-enabled GPU is available, otherwise use CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Loading BioBERT-based sentence transformer model: '{MODEL_NAME}'...")
    # This will download the model from Hugging Face the first time you run it.
    model = SentenceTransformer(MODEL_NAME, device=device)

    # 6. Generate embeddings
    print("Generating embeddings... (This will be slower than MiniLM and may take a while)")
    # We can specify a batch size that fits into our GPU/CPU memory
    embeddings = model.encode(
        chunks_for_embedding, 
        show_progress_bar=True, 
        convert_to_numpy=True,
        batch_size=32 # Adjust batch size based on your VRAM/RAM
    )
    embeddings = embeddings.astype('float32')
    
    # 7. Create and build the FAISS index
    embedding_dimension = embeddings.shape[1]
    print(f"Embeddings created with dimension: {embedding_dimension}")
    index = faiss.IndexFlatL2(embedding_dimension)
    index.add(embeddings)
    
    # 8. Save the FAISS index
    print(f"Saving FAISS index to '{FAISS_INDEX_PATH}'...")
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)
        
    print("\nArtifact creation complete!")
    print(f"Vector Index: {FAISS_INDEX_PATH} ({index.ntotal} vectors)")
    print(f"Content Store: {CHUNKS_JSON_PATH} ({len(chunks_with_metadata)} chunks)")

if __name__ == "__main__":
    create_and_save_artifacts()