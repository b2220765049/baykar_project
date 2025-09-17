# DATASET CONFIGURATION
PDF_DIRECTORY = "data/pdf_files"
INPUT_JSON_PATH = "data/extracted_files/extracted_data.json"
CSV_OUTPUT_FILE = "data/extracted_files/extracted_data.csv"
LOG_DATABASE_PATH = "data/logs/chat_logs.db"

# Chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 300

# Vector base files
CHUNKS_JSON_PATH = "data/extracted_files/hypertension_chunks_biobert_1000.json"
FAISS_INDEX_PATH = "data/vectorbase/hypertension_vectorbase_biobert_1000.index"

# API CONFIGURATION
import os
# Allow overriding API URL via environment so the UI container can reach the API
# at http://app:8000/query while local runs default to localhost.
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/query")

# CHATBOT UI CONFIGURATION
MAX_CHAT_ROWS = 50

# Model CONFIGURATION
EMBED_MODEL_NAME = 'dmis-lab/biobert-base-cased-v1.2'
PERPLEXITY_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
BERTSCORE_MODEL_TYPE = "dmis-lab/biobert-base-cased-v1.2"

# EVALUATION CONFIGURATION
RAW_RESULTS_FILE = "data/evaluate/raw_inference_resultsV3.json"
FINAL_CSV_REPORT = "data/evaluate/rag_evaluation_reportV3.csv"
QA_PAIRS_FILE = "data/evaluate/test_questionsV2.json"
MANUEL_METRICS_CSV = "data/evaluate/manuel_metricsV3.csv"