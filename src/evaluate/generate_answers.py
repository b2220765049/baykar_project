import config
import requests
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
QA_PAIRS_FILE = config.QA_PAIRS_FILE
API_URL = config.API_URL
RAW_RESULTS_FILE = config.RAW_RESULTS_FILE

def call_rag_api(question: str):
    """Sends a question to the RAG API and returns the full response."""
    payload = {"question": question}
    try:
        response = requests.post(API_URL, json=payload, timeout=180)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API call failed for question '{question}': {e}")
        return None

def run_inference():
    """
    Queries the RAG API for all questions in the QA pairs file and saves the raw results.
    """
    # --- Load QA Pairs ---
    try:
        with open(QA_PAIRS_FILE, 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f)
    except FileNotFoundError:
        print(f"FATAL: QA pairs file not found at '{QA_PAIRS_FILE}'")
        return

    all_results = []
    
    # --- Main Inference Loop ---
    print(f"--- Starting inference on {len(qa_pairs)} QA pairs ---")
    for item in tqdm(qa_pairs, desc="Querying RAG API"):
        query = item.get("question")
        expected_answer = item.get("answer")

        if not query or not expected_answer:
            continue

        # Get response from RAG API
        api_response = call_rag_api(query)
        
        result_entry = {
            "query": query,
            "expected_answer": expected_answer
        }

        if api_response:
            result_entry.update({
                "generated_answer": api_response.get("answer", ""),
                "retrieval_time_seconds": api_response.get("retrieval_time_seconds", 0),
                "generation_time_seconds": api_response.get("generation_time_seconds", 0),
                "total_time_seconds": api_response.get("total_time_seconds", 0),
            })
        else:
            result_entry["generated_answer"] = "API_CALL_FAILED"

        all_results.append(result_entry)

    # --- Save Raw Results ---
    with open(RAW_RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nInference complete. Raw results saved to '{RAW_RESULTS_FILE}'")

if __name__ == "__main__":
    run_inference()