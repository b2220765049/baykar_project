import json
import csv
import nltk
import torch
from tqdm import tqdm
from rouge_score import rouge_scorer
from bert_score import score as bert_score_func
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import config
import warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
RAW_RESULTS_FILE = config.RAW_RESULTS_FILE
FINAL_CSV_REPORT = config.FINAL_CSV_REPORT
PERPLEXITY_MODEL_NAME = config.PERPLEXITY_MODEL_NAME
BERTSCORE_MODEL_TYPE = config.BERTSCORE_MODEL_TYPE

# --- One-time NLTK Downloads ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except:
    print("Downloading NLTK resources (punkt, wordnet)...")
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)

def calculate_perplexity(text, model, tokenizer, device="cuda"):
    """Calculates the perplexity of a given text."""
    if not text or not text.strip():
        return float('nan')
    try:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        return torch.exp(outputs.loss).item()
    except Exception:
        return float('nan')

def calculate_all_metrics():
    """
    Reads raw inference results and calculates all evaluation metrics,
    using batching for BERTScore for improved performance.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: No GPU found. Metric calculation will be very slow.")

    # --- Load Raw Results ---
    try:
        with open(RAW_RESULTS_FILE, 'r', encoding='utf-8') as f:
            raw_results = json.load(f)
    except FileNotFoundError:
        print(f"FATAL: Raw results file not found at '{RAW_RESULTS_FILE}'. Run 'run_inference.py' first.")
        return

    # --- Load Models and Scorers ---
    print("--- Loading evaluation models and scorers ---")
    print(f"Loading perplexity model: {PERPLEXITY_MODEL_NAME}...")
    try:
        p_tokenizer = AutoTokenizer.from_pretrained(PERPLEXITY_MODEL_NAME)
        p_model = AutoModelForCausalLM.from_pretrained(PERPLEXITY_MODEL_NAME, torch_dtype=torch.float16).to(device)
        p_model.eval()
    except Exception as e:
        print(f"FATAL: Could not load perplexity model. Error: {e}")
        return
    
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    print(f"Using BERTScore model: {BERTSCORE_MODEL_TYPE}")

    # --- Main Calculation Loop (Pass 1: Non-batched metrics and data collection) ---
    print(f"\n--- Calculating non-batched metrics for {len(raw_results)} results ---")
    
    bert_candidates = []
    bert_references = []
    valid_indices = []

    for i, item in enumerate(tqdm(raw_results, desc="Calculating Metrics (non-batched)")):
        generated_answer = item.get("generated_answer", "")
        expected_answer = item.get("expected_answer", "")

        if generated_answer == "API_CALL_FAILED" or not generated_answer.strip():
            item.update({
                "bleu": 0, "meteor": 0, "rouge1": 0, "rouge2": 0, "rougeL": 0, 
                "bert_precision": 0, "bert_recall": 0, "bert_f1": 0, 
                "perplexity": float('nan')
            })
            continue

        # Calculate lexical metrics, ROUGE, and Perplexity...
        expected_tokens = nltk.word_tokenize(expected_answer)
        generated_tokens = nltk.word_tokenize(generated_answer)
        item["bleu"] = nltk.translate.bleu_score.sentence_bleu([expected_tokens], generated_tokens)
        item["meteor"] = nltk.translate.meteor_score.single_meteor_score(expected_tokens, generated_tokens)
        
        rouge_scores = rouge.score(expected_answer, generated_answer)
        item["rouge1"] = rouge_scores['rouge1'].fmeasure
        item["rouge2"] = rouge_scores['rouge2'].fmeasure
        item["rougeL"] = rouge_scores['rougeL'].fmeasure

        item["perplexity"] = calculate_perplexity(generated_answer, p_model, p_tokenizer, device)

        bert_candidates.append(generated_answer)
        bert_references.append(expected_answer)
        valid_indices.append(i)

    # --- Batched Calculation (Pass 2: BERTScore) ---
    if bert_candidates:
        print(f"\n--- Calculating BERTScore for {len(bert_candidates)} valid items in a batch ---")
        
        ### FIX: Dynamically get the number of layers from the model's config ###
        # This makes it robust if you change BERTSCORE_MODEL_TYPE later.
        # We load the model config only, not the full weights, so it's fast.
        bert_model_config = AutoModel.from_pretrained(BERTSCORE_MODEL_TYPE).config
        num_layers = bert_model_config.num_hidden_layers
        print(f"Detected {num_layers} layers for {BERTSCORE_MODEL_TYPE}.")

        P, R, F1 = bert_score_func(
            bert_candidates, 
            bert_references, 
            model_type=BERTSCORE_MODEL_TYPE,
            num_layers=num_layers, # Provide the number of layers
            lang="en", 
            device=device, 
            verbose=True
        )
        
        for i, (p_score, r_score, f1_score) in enumerate(zip(P, R, F1)):
            original_index = valid_indices[i]
            raw_results[original_index]["bert_precision"] = p_score.item()
            raw_results[original_index]["bert_recall"] = r_score.item()
            raw_results[original_index]["bert_f1"] = f1_score.item()

    # --- Write Final CSV Report ---
    print(f"\n--- Writing final report to {FINAL_CSV_REPORT} ---")
    csv_header = [
        "Query", "Query_Length", "Retrieval_Time_MS", "Generation_Time_MS", "Total_Time_MS",
        "Generated_Answer", "Expected_Answer", "BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L",
        "METEOR", "BERTScore-P", "BERTScore-R", "BERTScore-F1", "Perplexity"
    ]
    
    with open(FINAL_CSV_REPORT, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_header)
        for result in raw_results:
            writer.writerow([
                result.get("query", "N/A"), 
                len(result.get("query", "")), 
                result.get("retrieval_time_seconds", 0) * 1000,
                result.get("generation_time_seconds", 0) * 1000,
                result.get("total_time_seconds", 0) * 1000,
                result.get("generated_answer", "N/A"), 
                result.get("expected_answer", "N/A"),
                result.get("bleu", "N/A"), 
                result.get("rouge1", "N/A"), 
                result.get("rouge2", "N/A"),
                result.get("rougeL", "N/A"), 
                result.get("meteor", "N/A"), 
                result.get("bert_precision", "N/A"),
                result.get("bert_recall", "N/A"),
                result.get("bert_f1", "N/A"),
                result.get("perplexity", "N/A")
            ])

    print("\nEvaluation complete.")

if __name__ == "__main__":
    calculate_all_metrics()