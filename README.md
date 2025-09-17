# RAG API and UI (Dockerized)

Retrieval-Augmented Generation service with:
- API: FastAPI (Uvicorn), logs and chat management (SQLite)
- UI: Gradio web app with clickable example questions

## Features
- LLM: meta-llama/Meta-Llama-3-8B-Instruct
- Retriever: FAISS + BioBERT (dmis-lab/biobert-base-cased-v1.2)
- Persistence: SQLite (API logs, chat sessions/messages)
- Docker Compose: separate API (8000) and UI (7860) services

## Architecture
```
+-----------------+        Docker network        +-----------------+
|     UI (7860)   |  --->  http://app:8000  ---> |   API (8000)    |
|   Gradio app    |                              |  FastAPI+LLM    |
+-----------------+                              +-----------------+
            ^                                             |
            |                            HF Hub (token)   v
            |---------------------------------------------+
```

## Prerequisites
- Windows + Docker Desktop (WSL2 backend)
- NVIDIA GPU recommended
- Hugging Face account + accepted model terms + read token

## Hugging Face token and model access
This project requires a Hugging Face access token with at least "read" scope and explicit permission to use gated models (for example `meta-llama/Meta-Llama-3-8B-Instruct`).

Steps:

1. Create or sign in to a Hugging Face account at https://huggingface.co.
2. Visit the model page (for example https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) and accept any licensing or access terms shown on the model page. Some gated models require you to request or accept the license before downloads are allowed.
3. Create an access token:
  - Click your avatar → Settings → Access Tokens → New token.
  - Give it a name and select the "read" scope (no write/admin scope required for model downloads).
  - Copy the token; store it securely.
4. Place the token into your `.env` file as:

```ini
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxx
```

Quick test (PowerShell): verify the token can authenticate to Hugging Face's whoami endpoint:

```powershell
$env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxx"   # or ensure .env is loaded
Invoke-RestMethod -Uri "https://huggingface.co/api/whoami-v2" -Headers @{ Authorization = "Bearer $env:HF_TOKEN" }
```

Or test inside the running `app` container (Docker Compose):

```powershell
docker compose exec app python -c "from huggingface_hub import HfApi; import os; api=HfApi(token=os.getenv('HF_TOKEN')); print(api.whoami())"
```

Notes:
- If a model is gated you must accept its license on the model page before `transformers` or `snapshot_download` can fetch its files.
- Do not commit your token to version control. Use `.env` or Docker secrets in production.

## Setup
1) Create a `.env` next to `compose.yaml`:
```ini
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxx
```

2) Build and run:
```powershell
docker compose up --build
# subsequent runs:
docker compose up
```

3) Open:
- UI: http://localhost:7860
- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## REST API (overview)
- GET /health
- GET /meta
- POST /query
- GET /logs
- GET /chats
- GET /chats/{session_id}
- DELETE /chats/{session_id}

Interactive details and try-it-out: http://localhost:8000/docs

Example PowerShell call:
```powershell
$payload = @{ question = "What is hypertension?"; max_tokens = 150; temperature = 0.0 }
Invoke-RestMethod -Uri "http://localhost:8000/query" -Method Post -ContentType "application/json" -Body ($payload | ConvertTo-Json -Depth 5)
```

## Data & Evaluation
- SQLite DB: `data/logs/chat_logs.db` (api_logs, chat_sessions, chat_messages)
- Retriever assets:
  - `data/vectorbase/hypertension_vectorbase_biobert_1000.index`
  - `data/extracted_files/hypertension_chunks_biobert_1000.json`
- Combined metrics CSV: `data/evaluate/combined_evaluation_metrics.csv`
  - Produced from a 100-question QA set; includes human and automatic metrics.

Notebook for quick DB inspection:
- `src/utils/read_logs.ipynb`

## Troubleshooting
- HF token warning: ensure `.env` contains `HF_TOKEN`.
- UI can’t reach API: confirm both containers run; API on 8000, UI on 7860.
- First run slow: model shards download; later runs use cache.
- GPU OOM: lower max tokens/sequence length or use a larger GPU.

## Security
- Do not commit `.env` or `chat_logs.db`.
- Keep ports private or front with a reverse proxy as needed.

## Project Layout
```
c:\baykar_project
├─ api.py
├─ UI.py
├─ compose.yaml
├─ .env
├─ data\
│  ├─ logs\chat_logs.db
│  ├─ evaluate\combined_evaluation_metrics.csv
│  ├─ vectorbase\*.index
│  └─ extracted_files\*.json
└─ src\
   └─ utils\read_logs.ipynb
```