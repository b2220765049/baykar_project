# Run with Docker Hub (no local build)

These instructions run API and UI directly from the published Docker Hub image. Commands are for Windows PowerShell.

## Prerequisites
- Docker Desktop
- NVIDIA GPU (optional) + NVIDIA Container Toolkit (for `--gpus all`)
- Hugging Face token in your environment:
  ```powershell
  $env:HF_TOKEN = "<your_hf_token>"
  ```

Create local folders for persistence:
```powershell
New-Item -ItemType Directory -Force -Path .\data\logs | Out-Null
New-Item -ItemType Directory -Force -Path .\hf-cache | Out-Null
```

## Option A — Run API and UI as two containers (recommended)

1) Create an isolated network
```powershell
docker network create rag-net
```

2) Start the API
```powershell
docker run -d --gpus all --name medical-rag-api --network rag-net `
  -p 8000:8000 `
  -e HF_TOKEN="$env:HF_TOKEN" `
  -v ${PWD}\data\logs:/app/data/logs `
  -v ${PWD}\hf-cache:/home/appuser/.cache/huggingface `
  kuark7/medical_rag:latest
```

3) Start the UI (same image, run UI.py, point to API by container name)
```powershell
docker run -d --gpus all --name medical-rag-ui --network rag-net `
  -p 7860:7860 `
  -e API_URL="http://medical-rag-api:8000/query" `
  -e GRADIO_SERVER_NAME="0.0.0.0" `
  -e GRADIO_SERVER_PORT="7860" `
  -e HF_TOKEN="$env:HF_TOKEN" `
  -v ${PWD}\data\logs:/app/data/logs `
  -v ${PWD}\hf-cache:/home/appuser/.cache/huggingface `
  --entrypoint python `
  kuark7/medical_rag:latest UI.py
```

Open:
- API health: http://localhost:8000/health
- UI: http://localhost:7860

Notes:
- The UI reaches the API via Docker DNS name `medical-rag-api`.
- Chat history persists in `.\data\logs\chat_logs.db`.

## Option B — API only
```powershell
docker run -d --gpus all --name medical-rag-api `
  -p 8000:8000 `
  -e HF_TOKEN="$env:HF_TOKEN" `
  -v ${PWD}\data\logs:/app/data/logs `
  -v ${PWD}\hf-cache:/home/appuser/.cache/huggingface `
  kuark/medical_rag:latest
```
- Health: `Invoke-RestMethod http://127.0.0.1:8000/health`
- Query: POST http://localhost:8000/query

## Option C — UI only (point to a remote API)
```powershell
docker run -d --name medical-rag-ui `
  -p 7860:7860 `
  -e API_URL="http://YOUR_API_HOST:8000/query" `
  -e GRADIO_SERVER_NAME="0.0.0.0" `
  -e GRADIO_SERVER_PORT="7860" `
  -e HF_TOKEN="$env:HF_TOKEN" `
  kuark/medical_rag:latest `
  --entrypoint python UI.py
```

## Troubleshooting
- From the host, use http://localhost:8000 for API; container DNS names (e.g., `medical-rag-api`) work only inside Docker networks.
- If Postman fails but PowerShell works, disable Postman proxy for localhost/127.0.0.1.
- If port 8000/7860 is busy, change the left side of `-p`, e.g., `-p 18000:8000`.
- Ensure volumes exist and are writable: `.\data\logs`, `.\hf-cache`.
- Check logs:
  ```powershell
  docker logs medical-rag-api --tail 100
  docker logs medical-rag-ui --tail 100
  ```

## Stop/cleanup
```powershell
docker stop medical-rag-ui medical-rag-api
docker rm medical-rag-ui medical-rag-api
docker network rm rag-net
```