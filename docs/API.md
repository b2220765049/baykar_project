# Medical RAG API Documentation (FastAPI)

This document describes the REST API exposed by the Medical RAG service. The API is implemented with FastAPI and automatically provides interactive docs at:

- Swagger UI: /docs
- ReDoc: /redoc

Adjust the base URL as needed (for local runs: http://localhost:8000).

## Overview

- Title: Medical RAG API
- Version: 1.2.0
- Description: Retrieves relevant context and uses an LLM to generate grounded answers with source files.
- Content type: application/json
- Auth: None (expects gated model access via server-side Hugging Face token)

## Base URL

- Localhost: http://localhost:8000
- Docker Compose (from UI container): http://app:8000

## Endpoints

### 1) Health

- Method: GET
- Path: /health
- Response: { "ok": true }
- Purpose: Liveness probe.

### 2) Metadata

- Method: GET
- Path: /meta
- Response:
  - version: string
  - model: string
  - model_available: boolean
  - search_available: boolean
  - device: string (e.g., "cuda" or "cpu")

Example response:

```
{
  "version": "1.2.0",
  "model": "meta-llama/Meta-Llama-3-8B-Instruct",
  "model_available": true,
  "search_available": true,
  "device": "cuda"
}
```

### 3) Query (RAG: retrieval + generation)

- Method: POST
- Path: /query
- Request body (QueryRequest):
  - question: string (required)
  - max_tokens: integer (default 256)
  - temperature: number (default 0.0)
- Response (QueryResponse):
  - answer: string
  - retrieved_sources: string[] (filenames)
  - retrieval_time_seconds: number
  - generation_time_seconds: number
  - total_time_seconds: number

Example request:

```
{
  "question": "What is hypertension?",
  "max_tokens": 150,
  "temperature": 0.0
}
```

Example response:

```
{
  "answer": "... grounded answer with inline [Source: file.pdf] citations ...",
  "retrieved_sources": ["file1.pdf", "file2.pdf"],
  "retrieval_time_seconds": 0.018,
  "generation_time_seconds": 4.12,
  "total_time_seconds": 4.16
}
```

### 4) Logs (API call history)

- Method: GET
- Path: /logs
- Query params:
  - limit: integer (default 100)
  - offset: integer (default 0)
- Response:
  - logs: Array of log entries
    - id: number
    - timestamp: string (ISO-8601)
    - question: string
    - answer: string
    - retrieved_sources: string[]
    - retrieval_time: number
    - generation_time: number
    - total_time: number
  - limit: number
  - offset: number

Example response (truncated):

```
{
  "logs": [
    {
      "id": 1258,
      "timestamp": "2025-09-17T15:03:31.657730",
      "question": "What is hypertension?",
      "answer": "...",
      "retrieved_sources": ["file1.pdf", "file2.pdf"],
      "retrieval_time": 0.018,
      "generation_time": 4.12,
      "total_time": 4.16
    }
  ],
  "limit": 10,
  "offset": 0
}
```

### 5) Chats: list sessions

- Method: GET
- Path: /chats
- Query params:
  - limit: integer (default 100)
  - offset: integer (default 0)
- Response:
  - sessions: Array<{ session_id: string, created_at: string }>
  - limit: number
  - offset: number

### 6) Chat detail (messages)

- Method: GET
- Path: /chats/{session_id}
- Response:
  - session_id: string
  - messages: Array<{ timestamp: string, role: "user"|"assistant", content: string }>

### 7) Delete chat

- Method: DELETE
- Path: /chats/{session_id}
- Response:
  - deleted: boolean
  - session_id: string

## PowerShell examples

Set base URL:

```powershell
$base = "http://localhost:8000"
```

- Health:

```powershell
Invoke-RestMethod -Uri "$base/health" -Method Get
```

- Meta:

```powershell
Invoke-RestMethod -Uri "$base/meta" -Method Get
```

- Query (RAG):

```powershell
$payload = @{ question = "What is hypertension?"; max_tokens = 150; temperature = 0.0 }
Invoke-RestMethod -Uri "$base/query" -Method Post -ContentType "application/json" -Body ($payload | ConvertTo-Json -Depth 5)
```

- Logs:

```powershell
Invoke-RestMethod -Uri "$base/logs?limit=10&offset=0" -Method Get
```

- Chats:

```powershell
$chats = Invoke-RestMethod -Uri "$base/chats?limit=10&offset=0" -Method Get
$chats
```

- Chat detail and delete:

```powershell
$sid = if ($chats.sessions) { $chats.sessions[0].session_id } else { $null }
if ($sid) { Invoke-RestMethod -Uri "$base/chats/$sid" -Method Get }
if ($sid) { Invoke-RestMethod -Uri "$base/chats/$sid" -Method Delete }
```

## Notes

- Interactive API docs are served at /docs (Swagger) and /redoc (ReDoc).
- Retrieval requires the vector index and chunks JSON paths configured in `config.py`.
- Generation requires model weights access (Hugging Face token is handled server-side).
- All responses are JSON. Times are in seconds.
