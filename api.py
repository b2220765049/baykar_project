import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from huggingface_hub import snapshot_download, login as hf_login
from src.chatbot.retriever import VectorSearch
import time
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import threading
from pydantic import BaseModel
import config
from src.utils.database import (
    log_api_interaction,
    get_api_logs,
    list_chat_sessions,
    list_chat_messages_raw,
    delete_chat_session,
    create_chat_session,
)
from dotenv import load_dotenv


# Minimal, clean API without runtime HF login endpoints
api_description = """
Medical RAG API: retrieves context and uses an LLM to generate grounded answers.
"""

app = FastAPI(title="Medical RAG API", description=api_description, version="1.2.0")

# Allow cross-origin requests from any origin (safe for demo/internal; tighten in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    max_tokens: int = 256
    temperature: float = 0.0


class QueryResponse(BaseModel):
    answer: str
    retrieved_sources: list[str]
    retrieval_time_seconds: float
    generation_time_seconds: float
    total_time_seconds: float


class ChatCreateResponse(BaseModel):
    session_id: str
    created_at: str


 


SYSTEM_PROMPT = """You are a highly skilled medical AI assistant. Your persona is that of an expert whose knowledge is derived from a curated set of medical documents. You must answer questions based *only* on this knowledge.

**--- YOUR CORE DIRECTIVES ---**

1.  **The Golden Rule of Grounding:** Your entire response MUST be based exclusively on the information presented in the "KNOWLEDGE BASE" section of the input. You will act as if this is your own internal knowledge.

2.  **The Citation Mandate (CRITICAL):**
    - **Input Format:** The KNOWLEDGE BASE will be provided as a series of text blocks. Each block will begin with a `Source:` line identifying the document, followed by the `Content:`.
    - **Core Rule:** For EVERY piece of information you use to construct your answer, you MUST cite the corresponding `Source:` filename.
    - **Citation Placement:** Citations must be placed directly at the end of the sentence or clause they support.
    - **Citation Format:** The format is strictly `[Source: filename.pdf]`.
    - **Synthesizing Sources:** If you combine information from multiple sources to form a single sentence, you MUST cite all of them. Example: "...is a common condition [Source: doc1.pdf, doc2.pdf]."

    **--- CITATION EXAMPLE ---**
    **KNOWLEDGE BASE:**
    ---
    Source: hypertension_basics_v2.pdf
    Content: The White Coat Phenomenon (WCP) is a temporary elevation of blood pressure that occurs when a patient is in a medical environment.
    ---
    Source: advanced_cardiology_review.pdf
    Content: WCP is often due to anxiety or stress caused by the medical setting.
    ---
    **USER QUERY:** What is the White Coat Phenomenon?
    **CORRECT RESPONSE:** The White Coat Phenomenon (WCP) is a temporary elevation of blood pressure that occurs when a patient is in a medical environment [Source: hypertension_basics_v2.pdf]. This is often due to anxiety or stress caused by the medical setting [Source: advanced_cardiology_review.pdf].
    **--- END EXAMPLE ---**

3.  **The "Silent Treatment" Rule:**
    - **NEVER** mention the words "context", "provided text", "knowledge base", or any similar phrases in your response to the user. Your answers must be direct.
    - **DO NOT** introduce any outside information, facts, or medical classifications not explicitly found in the text. This is a strict prohibition on hallucination.
    - If the knowledge base does not mention a specific detail the user asks about, you **MUST simply omit it** from your answer. **DO NOT state that the information is missing.** Only report what IS present.

4.  **Response Style: Directness and Synthesis:**
    - **Prioritize Directness:** Your primary goal is to answer the user's question directly and concisely using the cited information.
    - **Avoid Over-Elaboration:** Do not include extensive background details or tangential information if it is not essential for a direct answer.
    - **Synthesize When Necessary:** Combine information from multiple sources (and cite them all) *only if necessary* to form a complete and direct answer. If one source provides a sufficient answer, prioritize it.

5.  **The Rejection Rule:**
    - If, and ONLY if, the KNOWLEDGE BASE contains no relevant information to answer the question, you MUST respond with the exact phrase: "I do not have sufficient information to answer that question." and nothing else.
"""


print("Starting server and attempting to load models/retriever...")

# Load environment variables from a local .env file if present
load_dotenv()

# Ensure proxies don't interfere with Hugging Face auth inside containers
_no_proxy_val = "huggingface.co,.huggingface.co,cdn-lfs.huggingface.co,cdn-lfs-us.huggingface.co"
if not os.getenv("NO_PROXY"):
    os.environ["NO_PROXY"] = _no_proxy_val
if not os.getenv("no_proxy"):
    os.environ["no_proxy"] = os.environ["NO_PROXY"]

# Ensure we target the official endpoint (avoid mirrors/overrides)
if not os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://huggingface.co"

# Support reading token from env or a file (e.g., Docker secret)
def _resolve_hf_token() -> str | None:
    # Preferred env var
    token = os.getenv("HF_TOKEN")
    if token:
        token = token.strip().strip('"').strip("'")
        return token

    # Optional: file-based secret (prefer HF_TOKEN_FILE; fallback to legacy var if present)
    token_file = os.getenv("HF_TOKEN_FILE") or os.getenv("HUGGINGFACE_HUB_TOKEN_FILE")
    if token_file and os.path.exists(token_file):
        try:
            with open(token_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    return content
        except Exception:
            pass
    return None

# Ensure the token is available to huggingface_hub/transformers before model load
_hf_token = _resolve_hf_token()
if _hf_token:
    # Set only HF_TOKEN for downstream use
    os.environ["HF_TOKEN"] = _hf_token

# Perform a non-interactive HF login so credentials are cached for downstream calls
try:
    if _hf_token:
        hf_login(token=_hf_token, add_to_git_credential=False)
except Exception as _e:
    # Proceed without failing hard; from_pretrained may still work with explicit token
    pass

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = None
tokenizer = None
model_available = True
model_load_error = None

try:
    # If a local pre-downloaded model directory is provided, prefer it
    local_model_dir = os.getenv("HF_LOCAL_MODEL_DIR")
    if local_model_dir and os.path.exists(local_model_dir):
        model = AutoModelForCausalLM.from_pretrained(
            local_model_dir,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
    else:
        common_auth = {}
        if _hf_token:
            # Use explicit token param consistently (matches current hub/transformers guidance)
            common_auth = {"token": _hf_token}

        try:
            # Primary path: direct load via Transformers (should attach token)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                **common_auth,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name, **common_auth)
        except Exception as direct_err:
            # Fallback: explicitly download snapshot with hub client, then load from local dir
            print("Direct load failed; attempting snapshot_download fallback:", direct_err)
            local_dir = snapshot_download(
                repo_id=model_name,
                token=_hf_token,
                revision="main",
                local_dir=None,
                local_dir_use_symlinks=False,
            )
            model = AutoModelForCausalLM.from_pretrained(
                local_dir,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(local_dir)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print(f"Loaded model {model_name}")
except Exception as e:
    model_available = False
    model_load_error = str(e)
    print("Warning: could not load generation model:", model_load_error)


search_engine = None
search_engine_available = False
search_engine_error = None
try:
    search_engine = VectorSearch(model_name=config.EMBED_MODEL_NAME, index_path=config.FAISS_INDEX_PATH, chunks_path=config.CHUNKS_JSON_PATH)
    search_engine_available = True
    print("VectorSearch initialized")
except Exception as e:
    search_engine_available = False
    search_engine_error = str(e)
    print("Warning: could not initialize VectorSearch:", search_engine_error)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/meta")
def metadata():
    return {
        "version": app.version,
        "model": model_name,
        "model_available": model_available,
        "search_available": search_engine_available,
        "device": device,
    }


 


@app.post("/query", response_model=QueryResponse)
async def process_user_query(request: QueryRequest):
    if not model_available:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=(f"Generation model not available: {model_load_error}."))

    if not (search_engine_available if 'search_engine_available' in globals() else False):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=(f"Search engine not available: {search_engine_error}."))

    total_start = time.perf_counter()
    question = request.question
    print(f"Received question: {question}")

    ret_start = time.perf_counter()
    results = search_engine.search(question, n=5)
    ret_end = time.perf_counter()
    retrieval_time = ret_end - ret_start

    retrieved_context = ""
    sources = set()
    for r in results:
        retrieved_context += f"Source: {r['source']}\nContent: {r['content']}\n\n"
        sources.add(r['source'])

    user_content = f"KNOWLEDGE BASE:\n{retrieved_context}\n---\nUSER QUESTION:\n{question}"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}]

    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)

    gen_start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(input_ids=inputs, max_new_tokens=request.max_tokens, do_sample=False, temperature=request.temperature, pad_token_id=tokenizer.eos_token_id)
    gen_end = time.perf_counter()
    generation_time = gen_end - gen_start

    input_len = inputs.shape[1]
    generated_tokens = outputs[0][input_len:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    total_end = time.perf_counter()
    total_time = total_end - total_start

    try:
        log_api_interaction(question=question, answer=answer, sources=list(sources), retrieval_time=retrieval_time, generation_time=generation_time, total_time=total_time)
    except Exception:
        pass

    return QueryResponse(answer=answer, retrieved_sources=list(sources), retrieval_time_seconds=retrieval_time, generation_time_seconds=generation_time, total_time_seconds=total_time)


@app.post("/query/stream")
async def process_user_query_stream(request: QueryRequest):
    """Server-sent streaming endpoint. Emits incremental generated text as SSE.

    Event format per message:
      data: <text-chunk>\n\n
    Final marker:
      event: end\n
      data: [DONE]\n\n
    Notes:
      - Keeps /query unchanged to avoid breaking existing clients.
      - Logs the final assembled answer to the database upon completion.
    """
    if not model_available:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=(f"Generation model not available: {model_load_error}."))
    if not (search_engine_available if 'search_engine_available' in globals() else False):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=(f"Search engine not available: {search_engine_error}."))

    question = request.question

    # Retrieval
    ret_start = time.perf_counter()
    results = search_engine.search(question, n=5)
    ret_end = time.perf_counter()
    retrieval_time = ret_end - ret_start

    retrieved_context = ""
    sources = set()
    for r in results:
        retrieved_context += f"Source: {r['source']}\nContent: {r['content']}\n\n"
        sources.add(r['source'])

    user_content = f"KNOWLEDGE BASE:\n{retrieved_context}\n---\nUSER QUESTION:\n{question}"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}]

    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = {
        "input_ids": inputs,
        "max_new_tokens": request.max_tokens,
        "do_sample": False,
        "temperature": request.temperature,
        "pad_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
    }

    def token_generator():
        total_start = time.perf_counter()
        # Run generation in a background thread so we can iterate the streamer
        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        assembled = []
        try:
            for text in streamer:
                if not text:
                    continue
                assembled.append(text)
                # SSE event: use only 'data' lines for broad compatibility
                yield f"data: {text}\n\n"
        except Exception as _e:
            # Emit an error event and stop
            yield f"event: error\n\n"
        finally:
            thread.join()

        # Finalize timings and log once
        total_end = time.perf_counter()
        generation_time = total_end - ret_end
        final_answer = "".join(assembled)
        try:
            log_api_interaction(question=question, answer=final_answer, sources=list(sources), retrieval_time=retrieval_time, generation_time=generation_time, total_time=(total_end - total_start))
        except Exception:
            pass
        # Send a final marker so clients can close
        yield "event: end\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")


@app.get("/logs")
def api_logs(limit: int = 100, offset: int = 0):
    try:
        logs = get_api_logs(limit=limit, offset=offset)
        return {"logs": logs, "limit": limit, "offset": offset}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chats")
def chats(limit: int = 100, offset: int = 0):
    try:
        sessions = list_chat_sessions(limit=limit, offset=offset)
        return {"sessions": sessions, "limit": limit, "offset": offset}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chats", response_model=ChatCreateResponse, status_code=201)
def chat_create():
    """Create a new chat session and return its identifier."""
    try:
        session_id = create_chat_session()
        # created_at is encoded inside DB when creating; return an ISO timestamp here as well
        # For simplicity, reuse the current time for created_at in response
        from datetime import datetime
        return {"session_id": session_id, "created_at": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chats/{session_id}")
def chat_detail(session_id: str):
    try:
        messages = list_chat_messages_raw(session_id)
        return {"session_id": session_id, "messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/chats/{session_id}")
def chat_delete(session_id: str):
    try:
        delete_chat_session(session_id)
        return {"deleted": True, "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)