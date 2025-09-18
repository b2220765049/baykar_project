import config
import gradio as gr
import requests
import os
import time
from functools import partial
from src.utils.database import (
    get_chat_history,
    create_chat_session,
    save_chat_message,
    get_session_messages,
    delete_chat_session
)

# --- Configuration ---
# Use configured API_URL as a hint, but auto-discover a reachable one at runtime.
API_URL_HINT = os.getenv("API_URL", config.API_URL)
PDF_DIRECTORY = config.PDF_DIRECTORY
MAX_CHAT_ROWS = config.MAX_CHAT_ROWS

# --- Helper Functions (SQLite Version) ---
def get_saved_chats():
    """Get list of saved chat sessions from SQLite."""
    sessions = get_chat_history()
    return [session[0] for session in sessions]  # Return session IDs

def load_chat_history(session_id):
    """Load chat history from SQLite for a specific session."""
    if session_id:
        return get_session_messages(session_id)
    return []

def save_chat_messages(messages, session_id):
    """Save chat messages to SQLite database."""
    if not messages or not session_id: return
    
    # Only save the last message (most recent addition)
    last_message = messages[-1]
    if last_message[0]:  # User message
        save_chat_message(session_id, "user", last_message[0])
    if last_message[1]:  # Bot message
        save_chat_message(session_id, "assistant", last_message[1])

def delete_chat_history(session_id):
    """Delete chat session from SQLite database."""
    if session_id:
        print(f"Deleted chat session: {session_id}")

# --- API helpers ---
_API_URL_CACHE = None

def _candidates() -> list[str]:
    # Treat candidates as full /query endpoints
    hints = []
    # 1) Env/config hint
    if API_URL_HINT:
        hints.append(API_URL_HINT)
    # 2) Common container DNS names
    hints += [
        "http://medical-rag-api:8000/query",
        "http://app:8000/query",
        "http://host.docker.internal:8000/query",
        "http://127.0.0.1:8000/query",
        "http://localhost:8000/query",
    ]
    # De-dup while preserving order
    seen, uniq = set(), []
    for h in hints:
        if h and h not in seen:
            seen.add(h)
            uniq.append(h)
    return uniq

def _api_base(api_url: str) -> str:
    # If endpoint ends with /query, return the base
    return api_url[: -len("/query")] if api_url.endswith("/query") else api_url

def _resolve_api_url() -> str:
    global _API_URL_CACHE
    if _API_URL_CACHE:
        return _API_URL_CACHE
    for cand in _candidates():
        base = _api_base(cand).rstrip("/")
        try:
            r = requests.get(f"{base}/health", timeout=2)
            if r.ok and isinstance(r.json(), dict) and r.json().get("ok") is True:
                _API_URL_CACHE = cand
                print(f"[UI] Using API URL: {_API_URL_CACHE}")
                return _API_URL_CACHE
        except Exception:
            continue
    # Fallback to hint even if not reachable; requests will surface errors later
    _API_URL_CACHE = API_URL_HINT
    print(f"[UI] Falling back to API URL hint: {_API_URL_CACHE}")
    return _API_URL_CACHE

# --- Core API Call Function (Unchanged for full reply) ---
def call_rag_api(message: str, history: list) -> str:
    payload = {"question": message, "max_tokens": 1024, "temperature": 0.0}
    try:
        api_url = _resolve_api_url()
        response = requests.post(api_url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        answer = data.get("answer", "Sorry, I received an invalid response from the backend.")
        sources = data.get("retrieved_sources", [])
        if sources:
            sources_list = "\n".join(f"- [{s}](/file={os.path.join(PDF_DIRECTORY, s)})" for s in sources)
            formatted_response = f"{answer}\n\n---\n**Sources:**\n{sources_list}"
        else:
            formatted_response = answer
        return formatted_response
    except requests.exceptions.RequestException as e:
        return f"Sorry, I couldn't connect to the backend service: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def stream_rag_api(message: str):
    """Consume the SSE streaming endpoint and yield partial text as it arrives.

    Falls back to non-streaming if the stream endpoint is unavailable.
    """
    api_url = _resolve_api_url()
    stream_url = api_url.rstrip("/") + "/stream"
    payload = {"question": message, "max_tokens": 1024, "temperature": 0.0}
    try:
        with requests.post(stream_url, json=payload, stream=True, timeout=300) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith(":"):
                    # Comment line in SSE, ignore
                    continue
                if line.startswith("event:"):
                    # We only care about 'end' to finish
                    if line.strip() == "event: end":
                        break
                    continue
                if line.startswith("data:"):
                    # Preserve leading/trailing spaces; only remove the optional single space after colon
                    data = line[len("data:"):]
                    if data.startswith(" "):
                        data = data[1:]
                    if data == "[DONE]":
                        break
                    yield data
    except requests.exceptions.RequestException:
        # Fallback: yield full answer once via non-streaming API
        yield call_rag_api(message, [])

# --- Gradio UI Definition using gr.Blocks ---
with gr.Blocks(theme=gr.themes.Soft(), css="footer {display: none !important}") as demo:
    current_chat_file_state = gr.State(None)

    # --- UI Layout ---
    gr.Markdown("# 🩺 Medical RAG Chatbot - Hypertension")
    gr.Markdown("Ask questions about hypertension. Chats are saved automatically.")

    with gr.Row():
        with gr.Column(scale=1, min_width=250):
            gr.Markdown("### Chat Controls")
            new_chat_btn = gr.Button("➕ New Chat")
            
            # --- BUILD PHASE: Create a fixed number of hidden rows ---
            chat_rows = []
            for _ in range(MAX_CHAT_ROWS):
                with gr.Row(visible=False, variant="panel") as row:
                    load_btn = gr.Button(scale=6)
                    delete_btn = gr.Button(value="🗑️", scale=1, min_width=40)
                    chat_rows.append((row, load_btn, delete_btn))

        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=600, label="Chat", type="messages")
            with gr.Row():
                msg_textbox = gr.Textbox(show_label=False, placeholder="Type your question here...", container=False, scale=8)
                send_btn = gr.Button("Send", scale=1)
            # Example questions for quick testing (click to populate the textbox)
            examples_list = [
                "What are the five most predictive variables for White Coat Effect (WCE) found by a machine learning algorithm study?",
                "Is White Coat Uncontrolled Hypertension (WUCH) associated with any increased cardiovascular risk?",
                "How did the frequency of LVDD differ between men and women in the study?",
                "What is barnidipine?",
                "How did the combination of barnidipine plus losartan affect left ventricular mass index compared to lercanidipine plus losartan in hypertensive patients with diabetes?",
                "What is masked hypertension (MHT)?",
                "What was the prevalence of hypertension in Bangladesh when using the 2017 ACC/AHA guideline?",
                "Which administrative divisions in Bangladesh had higher prevalence ratios for hypertension according to both JNC 7 and ACC/AHA guidelines?",
                "What is the P300 event-related potential presumed to reflect?",
                "What BP threshold did the 2017 ACC/AHA guidelines set for diagnosing hypertension?",
                "In Dahl salt-sensitive rats, what was the effect of switching from a high-salt to a low-salt diet on the vasoconstriction of mesenteric small arteries (MSAs) to norepinephrine?",
                "How did ACE inhibitor treatment affect vasodilatation in the MSAs of Dahl-SS rats?",
            ]
            gr.Examples(examples_list, inputs=[msg_textbox], label="Examples")

    # --- RUN PHASE: Functions that UPDATE the pre-built UI ---
    
    def update_chat_list_display():
        """
        This function is called during the "Run" phase. It does NOT create components.
        It returns a list of gr.update objects to change the visibility and content
        of the pre-built components.
        """
        chat_files = get_saved_chats()
        updates = []
        for i in range(MAX_CHAT_ROWS):
            if i < len(chat_files):
                filename = chat_files[i]
                # Make the row visible and set the button's text
                updates.extend([
                    gr.update(visible=True),
                    gr.update(value=filename.replace("chat_", "").replace(".json", "").replace("_", " ")),
                    gr.update(visible=True)
                ])
            else:
                # Hide the unused rows
                updates.extend([gr.update(visible=False), gr.update(value=""), gr.update(visible=False)])
        return updates

    def pairs_to_messages(pairs):
        messages = []
        for u, a in pairs or []:
            if u:
                messages.append({"role": "user", "content": u})
            if a:
                messages.append({"role": "assistant", "content": a})
        return messages

    def messages_to_pairs(messages):
        pairs = []
        for m in messages or []:
            role = m.get("role")
            content = m.get("content", "")
            if role == "user":
                pairs.append([content, None])
            elif role == "assistant":
                if pairs and pairs[-1][1] is None:
                    pairs[-1][1] = content
                else:
                    pairs.append([None, content])
        return pairs

    def load_chat_action(index):
        chat_files = get_saved_chats()
        if index < len(chat_files):
            filename = chat_files[index]
            history_pairs = load_chat_history(filename)
            return pairs_to_messages(history_pairs), filename
        return [], None

    def delete_chat_action(index):
        chat_files = get_saved_chats()
        if index < len(chat_files):
            filename = chat_files[index]
            # Try deleting via API first
            try:
                base = _api_base(_resolve_api_url())
                requests.delete(f"{base}/chats/{filename}", timeout=15)
            except Exception:
                pass
            # Also delete locally to keep UI state in sync (in case DBs differ)
            try:
                delete_chat_session(filename)
            except Exception:
                pass
        # After deleting, return updates to refresh the list and clear the chat
        return update_chat_list_display() + [[], None]

    # --- BUILD PHASE: Wire up events for ALL pre-built components ---
    
    all_chat_list_components = [component for row_tuple in chat_rows for component in row_tuple]

    for i, (row, load_btn, delete_btn) in enumerate(chat_rows):
        # Use partial to "bake" the index 'i' into the function call
        load_btn.click(
            fn=partial(load_chat_action, i),
            inputs=None,
            outputs=[chatbot, current_chat_file_state]
        )
        
        delete_btn.click(
            fn=partial(delete_chat_action, i),
            inputs=None,
            outputs=all_chat_list_components + [chatbot, current_chat_file_state]
        )

    # --- Event wiring for main components ---
    def user_message_handler(user_message, history):
        # Append only the user message in messages format; assistant will be streamed
        return "", (history or []) + [{"role": "user", "content": user_message}]

    def bot_response_handler(history, current_session_id):
        """True real-time streaming using the API's SSE endpoint."""
        # Find the last user message content
        user_message = None
        for m in reversed(history or []):
            if m.get("role") == "user":
                user_message = m.get("content", "")
                break
        if user_message is None:
            # Nothing to do
            yield history or [], current_session_id
            return

        # Ensure there's an assistant message to fill
        history = history or []
        history.append({"role": "assistant", "content": ""})

        collected = []
        for chunk in stream_rag_api(user_message):
            collected.append(chunk)
            history[-1]["content"] = "".join(collected)
            yield history, current_session_id

        # Persist only once after streaming completes
        if current_session_id is None:
            current_session_id = create_chat_session()
        save_chat_messages(messages_to_pairs(history), current_session_id)
        yield history, current_session_id

    msg_submit_event = msg_textbox.submit(user_message_handler, [msg_textbox, chatbot], [msg_textbox, chatbot]).then(
        bot_response_handler, [chatbot, current_chat_file_state], [chatbot, current_chat_file_state]
    )
    send_click_event = send_btn.click(user_message_handler, [msg_textbox, chatbot], [msg_textbox, chatbot]).then(
        bot_response_handler, [chatbot, current_chat_file_state], [chatbot, current_chat_file_state]
    )
    
    # After sending a message, refresh the chat list display
    msg_submit_event.then(fn=update_chat_list_display, inputs=None, outputs=all_chat_list_components)
    send_click_event.then(fn=update_chat_list_display, inputs=None, outputs=all_chat_list_components)

    def start_new_chat():
        return [], None

    new_chat_btn.click(start_new_chat, None, [chatbot, current_chat_file_state])
    
    # When the app loads, populate the list for the first time
    demo.load(fn=update_chat_list_display, inputs=None, outputs=all_chat_list_components)

if __name__ == "__main__":
    # Allow overriding the server host and port via environment variables so
    # the UI can be hosted inside Docker and be reachable from the host machine.
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", os.getenv("PORT", 7860)))

    print(f"Starting Gradio UI on {server_name}:{server_port}... Open the URL in your browser.")
    # Enable queuing so generator-based streaming works smoothly
    demo.queue()
    demo.launch(server_name=server_name, server_port=server_port, allowed_paths=[PDF_DIRECTORY])