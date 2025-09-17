import config
import json
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
    delete_chat_session,
)

# --- Configuration ---
API_URL = config.API_URL
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

# --- Core API Call Function (Unchanged) ---
def call_rag_api(message: str, history: list) -> str:
    payload = {"question": message, "max_tokens": 1024, "temperature": 0.0}
    try:
        response = requests.post(API_URL, json=payload, timeout=120)
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

# --- Gradio UI Definition using gr.Blocks ---
with gr.Blocks(
    theme=gr.themes.Soft(),
    css="footer {display: none !important}"
) as demo:
    current_chat_file_state = gr.State(None)
    # Track last session id seen by chat_fn to hard-reset history when switching
    last_session_state = gr.State(None)

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
            # Reuse explicit Chatbot and Textbox so left-panel controls can still
            # reference them (e.g., loading history, starting new chats).
            chatbot = gr.Chatbot(height=600, label="Chat", type="messages", elem_id="chatbot-box")
            msg_textbox = gr.Textbox(show_label=False, placeholder="Type your question here...", container=False)
            # Hidden event trigger to auto-refresh chat list when messages are saved/deleted
            chat_list_refresher = gr.State(0)

            # Hardcoded Accuracy=5 example questions from your CSV
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

            # Maintain explicit messages state to avoid bleed across sessions
            messages_state = gr.State([])

            def send_message(message, messages, current_session_id, last_session_id):
                # Normalize message text
                if isinstance(message, dict):
                    message_text = message.get("content", "")
                else:
                    message_text = (message or "").strip()
                if not message_text:
                    # No-op; return existing state
                    return messages or [], "", current_session_id, chat_list_refresher.value if hasattr(chat_list_refresher, 'value') else 0, last_session_id or current_session_id, messages or []

                # Reset messages if session switched or not set
                if not current_session_id or (last_session_id and last_session_id != current_session_id):
                    messages = []

                # Convert messages (list of {role, content}) to pairs for backend
                pairs_history = []
                for m in messages or []:
                    role = m.get("role")
                    content = m.get("content", "")
                    if role == "user":
                        pairs_history.append([content, None])
                    elif role == "assistant":
                        if pairs_history and pairs_history[-1][1] is None:
                            pairs_history[-1][1] = content
                        else:
                            pairs_history.append([None, content])

                # Call backend
                bot_message = call_rag_api(message_text, pairs_history)

                # Ensure session id
                if not current_session_id:
                    current_session_id = create_chat_session()

                # Persist last exchange to DB
                new_pairs = pairs_history + [[message_text, bot_message]]
                save_chat_messages(new_pairs, current_session_id)

                # Update messages state for UI (messages format)
                new_messages = (messages or []) + [
                    {"role": "user", "content": message_text},
                    {"role": "assistant", "content": bot_message},
                ]

                return new_messages, "", current_session_id, time.time(), current_session_id, new_messages

            # Wire sending on Enter
            msg_textbox.submit(
                fn=send_message,
                inputs=[msg_textbox, messages_state, current_chat_file_state, last_session_state],
                outputs=[chatbot, msg_textbox, current_chat_file_state, chat_list_refresher, last_session_state, messages_state],
            )

            # Clickable examples: populate the textbox for quick sending
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

    def load_chat_action(index):
        chat_files = get_saved_chats()
        if index < len(chat_files):
            filename = chat_files[index]
            history_pairs = load_chat_history(filename) or []
            # Convert stored pair history [[user, assistant], ...] into
            # messages format expected by Chatbot(type="messages").
            messages = []
            for u, a in history_pairs:
                if u:
                    messages.append({"role": "user", "content": u})
                if a:
                    messages.append({"role": "assistant", "content": a})
            return messages, filename, filename, messages
        return [], None, None, []

    def delete_chat_action(index):
        chat_files = get_saved_chats()
        if index < len(chat_files):
            filename = chat_files[index]
            # Actually delete from DB
            delete_chat_session(filename)
        # After deleting, return updates to refresh the list and clear the chat
        return update_chat_list_display() + [[], None, None, []]

    # --- BUILD PHASE: Wire up events for ALL pre-built components ---
    
    all_chat_list_components = [component for row_tuple in chat_rows for component in row_tuple]

    for i, (row, load_btn, delete_btn) in enumerate(chat_rows):
        # Use partial to "bake" the index 'i' into the function call
        load_btn.click(
            fn=partial(load_chat_action, i),
            inputs=None,
            outputs=[chatbot, current_chat_file_state, last_session_state, messages_state]
        )
        
        delete_btn.click(
            fn=partial(delete_chat_action, i),
            inputs=None,
            outputs=all_chat_list_components + [chatbot, current_chat_file_state, last_session_state, messages_state]
        )

    # Auto-refresh: whenever chat_list_refresher changes (after send/delete), update list
    chat_list_refresher.change(fn=update_chat_list_display, inputs=None, outputs=all_chat_list_components)

    def start_new_chat():
        # Clear chatbot, reset both session states, clear input box, and clear in-memory messages
        return [], None, None, "", []

    new_chat_btn.click(start_new_chat, None, [chatbot, current_chat_file_state, last_session_state, msg_textbox, messages_state])
    
    # When the app loads, populate the list for the first time
    demo.load(fn=update_chat_list_display, inputs=None, outputs=all_chat_list_components)

if __name__ == "__main__":
    # Allow overriding the server host and port via environment variables so
    # the UI can be hosted inside Docker and be reachable from the host machine.
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", os.getenv("PORT", 7860)))

    print(f"Starting Gradio UI on {server_name}:{server_port}... Open the URL in your browser.")
    demo.launch(server_name=server_name, server_port=server_port, allowed_paths=[PDF_DIRECTORY])