import sqlite3
import json
from datetime import datetime
from pathlib import Path
import config

DATABASE_PATH = Path(config.LOG_DATABASE_PATH)

def init_database():
    """Initialize the SQLite database with required tables."""
    DATABASE_PATH.parent.mkdir(exist_ok=True)
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Create table for API logs
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS api_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        retrieved_sources TEXT NOT NULL,
        retrieval_time REAL,
        generation_time REAL,
        total_time REAL
    )
    ''')
    
    # Create table for chat sessions
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chat_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    ''')
    
    # Create table for chat messages
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chat_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
    )
    ''')
    
    conn.commit()
    conn.close()

def log_api_interaction(question, answer, sources, retrieval_time, generation_time, total_time):
    """Log an API interaction to the database."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO api_logs (
        timestamp, question, answer, retrieved_sources,
        retrieval_time, generation_time, total_time
    ) VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        question,
        answer,
        json.dumps(sources),
        retrieval_time,
        generation_time,
        total_time
    ))
    
    conn.commit()
    conn.close()

def create_chat_session():
    """Create a new chat session and return its ID."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    cursor.execute(
        'INSERT INTO chat_sessions (session_id, created_at) VALUES (?, ?)',
        (session_id, datetime.now().isoformat())
    )
    
    conn.commit()
    conn.close()
    return session_id

def save_chat_message(session_id, role, content):
    """Save a chat message to the database."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO chat_messages (session_id, timestamp, role, content)
    VALUES (?, ?, ?, ?)
    ''', (session_id, datetime.now().isoformat(), role, content))
    
    conn.commit()
    conn.close()

def get_chat_history(session_id=None):
    """Retrieve chat history, optionally filtered by session_id."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    if session_id:
        cursor.execute('''
        SELECT timestamp, role, content FROM chat_messages
        WHERE session_id = ? ORDER BY timestamp
        ''', (session_id,))
    else:
        cursor.execute('''
        SELECT session_id, created_at FROM chat_sessions
        ORDER BY created_at DESC
        ''')
    
    results = cursor.fetchall()
    conn.close()
    return results

def get_session_messages(session_id):
    """Get all messages for a specific chat session."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT role, content FROM chat_messages
    WHERE session_id = ? ORDER BY timestamp
    ''', (session_id,))
    
    messages = cursor.fetchall()
    conn.close()
    return [[msg[1], None] if msg[0] == "user" else [None, msg[1]] for msg in messages]

def delete_chat_session(session_id: str):
    """Delete a chat session and all of its messages from the database."""
    if not session_id:
        return
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute('DELETE FROM chat_messages WHERE session_id = ?', (session_id,))
        cursor.execute('DELETE FROM chat_sessions WHERE session_id = ?', (session_id,))
        conn.commit()
    finally:
        conn.close()

def get_api_logs(limit: int = 100, offset: int = 0):
    """Fetch API logs from database with pagination."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute(
        '''SELECT id, timestamp, question, answer, retrieved_sources, retrieval_time, generation_time, total_time
           FROM api_logs ORDER BY id DESC LIMIT ? OFFSET ?''', (limit, offset)
    )
    rows = cursor.fetchall()
    conn.close()
    logs = []
    for r in rows:
        try:
            sources = json.loads(r[4]) if r[4] else []
        except Exception:
            sources = []
        logs.append({
            "id": r[0],
            "timestamp": r[1],
            "question": r[2],
            "answer": r[3],
            "retrieved_sources": sources,
            "retrieval_time": r[5],
            "generation_time": r[6],
            "total_time": r[7],
        })
    return logs

def list_chat_sessions(limit: int = 100, offset: int = 0):
    """List chat sessions with pagination."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute(
        '''SELECT session_id, created_at FROM chat_sessions ORDER BY created_at DESC LIMIT ? OFFSET ?''',
        (limit, offset)
    )
    rows = cursor.fetchall()
    conn.close()
    return [{"session_id": r[0], "created_at": r[1]} for r in rows]

def list_chat_messages_raw(session_id: str):
    """Return raw chat messages for a session: timestamp, role, content."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute(
        '''SELECT timestamp, role, content FROM chat_messages WHERE session_id = ? ORDER BY timestamp''',
        (session_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [{"timestamp": r[0], "role": r[1], "content": r[2]} for r in rows]

# Initialize the database when the module is imported
init_database()