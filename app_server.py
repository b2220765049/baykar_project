"""
app_server.py

This module exposes the FastAPI `app` from `api.py` so that
`uvicorn app_server:app` starts the single application that contains
all routes (query, auth, status, etc.).
"""

from api import app