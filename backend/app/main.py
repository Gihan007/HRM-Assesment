"""FastAPI application entrypoint.

Mounts the health and collections routers. Auth/session/Postgres wiring
is deferred until those pieces are implemented; nothing here depends on
them yet.
"""

from __future__ import annotations

from fastapi import FastAPI

from app.api import collections, health

app = FastAPI(title="RAG Generator")

app.include_router(health.router)
app.include_router(collections.router)
