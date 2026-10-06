"""FastAPI application entrypoint.

Creates the app, mounts the auth/documents/chat/health routers, and
initializes the Postgres schema (Base.metadata.create_all) on startup.
No logic implemented yet.
"""
