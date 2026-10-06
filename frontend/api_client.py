"""Thin HTTP client for the FastAPI backend, used by the Streamlit UI.

Backend URL comes from the BACKEND_BASE_URL env var (falls back to
http://localhost:8000 for local, non-docker runs).
"""

from __future__ import annotations

import os

import httpx

DEFAULT_BACKEND_URL = "http://localhost:8000"


class BackendError(Exception):
    """Raised when the backend can't be reached or returns an error."""


def get_backend_url() -> str:
    return os.environ.get("BACKEND_BASE_URL", DEFAULT_BACKEND_URL)


def _request(method: str, path: str, **kwargs):
    url = f"{get_backend_url()}{path}"
    try:
        response = httpx.request(method, url, timeout=60.0, **kwargs)
    except httpx.ConnectError as exc:
        raise BackendError(
            f"Could not reach the backend at {get_backend_url()}. Is it running?"
        ) from exc
    except httpx.TimeoutException as exc:
        raise BackendError("The backend took too long to respond.") from exc

    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise BackendError(str(detail))

    if response.status_code == 204 or not response.content:
        return {}
    return response.json()


def check_health() -> bool:
    try:
        _request("GET", "/health")
        return True
    except BackendError:
        return False


def list_collections() -> list[dict]:
    return _request("GET", "/collections")


def create_collection(name: str) -> dict:
    return _request("POST", "/collections", json={"name": name})


def delete_collection(name: str) -> None:
    _request("DELETE", f"/collections/{name}")


def list_documents(collection: str) -> list[dict]:
    return _request("GET", f"/collections/{collection}/documents")


def upload_documents(collection: str, files: list[tuple[str, bytes, str]]) -> dict:
    """`files` is a list of (filename, content_bytes, content_type)."""
    multipart = [("files", (name, content, content_type)) for name, content, content_type in files]
    return _request("POST", f"/collections/{collection}/documents", files=multipart)


def query_collection(collection: str, question: str) -> dict:
    return _request("POST", f"/collections/{collection}/query", json={"question": question})
