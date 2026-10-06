"""Streamlit frontend: create/select/delete collections, upload documents,
and chat with grounded, cited answers. Talks only to the backend REST API
(BACKEND_BASE_URL env var) via api_client.py.
"""

from __future__ import annotations

import streamlit as st

import api_client as api

st.set_page_config(page_title="RAG Generator", page_icon="📚", layout="wide")

if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}  # collection name -> list[{question, answer, sources}]
if "selected_collection" not in st.session_state:
    st.session_state.selected_collection = None


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📎 {len(sources)} source(s)"):
        for source in sources:
            location = source["filename"]
            if source.get("page") is not None:
                location += f", page {source['page']}"
            st.markdown(f"**[{source['index']}] {location}**")
            st.caption(source["snippet"])


def render_turn(turn: dict) -> None:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        render_sources(turn["sources"])


# --- Sidebar: collections ---------------------------------------------------

with st.sidebar:
    st.header("Collections")

    if not api.check_health():
        st.error(f"Backend not reachable at {api.get_backend_url()}")
        st.stop()

    with st.form("create_collection_form", clear_on_submit=True):
        new_name = st.text_input("New collection name")
        if st.form_submit_button("Create") and new_name.strip():
            try:
                with st.spinner(f"Creating '{new_name}'..."):
                    created = api.create_collection(new_name)
                st.session_state.selected_collection = created["name"]
                st.success(f"Created '{created['name']}'")
            except api.BackendError as exc:
                st.error(str(exc))

    try:
        with st.spinner("Loading collections..."):
            collections = api.list_collections()
    except api.BackendError as exc:
        st.error(str(exc))
        collections = []

    collection_names = [c["name"] for c in collections]

    if collection_names:
        counts = {c["name"]: c["document_count"] for c in collections}
        labels = [f"{name} ({counts[name]} docs)" for name in collection_names]
        if st.session_state.selected_collection in collection_names:
            default_index = collection_names.index(st.session_state.selected_collection)
        else:
            # Default to the seeded sample-papers collection (see
            # scripts/seed_samples.py) when nothing's been picked yet.
            preferred = next((n for n in collection_names if n.lower() == "ai_papers"), None)
            default_index = collection_names.index(preferred) if preferred else 0
        chosen_label = st.radio("Select a collection", labels, index=default_index)
        selected = collection_names[labels.index(chosen_label)]
        st.session_state.selected_collection = selected

        if st.button("🗑️ Delete selected collection"):
            try:
                with st.spinner(f"Deleting '{selected}'..."):
                    api.delete_collection(selected)
                st.session_state.chat_histories.pop(selected, None)
                st.session_state.selected_collection = None
                st.toast(f"Deleted '{selected}'", icon="🗑️")
                st.rerun()
            except api.BackendError as exc:
                st.error(str(exc))
    else:
        st.info("No collections yet. Create one above.")
        st.session_state.selected_collection = None

    selected = st.session_state.selected_collection

    if selected:
        st.divider()
        st.subheader("Upload documents")
        uploaded_files = st.file_uploader(
            "PDF, DOCX, TXT, or MD",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            key=f"uploader_{selected}",
        )
        if uploaded_files and st.button("Upload"):
            files_payload = [
                (f.name, f.getvalue(), f.type or "application/octet-stream") for f in uploaded_files
            ]
            try:
                with st.spinner(f"Uploading {len(files_payload)} file(s)..."):
                    result = api.upload_documents(selected, files_payload)
                for item in result["results"]:
                    if not item["success"]:
                        st.error(f"{item['filename']}: {item['error']}")
                    elif item["skipped"]:
                        st.info(f"{item['filename']}: already ingested, skipped.")
                    else:
                        st.success(f"{item['filename']}: {item['chunk_count']} chunks ingested.")
            except api.BackendError as exc:
                st.error(str(exc))

        st.divider()
        st.subheader("Documents")
        try:
            with st.spinner("Loading documents..."):
                documents = api.list_documents(selected)
            if documents:
                for doc in documents:
                    st.caption(f"📄 {doc['filename']} — {doc['chunk_count']} chunks")
            else:
                st.caption("No documents ingested yet.")
        except api.BackendError as exc:
            st.error(str(exc))


# --- Main area: chat ---------------------------------------------------------

st.title("RAG Generator")

selected = st.session_state.selected_collection

if not selected:
    st.info("Select or create a collection in the sidebar to start chatting.")
    st.stop()

st.caption(f"Chatting with collection: **{selected}**")

history = st.session_state.chat_histories.setdefault(selected, [])

for turn in history:
    render_turn(turn)

question = st.chat_input("Ask a question about this collection's documents...")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = api.query_collection(selected, question)
            except api.BackendError as exc:
                st.error(str(exc))
                response = None
        if response is not None:
            st.write(response["answer"])
            render_sources(response["sources"])
            history.append(
                {"question": question, "answer": response["answer"], "sources": response["sources"]}
            )
