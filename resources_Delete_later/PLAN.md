# RAG Generator — Implementation Plan

## 0. Scope recap

Assignment requires (from candidate brief, Option A):
- Accept documents at runtime (no code changes to add a new doc set)
- RAG over those documents
- Grounded, cited answers
- Works with different document sets without code changes

Decisions locked in from clarification round:
- **LLM**: Groq (open-source hosted models, e.g. Llama 3.x / Mixtral) as primary, OpenRouter as automatic fallback on error/timeout/rate-limit.
- **Embeddings**: open-source, local (`sentence-transformers`, e.g. `BAAI/bge-small-en-v1.5`) — runs inside the backend container, no external embedding API/cost.
- **Document scoping**: one global Weaviate collection, isolation enforced purely by metadata filtering (`user_id` + `document_set_id`) on every query — not separate Weaviate classes per user.
- **Auth**: JWT per user; every request is scoped to the authenticated `user_id`; users can never see another user's documents or chat history.
- **Session memory**: short-term (in-session conversational buffer) + long-term (cross-session rolling summary per user), both per-user.
- **Services**: 2 containers — `frontend` (Streamlit) and `backend` (FastAPI + LangGraph), plus infra containers `weaviate` and `postgres`.
- **File storage**: raw uploaded files on a local Docker volume, path tracked in Postgres. Demo docs will be dropped in `resources_Delete_later/` for manual testing.

---

## 1. Architecture (text diagram)

```
                         ┌─────────────────────────┐
                         │        frontend         │
                         │   (Streamlit, :8501)    │
                         │  login / upload / chat  │
                         └───────────┬─────────────┘
                                     │ REST (JSON, multipart) + Bearer JWT
                                     ▼
                         ┌─────────────────────────┐
                         │         backend          │
                         │   (FastAPI, :8000)       │
                         │                          │
                         │  ┌────────────────────┐  │
                         │  │   Auth routes       │  │──► Postgres (users, refresh tokens)
                         │  ├────────────────────┤  │
                         │  │  Documents routes   │  │──► local disk volume (raw files)
                         │  │  (ingestion pipeline│  │──► Weaviate (chunks+vectors)
                         │  │   loader→chunk→embed│  │──► Postgres (document metadata)
                         │  │   →upsert)          │  │
                         │  ├────────────────────┤  │
                         │  │  Chat routes        │  │
                         │  │  (LangGraph         │  │
                         │  │   supervisor graph) │  │
                         │  │   ┌───────────────┐ │  │
                         │  │   │ supervisor    │ │  │
                         │  │   │ node (router) │ │  │
                         │  │   └──────┬────────┘ │  │
                         │  │          ▼           │  │
                         │  │  ┌───────────────┐   │  │──► Weaviate (similarity search,
                         │  │  │ retrieve_tool │───┼──┼──►   filtered by user_id +
                         │  │  └──────┬────────┘   │  │      document_set_id)
                         │  │          ▼           │  │
                         │  │  ┌───────────────┐   │  │──► Groq (primary LLM)
                         │  │  │ generate node │───┼──┼──►  OpenRouter (fallback LLM)
                         │  │  │ (grounding +  │   │  │
                         │  │  │  citations)   │   │  │
                         │  │  └───────────────┘   │  │
                         │  │          ▼           │  │
                         │  │  short/long-term     │  │──► Postgres (checkpoints, summaries)
                         │  │  memory read/write    │  │
                         │  └────────────────────┘  │
                         └─────────────────────────┘
                                     │
                     ┌───────────────┴───────────────┐
                     ▼                                ▼
            ┌─────────────────┐             ┌──────────────────┐
            │     Weaviate      │             │    Postgres       │
            │  class: Chunk     │             │ users, documents, │
            │  (vectors + meta) │             │ chat_sessions,     │
            │                   │             │ chat_messages,     │
            │                   │             │ long_term_memory,  │
            │                   │             │ langgraph_checkpts │
            └─────────────────┘             └──────────────────┘
```

All 4 containers run via a single `docker-compose.yml`. Backend talks to Weaviate and Postgres over the compose network; frontend only ever talks to the backend's REST API (never touches Weaviate/Postgres directly) — this is the microservice boundary.

---

## 2. Folder structure

```
HRM-Assesment/
├── PLAN.md
├── docker-compose.yml
├── .env.example
├── resources_Delete_later/          # demo docs for manual testing (gitignored or kept, TBD)
│
├── backend/
│   ├── Dockerfile
│   ├── pyproject.toml / requirements.txt
│   ├── alembic/                     # DB migrations
│   ├── app/
│   │   ├── main.py                  # FastAPI app factory, router mounting
│   │   ├── core/
│   │   │   ├── config.py            # pydantic Settings, reads .env
│   │   │   ├── security.py          # JWT encode/decode, password hashing
│   │   │   └── deps.py              # get_current_user, get_db, etc.
│   │   ├── api/
│   │   │   ├── auth.py              # /auth/register /auth/login /auth/refresh
│   │   │   ├── documents.py         # /documents/upload /documents/sets /documents/{id}
│   │   │   ├── chat.py              # /chat/{session_id}/message, /chat/sessions
│   │   │   └── health.py
│   │   ├── db/
│   │   │   ├── models.py            # SQLAlchemy: User, Document, ChatSession,
│   │   │   │                        #   ChatMessage, LongTermMemory
│   │   │   └── session.py           # engine/session factory
│   │   ├── ingestion/
│   │   │   ├── loaders.py           # pdf/docx/txt/md -> list[PageText]
│   │   │   ├── chunker.py           # text -> list[Chunk] (text, offsets, page)
│   │   │   ├── embedder.py          # sentence-transformers wrapper
│   │   │   └── pipeline.py          # orchestrates load->chunk->embed->upsert
│   │   ├── vectorstore/
│   │   │   ├── weaviate_client.py   # connection, schema bootstrap
│   │   │   └── queries.py           # similarity_search(user_id, set_id, query, k)
│   │   ├── agent/
│   │   │   ├── graph.py             # LangGraph StateGraph: supervisor + nodes
│   │   │   ├── state.py             # TypedDict/pydantic graph state
│   │   │   ├── tools.py             # retrieve_documents tool (LangChain @tool)
│   │   │   ├── prompts.py           # system/grounding prompt templates
│   │   │   └── memory.py            # short-term checkpointer + long-term summarizer
│   │   ├── llm/
│   │   │   ├── groq_client.py
│   │   │   ├── openrouter_client.py
│   │   │   └── router.py            # try Groq, fall back to OpenRouter
│   │   └── schemas/                 # pydantic request/response models
│   └── tests/
│       ├── unit/                    # chunker, loaders, citation mapping, jwt
│       └── integration/             # ingestion + query against test Weaviate/Postgres
│
└── frontend/
    ├── Dockerfile
    ├── requirements.txt
    └── streamlit_app.py             # login, upload, document-set picker, chat UI w/ citations
```

---

## 3. Data flow

### 3.1 Ingestion

1. User logs in via frontend → `POST /auth/login` → backend returns access + refresh JWT. Frontend keeps these in `st.session_state`.
2. User picks or types a **document set name** (e.g. `"contracts-2024"`) and uploads one or more files via `POST /documents/upload` (multipart, `Authorization: Bearer <jwt>`, `document_set_id` form field).
3. Backend validates extension + size, saves the raw file to disk at `./data/{user_id}/{document_set_id}/{uuid}_{filename}`, and inserts a `documents` row (status=`processing`).
4. **Ingestion pipeline** (runs as a FastAPI `BackgroundTask` so upload returns fast):
   a. `loaders.py` dispatches by extension → extracts raw text, preserving page numbers (PDF) / headings (MD) where possible.
   b. `chunker.py` splits into overlapping chunks (strategy in §4), each chunk keeps `{document_id, filename, page, chunk_index, char_start, char_end, text}`.
   c. `embedder.py` encodes each chunk with the local embedding model.
   d. Chunks + vectors are upserted into the single Weaviate class `DocumentChunk`, tagged with `user_id` and `document_set_id`.
   e. `documents` row updated to status=`ready` (or `failed` with an error message).
5. Frontend polls `GET /documents/sets` to show ingestion status and list existing document sets for that user.

### 3.2 Querying

1. User opens/continues a chat session scoped to one `document_set_id`. `POST /chat/sessions` creates a `chat_sessions` row (`user_id`, `document_set_id`); `session_id` returned.
2. User sends a message → `POST /chat/{session_id}/message`.
3. Backend loads:
   - **short-term memory**: the LangGraph checkpoint for this `session_id` (recent turns), via a Postgres-backed checkpointer.
   - **long-term memory**: this user's rolling summary row from `long_term_memory` (facts/preferences/topics from past sessions).
4. LangGraph **supervisor graph** runs (see §3.3) → produces `answer_text`, `citations[]`.
5. Backend appends the turn to `chat_messages` + checkpoint, and returns `{answer, citations}` to the frontend, which renders the answer with numbered citation badges and an expandable "sources" panel (filename, page, snippet, score).
6. Every N turns (or on session close), a background task asks the LLM to update the long-term summary for that user from the session transcript.

### 3.3 LangGraph supervisor graph (simple agent + tool calling)

```
START → supervisor → (retrieve_documents tool) → generate → END
                 ↘________________no-retrieval-needed___________↗
```

- **supervisor node**: a lightweight router. Default path: always retrieve (RAG assignment ⇒ retrieval-first). It only skips retrieval for trivially non-informational turns (e.g. "hi", "thanks") by a cheap heuristic/classifier, answering directly from short-term memory instead.
- **retrieve_documents tool**: embeds the question, runs Weaviate `nearVector` search filtered by `user_id == X AND document_set_id == Y`, returns top-k chunks with similarity scores.
- **generate node**: builds a prompt with numbered context blocks `[1]..[k]`, short-term history, long-term summary, and the question; instructs the LLM to answer **only** from the numbered context and to cite with `[n]` markers; calls `llm/router.py` (Groq → OpenRouter fallback).
- **grounding gate** (inside generate node, before calling the LLM): if no chunk clears a minimum similarity threshold, skip the LLM call for the factual-answer path and return the fixed "not found in the uploaded documents" response directly — this guarantees the model can't hallucinate an answer with no retrieved evidence, independent of prompt-following.

State object carries: `user_id`, `document_set_id`, `session_id`, `question`, `retrieved_chunks`, `answer`, `citations`.

---

## 4. Chunking strategy

- Recursive character/token splitter (LangChain `RecursiveCharacterTextSplitter` or equivalent), **target ~800 tokens per chunk, ~100 token overlap (~12%)** — large enough for context, small enough for precise citation.
- Split boundaries prefer paragraph → sentence → word, in that order, so chunks don't cut mid-sentence where avoidable.
- Per-chunk metadata captured at split time (not recomputed later): `document_id`, `filename`, `document_set_id`, `user_id`, `page_number` (PDF) or `section_heading` (MD/DOCX, nearest preceding heading), `chunk_index`, `char_start`/`char_end` offsets into the source text.
- PDFs: extracted page-by-page so page numbers are known before chunking (a chunk that spans two pages is tagged with the starting page).
- Rationale: fixed small-ish chunks keep retrieval precision high and keep citations meaningfully scoped to "this specific passage," at the cost of occasionally splitting a concept across 2 chunks (mitigated by overlap + top-k retrieval pulling in neighbors).

## 5. Grounding & citations

- Every chunk stored in Weaviate retains enough metadata to be rendered as a human-readable source (`filename`, `page`/`section`, `chunk_index`).
- The generation prompt presents retrieved chunks as numbered blocks and explicitly instructs the model: *"Answer only using the numbered context. Cite every claim with its bracket number(s). If the context does not contain the answer, say so explicitly."*
- The backend parses `[n]` markers in the model output and maps them back to the actual chunk metadata, returning a structured `citations[]` array (not just inline text) so the frontend can show real source snippets rather than trusting the model's own claim.
- If the model cites a bracket number that wasn't actually provided (hallucinated citation), the backend drops it rather than surfacing a broken reference — a last-mile sanity check, not a grounding mechanism by itself.

## 6. Handling "not in the documents"

Two layers, not just prompting:
1. **Threshold gate (primary, deterministic)**: if the best retrieved chunk's similarity score is below `SIMILARITY_THRESHOLD` (configurable), the generate node never calls the LLM for a factual answer — it returns a fixed "I don't have information about that in these documents" response with zero citations. This is the backstop against hallucination.
2. **Prompt instruction (secondary)**: even when chunks pass the threshold, the system prompt instructs the model to say explicitly when the retrieved context is insufficient to fully answer, rather than filling gaps from its own knowledge.

Risk accepted: threshold tuning is empirical — too high and it under-answers borderline-relevant questions, too low and weak matches slip through to the LLM (relying then on layer 2). Plan to make `SIMILARITY_THRESHOLD` and `TOP_K` env-configurable so this can be tuned without a code change.

## 7. Supported file types

| Type | Library | Notes |
|---|---|---|
| PDF | `pypdf` (or `pdfplumber` if text extraction quality needs improvement) | per-page text + page numbers |
| DOCX | `python-docx` | paragraph + heading text, headings tracked for section metadata |
| TXT | built-in | plain read, UTF-8 (fallback latin-1) |
| MD | built-in + lightweight heading parse | headings tracked for section metadata |

Dispatch by file extension in `ingestion/loaders.py`; unsupported extensions rejected at upload with a clear 400 error. Adding a new file type later only touches `loaders.py` — doesn't affect chunking/embedding/retrieval, keeping "different document sets without code changes" intact for formats already supported.

## 8. Config via `.env`

```
# LLM
GROQ_API_KEY=
GROQ_MODEL=llama-3.3-70b-versatile
OPENROUTER_API_KEY=
OPENROUTER_MODEL=meta-llama/llama-3.1-70b-instruct
LLM_TIMEOUT_SECONDS=20

# Embeddings
EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5

# Retrieval
TOP_K=5
SIMILARITY_THRESHOLD=0.35

# Chunking
CHUNK_SIZE_TOKENS=800
CHUNK_OVERLAP_TOKENS=100

# Weaviate
WEAVIATE_URL=http://weaviate:8080

# Postgres
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=ragdb
POSTGRES_USER=raguser
POSTGRES_PASSWORD=
DATABASE_URL=postgresql+psycopg://raguser:${POSTGRES_PASSWORD}@postgres:5432/ragdb

# Auth
JWT_SECRET=
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Storage
UPLOAD_DIR=/data/uploads
MAX_UPLOAD_MB=25

# Memory
LONG_TERM_SUMMARY_EVERY_N_TURNS=6

# Service
BACKEND_PORT=8000
FRONTEND_PORT=8501
BACKEND_BASE_URL=http://backend:8000
```

`.env.example` committed with blank secrets; `.env` gitignored.

## 9. Testing approach

- **Unit tests** (no external services, fast, run in CI/pre-push):
  - Loaders: each file type → expected plain text + metadata (page numbers, headings).
  - Chunker: chunk size/overlap bounds, offset correctness, no mid-word splits.
  - Citation mapping: `[n]` markers → correct chunk metadata; hallucinated bracket numbers dropped.
  - JWT: issue/verify/expiry/tampered-token rejection.
  - LLM router: Groq failure → OpenRouter fallback triggered (mocked HTTP clients, no real API calls).
- **Integration tests** (docker-compose test stack or `testcontainers-python` spinning up ephemeral Weaviate + Postgres):
  - Full ingestion pipeline: upload a sample PDF/DOCX/TXT/MD from `resources_Delete_later/` → assert chunks land in Weaviate with correct metadata and in Postgres `documents` table.
  - Retrieval isolation: two users upload to the same `document_set_id` name → assert each only ever retrieves their own chunks (the core security property of the global-collection design).
  - Below-threshold query → asserts the fixed "not in documents" response with empty citations, and that the LLM was never called (mock asserts zero calls).
- **End-to-end smoke test**: `httpx.AsyncClient` against the FastAPI test app — register → login → upload → ask a question whose answer is verifiably in the demo doc → assert citations reference the right file/page.
- **Manual checklist** (Streamlit, since UI flows aren't worth automating for this assignment): login, upload across all 4 file types, create a second document set and confirm no cross-contamination, ask an in-document question and an out-of-scope question, verify citations are clickable/expandable and point to real snippets.
- LLM calls are mocked in all automated tests — no real Groq/OpenRouter spend in CI; a small number of manual "real" queries against demo docs serve as the acceptance check before submission.

## 10. Trade-offs & risks

1. **Single global Weaviate collection + metadata filtering** (vs. one Weaviate class per user/set): simpler schema management and no per-tenant schema migrations, but every query depends on the filter being applied correctly — a missing filter is a data leak, not just a wrong result. Mitigated by centralizing all retrieval through one `queries.py` function that always requires `user_id`, and covering it with the cross-user isolation integration test above.
2. **Groq primary / OpenRouter fallback**: gives resilience against Groq rate limits/outages, but the two providers may serve different underlying models, so answer quality/style can shift mid-session on fallback. Accepted for an assignment; would pin equivalent models on both sides for production.
3. **Local open-source embeddings**: no per-call cost and no external dependency for the core retrieval loop, but CPU-only inference adds latency on ingestion of large PDFs and increases the backend Docker image size (ML deps + model weights). Acceptable at assignment scale.
4. **Synchronous-ish ingestion via BackgroundTasks** (no queue/worker): fine for a handful of documents; a very large batch upload would block worker threads rather than scaling out. No Celery/queue in scope — noted as a known limitation, not solved.
5. **Streamlit frontend**: fastest to build and fully Python, but JWT/session handling in Streamlit is more manual than a real SPA (relies on `st.session_state`, lost on hard refresh unless tokens are also cached e.g. in a cookie/localStorage workaround). Acceptable trade for assignment speed.
6. **Similarity-threshold grounding gate**: deterministic and hallucination-resistant, but a single global threshold won't be equally right for every document set's embedding distribution; borderline-relevant questions may occasionally be under- or over-answered. Exposed via `.env` so it's tunable without code changes.
7. **Long-term memory via periodic LLM summarization**: adds extra LLM calls (cost + latency) on a background cadence; if summarization fails silently, long-term memory quietly stops improving (mitigated by logging/alerting-equivalent: a status flag on the `long_term_memory` row, checked in integration tests).

## 11. Open questions for you

- Do you want **multi-file upload in one request** (batch) or one-file-per-request initially? (Affects the `/documents/upload` contract — assuming batch is fine, flag if not.)
- Should document sets be **deletable/overwritable** by the user (re-ingest fresh), or append-only for this assignment? (Affects whether we need a delete-from-Weaviate path.)
- Is **Docker required to run on your machine for Weaviate** (i.e., confirm Docker Desktop is available on this Windows box), or should there be a fallback to an embedded/in-memory vector store for quick local runs without Docker?
- Any constraint on **model choice within Groq/OpenRouter** (e.g. must stay within free tier), or is that entirely my call?

Once these are answered (or you say "use your best judgement"), next step is scaffolding the repo structure above — still no application code until you confirm this plan.
