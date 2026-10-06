# RAG Generator — Implementation Plan (Corrected)

This supersedes `PLAN.md` with two changes the user requested:
1. No Alembic for now — Postgres schema is created directly at startup (`Base.metadata.create_all`), no migrations folder.
2. The retrieval step is expanded from a single "similarity search" call into a full multi-stage pipeline: enrich metadata → (index for hybrid search, at ingestion) → query understanding → query expansion → prefilter → hybrid retrieval → retrieve wide → rerank → post-filter.

Everything else (tech stack, scoping decisions, auth, memory model) is unchanged from the clarification round.

---

## 0. Scope recap

Assignment requires (from candidate brief, Option A):
- Accept documents at runtime (no code changes to add a new doc set)
- RAG over those documents
- Grounded, cited answers
- Works with different document sets without code changes

Locked-in decisions:
- **LLM**: Groq (open-source hosted models) as primary, OpenRouter as automatic fallback on error/timeout/rate-limit.
- **Embeddings**: open-source, local (`sentence-transformers`, e.g. `BAAI/bge-small-en-v1.5`), runs inside the backend container.
- **Reranker** (new, needed by the expanded retrieval pipeline): open-source cross-encoder, local (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2` or `BAAI/bge-reranker-base`), runs inside the backend container.
- **Document scoping**: one global Weaviate collection, isolation enforced purely by metadata filtering (`user_id` + `document_set_id`) on every query.
- **Auth**: JWT per user; every request scoped to the authenticated `user_id`.
- **Session memory**: short-term (in-session buffer) + long-term (cross-session rolling summary), both per-user.
- **Services**: 2 containers — `frontend` (Streamlit) and `backend` (FastAPI + LangGraph), plus infra containers `weaviate` and `postgres`.
- **File storage**: raw uploaded files on a local Docker volume, path tracked in Postgres.
- **Schema management**: no Alembic — SQLAlchemy models + `create_all()` on backend startup. Simple, fine for assignment scope; revisit if the project outlives this assessment.

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
                         ┌───────────────────────────────────────┐
                         │               backend                 │
                         │          (FastAPI, :8000)              │
                         │                                         │
                         │  Auth routes ──► Postgres (users)       │
                         │  Documents routes ──► disk + ingestion   │
                         │                       pipeline ──► Weaviate
                         │                                         │
                         │  Chat routes ──► LangGraph supervisor   │
                         │                  graph:                 │
                         │                                         │
                         │   START ─► supervisor ─► retrieval      │
                         │            pipeline (9 stages, §3.3)    │──► Weaviate
                         │                      │                   │   (hybrid search)
                         │                      ▼                   │
                         │                  generate node           │──► Groq / OpenRouter
                         │                  (grounding gate +       │
                         │                   citation mapping)      │
                         │                      │                   │
                         │                      ▼                   │
                         │            memory read/write             │──► Postgres
                         │                                         │   (checkpoints,
                         └───────────────────────────────────────┘    summaries)
                                     │
                     ┌───────────────┴───────────────┐
                     ▼                                ▼
            ┌─────────────────┐             ┌──────────────────┐
            │     Weaviate      │             │     Postgres       │
            │  class: Chunk     │             │ users, documents,  │
            │  (dense vector +  │             │ chat_sessions,     │
            │   BM25-indexed    │             │ chat_messages,     │
            │   text property)  │             │ long_term_memory,  │
            │                   │             │ langgraph_checkpts │
            └─────────────────┘             └──────────────────┘
```

---

## 2. Folder structure

```
HRM-Assesment/
├── PLAN.md
├── plan_recorrected.md
├── docker-compose.yml
├── .env.example
├── resources_Delete_later/          # demo docs for manual testing
│
├── backend/
│   ├── Dockerfile
│   ├── pyproject.toml / requirements.txt
│   ├── app/
│   │   ├── main.py                  # FastAPI app factory, router mounting,
│   │   │                            #   Base.metadata.create_all() on startup
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
│   │   │   └── session.py           # engine/session factory, create_all() call
│   │   ├── ingestion/
│   │   │   ├── loaders.py           # pdf/docx/txt/md -> list[PageText]
│   │   │   ├── chunker.py           # text -> list[Chunk] (text, offsets, page)
│   │   │   ├── embedder.py          # sentence-transformers wrapper
│   │   │   └── pipeline.py          # orchestrates load->chunk->embed->upsert
│   │   │                            #   (also responsible for the "index for
│   │   │                            #   hybrid search" step — see §3.3 Stage 0)
│   │   ├── retrieval/                # <- the expanded 9-stage pipeline lives here
│   │   │   ├── weaviate_client.py    # connection, schema bootstrap (vector +
│   │   │   │                        #   BM25-indexed text property)
│   │   │   ├── metadata_enricher.py  # Stage 1: enrich metadata
│   │   │   ├── query_understanding.py# Stage 3: intent classify, coreference
│   │   │   │                        #   resolution, filter/entity extraction
│   │   │   ├── query_expansion.py    # Stage 4: paraphrase / HyDE expansion
│   │   │   ├── prefilter.py          # Stage 5: build Weaviate `where` filter
│   │   │   ├── hybrid_search.py      # Stage 6+7: hybrid (BM25+vector) search,
│   │   │   │                        #   requesting a wide candidate pool
│   │   │   ├── reranker.py           # Stage 8: cross-encoder reranking
│   │   │   ├── postfilter.py         # Stage 9: threshold cutoff, dedupe, top_k
│   │   │   └── pipeline.py           # orchestrates all 9 stages end-to-end
│   │   ├── agent/
│   │   │   ├── graph.py             # LangGraph StateGraph: supervisor + nodes
│   │   │   ├── state.py             # TypedDict/pydantic graph state
│   │   │   ├── tools.py             # retrieve_documents tool (wraps retrieval/pipeline.py)
│   │   │   ├── prompts.py           # system/grounding prompt templates
│   │   │   └── memory.py            # short-term checkpointer + long-term summarizer
│   │   ├── llm/
│   │   │   ├── groq_client.py
│   │   │   ├── openrouter_client.py
│   │   │   └── router.py            # try Groq, fall back to OpenRouter
│   │   └── schemas/                 # pydantic request/response models
│   └── tests/
│       ├── unit/                    # chunker, loaders, each retrieval stage, jwt
│       └── integration/             # full pipeline against test Weaviate/Postgres
│
└── frontend/
    ├── Dockerfile
    ├── requirements.txt
    └── streamlit_app.py
```

Note: no `alembic/` directory, no migration tooling in `requirements.txt`. Postgres tables are created via `Base.metadata.create_all(engine)` called once at backend startup — acceptable since the schema is small and stable for this assignment; the trade-off is no safe way to evolve the schema on existing data later (noted in §10).

---

## 3. Data flow

### 3.1 Ingestion

Unchanged from `PLAN.md`, with one addition: when chunks are upserted into Weaviate, the `text` property is configured with inverted-index/tokenization enabled (not just stored), so it's queryable by BM25 — this is the prerequisite the hybrid retrieval stage depends on. Concretely:

1. User logs in → JWT issued.
2. User uploads file(s) under a `document_set_id` → saved to disk, `documents` row created (`status=processing`).
3. Ingestion pipeline (background task): `loaders.py` → `chunker.py` → `embedder.py` → upsert into Weaviate class `DocumentChunk` with both the dense vector and an indexed `text` field, tagged `user_id` + `document_set_id`. `documents` row updated to `ready`/`failed`.
4. Frontend polls `/documents/sets` for status.

### 3.2 Querying — high level

1. User sends a message in a chat session (`user_id`, `document_set_id`, `session_id` all resolved from JWT + session state).
2. LangGraph supervisor graph runs the **retrieval pipeline** (§3.3) as its tool call, then the **generate node** (grounding gate + citation mapping, unchanged from `PLAN.md` §5–6).
3. Turn is persisted to short-term memory (checkpoint) and `chat_messages`; long-term summary refreshed periodically.
4. Response `{answer, citations}` returned to frontend.

### 3.3 Retrieval pipeline (expanded — the core change in this revision)

Replaces the single `similarity_search()` call from `PLAN.md` with 9 stages. Implemented as one orchestrating function in `retrieval/pipeline.py`, each stage a separate testable module:

**Stage 0 — Index for hybrid search** *(ingestion-time prerequisite, not a per-query step)*
Weaviate's `DocumentChunk` class is configured so the `text` property is both stored and indexed for keyword (BM25) search, alongside the dense vector produced by the embedder. This is what makes Stage 6 "hybrid" possible — it's listed here because the query pipeline depends on it, but it actually happens once, during ingestion (§3.1).

**Stage 1 — Enrich metadata**
Before any retrieval logic runs, build a `RetrievalContext`: `user_id`, `document_set_id` (both mandatory, from JWT/session — the security boundary), plus soft context pulled from short-term memory (recently-mentioned filenames, last turn's topic) that later stages may use as hints.

**Stage 2 — Query understanding**
Classify the incoming message (`needs_retrieval: bool` — filters out pure chit-chat/thanks, same as the old supervisor heuristic), resolve coreferences against short-term memory ("the second one" → actual filename/entity), and extract any explicit filters the user names (a specific filename, section, or date mentioned in the question). Output: `rewritten_query` + `extracted_filters`. Cheap, can be rule-based first (regex/keyword heuristics) with an LLM-based fallback for ambiguous cases — avoids an LLM round-trip on every single message.

**Stage 3 — Query expansion**
From `rewritten_query`, generate 2–3 additional query variants to broaden recall: paraphrases and/or a HyDE-style hypothetical-answer passage (ask the LLM "write a short passage that would answer this question," embed that instead of/alongside the literal question — hypothetical passages often match document phrasing better than terse user questions). All variants are embedded; keyword terms extracted from the original query are kept for the BM25 side of hybrid search.

**Stage 4 — Prefilter**
Build the Weaviate `where` filter: `user_id == X AND document_set_id == Y` (always, non-negotiable), plus any `extracted_filters` from Stage 2 (e.g. `filename == "contract_v2.pdf"`) when present. This runs *before* the search, not after, so the candidate pool is already scoped and the hybrid search operates over a smaller, relevant subset.

**Stage 5 — Hybrid retrieval**
Run Weaviate's hybrid search (alpha-weighted fusion of BM25 keyword score + vector cosine similarity) using the Stage 3 query embedding(s) and keyword terms, constrained by the Stage 4 filter. Hybrid (vs. pure vector) matters here because exact terms — names, codes, numbers, acronyms — are common in uploaded business/legal/technical documents and are exactly what pure dense retrieval tends to miss.

**Stage 6 — Retrieve wide**
Request a larger candidate pool than the final answer needs (`TOP_K_WIDE`, e.g. 25–30) from Stage 5, rather than asking Weaviate directly for the final top-5. Over-fetching trades a bit of latency/compute for recall, which the next stage (rerank) needs in order to find the genuinely best chunks rather than whatever a single bi-encoder/BM25 fusion score happened to rank first.

**Stage 7 — Rerank**
Score every (original question, candidate chunk) pair with a local cross-encoder reranker. Cross-encoders are much more precise than bi-encoder similarity because they attend over the query and chunk jointly, at the cost of being too slow to run over the whole collection — which is exactly why Stage 6 first narrows the field to a manageable wide candidate set. Sort candidates by rerank score.

**Stage 8 — Post-filter**
Deterministic cleanup on the reranked list: drop anything below `RERANK_THRESHOLD`, deduplicate near-identical/adjacent chunks (e.g. consecutive `chunk_index` from the same document with heavy overlap), truncate to final `TOP_K` (e.g. 5). This is also where the **grounding gate** lives: if nothing survives post-filter, the generate node returns the fixed "not in the documents" response and never calls the LLM for a factual answer (same mechanism as `PLAN.md` §6, now anchored to the rerank score instead of raw similarity score — more reliable since rerank scores are query-aware).

Output of the whole pipeline: an ordered list of `{chunk_text, filename, page/section, chunk_id, rerank_score}` passed to the generate node exactly as before.

```
question
   │
   ▼
[1] enrich metadata  (user_id, document_set_id, short-term hints)
   │
   ▼
[2] query understanding  (needs_retrieval?, rewritten_query, extracted_filters)
   │
   ▼
[3] query expansion  (paraphrases / HyDE passage, keyword terms)
   │
   ▼
[4] prefilter  (build Weaviate `where`: user_id + document_set_id + extracted_filters)
   │
   ▼
[5] hybrid retrieval  (BM25 + vector, filtered)  ──uses Stage 0 index──┐
   │                                                                   │
   ▼                                                                   │
[6] retrieve wide  (top_k_wide candidates) ◄───────────────────────────┘
   │
   ▼
[7] rerank  (cross-encoder, query-aware scoring)
   │
   ▼
[8] post-filter  (threshold, dedupe, final top_k)  ──► generate node (grounding gate here)
```

---

## 4–9. (Unchanged from `PLAN.md`)

Chunking strategy, citations mechanism, "not in the documents" prompt-level handling, supported file types, and testing approach are unchanged — see `PLAN.md` §4, §5, §6, §7, §9. The grounding gate described there now reads the Stage 8 rerank score instead of a raw cosine similarity score; everything else is identical.

## 10. Config via `.env` (additions on top of `PLAN.md` §8)

```
# Reranker (new)
RERANKER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANK_THRESHOLD=0.0

# Retrieval pipeline (new/renamed)
TOP_K_WIDE=25
TOP_K=5
HYBRID_ALPHA=0.5          # 0 = pure BM25, 1 = pure vector
QUERY_EXPANSION_VARIANTS=2
ENABLE_QUERY_EXPANSION=true
```

`SIMILARITY_THRESHOLD` from `PLAN.md` is superseded by `RERANK_THRESHOLD` (the gate now runs post-rerank).

## 11. Trade-offs & risks (additions on top of `PLAN.md` §10)

8. **Retrieval latency**: the 9-stage pipeline (query understanding + expansion each may involve an LLM call, plus a local reranker pass over ~25–30 candidates) is meaningfully slower per-query than a single vector search — likely adds a few hundred ms to low-single-digit seconds depending on hardware. Acceptable for an assignment demo; a production system would cache query understanding/expansion for repeated question patterns and consider a lighter/distilled reranker.
9. **Two more LLM round-trips per query** (query understanding's LLM fallback, query expansion's paraphrase/HyDE generation) increase Groq/OpenRouter usage and the surface area for the fallback logic to trigger. Mitigated by making both stages rule-based-first with LLM only as a fallback for ambiguous input, and by `ENABLE_QUERY_EXPANSION` being a toggle if cost/latency becomes an issue during testing.
10. **No Alembic**: schema changes during development mean dropping/recreating tables (dev-only risk, not a production concern at this scope) rather than versioned migrations. Acceptable now; flagged as the first thing to add back if this ever needs to run against data worth preserving across schema changes.
11. **Reranker as another local ML model**: like the embedder, adds to backend image size and CPU inference time; for the handful of documents in an assignment-scale demo this is a non-issue, but it's the same category of trade-off as the embedding model choice in `PLAN.md` §10.3.
