# RAG Document Q&A

A RAG (Retrieval-Augmented Generation) system: upload documents into a collection, then ask questions and get answers grounded in those documents, with citations back to the source file/page.

## 1. Prerequisites

- Docker + Docker Compose installed.
- A `.env` file in the project root with real values filled in for:
  - `GROQ_API_KEY` (required — primary LLM)
  - `OPENROUTER_API_KEY` (required — fallback LLM)
  - `WEAVIATE_URL` (required — Weaviate Cloud cluster URL)
  - `WEAVIATE_API_KEY` (required — Weaviate Cloud API key)

  If any of these are missing or placeholder values, the backend will fail to start or fail on first query. Ask the project owner for a working `.env` if you don't have one.

## 2. How to run it

From the project root:

```bash
docker compose up --build
```

Wait until you see in the logs:

```
backend-1   | INFO:     Application startup complete.
frontend-1  | You can now view your Streamlit app in your browser.
```

Then open:

- Frontend (chat UI): http://localhost:8501
- Backend API docs: http://localhost:8000/docs

To stop everything:

```bash
docker compose down
```

## 3. Testing via the frontend (recommended for most people)

1. Open http://localhost:8501
2. In the sidebar, you should see a collection called **"Ai_papers"** already seeded with 12 AI research papers (Attention Is All You Need, BERT, GPT-3, LLaMA, LLaMA 2, InstructGPT, ReAct, RAG, Chain-of-Thought, etc). Select it.
3. Type a question in the chat box at the bottom and press Enter.
4. Check the answer, and expand "N source(s)" below it to see which document/page it cited.

You can also:

- Create a new collection (sidebar form) and upload your own PDF/DOCX/TXT/MD files to test ingestion.
- Delete a collection with the trash button.

## 4. What "good" looks like — test cases to try

**A) Greetings / small talk (should NOT trigger document search):**

```
hi
hello
thanks!
ok
```

Expected: a short "ask me a question about the documents" reply, with NO sources shown. These should answer instantly (no "Thinking..." delay).

**B) Real questions about the documents (should cite the right paper):**

| Question | Expected citation |
|---|---|
| "What is the transformer architecture?" | `attention_is_all_you_need.pdf` |
| "How does BERT's masked language modeling work?" | `bert.pdf` |
| "What is RLHF in InstructGPT?" | `instructgpt.pdf` |
| "What is Llama 2 about?" | `llama2.pdf` |

**C) Casually/badly phrased questions (should STILL work, not just clean ones):**

```
can u tell me about main points of attention all you need paper
wut is react paper abt
yo whats bert do
```

Expected: same quality of grounded, cited answer as the clean phrasing of the same question. If these come back empty / "couldn't find this", that's a regression in the query-rewrite step.

**D) Off-topic questions (should be correctly refused, no fake sources):**

```
What is the capital of France?
How do I bake a cake?
What color is the sky?
```

Expected: "I couldn't find this in the provided documents." with EMPTY sources. If it comes back with an answer that cites a document anyway, that's a false-positive bug — note which document/page it wrongly cited.

**E) Upload + ingest test:**

1. Create a new collection (e.g. "test").
2. Upload a PDF/TXT/DOCX file.
3. Confirm it shows up under "Documents" with a chunk count > 0.
4. Ask a question you know the answer to from that file and confirm the citation points to the right file/page.
5. Try uploading the SAME file again — it should say "already ingested, skipped" rather than duplicating it.

**F) Collection management:**

- Create a collection with a weird name (spaces, symbols, emoji) and confirm it doesn't error out (names get sanitized automatically).
- Delete a collection and confirm its chat history disappears from the sidebar too.

## 5. Testing via the API directly (for devs / automated checks)

List collections:

```bash
curl http://localhost:8000/collections
```

Ask a question (replace `ai_papers` with your collection name):

```bash
curl -X POST http://localhost:8000/collections/ai_papers/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the transformer architecture?"}'
```

On Windows (cmd.exe), use `^` line continuations and escaped quotes instead:

```cmd
curl -X POST http://localhost:8000/collections/ai_papers/query ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"What is the transformer architecture?\"}"
```

List documents in a collection:

```bash
curl http://localhost:8000/collections/ai_papers/documents
```

Full interactive API docs (try requests straight from the browser):

http://localhost:8000/docs
