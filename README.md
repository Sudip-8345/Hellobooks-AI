# HelloBooks RAG Q&A System

A Retrieval-Augmented Generation (RAG) based Question Answering system built on top of bookkeeping and accounting knowledge documents. It uses LangGraph for agent orchestration with short-term conversational memory and serves a chat UI via Panel.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM (primary) | Groq - Llama 3.3 70B |
| LLM (fallback) | Google Gemini 2.5 Flash |
| Embeddings | Google Generative AI Embeddings |
| Vector Store | FAISS (local, CPU) |
| Orchestration | LangGraph (StateGraph) |
| Framework | LangChain |
| Chat UI | Panel |
| Containerisation | Docker |

---

## Project Structure

```
.
├── app.py                          # Panel chat UI entry point
├── main.py                         # RAG pipeline (load, chunk, retrieve, generate)
├── agents.py                       # LangGraph agent with state, nodes and memory
├── config.py                       # All configuration / env variables
├── requirements.txt
├── Dockerfile
├── .env                            # API keys (you create this)
├── hellobooks_dataset/             # Source markdown documents
│   ├── balance_sheet.md
│   ├── bookkeeping_basics.md
│   ├── cash_flow_statement.md
│   ├── double_entry_bookkeeping.md
│   ├── financial_statements_relationship.md
│   ├── invoice_management.md
│   ├── invoices_overview.md
│   └── profit_and_loss_statement.md
└── RAG_Engine/
    ├── indexing.py                 # Document loading, chunking, FAISS indexing
    ├── retrieval.py                # Retriever, re-ranking, source extraction
    └── generation.py               # LLM invocation with Groq/Gemini fallback
```

---

## Prerequisites

- Python 3.11+
- A Groq API key and/or a Google AI API key

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd Assignment-meru-technosoft
```

### 2. Create a `.env` file

```
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
panel serve app.py --show
```

The chat interface will open at `http://localhost:5006/app`.

---

## Running with Docker

### Build

```bash
docker build -t hellobooks-rag .
```

### Run

```bash
docker run --env-file .env -p 5006:5006 hellobooks-rag
```

Open `http://localhost:5006/app` in your browser.

---

## How It Works

1. **Indexing** -- Markdown documents from `hellobooks_dataset/` are loaded, split into overlapping chunks, and indexed into a local FAISS vector store.
2. **Retrieval** -- The user query is embedded and the top-K most similar chunks are retrieved, then re-ranked by keyword overlap.
3. **Generation** -- Retrieved context is passed to an LLM (Groq Llama 3.3, with Gemini as fallback) to produce a grounded answer.
4. **Memory** -- LangGraph maintains a sliding window of the last N conversation turns so follow-up questions work naturally.

---

## Configuration

All tuneable parameters live in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| CHUNK_SIZE | 500 | Characters per chunk |
| CHUNK_OVERLAP | 40 | Overlap between chunks |
| TOP_K | 3 | Number of chunks retrieved |
| LLM_TEMPERATURE | 0.1 | LLM sampling temperature |
| MEMORY_MAX_TURNS | 5 | Conversation turns kept in memory |
