# Self-Optimizing RAG System Architecture Documentation

## Overview

The Self-Optimizing RAG (Retrieval-Augmented Generation) system is a Python-based, modular question-answering platform that combines vector search, keyword retrieval, cross-encoder reranking, and an LLM to produce answers and continuously improve via experiment logging and config selection. The implementation organizes responsibilities into clear packages for API, retrieval, control-plane routing, optimization, ingestion, evaluation, caching, observability, and UI.

## System Architecture

### High-Level Architecture

```text
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│   API Layer     │───▶│   Control Plane │
│  (UI/Direct)    │    │   (FastAPI)     │    │   (Model +      │
└─────────────────┘    └─────────────────┘    │   Knowledge +   │
                                              │   Config)       │
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│                 │◀──▶│   RAG Pipeline  │───▶│   Optimization  │
│                 │    │   (Core Logic)  │    │   Engine        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Caching &      │◀──▶│   Experiment    │◀──▶│   Dashboard     │
│  Semantic Memory│    │   Tracking (DB) │    │   (Streamlit)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Logging &     │    │   Observability │    │   Data Ingestion│
│   Eval Workers  │    │   Infrastructure│    │   Pipeline      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Technology Stack

- **Backend Framework**: FastAPI (app entry in `app/main.py`)
- **Vector Database**: ChromaDB (persistent client writing into `chroma_db/`)
- **Keyword Search**: BM25 (via `rank_bm25.BM25Okapi`, index built at startup from ChromaDB documents)
- **Reranking**: Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) via `sentence_transformers.CrossEncoder`
- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`) wrapped in `embeddings/embedding_service.py`
- **LLM**: Local Ollama HTTP endpoint (`http://localhost:11434/api/generate`) called from `llm/llm_service.py`
- **Evaluation**: Cosine similarity (`sklearn`) using centralized embeddings
- **Experiment Tracking**: SQLite (`optimization/experiment_db.py`) in WAL mode, populated by background evaluation workers
- **Caching**: MD5-based dict cache (`cache/query_cache.py`), Redis stub (`cache/redis_cache.py`), and ChromaDB-backed semantic memory (`cache/chroma_memory_store.py`)
- **Config Persistence**: PyYAML — `config_manager.py` reads/writes `configs/system_config.yaml` on every change
- **Cost Tracking**: `observability/cost_tracker.py` — records token usage and estimated USD cost per call, persisted to `logs/cost_log.json`
- **Observability**: Custom latency tracking (`observability/latency_tracker.py`) embedded in the pipeline
- **UI/Dashboard**: Streamlit apps in `ui/streamlit_app.py` and `dashboard/app.py`
- **Testing**: pytest + pytest-mock, 43 tests across 8 modules with global ML model mocking
- **Dependencies**: See `requirements.txt` — `chromadb`, `sentence-transformers`, `rank-bm25`, `scikit-learn`, `requests`, `numpy`, `pyyaml`, `streamlit`, `pandas`

### Core Components

#### 1. API Layer (`api/`)
- **Framework**: FastAPI
- **Purpose**: REST endpoints for query processing, feedback, index management, and health checks
- **Key Files**:
  - `api/routes/query_routes.py` — `POST /query` streams LLM answers word-by-word and emits JSON observability (including active config snapshot). `POST /feedback` calls `verify_memory` on `chroma_memory_store`. `POST /index/rebuild` triggers BM25 index refresh. `GET /config` returns current control-plane config version + params.
- **Behavior**: Streams incrementally (word chunks with a short sleep) and appends `__OBSERVABILITY_START__` followed by full JSON observability data including latencies, confidence, mode, and active config.

#### 2. Application Core (`app/`)
- **Framework**: FastAPI
- **Purpose**: App instantiation and startup initialization
- **Key Files**:
  - `app/main.py` — async `lifespan` context manager handles startup (SQLite schema init, BM25 index build from ChromaDB) and graceful shutdown. Includes `GET /` health endpoint.
- **Startup Process**:
  1. Calls `init_db()` to ensure SQLite schema exists.
  2. Reads all documents from ChromaDB collection.
  3. If documents present, builds BM25 index via `build_index(docs)`.
  4. Logs count to console.

#### 3. Retrieval Pipeline (`retrieval/`)
- **Purpose**: End-to-end request handling from query to `(answer, observability)` tuple
- **Main Flow**: Implemented in `retrieval/pipeline.py`.

##### Pipeline Flow (`retrieval/pipeline.py`)
Key steps executed by `rag_pipeline(question)`:
1. Initialize `LatencyTracker` and record start time.
2. Load global configs from `config_manager.get_config()`.
3. Query analysis via `query_processing.query_analyzer.analyze_query`.
4. Cache check via `cache.query_cache.get_cached` (skipped for coding queries or if disabled).
5. Model selection via `control_plane.model_router.route_model_with_exploration` (epsilon-greedy, ε=0.1).
6. Knowledge routing via `control_plane.knowledge_router.route_knowledge` → `LLM_ONLY`, `RAG`, or `HYBRID`.
7. **LLM_ONLY branch**: Retrieves semantic memories, generates answer, computes LLM-only confidence, stores cache and memory, returns early.
8. Config selection via `optimization.optimizer.choose_config()`.
9. Multi-hop retrieval — `query_planner.plan_query` decomposes if needed; multiple `hybrid_search` calls aggregated.
10. **Smart Fallback branch**: If 0 documents retrieved, promotes to parametric LLM generation with `FALLBACK_LLM` mode, stores cache and memory, returns early.
11. Deduplication and cross-encoder reranking via `retrieval.reranker.rerank`.
12. **Memory Augmentation** via `cache.chroma_memory_store.retrieve_memory`; memory prepended to context.
13. LLM generation via `llm.llm_service.generate_answer` (with cost tracking).
14. Sync evaluation via `evaluate_rag`; confidence via `compute_confidence`.
15. Sync memory store via `chroma_memory_store.store_memory` (confidence-gated, deduplicated, async cleanup).
16. Async evaluation worker: logs to SQLite, calls `adapt_config` for real-time parameter adjustment.
17. Returns `(answer, observability_dict)` and updates query cache.

##### Hybrid Retriever (`retrieval/hybrid_retriever.py`)
- Runs `vectorstore.chroma_store.search_documents` and `retrieval.keyword_retriever.keyword_search` in parallel, merges via Reciprocal Rank Fusion (RRF), deduplicates, and passes combined results to the reranker.

##### Keyword Retriever (`retrieval/keyword_retriever.py`)
- Module-level `BM25Okapi` index built at startup from ChromaDB's document collection.
- `keyword_search(query, k)` tokenizes and returns top-k BM25-ranked documents.
- `rebuild_from_chroma()` re-fetches all ChromaDB documents and rebuilds the index on demand.

##### Reranker (`retrieval/reranker.py`)
- Uses `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")` to score `(query, doc)` pairs and returns top-k sorted by relevance score.

#### 4. Vector Store (`vectorstore/`)
- **Implementation**: `vectorstore/chroma_store.py`
- **Client**: `chromadb.PersistentClient(path="chroma_db")`
- **Collection**: `documents`
- **Functions**: `add_documents(chunks)` and `search_documents(query, k)`.

#### 5. Large Language Model Service (`llm/`)
- **Client**: `llm/llm_service._call_llm` posts to `http://localhost:11434/api/generate`.
- **Cost Tracking**: Every `_call_llm` call records token estimates and cost via `cost_tracker.record()`.
- **Prompt Builder**: `llm/prompt_builder.build_prompt` selects template (code/rag/general/json), trims context.
- **Model Registry**: `llm/model_registry.py` defines `MODEL_REGISTRY` (category→model name), `MODEL_PROFILES` (capability+cost scores), and smart selection logic (`select_best_model`, `get_best_model_for_task`).
- **Fallback**: `generate_answer` triggers a secondary LLM call via `get_fallback_model` if confidence or LLM output signals are poor.

#### 6. Query Processing (`query_processing/`)
- **Key Files**:
  - `query_analyzer.py` — classifies `type` (coding/reasoning/general), `complexity` (low/medium/high), extracts keywords, `is_multi_hop`, `ambiguity`, and semantic need flags.
  - `query_decomposer.py` — LLM-based decomposition into atomic sub-questions; rule-based fallback for resilience.
  - `query_planner.py` — checks config flags and analysis result to return a single-query or multi-query list.

#### 7. Evaluation System (`evaluation/`)
- **Metrics**: `rag_evaluator.py` computes `answer_relevance` (question↔answer), `faithfulness` (answer↔context), and `hallucination_score` (1 − faithfulness) using cosine similarity.
- **Confidence**: `confidence_model.py` — in RAG mode: sigmoid-normalized reranker scores + memory boost − complexity penalty, multiplied by eval score. In LLM-only mode: content-aware heuristics (uncertainty phrases, length, code structure) modified by eval score multiplier. `sqrt()` was explicitly removed to prevent artificial confidence inflation.

#### 8. Optimization Engine (`optimization/`)
- **Components**:
  - `optimizer.py` — `choose_config()` queries `experiment_db` for the best historical config; `adapt_config()` reacts to poor evaluation scores by incrementing `top_k` or `chunk_size` and immediately persisting via `config_manager`.
  - `experiment_db.py` — SQLite schema + WAL mode, `log_experiment` inserts per-query rows, `get_best_config` queries for the highest-scoring config (with in-process cache).

#### 9. Data Ingestion (`ingestion/`)
- **Chunker**: `ingestion/chunker.chunk_document(text, chunk_size)`.
- **Script**: `scripts/ingest_documents.py` reads `.txt` files from `data/`, chunks, embeds, and stores in ChromaDB.

#### 10. Observability & Logging (`observability/`)
- **Latency**: `latency_tracker.py` — `start(key)` / `end(key)` / `get_all()` for per-stage pipeline timing returned in every API response.
- **Cost**: `cost_tracker.py` — thread-safe `CostTracker` singleton. `record(model, input_tokens, output_tokens)` accumulates session stats and persists all-time totals to `logs/cost_log.json` atomically. `estimate_tokens(text)` provides lightweight estimation without tiktoken.

#### 11. Embeddings (`embeddings/`)
- **Purpose**: Centralized text vectorization using `SentenceTransformer("all-MiniLM-L6-v2")`.
- **Key Files**:
  - `embedding_service.py` — `embed_text(text)` returns a 384-dimensional vector.

#### 12. Cache & Semantic Memory (`cache/`)
- **Purpose**: Multi-layer caching and persistent memory for high-confidence past answers.
- **Key Files**:
  - `cache/query_cache.py` — in-memory MD5 dict cache (volatile, exact match).
  - `cache/chroma_memory_store.py` — persistent ChromaDB semantic memory: stores query embeddings (not answer embeddings), near-duplicate detection (L2 < 0.1), async cleanup thread, TTL (7 days), capacity limit (200 items), confidence-gated retrieval.
  - `cache/redis_cache.py` — Redis client stub for future distributed cache.
  - `cache/memory_store.py` — ⚠️ DEPRECATED. Numpy-based in-memory prototype. Not imported anywhere in the active pipeline.

#### 13. Workers (`workers/`)
- **Key Files**:
  - `workers/evaluation_worker.py` — `run_evaluation_async` spawns a daemon thread that evaluates, logs to SQLite, stores memory in `chroma_memory_store`, and calls `adapt_config` for real-time parameter adjustment. Thread joins with 0.1s timeout to remain non-blocking.

#### 14. Control Plane (`control_plane/`)
- **Purpose**: Centralized decision-making and persistent configuration management.
- **Key Files**:
  - `control_plane/knowledge_router.py` — routes to `LLM_ONLY`, `RAG`, or `HYBRID` based on extracted query features.
  - `control_plane/model_router.py` — task-based routing first (rule-based), then capability scoring, then epsilon-greedy exploration (ε=0.1) for data collection.
  - `control_plane/config_manager.py` — thread-safe singleton with full YAML persistence (`configs/system_config.yaml`). Every write operation (`update_config`, `set_param`, `smart_update`) immediately saves to disk. On startup, loads from disk to restore learned state.

#### 15. User Interface & Dashboards (`ui/`, `dashboard/`)
- **Streamlit App**: `ui/streamlit_app.py` submits to `http://127.0.0.1:8000/query` and renders answer + observability metrics.
- **Dashboard App**: `dashboard/app.py` reads `experiments.db` to display experiment performance over time.

#### 16. Tests (`tests/`)
- **Status**: Full 43-test suite across 8 modules. Global ML model mocking via `sys.modules` in `conftest.py` enables CI without GPU or local Ollama dependencies. Autouse `mock_config_manager` fixture prevents cross-test config contamination.

#### 17. Configs (`configs/`)
- **`configs/system_config.yaml`**: Persisted control-plane state. Written by `config_manager` on every parameter update; read on startup. Solves the config amnesia problem.

## Data Flow

### Query Processing Flow

1. **User Query** → FastAPI endpoint (`/query`)
2. **Setup Tracking** → Initialize `LatencyTracker`, record start time
3. **Control Plane** → Load active config via `ConfigManager` (restored from YAML)
4. **Query Analysis** → `query_processing.analyze_query` classifies type, complexity, multi-hop
5. **Caching** → `query_cache.get_cached` (if enabled and not coding type)
6. **Model & Knowledge Routing** → Control plane selects model (epsilon-greedy) & mode
7. **Config Selection** → `optimization.optimizer.choose_config()` returns best historical config
8. **Hybrid Retrieval (Multi-hop)** → `query_planner` decomposes; aggregates hybrid searches (latency tracked). If 0 docs: Smart Fallback → `FALLBACK_LLM` branch
9. **Memory Augmentation** → `chroma_memory_store.retrieve_memory` injects verified past answers
10. **LLM Generation** → Ollama request with structured prompt (latency tracked; cost recorded)
11. **Confidence Computation** → Holistic score: reranker strength × eval score, memory boost, complexity penalty
12. **Sync Memory Storage** → `chroma_memory_store.store_memory` (confidence-gated, deduplicated)
13. **Async Background Logging** → Write to SQLite, call `adapt_config` (persists config changes to YAML)
14. **Observability Packaging** → Latencies, metrics, confidence, model, mode, active config
15. **Response** → Streaming word-by-word + JSON observability block

## Configuration and Parameters

### System Configurations
- **LLM Models**: Defined in `llm/model_registry.py` — `phi3:latest` (fast), `mistral:latest` (balanced), `qwen2.5:3b` (reasoning), `deepseek-coder:1.3b`/`6.7b` (coding).
- **Vector DB**: ChromaDB (`chroma_db/`)
- **Embedding Model**: `all-MiniLM-L6-v2` (384-dim, centralized in `embeddings/embedding_service.py`)
- **Reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Keyword Search**: `BM25Okapi`

### Optimization Parameters
- **top_k**: Retrieval document count. Default 5, dynamically adjusted by `adapt_config`.
- **chunk_size**: Document chunking size. Default 500, dynamically adjusted by `adapt_config`.
- **confidence_threshold**: Memory/cache gating threshold. Default 0.6.
- **enable_multi_hop**, **enable_decomposition**, **enable_fallback**: Feature flags for intelligence modes.
- **keyword_weight / semantic_weight**: Relative weighting in hybrid retrieval. Default 0.5/0.5.
- **All parameters persisted** to `configs/system_config.yaml` and restored on restart.

## Dependencies
- `fastapi`, `uvicorn`, `requests`
- `chromadb`, `rank-bm25`
- `sentence-transformers`, `scikit-learn`, `numpy`
- `pyyaml`
- `streamlit`, `pandas`
- `pytest`, `pytest-mock`

## Current Implementation Status

- **Core Pipeline**: Full hybrid retrieval, cross-encoder reranking, multi-hop decomposition, smart fallback for empty retrieval, content-aware confidence scoring, synchronous Chroma-backed semantic memory.
- **Config Persistence**: `config_manager.py` now reads/writes `configs/system_config.yaml` — learned config survives restarts.
- **Cost Tracking**: `cost_tracker.py` fully implemented and wired into `llm_service._call_llm` — every LLM call is recorded with token estimates and USD cost.
- **Memory Store Fixed**: `chroma_memory_store.py` now embeds query vectors (not answer vectors), includes near-duplicate detection, and runs cleanup in a background thread.
- **Deprecated Code Removed from Pipeline**: `cache/memory_store.py` (numpy-based) is fully deprecated and no longer imported anywhere in the active pipeline (`query_routes.py` and `evaluation_worker.py` both updated to use `chroma_memory_store`).
- **Test Suite**: 43 tests across 8 modules with proper ML model mocking, fixture isolation, and coverage of all 4 pipeline branches.
- **Evaluation Worker**: Now calls `chroma_memory_store.store_memory` and `adapt_config` for real-time self-optimization.

## Deployment and Usage (Local)

1. `pip install -r requirements.txt`
2. Start Ollama local server with desired models (e.g., `ollama pull llama3`)
3. Place `.txt` documents in `data/` and run: `python scripts/ingest_documents.py`
4. Start FastAPI server: `python -m uvicorn app.main:app --reload`
5. Chat UI: `streamlit run ui/streamlit_app.py`
6. Admin Dashboard: `streamlit run dashboard/app.py`
7. Run tests: `pytest tests/ -v`