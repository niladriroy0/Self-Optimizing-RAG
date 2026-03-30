# Self-Optimizing RAG System Architecture Documentation

## Overview

The Self-Optimizing RAG (Retrieval-Augmented Generation) system is a Python-based, modular question-answering platform that combines vector search, keyword retrieval, cross-encoder reranking, and an LLM to produce answers and continuously improve via experiment logging and config selection. The implementation organizes responsibilities into clear packages for API, retrieval, control-plane routing, optimization, ingestion, evaluation, caching, logging, UI, observability, and embeddings.

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
│   Data Plane    │◀──▶│   RAG Pipeline  │───▶│   Optimization  │
│(Stubs currently)│    │   (Core Logic)  │    │   Engine        │
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
- **Keyword Search**: BM25 (via `rank_bm25.BM25Okapi`, index built from loaded documents)
- **Reranking**: Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) via `sentence_transformers.CrossEncoder`
- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`) wrapped in `embeddings/embedding_service.py`
- **LLM**: Ollama HTTP endpoint (`http://localhost:11434/api/generate`) called from `llm/llm_service.py`
- **Evaluation**: Cosine similarity (`sklearn`) using embeddings
- **Experiment Tracking**: SQLite helper (`optimization/experiment_db.py`) populated by background evaluation workers.
- **Caching**: MD5-based dict caching (`cache/query_cache.py`), a Redis stub (`cache/redis_cache.py`), and ChromaDB-backed semantic memory (`cache/chroma_memory_store.py`).
- **Observability**: Custom performance tracking logic (`observability/latency_tracker.py`).
- **Logging**: SQLite database tracking via `optimization/experiment_db.py`.
- **UI/Dashboard**: Streamlit apps in `ui/streamlit_app.py` and `dashboard/app.py`.
- **Dependencies**: See `requirements.txt` — validate presence of `chromadb`, `sentence-transformers`, `rank-bm25`, `scikit-learn`, `requests`, `numpy`, etc.

### Core Components

#### 1. API Layer (`api/`)
- **Framework**: FastAPI
- **Purpose**: REST endpoint(s) for query processing, debugging, and health checks
- **Key Files**:
  - `api/routes/query_routes.py` — `POST /query` streams LLM answers word-by-word and emits JSON observability. `POST /feedback` verifies memory. `GET /config` returns current active Control Plane configs.
- **Behavior**: Streams incrementally (word chunks with a short sleep) and appends `__OBSERVABILITY_START__` followed by JSON observability data.

#### 2. Application Core (`app/`)
- **Framework**: FastAPI
- **Purpose**: App instantiation and startup initialization
- **Key Files**:
  - `app/main.py` — includes router and on-startup handler.
- **Startup Process**:
  - Reads the Chroma collection (`vectorstore/chroma_store.collection.get()`).
  - If documents present, calls `retrieval.keyword_retriever.build_index(docs)` to build BM25 index.
  - Logs initialization status to console.

#### 3. Retrieval Pipeline (`retrieval/`)
- **Purpose**: End-to-end request handling from query to answer + observability
- **Main Flow**: Implemented in `retrieval/pipeline.py`.

##### Pipeline Flow (`retrieval/pipeline.py`)
Key steps executed by `rag_pipeline(question)`:
1. Setup tracking via `observability.latency_tracker.LatencyTracker`.
2. Load global configs from Control Plane via `config_manager.get_config()`.
3. Query analysis via `query_processing.query_analyzer.analyze_query`.
4. Cache Check via `cache.query_cache.get_cached` (skipped for "coding" queries or if disabled in config).
5. Model selection via `control_plane.model_router.route_model_with_exploration`.
6. Knowledge routing via `control_plane.knowledge_router.route_knowledge`.
7. If `LLM_ONLY` -> generates answer and returns early, measuring LLM latency.
8. Optimizer Config via `optimization.optimizer.choose_config()`.
9. Multi-hop Retrieval (if enabled) via `query_processing.query_planner.decompose_query` and multiple `hybrid_search` calls.
10. Deduplication to remove redundantly retrieved documents.
11. Reranking (`retrieval.reranker.rerank`) to obtain top candidates, measuring rerank latency.
12. **Memory Augmentation** via `cache.chroma_memory_store.retrieve_memory(question)`. Prioritizes memory context explicitly via context fusion.
13. LLM generation via `llm.llm_service.generate_answer(...)`, measuring LLM latency.
14. Confidence Computation via `evaluation.confidence_model.compute_confidence` and sync evaluation via `evaluate_rag`.
15. **Sync Memory Storage** saves the successful response into ChromaDB memory if confidence is high enough.
16. Async worker processes SQLite logging. 
17. Returns answer and comprehensive observability metrics, including tracked latencies, and updates query cache.

##### Hybrid Retriever (`retrieval/hybrid_retriever.py`)
- Uses `vectorstore.chroma_store.search_documents(query, k=...)` and `retrieval.keyword_retriever.keyword_search(query, k=...)`, combines results (dedup), and applies `rerank(...)`.

##### Keyword Retriever (`retrieval/keyword_retriever.py`)
- Module-level `documents` and `bm25` globals populated by `build_index(chunks)`.
- `keyword_search(query, k=3)` returns the top-k BM25-ranked documents.

##### Reranker (`retrieval/reranker.py`)
- Uses `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")` to score (query, doc) pairs and returns top-k.

#### 4. Vector Store (`vectorstore/`)
- **Implementation**: `vectorstore/chroma_store.py`
- **Client**: `chromadb.PersistentClient(path="chroma_db")`
- **Collection**: `documents`
- **Functions**: `add_documents(chunks)` and `search_documents(query, k=3)`.

#### 5. Large Language Model Service (`llm/`)
- **Client**: `llm/llm_service.generate_answer` posts to `http://localhost:11434/api/generate`.
- **Prompt Builder**: `llm/prompt_builder.build_prompt(...)`.
- **Formatting**: `clean_output` and `format_code_block`.

#### 6. Query Processing (`query_processing/`)
- **Key Files**:
  - `query_analyzer.py` — Extracts basic features and classifies `type` (coding, reasoning, general) using heuristic keywords.
  - `query_planner.py` — Handles basic multi-hop decomposition for comparisons and chained queries.

#### 7. Evaluation System (`evaluation/`)
- **Metrics**: Cosine similarity computes `answer_relevance` and `faithfulness` leveraging the centralized embedding service.
- **Confidence**: `confidence_model.py` unifies retrieval scores, memory boosts, complexity penalties, and evaluation scores into a single confidence metric.

#### 8. Optimization Engine (`optimization/`)
- **Components**:
  - `optimizer.py` — `choose_config()` queries `experiment_db`.
  - `experiment_db.py` — SQLite DB `experiments.db` with logic to log configurations.

#### 9. Data Ingestion (`ingestion/`)
- **Chunker**: `ingestion/chunker.chunk_document(text, chunk_size=400)`.
- **Script**: `scripts/ingest_documents.py` reads `.txt` files in `data/`, chunks them, and stores in Chroma.

#### 10. Observability & Logging (`observability/` & `rag_logging/`)
- **Observability**: Features `latency_tracker.py` for fine-grained performance monitoring. Includes `cost_tracker.py` as a stub.
- **Log file**: Consolidated into `experiments.db` using `experiment_db.py`.

#### 11. Embeddings (`embeddings/`)
- **Purpose**: Centralized service for vectorizing text.
- **Key Files**:
  - `embedding_service.py` - Wraps `SentenceTransformer("all-MiniLM-L6-v2")` into a clean function `embed_text`.

#### 12. Cache & Semantic Memory (`cache/`)
- **Purpose**: Local caching and advanced persistent storage for high-confidence past answers.
- **Key Files**:
  - `cache/query_cache.py` - basic in-memory MD5 dict cache.
  - `cache/chroma_memory_store.py` - persistent ChromaDB-backed semantic memory handling answer storage, TTL, and confidence-gated bounds.
  - `cache/redis_cache.py` - skeleton/stub pointing towards future remote cache support.
  - `cache/memory_store.py` - Legacy numpy implementation.

#### 13. Workers (`workers/`)
- **Key Files**:
  - `workers/evaluation_worker.py` - runs performance logging asynchronously to SQLite (memory storing has been shifted directly to the pipeline).

#### 14. Control Plane (`control_plane/`)
- **Purpose**: Centralized decision-making and configuration management.
- **Key Files**:
  - `control_plane/knowledge_router.py` - routes `LLM_ONLY`, `HYBRID`, `RAG`.
  - `control_plane/model_router.py` - scores parameters with bandit exploration.
  - `control_plane/config_manager.py` - thread-safe singleton config manager that tracks system state and versioning for experiments.

#### 15. User Interface & Dashboards (`ui/`, `dashboard/`)
- **Streamlit App**: `ui/streamlit_app.py` submits to `http://127.0.0.1:8000/query`.
- **Dashboard App**: `dashboard/app.py` reads `logs/rag_logs.json` to showcase experiment performance metrics.

#### 16. Data Plane (`data_plane/`)
- **Status**: Completely empty; routing orchestrated mostly within `retrieval/pipeline.py`.

## Data Flow

### Query Processing Flow

1. **User Query** -> FastAPI endpoint (`/query`)
2. **Setup Tracking** -> Initialize latency trackers
3. **Control Plane** -> Load active multi-hop / top_k configs via `ConfigManager`
4. **Query Analysis** -> `query_processing.analyze_query`
5. **Caching** -> `query_cache.get_cached` (if enabled in `ConfigManager`)
6. **Model & Knowledge Routing** -> Control plane selects model & mode
7. **Config Selection** -> `optimization.optimizer.choose_config()` and `control_plane.config_manager.get_config()`
8. **Hybrid Retrieval (Multi-hop)** -> If complex, breaks query via `query_planner` and aggregates hybrid searches (tracked latency)
9. **Memory Augmentation** -> Combine retrieved docs + semantic memory (from `chroma_memory_store`)
10. **LLM Generation** -> Ollama request with tailored structured prompt (tracked latency)
11. **Confidence Computation** -> Calculates a holistic confidence score combining eval metrics and heuristics
12. **Sync Memory Storage** -> Direct high-confidence returns to `chroma_memory_store`
13. **Async Background Logging** -> Write observability history to SQLite
14. **Observability Packaging** -> Package latencies, metrics, and confidence paths
15. **Response** -> Streaming response + JSON observability block

## Configuration and Parameters

### System Configurations
- **LLM Model**: Profiles listed in `llm/model_registry.py`. Escalation models configured in `pipeline.py`.
- **Vector DB**: ChromaDB (`chroma_db/`)
- **Embedding Models**:
  - Centralized embeddings: `all-MiniLM-L6-v2` (`embeddings/embedding_service.py`)
  - Reranking: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Keyword Search**: BM25Okapi

### Optimization Parameters
- **top_k**: Options `[2,3,4]`.
- **chunk_size**: Options `[200,400,600]`.
- **Global Config**: Managed dynamically via `ConfigManager`.

## Dependencies
- `fastapi`, `uvicorn`, `requests`
- `chromadb`, `rank-bm25`
- `sentence-transformers`, `scikit-learn`, `numpy`
- `streamlit`, `pandas`

## Current Implementation Status & Known Issues

- **Core Pipeline**: Integrated advanced routing, evaluation, caching, latency observability, and hybrid retrieval. Features multi-hop query decomposition, confidence scoring, and synchronous Chroma-backed semantic memory updates.
- **Optimization Sync Resolved**: Evaluation worker correctly logs to SQLite `experiment_db`.
- **Data Plane / Control Plane Stubs**: `data_plane` is empty. Core pipeline handles choreography. `control_plane` hosts `config_manager`.
- **New Modules Added**: `embeddings/` for reusable encoders, `observability/` for tracing, `query_processing/query_planner.py` for multi-hop decomposition, and `cache/chroma_memory_store.py` for dynamic short-term memory limits.
- **Testing**: No automated tests present.

## Deployment and Usage (Local)

1. `pip install -r requirements.txt`
2. Start Ollama local server
3. Initialize documents: `python scripts/ingest_documents.py`
4. Start FastAPI server: `python -m uvicorn app.main:app --reload`
5. Chat UI: `streamlit run ui/streamlit_app.py`
6. Admin Dashboard: `streamlit run dashboard/app.py`