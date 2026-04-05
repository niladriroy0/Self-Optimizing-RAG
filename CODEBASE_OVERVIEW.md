# Codebase Overview: Self-Optimizing RAG

## `.\\api\\routes\\query_routes.py`
- **Contains:** FastAPI routes for handling queries, feedback, index rebuilding, and streaming responses. Includes a debug endpoint for active configurations.
- **Functions:**
  - `stream_response`: Streams the LLM response back to the client word-by-word.
  - `query`: Handles the main RAG querying logic, appends control-plane config snapshot to the observability payload.
  - `rebuild_index`: Triggers a rebuild of the BM25 keyword search index from ChromaDB.
  - `feedback`: Accepts user feedback on the quality of the answer, calling `verify_memory` on the active ChromaDB memory store.
  - `get_config`: Debug endpoint returning a versioned snapshot of the active control-plane config.
- **Purpose in Application:** Serves as the main entry point to the backend for the Streamlit UI and any other API clients. It glues the core RAG pipeline with HTTP requests.
- **Places to Improve:** Add authentication/rate-limiting. `stream_response` could use WebSockets for more robust streaming.
- **Need Rating:** 5/5

## `.\\app\\main.py`
- **Contains:** FastAPI application setup, async lifespan manager, and health check.
- **Functions:**
  - `lifespan`: On startup: initializes SQLite schema via `init_db()`, loads ChromaDB documents, and builds the BM25 index. Handles `asyncio.CancelledError` gracefully on shutdown.
  - `health`: Simple `GET /` endpoint returning `{"status": "running"}`.
- **Purpose in Application:** Bootstraps the entire backend. Essential entry point.
- **Places to Improve:** Add more comprehensive health checks (verifying DB connections, ChromaDB availability).
- **Need Rating:** 5/5

## `.\\cache\\chroma_memory_store.py`
- **Contains:** ChromaDB-powered persistent semantic memory system. Stores high-confidence past interactions and retrieves them for future similar queries.
- **Functions:**
  - `store_memory`: Confidence-gated (≥ 0.7) memory persistence. Embeds the **query** (not the answer) as the primary vector for accurate semantic retrieval. Performs near-duplicate detection (L2 distance < 0.1) before storing. Fires `_cleanup_memory` asynchronously in a background thread to avoid blocking the API.
  - `retrieve_memory`: Embeds the incoming query and performs a k-NN search against stored query embeddings. Filters results by `confidence ≥ 0.65` or `verified == True`.
  - `verify_memory`: Updates `verified=True` or deletes entries based on user feedback via the `/feedback` endpoint.
  - `_cleanup_memory`: Enforces the 200-item capacity cap by deleting the oldest entries (sorted by timestamp). Runs in a background thread.
  - `cleanup_old_memory`: TTL-based cleanup — deletes any memory older than 7 days.
- **Purpose in Application:** Provides persistent semantic memory that survives server restarts. Dramatically reduces hallucination and latency for similar future queries by injecting verified past answers into the LLM context.
- **Places to Improve:** The deduplication L2 threshold (`< 0.1`) is hardcoded — should be exposed as a configurable parameter in `config_manager.py`.
- **Need Rating:** 5/5

## `.\\cache\\memory_store.py`
- **Contains:** ⚠️ **DEPRECATED** — Legacy in-memory store using raw numpy cosine similarity. Explicitly marked as deprecated at the top of the file and **not imported or called anywhere in the active pipeline.**
- **Functions:**
  - `cosine_sim`: Manual cosine similarity via numpy dot products.
  - `store_memory`, `verify_memory`, `retrieve_memory`: Original memory lifecycle methods, fully superseded by `chroma_memory_store.py`.
- **Purpose in Application:** None currently. Was the original prototype before ChromaDB was introduced.
- **Places to Improve:** Safe to delete — no active references exist in the codebase.
- **Need Rating:** 0/5 (Deprecated — candidate for removal)

## `.\\cache\\query_cache.py`
- **Contains:** Caching layer specifically for query results to avoid redundant LLM calls.
- **Functions:**
  - `get_cache_key`: Generates an MD5 hash for a given query string.
  - `get_cached`: Retrieves existing RAG answers from the in-memory dict.
  - `set_cache`: Stores generated answers for future queries.
- **Purpose in Application:** Reduces latency and LLM costs by returning pre-computed answers for identical queries. Short-circuits the entire pipeline on a hit.
- **Places to Improve:** Upgrade to semantic similarity caching to catch paraphrased queries (not just exact matches).
- **Need Rating:** 4/5

## `.\\cache\\redis_cache.py`
- **Contains:** Redis client configuration and connection handling stub.
- **Functions:** *(No active functions — placeholder for future remote cache.)*
- **Purpose in Application:** Intended as the backend for a distributed cache that would replace the in-memory dict across multiple server instances.
- **Places to Improve:** Implement fully and wire into `query_cache.py` to enable multi-instance shared caching.
- **Need Rating:** 2/5

## `.\\chroma_db\\`
- **Contains:** Persistent on-disk ChromaDB storage directory.
- **Functions:** *(Internal ChromaDB binary/SQLite data.)*
- **Purpose in Application:** Stores the embedded document chunks (knowledge base) and the semantic memory collection (`memory_store`).
- **Places to Improve:** Regularly back up the SQLite file to prevent index corruption.
- **Need Rating:** 5/5

## `.\\configs\\system_config.yaml`
- **Contains:** YAML file that persists the active control-plane configuration to disk. Written by `config_manager.py` on every parameter change; read on startup to restore learned routing state.
- **Functions:** *(Data file — no functions.)*
- **Purpose in Application:** Solves the config amnesia problem — the optimizer's learned parameters now survive server restarts.
- **Places to Improve:** Add schema validation so malformed YAML doesn't silently fall back to defaults.
- **Need Rating:** 4/5

## `.\\control_plane\\config_manager.py`
- **Contains:** Thread-safe singleton configuration manager with full disk persistence.
- **Functions:**
  - `_init_config`: Initializes default config values and immediately calls `_load_from_disk()` to restore any previously saved state.
  - `_load_from_disk`: Reads `configs/system_config.yaml` and merges saved values over defaults.
  - `_save_to_disk`: Writes the current config to YAML after every write operation.
  - `get_config`, `get_param`, `get_version`: Thread-safe read methods using `threading.Lock`.
  - `update_config`, `set_param`, `reset_config`, `smart_update`: Thread-safe write methods — each one calls `_save_to_disk()` to persist state immediately.
  - `dump_config`: Returns version + config snapshot for the debug API endpoint.
- **Purpose in Application:** Ensures safe, concurrent global access to dynamic routing parameters. The optimizer's learned config now persists across restarts via YAML-backed storage.
- **Places to Improve:** The deduplication threshold for `chroma_memory_store` should be moved here as a first-class config parameter.
- **Need Rating:** 5/5

## `.\\control_plane\\knowledge_router.py`
- **Contains:** Decision logic to route each query to the correct knowledge retrieval mode.
- **Functions:**
  - `route_knowledge`: Returns `LLM_ONLY`, `RAG`, or `HYBRID` based on query features extracted from `extract_features()`. Coding queries bypass retrieval; high-ambiguity queries use RAG; general queries and complex reasoning use HYBRID.
- **Purpose in Application:** Enables the system to skip expensive retrieval operations for queries that don't need them (coding, simple factual) and route appropriately for those that do.
- **Places to Improve:** Could handle multi-domain queries by returning a ranked list of modes rather than a single one.
- **Need Rating:** 4/5

## `.\\control_plane\\model_router.py`
- **Contains:** Multi-strategy LLM selection with epsilon-greedy exploration.
- **Functions:**
  - `extract_features`: Extracts numerical feature vector from `query_analysis` (length, type, complexity score, ambiguity, multi-hop flag, code signal).
  - `route_by_task`: Fast rule-based routing — coding queries → `deepseek-coder`, reasoning/multi-hop → `qwen2.5:3b`, simple → `phi3`.
  - `score_model`: Scores each model in `MODEL_PROFILES` against the extracted features, applying cost and latency penalties.
  - `route_model_advanced`: Checks control-plane override first, then tries task routing, then falls back to scoring.
  - `route_model_with_exploration`: Wraps `route_model_advanced` with epsilon-greedy exploration (10% random model selection for data collection).
  - `get_primary_model`, `get_fallback_model`: Pipeline entry points.
- **Purpose in Application:** The self-optimizing engine's routing intelligence. Dynamically shifts traffic across models based on task and performance data.
- **Places to Improve:** `route_model_with_exploration` could incorporate historical evaluation scores into the exploration weighting (true contextual bandit).
- **Need Rating:** 5/5

## `.\\dashboard\\app.py`
- **Contains:** Streamlit observability dashboard that reads `experiments.db`.
- **Functions:**
  - `load_data`: Loads experiment data from SQLite for display.
- **Purpose in Application:** Provides a visual interface to monitor the system's performance, evaluation metrics, and model selections over time.
- **Places to Improve:** Make data loading asynchronous to prevent freezing on large datasets. Add cost tracking charts sourced from `cost_log.json`.
- **Need Rating:** 3/5

## `.\\embeddings\\embedding_service.py`
- **Contains:** Centralized embedding wrapper using `all-MiniLM-L6-v2`.
- **Functions:**
  - `embed_text`: Takes raw text and returns a 384-dimensional vector using `SentenceTransformer`.
- **Purpose in Application:** Single source of truth for all embedding operations — used by `chroma_store`, `chroma_memory_store`, and `keyword_retriever`.
- **Places to Improve:** Add batching support for multi-document embedding during ingestion. Add a fallback embedding model.
- **Need Rating:** 5/5

## `.\\evaluation\\confidence_model.py`
- **Contains:** Unified, content-aware confidence scoring for RAG and LLM-only responses.
- **Functions:**
  - `compute_confidence`: Calculates a 0.0–1.0 confidence score. In **RAG mode**: uses sigmoid-normalized reranker scores + memory boost − complexity penalty, then multiplied by evaluation score. In **LLM-only mode**: starts at 0.5 base, applies content-aware penalties (uncertainty phrases, short length, missing code structure for coding queries) and evaluation score as a strict multiplier.
- **Purpose in Application:** Provides the optimization and memory layers with a reliable signal of answer quality. Used to gate memory storage (≥ 0.7) and to drive the feedback loop.
- **Places to Improve:** Add integration with an LLM-as-judge for more semantically robust faithfulness scoring. The `sqrt()` normalization was deliberately removed to prevent artificial confidence inflation — this decision is documented in comments.
- **Need Rating:** 5/5

## `.\\evaluation\\rag_evaluator.py`
- **Contains:** Online evaluation of RAG responses using cosine similarity.
- **Functions:**
  - `evaluate_rag`: Embeds question, answer, and context. Computes `answer_relevance` (question ↔ answer similarity), `faithfulness` (answer ↔ context similarity), and `hallucination_score` (1 − faithfulness).
- **Purpose in Application:** Quantifies response quality so the optimization layer can make data-driven routing decisions. Returns a dict with all three metrics.
- **Places to Improve:** Add LLM-based judge for more granular faithfulness evaluation. Current cosine similarity is a fast but imprecise proxy for factual grounding.
- **Need Rating:** 5/5

## `.\\experiments.db`
- **Contains:** SQLite database storing evaluation metrics per query. Uses WAL journal mode and `timeout=30` for improved concurrency. Schema: `id`, `timestamp`, `question`, `config` (JSON), `answer_relevance`, `faithfulness`.
- **Functions:** *(Database file.)*
- **Purpose in Application:** The persistence layer for the self-optimization loop. Enables `get_best_config()` to return historically top-performing parameter sets.
- **Places to Improve:** Add caching to `get_best_config()` so it doesn't re-query the DB on every request. Move to PostgreSQL if concurrent writer load increases beyond a single server.
- **Need Rating:** 5/5

## `.\\ingestion\\chunker.py`
- **Contains:** Document chunking logic for pre-processing text before embedding.
- **Functions:**
  - `chunk_document`: Splits text into fixed-size chunks based on token/character limits.
- **Purpose in Application:** Prevents context window overflows and ensures retrieval hits precise, relevant sections of a document.
- **Places to Improve:** Implement semantic/paragraph-aware chunking. Add configurable chunk overlap to reduce context boundary issues.
- **Need Rating:** 4/5

## `.\\llm\\llm_service.py`
- **Contains:** The primary interface to the local Ollama LLM endpoint. Now includes cost tracking integration.
- **Functions:**
  - `clean_output`: Strips special tokens and artifacts from raw LLM output.
  - `format_code_block`: Wraps coding answers in proper markdown code fences.
  - `_call_llm`: Posts to `http://localhost:11434/api/generate`, records token usage and estimated cost via `cost_tracker.record()` after every call.
  - `estimate_confidence`: Simple heuristic confidence check on LLM output (length, uncertainty phrases).
  - `generate_answer`: Orchestrates prompt building, primary LLM call, confidence-gated fallback to a different model (using `get_fallback_model`), and output formatting.
- **Purpose in Application:** The generation engine for the entire RAG pipeline.
- **Places to Improve:** Add `tenacity`-based retry logic for transient Ollama connection failures. Add streaming support at this layer.
- **Need Rating:** 5/5

## `.\\llm\\model_registry.py`
- **Contains:** Model definitions, capability profiles, and selection logic.
- **Functions:**
  - `get_model`: Looks up a model name by task category from `MODEL_REGISTRY`.
  - `map_task_to_category`: Maps task strings to registry categories.
  - `select_best_model`: Scores all models in `MODEL_PROFILES` by task type (penalizing latency and cost) and returns the best.
  - `get_best_model_for_task`: Combines category lookup and scoring for a final model decision.
- **Purpose in Application:** Decouples model definitions from routing logic. Adding a new LLM only requires updating `MODEL_REGISTRY` and `MODEL_PROFILES`.
- **Places to Improve:** Dynamically load model availability from the Ollama API rather than a static registry.
- **Need Rating:** 4/5

## `.\\llm\\prompt_builder.py`
- **Contains:** Prompt templating and context management.
- **Functions:**
  - `trim_context`: Fits retrieved documents into the context window limit.
  - `get_verbosity`: Determines answer detail level based on query type and analysis.
  - `build_code_prompt`, `build_rag_prompt`, `build_general_prompt`, `build_json_prompt`: Specialized templates for coding, RAG grounded, general, and structured JSON responses.
  - `build_prompt`: Main router — selects the correct template, trims context, and builds the final prompt string.
- **Purpose in Application:** Ensures the LLM receives properly structured, context-appropriate instructions for every query type.
- **Places to Improve:** `trim_context` should prioritize highest-ranked documents rather than arbitrary cutoff. Add a system-prompt injection point for persona customization.
- **Need Rating:** 5/5

## `.\\logs\\rag_logs.json`
- **Contains:** Persistent structured query log.
- **Functions:** *(JSON data file.)*
- **Purpose in Application:** Human-readable record of query inputs, results, and generated metrics.
- **Places to Improve:** Add log rotation to prevent unbounded disk growth.
- **Need Rating:** 3/5

## `.\\observability\\cost_tracker.py`
- **Contains:** Thread-safe, persistent LLM token usage and cost tracker.
- **Functions:**
  - `record`: Records a single LLM call — computes cost from `MODEL_PRICING` table, accumulates session totals (`_session_*` counters), per-model breakdown (`_by_model`), and persists all-time totals to `logs/cost_log.json` atomically (via `.tmp` → rename).
  - `get_session_summary`: Returns session-scoped totals and per-model breakdown.
  - `get_all_time_totals`: Returns persisted all-time totals from `cost_log.json`.
  - `reset_session`: Clears in-memory session counters (does not touch the persisted file).
  - `estimate_tokens`: Lightweight token estimator (~4 chars/token), used when tiktoken is unavailable.
  - `_load_totals`, `_save_totals`: Disk persistence — loads existing totals on startup, writes atomically on every `record` call.
- **Purpose in Application:** Tracks token consumption and estimated USD cost per model, per session, and all-time. Wired into `llm_service._call_llm()` — every LLM call is automatically recorded.
- **Places to Improve:** Expose session cost summary via the API observability payload. Add alerting for budget thresholds.
- **Need Rating:** 5/5

## `.\\observability\\latency_tracker.py`
- **Contains:** Lightweight per-stage latency measurement tool.
- **Functions:**
  - `__init__`: Creates an empty `times` dict.
  - `start(key)`: Records the start timestamp for a named stage.
  - `end(key)`: Replaces the start timestamp with the elapsed duration.
  - `get_all()`: Returns the full timing dict for all tracked stages.
- **Purpose in Application:** Enables per-stage pipeline profiling (retrieval, rerank, LLM) returned in every API response's observability payload.
- **Places to Improve:** Could be implemented as a Python decorator `@track_latency` for cleaner integration. Add p50/p95 tracking across multiple requests.
- **Need Rating:** 3/5

## `.\\optimization\\experiment_db.py`
- **Contains:** SQLite interaction layer for experiment logging and config retrieval. Now includes WAL mode and concurrency optimizations.
- **Functions:**
  - `get_db_connection`: Returns a SQLite connection with `PRAGMA journal_mode=WAL` and `PRAGMA synchronous=NORMAL` for improved concurrency, plus `timeout=30` to prevent lock errors under load.
  - `init_db`: Initializes the `experiments` table schema on startup.
  - `log_experiment`: Inserts a row with timestamp, question, config (JSON), `answer_relevance`, and `faithfulness`.
  - `get_best_config`: Queries for the config with the highest average combined score. Uses an in-process cache (`_best_config_cache`) to avoid redundant DB queries.
- **Purpose in Application:** The persistence and query layer for the self-optimization engine.
- **Places to Improve:** The `_best_config_cache` is never invalidated after new experiments are logged — add TTL-based cache invalidation.
- **Need Rating:** 5/5

## `.\\optimization\\optimizer.py`
- **Contains:** The self-optimization decision engine.
- **Functions:**
  - `choose_config`: Queries `experiment_db` for the best-known config. Falls back to a random choice from `DEFAULT_CONFIGS` if no history exists.
  - `adapt_config`: Analyzes current relevance/faithfulness scores and reactively adjusts `top_k` (if faithfulness < 0.5) or `chunk_size` (if relevance < 0.5), then calls `config_manager.update_config()` to persist the change immediately.
- **Purpose in Application:** Drives the "self-optimizing" core — adjusts retrieval parameters based on observed performance. Changes are persisted via `config_manager` to `system_config.yaml`.
- **Places to Improve:** `adapt_config` uses fixed thresholds — should require a statistical minimum sample size before adjusting globally.
- **Need Rating:** 5/5

## `.\\query_processing\\query_analyzer.py`
- **Contains:** NLP-based query classification and feature extraction.
- **Functions:**
  - `analyze_query`: Extracts `type` (coding/reasoning/general), `complexity` (low/medium/high), `keywords`, `is_multi_hop`, `ambiguity`, `length`, and `needs_keyword`/`needs_semantic` signals from the raw query.
- **Purpose in Application:** Powers all downstream routing decisions in the control plane and determines whether multi-hop decomposition is needed.
- **Places to Improve:** Add query rewriting (expanding abbreviations, correcting typos) to improve retrieval recall.
- **Need Rating:** 4/5

## `.\\query_processing\\query_decomposer.py`
- **Contains:** LLM-based query decomposition with rule-based fallback.
- **Functions:**
  - `decompose_query_llm`: Sends the query to Ollama with a structured prompt to receive a JSON list of atomic sub-questions.
  - `fallback_decomposition`: Rule-based mechanical splitting on connectors like `compare`, `and`, `then`.
- **Purpose in Application:** Breaks complex multi-part queries into independent retrievable sub-questions, enabling multi-hop reasoning.
- **Places to Improve:** Expand fallback rules. Refine the LLM prompt to return more consistent JSON structure.
- **Need Rating:** 5/5

## `.\\query_processing\\query_planner.py`
- **Contains:** Orchestrator for multi-hop query planning.
- **Functions:**
  - `plan_query`: Checks `enable_multi_hop` and `enable_decomposition` from config, and `is_multi_hop` from query analysis. Returns a list of sub-queries (or a single-element list for simple queries).
- **Purpose in Application:** Decides whether to decompose a query before retrieval. Respects control-plane feature flags for dynamic disabling.
- **Places to Improve:** Could support dynamic execution graph planning where sub-queries have dependencies.
- **Need Rating:** 5/5

## `.\\retrieval\\hybrid_retriever.py`
- **Contains:** Merges semantic vector search and BM25 keyword search results.
- **Functions:**
  - `hybrid_search`: Runs both `search_documents` (ChromaDB) and `keyword_search` (BM25) in parallel, deduplicates, and merges rankings using Reciprocal Rank Fusion (RRF) before returning combined results.
- **Purpose in Application:** Provides best-of-both-worlds retrieval — semantic understanding from vector search AND exact-match precision from BM25.
- **Places to Improve:** The keyword/semantic weighting could be dynamically tuned by the optimizer based on query type.
- **Need Rating:** 5/5

## `.\\retrieval\\keyword_retriever.py`
- **Contains:** BM25 keyword search implementation.
- **Functions:**
  - `build_index`: Creates a `BM25Okapi` index from a list of document strings. Called on startup.
  - `rebuild_from_chroma`: Re-fetches all documents from ChromaDB and rebuilds the BM25 index. Triggered by `POST /index/rebuild`.
  - `keyword_search`: Tokenizes the query and returns top-k BM25-ranked documents.
- **Purpose in Application:** Ensures exact entity names, acronyms, and IDs are retrieved reliably — cases where vector search typically underperforms.
- **Places to Improve:** `rebuild_from_chroma` is a blocking operation — should run asynchronously. Serialize the BM25 index to disk to avoid rebuilding on restart.
- **Need Rating:** 4/5

## `.\\retrieval\\pipeline.py`
- **Contains:** The complete RAG pipeline orchestrator — the single source of truth for how a query becomes an answer.
- **Functions:**
  - `normalize_documents`: Normalizes retrieved docs from tuples, dicts, or raw strings into a uniform list of strings.
  - `rag_pipeline`: Full pipeline execution: latency tracking → query analysis → cache check → model routing → knowledge routing. Branches into **LLM_ONLY**, **FALLBACK_LLM** (empty retrieval), or **HYBRID/RAG** flows. Each flow handles retrieval, memory fusion, LLM generation, confidence computation, memory storage, and async evaluation logging. Returns `(answer, observability_dict)`.
- **Purpose in Application:** Ties the entire architecture together. All routing, retrieval, generation, evaluation, and memory persistence flow through this single function.
- **Places to Improve:** Extract branch logic into dedicated flow handlers for cleaner testing. Add agentic escalation for edge-case hallucination detection.
- **Need Rating:** 5/5

## `.\\retrieval\\reranker.py`
- **Contains:** Cross-encoder based re-scoring of retrieved documents.
- **Functions:**
  - `rerank`: Takes a query and list of candidate documents, scores each `(query, doc)` pair with `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`, sorts by score descending, and returns top-k results.
- **Purpose in Application:** Second-stage precision boost — after fast hybrid retrieval returns ~20 candidates, the cross-encoder re-scores with full attention over query+document pairs for maximum relevance.
- **Places to Improve:** Add a hard timeout to `rerank` to cap latency. Consider a lighter model for large candidate pools.
- **Need Rating:** 4/5

## `.\\scripts\\ingest_documents.py`
- **Contains:** One-time data ingestion script.
- **Functions:** *(Standalone script operations.)*
- **Purpose in Application:** Reads `.txt` files from the `data/` directory, chunks them via `chunk_document`, embeds them, and loads them into the ChromaDB `documents` collection.
- **Places to Improve:** Add chunk overlap support. Support formats beyond `.txt` (PDF, DOCX, HTML). Add concurrent ingestion for large document sets.
- **Need Rating:** 5/5

## `.\\summary_data.json`
- **Contains:** Pre-aggregated metrics snapshot for dashboard display.
- **Purpose in Application:** Enables fast dashboard page loads without querying SQLite on every render.
- **Need Rating:** 4/5

## `.\\tests\\`
- **Contains:** Full pytest test suite with 43 tests across 8 modules. Covers the entire stack — API routes, pipeline flows, config management, memory store, LLM service, model router, prompt builder, and query analyzer.
- **Key Files:**
  - `conftest.py`: Session-scoped `sys.modules` mocking for `sentence_transformers` and `chromadb` (enables CI without GPU/local dependencies). Provides `mock_db_session`, `mock_llm`, `mock_retriever`, and `mock_config_manager` (autouse, prevents cross-test config contamination) fixtures.
  - `pipeline/test_pipeline.py`: Tests all 4 pipeline branches — normal RAG flow, cache hit short-circuit, LLM-only mode, and empty retrieval fallback to FALLBACK_LLM.
  - `api/test_query_routes.py`, `cache/test_chroma_memory_store.py`, `config/test_config_manager.py`, `llm/test_llm_service.py`, `llm/test_model_router.py`, `llm/test_prompt_builder.py`, `query_processing/test_query_analyzer.py`, `retrieval/test_hybrid_retriever.py`.
- **Need Rating:** 5/5

## `.\\ui\\streamlit_app.py`
- **Contains:** The primary user-facing Streamlit interface.
- **Functions:**
  - `render_answer`: Displays the LLM payload, parsed observability metrics, confidence score, and latency breakdown.
  - `normalize`: Standardizes input variables for UI display formatting.
- **Purpose in Application:** The primary interaction surface for end-users to query the system and view response quality metadata.
- **Places to Improve:** Streamlit's state model makes complex feedback UX difficult. Moving to a React-based frontend would improve the feedback experience significantly.
- **Need Rating:** 5/5

## `.\\vectorstore\\chroma_store.py`
- **Contains:** ChromaDB integration for the primary document knowledge base.
- **Functions:**
  - `add_documents`: Embeds and loads document chunks into the `documents` collection.
  - `search_documents`: Executes k-NN vector similarity search, returning top-k matching chunks.
- **Purpose in Application:** The core semantic search backbone. Every RAG and HYBRID mode query hits this store.
- **Places to Improve:** Add metadata filtering to `search_documents` to enable domain-specific retrieval routing.
- **Need Rating:** 5/5

## `.\\workers\\evaluation_worker.py`
- **Contains:** Background evaluation and adaptation worker.
- **Functions:**
  - `run_evaluation_async`: Spawns a daemon thread that: evaluates the answer via `evaluate_rag`, logs the result to SQLite via `log_experiment`, computes a weighted confidence score, stores memory via `chroma_memory_store.store_memory`, and calls `adapt_config` to potentially update retrieval parameters in real-time.
- **Purpose in Application:** Decouples post-response evaluation from the user-facing API. Ensures heavy evaluation logging doesn't add latency to the streamed response. Also drives the real-time self-optimization feedback loop via `adapt_config`.
- **Places to Improve:** Replace the bare `threading.Thread` with a proper message queue (Celery/RabbitMQ) for scale. Consider a dead-letter queue for failed evaluation tasks.
- **Need Rating:** 4/5

---

## **Overview**

The **Self-Optimizing RAG** application is a highly advanced, modular framework built for retrieving and intelligently reasoning over data. The system is broken down into distinct layers:

1. **Ingestion & Vectorization:** Scripts chunk text (`chunker.py`) and generate semantic embeddings (`embedding_service.py`) for storage in ChromaDB (`chroma_store.py`). Handled locally by `ingest_documents.py`.
2. **Retrieval Engine:** Utilizes a robust hybrid approach — BM25 keyword matching (`keyword_retriever.py`) and semantic vector search — merged by Reciprocal Rank Fusion in `hybrid_retriever.py`, then re-scored by a cross-encoder (`reranker.py`). Multi-hop decomposition enables handling complex logical reasoning chains (`query_planner.py`).
3. **Core API & Control Plane:** The FastAPI backend exposes the pipeline while the control plane dynamically routes traffic to the most appropriate knowledge mode (`knowledge_router.py`) or LLM provider (`model_router.py`). All config state is managed by `config_manager.py` with full YAML-backed persistence.
4. **Self-Optimization & Observability:** The defining feature of the codebase. Latency tracking (`latency_tracker.py`), cost tracking (`cost_tracker.py`), confidence scoring (`confidence_model.py`), and SQLite telemetry (`evaluation_worker.py`, `experiment_db.py`) feed the `optimizer.py` — which reactively adjusts retrieval parameters and persists them across restarts.
5. **Presentation Layer:** A Streamlit chat interface (`streamlit_app.py`) and an observability dashboard (`dashboard/app.py`) built on top of the API.

---

### System Data Flow & Request Lifecycle
When a user submits a query through the UI (`ui\streamlit_app.py`) or the HTTP endpoints (`api\routes\query_routes.py`), the request undergoes the following lifecycle:

1. **Cache Check:** `query_cache.py` validates if this exact response is available in fast lookup paths.
2. **Query Analysis & Planning:** `query_analyzer.py` determines topic intent; `query_planner.py` splits the prompt if it is deemed a complex multi-hop sequence.
3. **Control Plane Routing:** `knowledge_router.py` picks a retrieval mode; `model_router.py` selects the most appropriate LLM using epsilon-greedy exploration. Settings are locked via `config_manager.py` (YAML-persisted).
4. **Retrieval & Re-ranking:** ChromaDB k-NN search via `chroma_store.py` and BM25 via `keyword_retriever.py` merge at `hybrid_retriever.py`, then pass through `reranker.py` for cross-encoder precision scoring.
5. **Context Memory & Fusion:** `chroma_memory_store.py` retrieves high-confidence verified past answers and fuses them with retrieved chunks.
6. **LLM Generation:** `prompt_builder.py` selects the correct template and trims context. `llm_service.py` posts to Ollama and automatically records token usage via `cost_tracker`.
7. **Offline Evaluation & Retention:** `pipeline.py` synchronously stores high-confidence answers in `chroma_memory_store`. `evaluation_worker.py` asynchronously logs SQLite telemetry and calls `adapt_config` to update system parameters — changes persist immediately to `system_config.yaml`.

### Environment & Persistence Context
- `data/*`: Raw `.txt` document drops for ingestion.
- `chroma_db/*`: On-disk vector index for documents AND semantic memory.
- `experiments.db`: SQLite tracker (WAL mode) for config × evaluation history.
- `configs/system_config.yaml`: Persisted control-plane config — the optimizer's learned state.
- `logs/cost_log.json`: Persisted all-time token usage and cost totals.
- `summary_data.json`: Pre-aggregated metrics for dashboard display.

---

### Logical Component Map

* **Entry Points & UI:** `app\main.py`, `api\routes\query_routes.py`, `ui\streamlit_app.py`, `dashboard\app.py`
* **Control Plane (Orchestration):** `control_plane\config_manager.py`, `control_plane\knowledge_router.py`, `control_plane\model_router.py`
* **Query Processing Pipeline:** `query_processing\query_analyzer.py`, `query_processing\query_decomposer.py`, `query_processing\query_planner.py`, `retrieval\pipeline.py`
* **Retrieval & Vector Logic:** `retrieval\hybrid_retriever.py`, `retrieval\keyword_retriever.py`, `retrieval\reranker.py`, `vectorstore\chroma_store.py`
* **LLM Integrations:** `llm\llm_service.py`, `llm\model_registry.py`, `llm\prompt_builder.py`
* **Optimization & Data Collection:** `optimization\optimizer.py`, `optimization\experiment_db.py`, `workers\evaluation_worker.py`, `evaluation\rag_evaluator.py`, `evaluation\confidence_model.py`
* **Data Ingestion:** `ingestion\chunker.py`, `embeddings\embedding_service.py`, `scripts\ingest_documents.py`
* **Caching & Memory:** `cache\chroma_memory_store.py`, `cache\query_cache.py`, `cache\redis_cache.py`
* **Observability:** `observability\latency_tracker.py`, `observability\cost_tracker.py`
* **Testing:** `tests\conftest.py` + 8 test modules (43 tests)

### Technology Integrations
* **FastAPI / Uvicorn** → `app\main.py` and `api\routes\query_routes.py`
* **ChromaDB** → `vectorstore\chroma_store.py` and `cache\chroma_memory_store.py`
* **Sentence-Transformers** → `embeddings\embedding_service.py` and `evaluation\rag_evaluator.py`
* **BM25Okapi (rank_bm25)** → `retrieval\keyword_retriever.py`
* **Cross-Encoder Model** → `retrieval\reranker.py`
* **Streamlit** → `ui\streamlit_app.py` and `dashboard\app.py`
* **Ollama (Local LLM Server)** → `llm\llm_service.py`
* **SQLite (WAL mode)** → `optimization\experiment_db.py`
* **PyYAML** → `control_plane\config_manager.py`
* **pytest + pytest-mock** → `tests\`

### Strategic Roadmap
Based on internal code reviews, the architecture is targeting the following major technical epics:
1. **Asynchronous Architecture Migration:** Transition intensive synchronous processes (reranking, BM25 rebuilding) to robust async brokers like **Celery** or **RabbitMQ** for stable API latencies under load.
2. **Semantic Query Caching:** Upgrade the current exact-match (`query_cache.py`) to vector-similarity caching, allowing high-confidence overlaps to bypass retrieval for equivalent user intents.
3. **LLM-as-Judge Evaluation:** Replace cosine similarity in `rag_evaluator.py` with a second LLM grading faithfulness and relevance for more semantically robust evaluation signals.
4. **Config Parameter Exposure:** Move hardcoded values (deduplication threshold, memory TTL, scoring weights) into `config_manager.py` as first-class dynamic parameters.
