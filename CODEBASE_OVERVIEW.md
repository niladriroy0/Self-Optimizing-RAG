# Codebase Overview: Self-Optimizing RAG

## `.\api\routes\query_routes.py`
- **Contains:** FastAPI routes for handling queries, feedback, index rebuilding, and streaming responses. Includes debugging endpoints for active configurations.
- **Functions:**
  - `stream_response`: Streams the LLM response back to the client word-by-word.
  - `query`: Handles the main RAG querying logic.
  - `rebuild_index`: Triggers a background job to rebuild the search index.
  - `feedback`: Accepts user feedback on the quality of the answer.
  - `get_config`: Debug endpoint returning a snapshot of the active control-plane config.
- **Purpose in Application:** Serves as the main entry point to the backend for the Streamlit UI and any other API clients. It glues the core RAG pipeline with HTTP requests.
- **Places to Improve:** Move heavy computation out of route handlers and into separate background workers. Add authentication/rate-limiting to the routes. `stream_response` could handle WebSocket connections for more robust streaming.
- **Need Rating:** 5/5

## `.\app\main.py`
- **Contains:** FastAPI application setup, global middleware, and lifespan managers.
- **Functions:**
  - `lifespan`: Handles startup and shutdown events (e.g., initializing databases, cleaning up caches).
  - `health`: Simple endpoint to check if the API is alive.
- **Purpose in Application:** Initializes and starts the FastAPI server. Essential for bootstrapping the backend.
- **Places to Improve:** Could add more comprehensive health checks (e.g., verifying DB connections, checking memory usage).
- **Need Rating:** 5/5

## `.\cache\chroma_memory_store.py`
- **Contains:** ChromaDB-powered persistent memory system allowing the RAG system to learn confident past interactions.
- **Functions:**
  - `store_memory`: Saves context into a Chroma collection for high-confidence LLM answers.
  - `retrieve_memory`: Fetches relevant memory for the current query if vector dots match closely.
  - `verify_memory`: Manual intervention or rule-based updates validating past memory.
  - `_cleanup_memory`, `cleanup_old_memory`: Enforces TTL and strict bounding box (e.g., 200 items max) to prevent unbounded memory growth.
- **Purpose in Application:** Radically cuts down hallucination and response time by instantly providing semantic 'past successful' memory context into the builder prompt.
- **Places to Improve:** Optimize the cleanup functions to run asynchronously instead of halting the store operation execution.
- **Need Rating:** 5/5

## `.\cache\memory_store.py`
- **Contains:** ⚠️ **DEPRECATED** — Legacy in-memory store using raw numpy cosine similarity. Explicitly marked as deprecated at the top of the file and **not imported or called anywhere in the active pipeline.**
- **Functions:**
  - `cosine_sim`: Manual cosine similarity via numpy dot products.
  - `store_memory`, `verify_memory`, `retrieve_memory`: Original memory lifecycle methods, fully superseded by `chroma_memory_store.py`.
- **Purpose in Application:** None currently. Was the original prototype before ChromaDB was introduced.
- **Places to Improve:** Safe to delete — no active references exist in the codebase.
- **Need Rating:** 0/5 (Deprecated — candidate for removal)


## `.\cache\query_cache.py`
- **Contains:** Caching layer specifically for query results to avoid redundant LLM calls.
- **Functions:**
  - `get_cache_key`: Generates a deterministic hash or key for a given semantic query.
  - `get_cached`: Retrieves existing RAG answers.
  - `set_cache`: Stores generated answers for future queries.
- **Purpose in Application:** Reduces latency and LLM costs by returning pre-computed answers for frequent or exact-match queries.
- **Places to Improve:** Semantic caching (matching "similar" queries not just exact ones) would be a huge optimization. (Addressed somewhat by `chroma_memory_store`).
- **Need Rating:** 4/5

## `.\cache\redis_cache.py`
- **Contains:** Redis client configuration and connection handling.
- **Functions:** *(None defined as top-level functions, likely uses a class or global client).*
- **Purpose in Application:** Provides the backend implementation for the caching mechanisms mentioned above.
- **Places to Improve:** Add fallback mechanisms if Redis is down, or implement connection pooling.
- **Need Rating:** 3/5

## `.\chroma_db\`
- **Contains:** Persistent storage directory for ChromaDB vector index.
- **Functions:** *(Internal ChromaDB binary data).*
- **Purpose in Application:** Maintains the on-disk embeddings for the application.
- **Places to Improve:** Regularly back up the SQLite database to prevent index corruption.
- **Need Rating:** 5/5

## `.\configs\system_config.yaml`
- **Contains:** Empty configuration YAML file.
- **Functions:** *(None).*
- **Purpose in Application:** Intended for loading system-level configuration parameters outside of code.
- **Places to Improve:** Populate and wire to `config_manager.py`.
- **Need Rating:** 1/5

## `.\control_plane\config_manager.py`
- **Contains:** Central singleton control plane configuration manager.
- **Functions:**
  - `_init_config`: Initializes default system configuration values (e.g., `top_k`, `chunk_size`).
  - `get_config`, `get_param`, `get_version`: Read methods with thread-safety locks.
  - `update_config`, `set_param`, `reset_config`, `smart_update`: Write methods that alter the active config and track versioning safely.
- **Purpose in Application:** Ensures safe, consistent, and concurrent global access to dynamic parameters routing decisions in the RAG pipeline.
- **Places to Improve:** The `ConfigManager` configuration state resets on server restart; it should sync state back to a persistent storage or database.
- **Need Rating:** 5/5

## `.\control_plane\knowledge_router.py`
- **Contains:** Logic to decide which knowledge base or index to query based on the user's input.
- **Functions:**
  - `route_knowledge`: Analyzes the query and routes it to the correct downstream retriever.
- **Purpose in Application:** Acts as an orchestrator for multiple domains of knowledge, enabling specialized retrieval.
- **Places to Improve:** The routing logic could be improved to handle multi-domain queries by splitting them instead of just picking one.
- **Need Rating:** 4/5

## `.\control_plane\model_router.py`
- **Contains:** Logic to dynamically select the best LLM or routing strategy for a specific query.
- **Functions:**
  - `extract_features`: Extracts features from the query to decide routing.
  - `score_model`: Scores different available LLMs based on cost, latency, or capability.
  - `route_model_advanced`: Selects the best LLM using the extracted features and scores.
  - `route_model_with_exploration`: Sporadically routes to different models to collect evaluation data (exploration vs exploitation).
- **Purpose in Application:** Implements the "Self-Optimizing" part of the application by dynamically shifting traffic across models based on performance.
- **Places to Improve:** `route_model_with_exploration` could implement multi-armed bandit strategies more robustly for optimal learning.
- **Need Rating:** 5/5

## `.\dashboard\app.py`
- **Contains:** Code for an observability or monitoring dashboard.
- **Functions:**
  - `load_data`: Loads experiment or evaluation data into the dashboard.
- **Purpose in Application:** Provides a visual interface to monitor the system's performance, cost, and evaluation metrics.
- **Places to Improve:** Make loading asynchronous to prevent dashboard freezing on large datasets.
- **Need Rating:** 3/5

## `.\embeddings\embedding_service.py`
- **Contains:** Wrapper for generating embeddings of text.
- **Functions:**
  - `embed_text`: Takes raw text and converts it into a vector representation.
- **Purpose in Application:** Essential for indexing documents, building confidence scoring matrices, and managing memory stores via `all-MiniLM-L6-v2`.
- **Places to Improve:** Implement batching to `embed_text` to handle multiple documents faster during ingestion. Add fallback embedding models.
- **Need Rating:** 5/5

## `.\evaluation\confidence_model.py`
- **Contains:** Unified confidence scoring logic for evaluating system responses.
- **Functions:**
  - `compute_confidence`: Calculates an overall confidence score (0.0 to 1.0) using retrieval strength, memory boosting, complexity penalties, and evaluation scores. Also features context-awareness for LLM-Only modes, penalizing AI uncertainty language.
- **Purpose in Application:** Gives the self-optimizing engine a numerical representation of certainty. This affects caching, fallback actions, and dynamic route choices.
- **Places to Improve:** Add an integration with an ensemble model. Expand normalization mechanisms (currently defaults to min-max over top-k scores which could be sensitive to outliers).
- **Need Rating:** 5/5

## `.\evaluation\rag_evaluator.py`
- **Contains:** Logic for offline or online evaluation of RAG responses.
- **Functions:**
  - `evaluate_rag`: Compares the generated answer to the ground truth or context (e.g., checking for context relevance, answer accuracy).
- **Purpose in Application:** Quantifies the performance of the system so the optimization layer can make routing decisions.
- **Places to Improve:** Add more granular metrics (e.g., faithfulness, hallucination rate) instead of a single overall evaluation score.
- **Need Rating:** 5/5

## `.\experiments.db`
- **Contains:** SQLite database containing evaluation metrics.
- **Functions:** *(Database file).*
- **Purpose in Application:** Stores logs of model decisions, configurations, and resulting evaluation scores to drive the self-optimization engine.
- **Places to Improve:** Move to a more robust database like PostgreSQL if concurrency scales up.
- **Need Rating:** 5/5

## `.\ingestion\chunker.py`
- **Contains:** Logic to split large documents into smaller, processable chunks before embedding.
- **Functions:**
  - `chunk_document`: Slices text based on token limits, semantic boundaries, or fixed lengths.
- **Purpose in Application:** Prevents context window overflows and ensures that retrieval hits specific, relevant parts of a document.
- **Places to Improve:** Implement semantic chunking (splitting by paragraphs/sentences) rather than purely fixed token length.
- **Need Rating:** 4/5

## `.\llm\llm_service.py`
- **Contains:** The primary interface to the LLM providers (e.g., OpenAI, Anthropic, Ollama).
- **Functions:**
  - `clean_output`: Removes markdown formatting or unwanted spaces from the LLM output.
  - `format_code_block`: Ensures any code provided by the LLM is properly formatted.
  - `generate_answer`: Submits the final prompt to the LLM and gets the response.
- **Purpose in Application:** The brain of the actual generation step in RAG.
- **Places to Improve:** Implement automatic retry logic for transient API failures in `generate_answer`.
- **Need Rating:** 5/5

## `.\llm\model_registry.py`
- **Contains:** A configuration mapping of available models and their endpoints/keys.
- **Functions:**
  - `get_model`: Retrieves the specific client/configuration for a given model name.
- **Purpose in Application:** Decouples model definitions from the core logic, allowing easy addition of new LLMs.
- **Places to Improve:** Could dynamically update API configurations based on a database rather than a static registry.
- **Need Rating:** 4/5

## `.\llm\prompt_builder.py`
- **Contains:** Templates and logic to construct the final prompts sent to the LLM.
- **Functions:**
  - `trim_context`: Fits the retrieved documents into the context window limit.
  - `get_verbosity`: Determines how detailed the answer should be.
  - `build_code_prompt`, `build_rag_prompt`, `build_general_prompt`, `build_json_prompt`: Specialized templates for different tasks.
  - `build_prompt`: The main router that picks the correct template and builds the final string.
- **Purpose in Application:** Ensures the LLM is properly instructed based on the user's specific context and constraints.
- **Places to Improve:** Optimize `trim_context` to prioritize the highest-ranked documents rather than just arbitrarily cutting off the end.
- **Need Rating:** 5/5

## `.\logs\rag_logs.json`
- **Contains:** Persistent query log output.
- **Functions:** *(JSON Data File).*
- **Purpose in Application:** Stores user query inputs, results, and generated metrics in a text-readable format.
- **Places to Improve:** Add log rotation limits to prevent infinite disk space scaling.
- **Need Rating:** 3/5

## `.\observability\cost_tracker.py`
- **Contains:** Empty placeholder file intended for tracking token usage / application costs. 
- **Functions:** *(None)*
- **Purpose in Application:** Expected to manage cost attribution by tracking LLM input/output tokens to optimize application-wide ROI limits.
- **Places to Improve:** Fully implement the module. Integrate token counting hooks with the LLM models (e.g., using `tiktoken` for OpenAI wrappers).
- **Need Rating:** 2/5 (Required but currently un-implemented)

## `.\observability\latency_tracker.py`
- **Contains:** Tools for measuring how long different parts of the pipeline take.
- **Functions:**
  - `__init__`, `start`, `end`, `get_all`: Lifecycle methods for a timer context.
- **Purpose in Application:** Logs latency stats to optimize the system and feed data to the dashboard.
- **Places to Improve:** Could be implemented as a Python decorator `@track_latency` for cleaner integration across the codebase.
- **Need Rating:** 3/5

## `.\optimization\experiment_db.py`
- **Contains:** Database interactions to store results of A/B tests and evaluations.
- **Functions:**
  - `init_db`: Sets up the SQLite tables for experiments.
  - `log_experiment`: Records the variables used and the resulting evaluation score.
  - `get_best_config`: Queries the DB for the highest-performing configuration so far.
- **Purpose in Application:** The persistence layer for the "Self-Optimizing" engine. Enables data-driven routing.
- **Places to Improve:** `get_best_config` should have a caching layer so it doesn't query the DB on every single request.
- **Need Rating:** 5/5

## `.\optimization\optimizer.py`
- **Contains:** The heart of the self-optimizing functionality.
- **Functions:**
  - `choose_config`: Selects the RAG parameters (chunk size, model, top-k) based on history.
  - `adapt_config`: Analyzes recent failures and adjusts underlying system parameters.
- **Purpose in Application:** Dynamically changes the system's behavior to maximize the evaluation metrics and minimize cost/latency.
- **Places to Improve:** `adapt_config` could use a more robust statistical threshold before deciding to change a parameter globally.
- **Need Rating:** 5/5

## `.\query_processing\query_analyzer.py`
- **Contains:** NLP logic to understand the user's raw query before retrieval.
- **Functions:**
  - `analyze_query`: Extracts keywords, intent, and domain from the query.
- **Purpose in Application:** Powers the routing logic. By understanding the query intent, the system knows what index or model to use.
- **Places to Improve:** Add query rewriting capabilities (expanding abbreviations, correcting typos) to improve retrieval performance.
- **Need Rating:** 4/5

## `.\query_processing\query_decomposer.py`
- **Contains:** LLM-based query decomposition logic with rule-based fallbacks.
- **Functions:**
  - `decompose_query_llm`: Uses an LLM to smartly break down complex queries into sub-questions.
  - `fallback_decomposition`: Fallback splitting queries mechanically with 'compare', 'and', or 'then'.
- **Purpose in Application:** Separates independent question elements to optimize hybrid vector-search limits.
- **Places to Improve:** Expand context-aware rules in the fallback. Refine the LLM prompt.
- **Need Rating:** 5/5

## `.\query_processing\query_planner.py`
- **Contains:** Decision logic to coordinate when/if a query should be decomposed.
- **Functions:**
  - `plan_query`: Based on the config manager settings and query analysis, decides to return a single query or list of multiple queries.
- **Purpose in Application:** Crucial for servicing complex requests that require gathering multiple independent sources of evidence before answering.
- **Places to Improve:** Could allow dynamic execution graph planning.
- **Need Rating:** 5/5

## `.\project_structure.md`
- **Contains:** Outdated markdown summary of project structure.
- **Purpose in Application:** Redundant context descriptor largely subsumed by `CODEBASE_OVERVIEW.md`.
- **Need Rating:** 1/5

## `.\rag_logging\__init__.py`
- **Contains:** Placeholder for log tooling package.
- **Purpose in Application:** Initializes the package. Needs to contain actual implementations to write to `rag_logs.json`.
- **Need Rating:** 2/5

## `.\requirements.txt`
- **Contains:** pip dependencies and package requirements.
- **Purpose in Application:** Needed for deployment.
- **Need Rating:** 5/5

## `.\retrieval\hybrid_retriever.py`
- **Contains:** Mechanism to combine keyword and vector search.
- **Functions:**
  - `hybrid_search`: Merges and deduplicates results from different search algorithms, usually using Reciprocal Rank Fusion (RRF).
- **Purpose in Application:** Provides the highest quality retrieval by covering both exact keyword matches and semantic meaning.
- **Places to Improve:** The weighting between vector and keyword scores in `hybrid_search` could be dynamically tuned by the optimizer.
- **Need Rating:** 5/5

## `.\retrieval\keyword_retriever.py`
- **Contains:** BM25 or raw text search implementation.
- **Functions:**
  - `build_index`: Creates the TF-IDF/BM25 index from documents.
  - `rebuild_from_chroma`: Syncs the keyword index using data already loaded in the vector database.
  - `keyword_search`: Executes the exact-match search.
- **Purpose in Application:** Necessary for queries that require exact entity names, acronyms, or IDs where vector search fails.
- **Places to Improve:** `rebuild_from_chroma` can be a slow operation; it should be done asynchronously in a background worker.
- **Need Rating:** 4/5

## `.\retrieval\pipeline.py`
- **Contains:** The main execution flow that chains all components together.
- **Functions:**
  - `normalize_documents`: Cleans up the formatting of retrieved text.
  - `rag_pipeline`: Orchestrates retrieving, prompt building, and LLM generation. Incorporates a Smart Fallback to parametric intelligence when retrieval finds 0 documents. Sync stores Chroma memory dynamically based on evaluator responses.
- **Purpose in Application:** The single source of truth for how a query becomes an answer. Ties the entire architecture together.
- **Places to Improve:** Auto-escalation behaviors could be safely reintroduced or augmented with lightweight agents to handle edge-case hallucinated data further.
- **Need Rating:** 5/5

## `.\retrieval\reranker.py`
- **Contains:** Re-scoring logic applied to retrieved documents before they go to the LLM.
- **Functions:**
  - `rerank`: Uses a cross-encoder model to accurately re-sort documents based on relevance to the query.
- **Purpose in Application:** Maximizes accuracy by ensuring the most relevant documents are at the very top of the context window.
- **Places to Improve:** Reranking is computationally heavy. `rerank` should have a hard timeout or utilize a faster, smaller model.
- **Need Rating:** 4/5

## `.\scripts\ingest_documents.py`
- **Contains:** Script used to ingest documents into the vector database.
- **Functions:** *(Standalone initialization script operations).*
- **Purpose in Application:** Takes raw `.txt` files from the `data` directory, uses the `chunk_document` method from `chunker.py`, and loads them into Chroma search instances for RAG availability.
- **Places to Improve:** Should implement chunk overlapping parameterization. Allow for bulk-concurrent ingest streams rather than a manual loop. Support more formats than just `.txt`.
- **Need Rating:** 5/5

## `.\summary_data.json`
- **Contains:** Evaluated snapshot metrics for dashboard display.
- **Purpose in Application:** Fast-retrieval summarized analytics for Streamlit components.
- **Need Rating:** 4/5

## `.\tests\__init__.py`
- **Contains:** Empty definitions for a testing suite.
- **Purpose in Application:** Placeholder for future automated evaluation.
- **Need Rating:** 2/5

## `.\ui\streamlit_app.py`
- **Contains:** The frontend user interface code using Streamlit.
- **Functions:**
  - `render_answer`: Displays the LLM payload, citations, and metadata on the screen.
  - `normalize`: Standardizes input variables for UI display.
- **Purpose in Application:** The mechanism for users to actually interact with the system.
- **Places to Improve:** UI state management can be finicky in Streamlit. Moving to a more robust React-based frontend if the application scales.
- **Need Rating:** 5/5

## `.\vectorstore\chroma_store.py`
- **Contains:** Integration with ChromaDB.
- **Functions:**
  - `add_documents`: Loads embedded chunks into the database.
  - `search_documents`: Executes the vector similarity search (k-NN).
- **Purpose in Application:** The fundamental data storage mechanism that makes semantic search possible.
- **Places to Improve:** Add filtering capabilities to `search_documents` using metadata routing.
- **Need Rating:** 5/5

## `.\workers\evaluation_worker.py`
- **Contains:** Asynchronous worker processing for offline evaluation logging.
- **Functions:**
  - `run_evaluation_async`: Handles the slow logging of evaluation telemetry to SQLite in the background without blocking the API.
  - `task`: The main loop or job execution method.
- **Purpose in Application:** Prevents user latency. Records metrics like answer relevance *after* the user has already received their response.
- **Places to Improve:** Implement proper message queues (like Celery/RabbitMQ) instead of simple async tasks if the scale increases.
- **Need Rating:** 4/5

---

## **Overview**

The **Self-Optimizing RAG** application is a highly advanced, modular framework built for retrieving and intelligently reasoning over data. The system is broken down into distinct layers:

1. **Ingestion & Vectorization:** Scripts chunk text (`chunker.py`) and generate semantic embeddings (`embedding_service.py`) for storage in ChromaDB (`chroma_store.py`). Handled locally by `ingest_documents.py`.
2. **Retrieval Engine:** It utilizes a robust hybrid approach, combining exact keyword matching (`keyword_retriever.py`) and semantic search, followed by cross-encoder reranking (`reranker.py`) to fetch the most accurate context. Multi-hop breakdown enables handling complicated logical reasoning chains (`query_planner.py`).
3. **Core API & Control Plane:** The FastAPI backend securely exposes the pipeline, while the control plane dynamically routes traffic to the most appropriate knowledge base (`knowledge_router.py`) or LLM provider (`model_router.py`). Core logic values are managed seamlessly via `config_manager.py`.
4. **Self-Optimization & Observability:** The defining feature of the codebase. By monitoring latency (`latency_tracker.py`), calculating response confidence (`pipeline.py`, `confidence_model.py`), and logging telemetry via workers (`evaluation_worker.py`), the app relies strictly on data (`experiment_db.py`) to systematically self-adjust heuristics.
5. **Presentation Layer:** A Streamlit dashboard interface (`streamlit_app.py`) built directly on top of the API routes provides an immediate workspace for testing and interacting with the AI agent.

---

### System Data Flow & Request Lifecycle
When a user submits a query through the UI (`ui\streamlit_app.py`) or the HTTP endpoints (`api\routes\query_routes.py`), the request undergoes the following lifecycle:

1. **Cache Check:** The `query_cache.py` validates if this exact response is already available in fast lookup paths.
2. **Query Analysis & Planning:** `query_analyzer.py` determines the topic intent, while `query_planner.py` splits the prompt apart if it is deemed a complex sequence (e.g., comparing two items simultaneously).
3. **Control Plane Routing:** `knowledge_router.py` picks a retrieval vector space based on intent, and `model_router.py` actively picks the most robust available LLM model configuration from the `experiment_db.py` logic. Settings lock-in using `config_manager.py`.
4. **Retrieval & Re-ranking:** ChromaDB kicks off semantic k-NN searches via `chroma_store.py`. Concurrently `keyword_retriever.py` searches through BM25 metadata. Both stacks intersect at `hybrid_retriever.py`, which is then sent directly through `reranker.py` for context alignment and penalty-trimming.
5. **Context Memory & Fusion:** The context pulls from legacy cache locations or specifically via `chroma_memory_store.py` (which intelligently retrieves high-confidence, contextually relevant exact prior answers leveraging local ML loops).
6. **LLM Generation:** `prompt_builder.py` trims the context and wraps the payload templates. Sent sequentially through the `llm_service.py` to target provider platforms.
7. **Offline Evaluation & Retention:** As the user views their completed response, `pipeline.py` executes synchronous memory storing parameters locking down the answers via `chroma_memory_store.py` while asynchronously passing telemetry to `evaluation_worker.py` to adjust statistical logs driving dynamic logic parameterization into `optimizer.py`.

### Environment & Persistence Context
By design, the Self-Optimizing model expects high amounts of caching manipulation and dynamic memory:
* `data/*`: Active un-formatted text drops holding baseline application knowledge.
* `chroma_db/*`: Retained, tokenized index embeddings loaded from the document scripts *and* actively saving RAG state interaction history via `chroma_memory_store.py`.
* `experiments.db`: A persistent SQLite tracker holding parameter combinations vs. evaluation scores representing application-learned heuristic efficiency metrics.
* `summary_data.json`: Quick-look summaries on metric data for `dashboard/app.py`.

---

### Logical Component Map
To navigate the codebase files effectively, they are grouped into the following conceptual domains:

* **Entry Points & UI:** `app\main.py`, `api\routes\query_routes.py`, `ui\streamlit_app.py`, `dashboard\app.py`
* **Control Plane (Orchestration):** `control_plane\config_manager.py`, `control_plane\knowledge_router.py`, `control_plane\model_router.py`
* **Query Processing Pipeline:** `query_processing\query_analyzer.py`, `query_processing\query_planner.py`, `retrieval\pipeline.py`
* **Retrieval & Vector Logic:** `retrieval\hybrid_retriever.py`, `retrieval\keyword_retriever.py`, `retrieval\reranker.py`, `vectorstore\chroma_store.py`
* **LLM Integrations:** `llm\llm_service.py`, `llm\model_registry.py`, `llm\prompt_builder.py`
* **Optimization & Data Collection:** `optimization\optimizer.py`, `optimization\experiment_db.py`, `workers\evaluation_worker.py`, `evaluation\rag_evaluator.py`, `evaluation\confidence_model.py`
* **Data Ingestion:** `ingestion\chunker.py`, `embeddings\embedding_service.py`, `scripts\ingest_documents.py`
* **Caching & Memory:** `cache\chroma_memory_store.py`, `cache\query_cache.py`, `cache\redis_cache.py`, `cache\memory_store.py`
* **Observability:** `observability\latency_tracker.py`, `observability\cost_tracker.py`

### Technology Integrations
Specific third-party libraries map directly to wrapper files within this codebase:
* **FastAPI / Uvicorn** -> Managed inside `app\main.py` and `api\routes\query_routes.py`
* **ChromaDB** -> Handled entirely within `vectorstore\chroma_store.py` and actively managing memory stores in `cache\chroma_memory_store.py`
* **Sentence-Transformers** -> Wrapped cleanly in `embeddings\embedding_service.py` 
* **BM25Okapi (rank_bm25)** -> Integrated within `retrieval\keyword_retriever.py`
* **Cross-Encoder Model** -> Executed within `retrieval\reranker.py`
* **Streamlit** -> Powers `ui\streamlit_app.py` and `dashboard\app.py`
* **Ollama (Local LLM Server)** -> Referenced dynamically via HTTP inside `llm\llm_service.py`

### Strategic Roadmap
Based on internal code reviews and "Places to Improve" items, the architecture is currently targeting the following major technical epics:
1. **Asynchronous Architecture Migration:** Transition intensive synchronous processes (like reranking, metrics logging, or BM25 index rebuilding) to robust async brokers like **Celery** or **RabbitMQ** to ensure stable API latencies under load.
2. **Semantic Query Caching:** Upgrade the current exact-match caching inside `query_cache.py` to utilize intermediate vector-similarity caching. This allows high-confidence overlaps to bypass standard retrieval limits heavily reducing LLM generation costs.
3. **Control Plane Persistence:** Move dynamic routing configurations generated by `config_manager.py` out of local thread-memory into a durable state layer (e.g. Redis or SQLite) to prevent "amnesia" on server restarts.
4. **Small-LLM Utility Agents:** Offload heuristic query parsing (`query_planner.py`) and memory contradiction checking (`chroma_memory_store.py`) to localized, lightweight SLMs rather than regex string matching or expensive generation-tier models.
