<div align="center">
  <h1>🧠 Self-Optimizing RAG</h1>
  <p><i>An advanced Retrieval-Augmented Generation system that learns, adapts, and optimizes itself based on interaction history and evaluation feedback.</i></p>

  [![Python version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
  [![FastAPI Core](https://img.shields.io/badge/FastAPI-Framework-00a393.svg)](https://fastapi.tiangolo.com/)
  [![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-ff5e5e.svg)](https://www.trychroma.com/)
  [![Streamlit Dash](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg)](https://streamlit.io/)
  [![pytest](https://img.shields.io/badge/Tests-43%20passing-brightgreen.svg)](https://pytest.org/)
</div>

---

## 🚀 Overview

The **Self-Optimizing RAG** application is a modular framework built for retrieving and intelligently reasoning over data. Unlike static RAG pipelines, this system dynamically adjusts its parameters—such as retrieval chunk size, model selection, and top-k documents—based on past performance, confidence scores, and offline evaluation metrics. All learned configuration persists across restarts via YAML-backed storage.

## ✨ Key Features

- 🔄 **Self-Optimizing Engine**: Continuously learns from evaluation results (cosine similarity, answer relevance, faithfulness) to reroute queries to optimal configurations. Learned state persists to `configs/system_config.yaml` and survives server restarts.
- 🧠 **Semantic Memory Store**: High-confidence interactions are saved into a persistent ChromaDB memory module using **query embeddings** (not answer embeddings) for accurate semantic retrieval. Near-duplicate detection and async background cleanup prevent memory contamination.
- 🔀 **Hybrid Retrieval & Reranking**: Combines semantic vector search with exact keyword matching (BM25) via Reciprocal Rank Fusion. Results are re-scored by a Cross-Encoder for maximum accuracy.
- 🧮 **Multi-Hop Query Planning**: Dynamically decomposes complex questions into smaller, independent sub-queries using LLM analysis, with rule-based fallback for resilience.
- 💰 **Cost Tracking**: Every LLM call records token usage and estimated USD cost per model, persisted to `logs/cost_log.json`. Supports per-model pricing for both cloud and local (Ollama) models.
- 📊 **Observability & Analytics**: Per-stage latency tracking, confidence scoring, and an evaluation pipeline that feeds back into model routing. Every API response includes a full observability payload.
- 🧪 **Test Suite**: 43 tests across 8 modules covering all pipeline branches, with global ML model mocking for CI compatibility.

## 🏗️ System Architecture

```mermaid
graph TD
    User([User]) -->|Query| API[FastAPI Entry Point]
    API --> ControlPlane{Control Plane\nConfig & Routing}
    
    ControlPlane --> Cache[Query Cache]
    Cache -->|Miss| Analyzer[Query Analyzer & Planner]
    
    Analyzer --> Retrieval[Hybrid Retrieval Layer]
    
    subgraph Retrieval Pipeline
        Retrieval --> Vector[Semantic Search\nChromaDB]
        Retrieval --> Keyword[Keyword Search\nBM25]
        Vector --> Reranker[Cross-Encoder Reranker]
        Keyword --> Reranker
    end
    
    Reranker -->|0 docs: Smart Fallback| Fallback[Parametric LLM\nFALLBACK_LLM mode]
    Reranker --> Memory[Memory Store \nRetrieval & Fusion]
    Memory --> Builder[Prompt Builder]
    Builder --> LLM[LLM Generation\nOllama + Cost Tracker]
    
    LLM --> Evaluation[Evaluation & Confidence Model]
    Evaluation --> |High Confidence| Persistence[(Semantic Memory DB\nChromaDB)]
    Evaluation --> OptDB[(SQLite Experiments DB\nWAL mode)]
    
    OptDB -.->|Updates Heuristics| ControlPlane
    ControlPlane -.->|Persists to| YAML[(system_config.yaml)]
    LLM --> API
    API -->|Streaming Response| User
    
    subgraph Dashboards
        UI[Main Streamlit UI] --> API
        Dash[Observability Dashboard] --> OptDB
    end
```

## 🛠️ Technology Stack

- **Backend Framework:** FastAPI
- **Vector Database:** ChromaDB (persistent, local)
- **Keyword Search:** BM25 (`rank_bm25.BM25Okapi`)
- **Reranking:** Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim)
- **LLM Provider:** Local Ollama Endpoint (phi3, mistral, qwen2.5, deepseek-coder)
- **Experiment Tracking:** SQLite (WAL mode)
- **Config Persistence:** PyYAML (`configs/system_config.yaml`)
- **Cost Tracking:** Custom `CostTracker` → `logs/cost_log.json`
- **Testing:** pytest + pytest-mock (43 tests)
- **UI & Dashboard:** Streamlit

## 💻 Getting Started

### Prerequisites
- Python 3.9+
- [Ollama](https://ollama.com/) installed and running locally with your preferred models (e.g., `ollama pull llama3`).

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/niladriroy0/Self-Optimizing-RAG.git
   cd Self-Optimizing-RAG
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ingest Documents:**
   Place your raw `.txt` files in the `data/` directory, then run the ingestion script to populate ChromaDB.
   ```bash
   python scripts/ingest_documents.py
   ```

### Running the Services

The application requires multiple components to be running concurrently.

1. **Start the FastAPI Backend:**
   ```bash
   python -m uvicorn app.main:app --reload
   ```

2. **Launch the User Interface:**
   ```bash
   streamlit run ui/streamlit_app.py
   ```

3. **Launch the Observability Dashboard:**
   ```bash
   streamlit run dashboard/app.py
   ```

4. **Run Tests:**
   ```bash
   pytest tests/ -v
   ```

## 📂 Codebase Structure

If you'd like an in-depth dive into the directories and files, check out our comprehensive internal documentation:

- 📖 [`CODEBASE_OVERVIEW.md`](./CODEBASE_OVERVIEW.md) - Detailed breakdown of every file, its functions, purpose, and improvement notes.
- 🏛️ [`SYSTEM_ARCHITECTURE.md`](./SYSTEM_ARCHITECTURE.md) - High-level system design, data flow, and deployment mapping.
- 🎓 [`interview_guide.md`](./interview_guide.md) - Comprehensive guide on how to pitch and explain this project in interviews.

## 🚀 Roadmap
- **Asynchronous Architecture:** Transition offline evaluation and BM25 index rebuilding to message brokers (e.g., Celery/RabbitMQ) for stable API latencies under load.
- **Semantic Query Caching:** Upgrade exact-match caching to vector-similarity to bypass generation for semantically equivalent user intents.
- **LLM-as-Judge Evaluation:** Replace cosine-similarity evaluation in `rag_evaluator.py` with a second LLM grading faithfulness and relevance for more robust quality signals.
- **Config Parameter Exposure:** Move hardcoded values (deduplication threshold, memory TTL, scoring weights) into `config_manager.py` as first-class dynamic parameters.

---
<div align="center">
  <i>Built with ❤️ By Niladri Roy.</i>
</div>
