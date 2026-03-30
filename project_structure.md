# Self-Optimizing RAG Project Structure

## Overview
A sophisticated AI-powered question-answering platform that automatically improves performance through continuous experimentation and optimization. It combines vector-based semantic search, keyword-based retrieval (BM25), and large language model generation to provide accurate answers with self-optimization.

## Root-Level Files

- **README.md** — Project documentation
- **SYSTEM_ARCHITECTURE.md** — System design and architecture documentation
- **requirements.txt** — Python dependency list (FastAPI, ChromaDB, sentence-transformers, rank-bm25, Streamlit, etc.)
- **project_structure.md** — This file (generated/updated from repository contents)

## api/
FastAPI Application Layer

- **__init__.py** — Package initialization
- **routes/**
	- **query_routes.py** — Defines the REST API endpoint (POST /query) and streaming response handling

## app/
Core Application Initialization

- **__init__.py** — Package initialization
- **main.py** — FastAPI app startup: loads stores/indexes and mounts routers

## cache/
In-memory / Redis cache wrappers

- **__init__.py** — Package initialization
- **query_cache.py** — Local in-memory query caching utilities
- **redis_cache.py** — Redis-backed cache implementation

## chroma_db/
Vector Database Storage

- **chroma.sqlite3** — Persistent ChromaDB store (SQLite file)
- **7efb519e-e0c6-44f1-bdb9-f02286f001a1/** — Chroma collection directory

## configs/
Configuration Management

- **__init__.py** — Package initialization
- **system_config.yaml** — YAML configuration for system settings

## control_plane/
Control and routing components

- **__init__.py** — Package initialization
- **config_manager.py** — Loads and exposes configuration
- **experiment_engine.py** — Experiment orchestration and scheduling
- **knowledge_router.py** — Routes queries to appropriate knowledge sources
- **model_router.py** — Maps requests to LLM/model backends
- **prompt_registry.py** — Central place for prompt templates

## dashboard/
Dashboard / Admin UI (placeholder)

- **__init__.py** — Package initialization
- **app.py** — Dashboard application placeholder

## data/
Training/Ingestion Data (text sources)

- **attention.txt**
- **bert.txt**
- **gpt.txt**
- **nlp_intro.txt**
- **transformer.txt**

## data_plane/
Pipeline and processing helpers

- **context_builder.py** — Builds LLM context from retrieved documents
- **query_processor.py** — Pre-processes incoming queries
- **rag_pipeline.py** — High-level orchestration of the RAG flow

## embeddings/
Text Embedding Service

- **__init__.py** — Package initialization
- **embedding_service.py** — Utility for producing text embeddings

## evaluation/
RAG Quality Assessment

- **__init__.py** — Package initialization
- **rag_evaluator.py** — Computes relevance and faithfulness metrics

## ingestion/
Document ingestion utilities

- **__init__.py** — Package initialization
- **chunker.py** — Splits text files into chunks for vector ingestion

## llm/
Language Model integrations

- **__init__.py** — Package initialization
- **llm_service.py** — Adapter for calling the LLM backend
- **model_registry.py** — Registry of available models and settings
- **prompt_builder.py** — Helpers to build prompts from templates and context

## logs/
Logging and persistent logs

- **rag_logs.json** — JSON log file for query/answer records

## observability/
Metrics and monitoring helpers

- **__init__.py** — Package initialization
- **cost_tracker.py** — Tracks cost per request (if available)
- **latency_tracker.py** — Measures latency across pipeline stages
- **metrics_collector.py** — Aggregates and exposes metrics

## optimization/
Self-optimization and experimentation

- **__init__.py** — Package initialization
- **experiment_db.py** — SQLite-backed experiment persistence
- **experiment_tracker.py** — In-memory experiment tracker
- **optimizer.py** — Logic to select or propose retrieval/configuration

## query_processing/
Query analysis and scoring

- **__init__.py** — Package initialization
- **complexity_scorer.py** — Scores queries for complexity-based routing
- **intent_classifier.py** — (Placeholder) classify query intent
- **query_analyzer.py** — Heuristics for query characteristics

## rag_logging/
Query logging utilities

- **__init__.py** — Package initialization
- **logger.py** — Functions to append/query rag_logs.json

## retrieval/
Retrieval and reranking core

- **__init__.py** — Package initialization
- **hybrid_retriever.py** — Combines vector + keyword retrieval
- **keyword_retriever.py** — BM25 keyword search utilities
- **pipeline.py** — RAG pipeline orchestration
- **reranker.py** — Cross-encoder reranking utilities

## scripts/
Project scripts and automation

- **__init__.py** — Package initialization
- **ingest_documents.py** — Script to ingest text files into the vector store

## tests/
Testing utilities

- **__init__.py** — Package initialization for tests

## ui/
User-facing interface

- **__init__.py** — Package initialization
- **streamlit_app.py** — Streamlit UI for interactive querying and observability

## vectorstore/
ChromaDB integration layer

- **__init__.py** — Package initialization
- **chroma_store.py** — Wrapper around ChromaDB collection operations

## workers/
Background workers for async tasks

- **__init__.py** — Package initialization
- **evaluation_worker.py** — Background evaluation worker
- **optimization_worker.py** — Background optimization/experiment worker

## Additional Notes

- Many packages include `__pycache__/` directories generated by Python. These are runtime artifacts and not required in VCS.

## Key System Characteristics

- **Modular Architecture**: Clear separation of concerns across retrieval, generation, evaluation, and optimization
- **Dual Retrieval**: Combines semantic (vector) and syntactic (keyword/BM25) search for comprehensive coverage
- **Self-Optimization**: Automatically improves by tracking configuration performance and preferring better-performing parameter sets
- **Observability**: Comprehensive logging with metrics on answer relevance and faithfulness for every query
- **Streaming UI**: Real-time response streaming for better user experience via Streamlit
- **Local-First**: Uses local model adapters and ChromaDB for local embeddings—minimizes external dependencies