import pytest
import sys
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

# Global mocking to prevent ML model downloads and DB locking during test collection
sys.modules['sentence_transformers'] = MagicMock()
mock_chromadb = MagicMock()
# Ensure that when collection.get() is called during lifespan, it returns an empty list
mock_chromadb.PersistentClient.return_value.get_or_create_collection.return_value.get.return_value = {"documents": []}
sys.modules['chromadb'] = mock_chromadb

# Import the actual application
from app.main import app
from control_plane.config_manager import ConfigManager, config_manager as real_config_manager

@pytest.fixture(scope="session")
def app_client():
    """Provides a global FastAPI TestClient."""
    with TestClient(app) as client:
        yield client

@pytest.fixture(autouse=True)
def mock_config_manager(monkeypatch):
    """
    Globally mocks the config manager across all tests
    to prevent cross-test contamination of system settings.
    """
    # Create the mock manager but disable disk I/O first
    # Or, we can just replace its _load_from_disk and _save_to_disk methods
    mock_manager = ConfigManager()
    monkeypatch.setattr(mock_manager, "_load_from_disk", lambda: None)
    monkeypatch.setattr(mock_manager, "_save_to_disk", lambda: None)
    
    mock_manager._init_config() # Ensure clean state
    # Re-apply the mock methods in case _init_config re-creates or overwrites
    monkeypatch.setattr(mock_manager, "_load_from_disk", lambda: None)
    monkeypatch.setattr(mock_manager, "_save_to_disk", lambda: None)

    monkeypatch.setattr("control_plane.config_manager.config_manager", mock_manager)
    return mock_manager

@pytest.fixture
def mock_db_session(mocker):
    """
    Mocks both SQLite (experiments.db) and ChromaDB
    preventing any disk writes during tests.
    """
    # Mocking SQLite
    mocker.patch("optimization.experiment_db.init_db")
    mocker.patch("optimization.experiment_db.log_experiment")
    
    # Mocking ChromaDB
    mock_collection = MagicMock()
    mock_collection.get.return_value = {"documents": []}
    mocker.patch("vectorstore.chroma_store.collection", mock_collection)
    mocker.patch("cache.chroma_memory_store.memory_collection", mock_collection)
    
    return mock_collection

@pytest.fixture
def mock_llm(mocker):
    """
    Mocks the LLM service to return deterministic answers
    without calling OpenAI or local LLM instances.
    """
    mock_generate = mocker.patch("llm.llm_service.generate_answer")
    mock_generate.return_value = "This is a mocked LLM response."
    return mock_generate

@pytest.fixture
def mock_retriever(mocker):
    """
    Mocks the hybrid retriever to avoid BM25 and Vector Search overhead.
    """
    mock_search = mocker.patch("retrieval.hybrid_retriever.hybrid_search")
    mock_search.return_value = [
        {"text": "Mocked retrieved document 1", "score": 0.9, "metadata": {"source": "fake.txt"}},
        {"text": "Mocked retrieved document 2", "score": 0.7, "metadata": {"source": "fake2.txt"}}
    ]
    return mock_search
