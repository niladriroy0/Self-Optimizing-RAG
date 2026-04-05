import pytest
from cache.chroma_memory_store import store_memory, retrieve_memory, verify_memory, _cleanup_memory

@pytest.fixture(autouse=True)
def mock_dependencies(mocker):
    # Mock embed_text to avoid real SentenceTransformer calls
    mocker.patch("cache.chroma_memory_store.embed_text", return_value=[0.1, 0.2, 0.3])
    # Also ensure threading in _cleanup_memory happens synchronously in test or is ignored
    mocker.patch("threading.Thread")


def test_store_memory_low_confidence(mock_db_session):
    """Test that low confidence answers are not stored."""
    store_memory("What is AI?", "AI is artificial intelligence.", 0.5)
    mock_db_session.add.assert_not_called()

def test_store_memory_high_confidence(mock_db_session):
    """Test that high confidence answers are stored."""
    mock_db_session.count.return_value = 0 # Empty collection
    store_memory("What is AI?", "AI is artificial intelligence.", 0.9)
    
    mock_db_session.add.assert_called_once()
    kwargs = mock_db_session.add.call_args[1]
    assert kwargs["documents"] == ["AI is artificial intelligence."]
    assert kwargs["metadatas"][0]["query"] == "What is AI?"
    assert kwargs["metadatas"][0]["confidence"] == 0.9

def test_store_memory_duplicate_prevention(mock_db_session, mocker):
    """Test that duplicate memories are skipped based on L2 distance."""
    mock_db_session.count.return_value = 1
    # Mocking extremely close distance
    mock_db_session.query.return_value = {"distances": [[0.05]]}
    
    store_memory("What is AI?", "AI is artificial intelligence.", 0.9)
    mock_db_session.add.assert_not_called()

def test_retrieve_memory_empty(mock_db_session):
    mock_db_session.count.return_value = 0
    results = retrieve_memory("Test Query")
    assert results == []

def test_retrieve_memory(mock_db_session):
    mock_db_session.count.return_value = 1
    mock_db_session.query.return_value = {
        "documents": [["Past answer 1", "Past answer 2"]],
        "metadatas": [
            [{"confidence": 0.95, "verified": False}, {"confidence": 0.5, "verified": False}]
        ]
    }
    
    # Should only return the high confidence answer since unverified low conf is filtered
    results = retrieve_memory("Query")
    assert len(results) == 1
    assert results[0] == "Past answer 1"

def test_verify_memory(mock_db_session):
    mock_db_session.get.return_value = {
        "ids": ["id-123"],
        "metadatas": [{"query": "test", "verified": False}]
    }

    # Verify correct
    res = verify_memory("test", is_correct=True)
    assert res is True
    mock_db_session.update.assert_called_once()
    
    # Verify incorrect
    res = verify_memory("test", is_correct=False)
    assert res is True
    mock_db_session.delete.assert_called_once_with(ids=["id-123"])
