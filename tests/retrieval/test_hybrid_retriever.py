import pytest
from retrieval.hybrid_retriever import hybrid_search

@pytest.fixture
def mock_search_backends(mocker):
    vector_mock = mocker.patch("retrieval.hybrid_retriever.search_documents")
    keyword_mock = mocker.patch("retrieval.hybrid_retriever.keyword_search")
    
    vector_mock.return_value = ["doc1", "doc2"]
    keyword_mock.return_value = ["doc2", "doc3"]
    
    return vector_mock, keyword_mock

def test_hybrid_search_merging(mock_search_backends):
    vector_mock, keyword_mock = mock_search_backends
    
    query_analysis = {"needs_keyword": True, "needs_semantic": True}
    results = hybrid_search("test query", query_analysis=query_analysis, top_k=5)
    
    # Should deduplicate doc2
    assert len(results) == 3
    assert set(results) == {"doc1", "doc2", "doc3"}
    
    vector_mock.assert_called_once()
    keyword_mock.assert_called_once()

def test_hybrid_search_semantic_only(mock_search_backends):
    vector_mock, keyword_mock = mock_search_backends
    
    query_analysis = {"needs_keyword": False, "needs_semantic": True}
    results = hybrid_search("test query", query_analysis=query_analysis)
    
    vector_mock.assert_called_once()
    keyword_mock.assert_not_called()
    assert results == ["doc1", "doc2"]

def test_hybrid_search_fallback(mock_search_backends):
    vector_mock, keyword_mock = mock_search_backends
    
    # Simulate both returning nothing initially
    vector_mock.side_effect = [[], ["fallback_doc"]]
    keyword_mock.return_value = []
    
    query_analysis = {"needs_keyword": True, "needs_semantic": True}
    results = hybrid_search("test query", query_analysis=query_analysis)
    
    # Initial vector call + fallback vector call
    assert vector_mock.call_count == 2
    assert results == ["fallback_doc"]
