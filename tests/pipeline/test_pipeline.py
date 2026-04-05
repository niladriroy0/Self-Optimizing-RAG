import pytest
from retrieval.pipeline import rag_pipeline

@pytest.fixture
def mock_pipeline_dependencies(mocker):
    # Mocking Heavy Operations
    mocker.patch("retrieval.pipeline.analyze_query", return_value={"type": "reasoning", "complexity": "low"})
    mocker.patch("retrieval.pipeline.route_model_with_exploration", return_value="test_model")
    mocker.patch("retrieval.pipeline.route_knowledge", return_value="RAG")
    mocker.patch("retrieval.pipeline.get_cached", return_value=None)
    mocker.patch("retrieval.pipeline.set_cache")
    mocker.patch("retrieval.pipeline.store_memory")
    mocker.patch("retrieval.pipeline.retrieve_memory", return_value=["Memory 1"])
    mocker.patch("retrieval.pipeline.choose_config", return_value={"top_k": 3})
    mocker.patch("retrieval.pipeline.plan_query", return_value=["Query 1"])
    mocker.patch("retrieval.pipeline.hybrid_search", return_value=["Doc 1"])
    mocker.patch("retrieval.pipeline.rerank", return_value=[("Doc 1", 0.99)])
    mocker.patch("retrieval.pipeline.run_evaluation_async")
    
    mocker.patch("retrieval.pipeline.evaluate_rag", return_value={"score": 1.0})
    mocker.patch("retrieval.pipeline.compute_confidence", return_value=0.88)
    mocker.patch("retrieval.pipeline.generate_answer", return_value="Pipeline Answer.")

def test_pipeline_normal_flow(mock_pipeline_dependencies, mocker):
    """Test full RAG pipeline flow with no cache hit."""
    answer, observe = rag_pipeline("What is AI?")
    
    assert answer == "Pipeline Answer."
    assert observe["confidence"] == 0.88
    assert observe["mode"] == "RAG"

def test_pipeline_cache_hit(mocker):
    # Fast path: cache hit
    mocker.patch("retrieval.pipeline.analyze_query", return_value={"type": "reasoning"})
    mocker.patch("retrieval.pipeline.get_cached", return_value=("Cached Answer", {"mode": "cache"}))
    
    answer, observe = rag_pipeline("What is AI?")
    
    assert answer == "Cached Answer"
    assert observe["mode"] == "cache"

def test_pipeline_llm_only_mode(mock_pipeline_dependencies, mocker):
    """Test pipeline branching to LLM_ONLY when requested."""
    mocker.patch("retrieval.pipeline.route_knowledge", return_value="LLM_ONLY")
    
    answer, observe = rag_pipeline("Say hello")
    
    assert answer == "Pipeline Answer."
    assert observe["mode"] == "LLM_ONLY"

def test_pipeline_fallback_when_empty_retrieval(mock_pipeline_dependencies, mocker):
    """Test pipeline fallback triggered when retrieval returns 0 documents."""
    # Force empty retrieval
    mocker.patch("retrieval.pipeline.hybrid_search", return_value=[])
    
    answer, observe = rag_pipeline("Obscure question")
    
    # Should automatically trigger fallback route and jump to generate_answer
    assert answer == "Pipeline Answer."
    assert observe["mode"] == "FALLBACK_LLM"
