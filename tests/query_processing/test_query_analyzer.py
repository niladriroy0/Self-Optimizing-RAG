import pytest
from query_processing.query_analyzer import analyze_query

def test_analyze_query_coding():
    res = analyze_query("write python code loop") # length < 5 to get 'low' complexity
    assert res["type"] == "coding"
    assert res["complexity"] == "low"
    assert res["needs_keyword"] is False

def test_analyze_query_reasoning_high_complexity():
    res = analyze_query("explain the architecture difference and compare the approaches")
    assert res["type"] == "reasoning"
    assert res["complexity"] == "high"
    assert res["is_multi_hop"] is True
    assert res["needs_semantic"] is True

def test_analyze_query_hybrid_signals():
    res = analyze_query("exact syntax error code 500 explain why")
    assert res["needs_keyword"] is True
    assert res["needs_semantic"] is True

def test_analyze_query_ambiguity():
    # High ambiguity (contains "it" and short)
    res1 = analyze_query("how does it work")
    assert res1["ambiguity"] == 0.8
    
    # Low ambiguity
    res2 = analyze_query("explain the retrieval augmented generation system completely")
    assert res2["ambiguity"] == 0.3  # Default is 0.3
