import pytest
from llm.prompt_builder import trim_context, get_verbosity, build_prompt
from control_plane.config_manager import config_manager

def test_trim_context():
    assert trim_context("abc", 2) == "ab"
    assert trim_context("", 10) == ""
    assert trim_context("hello", 100) == "hello"

def test_get_verbosity():
    # Long high complexity
    assert get_verbosity({"length": 20, "complexity": "high"}) == "detailed"
    
    # Short low complexity
    assert get_verbosity({"length": 4, "complexity": "low"}) == "very_short"
    
    # Medium
    assert get_verbosity({"length": 10, "complexity": "medium"}) == "medium"

def test_build_code_prompt():
    prompt = build_prompt(context=None, question="Write a loop", query_type="coding")
    assert "You are a senior Python engineer" in prompt
    assert "Write a loop" in prompt
    assert "CONTEXT:" not in prompt

def test_build_json_prompt():
    prompt = build_prompt(context=None, question="Give me stats", output_format="json")
    assert "Return ONLY valid JSON." in prompt
    assert "Give me stats" in prompt

def test_build_rag_prompt():
    prompt = build_prompt(context="Some knowledge", question="What is it?")
    assert "You are a retrieval-augmented assistant" in prompt
    assert "CONTEXT:" in prompt
    assert "Some knowledge" in prompt
    assert "What is it?" in prompt

def test_build_general_prompt():
    prompt = build_prompt(context=None, question="Say hello")
    assert "You are a highly intelligent AI assistant" in prompt
    assert "Say hello" in prompt
    assert "CONTEXT:" not in prompt
