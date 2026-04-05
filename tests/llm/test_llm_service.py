import pytest
from llm.llm_service import clean_output, format_code_block, estimate_confidence, generate_answer

def test_clean_output():
    text = "<|endoftext|> hello <s>"
    cleaned = clean_output(text)
    assert cleaned == "hello"

def test_format_code_block():
    code = "print('hello')"
    # Coding query adds python blocks
    formatted = format_code_block(code, "coding")
    assert "```python\n" in formatted
    assert formatted.endswith("```")
    
    # Non coding returns as is
    formatted_generic = format_code_block(code, "general")
    assert formatted_generic == "print('hello')"

def test_estimate_confidence():
    assert estimate_confidence("I don't know the answer") == 0.2
    assert estimate_confidence("Short") == 0.3
    assert estimate_confidence("This is a sufficiently long and confident answer to the query.") == 0.7

def test_generate_answer_success(mocker):
    # Mocking the actual request to Ollama
    mocker.patch("llm.llm_service._call_llm", return_value="Test answer.")
    
    # Needs a mock context and question
    res = generate_answer("Context", "Question", "test_model")
    assert res == "Test answer."

def test_generate_answer_fallback_triggered(mocker, mock_config_manager):
    # Enable fallback
    mock_config_manager.set_param("enable_fallback", True)
    
    # First call returns low confidence
    mock_call_llm = mocker.patch("llm.llm_service._call_llm", side_effect=["I don't have enough information", "Fallback Answer"])
    mocker.patch("llm.llm_service.get_fallback_model", return_value="fallback_model")
    
    res = generate_answer("Context", "Question", "primary")
    
    assert res == "Fallback Answer"
    assert mock_call_llm.call_count == 2
