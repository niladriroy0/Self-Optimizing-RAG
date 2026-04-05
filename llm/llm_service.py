# llm/llm_service.py

import requests

from llm.prompt_builder import build_prompt
from control_plane.config_manager import config_manager
from control_plane.model_router import get_fallback_model
from observability.cost_tracker import cost_tracker


OLLAMA_URL = "http://localhost:11434/api/generate"


# ----------------------------------
# CLEAN OUTPUT
# ----------------------------------

def clean_output(text: str) -> str:

    bad_tokens = [
        "<｜begin▁of▁sentence｜>",
        "<|endoftext|>",
        "<s>",
        "</s>"
    ]

    for token in bad_tokens:
        text = text.replace(token, "")

    return text.strip()


# ----------------------------------
# CODE FORMATTER
# ----------------------------------

def format_code_block(text: str, query_type: str) -> str:

    if query_type != "coding":
        return text

    text = text.strip()

    if not text.startswith("```"):
        text = "```python\n" + text

    if not text.endswith("```"):
        text += "\n```"

    return text


# ----------------------------------
# CORE LLM CALL
# ----------------------------------

def _call_llm(model, prompt):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    result = response.json()
    answer = result.get("response", "")

    # 🔥 Track token usage and cost for every LLM call
    input_tokens  = cost_tracker.estimate_tokens(prompt)
    output_tokens = cost_tracker.estimate_tokens(answer)
    cost_tracker.record(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens
    )

    return answer


# ----------------------------------
# SIMPLE CONFIDENCE HEURISTIC
# ----------------------------------

def estimate_confidence(answer: str) -> float:

    if not answer or len(answer) < 20:
        return 0.3

    if "I don't know" in answer:
        return 0.2

    return 0.7


# ----------------------------------
# MAIN ENTRY (WITH FALLBACK)
# ----------------------------------

def generate_answer(
    context,
    question,
    model,
    query_type=None,
    query_analysis=None
):

    prompt = build_prompt(
        context=context,
        question=question,
        query_type=query_type,
        query_analysis=query_analysis
    )

    try:

        # --------------------------
        # PRIMARY CALL
        # --------------------------

        answer = _call_llm(model, prompt)
        answer = clean_output(answer)

        confidence = estimate_confidence(answer)

        # --------------------------
        # 🔥 FALLBACK LOGIC
        # --------------------------

        if config_manager.get_param("enable_fallback", True):

            threshold = config_manager.get_param("confidence_threshold", 0.6)

            if confidence < threshold or "I don't have enough information" in answer:

                fallback_model = get_fallback_model(query_analysis)

                # 🔥 If RAG context failed, fallback without context (use parametric knowledge)
                fallback_prompt = build_prompt(
                    context="", 
                    question=question,
                    query_type=query_type,
                    query_analysis=query_analysis
                )

                fallback_answer = _call_llm(fallback_model, fallback_prompt)
                fallback_answer = clean_output(fallback_answer)

                answer = fallback_answer

        # --------------------------
        # FORMAT
        # --------------------------

        answer = format_code_block(answer, query_type)

        return answer

    except Exception as e:
        return f"LLM Error: {str(e)}"