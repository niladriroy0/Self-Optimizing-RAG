# query_processing/query_decomposer.py

import json

from llm.model_registry import get_model


# 🔥 SAFE IMPORT
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


def decompose_query_llm(query: str):

    if not OLLAMA_AVAILABLE:
        return fallback_decomposition(query)

    model = get_model("decomposition")

    prompt = f"""
    Break the following query into atomic sub-questions.

    Rules:
    - Keep them independent
    - Keep them short
    - Return ONLY JSON list

    Query:
    {query}

    Output:
    ["sub-question 1", "sub-question 2"]
    """

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )

        content = response["message"]["content"]

        return json.loads(content)

    except Exception:
        return fallback_decomposition(query)


# ----------------------------------
# 🔁 RULE-BASED FALLBACK
# ----------------------------------

def fallback_decomposition(query: str):

    q = query.lower()

    if "compare" in q or "difference between" in q:
        parts = query.replace("compare", "").split("and")
        return [p.strip() for p in parts if p.strip()]

    if " and " in q:
        return [p.strip() for p in query.split(" and ") if p.strip()]

    if " then " in q:
        return [p.strip() for p in query.split(" then ") if p.strip()]

    return [query]