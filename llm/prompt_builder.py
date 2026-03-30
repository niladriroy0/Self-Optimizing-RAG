# llm/prompt_builder.py

from typing import Optional
from control_plane.config_manager import config_manager


MAX_CONTEXT_CHARS = 4000
DEFAULT_FORMAT = "text"


# ----------------------------------
# CONTEXT TRIMMER
# ----------------------------------

def trim_context(context: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    if not context:
        return ""

    return context[:max_chars]


# ----------------------------------
# VERBOSITY LOGIC
# ----------------------------------

def get_verbosity(query_analysis: dict) -> str:

    length = query_analysis.get("length", 0)

    complexity_map = {
        "low": 0.3,
        "medium": 0.6,
        "high": 1.0
    }

    complexity = complexity_map.get(
        query_analysis.get("complexity", "medium"),
        0.5
    )

    if length < 6 and complexity < 0.5:
        return "very_short"

    if complexity > 0.7:
        return "detailed"

    return "medium"


# ----------------------------------
# SYSTEM PROMPTS
# ----------------------------------

SYSTEM_BASE = """
You are a highly intelligent AI assistant.
Be accurate, concise, and structured.
Avoid hallucination.
"""

SYSTEM_CODE = """
You are a senior Python engineer.
Write clean, production-quality code.
No explanation unless asked.
"""

SYSTEM_RAG = """
You are a retrieval-augmented assistant.

STRICT RULES:
- Use ONLY the given context
- If answer not in context → say:
  "I don't have enough information"
- DO NOT hallucinate
"""


# ----------------------------------
# PROMPT BUILDERS
# ----------------------------------

def build_code_prompt(question: str) -> str:
    return f"""
{SYSTEM_CODE}

TASK:
Write clean Python code.

QUESTION:
{question}

OUTPUT:
"""


def build_rag_prompt(context: str, question: str, verbosity: str) -> str:

    context = trim_context(context)

    return f"""
{SYSTEM_RAG}

CONTEXT:
{context}

QUESTION:
{question}

STYLE:
{verbosity}

ANSWER:
"""


def build_general_prompt(question: str, verbosity: str) -> str:

    tone_map = {
        "very_short": "Answer in 1-2 lines.",
        "medium": "Answer clearly and concisely.",
        "detailed": "Give a structured answer with reasoning."
    }

    return f"""
{SYSTEM_BASE}

INSTRUCTIONS:
{tone_map.get(verbosity)}

QUESTION:
{question}

ANSWER:
"""


def build_json_prompt(question: str) -> str:
    return f"""
You are an API assistant.

Return ONLY valid JSON.

Question:
{question}

Output:
{{
  "answer": "...",
  "confidence": 0.0
}}
"""


# ----------------------------------
# MAIN ENTRY
# ----------------------------------

def build_prompt(
    context: Optional[str],
    question: str,
    query_type: Optional[str] = None,
    query_analysis: Optional[dict] = None,
    output_format: str = DEFAULT_FORMAT
):

    query_analysis = query_analysis or {}
    verbosity = get_verbosity(query_analysis)

    # 🔥 CONTROL PLANE OVERRIDE
    template = config_manager.get_param("prompt_template")

    # --------------------------
    # PRIORITY ORDER
    # --------------------------

    if query_type == "coding":
        return build_code_prompt(question)

    if output_format == "json":
        return build_json_prompt(question)

    if context:
        return build_rag_prompt(context, question, verbosity)

    return build_general_prompt(question, verbosity)