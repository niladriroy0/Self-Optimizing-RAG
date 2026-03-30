from typing import Optional
from control_plane.config_manager import config_manager  # ✅ NEW


MAX_CONTEXT_CHARS = 4000
DEFAULT_TONE = "concise"
DEFAULT_FORMAT = "text"


def trim_context(context: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    if not context:
        return ""

    if len(context) <= max_chars:
        return context

    return context[:max_chars]


def get_verbosity(query_analysis: dict) -> str:

    length = query_analysis.get("length", 0)

    # 🔥 FIX: normalize complexity
    complexity_map = {
        "low": 0.3,
        "medium": 0.6,
        "high": 1.0
    }

    raw_complexity = query_analysis.get("complexity", "medium")
    complexity = complexity_map.get(raw_complexity, 0.5)

    if length < 6 and complexity < 0.5:
        return "very_short"

    if complexity > 0.7:
        return "detailed"

    return "medium"


SYSTEM_BASE = """
You are a highly intelligent AI assistant.

Follow instructions strictly.
Be accurate, concise, and structured.
Avoid hallucination.
"""

SYSTEM_CODE = """
You are a senior Python engineer.

Write production-quality code.
No unnecessary explanation.
Clean, readable, correct.
"""

SYSTEM_RAG = """
You are a retrieval-augmented assistant.

Strictly follow the context.
If answer is not in context, say:
"I don't have enough information."
Do NOT hallucinate.
"""


def build_code_prompt(question: str, verbosity: str) -> str:

    return f"""
{SYSTEM_CODE}

TASK:
Write clean Python code.

RULES:
- Only return code
- No explanation
- Proper formatting
- No markdown unless required

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

INSTRUCTIONS:
- Answer ONLY from context
- Be concise
- Do not assume anything outside context

ANSWER:
"""


def build_general_prompt(question: str, verbosity: str) -> str:

    tone_map = {
        "very_short": "Answer in 1-2 lines.",
        "medium": "Answer clearly and concisely.",
        "detailed": "Give a structured and detailed answer."
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

Return output strictly in JSON format.

Question:
{question}

Output format:
{{
  "answer": "...",
  "confidence": 0.0
}}
"""


def build_prompt(
    context: Optional[str],
    question: str,
    query_type: Optional[str] = None,
    query_analysis: Optional[dict] = None,
    output_format: str = DEFAULT_FORMAT
):

    query_analysis = query_analysis or {}

    verbosity = get_verbosity(query_analysis)

    # 🔥 CONTROL PLANE TEMPLATE (NEW)
    template = config_manager.get_param("prompt_template")

    if query_type == "coding":
        return build_code_prompt(question, verbosity)

    if output_format == "json":
        return build_json_prompt(question)

    if context:
        return build_rag_prompt(context, question, verbosity)

    return build_general_prompt(question, verbosity)