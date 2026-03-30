import requests
from llm.prompt_builder import build_prompt

OLLAMA_URL = "http://localhost:11434/api/generate"


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


def format_code_block(text: str, query_type: str) -> str:

    if query_type != "coding":
        return text

    text = text.strip()

    if not text.startswith("```"):
        text = "```python\n" + text

    if not text.endswith("```"):
        text = text + "\n```"

    return text


def generate_answer(context, question, model, query_type=None, query_analysis=None):

    # 🔥 BUILD ADVANCED PROMPT
    prompt = build_prompt(
        context=context,
        question=question,
        query_type=query_type,
        query_analysis=query_analysis
    )

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=12000
        )

        result = response.json()

        answer = result.get("response", "No response from model")

        # 🔥 CLEAN + FORMAT
        answer = clean_output(answer)
        answer = format_code_block(answer, query_type)

        return answer

    except Exception as e:

        return f"LLM Error: {str(e)}"