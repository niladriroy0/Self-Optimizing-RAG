from typing import List, Tuple, Dict
import math


def compute_confidence(
    clean_reranked: List[Tuple[str, float]],
    top_k: int,
    memory_context: List[str],
    query_analysis: Dict,
    evaluation_scores: Dict = None,
    answer: str = None   # 🔥 NEW
) -> float:
    """
    Advanced Confidence Model

    Handles:
    - RAG / HYBRID
    - LLM_ONLY (content-aware)
    """

    # ----------------------------------
    # 🔥 LLM_ONLY MODE (CONTENT-AWARE)
    # ----------------------------------
    if not clean_reranked:

        answer = answer or ""

        base_conf = 0.6

        # --------------------------
        # TYPE BOOST
        # --------------------------
        if query_analysis.get("type") == "coding":
            base_conf += 0.15

        # --------------------------
        # LENGTH SIGNAL
        # --------------------------
        if len(answer) > 200:
            base_conf += 0.1
        elif len(answer) < 50:
            base_conf -= 0.2

        # --------------------------
        # STRUCTURE SIGNAL
        # --------------------------
        if "def " in answer or "class " in answer:
            base_conf += 0.1

        if "\n" in answer:
            base_conf += 0.05

        # --------------------------
        # BAD SIGNALS
        # --------------------------
        if "I don't know" in answer or "not sure" in answer:
            base_conf -= 0.4

        # --------------------------
        # EVALUATION BOOST
        # --------------------------
        if evaluation_scores:
            faithfulness = evaluation_scores.get("faithfulness", 0)
            relevance = evaluation_scores.get("answer_relevance", 0)

            eval_score = (faithfulness + relevance) / 2

            base_conf = (base_conf * 0.6) + (eval_score * 0.4)

        return max(0.0, min(base_conf, 1.0))

    # ----------------------------------
    # 🔍 RAG MODE (RETRIEVAL BASED)
    # ----------------------------------

    scores = [score for _, score in clean_reranked[:top_k]]
    avg_score = sum(scores) / len(scores)

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    retrieval_score = sigmoid(avg_score)

    # ----------------------------------
    # MEMORY BOOST
    # ----------------------------------
    memory_boost = min(len(memory_context) * 0.05, 0.2)

    # ----------------------------------
    # COMPLEXITY PENALTY
    # ----------------------------------
    complexity_penalty = 0.05 if query_analysis.get("complexity") == "high" else 0

    # ----------------------------------
    # BASE CONFIDENCE
    # ----------------------------------
    confidence = retrieval_score + memory_boost - complexity_penalty

    # ----------------------------------
    # EVALUATION INTEGRATION
    # ----------------------------------
    if evaluation_scores:
        faithfulness = evaluation_scores.get("faithfulness", 0)
        relevance = evaluation_scores.get("answer_relevance", 0)

        eval_score = (faithfulness + relevance) / 2

        eval_weight = 0.25
        confidence = (confidence * (1 - eval_weight)) + (eval_score * eval_weight)

    # ----------------------------------
    # CLAMP BEFORE TRANSFORM
    # ----------------------------------
    confidence = max(0.0, min(confidence, 1.0))

    # ----------------------------------
    # 🔥 SMOOTHING
    # ----------------------------------
    confidence = math.sqrt(confidence)

    confidence = 0.9 * confidence + 0.1 * (
        1 / (1 + math.exp(-3 * (confidence - 0.5)))
    )

    return max(0.0, min(confidence, 1.0))