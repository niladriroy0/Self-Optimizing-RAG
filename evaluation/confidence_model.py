from typing import List, Tuple, Dict
import math


def compute_confidence(
    clean_reranked: List[Tuple[str, float]],
    top_k: int,
    memory_context: List[str],
    query_analysis: Dict,
    evaluation_scores: Dict = None
) -> float:
    """
    Balanced + Mode-Aware Confidence Model

    Handles:
    - RAG / HYBRID (retrieval-based)
    - LLM_ONLY (fallback confidence)
    """

    # ----------------------------------
    # 🔥 LLM_ONLY FALLBACK (IMPORTANT)
    # ----------------------------------
    if not clean_reranked:

        base_conf = 0.65

        # 🔥 Coding queries are more reliable
        if query_analysis.get("type") == "coding":
            base_conf = 0.75

        if evaluation_scores:
            faithfulness = evaluation_scores.get("faithfulness", 0)
            relevance = evaluation_scores.get("answer_relevance", 0)

            eval_score = (faithfulness + relevance) / 2

            confidence = (base_conf * 0.6) + (eval_score * 0.4)
        else:
            confidence = base_conf

        return max(0.0, min(confidence, 1.0))

    # ----------------------------------
    # RETRIEVAL SCORE (RAW AVG + SIGMOID)
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

        eval_weight = 0.25  # balanced influence
        confidence = (confidence * (1 - eval_weight)) + (eval_score * eval_weight)

    # ----------------------------------
    # CLAMP BEFORE TRANSFORM
    # ----------------------------------
    confidence = max(0.0, min(confidence, 1.0))

    # ----------------------------------
    # 🔥 NON-LINEAR SMOOTHING
    # ----------------------------------
    confidence = math.sqrt(confidence)

    confidence = 0.9 * confidence + 0.1 * (
        1 / (1 + math.exp(-3 * (confidence - 0.5)))
    )

    # ----------------------------------
    # FINAL CLAMP
    # ----------------------------------
    return max(0.0, min(confidence, 1.0))