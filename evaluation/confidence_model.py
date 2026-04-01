from typing import List, Tuple, Dict
import math


def compute_confidence(
    clean_reranked: List[Tuple[str, float]],
    top_k: int,
    memory_context: List[str],
    query_analysis: Dict,
    evaluation_scores: Dict = None,
    answer: str = None
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

        base_conf = 0.5  # Start lower. A purely LLM response without retrieval is inherently uncertain.

        # --------------------------
        # BAD SIGNALS
        # --------------------------
        answer_lower = answer.lower()
        if "i don't know" in answer_lower or "not sure" in answer_lower or "as an ai" in answer_lower:
            base_conf -= 0.5

        # --------------------------
        # LENGTH SIGNAL
        # --------------------------
        if len(answer) < 50:
            base_conf -= 0.2

        if len(answer) > 200:
            base_conf += 0.1

        # --------------------------
        # TYPE BOOST & STRUCTURE
        # --------------------------
        if query_analysis.get("type") == "coding":
            if "def " in answer or "class " in answer or "```" in answer:
                base_conf += 0.2
            else:
                base_conf -= 0.3  # Penalize if it claims to be coding but lacks code structure

        if "\n" in answer:
            base_conf += 0.05

        # --------------------------
        # EVALUATION INTEGRATION
        # --------------------------
        if evaluation_scores:
            faithfulness = evaluation_scores.get("faithfulness", 0)
            relevance = evaluation_scores.get("answer_relevance", 0)

            eval_score = (faithfulness + relevance) / 2

            # Use evaluation as a strict multiplier. 
            # If the eval score is terrible (e.g. 0.2), confidence drops by 80%.
            base_conf = base_conf * eval_score

        return max(0.0, min(base_conf, 1.0))

    # ----------------------------------
    # 🔍 RAG MODE (RETRIEVAL BASED)
    # ----------------------------------

    scores = [score for _, score in clean_reranked[:top_k]]
    
    if not scores:
        avg_score = -2.0
    else:
        avg_score = sum(scores) / len(scores)

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    retrieval_score = sigmoid(avg_score)

    # ----------------------------------
    # MEMORY BOOST
    # ----------------------------------
    memory_boost = min(len(memory_context) * 0.05, 0.15)

    # ----------------------------------
    # COMPLEXITY PENALTY
    # ----------------------------------
    complexity_penalty = 0.1 if query_analysis.get("complexity") == "high" else 0

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

        # Strong penalty for bad evaluations. A low evaluation score will now drag the confidence
        # entirely down, instead of barely moving it.
        confidence = confidence * eval_score

    # ----------------------------------
    # CLAMP (No artificial inflation)
    # ----------------------------------
    # math.sqrt() was removed here because sqrt(0.4) = 0.63, which artificially 
    # pushed deeply lacking answers into "high confidence" thresholds!
    
    return max(0.0, min(confidence, 1.0))