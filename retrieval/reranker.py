from sentence_transformers import CrossEncoder
from control_plane.config_manager import config_manager
import math

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

MAX_RERANK_NORMAL = 20       # default candidate cap
MAX_RERANK_LOW_RESOURCE = 5  # CPU-only cap (4x faster)


def softmax(scores):
    """Convert logits → probabilities"""
    max_score = max(scores)  # for numerical stability
    exp_scores = [math.exp(s - max_score) for s in scores]
    total = sum(exp_scores)
    return [s / total for s in exp_scores]


def rerank(query, docs, top_k=3):

    if not docs:
        return []

    # 🔥 Respect low_resource_mode for CPU-only machines
    low_resource = config_manager.get_param("low_resource_mode", False)
    max_rerank = MAX_RERANK_LOW_RESOURCE if low_resource else MAX_RERANK_NORMAL

    # ----------------------------------
    # 🔥 Remove duplicates (IMPORTANT)
    # ----------------------------------
    docs = list(dict.fromkeys(docs))

    # ----------------------------------
    # 🔥 Limit candidates
    # ----------------------------------
    docs = docs[:max_rerank]

    print(f"Reranker candidates: {len(docs)} (low_resource_mode={low_resource})")

    pairs = [(query, doc) for doc in docs]

    # Raw logits
    scores = model.predict(pairs)

    # ----------------------------------
    # 🔥 NORMALIZE SCORES (CRITICAL FIX)
    # ----------------------------------
    normalized_scores = softmax(scores)

    ranked = sorted(
        zip(docs, normalized_scores),
        key=lambda x: x[1],
        reverse=True
    )

    print("Reranker selected:", ranked[:top_k])

    return ranked[:top_k]