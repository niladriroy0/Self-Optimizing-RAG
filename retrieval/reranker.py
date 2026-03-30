from sentence_transformers import CrossEncoder
import math

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

MAX_RERANK = 20   # cap


def softmax(scores):
    """Convert logits → probabilities"""
    max_score = max(scores)  # for numerical stability
    exp_scores = [math.exp(s - max_score) for s in scores]
    total = sum(exp_scores)
    return [s / total for s in exp_scores]


def rerank(query, docs, top_k=3):

    if not docs:
        return []

    # ----------------------------------
    # 🔥 Remove duplicates (IMPORTANT)
    # ----------------------------------
    docs = list(dict.fromkeys(docs))

    # ----------------------------------
    # 🔥 Limit candidates
    # ----------------------------------
    docs = docs[:MAX_RERANK]

    print("Reranker candidates:", len(docs))

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