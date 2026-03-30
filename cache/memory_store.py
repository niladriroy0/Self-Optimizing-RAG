# ⚠️ DEPRECATED
# This file is replaced by chroma_memory_store.py
# Used only for testing / fallback


from embeddings.embedding_service import embed_text
import numpy as np

MEMORY = []
MAX_MEMORY = 100


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def store_memory(query, answer, confidence):

    # 🔥 Balanced threshold
    if confidence < 0.7:
        return

    # ----------------------------------
    # Deduplication (by query)
    # ----------------------------------
    for item in MEMORY:
        if item["query"] == query:
            return

    MEMORY.append({
        "query": query,
        "answer": answer,

        # 🔥 DUAL EMBEDDING (IMPORTANT)
        "query_embedding": embed_text(query),
        "answer_embedding": embed_text(answer),

        "confidence": confidence,
        "verified": False
    })

    print(f"📥 MEMORY STORED | Conf: {round(confidence, 3)} | Total: {len(MEMORY)}")

    # ----------------------------------
    # Memory cap
    # ----------------------------------
    if len(MEMORY) > MAX_MEMORY:
        MEMORY.pop(0)


def verify_memory(query: str, is_correct: bool) -> bool:
    success = False

    for i in range(len(MEMORY) - 1, -1, -1):
        if MEMORY[i]["query"] == query:
            if is_correct:
                MEMORY[i]["verified"] = True
            else:
                MEMORY.pop(i)
            success = True
            break

    return success


def retrieve_memory(query, top_k=2):

    if not MEMORY:
        print("📤 MEMORY EMPTY")
        return []

    query_emb = embed_text(query)

    scored = []

    for item in MEMORY:

        # ----------------------------------
        # Usage filter
        # ----------------------------------
        if item["confidence"] < 0.65 and not item["verified"]:
            continue

        # ----------------------------------
        # 🔥 DUAL SIMILARITY
        # ----------------------------------
        score_q = cosine_sim(query_emb, item["query_embedding"])
        score_a = cosine_sim(query_emb, item["answer_embedding"])

        # 🔥 HYBRID SCORE (balanced)
        score = (0.6 * score_q) + (0.4 * score_a)

        # 🔥 Relaxed threshold (important)
        if score < 0.35:
            continue

        # 🔥 Boost verified memory
        if item["verified"]:
            score += 0.1

        scored.append((score, item["answer"]))

    scored.sort(reverse=True)

    results = [ans for _, ans in scored[:top_k]]

    print(f"📤 MEMORY RETRIEVED: {len(results)} / {len(MEMORY)}")

    return results