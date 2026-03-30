from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")


def evaluate_rag(question, answer, context):

    question_emb = model.encode([question])
    answer_emb = model.encode([answer])
    context_emb = model.encode([context])

    relevance = cosine_similarity(question_emb, answer_emb)[0][0]
    faithfulness = cosine_similarity(answer_emb, context_emb)[0][0]

    # 🔥 NEW: hallucination signal
    hallucination = 1 - faithfulness

    result = {
        "answer_relevance": float(relevance),
        "faithfulness": float(faithfulness),
        "hallucination_score": float(hallucination)
    }

    return result