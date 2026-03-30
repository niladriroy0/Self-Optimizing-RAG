from vectorstore.chroma_store import search_documents
from retrieval.keyword_retriever import keyword_search
from retrieval.reranker import rerank


# hybrid_retriever.py

def hybrid_search(query, top_k=3):

    vector_results = search_documents(query, k=top_k * 2)
    keyword_results = keyword_search(query, k=top_k * 2)

    combined = list(set(vector_results + keyword_results))

    # ❌ REMOVE rerank here
    return combined[:20]