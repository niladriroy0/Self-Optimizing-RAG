from rank_bm25 import BM25Okapi

documents = []
bm25 = None


def build_index(chunks):
    global documents, bm25

    documents = chunks
    tokenized = [doc.split() for doc in chunks]

    bm25 = BM25Okapi(tokenized)

def rebuild_from_chroma():
    from vectorstore.chroma_store import collection
    results = collection.get()
    docs = results.get("documents", [])
    if docs:
        build_index(docs)
    print(f"Dynamically rebuilt BM25 index with {len(docs)} documents.")

def keyword_search(query, k=3):
    global bm25

    if bm25 is None:
        raise Exception("BM25 index not built yet")

    tokenized_query = query.split()

    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, _ in ranked[:k]]