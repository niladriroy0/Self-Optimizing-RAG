import time
import uuid
import threading
from embeddings.embedding_service import embed_text
from vectorstore.chroma_store import client

# ----------------------------------
# CONFIG
# ----------------------------------

COLLECTION_NAME = "memory_store"
MAX_MEMORY = 200
TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days

memory_collection = client.get_or_create_collection(COLLECTION_NAME)


# ----------------------------------
# STORE MEMORY
# ----------------------------------

def store_memory(query, answer, confidence):

    if confidence < 0.7:
        return

    query_emb = embed_text(query)

    # 1. Deduplication Check (prevent identical/near-identical queries from flooding memory)
    if memory_collection.count() > 0:
        existing = memory_collection.query(
            query_embeddings=[query_emb],
            n_results=1
        )
        # ChromaDB default distance is L2. < 0.1 indicates extreme semantic similarity (near duplicate)
        if existing and existing.get("distances") and existing["distances"][0]:
            if existing["distances"][0][0] < 0.1:
                print("⚠️ MEMORY DUPLICATE DETECTED: Skipping.")
                return

    # 2. Store with QUERY embedding instead of answer embedding for accurate retrieval
    memory_collection.add(
        documents=[answer],
        embeddings=[query_emb],
        metadatas=[{
            "query": query,
            "confidence": float(confidence),
            "timestamp": time.time(),
            "verified": False
        }],
        ids=[str(uuid.uuid4())]
    )

    print(f"📥 MEMORY STORED (Chroma) | Conf: {round(confidence, 3)}")

    # 3. Fire-and-forget sync (async memory limits to avoid blocking RAG API latency)
    threading.Thread(target=_cleanup_memory, daemon=True).start()


# ----------------------------------
# RETRIEVE MEMORY
# ----------------------------------

def retrieve_memory(query, top_k=2):

    if memory_collection.count() == 0:
        print("📤 MEMORY EMPTY")
        return []

    query_emb = embed_text(query)

    results = memory_collection.query(
        query_embeddings=[query_emb],
        n_results=top_k
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    filtered = []

    for doc, meta in zip(docs, metas):

        if meta["confidence"] < 0.65 and not meta.get("verified", False):
            continue

        filtered.append(doc)

    print(f"📤 MEMORY RETRIEVED: {len(filtered)}")

    return filtered


# ----------------------------------
# VERIFY MEMORY
# ----------------------------------

def verify_memory(query, is_correct=True):

    results = memory_collection.get(where={"query": query})

    ids = results.get("ids", [])

    if not ids:
        return False

    if is_correct:
        memory_collection.update(
            ids=ids,
            metadatas=[{**m, "verified": True} for m in results["metadatas"]]
        )
    else:
        memory_collection.delete(ids=ids)

    return True


# ----------------------------------
# CLEANUP (IMPORTANT)
# ----------------------------------

def _cleanup_memory():

    count = memory_collection.count()

    if count <= MAX_MEMORY:
        return

    results = memory_collection.get()

    metas = results["metadatas"]
    ids = results["ids"]

    # sort by oldest timestamp
    sorted_items = sorted(zip(ids, metas), key=lambda x: x[1]["timestamp"])

    to_delete = sorted_items[: count - MAX_MEMORY]

    delete_ids = [item[0] for item in to_delete]

    memory_collection.delete(ids=delete_ids)


# ----------------------------------
# OPTIONAL: TTL CLEANUP
# ----------------------------------

def cleanup_old_memory():

    now = time.time()

    results = memory_collection.get()

    ids = []
    metas = results.get("metadatas", [])

    for i, meta in enumerate(metas):
        if now - meta["timestamp"] > TTL_SECONDS:
            ids.append(results["ids"][i])

    if ids:
        memory_collection.delete(ids=ids)
        print(f"🧹 Deleted {len(ids)} old memories")