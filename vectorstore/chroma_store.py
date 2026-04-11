import chromadb

# persistent client
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(
    name="documents"
)


def add_documents(chunks):

    ids = [str(i) for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        ids=ids
    )

    print("Inserted", len(chunks), "documents")


def search_documents(query, k=3):

    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "distances"]
    )

    docs = results["documents"][0]
    distances = results["distances"][0]

    # Return (doc, distance) tuples — lower distance = more similar
    return list(zip(docs, distances))