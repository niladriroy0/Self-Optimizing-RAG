import os
from vectorstore.chroma_store import add_documents
from ingestion.chunker import chunk_document

DATA_DIR = "data"

documents = []

for filename in os.listdir(DATA_DIR):

    if filename.endswith(".txt"):

        filepath = os.path.join(DATA_DIR, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            print(f"Skipping empty file: {filename}")
            continue

        chunks = chunk_document(text)

        documents.extend(chunks)

print(f"Loaded {len(documents)} chunks from data folder")

if len(documents) == 0:
    print("No documents found in data folder.")
    exit()

add_documents(documents)

print("Documents indexed for vector and keyword retrieval.")