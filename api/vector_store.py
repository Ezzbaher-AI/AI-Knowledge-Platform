from sentence_transformers import SentenceTransformer
import chromadb
import os
import pickle

model = SentenceTransformer("all-MiniLM-L6-v2")


def get_collection():
    os.makedirs("/app/chroma_db", exist_ok=True)

    client = chromadb.PersistentClient(path="/app/chroma_db")

    return client.get_or_create_collection(name="documents")


def create_embedding(text):
    return model.encode(text).tolist()


def save_embedding(filename, embedding):
    os.makedirs("vectors", exist_ok=True)

    with open(f"vectors/{filename}.pkl", "wb") as f:
        pickle.dump(embedding, f)


def chunk_text(text, chunk_size=1000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def store_embedding(doc_id, text):
    collection = get_collection()

    chunks = chunk_text(text)

    ids = []
    documents = []
    embeddings = []

    for i, chunk in enumerate(chunks):
        print(f"Embedding chunk {i+1}/{len(chunks)}")

        ids.append(f"{doc_id}_{i}")
        documents.append(chunk)
        embeddings.append(create_embedding(chunk))

    print("Adding all chunks to Chroma...")

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings
    )

    print("Chroma add completed")