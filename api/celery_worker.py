from celery import Celery
from pypdf import PdfReader
import os
from vector_store import create_embedding, save_embedding, store_embedding

celery_app = Celery(
    "worker",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/0"
)


@celery_app.task
def process_document(filename: str):
    file_path = f"uploads/{filename}"

    if not os.path.exists(file_path):
        return "File not found"

    print(f"Processing: {filename}")

    # Extract PDF text
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    if not text.strip():
        return "No text extracted"

    print("Text extracted")

    # Save extracted text
    output_path = f"uploads/{filename}.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print("Creating embedding...")

    # Create embedding
    embedding = create_embedding(text)

    print("Embedding created")

    # Store in vector DB
    print("Sending text to vector store...")

    store_embedding(
        doc_id=filename,
        text=text
    )

    print("Stored in vector DB")
    
    # Optional backup
    save_embedding(filename, embedding)

    return f"{filename} embedded successfully"