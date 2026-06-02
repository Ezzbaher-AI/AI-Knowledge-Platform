from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.security import (
    HTTPBearer,
    HTTPAuthorizationCredentials,
    OAuth2PasswordBearer,
    OAuth2PasswordRequestForm
)
from sqlalchemy import text
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

import os
import chromadb
from openai import OpenAI

from database import engine, SessionLocal
from models import Base, User
from schemas import UserCreate, AskRequest
from security import hash_password, verify_password
from auth import create_access_token, verify_token
from celery_worker import process_document
from vector_store import create_embedding
from memory import save_message, get_history, get_last_assistant_message
from memory import list_user_chats

# -----------------------------
# App setup
# -----------------------------
app = FastAPI(title="AI Knowledge Platform")

Base.metadata.create_all(bind=engine)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("documents")

security = HTTPBearer()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


# -----------------------------
# Database
# -----------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -----------------------------
# Authentication
# -----------------------------
def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    token = credentials.credentials
    payload = verify_token(token)

    if payload is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token"
        )

    return payload


# -----------------------------
# Root routes
# -----------------------------
@app.get("/")
def root():
    return {"message": "AI Knowledge Platform Running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/db-check")
def db_check():
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))
        return {"database_connected": result.scalar()}


# -----------------------------
# Register
# -----------------------------
@app.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    new_user = User(
        username=user.username,
        email=user.email,
        password=hash_password(user.password)
    )

    try:
        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        return {
            "message": "User registered successfully",
            "user_id": new_user.id
        }

    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=400,
            detail="Username or email already exists"
        )


# -----------------------------
# Login
# -----------------------------
@app.post("/login")
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    db_user = db.query(User).filter(
        User.email == form_data.username
    ).first()

    if not db_user or not verify_password(
        form_data.password,
        db_user.password
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )

    token = create_access_token({"sub": db_user.email})

    return {
        "access_token": token,
        "token_type": "bearer"
    }


# -----------------------------
# Profile
# -----------------------------
@app.get("/profile")
def profile(current_user=Depends(get_current_user)):
    return {
        "message": "Protected route accessed",
        "user": current_user
    }


# -----------------------------
# Upload PDF
# -----------------------------
@app.post("/upload")
def upload_file(
    file: UploadFile = File(...),
    current_user=Depends(get_current_user)
):
    os.makedirs("uploads", exist_ok=True)

    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    task = process_document.delay(file.filename)

    return {
        "message": "File uploaded and queued",
        "task_id": task.id
    }


# -----------------------------
# Ask Questions
# -----------------------------
@app.post("/ask")
def ask_question(
    request: AskRequest,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        db_user = db.query(User).filter(
            User.email == current_user["sub"]
        ).first()

        if not db_user:
            raise HTTPException(
                status_code=404,
                detail="User not found"
            )

        message = request.question.lower()

        followups = [
            "simplify it",
            "simplify",
            "make it simpler",
            "explain simpler",
            "summarize that"
        ]

        # Handle follow-up requests
        if message in followups:
            last_answer = get_last_assistant_message(
                db_user.id,
                request.chat_id
            )

            if not last_answer:
                return {
                    "answer": "No previous answer found."
                }

            prompt = f"Rewrite this in simpler words:\n\n{last_answer}"

            history = get_history(db_user.id, request.chat_id)

        else:
            question_embedding = create_embedding(request.question)

            results = collection.query(
                query_embeddings=[question_embedding],
                n_results=3
            )

            context = "\n".join(results["documents"][0])

            prompt = f"""
Context:
{context}

Question:
{request.question}
"""

        history = get_history(db_user.id, request.chat_id)

        save_message(
            db_user.id,
            request.chat_id,
            "user",
            request.question
        )

        answer_response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=history + [
                {
                    "role": "system",
                    "content": """
You are a helpful assistant.
Always answer based on the conversation context.
If the user says 'simplify', simplify only your last answer.
"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = answer_response.choices[0].message.content
        save_message(
            db_user.id,
            request.chat_id,
            "assistant",
            answer
        )

        return {
            "question": request.question,
            "answer": answer
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    
@app.get("/chats")
def get_chats(current_user=Depends(get_current_user),
              db: Session = Depends(get_db)):

    db_user = db.query(User).filter(
        User.email == current_user["sub"]
    ).first()

    return {
        "chats": list_user_chats(db_user.id)
    }