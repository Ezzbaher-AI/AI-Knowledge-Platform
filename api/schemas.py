from pydantic import BaseModel


class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class AskRequest(BaseModel):
    question: str
    chat_id: int

class ChatCreate(BaseModel):
    title: str


class MessageCreate(BaseModel):
    content: str