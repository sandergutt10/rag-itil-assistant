from fastapi import FastAPI
from pydantic import BaseModel
from generator import generate_answer
from retriever import retrieve

app = FastAPI(title="ITIL RAG Bot", version="1.0")


class AskRequest(BaseModel):
    query: str
    category: str | None = None
    k: int = 3


@app.get("/health")
def health():
    """Проверка, что сервис жив."""
    return {"status": "ok"}


@app.post("/ask")
def ask(request: AskRequest):
    """Главный эндпоинт: вопрос → ответ + источники."""
    result = generate_answer(
        query=request.query,
        k=request.k,
        category=request.category,
    )
    return result


@app.post("/search")
def search(request: AskRequest):
    """Только поиск, без генерации ответа."""
    results = retrieve(query=request.query, k=request.k, category=request.category)
    return {"results": results}
