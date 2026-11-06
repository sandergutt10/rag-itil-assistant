import chromadb
import os
import json
import urllib.request

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_data")

# Подключение к Ollama
OLLAMA_URL = "http://ollama:11434"

# Модель для векторизации запросов
EMBED_MODEL = "nomic-embed-text:latest"


def get_embedding(text: str):
    """Получает вектор запроса от Ollama."""

    data = json.dumps({
        "model": EMBED_MODEL,
        "input": f"search_query: {text}"
    }).encode("utf-8")

    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/embed",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result["embeddings"][0]


def get_collection():
    """Возвращает коллекцию из ChromaDB."""

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    return client.get_collection(name="itil_kb")


def retrieve(query: str, k: int = 3, category: str = None):
    """
    Ищет k ближайших статей к запросу query.
    Если указан category — фильтрует по категории.
    Возвращает список словарей с полями: id, title, content, score.
    """

    collection = get_collection()

    # Получаем эмбеддинг пользовательского запроса
    query_embedding = get_embedding(query)

    # Если фильтр по категории задан — добавляем условие where
    where_filter = {"category": category} if category else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where_filter,
    )

    # Извлекаем результаты в удобном формате
    found = []

    for i in range(len(results["ids"][0])):
        found.append({
            "id": results["ids"][0][i],
            "title": results["metadatas"][0][i]["title"],
            "category": results["metadatas"][0][i]["category"],
            "content": results["documents"][0][i],
            "score": results["distances"][0][i],
        })

    return found


if __name__ == "__main__":
    # Простой тест
    result = retrieve("Как восстановить работу сервиса после сбоя?")

    for r in result:
        print(
            f"  [{r['id']}] {r['title']} "
            f"(score: {r['score']:.4f})"
        )