import json
import os
import chromadb
import urllib.request

# Папка со статьями
KB_DIR = os.path.join(os.path.dirname(__file__), "kb")

# Папка для хранения векторов (ChromaDB будет писать сюда на диск)
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_data")

# Подключение к Ollama
OLLAMA_URL = "http://ollama:11434"

# Модель для векторизации текста
EMBED_MODEL = "nomic-embed-text:latest"

def get_embeddings(texts):
    """Получает векторы текстов от Ollama."""

    data = json.dumps({
        "model": EMBED_MODEL,
        "input": [f"search_document: {text}" for text in texts]
    }).encode("utf-8")

    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/embed",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result["embeddings"]

def load_articles():
    """Читает все JSON-файлы из папки kb и возвращает список статей."""
    articles = []
    for filename in os.listdir(KB_DIR):
        if filename.endswith(".json"):
            with open(os.path.join(KB_DIR, filename), "r", encoding="utf-8") as f:
                articles.append(json.load(f))
    print(f"Загружено статей: {len(articles)}")
    return articles


def build_index():
    """Создаёт векторную базу и индексирует все статьи."""

    # Создаём клиент ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Загружаем статьи
    articles = load_articles()

    if not articles:
        print("Статьи для индексации не найдены.")
        return

    # Готовим данные для загрузки
    documents = []
    metadatas = []
    ids = []

    for article in articles:
        documents.append(article["content"])

        metadatas.append({
            "title": article["title"],
            "category": article["category"],
            "id": article["id"],
        })

        ids.append(article["id"])

    # Получаем эмбеддинги от Ollama
    print("Получение эмбеддингов от Ollama...")

    embeddings = get_embeddings(documents)

    # Удаляем старую коллекцию, если она существует
    # Это необходимо при переходе с MiniLM на Nomic
    if "itil_kb" in [c.name for c in client.list_collections()]:
        client.delete_collection(name="itil_kb")

    # Создаём новую коллекцию
    collection = client.create_collection(name="itil_kb")

    # Загружаем данные и готовые векторы в ChromaDB
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )

    print(f"Проиндексировано статей: {len(articles)}")
    print("Индексация завершена.")

if __name__ == "__main__":
    build_index()