from retriever import retrieve
import json
import urllib.request

# Подключение к Ollama
OLLAMA_URL = "http://ollama:11434"

# Модель для генерации ответов
LLM_MODEL = "gemma2:2b"


def build_prompt(query: str, fragments: list) -> str:
    """Собирает промпт для LLM из найденных фрагментов."""

    context_block = ""

    for i, frag in enumerate(fragments, 1):
        context_block += (
            f"[{i}] (id: {frag['id']}) "
            f"{frag['title']}: {frag['content']}\n\n"
        )

    prompt = f"""Ты — агент службы поддержки, работающий по ITIL.

Ответь на вопрос пользователя, опираясь ТОЛЬКО на приведённые ниже фрагменты базы знаний.

Правила:
1. Не используй информацию, которой нет в базе знаний.
2. Объясняй ответ понятно и последовательно.
3. Обязательно указывай ID использованных статей в формате [id].
4. Если информации недостаточно, скажи об этом прямо и предложи создать заявку.

Фрагменты базы знаний:
{context_block}

Вопрос: {query}

Ответ:"""

    return prompt


def generate_answer(query: str, k: int = 3, category: str = None):
    """
    Полный цикл RAG: поиск + генерация ответа через Ollama.
    Возвращает словарь с ответом и списком источников.
    """

    # Получаем релевантные статьи из ChromaDB
    fragments = retrieve(query, k=k, category=category)

    # Формируем промпт
    prompt = build_prompt(query, fragments)

    # Отправляем запрос в Ollama
    data = json.dumps({
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False
    }).encode("utf-8")

    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.loads(response.read().decode("utf-8"))

    # Получаем ответ модели
    answer = result["response"].strip()

    # Формируем список источников
    sources = [
        {
            "id": f["id"],
            "title": f["title"],
            "score": f["score"]
        }
        for f in fragments
    ]

    return {
        "answer": answer,
        "sources": sources,
        "prompt": prompt
    }


if __name__ == "__main__":
    result = generate_answer("Как избежать повторения инцидента?")

    print("ОТВЕТ:")
    print(result["answer"])

    print("\nИСТОЧНИКИ:")

    for s in result["sources"]:
        print(
            f"  [{s['id']}] {s['title']} "
            f"(score: {s['score']:.4f})"
        )