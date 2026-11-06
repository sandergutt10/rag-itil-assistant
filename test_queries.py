from retriever import retrieve

# Тестовые запросы и ожидаемые статьи
TESTS = [
    {"query": "Как восстановить работу сервиса после сбоя?", "expected": "incident-01"},
    {"query": "Как найти корневую причину повторяющихся ошибок?", "expected": "problem-01"},
    {"query": "Как согласовать изменение в инфраструктуре?", "expected": "change-01"},
    {"query": "Какие задачи у службы поддержки первой линии?", "expected": "servicedesk-01"},
    {"query": "Как вести базу знаний для поддержки?", "expected": "knowledge-01"},
]

def run_tests(k=3):
    """Считает Recall@k и MRR."""
    hits = 0
    mrr_sum = 0.0

    for test in TESTS:
        results = retrieve(test["query"], k=k)
        found_ids = [r["id"] for r in results]

        if test["expected"] in found_ids:
            hits += 1
            rank = found_ids.index(test["expected"]) + 1
            mrr_sum += 1.0 / rank

        print(f"  Запрос: {test['query']}")
        print(f"  Ожидается: {test['expected']}")
        print(f"  Найдено: {found_ids}")
        print(f"  {'✅ HIT' if test['expected'] in found_ids else '❌ MISS'}")
        print()

    total = len(TESTS)
    recall = hits / total
    mrr = mrr_sum / total

    print(f"Recall@{k}: {recall:.2f} ({hits}/{total})")
    print(f"MRR: {mrr:.4f}")
    return recall, mrr

if __name__ == "__main__":
    run_tests(k=3)
