# Базовый образ — Python
FROM python:3.13-slim

# Рабочая папка внутри контейнера
WORKDIR /app

# Копируем requirements и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь код
COPY . .

# Открываем порт
EXPOSE 8000

# Запускаем индексацию, затем стартуем API
CMD ["sh", "-c", "python indexer.py && uvicorn app:app --host 0.0.0.0 --port 8000"]