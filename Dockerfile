# Используем легкий образ Python
FROM python:3.11-slim

# Устанавливаем системные зависимости (нужны для некоторых библиотек)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория внутри контейнера
WORKDIR /app

# Копируем зависимости и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь код проект
COPY . .

# Открываем порт (по умолчанию 8000 для FastAPI)
EXPOSE 8000

# Команда запуска сервера
# Важно: host 0.0.0.0 обязателен для работы внутри контейнера
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]