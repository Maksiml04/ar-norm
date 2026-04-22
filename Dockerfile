FROM python:3.11-slim

# Установка рабочей директории
WORKDIR /app

# Установка системных зависимостей (для компиляции некоторых пакетов)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Копирование требований и установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Создание директорий для логов и данных
RUN mkdir -p logs data

# Переменные окружения по умолчанию
ENV PYTHONPATH=/app
ENV LOG_LEVEL=INFO

# Порт приложения
EXPOSE 8000

# Команда запуска
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]