"""
FastAPI backend для AI-нормконтролера.
"""
import os
import sys
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Добавляем корень проекта в PYTHONPATH динамически
current_dir = Path(__file__).resolve().parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src.main import AINormkontroler
from src.logging_config import get_logger

logger = get_logger(__name__)

# Инициализация приложения
app = FastAPI(
    title="AI Нормконтролер",
    description="Система автоматического анализа инженерных документов на соответствие ГОСТ",
    version="1.0.0"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальная переменная для нормконтролера
normkontroler: Optional[AINormkontroler] = None


# --- Модели данных ---

class AnalysisResult(BaseModel):
    """Модель результата анализа одного чанка."""
    chunk_id: str
    has_violation: bool = False
    violations: list = []
    is_correct: bool = True
    confidence: float = 0.0
    location: Optional[dict] = None
    applied_rules: list = []
    error: Optional[str] = None
    warning: Optional[str] = None


class AnalysisRequest(BaseModel):
    """Запрос на анализ текста."""
    text: str
    chunk_type: str = "text"


class HealthCheck(BaseModel):
    """Модель ответа проверки здоровья."""
    status: str
    message: str
    rules_loaded: int = 0


# --- События старта/остановки ---

@app.on_event("startup")
async def startup_event():
    """Инициализация системы при старте."""
    global normkontroler
    logger.info("Запуск системы AI Нормконтролер...")

    try:
        # Пути к данным
        data_dir = current_dir / "data"
        index_path = data_dir / "gost.index"
        meta_path = data_dir / "gost_rules_meta.pkl"

        api_key = os.getenv("OPENROUTER_API_KEY")

        if index_path.exists() and meta_path.exists():
            logger.info(f"Загрузка индекса из {index_path}")
            normkontroler = AINormkontroler.load_from_index(
                index_path=str(index_path),
                meta_path=str(meta_path),
                api_key=api_key
            )
            logger.info(f"Система готова. Загружено правил: {len(normkontroler.rules)}")
        else:
            logger.warning("Индекс не найден. Запустите src/index_builder.py для построения.")
            logger.warning("Сервер запущен в ограниченном режиме (без поиска правил).")

    except Exception as e:
        logger.error(f"Ошибка инициализации: {e}", exc_info=True)
        raise


@app.get("/", response_model=HealthCheck)
async def root():
    """Проверка здоровья сервиса."""
    if normkontroler:
        return HealthCheck(
            status="healthy",
            message="Система готова к работе",
            rules_loaded=len(normkontroler.rules)
        )
    return HealthCheck(
        status="degraded",
        message="Индекс правил не загружен",
        rules_loaded=0
    )


@app.post("/api/analyze", response_model=List[AnalysisResult])
async def analyze_text(request: AnalysisRequest):
    """Анализ текстового фрагмента."""
    if not normkontroler:
        raise HTTPException(status_code=503, detail="Система не инициализирована (нет индекса правил)")

    try:
        logger.info(f"Получен запрос на анализ: тип={request.chunk_type}, длина={len(request.text)}")

        chunk = {
            "id": "manual_request",
            "text": request.text,
            "chunk_type": request.chunk_type,
            "location": {}
        }

        result = normkontroler.analyze_chunk(chunk)

        return [AnalysisResult(**result)]

    except Exception as e:
        logger.error(f"Ошибка при анализе: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Загрузка файла для анализа (заглушка)."""
    if not normkontroler:
        raise HTTPException(status_code=503, detail="Система не инициализирована")

    logger.info(f"Получен файл: {file.filename}, размер: {file.size}")

    # Здесь должна быть логика парсинга PDF
    # Пока возвращаем заглушку
    return {
        "filename": file.filename,
        "status": "received",
        "message": "Функция парсинга PDF в разработке"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)