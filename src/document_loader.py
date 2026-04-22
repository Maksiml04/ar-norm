"""
FastAPI Backend для AI-нормконтролера
"""
import os
import sys
import uuid
import tempfile
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Добавляем src в путь для импорта
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Настраиваем логирование ПЕРЕД импортом остальных модулей
os.environ.setdefault('LOG_TO_FILE', 'true')
os.environ.setdefault('LOG_LEVEL', 'INFO')

from pdf_parser import PDFChunker
try:
    from main import AINormkontroler
except ImportError:
    AINormkontroler = None

# Импортируем логгер после настройки
from logging_config import get_logger

logger = get_logger(__name__)


app = FastAPI(
    title="AI Нормконтролер",
    description="Сервис для анализа инженерных документов на соответствие ГОСТ",
    version="1.0.0"
)

# CORS для фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальный экземпляр нормконтролера (инициализируется при старте)
normkontroler = None


class AnalysisRequest(BaseModel):
    """Модель запроса на анализ"""
    file_path: str
    custom_question: str = None


class AnalysisResult(BaseModel):
    """Модель результата анализа"""
    file_id: str
    filename: str
    total_chunks: int
    total_violations: int
    violations_by_category: Dict[str, int]
    violations: List[Dict[str, Any]]
    summary: str


@app.on_event("startup")
async def startup_event():
    """Инициализация нормконтролера при старте приложения"""
    global normkontroler

    logger.info("=" * 60)
    logger.info("AI НОРМКОНТРОЛЕР - ЗАПУСК BACKEND")
    logger.info("=" * 60)

    # Монтируем статические файлы (фронтенд)
    frontend_path = os.path.join(os.path.dirname(__file__), '..', 'frontend')
    if os.path.exists(frontend_path):
        logger.info(f"Монтирование фронтенда из: {frontend_path}")
        app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
    else:
        logger.warning(f"Фронтенд не найден: {frontend_path}")

    # Проверяем наличие индекса - ищем в нескольких местах (для Docker и локальной разработки)
    possible_index_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'gost.index'),
        os.path.join(os.path.dirname(__file__), '..', 'data', 'gost.index'),
        '/app/data/gost.index',
    ]
    possible_meta_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'gost_rules_meta.pkl'),
        os.path.join(os.path.dirname(__file__), '..', 'data', 'gost_rules_meta.pkl'),
        '/app/data/gost_rules_meta.pkl',
    ]

    index_path = None
    meta_path = None

    for path in possible_index_paths:
        if os.path.exists(path):
            index_path = path
            logger.info(f"Найден индекс FAISS: {path}")
            break

    for path in possible_meta_paths:
        if os.path.exists(path):
            meta_path = path
            logger.info(f"Найдены метаданные: {path}")
            break

    if not index_path or not meta_path:
        logger.warning("Индекс FAISS не найден. Запустите src/index_builder.py для создания.")
        logger.warning("Анализ будет работать в ограниченном режиме.")
        normkontroler = None
    else:
        try:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if api_key:
                logger.info("API ключ OpenRouter найден")
            else:
                logger.warning("API ключ OpenRouter не установлен (LLM-анализ недоступен)")

            logger.info(f"Загрузка нормконтролера из индекса: {index_path}")
            normkontroler = AINormkontroler.load_from_index(
                index_path=index_path,
                meta_path=meta_path,
                api_key=api_key
            )
            logger.info(f"✓ AI Нормконтролер успешно инициализирован (индекс: {index_path})")
        except Exception as e:
            logger.error(f"Ошибка инициализации нормконтролера: {e}", exc_info=True)
            normkontroler = None

    logger.info("-" * 60)
    logger.info("BACKEND ГОТОВ К РАБОТЕ")
    logger.info("-" * 60)


@app.get("/api/health")
async def health_check():
    """Проверка состояния сервиса"""
    # Проверяем наличие индекса в возможных локациях
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'gost.index'),
        os.path.join(os.path.dirname(__file__), '..', 'data', 'gost.index'),
        '/app/data/gost.index',
    ]
    index_exists = any(os.path.exists(p) for p in possible_paths)

    return {
        "status": "healthy",
        "normkontroler_ready": normkontroler is not None,
        "index_exists": index_exists
    }


# Убираем root endpoint - теперь там фронтенд


@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze_document(
    file: UploadFile = File(..., description="PDF файл для анализа"),
    question: str = Form(None, description="Дополнительный вопрос для анализа")
):
    """
    Анализ загруженного PDF документа на соответствие ГОСТ

    - **file**: PDF файл для анализа
    - **question**: Опциональный вопрос пользователя
    """
    logger.info(f"Получен запрос на анализ файла: {file.filename}")

    try:
        # Проверка типа файла
        if not file.filename.lower().endswith('.pdf'):
            logger.warning(f"Отклонён файл неверного типа: {file.filename}")
            raise HTTPException(status_code=400, detail="Только PDF файлы поддерживаются")

        # Проверка размера файла (макс 50MB)
        content = await file.read()
        file_size = len(content)
        logger.info(f"Размер файла: {file_size / 1024 / 1024:.2f} MB")

        if file_size > 50 * 1024 * 1024:
            logger.warning(f"Файл слишком большой: {file_size} байт")
            raise HTTPException(status_code=400, detail="Размер файла не должен превышать 50MB")

        # Создаём временный файл
        temp_dir = tempfile.gettempdir()
        file_id = str(uuid.uuid4())
        temp_file_path = os.path.join(temp_dir, f"{file_id}_{file.filename}")

        try:
            # Сохраняем файл
            logger.debug(f"Сохранение временного файла: {temp_file_path}")
            with open(temp_file_path, "wb") as f:
                f.write(content)

            # Парсим PDF
            logger.info("Парсинг PDF документа")
            chunker = PDFChunker()
            chunks_obj = chunker.chunk_pdf(temp_file_path)

            # Конвертируем в формат словарей для совместимости
            chunks = [chunk.to_dict() for chunk in chunks_obj]

            if not chunks:
                logger.error("Не удалось извлечь текст из PDF")
                raise HTTPException(status_code=400, detail="Не удалось извлечь текст из PDF")

            logger.info(f"Извлечено {len(chunks)} чанков из PDF")

            # Анализируем чанки
            if normkontroler:
                logger.info("Начало анализа чанков")
                results = []
                for i, chunk in enumerate(chunks, 1):
                    logger.debug(f"Анализ чанка {i}/{len(chunks)}")
                    result = normkontroler.analyze_chunk(chunk)
                    results.append(result)

                # Агрегируем результаты
                violations = [r for r in results if r.get('has_violation', False)]
                violations_by_category = {}

                for v in violations:
                    category = v.get('category', 'other')
                    violations_by_category[category] = violations_by_category.get(category, 0) + 1

                # Генерируем сводку
                summary = generate_summary(violations, len(chunks))

                logger.info(f"Анализ завершён: найдено {len(violations)} нарушений")

                return AnalysisResult(
                    file_id=file_id,
                    filename=file.filename,
                    total_chunks=len(chunks),
                    total_violations=len(violations),
                    violations_by_category=violations_by_category,
                    violations=violations,
                    summary=summary
                )
            else:
                logger.warning("Нормконтролер не инициализирован, работа в ограниченном режиме")
                # Режим без LLM - только базовая проверка
                return AnalysisResult(
                    file_id=file_id,
                    filename=file.filename,
                    total_chunks=len(chunks),
                    total_violations=0,
                    violations_by_category={},
                    violations=[],
                    summary="Сервис работает в ограниченном режиме. Индекс правил не найден."
                )

        finally:
            # Удаляем временный файл
            if os.path.exists(temp_file_path):
                logger.debug(f"Удаление временного файла: {temp_file_path}")
                os.remove(temp_file_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Критическая ошибка при анализе документа: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")


def generate_summary(violations: List[Dict], total_chunks: int) -> str:
    """Генерирует текстовую сводку по результатам анализа"""
    if not violations:
        return f"✓ Нарушений не найдено. Проанализировано {total_chunks} блоков текста."

    total = len(violations)
    critical = sum(1 for v in violations if v.get('criticality') == 'critical')
    major = sum(1 for v in violations if v.get('criticality') == 'major')
    minor = sum(1 for v in violations if v.get('criticality') == 'minor')

    summary_parts = [
        f"Найдено {total} нарушений в {total_chunks} блоках текста.",
    ]

    if critical > 0:
        summary_parts.append(f"❗ {critical} критических ошибок")
    if major > 0:
        summary_parts.append(f"⚠️  {major} серьёзных нарушений")
    if minor > 0:
        summary_parts.append(f"ℹ️  {minor} мелких замечаний")

    return " ".join(summary_parts)


@app.post("/api/analyze/text")
async def analyze_text(
    text: str = Form(..., description="Текст для анализа"),
    question: str = Form(None, description="Дополнительный вопрос")
):
    """Анализ текста без загрузки файла"""
    logger.info("Получен запрос на анализ текста")

    try:
        if not text.strip():
            logger.warning("Получен пустой текст для анализа")
            raise HTTPException(status_code=400, detail="Текст не может быть пустым")

        # Парсим текст
        chunker = PDFChunker()
        chunks_obj = chunker.chunk_pdf(text=text) if hasattr(chunker, 'chunk_pdf') and text else []
        # Для текста используем простой подход - создаем один чанк
        if not chunks_obj and text:
            chunks_obj = [type('Chunk', (), {'to_dict': lambda self: {
                "id": "text_chunk_1",
                "text": text,
                "chunk_type": "text",
                "location": {"page": 1},
                "metadata": {},
                "context_query": "правила оформления технической документации ГОСТ"
            }})()]
        chunks = [c.to_dict() if hasattr(c, 'to_dict') else c for c in chunks_obj]

        logger.info(f"Разбито на {len(chunks)} чанков")

        if normkontroler:
            results = []
            for i, chunk in enumerate(chunks, 1):
                logger.debug(f"Анализ чанка {i}/{len(chunks)}")
                result = normkontroler.analyze_chunk(chunk)
                results.append(result)

            violations = [r for r in results if r.get('has_violation', False)]
            violations_by_category = {}

            for v in violations:
                category = v.get('category', 'other')
                violations_by_category[category] = violations_by_category.get(category, 0) + 1

            summary = generate_summary(violations, len(chunks))

            logger.info(f"Анализ текста завершён: найдено {len(violations)} нарушений")

            return {
                "total_chunks": len(chunks),
                "total_violations": len(violations),
                "violations_by_category": violations_by_category,
                "violations": violations,
                "summary": summary
            }
        else:
            logger.warning("Нормконтролер не инициализирован")
            return {
                "total_chunks": len(chunks),
                "total_violations": 0,
                "violations_by_category": {},
                "violations": [],
                "summary": "Сервис работает в ограниченном режиме."
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при анализе текста: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    # Запуск с автоперезагрузкой и хостом 0.0.0.0 для доступа извне
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)