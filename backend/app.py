#from __future__ import annotations
import httpx
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional, List
from slowapi import Limiter
from slowapi.util import get_remote_address
from dotenv import load_dotenv
import os
limiter = Limiter(key_func=get_remote_address)
from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import openai

# Подключаем наш чанкер и пайплайн
from src.chunker import DocumentChunker, chunk_docx
from src.orchestrator import NormcontrolPipeline
from src.schemas import DocumentChunk
from src.logging_config import get_logger

# Корень проекта в PYTHONPATH
current_dir = Path(__file__).resolve().parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))



# Загружаем переменные из .env файла
load_dotenv()
logger = get_logger(__name__)

# ─── Приложение ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Нормконтролер",
    description="Автоматическая проверка технической документации на соответствие ГОСТ (Native Pipeline v3.0)",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline: Optional[NormcontrolPipeline] = None


# ─── Pydantic-модели ──────────────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    text: str
    chunk_type: str = "TEXT_BODY"
    is_centered: Optional[bool] = None
    is_bold: Optional[bool] = None
    font_size: Optional[float] = None


class ViolationDetail(BaseModel):
    rule_id: str
    violation_type: str
    explanation: str
    severity: str
    step_by_step_analysis: Optional[str] = None


class AnalysisResult(BaseModel):
    chunk_id: str
    validator: str
    has_violation: bool = False
    violations: List[ViolationDetail] = []
    error: Optional[str] = None
    location: Optional[dict] = None


class HealthCheck(BaseModel):
    status: str
    message: str
    pipeline_ready: bool


class ChatMessage(BaseModel):
    message: str
    history: list = []

class UploadResponse(BaseModel):
    filename: str
    total_pages: int
    chunks_analyzed: int
    violations_found: int
    status: str
    details: list[dict]


# ─── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event() -> None:
    global pipeline
    logger.info("Запуск гибридного пайплайна AI Нормконтролера...")

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        logger.warning("API ключ не установлен! LLM-анализ работать не будет.")
    else:
        logger.info(f"API ключ найден: {api_key[:5]}…")

    try:
        pipeline = NormcontrolPipeline()
        logger.info("✅ Система готова. Rule-Based и LLM движки загружены.")
    except Exception as e:
        logger.error(f"Ошибка инициализации пайплайна: {e}", exc_info=True)
        raise


# ─── Эндпоинты ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthCheck)
async def health_check() -> HealthCheck:
    if pipeline:
        return HealthCheck(
            status="healthy",
            message="Пайплайн готов к работе",
            pipeline_ready=True,
        )
    return HealthCheck(status="degraded", message="Пайплайн не загружен", pipeline_ready=False)


@app.post("/api/analyze", response_model=list[AnalysisResult])
async def analyze_text(request: AnalysisRequest) -> list[AnalysisResult]:
    """Анализирует переданный текст (ручной ввод с фронтенда)."""
    if pipeline is None:
        raise HTTPException(503, "Система не инициализирована")

    manual_chunk = DocumentChunk(
        chunk_id="manual_request",
        chunk_type=request.chunk_type,
        text=request.text,
        location={},
        metadata={
            "is_centered": request.is_centered,
            "is_bold": request.is_bold,
            "font_size": request.font_size
        }
    )

    try:
        results = pipeline.process_document_async([manual_chunk])
        return [AnalysisResult(**res.model_dump()) for res in results]
    except Exception as e:
        logger.error(f"Ошибка анализа: {e}", exc_info=True)
        raise HTTPException(500, str(e))



@app.post("/api/upload", response_model=UploadResponse)
@limiter.limit("10/minute")
async def upload_and_analyze(request: Request, file: UploadFile = File(...)):
    if pipeline is None:
        raise HTTPException(503, "Система не инициализирована")

    filename = file.filename.lower()
    if not (filename.endswith(".pdf") or filename.endswith(".docx")):
        raise HTTPException(400, "Только PDF и DOCX файлы")

    tmp_path = None
    try:
        # Сохраняем файл во временную директорию
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # 1. Извлекаем текст в зависимости от формата
        if filename.endswith(".pdf"):
            chunker = DocumentChunker()
            pipeline_chunks = chunker.chunk_pdf(tmp_path)
        else:
            # Предполагаем, что функция chunk_docx импортирована
            pipeline_chunks = chunk_docx(tmp_path)

        if not pipeline_chunks:
            raise HTTPException(400, "Не удалось извлечь текст из файла")

        logger.info(f"Чанков извлечено: {len(pipeline_chunks)}")

        # 2. Анализ
        # ЗАМЕНИТЕ 'ВАШ_МЕТОД' на то, что нашли через dir(pipeline)
        # Если метод async, добавьте await
        # 2. Анализ
        # 2. Анализ
        results_tuple = await pipeline.process_document_async(pipeline_chunks)
        all_results = results_tuple[0]  # Список всех найденных нарушений

        # 3. Формирование ответа
        violations_count = len(all_results)  # Считаем количество ошибок

        real_total_pages = (
            pipeline_chunks[0].location.get("page_total")
            if pipeline_chunks and pipeline_chunks[0].location
            else 0
        )

        return {
            "filename": file.filename,
            "total_pages": real_total_pages,

            # Исправлено имя поля, чтобы оно соответствовало модели UploadResponse
            "chunks_analyzed": len(pipeline_chunks),

            "violations_found": violations_count,
            "status": "FAIL" if violations_count > 0 else "PASS",
            "details": [r.model_dump() for r in all_results]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        raise HTTPException(500, str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass






# ... ваш существующий код ...

@app.post("/api/chat")
async def chat_with_gost(request: ChatMessage) -> dict:
    """Чат с AI-нормоконтролёром (через OpenRouter/OpenAI)."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return {"response": "⚠️ Ошибка: Не настроен API ключ (OPENROUTER_API_KEY) в .env", "error": True}

    # Формируем системный промпт
    system_prompt = (
        "Ты строгий эксперт-нормоконтролёр по ГОСТ 2.105-95, ГОСТ 2.304-81 и ГОСТ Р 7.0.100. "
        "Отвечай чётко, ссылайся на конкретные пункты стандартов. Формат ответов — Markdown."
    )

    messages = [{"role": "system", "content": system_prompt}]

    # Безопасно добавляем историю (поддерживаем оба формата от фронтенда)
    for m in request.history:
        role = m.get("role", "user")
        content = m.get("content") or m.get("text", "")
        if content:
            messages.append({"role": role, "content": content})

    # Текущий вопрос
    messages.append({"role": "user", "content": request.message})

    try:
        # Синхронный клиент оборачиваем в asyncio.to_thread, чтобы не блокировать FastAPI
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={"HTTP-Referer": "http://localhost:8000", "X-Title": "GOST-Chat"}
        )

        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="deepseek/deepseek-v4-pro",
            messages=messages,
            temperature=0.7,
            max_tokens=1024
        )

        ai_text = response.choices[0].message.content or "..."
        return {"response": ai_text, "error": False}

    except Exception as e:
        logger.error(f"Chat API error: {e}")
        return {"response": f"⚠️ Ошибка соединения: {str(e)}", "error": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)