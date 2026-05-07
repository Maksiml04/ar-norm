"""
FastAPI backend для AI-нормконтролера v2.
Главное изменение: /api/upload теперь использует PDFChunker (не pypdf+langchain).
Все чанки получают правильные metadata → LLM работает корректно.
"""
from __future__ import annotations
import httpx
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

import os
import shutil
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
# Корень проекта в PYTHONPATH
current_dir = Path(__file__).resolve().parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src.main import AINormkontroler
from src.pdf_parser import PDFChunker          # ← используем наш парсер
from src.logging_config import get_logger

logger = get_logger(__name__)

# ─── Приложение ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Нормконтролер",
    description="Автоматическая проверка технической документации на соответствие ГОСТ",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

normkontroler: Optional[AINormkontroler] = None


# ─── Pydantic-модели ──────────────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    text: str
    chunk_type: str = "text"
    # Опциональные метаданные для ручного запроса
    is_centered: Optional[bool] = None
    is_bold: Optional[bool] = None
    font_size: Optional[float] = None


class AnalysisResult(BaseModel):
    chunk_id: str
    has_violation: bool = False
    violations: list = []
    is_correct: bool = True
    confidence: float = 0.0
    location: Optional[dict] = None
    applied_rules: list = []
    text: Optional[str] = None
    error: Optional[str] = None


class HealthCheck(BaseModel):
    status: str
    message: str
    rules_loaded: int = 0

class ChatMessage(BaseModel):
    message: str
    history: list = []


# ─── HTML-интерфейс ───────────────────────────────────────────────────────────




# ─── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event() -> None:
    global normkontroler
    logger.info("Запуск AI Нормконтролер...")

    data_dir   = current_dir / "data"
    index_path = data_dir / "gost.index"
    meta_path  = data_dir / "gost_rules_meta.pkl"
    api_key    = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        logger.error("OPENROUTER_API_KEY не установлен!")
    else:
        logger.info(f"API ключ: {api_key[:5]}…")

    if index_path.exists() and meta_path.exists():
        try:
            normkontroler = AINormkontroler.load_from_index(
                index_path=str(index_path),
                meta_path=str(meta_path),
                api_key=api_key,
            )
            logger.info(f"✅ Система готова. Правил: {len(normkontroler.rules)}")
        except Exception as e:
            logger.error(f"Ошибка инициализации: {e}", exc_info=True)
            raise
    else:
        logger.warning("Индекс не найден. Запуск в деградированном режиме.")
        normkontroler = None


# ─── Эндпоинты ────────────────────────────────────────────────────────────────

# @app.get("/", response_class=HTMLResponse)
# async def read_root() -> str:
#     """Встроенный HTML-интерфейс."""
#     html_file = Path(__file__).parent / "index.html"
#     if html_file.exists():
#         return html_file.read_text(encoding="utf-8")
#     return HTML_CONTENT


@app.get("/health", response_model=HealthCheck)
async def health_check() -> HealthCheck:
    if normkontroler:
        return HealthCheck(
            status="healthy",
            message="Готов к работе",
            rules_loaded=len(normkontroler.rules),
        )
    return HealthCheck(status="degraded", message="Индекс не загружен", rules_loaded=0)


@app.post("/api/analyze", response_model=list[AnalysisResult])
async def analyze_text(request: AnalysisRequest) -> list[AnalysisResult]:
    """Анализирует переданный текст (ручной ввод)."""
    if normkontroler is None:
        raise HTTPException(503, "Система не инициализирована")

    # Собираем чанк в том же формате что PDFChunker
    chunk: dict[str, Any] = {
        "id":         "manual_request",
        "text":       request.text,
        "chunk_type": request.chunk_type,
        "location":   {},
        "metadata": {
            # Пробрасываем метаданные если переданы
            "known_abbreviations": {},
            **({"is_centered": request.is_centered}
               if request.is_centered is not None else {}),
            **({"is_bold": request.is_bold}
               if request.is_bold is not None else {}),
            **({"font_size": request.font_size}
               if request.font_size is not None else {}),
        },
        "context_query": "",
    }

    try:
        result = normkontroler.analyze_chunk(chunk)
        return [AnalysisResult(**{
            k: result.get(k)
            for k in AnalysisResult.model_fields
        })]
    except Exception as e:
        logger.error(f"Ошибка анализа: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.post("/api/upload")
async def upload_and_analyze(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Загружает PDF, разбивает через PDFChunker и анализирует каждый чанк.

    Теперь чанки имеют:
    - правильный chunk_type (section_header, table_ref, …)
    - metadata с is_centered, is_bold, known_abbreviations
    - context_query для FAISS-retriever
    """
    if normkontroler is None:
        raise HTTPException(503, "Система не инициализирована")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Только PDF файлы")

    tmp_path = None
    try:
        # Сохраняем во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # ── Разбивка через PDFChunker (не pypdf!) ─────────────────────────────
        chunker = PDFChunker()
        doc_chunks = chunker.chunk_pdf(tmp_path)

        if not doc_chunks:
            raise HTTPException(400, "Не удалось извлечь текст (возможно, скан)")

        logger.info(f"Чанков извлечено: {len(doc_chunks)}")

        # ── Анализ ────────────────────────────────────────────────────────────
        # Используем батч-метод: один encode-вызов для всех запросов
        chunk_dicts = [c.to_dict() for c in doc_chunks]
        results     = normkontroler.analyze_chunks_batch(chunk_dicts)

        violations_count = sum(1 for r in results if r.get("has_violation"))

        return {
            "filename":        file.filename,
            "total_pages":     max(
                (c.location.get("page", 0) for c in doc_chunks), default=0
            ),
            "chunks_analyzed": len(doc_chunks),
            "violations_found": violations_count,
            "status":          "FAIL" if violations_count > 0 else "PASS",
            "details":         results,
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


@app.post("/api/chat")
async def chat_with_gost(request: ChatMessage) -> dict[str, Any]:
    """
    Обработчик чата. Использует DeepSeek API через бэкенд,
    чтобы не светить ключ на фронтенде.
    """
    import httpx

    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")  # Лучше хранить в .env
    if not deepseek_api_key:
        # Fallback если ключа нет, можно вернуть ошибку или моковый ответ
        return {
            "response": "⚠️ Ошибка: Не настроен API ключ DeepSeek на сервере.",
            "error": True
        }

    system_prompt = """Вы — эксперт по ГОСТ 2.105-95 (общие требования к текстовым документам) и ГОСТ 2.304-81 (шрифты для чертежей). Отвечайте на вопросы инженеров, ссылайтесь на конкретные пункты стандартов. Формат ответов — markdown."""

    messages = [
        {"role": "system", "content": system_prompt},
        *request.history,
        {"role": "user", "content": request.message}
    ]

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {deepseek_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 1024
                }
            )
            response.raise_for_status()
            data = response.json()
            ai_text = data["choices"][0]["message"]["content"]
            return {"response": ai_text, "error": False}
    except Exception as e:
        logger.error(f"Ошибка DeepSeek API: {e}")
        return {"response": f"Ошибка соединения с AI: {str(e)}", "error": True}


# Создание папки static если нет
# STATIC_DIR = Path(__file__).parent / "static"
# STATIC_DIR.mkdir(exist_ok=True)
#
# # Монтирование статики
# app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets"), html=True), name="assets")
#
# @app.get("/{full_path:path}")
# async def serve_react(full_path: str):
#     """Отдаёт index.html для всех путей, чтобы работал React Router (если будет)"""
#     index_file = STATIC_DIR / "index.html"
#     if index_file.exists():
#         return FileResponse(str(index_file))
#     return HTMLResponse(content="Frontend not built yet. Run 'npm run build' in frontend folder.", status_code=503)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)