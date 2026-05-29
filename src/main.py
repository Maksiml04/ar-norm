from __future__ import annotations
import os
import tempfile
import logging
from typing import Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.chunker import DocumentChunker
from src.orchestrator import NormcontrolPipeline
from src.schemas import ChunkType

logger = logging.getLogger(__name__)

_pipeline_instance: Optional[NormcontrolPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI startup/shutdown handler"""
    global _pipeline_instance
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = "https://openrouter.ai/api/v1"

    _pipeline_instance = NormcontrolPipeline(
        api_key=api_key,
        model=os.getenv("LLM_MODEL", "deepseek/deepseek-chat"),
        base_url=base_url
    )
    logger.info("✅ Pipeline инициализирован через OpenRouter.")
    yield
    # Очистка при выключении (если требуется)


app = FastAPI(title="AI Нормконтролер v3", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])


# Dependency Injection
def get_pipeline() -> NormcontrolPipeline:
    if not _pipeline_instance:
        raise HTTPException(503, "Пайплайн еще не готов")
    return _pipeline_instance


class AnalyzeTextRequest(BaseModel):
    text: str
    chunk_type: str = "TEXT_BODY"


class ChatMessage(BaseModel):
    message: str
    history: list = []


@app.post("/api/upload")
async def upload_and_analyze(file: UploadFile = File(...), pipeline: NormcontrolPipeline = Depends(get_pipeline)) -> \
dict[str, Any]:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Принимаются только PDF файлы")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        chunks = DocumentChunker().chunk_pdf(tmp_path)
        if not chunks:
            raise HTTPException(400, "Скан-копия или пустой файл.")

        # ВАЖНО: Вызов через await! Мы больше не блокируем воркеры сервера.
        results, stats = await pipeline.process_document_async(chunks)

        severity_counts = {"critical": 0, "major": 0, "minor": 0}
        details = []

        for res in results:
            for v in res.violations:
                severity_counts[v.severity.value] += 1
                details.append({
                    "chunk_id": res.chunk_id,
                    "rule_id": v.rule_id,
                    "violation_type": v.violation_type,
                    "explanation": v.explanation,
                    "severity": v.severity.value,
                    "validator": res.validator.value,
                })

        overall_status = "FAIL" if severity_counts["critical"] or severity_counts["major"] else "WARN" if \
        severity_counts["minor"] else "PASS"

        return {
            "filename": file.filename,
            "violations_found": stats.total_violations,
            "severity_counts": severity_counts,
            "status": overall_status,
            "details": details,
        }

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/chat")
async def chat(request: ChatMessage) -> dict[str, Any]:
    import httpx
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return {"response": "⚠️ OPENROUTER_API_KEY не настроен.", "error": True}

    messages = [
        {"role": "system", "content": "Вы — эксперт по ГОСТ. Формат ответов — markdown."},
        *request.history,
        {"role": "user", "content": request.message},
    ]

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "HTTP-Referer": "http://localhost"},
                json={"model": "deepseek/deepseek-chat", "messages": messages, "temperature": 0.7}
            )
            resp.raise_for_status()
            return {"response": resp.json()["choices"][0]["message"]["content"], "error": False}
    except Exception as e:
        return {"response": f"Ошибка: {e}", "error": True}