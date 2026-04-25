"""
FastAPI backend для AI-нормконтролера v2.
Главное изменение: /api/upload теперь использует PDFChunker (не pypdf+langchain).
Все чанки получают правильные metadata → LLM работает корректно.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

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


# ─── HTML-интерфейс ───────────────────────────────────────────────────────────

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Нормконтролер</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #f4f6f9; color: #333; }
        .container { background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; text-align: center; }
        .subtitle { text-align: center; color: #7f8c8d; margin-bottom: 30px; }
        .tabs { display: flex; justify-content: center; margin-bottom: 20px; border-bottom: 2px solid #eee; }
        .tab { padding: 10px 20px; cursor: pointer; border: none; background: none; font-size: 16px; color: #7f8c8d; }
        .tab.active { color: #3498db; border-bottom: 3px solid #3498db; font-weight: bold; }
        .panel { display: none; }
        .panel.active { display: block; }
        textarea { width: 100%; height: 200px; padding: 15px; border: 1px solid #ddd; border-radius: 8px; resize: vertical; font-size: 14px; box-sizing: border-box; }
        .file-upload { border: 2px dashed #ddd; padding: 30px; text-align: center; border-radius: 8px; cursor: pointer; }
        .file-upload:hover { border-color: #3498db; background: #f0f8ff; }
        input[type="file"] { display: none; }
        button { background: #3498db; color: white; border: none; padding: 12px 25px; border-radius: 6px; cursor: pointer; font-size: 16px; margin-top: 15px; width: 100%; font-weight: 600; }
        button:hover { background: #2980b9; }
        button:disabled { background: #bdc3c7; cursor: not-allowed; }
        .result { margin-top: 25px; padding: 20px; background: #f8f9fa; border-radius: 8px; border-left: 5px solid #bdc3c7; }
        .result.error   { border-left-color: #e74c3c; background: #fdedec; }
        .result.success { border-left-color: #2ecc71; background: #eafaf1; }
        .violation-item { background: white; padding: 10px; margin-top: 10px; border-radius: 4px; border: 1px solid #eee; }
        .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; color: white; }
        .badge-critical { background: #c0392b; }
        .badge-major    { background: #e74c3c; }
        .badge-minor    { background: #f39c12; }
        .spinner { display: inline-block; width: 18px; height: 18px; border: 3px solid rgba(255,255,255,.3); border-radius: 50%; border-top-color: #fff; animation: spin 1s linear infinite; vertical-align: middle; margin-right: 8px; }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
<div class="container">
    <h1>🔍 AI Нормконтролер</h1>
    <p class="subtitle">Автоматическая проверка документации по ГОСТ 2.105-95</p>

    <div class="tabs">
        <button class="tab active" onclick="switchTab('text', event)">Текст</button>
        <button class="tab"        onclick="switchTab('file', event)">Файл (PDF)</button>
    </div>

    <div id="panel-text" class="panel active">
        <textarea id="textInput" placeholder="Вставьте фрагмент текста для проверки..."></textarea>
        <button onclick="analyzeText()" id="btnText">Проверить текст</button>
    </div>

    <div id="panel-file" class="panel">
        <label class="file-upload">
            <input type="file" id="fileInput" accept=".pdf" onchange="updateFileName()">
            <span id="fileName">Нажмите, чтобы выбрать PDF файл</span>
        </label>
        <button onclick="uploadFile()" id="btnFile">Загрузить и анализировать</button>
    </div>

    <div id="result" class="result" style="display:none;"></div>
</div>

<script>
const API = '';

function switchTab(tab, e) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    e.target.classList.add('active');
    document.getElementById('panel-' + tab).classList.add('active');
    document.getElementById('result').style.display = 'none';
}

function updateFileName() {
    const f = document.getElementById('fileInput').files[0];
    if (f) {
        const span = document.getElementById('fileName');
        span.textContent = 'Выбран: ' + f.name;
        span.style.color = '#27ae60';
    }
}

function showResult(html, type) {
    const d = document.getElementById('result');
    d.className = 'result ' + type;
    d.innerHTML = html;
    d.style.display = 'block';
    d.scrollIntoView({ behavior: 'smooth' });
}

async function analyzeText() {
    const text = document.getElementById('textInput').value.trim();
    const btn  = document.getElementById('btnText');
    if (!text) return alert('Введите текст');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>Анализ...';
    try {
        const res  = await fetch(API + '/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, chunk_type: 'text' })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Ошибка сервера');
        renderSingle(data[0]);
    } catch(e) {
        showResult('<strong>Ошибка:</strong> ' + e.message, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Проверить текст';
    }
}

async function uploadFile() {
    const input = document.getElementById('fileInput');
    const btn   = document.getElementById('btnFile');
    if (!input.files.length) return alert('Выберите файл');
    const fd = new FormData();
    fd.append('file', input.files[0]);
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>Загрузка и анализ...';
    try {
        const res  = await fetch(API + '/api/upload', { method: 'POST', body: fd });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Ошибка сервера');
        renderReport(data);
        console.log('Полный отчёт:', data);
    } catch(e) {
        showResult('<strong>Ошибка:</strong> ' + e.message, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Загрузить и анализировать';
    }
}

function severityClass(s) {
    return s === 'critical' ? 'badge-critical' : s === 'major' ? 'badge-major' : 'badge-minor';
}

function renderSingle(r) {
    let html = '<strong>' + (r.has_violation ? '❌ Нарушения найдены' : '✅ Нарушений нет') + '</strong><br>';
    html += 'Уверенность: ' + ((r.confidence || 0) * 100).toFixed(1) + '%<br>';
    (r.violations || []).forEach(v => {
        html += '<div class="violation-item"><span class="badge ' + severityClass(v.severity) + '">' +
                (v.rule_id || '?') + '</span> ' + v.violation_type + '<br><small>' + v.explanation + '</small></div>';
    });
    showResult(html, r.has_violation ? 'error' : 'success');
}

function renderReport(data) {
    let html = '<strong>Файл:</strong> ' + data.filename + '<br>';
    html += 'Страниц: ' + data.total_pages + ' | Чанков: ' + data.chunks_analyzed + '<br>';
    html += '<strong>Нарушений:</strong> ' + data.violations_found + ' | ';
    html += '<strong>Статус:</strong> <span style="color:' + (data.status==='PASS'?'green':'red') + '">' + data.status + '</span><hr>';

    let shown = 0;
    (data.details || []).forEach(chunk => {
        if (!chunk.has_violation || shown >= 5) return;
        shown++;
        html += '<div class="violation-item"><strong>Стр. ' + (chunk.location?.page || '?') +
                ' [' + (chunk.chunk_type || 'text') + ']:</strong><br>';
        html += '<small>' + (chunk.text || '').substring(0, 120) + '…</small>';
        (chunk.violations || []).forEach(v => {
            html += '<div style="margin-top:6px"><span class="badge ' + severityClass(v.severity) + '">' +
                    (v.rule_id || '?') + '</span> ' + v.violation_type + '<br><small>' + v.explanation + '</small></div>';
        });
        html += '</div>';
    });

    if (data.violations_found > 5)
        html += '<p>...и ещё ' + (data.violations_found - 5) + ' нарушений (см. консоль)</p>';
    if (data.violations_found === 0)
        html += '<p>Нарушений не найдено!</p>';

    showResult(html, data.status === 'PASS' ? 'success' : 'error');
}
</script>
</body>
</html>
"""


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

@app.get("/", response_class=HTMLResponse)
async def read_root() -> str:
    """Встроенный HTML-интерфейс."""
    html_file = Path(__file__).parent / "index.html"
    if html_file.exists():
        return html_file.read_text(encoding="utf-8")
    return HTML_CONTENT


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)