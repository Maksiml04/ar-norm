"""
FastAPI backend для AI-нормконтролера.
Полная версия с поддержкой загрузки PDF, анализа и встроенным UI.
"""
import os
import sys
import tempfile
from typing import List, Optional, Dict, Any
from pathlib import Path
from fastapi.responses import JSONResponse, HTMLResponse

# Добавляем корень проекта в PYTHONPATH
current_dir = Path(__file__).resolve().parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Импорты из проекта
from src.main import AINormkontroler
from src.logging_config import get_logger

# Сторонние библиотеки
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None

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

# Глобальная переменная
normkontroler: Optional[AINormkontroler] = None

# --- Модели данных ---
class AnalysisResult(BaseModel):
    chunk_id: str
    has_violation: bool = False
    violations: list = []
    is_correct: bool = True
    confidence: float = 0.0
    location: Optional[dict] = None
    applied_rules: list = []
    error: Optional[str] = None

class AnalysisRequest(BaseModel):
    text: str
    chunk_type: str = "text"

class HealthCheck(BaseModel):
    status: str
    message: str
    rules_loaded: int = 0

# --- HTML Интерфейс (встроенный) ---
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Нормконтролер</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #f4f6f9; color: #333; }
        .container { background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; text-align: center; margin-bottom: 10px; }
        .subtitle { text-align: center; color: #7f8c8d; margin-bottom: 30px; }
        
        .tabs { display: flex; justify-content: center; margin-bottom: 20px; border-bottom: 2px solid #eee; }
        .tab { padding: 10px 20px; cursor: pointer; border: none; background: none; font-size: 16px; color: #7f8c8d; transition: 0.3s; }
        .tab.active { color: #3498db; border-bottom: 3px solid #3498db; font-weight: bold; }
        .tab:hover { color: #2980b9; }
        
        .panel { display: none; }
        .panel.active { display: block; animation: fadeIn 0.5s; }
        
        textarea { width: 100%; height: 200px; padding: 15px; border: 1px solid #ddd; border-radius: 8px; resize: vertical; font-family: inherit; font-size: 14px; box-sizing: border-box; }
        textarea:focus { outline: none; border-color: #3498db; box-shadow: 0 0 5px rgba(52,152,219,0.3); }
        
        .file-upload { border: 2px dashed #ddd; padding: 30px; text-align: center; border-radius: 8px; cursor: pointer; transition: 0.3s; }
        .file-upload:hover { border-color: #3498db; background: #f0f8ff; }
        input[type="file"] { display: none; }
        
        button { background: #3498db; color: white; border: none; padding: 12px 25px; border-radius: 6px; cursor: pointer; font-size: 16px; margin-top: 15px; width: 100%; transition: 0.3s; font-weight: 600; }
        button:hover { background: #2980b9; transform: translateY(-1px); }
        button:disabled { background: #bdc3c7; cursor: not-allowed; transform: none; }
        
        .result { margin-top: 25px; padding: 20px; background: #f8f9fa; border-radius: 8px; border-left: 5px solid #bdc3c7; }
        .result.error { border-left-color: #e74c3c; background: #fdedec; }
        .result.success { border-left-color: #2ecc71; background: #eafaf1; }
        .result.warning { border-left-color: #f39c12; background: #fef9e7; }
        
        .loading { opacity: 0.7; pointer-events: none; position: relative; }
        .spinner { display: inline-block; width: 20px; height: 20px; border: 3px solid rgba(255,255,255,.3); border-radius: 50%; border-top-color: #fff; animation: spin 1s ease-in-out infinite; margin-right: 10px; vertical-align: middle; }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        
        .violation-item { background: white; padding: 10px; margin-top: 10px; border-radius: 4px; border: 1px solid #eee; }
        .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; color: white; }
        .badge-high { background: #e74c3c; }
        .badge-medium { background: #f39c12; }
        .badge-low { background: #3498db; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 AI Нормконтролер</h1>
        <p class="subtitle">Автоматическая проверка документации по ГОСТ 2.105-95</p>

        <div class="tabs">
            <button class="tab active" onclick="switchTab('text')">Текст</button>
            <button class="tab" onclick="switchTab('file')">Файл (PDF)</button>
        </div>

        <!-- Панель текста -->
        <div id="panel-text" class="panel active">
            <textarea id="textInput" placeholder="Вставьте фрагмент текста документа здесь для проверки..."></textarea>
            <button onclick="analyzeText()" id="btnText">Проверить текст</button>
        </div>

        <!-- Панель файла -->
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
        const API_URL = 'http://localhost:8000';

        function switchTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            
            event.target.classList.add('active');
            document.getElementById(`panel-${tab}`).classList.add('active');
            document.getElementById('result').style.display = 'none';
        }

        function updateFileName() {
            const input = document.getElementById('fileInput');
            const span = document.getElementById('fileName');
            if (input.files.length > 0) {
                span.textContent = `Выбран файл: ${input.files[0].name}`;
                span.style.color = '#27ae60';
                span.style.fontWeight = 'bold';
            }
        }

        function showResult(html, type) {
            const div = document.getElementById('result');
            div.className = `result ${type}`;
            div.innerHTML = html;
            div.style.display = 'block';
            div.scrollIntoView({ behavior: 'smooth' });
        }

        async function analyzeText() {
            const text = document.getElementById('textInput').value;
            const btn = document.getElementById('btnText');
            if (!text.trim()) return alert('Введите текст');

            btn.classList.add('loading');
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span> Анализ...';

            try {
                const res = await fetch(`${API_URL}/api/analyze`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text, chunk_type: 'text' })
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || 'Ошибка сервера');
                
                renderSingleResult(data[0]);
            } catch (e) {
                showResult(`<strong>Ошибка:</strong> ${e.message}`, 'error');
            } finally {
                btn.classList.remove('loading');
                btn.disabled = false;
                btn.textContent = 'Проверить текст';
            }
        }

        async function uploadFile() {
            const input = document.getElementById('fileInput');
            const btn = document.getElementById('btnFile');
            if (!input.files.length) return alert('Выберите файл');

            const formData = new FormData();
            formData.append('file', input.files[0]);

            btn.classList.add('loading');
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span> Загрузка и анализ...';

            try {
                const res = await fetch(`${API_URL}/api/upload`, {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || 'Ошибка сервера');

                let html = `<strong>Отчет по файлу:</strong> ${data.filename}<br>`;
                html += `Страниц: ${data.total_pages} | Чанков: ${data.chunks_analyzed}<br>`;
                html += `<strong>Статус:</strong> <span style="color:${data.status === 'PASS' ? 'green' : 'red'}">${data.status}</span><br>`;
                html += `<strong>Найдено нарушений:</strong> ${data.violations_found}<hr>`;
                
                // Показываем первые 5 нарушений подробно
                let count = 0;
                data.details.forEach((chunk, idx) => {
                    if (chunk.has_violation && count < 5) {
                        count++;
                        html += `<div class="violation-item">
                            <strong>Стр. ${chunk.location?.page || '?'}:</strong> Найдено нарушений<br>
                            <small>${chunk.text.substring(0, 100)}...</small><br>
                            <ul>`;
                        chunk.violations.forEach(v => {
                            const severityClass = v.severity === 'high' ? 'badge-high' : (v.severity === 'medium' ? 'badge-medium' : 'badge-low');
                            html += `<li><span class="badge ${severityClass}">${v.rule_id}</span> ${v.description}</li>`;
                        });
                        html += `</ul></div>`;
                    }
                });
                
                if (data.violations_found > 5) html += `<p>...и еще ${data.violations_found - 5} нарушений (см. полный JSON в консоли)</p>`;
                if (data.violations_found === 0) html += `<p>Нарушений не найдено! Документ соответствует нормам.</p>`;

                showResult(html, data.status === 'PASS' ? 'success' : 'error');
                console.log('Полный отчет:', data);

            } catch (e) {
                showResult(`<strong>Ошибка:</strong> ${e.message}`, 'error');
            } finally {
                btn.classList.remove('loading');
                btn.disabled = false;
                btn.textContent = 'Загрузить и анализировать';
            }
        }

        function renderSingleResult(result) {
            const div = document.getElementById('result');
            const type = result.has_violation ? 'error' : 'success';
            
            let html = `<strong>Результат:</strong> ${result.has_violation ? '❌ Нарушения найдены' : '✅ Нарушений нет'}<br>`;
            html += `<strong>Уверенность:</strong> ${(result.confidence * 100).toFixed(1)}%<br>`;

            if (result.violations && result.violations.length > 0) {
                html += '<hr><strong>Детали:</strong>';
                result.violations.forEach(v => {
                    const sev = v.severity || 'medium';
                    const badgeClass = sev === 'high' ? 'badge-high' : (sev === 'medium' ? 'badge-medium' : 'badge-low');
                    html += `<div class="violation-item">
                        <span class="badge ${badgeClass}">${v.rule_id}</span> 
                        <strong>${v.description}</strong><br>
                        <small>ГОСТ: ${v.rule_id}</small>
                    </div>`;
                });
            }
            showResult(html, type);
        }
    </script>
</body>
</html>
"""

# --- События старта ---
@app.on_event("startup")
async def startup_event():
    global normkontroler
    logger.info("Запуск системы AI Нормконтролер...")
    try:
        data_dir = current_dir / "data"
        index_path = data_dir / "gost.index"
        meta_path = data_dir / "gost_rules_meta.pkl"
        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            logger.error("❌ КРИТИЧЕСКАЯ ОШИБКА: OPENROUTER_API_KEY не установлена!")
        else:
            logger.info(f"API ключ найден ({api_key[:5]}...)")

        if index_path.exists() and meta_path.exists():
            logger.info(f"Загрузка индекса из {index_path}")
            normkontroler = AINormkontroler.load_from_index(
                index_path=str(index_path),
                meta_path=str(meta_path),
                api_key=api_key
            )
            if hasattr(normkontroler, 'analyzer') and normkontroler.analyzer:
                logger.info("✅ Система полностью готова (Rules + LLM).")
            else:
                logger.warning("⚠️ LLM анализатор отсутствует.")
            logger.info(f"Загружено правил: {len(normkontroler.rules)}")
        else:
            logger.warning("Индекс не найден. Режим ограниченный.")
            normkontroler = None
    except Exception as e:
        logger.error(f"Ошибка инициализации: {e}", exc_info=True)
        raise

# --- Эндпоинты ---

@app.get("/", response_class=HTMLResponse)
async def get_upload_page():
    """Отдает главную страницу для пользователя."""
    return """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Нормконтролер</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f6f9; margin: 0; padding: 20px; color: #333; }
            .container { max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
            h1 { text-align: center; color: #2c3e50; margin-bottom: 10px; }
            p.subtitle { text-align: center; color: #7f8c8d; margin-bottom: 30px; }

            .upload-area { border: 2px dashed #3498db; border-radius: 8px; padding: 40px; text-align: center; background: #f8fbff; transition: all 0.3s; cursor: pointer; }
            .upload-area:hover { background: #eaf6ff; border-color: #2980b9; }
            .upload-area.dragover { background: #d6eaf8; border-color: #2980b9; }

            input[type="file"] { display: none; }
            .btn { background: #3498db; color: white; border: none; padding: 12px 25px; font-size: 16px; border-radius: 6px; cursor: pointer; transition: background 0.3s; display: inline-block; margin-top: 15px; }
            .btn:hover { background: #2980b9; }
            .btn:disabled { background: #95a5a6; cursor: not-allowed; }

            #status { margin-top: 20px; text-align: center; font-weight: bold; }
            .success { color: #27ae60; }
            .error { color: #c0392b; }

            #results { margin-top: 30px; display: none; }
            .summary { background: #ecf0f1; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
            .violation-card { border-left: 4px solid #e74c3c; background: #fdedec; padding: 15px; margin-bottom: 10px; border-radius: 4px; }
            .violation-card.correct { border-left-color: #27ae60; background: #eafaf1; }
            .badge { display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; color: white; }
            .badge-high { background: #c0392b; }
            .badge-medium { background: #f39c12; }
            .badge-low { background: #3498db; }

            .loader { border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%; width: 30px; height: 30px; animation: spin 1s linear infinite; margin: 20px auto; display: none; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 AI Нормконтролер</h1>
            <p class="subtitle">Загрузите PDF документ для проверки на соответствие ГОСТ 2.105-95</p>

            <div class="upload-area" id="dropZone" onclick="document.getElementById('fileInput').click()">
                <p>📄 Перетащите файл сюда или нажмите для выбора</p>
                <input type="file" id="fileInput" accept=".pdf">
                <button class="btn" id="uploadBtn">Проверить документ</button>
            </div>

            <div class="loader" id="loader"></div>
            <div id="status"></div>

            <div id="results">
                <div class="summary" id="summaryBox"></div>
                <div id="violationsList"></div>
            </div>
        </div>

        <script>
            const dropZone = document.getElementById('dropZone');
            const fileInput = document.getElementById('fileInput');
            const uploadBtn = document.getElementById('uploadBtn');
            const statusDiv = document.getElementById('status');
            const loader = document.getElementById('loader');
            const resultsDiv = document.getElementById('results');
            const summaryBox = document.getElementById('summaryBox');
            const violationsList = document.getElementById('violationsList');

            let selectedFile = null;

            // Обработка выбора файла
            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length) handleFile(e.target.files[0]);
            });

            function handleFile(file) {
                selectedFile = file;
                dropZone.innerHTML = `<p><strong>Выбран файл:</strong> ${file.name}</p><button class="btn" onclick="event.stopPropagation(); document.getElementById('fileInput').click()">Изменить</button>`;
                statusDiv.textContent = "";
                resultsDiv.style.display = "none";
            }

            // Drag & Drop эффекты
            ['dragenter', 'dragover'].forEach(eventName => {
                dropZone.addEventListener(eventName, (e) => {
                    e.preventDefault();
                    dropZone.classList.add('dragover');
                }, false);
            });
            ['dragleave', 'drop'].forEach(eventName => {
                dropZone.addEventListener(eventName, (e) => {
                    e.preventDefault();
                    dropZone.classList.remove('dragover');
                }, false);
            });

            dropZone.addEventListener('drop', (e) => {
                const dt = e.dataTransfer;
                const files = dt.files;
                if (files.length) handleFile(files[0]);
            });

            // Загрузка на сервер
            uploadBtn.addEventListener('click', async () => {
                if (!selectedFile) {
                    statusDiv.textContent = "⚠️ Пожалуйста, выберите файл.";
                    statusDiv.className = "error";
                    return;
                }

                const formData = new FormData();
                formData.append("file", selectedFile);

                uploadBtn.disabled = true;
                loader.style.display = "block";
                statusDiv.textContent = "⏳ Анализ документа... Это может занять несколько минут.";
                statusDiv.className = "";
                resultsDiv.style.display = "none";

                try {
                    const response = await fetch('/api/upload', {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) throw new Error(`Ошибка сервера: ${response.status}`);

                    const data = await response.json();
                    displayResults(data);

                } catch (error) {
                    statusDiv.textContent = `❌ Ошибка: ${error.message}`;
                    statusDiv.className = "error";
                } finally {
                    uploadBtn.disabled = false;
                    loader.style.display = "none";
                }
            });

            function displayResults(data) {
                resultsDiv.style.display = "block";

                const statusColor = data.status === "PASS" ? "#27ae60" : "#c0392b";
                const statusIcon = data.status === "PASS" ? "✅" : "❌";

                summaryBox.innerHTML = `
                    <h3>Результат: <span style="color:${statusColor}">${statusIcon} ${data.status}</span></h3>
                    <p><strong>Файл:</strong> ${data.filename}</p>
                    <p><strong>Страниц:</strong> ${data.total_pages} | <strong>Проверено блоков:</strong> ${data.chunks_analyzed}</p>
                    <p><strong>Найдено нарушений:</strong> <span style="color:${statusColor}; font-weight:bold; font-size:1.2em">${data.violations_found}</span></p>
                `;

                violationsList.innerHTML = "<h3>Детальный отчет:</h3>";

                if (data.violations_found === 0) {
                    violationsList.innerHTML += `<div class="violation-card correct">Нарушений не найдено! Документ соответствует правилам.</div>`;
                    return;
                }

                data.details.forEach((chunk, index) => {
                    if (chunk.has_violation && chunk.violations && chunk.violations.length > 0) {
                        chunk.violations.forEach(v => {
                            const severityClass = v.severity === 'high' ? 'badge-high' : (v.severity === 'medium' ? 'badge-medium' : 'badge-low');
                            const html = `
                                <div class="violation-card">
                                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:5px;">
                                        <strong>Стр. ${chunk.location?.page || '?'} | Правило ${v.rule_id}</strong>
                                        <span class="badge ${severityClass}">${v.severity || 'info'}</span>
                                    </div>
                                    <p style="margin: 5px 0;">${v.description}</p>
                                    <small style="color:#7f8c8d">Текст фрагмента: "${chunk.text.substring(0, 100)}..."</small>
                                </div>
                            `;
                            violationsList.innerHTML += html;
                        });
                    }
                });
            }
        </script>
    </body>
    </html>
    """

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Отдает HTML интерфейс пользователя."""
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>Интерфейс не найден. Используйте /docs для API.</h1>"

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Проверка здоровья сервиса (JSON)."""
    if normkontroler:
        return HealthCheck(status="healthy", message="Готов к работе", rules_loaded=len(normkontroler.rules))
    return HealthCheck(status="degraded", message="Индекс не загружен", rules_loaded=0)

@app.post("/api/analyze", response_model=List[AnalysisResult])
async def analyze_text(request: AnalysisRequest):
    if not normkontroler:
        raise HTTPException(status_code=503, detail="Система не инициализирована")
    try:
        logger.info(f"Запрос анализа: длина={len(request.text)}")
        chunk = {
            "id": "manual_request",
            "text": request.text,
            "chunk_type": request.chunk_type,
            "location": {}
        }
        result_dict = normkontroler.analyze_chunk(chunk)
        return [AnalysisResult(**result_dict)]
    except Exception as e:
        logger.error(f"Ошибка анализа: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_and_analyze(file: UploadFile = File(...)):
    if not normkontroler:
        raise HTTPException(status_code=503, detail="Система не инициализирована")
    if PdfReader is None:
        raise HTTPException(status_code=500, detail="Библиотека pypdf не установлена")

    logger.info(f"Загрузка файла: {file.filename}")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Только PDF файлы")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        reader = PdfReader(tmp_path)
        chunks = []

        # Простой сплиттер, если langchain нет
        if RecursiveCharacterTextSplitter:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        else:
            text_splitter = None

        logger.info(f"Обработка {len(reader.pages)} страниц...")
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text or not text.strip(): continue

            sub_chunks = text_splitter.split_text(text) if text_splitter else [text]
            for j, sub_text in enumerate(sub_chunks):
                if len(sub_text.strip()) < 10: continue
                chunks.append({
                    "id": f"page_{i+1}_chunk_{j}",
                    "text": sub_text,
                    "chunk_type": "text",
                    "location": {"page": i+1}
                })

        if not chunks:
            raise HTTPException(status_code=400, detail="Не удалось извлечь текст (скан?)")

        results = []
        violations_count = 0
        for chunk in chunks:
            try:
                result = normkontroler.analyze_chunk(chunk)
                if result.get("has_violation"): violations_count += 1
                results.append(result)
            except Exception as e:
                logger.error(f"Ошибка чанка {chunk['id']}: {e}")
                results.append({"chunk_id": chunk["id"], "error": str(e), "has_violation": False, "violations": []})

        return {
            "filename": file.filename,
            "total_pages": len(reader.pages),
            "chunks_analyzed": len(chunks),
            "violations_found": violations_count,
            "status": "FAIL" if violations_count > 0 else "PASS",
            "details": results
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try: os.unlink(tmp_path)
            except: pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)