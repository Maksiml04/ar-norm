import React, { useState, useEffect, useRef } from 'react';
import {
  Search, FileText, MessageSquare, History, Settings, Sun, Moon,
  Upload, Send, AlertTriangle, AlertCircle, CheckCircle, X, Trash2, ChevronRight, Sparkles, Zap, Shield
} from 'lucide-react';

// ============================================================
// КОНФИГУРАЦИЯ БЭКЕНДА
// ============================================================
// В режиме разработки (npm run dev) указываем полный URL.
// В продакшене (после сборки) можно оставить пустым '', если настроен прокси.
const API_BASE_URL = 'http://localhost:8000';

// ============================================================
// MOCK GOST ERROR DATA (Для демонстрации UI анализа)
// ============================================================
const MOCK_ERRORS = [
  {
    id: 1,
    severity: 'error',
    standard: 'ГОСТ 2.105-95',
    clause: 'п. 2.1.3',
    title: 'Недопустимая высота строки',
    description: 'Межстрочный интервал менее 2.5 мм.',
    page: 1,
  },
  {
    id: 2,
    severity: 'warning',
    standard: 'ГОСТ 2.105-95',
    clause: 'п. 2.4.1',
    title: 'Недостаточное поле документа',
    description: 'Левое поле составляет 15 мм — минимальное требование 20 мм.',
    page: 1,
  },
];

// ============================================================
// API CALLS TO BACKEND
// ============================================================

// Отправка сообщения в чат (через ваш app.py -> /api/chat)
async function sendMessageToBackend(message, conversationHistory = []) {
  const historyFormatted = conversationHistory.map(m => ({
    role: m.role,
    content: m.content || m.text
  }));

  const response = await fetch(`${API_BASE_URL}/api/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      message: message,
      history: historyFormatted
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      errorData.detail || errorData.message || `Ошибка сервера: ${response.status}`
    );
  }

  const data = await response.json();
  if (data.error) {
    throw new Error(data.response || "Ошибка обработки запроса");
  }

  return data.response;
}

// Загрузка файла (через ваш app.py -> /api/upload)
async function uploadFileToBackend(file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/api/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      errorData.detail || `Ошибка загрузки: ${response.status}`
    );
  }

  return await response.json();
}

// ============================================================
// SIMPLE MARKDOWN RENDERER
// ============================================================
function renderMarkdown(text) {
  if (!text) return null;

  const lines = text.split('\n');
  const elements = [];
  let inCodeBlock = false;
  let codeContent = [];
  let key = 0;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    if (line.startsWith('```')) {
      if (inCodeBlock) {
        elements.push(
          <pre key={key++} className="bg-gray-900 text-green-400 p-3 rounded-lg text-xs overflow-x-auto my-2 font-mono">
            {codeContent.join('\n')}
          </pre>
        );
        codeContent = [];
        inCodeBlock = false;
      } else {
        inCodeBlock = true;
      }
      continue;
    }

    if (inCodeBlock) {
      codeContent.push(line);
      continue;
    }

    if (line.startsWith('### ')) {
      elements.push(<h3 key={key++} className="font-bold text-base mt-3 mb-1">{processInline(line.slice(4))}</h3>);
    } else if (line.startsWith('## ')) {
      elements.push(<h2 key={key++} className="font-bold text-lg mt-3 mb-1">{processInline(line.slice(3))}</h2>);
    } else if (line.startsWith('# ')) {
      elements.push(<h1 key={key++} className="font-bold text-xl mt-3 mb-1">{processInline(line.slice(2))}</h1>);
    }
    else if (line.startsWith('**') && line.endsWith('**')) {
      elements.push(<p key={key++} className="font-bold my-1">{processInline(line.slice(2, -2))}</p>);
    }
    else if (line.startsWith('- ') || line.startsWith('* ')) {
      elements.push(<li key={key++} className="ml-4 list-disc">{processInline(line.slice(2))}</li>);
    }
    else if (/^\d+[\.\)]\s/.test(line)) {
      elements.push(<li key={key++} className="ml-4 list-decimal">{processInline(line.replace(/^\d+[\.\)]\s/, ''))}</li>);
    }
    else if (line.trim() === '') {
      elements.push(<div key={key++} className="h-2" />);
    }
    else {
      elements.push(<p key={key++} className="my-1 leading-relaxed">{processInline(line)}</p>);
    }
  }

  return elements;
}

function processInline(text) {
  const parts = [];
  const regex = /(\*\*.*?\*\*|`.*?`|__.*?__|\*.*?\*|_.*?_)/g;
  let lastIndex = 0;
  let match;
  let key = 0;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }

    const content = match[0];
    if (content.startsWith('**') && content.endsWith('**')) {
      parts.push(<strong key={key++}>{content.slice(2, -2)}</strong>);
    } else if (content.startsWith('`') && content.endsWith('`')) {
      parts.push(
        <code key={key++} className="bg-gray-200 dark:bg-gray-700 px-1 py-0.5 rounded text-xs font-mono">
          {content.slice(1, -1)}
        </code>
      );
    } else if ((content.startsWith('*') && content.endsWith('*')) || (content.startsWith('_') && content.endsWith('_'))) {
      parts.push(<em key={key++}>{content.slice(1, -1)}</em>);
    }

    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return parts.length > 0 ? parts : text;
}

// ============================================================
// ANALYSIS TAB
// ============================================================
function AnalysisTab({ dark }) {
  const [pdfFile, setPdfFile] = useState(null);
  const [pdfUrl, setPdfUrl] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [errors, setErrors] = useState([]);
  const [selectedError, setSelectedError] = useState(null);
  const [uploadStatus, setUploadStatus] = useState(null); // 'success' | 'error' | null
  const [statusMessage, setStatusMessage] = useState('');

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file && file.type === 'application/pdf') {
      if (pdfUrl) URL.revokeObjectURL(pdfUrl);
      setPdfFile(file);
      setPdfUrl(URL.createObjectURL(file));
      setErrors([]);
      setSelectedError(null);
      setUploadStatus(null);
      startAnalysis(file);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type === 'application/pdf') {
      if (pdfUrl) URL.revokeObjectURL(pdfUrl);
      setPdfFile(file);
      setPdfUrl(URL.createObjectURL(file));
      setErrors([]);
      setSelectedError(null);
      setUploadStatus(null);
      startAnalysis(file);
    }
  };

  const handleDragOver = (e) => e.preventDefault();

  const startAnalysis = async (file) => {
    setAnalyzing(true);
    setUploadStatus(null);
    try {
      const result = await uploadFileToBackend(file);

      // Преобразуем результат бэкенда в формат для UI
      // Бэкенд возвращает: { details: [...], violations_found: N, status: 'FAIL'/'PASS' }
      if (result.violations_found > 0) {
        const formattedErrors = result.details
          .filter(chunk => chunk.has_violation)
          .map((chunk, idx) => ({
            id: idx,
            severity: chunk.violations?.[0]?.severity === 'critical' ? 'error' : 'warning',
            standard: chunk.violations?.[0]?.rule_id || 'ГОСТ',
            clause: chunk.violations?.[0]?.rule_id || '',
            title: chunk.violations?.[0]?.violation_type || 'Нарушение найдено',
            description: chunk.violations?.[0]?.explanation || chunk.text,
            page: chunk.location?.page || '?',
          }));

        setErrors(formattedErrors.length > 0 ? formattedErrors : MOCK_ERRORS);
        setStatusMessage(`Найдено нарушений: ${result.violations_found}`);
      } else {
        setErrors([]);
        setStatusMessage('Нарушений не найдено! Документ соответствует ГОСТ.');
      }
      setUploadStatus('success');
    } catch (err) {
      console.error(err);
      setErrors(MOCK_ERRORS); // Fallback для демо
      setStatusMessage(`Ошибка анализа: ${err.message}. Показан демо-режим.`);
      setUploadStatus('error');
    } finally {
      setAnalyzing(false);
    }
  };

  const resetUpload = () => {
    if (pdfUrl) URL.revokeObjectURL(pdfUrl);
    setPdfFile(null);
    setPdfUrl(null);
    setAnalyzing(false);
    setErrors([]);
    setSelectedError(null);
    setUploadStatus(null);
    setStatusMessage('');
  };

  const errorCount = errors.filter((e) => e.severity === 'error').length;
  const warningCount = errors.filter((e) => e.severity === 'warning').length;

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className={`text-3xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent`}>
            Анализ документа
          </h2>
          <p className={`text-xs mt-1 ${dark ? 'text-zinc-500' : 'text-slate-400'}`}>AI-проверка на соответствие ГОСТ</p>
        </div>
        {pdfFile && (
          <button
            onClick={resetUpload}
            className={`flex items-center gap-2 px-4 py-2 text-sm rounded-xl transition-all ${
              dark ? 'text-zinc-400 hover:text-red-400 hover:bg-red-500/10' : 'text-slate-500 hover:text-red-500 hover:bg-red-50'
            }`}
          >
            <X size={16} />
            Сбросить
          </button>
        )}
      </div>

      {!pdfUrl ? (
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          className={`flex-1 flex flex-col items-center justify-center border-2 border-dashed rounded-3xl transition-all cursor-pointer group ${
            dark
              ? 'border-zinc-700 bg-zinc-900/30 hover:border-blue-500/50 hover:bg-blue-500/5'
              : 'border-slate-300 bg-white hover:border-blue-400 hover:bg-blue-50/30'
          }`}
          onClick={() => document.getElementById('pdfFileInput')?.click()}
        >
          <input
            id="pdfFileInput"
            type="file"
            accept=".pdf,application/pdf,.docs"
            onChange={handleFileChange}
            className="hidden"
          />
          {/* Анимированная иконка */}
          <div className="relative mb-6">
            <div className="absolute inset-0 bg-gradient-to-br from-blue-500 via-purple-500 to-pink-500 rounded-3xl blur-xl opacity-20 group-hover:opacity-40 transition-opacity animate-pulse-slow" />
            <div className={`relative w-20 h-20 rounded-3xl flex items-center justify-center ${
              dark ? 'bg-zinc-800 group-hover:bg-zinc-700' : 'bg-slate-100 group-hover:bg-white'
            } transition-colors shadow-xl`}>
              <Upload size={32} className={`${dark ? 'text-zinc-500 group-hover:text-blue-400' : 'text-slate-400 group-hover:text-blue-500'} transition-colors`} />
            </div>
          </div>
          <p className={`text-xl font-bold ${dark ? 'text-zinc-200' : 'text-slate-700'}`}>
            Загрузите PDF-документ
          </p>
          <p className={`text-sm mt-2 ${dark ? 'text-zinc-500' : 'text-slate-400'}`}>
            Перетащите файл или нажмите для выбора
          </p>
          <div className={`mt-6 flex items-center gap-2 px-4 py-2 rounded-full text-xs ${dark ? 'bg-zinc-800 text-zinc-400' : 'bg-slate-100 text-slate-500'}`}>
            <Sparkles size={12} className="text-purple-400" />
            <span>AI анализирует за секунды</span>
          </div>
        </div>
      ) : (
        <div className="flex flex-1 gap-5 min-h-0">
          <div className={`flex-[6] rounded-2xl overflow-hidden flex flex-col ${
            dark ? 'bg-zinc-900 ring-1 ring-zinc-800' : 'bg-white border border-slate-200 shadow-sm'
          }`}>
            <div className={`px-4 py-2.5 flex items-center gap-3 border-b ${
              dark ? 'border-zinc-800 bg-zinc-900' : 'border-slate-100 bg-slate-50'
            }`}>
              <FileText size={16} className={dark ? 'text-zinc-500' : 'text-slate-400'} />
              <span className={`text-sm font-medium truncate ${dark ? 'text-zinc-300' : 'text-slate-600'}`}>
                {pdfFile?.name}
              </span>
            </div>
            <div className="flex-1 relative">
              {analyzing && (
                <div className={`absolute inset-0 z-10 flex flex-col items-center justify-center backdrop-blur-sm ${
                  dark ? 'bg-zinc-900/70' : 'bg-white/70'
                }`}>
                  <div className="w-14 h-14 rounded-full border-4 border-blue-500/30 border-t-blue-500 animate-spin" />
                  <p className={`mt-4 text-sm font-medium ${dark ? 'text-zinc-300' : 'text-slate-600'}`}>
                    AI анализирует документ...
                  </p>
                  <p className={`mt-1 text-xs ${dark ? 'text-zinc-500' : 'text-slate-400'}`}>
                    Это может занять некоторое время
                  </p>
                </div>
              )}
              <iframe src={pdfUrl} className="w-full h-full" title="PDF Preview" />
            </div>
          </div>

          <div className={`flex-[4] rounded-2xl overflow-hidden flex flex-col ${
            dark ? 'bg-zinc-900 ring-1 ring-zinc-800' : 'bg-white border border-slate-200 shadow-sm'
          }`}>
            <div className={`px-4 py-3 border-b ${dark ? 'border-zinc-800' : 'border-slate-100'}`}>
              <h3 className={`text-sm font-semibold uppercase tracking-wider ${dark ? 'text-zinc-400' : 'text-slate-500'}`}>
                Результаты
              </h3>
            </div>
            <div className="flex-1 overflow-auto p-4">
              {analyzing ? (
                <div className="flex flex-col items-center justify-center py-16">
                  <div className="w-10 h-10 rounded-full border-4 border-blue-500/30 border-t-blue-500 animate-spin" />
                </div>
              ) : (
                <>
                  {statusMessage && (
                    <div className={`mb-4 p-3 rounded-xl text-sm ${
                      uploadStatus === 'error'
                        ? (dark ? 'bg-red-900/20 text-red-400' : 'bg-red-50 text-red-600')
                        : (dark ? 'bg-green-900/20 text-green-400' : 'bg-green-50 text-green-600')
                    }`}>
                      {statusMessage}
                    </div>
                  )}

                  {errors.length === 0 && !statusMessage ? (
                     <div className="flex flex-col items-center justify-center py-16">
                       <CheckCircle size={40} className="text-green-500 mb-3" />
                       <p className={`text-sm ${dark ? 'text-zinc-400' : 'text-slate-500'}`}>Загрузите файл для начала</p>
                     </div>
                  ) : errors.length === 0 ? (
                    <div className="flex flex-col items-center justify-center py-16">
                      <CheckCircle size={40} className="text-green-500 mb-3" />
                      <p className={`text-sm font-medium ${dark ? 'text-zinc-300' : 'text-slate-600'}`}>Нет несоответствий</p>
                    </div>
                  ) : (
                    <>
                      <div className={`flex gap-3 mb-4 p-3 rounded-xl text-sm font-semibold ${dark ? 'bg-zinc-800' : 'bg-slate-50'}`}>
                        <span className="flex items-center gap-1.5 text-red-500"><AlertCircle size={16} /> {errorCount}</span>
                        <span className="flex items-center gap-1.5 text-amber-500"><AlertTriangle size={16} /> {warningCount}</span>
                      </div>
                      <div className="space-y-2">
                        {errors.map((err) => (
                          <button
                            key={err.id}
                            onClick={() => setSelectedError(selectedError === err.id ? null : err.id)}
                            className={`w-full text-left p-3 rounded-xl border transition-all ${
                              selectedError === err.id
                                ? dark ? 'border-blue-500/50 bg-blue-500/10' : 'border-blue-300 bg-blue-50'
                                : dark ? 'border-zinc-800 hover:border-zinc-700 bg-zinc-800/50' : 'border-slate-100 hover:border-slate-200 hover:bg-slate-50'
                            }`}
                          >
                            <div className="flex items-start gap-2.5">
                              {err.severity === 'error' ? (
                                <AlertCircle size={18} className="text-red-500 mt-0.5 shrink-0" />
                              ) : (
                                <AlertTriangle size={18} className="text-amber-500 mt-0.5 shrink-0" />
                              )}
                              <div className="min-w-0">
                                <p className={`text-sm font-semibold truncate ${dark ? 'text-zinc-200' : 'text-slate-700'}`}>
                                  {err.title}
                                </p>
                                <div className="flex items-center gap-2 flex-wrap mt-1">
                                  <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded ${dark ? 'bg-purple-500/20 text-purple-400' : 'bg-purple-100 text-purple-600'}`}>
                                    {err.standard}
                                  </span>
                                  <span className={`text-[10px] ${dark ? 'text-zinc-500' : 'text-slate-400'}`}>стр. {err.page}</span>
                                </div>
                              </div>
                              <ChevronRight size={16} className={`shrink-0 mt-1 transition-transform ${selectedError === err.id ? 'rotate-90' : ''} ${dark ? 'text-zinc-600' : 'text-slate-300'}`} />
                            </div>
                            {selectedError === err.id && (
                              <div className={`mt-3 pt-3 border-t text-xs leading-relaxed ${dark ? 'border-zinc-700 text-zinc-400' : 'border-slate-200 text-slate-500'}`}>
                                {err.description}
                              </div>
                            )}
                          </button>
                        ))}
                      </div>
                    </>
                  )}
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================================
// CHAT TAB
// ============================================================
function ChatTab({ dark }) {
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: 'ai',
      text: 'Здравствуйте! Я — AI-ассистент **GOST-Check**. Задайте вопрос о требованиях ГОСТ.',
    },
  ]);
  const [input, setInput] = useState('');
  const [typing, setTyping] = useState(false);
  const [apiError, setApiError] = useState(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, typing]);

  const handleSend = async () => {
    if (!input.trim() || typing) return;

    const userText = input.trim();
    setInput('');
    setApiError(null);

    const userMsg = { id: Date.now(), role: 'user', text: userText };
    setMessages((prev) => [...prev, userMsg]);
    setTyping(true);

    try {
      // Фильтруем системные сообщения и формируем историю
      const conversationHistory = messages
        .filter((m) => m.role !== 'system')
        .map((m) => ({ role: m.role, content: m.text }));

      const aiResponse = await sendMessageToBackend(userText, conversationHistory);

      setMessages((prev) => [
        ...prev,
        { id: Date.now() + 1, role: 'ai', text: aiResponse },
      ]);
    } catch (err) {
      console.error('Backend API Error:', err);
      setApiError(err.message);
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          role: 'ai',
          text: `⚠️ **Ошибка соединения.**\n\n${err.message}\n\nУбедитесь, что сервер запущен (порт 8000) и переменные окружения настроены.`,
        },
      ]);
    } finally {
      setTyping(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between mb-5">
        <div>
          <h2 className={`text-2xl font-bold ${dark ? 'text-zinc-100' : 'text-slate-800'}`}>AI-консультант</h2>
          <p className={`text-xs mt-0.5 ${dark ? 'text-zinc-500' : 'text-slate-400'}`}>Подключено к локальному серверу</p>
        </div>
        {apiError && (
          <span className="text-xs text-red-500 flex items-center gap-1"><AlertCircle size={12} /> Ошибка API</span>
        )}
      </div>

      <div className={`flex-1 overflow-auto rounded-2xl p-5 mb-4 space-y-4 ${dark ? 'bg-zinc-900 ring-1 ring-zinc-800' : 'bg-white border border-slate-200 shadow-sm'}`}>
        {messages.map((msg) => (
          <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-xl rounded-2xl px-4 py-3 text-sm leading-relaxed ${
              msg.role === 'user'
                ? 'bg-blue-600 text-white'
                : dark ? 'bg-zinc-800 text-zinc-200' : 'bg-slate-50 text-slate-700'
            }`}>
              {msg.role === 'ai' && (
                <div className="flex items-center gap-2 mb-2 pb-2 border-b border-white/10">
                  <div className="w-5 h-5 rounded-full bg-blue-500 flex items-center justify-center shrink-0">
                    <FileText size={11} className="text-white" />
                  </div>
                  <span className="text-[10px] font-bold uppercase tracking-wider opacity-60">GOST-Check</span>
                </div>
              )}
              <div>{msg.role === 'ai' ? renderMarkdown(msg.text) : msg.text}</div>
            </div>
          </div>
        ))}
        {typing && (
          <div className="flex justify-start">
            <div className={`rounded-2xl px-4 py-3 ${dark ? 'bg-zinc-800' : 'bg-slate-50'}`}>
              <div className="flex gap-1.5">
                <span className={`w-2 h-2 rounded-full animate-bounce ${dark ? 'bg-zinc-600' : 'bg-slate-400'}`} style={{ animationDelay: '0ms' }} />
                <span className={`w-2 h-2 rounded-full animate-bounce ${dark ? 'bg-zinc-600' : 'bg-slate-400'}`} style={{ animationDelay: '150ms' }} />
                <span className={`w-2 h-2 rounded-full animate-bounce ${dark ? 'bg-zinc-600' : 'bg-slate-400'}`} style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="flex gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
          placeholder="Спросите о требованиях ГОСТ..."
          disabled={typing}
          className={`flex-1 px-4 py-3 rounded-xl text-sm outline-none transition-colors ${
            dark
              ? 'bg-zinc-900 text-zinc-200 border border-zinc-700 focus:border-blue-500 placeholder-zinc-600 disabled:opacity-50'
              : 'bg-white text-slate-700 border border-slate-200 focus:border-blue-400 placeholder-slate-400 disabled:opacity-50'
          }`}
        />
        <button
          onClick={handleSend}
          disabled={!input.trim() || typing}
          className="px-4 py-3 rounded-xl bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-30 disabled:cursor-not-allowed transition-all active:scale-95"
        >
          <Send size={18} />
        </button>
      </div>
    </div>
  );
}

// ============================================================
// HISTORY TAB
// ============================================================
function HistoryTab({ dark }) {
  const [docs] = useState([
    { id: 1, name: 'Спецификация_КМ-01.pdf', date: '14.04.2026', time: '16:32', status: 'passed', errors: 0, size: '245 KB' },
    { id: 2, name: 'Чертеж_Фундамент.pdf', date: '13.04.2026', time: '11:15', status: 'failed', errors: 5, size: '1.2 MB' },
    { id: 3, name: 'ПЗ_Раздел_5.pdf', date: '12.04.2026', time: '09:47', status: 'warning', errors: 2, size: '890 KB' },
  ]);

  const statusConfig = {
    passed: { label: 'Соответствует', color: 'text-emerald-500', bg: dark ? 'bg-emerald-500/10' : 'bg-emerald-50', icon: CheckCircle },
    warning: { label: 'Предупреждения', color: 'text-amber-500', bg: dark ? 'bg-amber-500/10' : 'bg-amber-50', icon: AlertTriangle },
    failed: { label: 'Несоответствия', color: 'text-red-500', bg: dark ? 'bg-red-500/10' : 'bg-red-50', icon: AlertCircle },
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between mb-5">
        <h2 className={`text-2xl font-bold ${dark ? 'text-zinc-100' : 'text-slate-800'}`}>История проверок</h2>
        <button className={`flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-lg transition-colors ${dark ? 'text-zinc-400 hover:text-red-400 hover:bg-zinc-800' : 'text-slate-500 hover:text-red-500 hover:bg-red-50'}`}>
          <Trash2 size={14} /> Очистить
        </button>
      </div>

      <div className={`rounded-2xl overflow-hidden ${dark ? 'bg-zinc-900 ring-1 ring-zinc-800' : 'bg-white border border-slate-200 shadow-sm'}`}>
        <div className={`grid grid-cols-12 gap-4 px-5 py-3 text-[10px] font-bold uppercase tracking-widest ${dark ? 'bg-zinc-950 text-zinc-500' : 'bg-slate-50 text-slate-400'}`}>
          <div className="col-span-5">Документ</div>
          <div className="col-span-2">Дата</div>
          <div className="col-span-2">Статус</div>
          <div className="col-span-3 text-right">Ошибки</div>
        </div>
        {docs.map((doc) => {
          const cfg = statusConfig[doc.status];
          const Icon = cfg.icon;
          return (
            <div key={doc.id} className={`grid grid-cols-12 gap-4 px-5 py-3.5 items-center border-t cursor-pointer transition-colors ${dark ? 'border-zinc-800 hover:bg-zinc-800/60' : 'border-slate-50 hover:bg-slate-50'}`}>
              <div className={`col-span-5 flex items-center gap-3 ${dark ? 'text-zinc-200' : 'text-slate-700'}`}>
                <div className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 ${dark ? 'bg-zinc-800' : 'bg-slate-100'}`}>
                  <FileText size={14} className={dark ? 'text-zinc-500' : 'text-slate-400'} />
                </div>
                <span className="text-sm truncate font-medium">{doc.name}</span>
              </div>
              <div className={`col-span-2 text-xs ${dark ? 'text-zinc-500' : 'text-slate-400'}`}>
                <div>{doc.date}</div>
                <div className="opacity-60">{doc.time}</div>
              </div>
              <div className="col-span-2">
                <span className={`inline-flex items-center gap-1.5 text-[11px] font-semibold px-2.5 py-1 rounded-full ${cfg.bg} ${cfg.color}`}>
                  <Icon size={12} /> {cfg.label}
                </span>
              </div>
              <div className={`col-span-3 text-sm font-bold text-right ${doc.errors > 0 ? 'text-red-500' : dark ? 'text-zinc-600' : 'text-slate-300'}`}>
                {doc.errors > 0 ? doc.errors : '—'}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ============================================================
// MAIN APP
// ============================================================
export default function App() {
  const [activeTab, setActiveTab] = useState('analysis');
  const [dark, setDark] = useState(true);

  const tabs = [
    { id: 'analysis', label: 'Анализ', icon: Search },
    { id: 'chat', label: 'AI-чат', icon: MessageSquare },
    { id: 'history', label: 'История', icon: History },
  ];

  return (
    <div className={`h-screen flex ${dark ? 'dark' : ''}`}>
      {/* Sidebar с градиентным фоном */}
      <aside className={`w-[84px] flex flex-col items-center py-6 shrink-0 relative overflow-hidden ${dark ? 'bg-gradient-to-b from-zinc-950 via-zinc-900 to-zinc-950 border-r border-zinc-800/50' : 'bg-gradient-to-b from-slate-50 via-white to-slate-50 border-r border-slate-200/50'}`}>
        {/* Анимированный фоновый элемент */}
        <div className={`absolute inset-0 opacity-30 ${dark ? 'bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-blue-900/20 via-transparent to-transparent' : 'bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-blue-100/40 via-transparent to-transparent'}`} />
        
        {/* Логотип с эффектом свечения */}
        <div className="mb-8 relative group">
          <div className="absolute inset-0 bg-gradient-to-br from-blue-500 via-purple-500 to-pink-500 rounded-2xl blur-lg opacity-50 group-hover:opacity-70 transition-opacity animate-glow" />
          <div className="relative w-12 h-12 rounded-2xl bg-gradient-to-br from-blue-500 via-purple-500 to-pink-500 flex items-center justify-center text-white font-bold text-xs shadow-2xl">
            ГОСТ
          </div>
        </div>
        
        <div className={`w-10 h-px mb-6 ${dark ? 'bg-gradient-to-r from-transparent via-zinc-700 to-transparent' : 'bg-gradient-to-r from-transparent via-slate-300 to-transparent'}`} />
        
        {/* Навигация */}
        <nav className="flex flex-col gap-2 w-full px-3 relative z-10">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                title={tab.label}
                className={`group relative flex flex-col items-center justify-center py-3 px-2 rounded-2xl transition-all duration-300 ${
                  isActive
                    ? 'bg-gradient-to-br from-blue-500 to-blue-600 text-white shadow-xl shadow-blue-500/30 scale-105'
                    : dark ? 'text-zinc-500 hover:text-zinc-200 hover:bg-zinc-800/50' : 'text-slate-400 hover:text-slate-600 hover:bg-slate-100'
                }`}
              >
                {isActive && (
                  <div className="absolute inset-0 bg-gradient-to-br from-white/10 to-transparent rounded-2xl" />
                )}
                <Icon size={22} strokeWidth={isActive ? 2.5 : 1.5} className="relative z-10" />
                <span className="text-[9px] mt-1.5 font-semibold tracking-wide relative z-10">{tab.label}</span>
              </button>
            );
          })}
        </nav>
        
        <div className="flex-1" />
        
        {/* Переключатель темы */}
        <button
          onClick={() => setDark(!dark)}
          className={`group relative flex flex-col items-center justify-center py-3 px-2 rounded-2xl transition-all duration-300 w-full mb-2 ${dark ? 'text-zinc-500 hover:text-amber-400 hover:bg-amber-500/10' : 'text-slate-400 hover:text-amber-500 hover:bg-amber-50'}`}
        >
          <div className={`absolute inset-0 rounded-2xl transition-opacity ${dark ? 'group-hover:bg-amber-500/5' : 'group-hover:bg-amber-100/50'}`} />
          {dark ? <Sun size={22} strokeWidth={1.5} /> : <Moon size={22} strokeWidth={1.5} />}
          <span className="text-[9px] mt-1.5 font-semibold">{dark ? 'Свет' : 'Тёмн.'}</span>
        </button>
      </aside>

      {/* Main content area */}
      <main className={`flex-1 p-8 overflow-auto relative ${dark ? 'bg-gradient-to-br from-zinc-950 via-zinc-900 to-zinc-950' : 'bg-gradient-to-br from-slate-50 via-slate-100 to-slate-50'}`}>
        {/* Фоновые декоративные элементы */}
        <div className="fixed top-0 right-0 w-96 h-96 bg-blue-500/5 rounded-full blur-3xl pointer-events-none" />
        <div className="fixed bottom-0 left-0 w-96 h-96 bg-purple-500/5 rounded-full blur-3xl pointer-events-none" />
        
        {activeTab === 'analysis' && <AnalysisTab dark={dark} />}
        {activeTab === 'chat' && <ChatTab dark={dark} />}
        {activeTab === 'history' && <HistoryTab dark={dark} />}
      </main>
    </div>
  );
}