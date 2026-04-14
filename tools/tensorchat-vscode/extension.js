// TensorChat Studio — VS Code Extension
// Ollama-compatible AI chat + code engine with accurate tok/s and batched streaming
const vscode = require('vscode');

let chatPanel = null;
let codePanel = null;
let conversationHistory = [];
let cachedModels = [];
let activeAbortController = null;

// ── Config ──────────────────────────────────────────────────────────────
function cfg(key, fallback) {
  return vscode.workspace.getConfiguration('tensorchat').get(key, fallback);
}

function getBaseUrl() {
  return String(cfg('baseUrl', 'http://127.0.0.1:11434')).replace(/\/+$/, '');
}

function getProvider() {
  return String(cfg('provider', 'ollama'));
}

// ── API abstraction ─────────────────────────────────────────────────────
async function listModelsFromProvider() {
  const provider = getProvider();
  const baseUrl = getBaseUrl();
  try {
    if (provider === 'hypertensor') {
      const resp = await fetch(`${baseUrl}/v1/models`, { signal: AbortSignal.timeout(5000) });
      if (!resp.ok) return [];
      const json = await resp.json();
      return [{
        name: json.model || 'unknown',
        size: 0,
        modified: '',
        paramSize: '',
        quantization: '',
        family: json.arch || '',
        layers: json.layers,
        dim: json.dim,
        vocab: json.vocab,
        backend: json.backend || '',
        contextTokens: json.context_tokens,
        contextMax: json.context_max,
        vramMb: json.vram_mb
      }];
    } else {
      // Ollama / OpenAI-compatible
      const resp = await fetch(`${baseUrl}/api/tags`, { signal: AbortSignal.timeout(5000) });
      if (!resp.ok) return [];
      const json = await resp.json();
      return (json.models || []).map(m => ({
        name: m.name,
        size: m.size,
        modified: m.modified_at,
        paramSize: m.details?.parameter_size || '',
        quantization: m.details?.quantization_level || '',
        family: m.details?.family || ''
      }));
    }
  } catch { return []; }
}

async function showModelDetail(name) {
  const provider = getProvider();
  const baseUrl = getBaseUrl();
  try {
    if (provider === 'hypertensor') {
      const resp = await fetch(`${baseUrl}/v1/models`, { signal: AbortSignal.timeout(5000) });
      if (!resp.ok) return null;
      return await resp.json();
    } else {
      const resp = await fetch(`${baseUrl}/api/show`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
        signal: AbortSignal.timeout(5000)
      });
      if (!resp.ok) return null;
      return await resp.json();
    }
  } catch { return null; }
}

async function resolveModel() {
  let model = String(cfg('model', '')).trim();
  if (model) return model;
  if (cachedModels.length === 0) {
    cachedModels = await listModelsFromProvider();
  }
  if (cachedModels.length > 0) return cachedModels[0].name;
  return 'llama3';
}

// ── Streaming chat with batched token delivery ──────────────────────────
async function streamChat(messages, onTokenBatch, onStats, externalSignal) {
  const provider = getProvider();
  const baseUrl = getBaseUrl();
  const model = await resolveModel();
  const temperature = Number(cfg('temperature', 0.7));
  const maxTokens = Number(cfg('maxTokens', -1));
  const timeoutMs = Number(cfg('timeoutMs', 120000));

  // ── HyperTensor: non-streaming single request ──
  if (provider === 'hypertensor') {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    if (externalSignal) {
      externalSignal.addEventListener('abort', () => controller.abort());
    }

    const t0 = performance.now();

    // Extract last user message for HyperTensor's simpler chat API
    const lastUser = messages.filter(m => m.role === 'user').pop();
    const htBody = JSON.stringify({
      messages,
      ...(maxTokens > 0 ? { max_tokens: maxTokens } : {}),
      temperature
    });

    try {
      const resp = await fetch(`${baseUrl}/v1/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: htBody,
        signal: controller.signal
      });

      clearTimeout(timer);

      if (!resp.ok) {
        const errText = await resp.text().catch(() => '');
        throw new Error(`HTTP ${resp.status}: ${errText.slice(0, 200)}`);
      }

      const json = await resp.json();
      const totalMs = performance.now() - t0;
      const text = json.message?.content || json.response || '';
      const tokenCount = json.tokens || text.split(/\s+/).length;
      const tokPerSec = json.tokens_per_sec || (tokenCount / Math.max(totalMs / 1000, 0.001));

      // Deliver full response as single batch
      onTokenBatch(text, tokenCount, tokPerSec);

      return {
        fullText: text,
        tokenCount,
        totalMs: json.elapsed_ms || totalMs,
        tokPerSec,
        firstTokenMs: json.prefill_ms || totalMs,
        ollamaStats: null,
        htStats: {
          contextTokens: json.context_tokens,
          thinkingTokens: json.thinking_tokens,
          prefillMs: json.prefill_ms,
          vramMb: json.vram_mb
        }
      };
    } finally {
      clearTimeout(timer);
    }
  }

  // ── Ollama / OpenAI-compatible: streaming ──
  let url, body, parseChunk;

  if (provider === 'ollama') {
    url = `${baseUrl}/api/chat`;
    const opts = { temperature };
    if (maxTokens > 0) opts.num_predict = maxTokens;
    body = JSON.stringify({
      model,
      messages,
      stream: true,
      options: opts
    });
    parseChunk = (json) => ({
      text: json.message?.content || '',
      done: !!json.done,
      stats: json.done ? {
        totalDuration: json.total_duration,
        loadDuration: json.load_duration,
        promptEvalCount: json.prompt_eval_count,
        promptEvalDuration: json.prompt_eval_duration,
        evalCount: json.eval_count,
        evalDuration: json.eval_duration
      } : null
    });
  } else {
    url = `${baseUrl}/v1/chat/completions`;
    body = JSON.stringify({
      model,
      messages,
      stream: true,
      temperature,
      ...(maxTokens > 0 ? { max_tokens: maxTokens } : {})
    });
    parseChunk = (json) => {
      const choice = json.choices?.[0];
      return {
        text: choice?.delta?.content || '',
        done: choice?.finish_reason === 'stop',
        stats: null
      };
    };
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  // Allow external cancellation
  if (externalSignal) {
    externalSignal.addEventListener('abort', () => controller.abort());
  }
  let fullText = '';
  let tokenCount = 0;
  const t0 = performance.now();
  let firstTokenMs = 0;
  let firstTokenTime = 0;
  let ollamaStats = null;

  // Token batching: accumulate tokens and flush at ~30fps
  let tokenBatch = '';
  let batchTimer = null;

  function flushBatch() {
    if (tokenBatch) {
      // tok/s from first token (excludes prompt eval)
      const genElapsed = firstTokenTime ? (performance.now() - firstTokenTime) / 1000 : 0.001;
      const tokPerSec = tokenCount / Math.max(genElapsed, 0.001);
      onTokenBatch(tokenBatch, tokenCount, tokPerSec);
      tokenBatch = '';
    }
    batchTimer = null;
  }

  try {
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
      signal: controller.signal
    });

    if (!resp.ok) {
      const errText = await resp.text().catch(() => '');
      throw new Error(`HTTP ${resp.status}: ${errText.slice(0, 200)}`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;

        const payload = trimmed.startsWith('data:') ? trimmed.slice(5).trim() : trimmed;
        if (payload === '[DONE]') continue;

        try {
          const json = JSON.parse(payload);
          const parsed = parseChunk(json);

          if (parsed.text) {
            fullText += parsed.text;
            tokenCount++;

            if (tokenCount === 1) {
              firstTokenMs = performance.now() - t0;
              firstTokenTime = performance.now();
            }

            // Accumulate into batch
            tokenBatch += parsed.text;
            if (!batchTimer) {
              batchTimer = setTimeout(flushBatch, 33); // ~30fps
            }
          }

          if (parsed.done && parsed.stats) {
            ollamaStats = parsed.stats;
            onStats(parsed.stats);
          }
        } catch { /* skip malformed lines */ }
      }
    }

    // Process any remaining data left in buffer (Ollama final stats chunk)
    if (buffer.trim()) {
      const payload = buffer.trim().startsWith('data:') ? buffer.trim().slice(5).trim() : buffer.trim();
      if (payload && payload !== '[DONE]') {
        try {
          const json = JSON.parse(payload);
          const parsed = parseChunk(json);
          if (parsed.text) {
            fullText += parsed.text;
            tokenCount++;
            tokenBatch += parsed.text;
          }
          if (parsed.done && parsed.stats) {
            ollamaStats = parsed.stats;
            onStats(parsed.stats);
          }
        } catch { /* skip malformed */ }
      }
    }

    // Flush any remaining tokens
    if (batchTimer) {
      clearTimeout(batchTimer);
      batchTimer = null;
    }
    flushBatch();

  } finally {
    clearTimeout(timer);
  }

  const totalMs = performance.now() - t0;

  // Use Ollama native eval stats for accurate tok/s (excludes prefill)
  let finalTokPerSec = tokenCount / Math.max(totalMs / 1000, 0.001);
  let finalTokenCount = tokenCount;

  if (ollamaStats?.evalCount && ollamaStats?.evalDuration) {
    finalTokPerSec = ollamaStats.evalCount / (ollamaStats.evalDuration / 1e9);
    finalTokenCount = ollamaStats.evalCount;
  }

  return {
    fullText,
    tokenCount: finalTokenCount,
    totalMs,
    tokPerSec: finalTokPerSec,
    firstTokenMs,
    ollamaStats
  };
}

// ── Non-streaming fallback ──────────────────────────────────────────────
async function chatOnce(messages) {
  const provider = getProvider();
  const baseUrl = getBaseUrl();
  const model = await resolveModel();
  const temperature = Number(cfg('temperature', 0.7));
  const maxTokens = Number(cfg('maxTokens', -1));
  const timeoutMs = Number(cfg('timeoutMs', 120000));

  let url, body;
  if (provider === 'hypertensor') {
    url = `${baseUrl}/v1/chat`;
    body = JSON.stringify({ messages, temperature,
      ...(maxTokens > 0 ? { max_tokens: maxTokens } : {}) });
  } else if (provider === 'ollama') {
    url = `${baseUrl}/api/chat`;
    const opts = { temperature };
    if (maxTokens > 0) opts.num_predict = maxTokens;
    body = JSON.stringify({ model, messages, stream: false, options: opts });
  } else {
    url = `${baseUrl}/v1/chat/completions`;
    body = JSON.stringify({ model, messages, stream: false,
      temperature, ...(maxTokens > 0 ? { max_tokens: maxTokens } : {}) });
  }

  const t0 = performance.now();
  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body,
    signal: AbortSignal.timeout(timeoutMs)
  });

  if (!resp.ok) {
    const errText = await resp.text().catch(() => '');
    throw new Error(`HTTP ${resp.status}: ${errText.slice(0, 200)}`);
  }

  const json = await resp.json();
  const totalMs = performance.now() - t0;

  let text = '';
  let stats = null;

  if (provider === 'hypertensor') {
    text = json.message?.content || json.response || '';
    stats = { tokPerSec: json.tokens_per_sec, elapsedMs: json.elapsed_ms };
  } else if (provider === 'ollama') {
    text = json.message?.content || '';
    stats = {
      totalDuration: json.total_duration,
      evalCount: json.eval_count,
      evalDuration: json.eval_duration,
      promptEvalCount: json.prompt_eval_count,
      promptEvalDuration: json.prompt_eval_duration
    };
  } else {
    text = json.choices?.[0]?.message?.content || '';
  }

  return { text, stats, totalMs };
}

// ── Workspace helpers ───────────────────────────────────────────────────
function getWorkspaceFiles() {
  const folders = vscode.workspace.workspaceFolders;
  if (!folders) return [];
  return folders.map(f => f.uri.fsPath);
}

function getActiveEditorInfo() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) return null;
  const doc = editor.document;
  const selection = editor.selection;
  return {
    fileName: doc.fileName,
    language: doc.languageId,
    lineCount: doc.lineCount,
    selectedText: doc.getText(selection),
    fullText: doc.getText(),
    cursorLine: selection.active.line
  };
}

// ── Panel creation ──────────────────────────────────────────────────────
function createPanel(context, mode) {
  const title = mode === 'chat' ? 'TensorChat' : 'TensorChat Code';
  const column = mode === 'chat' ? vscode.ViewColumn.One : vscode.ViewColumn.Two;

  const panel = vscode.window.createWebviewPanel(
    `tensorchat-${mode}`,
    title,
    column,
    {
      enableScripts: true,
      retainContextWhenHidden: true,
      localResourceRoots: [vscode.Uri.joinPath(context.extensionUri, 'media')]
    }
  );

  const styleUri = panel.webview.asWebviewUri(
    vscode.Uri.joinPath(context.extensionUri, 'media', 'webview.css'));
  const scriptUri = panel.webview.asWebviewUri(
    vscode.Uri.joinPath(context.extensionUri, 'media', 'webview.js'));
  const nonce = getNonce();

  panel.webview.html = buildHtml(panel.webview, styleUri, scriptUri, nonce, mode);

  resolveModel().then(model => {
    panel.webview.postMessage({
      type: 'init', mode, model,
      provider: getProvider(),
      baseUrl: getBaseUrl(),
      temperature: Number(cfg('temperature', 0.7)),
      maxTokens: Number(cfg('maxTokens', -1)),
      timeoutMs: Number(cfg('timeoutMs', 120000)),
      systemPrompt: String(cfg('systemPrompt', ''))
    });
  });

  panel.webview.onDidReceiveMessage(async (msg) => {
    try {
      await handleMessage(panel, msg, context);
    } catch (err) {
      panel.webview.postMessage({
        type: 'error', text: err.message || String(err)
      });
    }
  }, undefined, context.subscriptions);

  return panel;
}

async function handleMessage(panel, msg, context) {
  switch (msg.type) {
    case 'chat': {
      const systemPrompt = String(cfg('systemPrompt', ''));
      const messages = [];
      if (systemPrompt) messages.push({ role: 'system', content: systemPrompt });

      if (msg.context) {
        messages.push({ role: 'system', content: `[Workspace Context]\n${msg.context}` });
      }

      for (const h of msg.history || []) {
        messages.push({ role: h.role, content: h.content });
      }

      messages.push({ role: 'user', content: msg.text });

      const model = await resolveModel();
      panel.webview.postMessage({ type: 'streamStart', model });

      activeAbortController = new AbortController();
      try {
        const result = await streamChat(
          messages,
          (tokens, count, tokPerSec) => {
            panel.webview.postMessage({
              type: 'streamTokenBatch', tokens, tokenCount: count, tokPerSec
            });
          },
          (stats) => {
            // Stats captured in result, no need to send separately
          },
          activeAbortController.signal
        );

        panel.webview.postMessage({
          type: 'streamEnd',
          tokenCount: result.tokenCount,
          totalMs: result.totalMs,
          tokPerSec: result.tokPerSec,
          ttft: result.firstTokenMs
        });
      } catch (err) {
        const cancelled = err.name === 'AbortError';
        panel.webview.postMessage({
          type: cancelled ? 'streamCancelled' : 'streamError',
          text: cancelled ? 'Request cancelled' : (err.message || String(err))
        });
      } finally {
        activeAbortController = null;
      }
      break;
    }

    case 'cancelRequest': {
      if (activeAbortController) {
        activeAbortController.abort();
        activeAbortController = null;
      }
      break;
    }

    case 'codeAction': {
      const info = getActiveEditorInfo();
      const lang = msg.language || info?.language || 'code';
      const actionPrompts = {
        explain: `Explain this ${lang} code in detail:\n\`\`\`${lang}\n${msg.code}\n\`\`\``,
        refactor: `Refactor this ${lang} code for readability and performance. Return ONLY the refactored code:\n\`\`\`${lang}\n${msg.code}\n\`\`\``,
        tests: `Generate comprehensive unit tests for this ${lang} code:\n\`\`\`${lang}\n${msg.code}\n\`\`\``,
        review: `Review this ${lang} code for bugs, security issues, and improvements:\n\`\`\`${lang}\n${msg.code}\n\`\`\``,
        optimize: `Optimize this ${lang} code for maximum performance. Return ONLY the optimized code:\n\`\`\`${lang}\n${msg.code}\n\`\`\``,
        document: `Add comprehensive documentation to this ${lang} code. Return the documented code:\n\`\`\`${lang}\n${msg.code}\n\`\`\``
      };

      const prompt = actionPrompts[msg.action] || msg.code;
      const messages = [
        { role: 'system', content: cfg('systemPrompt', '') },
        { role: 'user', content: prompt }
      ];

      const model = await resolveModel();
      panel.webview.postMessage({ type: 'streamStart', model, action: msg.action });

      activeAbortController = new AbortController();
      try {
        const result = await streamChat(
          messages,
          (tokens, count, tokPerSec) => {
            panel.webview.postMessage({
              type: 'streamTokenBatch', tokens, tokenCount: count, tokPerSec
            });
          },
          (stats) => {},
          activeAbortController.signal
        );
        panel.webview.postMessage({
          type: 'streamEnd',
          tokenCount: result.tokenCount,
          totalMs: result.totalMs,
          tokPerSec: result.tokPerSec,
          ttft: result.firstTokenMs
        });
      } catch (err) {
        const cancelled = err.name === 'AbortError';
        panel.webview.postMessage({
          type: cancelled ? 'streamCancelled' : 'streamError',
          text: cancelled ? 'Request cancelled' : (err.message || String(err))
        });
      } finally {
        activeAbortController = null;
      }
      break;
    }

    case 'getEditor': {
      const info = getActiveEditorInfo();
      panel.webview.postMessage({ type: 'editorContent', info });
      break;
    }

    case 'applyCode': {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showWarningMessage('No active editor to apply code to.');
        return;
      }
      await editor.edit(eb => {
        if (msg.replaceSelection && !editor.selection.isEmpty) {
          eb.replace(editor.selection, msg.code);
        } else {
          const full = new vscode.Range(
            editor.document.positionAt(0),
            editor.document.positionAt(editor.document.getText().length)
          );
          eb.replace(full, msg.code);
        }
      });
      vscode.window.showInformationMessage('Code applied to editor.');
      break;
    }

    case 'insertCode': {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        const doc = await vscode.workspace.openTextDocument({ content: msg.code, language: msg.language || 'plaintext' });
        await vscode.window.showTextDocument(doc);
        return;
      }
      await editor.edit(eb => {
        eb.insert(editor.selection.active, msg.code);
      });
      break;
    }

    case 'openFile': {
      const doc = await vscode.workspace.openTextDocument(msg.path);
      await vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside);
      break;
    }

    case 'newFile': {
      const doc = await vscode.workspace.openTextDocument({
        content: msg.code || '', language: msg.language || 'plaintext'
      });
      await vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside);
      break;
    }

    case 'listModels': {
      cachedModels = await listModelsFromProvider();
      panel.webview.postMessage({ type: 'modelList', models: cachedModels });
      break;
    }

    case 'setModel': {
      await vscode.workspace.getConfiguration('tensorchat').update('model', msg.model, true);
      panel.webview.postMessage({ type: 'modelChanged', model: msg.model });
      break;
    }

    case 'getWorkspace': {
      const folders = getWorkspaceFiles();
      panel.webview.postMessage({ type: 'workspaceInfo', folders });
      break;
    }

    case 'copyToClipboard': {
      await vscode.env.clipboard.writeText(msg.text);
      vscode.window.showInformationMessage('Copied to clipboard');
      break;
    }

    case 'updateSetting': {
      const config = vscode.workspace.getConfiguration('tensorchat');
      try {
        await config.update(msg.key, msg.value, true);
      } catch (err) {
        panel.webview.postMessage({ type: 'error', text: `Failed to update ${msg.key}: ${err.message}` });
      }
      break;
    }

    case 'testConnection': {
      const baseUrl = getBaseUrl();
      const provider = getProvider();
      const testUrl = provider === 'hypertensor' ? `${baseUrl}/v1/models` : `${baseUrl}/api/tags`;
      const t0 = performance.now();
      try {
        const resp = await fetch(testUrl, { signal: AbortSignal.timeout(5000) });
        const latency = Math.round(performance.now() - t0);
        if (resp.ok) {
          const json = await resp.json();
          const count = provider === 'hypertensor' ? 1 : (json.models || []).length;
          panel.webview.postMessage({
            type: 'connectionTest',
            success: true,
            latency,
            modelCount: count
          });
        } else {
          panel.webview.postMessage({
            type: 'connectionTest',
            success: false,
            error: `HTTP ${resp.status}`
          });
        }
      } catch (err) {
        panel.webview.postMessage({
          type: 'connectionTest',
          success: false,
          error: err.message || 'Connection failed'
        });
      }
      break;
    }

    case 'showModelDetail': {
      try {
        const detail = await showModelDetail(msg.model);
        if (detail) {
          const provider = getProvider();
          if (provider === 'hypertensor') {
            panel.webview.postMessage({
              type: 'modelDetail',
              detail: {
                name: detail.model || msg.model,
                family: detail.arch || '',
                paramSize: '',
                quantization: '',
                format: 'GGUF',
                contextLength: detail.context_max || '',
                size: 0,
                template: '',
                layers: detail.layers,
                dim: detail.dim,
                vocab: detail.vocab,
                backend: detail.backend || '',
                contextTokens: detail.context_tokens,
                vramMb: detail.vram_mb
              }
            });
          } else {
            panel.webview.postMessage({
              type: 'modelDetail',
              detail: {
                name: msg.model,
                family: detail.details?.family || '',
                paramSize: detail.details?.parameter_size || '',
                quantization: detail.details?.quantization_level || '',
                format: detail.details?.format || '',
                contextLength: detail.model_info?.['general.context_length'] || detail.details?.context_length || '',
                size: detail.size || 0,
                template: detail.template || ''
              }
            });
          }
        }
      } catch {}
      break;
    }

    case 'getSettings': {
      panel.webview.postMessage({
        type: 'settingsData',
        provider: getProvider(),
        baseUrl: getBaseUrl(),
        temperature: Number(cfg('temperature', 0.7)),
        maxTokens: Number(cfg('maxTokens', -1)),
        timeoutMs: Number(cfg('timeoutMs', 120000)),
        systemPrompt: String(cfg('systemPrompt', ''))
      });
      break;
    }
  }
}

// ── HTML builder ────────────────────────────────────────────────────────
function buildHtml(webview, styleUri, scriptUri, nonce, mode) {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Security-Policy"
    content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline' https://cdn.jsdelivr.net; script-src 'nonce-${nonce}' https://cdn.jsdelivr.net; font-src ${webview.cspSource} https://cdn.jsdelivr.net; img-src ${webview.cspSource} data:;">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
  <link rel="stylesheet" href="${styleUri}">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
  <title>TensorChat</title>
</head>
<body data-mode="${mode}">
  <div id="app"></div>
  <script nonce="${nonce}" src="${scriptUri}"></script>
</body>
</html>`;
}

function getNonce() {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let nonce = '';
  for (let i = 0; i < 32; i++) {
    nonce += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return nonce;
}

// ── Activation ──────────────────────────────────────────────────────────
function activate(context) {
  context.subscriptions.push(
    vscode.commands.registerCommand('tensorchat.openChat', () => {
      if (chatPanel) { chatPanel.reveal(); return; }
      chatPanel = createPanel(context, 'chat');
      chatPanel.onDidDispose(() => { chatPanel = null; }, null, context.subscriptions);
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('tensorchat.openCode', () => {
      if (codePanel) { codePanel.reveal(); return; }
      codePanel = createPanel(context, 'code');
      codePanel.onDidDispose(() => { codePanel = null; }, null, context.subscriptions);
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('tensorchat.explainSelection', () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) return;
      const text = editor.document.getText(editor.selection);
      if (!text) return;
      if (!chatPanel) {
        chatPanel = createPanel(context, 'chat');
        chatPanel.onDidDispose(() => { chatPanel = null; }, null, context.subscriptions);
      }
      chatPanel.reveal();
      setTimeout(() => {
        chatPanel.webview.postMessage({
          type: 'injectPrompt',
          text: `Explain this code:\n\`\`\`\n${text}\n\`\`\``
        });
      }, 500);
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('tensorchat.refactorSelection', () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) return;
      const text = editor.document.getText(editor.selection);
      if (!text) return;
      if (!codePanel) {
        codePanel = createPanel(context, 'code');
        codePanel.onDidDispose(() => { codePanel = null; }, null, context.subscriptions);
      }
      codePanel.reveal();
      setTimeout(() => {
        codePanel.webview.postMessage({
          type: 'injectCode', code: text,
          action: 'refactor',
          language: editor.document.languageId
        });
      }, 500);
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('tensorchat.generateTests', () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) return;
      const text = editor.document.getText(editor.selection);
      if (!text) return;
      if (!codePanel) {
        codePanel = createPanel(context, 'code');
        codePanel.onDidDispose(() => { codePanel = null; }, null, context.subscriptions);
      }
      codePanel.reveal();
      setTimeout(() => {
        codePanel.webview.postMessage({
          type: 'injectCode', code: text,
          action: 'tests',
          language: editor.document.languageId
        });
      }, 500);
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('tensorchat.listModels', async () => {
      cachedModels = await listModelsFromProvider();
      if (cachedModels.length === 0) {
        vscode.window.showWarningMessage('No models found. Is the inference server running?');
        return;
      }
      const items = cachedModels.map(m => ({
        label: m.name,
        description: `${m.paramSize} ${m.quantization}`.trim(),
        detail: m.family
      }));
      const picked = await vscode.window.showQuickPick(items, {
        placeHolder: 'Select a model'
      });
      if (picked) {
        await vscode.workspace.getConfiguration('tensorchat').update('model', picked.label, true);
        vscode.window.showInformationMessage(`Model set to: ${picked.label}`);
        if (chatPanel) chatPanel.webview.postMessage({ type: 'modelChanged', model: picked.label });
        if (codePanel) codePanel.webview.postMessage({ type: 'modelChanged', model: picked.label });
      }
    })
  );
}

function deactivate() {}

module.exports = { activate, deactivate };
