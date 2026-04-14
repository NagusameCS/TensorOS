// TensorChat Studio — Webview Frontend
// Pure black theme, KaTeX LaTeX, batched rendering for performance
const vscode = acquireVsCodeApi();

let currentTab = 'chat';
let currentModel = '';
let isStreaming = false;
let streamingNode = null;
let streamBuffer = '';
let chatHistory = [];
let perfStats = { tokPerSec: 0, tokenCount: 0, totalMs: 0, ttft: 0 };
let connectionStatus = 'disconnected';
let models = [];
let codeLanguage = 'plaintext';
let renderPending = false;
let codeRenderPending = false;

function init() {
  const mode = document.body.dataset.mode || 'chat';
  currentTab = mode;
  document.getElementById('app').innerHTML = buildShell();
  bindEvents();
  switchTab(currentTab);
  vscode.postMessage({ type: 'listModels' });
  vscode.postMessage({ type: 'getEditor' });
  restoreState();
}

function buildShell() {
  return `
    <div class="toolbar">
      <span class="brand">TensorChat</span>
      <div class="tab-bar">
        <button class="tab" data-tab="chat">Chat</button>
        <button class="tab" data-tab="code">Code</button>
        <button class="tab" data-tab="settings">Settings</button>
      </div>
      <div class="toolbar-spacer"></div>
      <div class="toolbar-status">
        <span class="status-dot" id="statusDot"></span>
        <span id="statusText">Connecting...</span>
        <select class="model-selector" id="modelSelect">
          <option value="">(detecting...)</option>
        </select>
      </div>
    </div>

    <div class="page chat-page" id="page-chat">
      <div class="chat-messages" id="chatMessages">
        <div class="empty-state" id="chatEmpty">
          <h2>TensorChat</h2>
          <p>AI-powered chat and code assistant</p>
          <div class="shortcuts">
            <div class="shortcut"><span class="key">Enter</span> Send message</div>
            <div class="shortcut"><span class="key">Shift+Enter</span> New line</div>
            <div class="shortcut"><span class="key">Ctrl+L</span> Clear chat</div>
          </div>
        </div>
      </div>
      <div class="perf-bar" id="perfBar">
        <div class="stat">
          <span class="stat-label">tok/s</span>
          <span class="stat-value" id="tokPerSec">&mdash;</span>
        </div>
        <div class="stat">
          <span class="stat-label">tokens</span>
          <span class="stat-value" id="tokenCount">&mdash;</span>
        </div>
        <div class="stat">
          <span class="stat-label">ttft</span>
          <span class="stat-value" id="ttft">&mdash;</span>
        </div>
        <div class="stat">
          <span class="stat-label">total</span>
          <span class="stat-value" id="totalMs">&mdash;</span>
        </div>
        <div class="stat">
          <span class="stat-label">model</span>
          <span class="stat-value" id="modelName">&mdash;</span>
        </div>
      </div>
      <div class="chat-input-area">
        <textarea class="chat-input" id="chatInput"
          placeholder="Ask anything..."
          rows="1"></textarea>
        <button class="send-btn" id="sendBtn">Send</button>
      </div>
    </div>

    <div class="page code-page" id="page-code">
      <div class="code-toolbar">
        <button class="code-action-btn" data-action="explain">Explain</button>
        <button class="code-action-btn" data-action="refactor">Refactor</button>
        <button class="code-action-btn" data-action="optimize">Optimize</button>
        <button class="code-action-btn" data-action="tests">Tests</button>
        <button class="code-action-btn" data-action="review">Review</button>
        <button class="code-action-btn" data-action="document">Document</button>
        <div class="toolbar-spacer"></div>
        <button class="code-action-btn" id="pullEditorBtn">Pull Editor</button>
        <button class="code-action-btn" id="applyCodeBtn">Apply</button>
        <button class="code-action-btn" id="newFileBtn">New File</button>
      </div>
      <div class="code-split">
        <div class="code-pane">
          <div class="code-pane-header">
            <span>Input</span>
            <span class="lang-badge" id="codeLangBadge">plaintext</span>
          </div>
          <textarea class="code-editor" id="codeInput"
            placeholder="Paste or pull code from the active editor..."
            spellcheck="false"></textarea>
        </div>
        <div class="code-pane">
          <div class="code-pane-header">
            <span>Output</span>
            <span class="lang-badge" id="codeOutputBadge">output</span>
          </div>
          <div class="code-output" id="codeOutput">
            <div class="empty-state">
              <h2>Code Assistant</h2>
              <p>Paste code on the left, then use the toolbar to analyze it.</p>
            </div>
          </div>
        </div>
      </div>
      <div class="perf-bar" id="codePerfBar">
        <div class="stat">
          <span class="stat-label">tok/s</span>
          <span class="stat-value" id="codeTokPerSec">&mdash;</span>
        </div>
        <div class="stat">
          <span class="stat-label">tokens</span>
          <span class="stat-value" id="codeTokenCount">&mdash;</span>
        </div>
        <div class="stat">
          <span class="stat-label">total</span>
          <span class="stat-value" id="codeTotalMs">&mdash;</span>
        </div>
      </div>
    </div>

    <div class="page settings-page" id="page-settings">
      <div class="settings-columns">
        <div class="settings-col">
          <div class="settings-group">
            <h3>Connection</h3>
            <div class="setting-row">
              <span class="setting-label">Provider</span>
              <select class="setting-select" id="settingsProvider" data-key="provider">
                <option value="hypertensor">HyperTensor</option>
                <option value="ollama">Ollama</option>
                <option value="openai-compatible">OpenAI-compatible</option>
              </select>
            </div>
            <div class="setting-row">
              <span class="setting-label">Base URL</span>
              <input class="setting-input" id="settingsBaseUrl" data-key="baseUrl" type="text" placeholder="http://127.0.0.1:8080">
            </div>
            <div class="setting-row">
              <span class="setting-label">Status</span>
              <span class="setting-row-right">
                <span id="settingsStatusDot" class="status-dot"></span>
                <span id="settingsStatus" class="stat-value">Checking...</span>
              </span>
            </div>
            <div class="setting-row">
              <span class="setting-label">Timeout</span>
              <div class="setting-row-right">
                <input class="setting-input setting-input-sm" id="settingsTimeout" data-key="timeoutMs" type="number" min="5000" max="600000" step="5000">
                <span class="setting-unit">ms</span>
              </div>
            </div>
            <div class="setting-actions">
              <button class="setting-btn" id="testConnectionBtn">Test Connection</button>
              <button class="setting-btn" id="refreshModelsBtn">Refresh Models</button>
            </div>
          </div>

          <div class="settings-group">
            <h3>Generation</h3>
            <div class="setting-row">
              <span class="setting-label">Temperature</span>
              <div class="setting-row-right">
                <input class="setting-range" id="settingsTemperature" data-key="temperature" type="range" min="0" max="2" step="0.05">
                <span class="setting-range-value" id="settingsTemperatureValue">0.7</span>
              </div>
            </div>
            <div class="setting-row">
              <span class="setting-label">Max Tokens</span>
              <input class="setting-input setting-input-sm" id="settingsMaxTokens" data-key="maxTokens" type="number" min="-1" max="131072" step="64" title="-1 = unlimited">
            </div>
            <div class="setting-row setting-row-full">
              <span class="setting-label">System Prompt</span>
              <textarea class="setting-textarea" id="settingsSystemPrompt" data-key="systemPrompt" rows="4" placeholder="System prompt..."></textarea>
            </div>
          </div>

          <div class="settings-group">
            <h3>Performance</h3>
            <div class="settings-stats-grid">
              <div class="settings-stat-card">
                <span class="settings-stat-num" id="settingsTokPerSec">&mdash;</span>
                <span class="settings-stat-label">Last tok/s</span>
              </div>
              <div class="settings-stat-card">
                <span class="settings-stat-num" id="settingsTotalTokens">0</span>
                <span class="settings-stat-label">Total tokens</span>
              </div>
              <div class="settings-stat-card">
                <span class="settings-stat-num" id="settingsTotalRequests">0</span>
                <span class="settings-stat-label">Requests</span>
              </div>
              <div class="settings-stat-card">
                <span class="settings-stat-num" id="settingsAvgTokPerSec">&mdash;</span>
                <span class="settings-stat-label">Avg tok/s</span>
              </div>
            </div>
            <div class="setting-actions">
              <button class="setting-btn setting-btn-danger" id="resetStatsBtn">Reset Stats</button>
            </div>
          </div>
        </div>

        <div class="settings-col">
          <div class="settings-group">
            <h3>Models</h3>
            <div class="model-grid" id="modelGrid">
              <div class="model-card" style="opacity:0.5">
                <div class="model-name">Loading...</div>
                <div class="model-meta">Connecting to backend</div>
              </div>
            </div>
          </div>

          <div class="settings-group" id="modelDetailGroup" style="display:none">
            <h3>Model Details</h3>
            <div id="modelDetailContent"></div>
          </div>

          <div class="settings-group">
            <h3>Data</h3>
            <div class="setting-actions">
              <button class="setting-btn setting-btn-danger" id="clearHistoryBtn">Clear Chat History</button>
              <button class="setting-btn" id="exportChatBtn">Export Chat</button>
            </div>
          </div>

          <div class="settings-group">
            <h3>Shortcuts</h3>
            <div class="setting-row"><span class="setting-label">Send</span><span class="key">Enter</span></div>
            <div class="setting-row"><span class="setting-label">New line</span><span class="key">Shift+Enter</span></div>
            <div class="setting-row"><span class="setting-label">Clear chat</span><span class="key">Ctrl+L</span></div>
            <div class="setting-row"><span class="setting-label">Open chat</span><span class="key">Ctrl+Shift+T</span></div>
          </div>

          <div class="settings-group">
            <h3>About</h3>
            <div class="setting-row">
              <span class="setting-label">Version</span>
              <span class="stat-value">0.2.0</span>
            </div>
            <div class="setting-row">
              <span class="setting-label">Engine</span>
              <span class="stat-value">Ollama-compatible API</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;
}

// ── Tab switching ───────────────────────────────────────────────────────
function switchTab(tab) {
  currentTab = tab;
  document.querySelectorAll('.tab').forEach(t =>
    t.classList.toggle('active', t.dataset.tab === tab));
  document.querySelectorAll('.page').forEach(p =>
    p.classList.toggle('active', p.id === `page-${tab}`));
  saveState();
}

// ── Events ──────────────────────────────────────────────────────────────
function bindEvents() {
  document.querySelectorAll('.tab').forEach(t =>
    t.addEventListener('click', () => switchTab(t.dataset.tab)));

  const chatInput = document.getElementById('chatInput');
  const sendBtn = document.getElementById('sendBtn');

  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendChat();
    }
  });

  chatInput.addEventListener('input', () => autoResize(chatInput));
  sendBtn.addEventListener('click', sendChat);

  document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'l') {
      e.preventDefault();
      clearChat();
    }
  });

  document.querySelectorAll('.code-action-btn[data-action]').forEach(btn =>
    btn.addEventListener('click', () => runCodeAction(btn.dataset.action)));

  document.getElementById('pullEditorBtn')?.addEventListener('click', () => {
    vscode.postMessage({ type: 'getEditor' });
  });

  document.getElementById('applyCodeBtn')?.addEventListener('click', () => {
    const output = document.getElementById('codeOutput')?.textContent || '';
    const codeMatch = output.match(/```[\w]*\n([\s\S]*?)```/);
    const code = codeMatch ? codeMatch[1] : output;
    vscode.postMessage({ type: 'applyCode', code, replaceSelection: false });
  });

  document.getElementById('newFileBtn')?.addEventListener('click', () => {
    const output = document.getElementById('codeOutput')?.textContent || '';
    const codeMatch = output.match(/```[\w]*\n([\s\S]*?)```/);
    const code = codeMatch ? codeMatch[1] : output;
    vscode.postMessage({ type: 'newFile', code, language: codeLanguage });
  });

  document.getElementById('modelSelect')?.addEventListener('change', (e) => {
    const model = e.target.value;
    if (model) vscode.postMessage({ type: 'setModel', model });
  });

  // Settings controls
  document.getElementById('testConnectionBtn')?.addEventListener('click', () => {
    const btn = document.getElementById('testConnectionBtn');
    btn.textContent = 'Testing...';
    btn.disabled = true;
    vscode.postMessage({ type: 'testConnection' });
  });

  document.getElementById('refreshModelsBtn')?.addEventListener('click', () => {
    vscode.postMessage({ type: 'listModels' });
  });

  document.getElementById('resetStatsBtn')?.addEventListener('click', () => {
    const state = vscode.getState() || {};
    vscode.setState({ ...state, totalTokens: 0, totalRequests: 0, tokPerSecHistory: [] });
    const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
    set('settingsTotalTokens', '0');
    set('settingsTotalRequests', '0');
    set('settingsAvgTokPerSec', '\u2014');
    set('settingsTokPerSec', '\u2014');
  });

  document.getElementById('clearHistoryBtn')?.addEventListener('click', () => {
    chatHistory = [];
    const container = document.getElementById('chatMessages');
    if (container) container.innerHTML = '';
    const state = vscode.getState() || {};
    vscode.setState({ ...state, chatHistory: [] });
    addMessage('system', 'Chat history cleared');
  });

  document.getElementById('exportChatBtn')?.addEventListener('click', () => {
    if (chatHistory.length === 0) return;
    const text = chatHistory.map(m => `[${m.role}]\n${m.content}`).join('\n\n---\n\n');
    vscode.postMessage({ type: 'newFile', code: text, language: 'markdown' });
  });

  // Settings inputs — debounced save
  let settingSaveTimer = null;
  function debouncedSave(key, value) {
    clearTimeout(settingSaveTimer);
    settingSaveTimer = setTimeout(() => {
      vscode.postMessage({ type: 'updateSetting', key, value });
    }, 500);
  }

  document.getElementById('settingsProvider')?.addEventListener('change', (e) => {
    vscode.postMessage({ type: 'updateSetting', key: 'provider', value: e.target.value });
  });

  document.getElementById('settingsBaseUrl')?.addEventListener('input', (e) => {
    debouncedSave('baseUrl', e.target.value);
  });

  document.getElementById('settingsTimeout')?.addEventListener('change', (e) => {
    vscode.postMessage({ type: 'updateSetting', key: 'timeoutMs', value: parseInt(e.target.value) });
  });

  document.getElementById('settingsTemperature')?.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    document.getElementById('settingsTemperatureValue').textContent = val.toFixed(2);
    debouncedSave('temperature', val);
  });

  document.getElementById('settingsMaxTokens')?.addEventListener('change', (e) => {
    vscode.postMessage({ type: 'updateSetting', key: 'maxTokens', value: parseInt(e.target.value) });
  });

  document.getElementById('settingsSystemPrompt')?.addEventListener('input', (e) => {
    debouncedSave('systemPrompt', e.target.value);
  });
}

// ── Chat ────────────────────────────────────────────────────────────────
function sendChat() {
  if (isStreaming) {
    vscode.postMessage({ type: 'cancelRequest' });
    return;
  }
  const input = document.getElementById('chatInput');
  const text = input.value.trim();
  if (!text) return;

  const empty = document.getElementById('chatEmpty');
  if (empty) empty.remove();

  addMessage('user', text);
  chatHistory.push({ role: 'user', content: text });

  const codeInput = document.getElementById('codeInput');
  const context = codeInput?.value?.trim()?.slice(0, 8000) || '';

  input.value = '';
  autoResize(input);

  vscode.postMessage({
    type: 'chat',
    text,
    history: chatHistory.slice(-20),
    context: context || undefined
  });
}

function addMessage(role, text) {
  const container = document.getElementById('chatMessages');
  const msg = document.createElement('div');
  msg.className = `msg ${role}`;

  if (role === 'assistant') {
    msg.innerHTML = renderMarkdown(text);
  } else {
    msg.textContent = text;
  }

  if (role === 'assistant' && text) {
    const actions = document.createElement('div');
    actions.className = 'msg-actions';
    actions.innerHTML = `
      <button class="msg-action-btn" data-action="copy">Copy</button>
      <button class="msg-action-btn" data-action="insert">Insert</button>
    `;
    actions.addEventListener('click', (e) => {
      const action = e.target.dataset.action;
      if (action === 'copy') {
        vscode.postMessage({ type: 'copyToClipboard', text });
      } else if (action === 'insert') {
        const codeMatch = text.match(/```[\w]*\n([\s\S]*?)```/);
        vscode.postMessage({ type: 'insertCode', code: codeMatch ? codeMatch[1] : text });
      }
    });
    msg.appendChild(actions);
  }

  container.appendChild(msg);
  container.scrollTop = container.scrollHeight;
  return msg;
}

// Batched streaming: only render via requestAnimationFrame
function startStreaming() {
  isStreaming = true;
  streamBuffer = '';
  renderPending = false;
  const container = document.getElementById('chatMessages');
  streamingNode = document.createElement('div');
  streamingNode.className = 'msg assistant streaming-cursor';
  container.appendChild(streamingNode);
  container.scrollTop = container.scrollHeight;

  const btn = document.getElementById('sendBtn');
  btn.disabled = false;
  btn.textContent = 'Stop';
  btn.classList.add('stop-btn');
  setStatus('streaming');
}

function appendStreamTokens(tokens) {
  if (!streamingNode) return;
  streamBuffer += tokens;
  // Batch rendering via rAF to avoid O(n^2) re-rendering per token
  if (!renderPending) {
    renderPending = true;
    requestAnimationFrame(() => {
      renderPending = false;
      if (streamingNode) {
        streamingNode.innerHTML = renderMarkdown(streamBuffer);
        const container = document.getElementById('chatMessages');
        container.scrollTop = container.scrollHeight;
      }
    });
  }
}

function endStreaming(tokenCount, totalMs, tokPerSec, ttft) {
  isStreaming = false;
  if (streamingNode) {
    streamingNode.classList.remove('streaming-cursor');
    streamingNode.innerHTML = renderMarkdown(streamBuffer);

    const actions = document.createElement('div');
    actions.className = 'msg-actions';
    actions.innerHTML = `
      <button class="msg-action-btn" data-action="copy">Copy</button>
      <button class="msg-action-btn" data-action="insert">Insert</button>
    `;
    const finalText = streamBuffer;
    actions.addEventListener('click', (e) => {
      const action = e.target.dataset.action;
      if (action === 'copy') {
        vscode.postMessage({ type: 'copyToClipboard', text: finalText });
      } else if (action === 'insert') {
        const codeMatch = finalText.match(/```[\w]*\n([\s\S]*?)```/);
        vscode.postMessage({ type: 'insertCode', code: codeMatch ? codeMatch[1] : finalText });
      }
    });
    streamingNode.appendChild(actions);

    chatHistory.push({ role: 'assistant', content: streamBuffer });
  }
  streamingNode = null;

  const btn = document.getElementById('sendBtn');
  btn.disabled = false;
  btn.textContent = 'Send';
  btn.classList.remove('stop-btn');

  updatePerfStats(tokenCount, totalMs, tokPerSec, ttft);
  setStatus('connected');
  saveState();
}

function updatePerfStats(tokenCount, totalMs, tokPerSec, ttft) {
  perfStats = { tokenCount, totalMs, tokPerSec, ttft: ttft || 0 };

  const set = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  };

  set('tokPerSec', tokPerSec.toFixed(1));
  set('tokenCount', tokenCount);
  set('totalMs', totalMs < 1000 ? `${Math.round(totalMs)}ms` : `${(totalMs/1000).toFixed(1)}s`);
  set('ttft', ttft ? `${Math.round(ttft)}ms` : '\u2014');
  set('modelName', currentModel || '\u2014');

  set('codeTokPerSec', tokPerSec.toFixed(1));
  set('codeTokenCount', tokenCount);
  set('codeTotalMs', totalMs < 1000 ? `${Math.round(totalMs)}ms` : `${(totalMs/1000).toFixed(1)}s`);

  set('settingsTokPerSec', `${tokPerSec.toFixed(1)} tok/s`);

  const state = vscode.getState() || {};
  const totalTokens = (state.totalTokens || 0) + tokenCount;
  const totalRequests = (state.totalRequests || 0) + 1;
  const tokPerSecHistory = (state.tokPerSecHistory || []).concat(tokPerSec).slice(-100);
  const avgTokPerSec = tokPerSecHistory.reduce((a, b) => a + b, 0) / tokPerSecHistory.length;
  set('settingsTotalTokens', totalTokens);
  set('settingsTotalRequests', totalRequests);
  set('settingsAvgTokPerSec', avgTokPerSec.toFixed(1));
  vscode.setState({ ...state, totalTokens, totalRequests, tokPerSecHistory });
}

function clearChat() {
  chatHistory = [];
  const container = document.getElementById('chatMessages');
  container.innerHTML = '';
  addMessage('system', 'Chat cleared');
  saveState();
}

// ── Code actions ────────────────────────────────────────────────────────
function runCodeAction(action) {
  if (isStreaming) return;
  const codeInput = document.getElementById('codeInput');
  const code = codeInput.value.trim();
  if (!code) {
    vscode.postMessage({ type: 'getEditor' });
    return;
  }

  const output = document.getElementById('codeOutput');
  output.innerHTML = '<span style="color:var(--text-muted)">Processing...</span>';
  output.classList.add('streaming-cursor');
  streamBuffer = '';
  codeRenderPending = false;
  isStreaming = true;

  vscode.postMessage({ type: 'codeAction', action, code, language: codeLanguage });
}

// ── Status ──────────────────────────────────────────────────────────────
function setStatus(status) {
  connectionStatus = status;
  const dot = document.getElementById('statusDot');
  const text = document.getElementById('statusText');
  if (!dot || !text) return;

  dot.className = 'status-dot';
  switch (status) {
    case 'connected':
      dot.classList.add('connected');
      text.textContent = currentModel || 'Connected';
      break;
    case 'streaming':
      dot.classList.add('streaming');
      text.textContent = 'Generating...';
      break;
    case 'error':
      dot.classList.add('error');
      text.textContent = 'Error';
      break;
    default:
      text.textContent = 'Disconnected';
  }

  const settingsStatus = document.getElementById('settingsStatus');
  if (settingsStatus) settingsStatus.textContent = status;
}

// ── Models ──────────────────────────────────────────────────────────────
function updateModelUI() {
  const select = document.getElementById('modelSelect');
  if (select) {
    select.innerHTML = '';
    if (models.length === 0) {
      select.innerHTML = '<option value="">(no models)</option>';
    } else {
      models.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.name;
        opt.textContent = m.name;
        opt.selected = m.name === currentModel;
        select.appendChild(opt);
      });
    }
  }

  const grid = document.getElementById('modelGrid');
  if (grid) {
    if (models.length === 0) {
      grid.innerHTML = `
        <div class="model-card" style="opacity:0.5">
          <div class="model-name">No models found</div>
          <div class="model-meta">Ensure Ollama is running</div>
        </div>
      `;
    } else {
      grid.innerHTML = models.map(m => `
        <div class="model-card ${m.name === currentModel ? 'selected' : ''}"
             data-model="${esc(m.name)}">
          <div class="model-name">${esc(m.name)}</div>
          <div class="model-meta">
            ${esc(m.paramSize)}${m.quantization ? ' / ' + esc(m.quantization) : ''}${m.family ? ' / ' + esc(m.family) : ''}
          </div>
        </div>
      `).join('');

      grid.querySelectorAll('.model-card[data-model]').forEach(card =>
        card.addEventListener('click', () => {
          vscode.postMessage({ type: 'setModel', model: card.dataset.model });
          vscode.postMessage({ type: 'showModelDetail', model: card.dataset.model });
        })
      );
    }
  }
}

// ── Settings population ─────────────────────────────────────────────────
function populateSettings(data) {
  const setVal = (id, val) => {
    const el = document.getElementById(id);
    if (!el) return;
    if (el.tagName === 'SELECT') el.value = val;
    else if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') el.value = val;
    else el.textContent = val;
  };

  if (data.provider) setVal('settingsProvider', data.provider);
  if (data.baseUrl) setVal('settingsBaseUrl', data.baseUrl);
  if (data.timeoutMs) setVal('settingsTimeout', data.timeoutMs);
  if (data.temperature != null) {
    setVal('settingsTemperature', data.temperature);
    const valEl = document.getElementById('settingsTemperatureValue');
    if (valEl) valEl.textContent = Number(data.temperature).toFixed(2);
  }
  if (data.maxTokens != null) setVal('settingsMaxTokens', data.maxTokens);
  if (data.systemPrompt != null) setVal('settingsSystemPrompt', data.systemPrompt);

  // Restore accumulated stats
  const state = vscode.getState() || {};
  const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
  set('settingsTotalTokens', state.totalTokens || 0);
  set('settingsTotalRequests', state.totalRequests || 0);
  if (state.tokPerSecHistory?.length > 0) {
    const avg = state.tokPerSecHistory.reduce((a, b) => a + b, 0) / state.tokPerSecHistory.length;
    set('settingsAvgTokPerSec', avg.toFixed(1));
  }
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
  if (bytes < 1073741824) return (bytes / 1048576).toFixed(1) + ' MB';
  return (bytes / 1073741824).toFixed(2) + ' GB';
}

// ── Markdown rendering with KaTeX LaTeX support ─────────────────────────
function renderMarkdown(text) {
  if (!text) return '';

  // Escape HTML first
  let html = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  // Extract and protect code blocks from further processing
  const codeBlocks = [];
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
    const id = `__CODE_${codeBlocks.length}__`;
    const langLabel = lang || 'code';
    codeBlocks.push(
      `<div class="code-block-header"><span>${langLabel}</span><button class="copy-btn" onclick="copyCode(this)">copy</button></div>` +
      `<pre><code>${code}</code></pre>`
    );
    return id;
  });

  // LaTeX display math $$...$$
  html = html.replace(/\$\$([\s\S]*?)\$\$/g, (_, tex) => {
    try {
      if (typeof katex !== 'undefined') {
        return katex.renderToString(tex.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>'), {
          displayMode: true, throwOnError: false
        });
      }
    } catch {}
    return `<pre>${tex}</pre>`;
  });

  // LaTeX inline math $...$  (not preceded/followed by space+digit pattern that looks like currency)
  html = html.replace(/(?<!\w)\$([^\$\n]+?)\$(?!\w)/g, (_, tex) => {
    try {
      if (typeof katex !== 'undefined') {
        return katex.renderToString(tex.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>'), {
          displayMode: false, throwOnError: false
        });
      }
    } catch {}
    return `<code>${tex}</code>`;
  });

  // Inline code
  html = html.replace(/`([^`\n]+)`/g, '<code>$1</code>');

  // Bold
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

  // Italic
  html = html.replace(/(?<!\*)\*([^*]+?)\*(?!\*)/g, '<em>$1</em>');

  // Headings
  html = html.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

  // Blockquotes
  html = html.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');

  // Horizontal rule
  html = html.replace(/^---$/gm, '<hr>');

  // Unordered lists
  html = html.replace(/^- (.+)$/gm, '<li>$1</li>');

  // Ordered lists
  html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

  // Line breaks (but not inside block elements)
  html = html.replace(/\n/g, '<br>');

  // Restore code blocks
  codeBlocks.forEach((block, i) => {
    html = html.replace(`__CODE_${i}__`, block);
  });

  return `<div style="word-break:break-word">${html}</div>`;
}

// ── Utility ─────────────────────────────────────────────────────────────
function esc(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 200) + 'px';
}

function saveState() {
  const state = vscode.getState() || {};
  vscode.setState({
    ...state,
    currentTab,
    chatHistory: chatHistory.slice(-50),
    currentModel
  });
}

function restoreState() {
  const state = vscode.getState();
  if (!state) return;
  if (state.currentTab) switchTab(state.currentTab);
  if (state.currentModel) currentModel = state.currentModel;
  if (state.chatHistory?.length) {
    const empty = document.getElementById('chatEmpty');
    if (empty) empty.remove();
    chatHistory = state.chatHistory;
    chatHistory.forEach(msg => addMessage(msg.role, msg.content));
  }
}

window.copyCode = function(btn) {
  const pre = btn.closest('.code-block-header')?.nextElementSibling || btn.closest('pre');
  const code = pre.querySelector('code') || pre;
  vscode.postMessage({ type: 'copyToClipboard', text: code.textContent });
  btn.textContent = '\u2713';
  setTimeout(() => { btn.textContent = 'copy'; }, 1500);
};

// ── Message handler ─────────────────────────────────────────────────────
window.addEventListener('message', (event) => {
  const msg = event.data;
  switch (msg.type) {
    case 'init':
      currentModel = msg.model || '';
      populateSettings(msg);
      break;

    case 'streamStart':
      currentModel = msg.model || currentModel;
      if (currentTab === 'chat') {
        startStreaming();
      } else {
        isStreaming = true;
        streamBuffer = '';
        codeRenderPending = false;
        const output = document.getElementById('codeOutput');
        if (output) {
          output.innerHTML = '';
          output.classList.add('streaming-cursor');
        }
      }
      setStatus('streaming');
      break;

    // Batched token message — extension sends accumulated tokens
    case 'streamTokenBatch': {
      if (currentTab === 'chat') {
        appendStreamTokens(msg.tokens);
      } else {
        streamBuffer += msg.tokens;
        if (!codeRenderPending) {
          codeRenderPending = true;
          requestAnimationFrame(() => {
            codeRenderPending = false;
            const output = document.getElementById('codeOutput');
            if (output) output.innerHTML = renderMarkdown(streamBuffer);
          });
        }
      }
      // Update live tok/s
      const tps = document.getElementById(currentTab === 'code' ? 'codeTokPerSec' : 'tokPerSec');
      if (tps && msg.tokPerSec) tps.textContent = msg.tokPerSec.toFixed(1);
      break;
    }

    // Legacy single-token message (backwards compat)
    case 'streamToken': {
      if (currentTab === 'chat') {
        appendStreamTokens(msg.token);
      } else {
        streamBuffer += msg.token;
        if (!codeRenderPending) {
          codeRenderPending = true;
          requestAnimationFrame(() => {
            codeRenderPending = false;
            const output = document.getElementById('codeOutput');
            if (output) output.innerHTML = renderMarkdown(streamBuffer);
          });
        }
      }
      const tps2 = document.getElementById(currentTab === 'code' ? 'codeTokPerSec' : 'tokPerSec');
      if (tps2 && msg.tokPerSec) tps2.textContent = msg.tokPerSec.toFixed(1);
      break;
    }

    case 'streamEnd':
      if (currentTab === 'chat') {
        endStreaming(msg.tokenCount, msg.totalMs, msg.tokPerSec, msg.ttft);
      } else {
        isStreaming = false;
        const output = document.getElementById('codeOutput');
        if (output) {
          output.classList.remove('streaming-cursor');
          output.innerHTML = renderMarkdown(streamBuffer);
        }
        updatePerfStats(msg.tokenCount, msg.totalMs, msg.tokPerSec, msg.ttft);
        setStatus('connected');
      }
      break;

    case 'streamError':
      isStreaming = false;
      if (currentTab === 'chat') {
        if (streamingNode) {
          streamingNode.classList.remove('streaming-cursor');
          streamingNode.className = 'msg error';
          streamingNode.textContent = msg.text;
          streamingNode = null;
        } else {
          addMessage('error', msg.text);
        }
        const errBtn = document.getElementById('sendBtn');
        errBtn.disabled = false;
        errBtn.textContent = 'Send';
        errBtn.classList.remove('stop-btn');
      } else {
        const output = document.getElementById('codeOutput');
        if (output) {
          output.classList.remove('streaming-cursor');
          output.innerHTML = `<div style="color:var(--error)">${esc(msg.text)}</div>`;
        }
      }
      setStatus('error');
      break;

    case 'streamCancelled':
      if (currentTab === 'chat') {
        endStreaming(0, 0, 0, 0);
      } else {
        isStreaming = false;
        const output = document.getElementById('codeOutput');
        if (output) output.classList.remove('streaming-cursor');
      }
      setStatus('connected');
      break;

    case 'error':
      if (currentTab === 'chat') {
        addMessage('error', msg.text);
      } else {
        const output = document.getElementById('codeOutput');
        if (output) output.innerHTML = `<div style="color:var(--error)">${esc(msg.text)}</div>`;
      }
      setStatus('error');
      break;

    case 'modelList':
      models = msg.models || [];
      if (models.length > 0) {
        setStatus('connected');
        if (!currentModel) currentModel = models[0].name;
      }
      updateModelUI();
      break;

    case 'modelChanged':
      currentModel = msg.model;
      updateModelUI();
      setStatus('connected');
      break;

    case 'editorContent':
      if (msg.info) {
        const codeInput = document.getElementById('codeInput');
        if (codeInput) {
          codeInput.value = msg.info.selectedText || msg.info.fullText || '';
          codeLanguage = msg.info.language || 'plaintext';
          const badge = document.getElementById('codeLangBadge');
          if (badge) badge.textContent = codeLanguage;
        }
      }
      break;

    case 'injectPrompt': {
      switchTab('chat');
      const input = document.getElementById('chatInput');
      if (input) {
        input.value = msg.text;
        autoResize(input);
        input.focus();
      }
      break;
    }

    case 'injectCode': {
      switchTab('code');
      const codeInput = document.getElementById('codeInput');
      if (codeInput) {
        codeInput.value = msg.code;
        codeLanguage = msg.language || 'plaintext';
        const badge = document.getElementById('codeLangBadge');
        if (badge) badge.textContent = codeLanguage;
      }
      if (msg.action) {
        setTimeout(() => runCodeAction(msg.action), 300);
      }
      break;
    }

    case 'workspaceInfo':
      break;

    case 'settingsData':
      populateSettings(msg);
      break;

    case 'connectionTest': {
      const btn = document.getElementById('testConnectionBtn');
      const dot = document.getElementById('settingsStatusDot');
      const statusEl = document.getElementById('settingsStatus');
      if (btn) { btn.textContent = 'Test Connection'; btn.disabled = false; }
      if (msg.success) {
        if (dot) { dot.className = 'status-dot connected'; }
        if (statusEl) statusEl.textContent = `Connected (${msg.latency}ms)`;
        setStatus('connected');
      } else {
        if (dot) { dot.className = 'status-dot error'; }
        if (statusEl) statusEl.textContent = msg.error || 'Failed';
        setStatus('error');
      }
      break;
    }

    case 'modelDetail': {
      const group = document.getElementById('modelDetailGroup');
      const content = document.getElementById('modelDetailContent');
      if (group && content && msg.detail) {
        group.style.display = '';
        const d = msg.detail;
        content.innerHTML = `
          <div class="setting-row"><span class="setting-label">Name</span><span class="stat-value">${esc(d.name || '')}</span></div>
          <div class="setting-row"><span class="setting-label">Family</span><span class="stat-value">${esc(d.family || '\u2014')}</span></div>
          <div class="setting-row"><span class="setting-label">Parameters</span><span class="stat-value">${esc(d.paramSize || '\u2014')}</span></div>
          <div class="setting-row"><span class="setting-label">Quantization</span><span class="stat-value">${esc(d.quantization || '\u2014')}</span></div>
          <div class="setting-row"><span class="setting-label">Size</span><span class="stat-value">${d.size ? formatBytes(d.size) : '\u2014'}</span></div>
          <div class="setting-row"><span class="setting-label">Context Length</span><span class="stat-value">${esc(d.contextLength || '\u2014')}</span></div>
          <div class="setting-row"><span class="setting-label">Format</span><span class="stat-value">${esc(d.format || '\u2014')}</span></div>
          ${d.template ? `<div class="setting-row setting-row-full"><span class="setting-label">Template</span><pre class="setting-pre">${esc(d.template)}</pre></div>` : ''}
        `;
      }
      break;
    }
  }
});

init();
