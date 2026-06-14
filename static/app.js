'use strict';

const MODE_FIRST_EXAMPLES = {
  demo:        'Should a diabetic applicant with A1C below 7.0 qualify for preferred term life?',
  openai:      'Should a diabetic applicant with A1C below 7.0 qualify for preferred term life?',
  text2cypher: 'Which underwriting rules apply to John Smith?',
  auto:        'Should a diabetic applicant with A1C below 7.0 qualify for preferred term life?',
};

const ALL_KNOWN_SAMPLES = new Set([
  'Should a diabetic applicant with A1C below 7.0 qualify for preferred term life?',
  'How does controlled diabetes affect preferred underwriting?',
  'What role does tobacco use play in the risk profile?',
  'Explain how controlled diabetes affects preferred underwriting.',
  'What does the underwriting manual say about tobacco use?',
  'Which underwriting rules apply to John Smith?',
  'What risk factors does John Smith have?',
  'What policy is John Smith applying for?',
  "Based on John Smith's profile and underwriting rules, what is your recommendation?",
]);

// ── DOM refs ─────────────────────────────────────────────────────────────────
const questionEl  = document.getElementById('question');
const askBtn      = document.getElementById('ask-btn');
const loadingEl   = document.getElementById('loading');
const errorEl     = document.getElementById('error');
const resultsEl   = document.getElementById('results');

// ── Init ──────────────────────────────────────────────────────────────────────
questionEl.value = MODE_FIRST_EXAMPLES.demo;

askBtn.addEventListener('click', runQuery);

questionEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) runQuery();
});

document.querySelectorAll('input[name="mode"]').forEach(r =>
  r.addEventListener('change', onModeChange)
);

// One delegated listener covers all four hint strips
document.getElementById('query-card').addEventListener('click', e => {
  if (e.target.classList.contains('auto-example') ||
      e.target.classList.contains('t2c-example')) {
    questionEl.value = e.target.textContent;
  }
});

// Show the correct strip for the default selected mode on page load
onModeChange();

// ── Main query flow ───────────────────────────────────────────────────────────
async function runQuery() {
  const question = questionEl.value.trim();
  if (!question) return;

  const mode = document.querySelector('input[name="mode"]:checked')?.value || 'demo';

  setLoading(true, mode);
  clearError();
  hide(resultsEl);

  try {
    const res = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, mode }),
    });

    const data = await res.json();

    if (!res.ok) {
      showError(data.detail || `Error ${res.status}`);
      return;
    }

    render(data);
    show(resultsEl);
    resultsEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (err) {
    showError('Network error — is the server running? ' + err.message);
  } finally {
    setLoading(false);
  }
}

// ── Render all sections ───────────────────────────────────────────────────────
function render(data) {
  const isT2C  = data.mode === 'text2cypher';
  const isAuto = data.mode === 'auto';
  const strategy = data.selected_strategy; // "openai_graph" | "text2cypher" | "hybrid" | undefined

  renderProviderBar(data);
  renderCompatWarning(data.compatibility_warning);
  renderReindexNotice(data);
  renderRouterReason(data);
  renderQuestion(data.question);

  const showGraphRAG = !isT2C && (!isAuto || strategy === 'openai_graph' || strategy === 'hybrid');
  const showT2C      = isT2C  || (isAuto && (strategy === 'text2cypher' || strategy === 'hybrid'));

  showEl('phase1-card',      showGraphRAG);
  showEl('phase2-card',      showGraphRAG);
  showEl('decision-card',    showGraphRAG);
  showEl('citations-card',   showGraphRAG);
  showEl('t2c-cypher-card',  showT2C);
  showEl('t2c-results-card', showT2C);

  const pureT2C = isT2C || (isAuto && strategy === 'text2cypher');
  document.getElementById('reasoning-badge').textContent = pureT2C ? 'Answer' : 'Reasoning';

  if (showT2C)      renderText2Cypher(data);
  if (showGraphRAG) {
    renderChunks(data.matched_chunks || []);
    renderGraphContext(data.graph_context || {});
    renderDecision(data.decision);
    renderCitations(data.citations || []);
  }

  renderReasoning(data.reasoning || []);
}

// Section 1 — Question
function renderQuestion(question) {
  document.getElementById('display-question').textContent = question;
}

// Section 2 — Phase 1: Vector retrieval
function renderChunks(chunks) {
  const el = document.getElementById('chunks-list');
  el.innerHTML = '';

  if (!chunks.length) {
    el.appendChild(emptyState('No document chunks matched.'));
    return;
  }

  chunks.forEach(c => {
    const item = div('chunk-item');
    const header = div('chunk-header');

    const src = span('chunk-source', esc(c.source || '—'));
    const score = c.score != null
      ? span('chunk-score', `similarity ${Number(c.score).toFixed(4)}`)
      : span('chunk-score', '');

    header.append(src, score);
    item.append(header, span('chunk-text', esc(c.text || '')));
    el.appendChild(item);
  });
}

// Section 3 — Phase 2: Graph traversal
function renderGraphContext(ctx) {
  const el = document.getElementById('graph-context');
  el.innerHTML = '';

  const grid = div('context-grid');

  grid.appendChild(contextGroup('Applicants',
    (ctx.applicants || []).map(a => `${a.name}, age ${a.age}`)
  ));

  grid.appendChild(contextGroup('Policies',
    (ctx.policies || []).map(p => `${p.name}  (${p.type || 'n/a'})`)
  ));

  grid.appendChild(contextGroup('Risk Factors',
    (ctx.risk_factors || []).map(rf => `${rf.name}  [${rf.category}]`)
  ));

  // Rules — full width
  const rulesDiv = div('context-group rules-group');
  const rulesH3 = document.createElement('h3');
  rulesH3.textContent = `Underwriting Rules (${(ctx.rules || []).length})`;
  rulesDiv.appendChild(rulesH3);

  if (!(ctx.rules || []).length) {
    rulesDiv.appendChild(emptyState('None'));
  } else {
    ctx.rules.forEach(r => {
      const row = div('rule-item');
      const badge = span('rule-decision-tag', r.decision || '');
      badge.style.cssText = decisionStyle(r.decision);
      const title = span('rule-title', r.title || '');
      row.append(badge, title);
      rulesDiv.appendChild(row);
    });
  }

  grid.appendChild(rulesDiv);
  el.appendChild(grid);
}

function contextGroup(label, items) {
  const groupDiv = div('context-group');
  const h3 = document.createElement('h3');
  h3.textContent = `${label} (${items.length})`;
  groupDiv.appendChild(h3);

  if (!items.length) {
    groupDiv.appendChild(emptyState('None'));
  } else {
    items.forEach(text => {
      const tag = span('context-tag', esc(text));
      groupDiv.appendChild(tag);
    });
  }
  return groupDiv;
}

// Section 4 — Decision
function renderDecision(decision) {
  const el = document.getElementById('decision-badge');
  const label = (decision || 'UNKNOWN').replace(/_/g, ' ');   // non-breaking space
  const badge = span(`decision-badge decision-${decision || 'UNKNOWN'}`, label);
  el.innerHTML = '';
  el.appendChild(badge);
}

// Section 5 — Reasoning
function renderReasoning(reasoning) {
  const ol = document.getElementById('reasoning-list');
  ol.innerHTML = '';
  if (!reasoning.length) {
    ol.insertAdjacentHTML('afterbegin',
      '<li class="empty-state">No reasoning provided.</li>');
    return;
  }
  reasoning.forEach(text => {
    const li = document.createElement('li');
    li.textContent = text;
    ol.appendChild(li);
  });
}

// Section 6 — Citations
function renderCitations(citations) {
  const el = document.getElementById('citations-list');
  el.innerHTML = '';

  if (!citations.length) {
    el.appendChild(emptyState('No citations.'));
    return;
  }

  citations.forEach(cit => {
    const row = div('citation-item');
    let badgeClass, badgeText, citText;

    if (typeof cit === 'string') {
      badgeClass = 'badge-other';
      badgeText  = 'Source';
      citText    = cit;
    } else if (cit.type === 'DocumentChunk') {
      badgeClass = 'badge-chunk';
      badgeText  = 'Chunk';
      const s    = cit.relevance_score != null ? `  (score ${cit.relevance_score})` : '';
      citText    = `${cit.source}${s}`;
    } else {
      badgeClass = 'badge-rule';
      badgeText  = 'Rule';
      citText    = `${cit.title}  →  ${cit.decision}`;
    }

    row.append(
      span(`citation-badge ${badgeClass}`, badgeText),
      span('citation-text', esc(citText))
    );
    el.appendChild(row);
  });
}

// Provider bar
function renderProviderBar(data) {
  const modeLabels = { openai: 'OpenAI', text2cypher: 'Text2Cypher', demo: 'Learning', auto: 'Auto' };
  document.getElementById('mode-display').textContent = modeLabels[data.mode] || 'Learning';

  const isAuto = data.mode === 'auto';

  // Toggle standard vs auto fields
  showEl('std-embedding-sep',   !isAuto);
  showEl('std-embedding-group', !isAuto);
  showEl('std-llm-sep',         !isAuto);
  showEl('std-llm-group',       !isAuto);
  showEl('auto-router-sep',     isAuto);
  showEl('auto-router-group',   isAuto);

  const hasSelected = isAuto && !!data.selected_strategy;
  showEl('auto-selected-sep',   hasSelected);
  showEl('auto-selected-group', hasSelected);

  if (!isAuto) {
    document.getElementById('embedding-display').textContent = data.embedding_provider || '—';
    document.getElementById('llm-display').textContent       = data.llm_provider || '—';
  }
  if (hasSelected) {
    document.getElementById('auto-selected-display').textContent = data.selected_strategy;
  }

  const hasStrategy = !!data.retrieval_strategy && !isAuto;
  showEl('strategy-sep',   hasStrategy);
  showEl('strategy-label', hasStrategy);
  if (hasStrategy) {
    document.getElementById('strategy-display').textContent = data.retrieval_strategy;
  }
}

// Compatibility warning — only shown if auto-reseed failed
function renderCompatWarning(warning) {
  const el = document.getElementById('compat-warning');
  if (warning) {
    document.getElementById('compat-warning-text').textContent = warning;
    show(el);
  } else {
    hide(el);
  }
}

// Reindex notice — shown once after automatic embedding switch
function renderReindexNotice(data) {
  const el = document.getElementById('reindex-notice');
  if (data.reindexed) {
    const modeLabels = { openai: 'OpenAI', text2cypher: 'Text2Cypher', demo: 'Learning', auto: 'Auto' };
    document.getElementById('reindex-mode').textContent = modeLabels[data.mode] || 'Learning';
    show(el);
  } else {
    hide(el);
  }
}

// Router reason callout — auto mode only
function renderRouterReason(data) {
  const box = document.getElementById('router-reason-box');
  if (data.mode === 'auto' && data.router_reason) {
    document.getElementById('router-selected-badge').textContent = data.selected_strategy || '—';
    document.getElementById('router-reason-text').textContent    = data.router_reason;
    show(box);
  } else {
    hide(box);
  }
}

// ── Mode hint strips ──────────────────────────────────────────────────────────
function getSelectedMode() {
  return document.querySelector('input[name="mode"]:checked')?.value || 'demo';
}

function onModeChange() {
  const mode = getSelectedMode();

  const q = questionEl.value.trim();
  if (ALL_KNOWN_SAMPLES.has(q)) {
    questionEl.value = MODE_FIRST_EXAMPLES[mode] || q;
  }

  showEl('demo-hints',   mode === 'demo');
  showEl('openai-hints', mode === 'openai');
  showEl('t2c-hints',    mode === 'text2cypher');
  showEl('auto-hints',   mode === 'auto');
}

function renderText2Cypher(data) {
  document.getElementById('t2c-cypher-block').textContent =
    data.generated_cypher || '(no query generated)';
  renderRawResults(data.raw_query_results || []);
}

function renderRawResults(records) {
  const container = document.getElementById('t2c-results-body');
  container.innerHTML = '';

  if (!records.length) {
    container.appendChild(emptyState('No records returned.'));
    return;
  }

  const keys = Object.keys(records[0]);
  if (!keys.length) {
    container.appendChild(emptyState('Records returned no columns.'));
    return;
  }

  const table = document.createElement('table');
  table.className = 'raw-results-table';

  const thead = table.createTHead();
  const hrow  = thead.insertRow();
  keys.forEach(k => {
    const th = document.createElement('th');
    th.textContent = k;
    hrow.appendChild(th);
  });

  const tbody = table.createTBody();
  records.forEach(rec => {
    const row = tbody.insertRow();
    keys.forEach(k => {
      const td  = row.insertCell();
      const val = rec[k];
      td.textContent = val == null ? '—'
        : typeof val === 'object' ? JSON.stringify(val)
        : String(val);
    });
  });

  container.appendChild(table);
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function div(className) {
  const el = document.createElement('div');
  if (className) el.className = className;
  return el;
}

function span(className, text) {
  const el = document.createElement('span');
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}

function emptyState(msg) {
  return span('empty-state', msg);
}

function esc(str) {
  // Escape for textContent assignment — textContent is already XSS-safe,
  // but we use innerHTML in a few places so keep this available.
  return String(str);
}

function decisionStyle(decision) {
  const styles = {
    APPROVE:                   'background:#dcfce7;color:#15803d',
    REFER_FOR_REVIEW:          'background:#fef9c3;color:#a16207',
    REQUIRE_ADDITIONAL_REVIEW: 'background:#ffedd5;color:#c2410c',
    DECLINE:                   'background:#fee2e2;color:#b91c1c',
    APPROVE_FACTOR:            'background:#dcfce7;color:#15803d',
    REFER_IF_UNCONTROLLED:     'background:#fef9c3;color:#a16207',
  };
  return styles[decision] || 'background:#f3f4f6;color:#374151';
}

function setLoading(on, mode) {
  askBtn.disabled = on;
  askBtn.textContent = on ? 'Running…' : 'Ask →';
  if (on) {
    const modeLabels = { openai: 'OpenAI', text2cypher: 'Text2Cypher', demo: 'Learning', auto: 'Auto' };
    const modeLabel  = modeLabels[mode] || 'Learning';
    const hint = mode === 'openai'
      ? ' <span class="loading-hint">(first switch re-indexes embeddings)</span>'
      : '';
    loadingEl.innerHTML =
      `<span class="spinner"></span> Running ${modeLabel} pipeline${hint}…`;
    show(loadingEl);
  } else {
    hide(loadingEl);
  }
}

function showError(msg) {
  errorEl.textContent = `⚠ ${msg}`;
  show(errorEl);
}

function clearError() {
  errorEl.textContent = '';
  hide(errorEl);
}

function showEl(id, visible) {
  document.getElementById(id).classList.toggle('hidden', !visible);
}
function show(el) { el.classList.remove('hidden'); }
function hide(el) { el.classList.add('hidden'); }
