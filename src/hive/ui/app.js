/* HiveMindRL dashboard — vanilla JS, no external dependencies. */
'use strict';

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const state = { options: null, graphResult: null, graphJobId: null, running: new Set() };

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function api(path, options = {}) {
  const opts = { ...options };
  if (opts.body !== undefined && !(opts.body instanceof FormData)) {
    opts.headers = { 'Content-Type': 'application/json', ...(opts.headers || {}) };
  }
  const response = await fetch(path, opts);
  const text = await response.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = null; }
  if (!response.ok) {
    throw new ApiError((data && data.error) || `HTTP ${response.status}`, response.status);
  }
  return data;
}

/* --------------------------------------------------------------- formatting */

const nf = new Intl.NumberFormat('vi-VN', { maximumFractionDigits: 1 });
const nf2 = new Intl.NumberFormat('vi-VN', { maximumFractionDigits: 2 });
const nf3 = new Intl.NumberFormat('vi-VN', { maximumFractionDigits: 3 });

function fmt(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  if (!Number.isFinite(value)) return '∞';
  const formatter = digits <= 1 ? nf : digits === 2 ? nf2 : nf3;
  return formatter.format(Number(value));
}

function fmtTau(value) {
  if (value === null || value === undefined) return '∞';
  if (!Number.isFinite(value)) return '∞';
  if (Math.abs(value) >= 100000) {
    return value.toExponential(1).replace('.', ',').replace('e+', 'e');
  }
  return fmt(value, 1);
}

function pct(value) {
  if (value === null || value === undefined) return '—';
  return `${(Number(value) * 100).toLocaleString('vi-VN', { maximumFractionDigits: 1 })}%`;
}

function stepLabel(entry) {
  if (!entry) return '—';
  return entry.hit ? `${fmt(entry.step, 0)} bước` : 'censor';
}

function fmtBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (ch) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
}[ch]));

const viText = (value) => escapeHtml(String(value).replace(/(\d)\.(\d)/g, '$1,$2'));

function metric(label, value, note = '', tone = '') {
  return `<div class="metric ${tone}">
    <span>${escapeHtml(label)}</span>
    <strong>${escapeHtml(value)}</strong>
    ${note ? `<small>${escapeHtml(note)}</small>` : ''}
  </div>`;
}

/* ------------------------------------------------------------------- toast */

let toastTimer = null;
function toast(message, kind = 'ok') {
  const el = $('#toast');
  el.textContent = message;
  el.className = `toast${kind === 'error' ? ' is-error' : ''}`;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 4200);
}

/* --------------------------------------------------------------- navigation */

function activateTab(tab) {
  if (!$(`#panel-${tab}`)) tab = 'overview';
  let title = 'Tổng quan';
  $$('.tab').forEach((button) => {
    const active = button.dataset.tab === tab;
    button.classList.toggle('is-active', active);
    button.setAttribute('aria-selected', String(active));
    if (active) title = button.dataset.title || button.textContent.trim();
  });
  $$('.panel').forEach((panel) => panel.classList.toggle('is-active', panel.id === `panel-${tab}`));
  const titleEl = $('#page-title');
  if (titleEl) titleEl.textContent = title;
  document.title = `${title} · HiveMindRL`;
  if (location.hash !== `#${tab}`) history.replaceState(null, '', `#${tab}`);
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function setupTabs() {
  $$('.tab').forEach((button) => {
    button.addEventListener('click', () => activateTab(button.dataset.tab));
  });
  window.addEventListener('hashchange', () => activateTab(location.hash.slice(1)));
  document.addEventListener('click', (event) => {
    const jump = event.target.closest('[data-jump]');
    if (jump) {
      event.preventDefault();
      activateTab(jump.dataset.jump);
    }
  });
}

/* ------------------------------------------------------------- server status */

function markServer(online, detail = '') {
  const el = $('#server-status');
  el.classList.toggle('is-offline', !online);
  $('.server-text', el).textContent = online ? 'máy chủ cục bộ' : (detail || 'mất kết nối');
}

async function checkHealth() {
  try {
    await api('/api/health');
    markServer(true);
  } catch {
    markServer(false);
  }
}

/* ------------------------------------------------------------------ controls */

function renderControls() {
  const options = state.options;
  const scenarioLabels = {
    low: 'Thấp [0,1; 0,4]',
    middle: 'Trung bình [0,4; 0,7]',
    high: 'Cao [0,6; 0,9]',
  };
  $$('[data-role="scenario"]').forEach((select) => {
    select.innerHTML = options.scenarios
      .map((name) => `<option value="${name}">${scenarioLabels[name] || name}</option>`)
      .join('');
    select.value = options.defaults.scenario;
  });

  const familySelect = $('#graph-family');
  familySelect.innerHTML = options.families
    .map((family) => `<option value="${family.id}">${family.label}</option>`)
    .join('');
  familySelect.value = 'ws';
  applyFamilyMeta('ws');

  const container = $('#compare-configs');
  container.innerHTML = options.default_compare.map((config) => {
    const meta = options.families.find((family) => family.id === config.family);
    const label = meta ? meta.short : config.family;
    const param = config.family === 'complete' ? '—' : fmt(config.param, 3);
    return `<label class="config-item">
      <input type="checkbox" checked data-family="${config.family}" data-param="${config.param}">
      <span class="family">${escapeHtml(label)}</span>
      <span class="param">${escapeHtml(param)}</span>
    </label>`;
  }).join('');
}

function applyFamilyMeta(familyId) {
  const meta = state.options.families.find((family) => family.id === familyId);
  if (!meta) return;
  const input = $('#graph-param');
  input.min = meta.min;
  input.max = meta.max;
  input.step = meta.step;
  input.value = meta.default;
  input.disabled = meta.id === 'complete';
  $('#graph-param-label').textContent = meta.param_label;
  $('#family-hint').textContent = meta.hint;
}

function setupGraphFamily() {
  $('#graph-family').addEventListener('change', (event) => applyFamilyMeta(event.target.value));
}

function setupPresets() {
  $$('.preset-group .chip').forEach((button) => {
    button.addEventListener('click', () => {
      if (!state.options) return;
      const target = button.closest('.preset-group').dataset.target;
      const preset = state.options.presets[button.dataset.preset];
      if (!preset) return;
      const key = target === 'rep' ? 'reproduction' : target;
      const config = preset[key];
      const form = $(`#${target}-form`);
      Object.entries(config).forEach(([name, value]) => {
        const field = form.elements.namedItem(name);
        if (field) field.value = value;
      });
      toast(`Đã áp preset “${preset.label}”`);
    });
  });
}

/* ------------------------------------------------------------------- job UI */

function makeUi(prefix) {
  return {
    prefix,
    form: $(`#${prefix}-form`),
    output: $(`#${prefix}-output`),
    status: $(`#${prefix}-status`),
    bar: $(`#${prefix}-bar`),
    progress: $(`#${prefix}-progress`),
    metrics: $(`#${prefix}-metrics`),
    elapsed: $(`#${prefix}-elapsed`),
    error: $(`#${prefix}-error`),
    button: $(`#${prefix}-run`),
    originalText: null,
  };
}

function setRunning(ui, running) {
  if (!ui.button) return;
  if (running) {
    ui.originalText = ui.button.textContent;
    ui.button.disabled = true;
    ui.button.textContent = 'Đang chạy…';
  } else {
    ui.button.disabled = false;
    ui.button.textContent = ui.originalText || 'Chạy mô phỏng';
  }
  if (ui.output) ui.output.classList.toggle('is-running', running);
}

function paintJob(ui, job) {
  const percent = Math.max(1, Math.round((job.progress || 0) * 100));
  ui.bar.style.width = `${percent}%`;
  ui.progress.setAttribute('aria-valuenow', String(percent));
  ui.status.textContent = job.message || job.status;
  ui.status.classList.toggle('is-error', job.status === 'error');
  if (job.elapsed !== undefined && job.elapsed !== null) {
    ui.elapsed.textContent = `${job.elapsed.toFixed(1)} s`;
  }
}

async function pollJob(id, ui) {
  const deadline = Date.now() + 20 * 60 * 1000;
  while (Date.now() < deadline) {
    const job = await api(`/api/jobs/${id}`);
    paintJob(ui, job);
    if (job.status === 'done' || job.status === 'error') return job;
    await sleep(350);
  }
  throw new Error('Quá thời gian chờ (20 phút).');
}

async function runJob(kind, payload, ui) {
  setRunning(ui, true);
  if (ui.error) ui.error.textContent = '';
  paintJob(ui, { progress: 0.01, message: 'Đang gửi yêu cầu…', status: 'queued' });
  try {
    const job = await api('/api/jobs', { method: 'POST', body: JSON.stringify({ kind, ...payload }) });
    const finished = await pollJob(job.id, ui);
    if (finished.status === 'error') throw new Error(finished.error || 'Mô phỏng thất bại.');
    paintJob(ui, { ...finished, progress: 1, message: 'Hoàn tất' });
    const result = await api(`/api/jobs/${job.id}/result`);
    return { result, id: job.id, job: finished };
  } catch (error) {
    ui.status.textContent = 'Lỗi';
    ui.status.classList.add('is-error');
    if (ui.error) ui.error.textContent = error.message;
    toast(error.message, 'error');
    return null;
  } finally {
    setRunning(ui, false);
  }
}

/* ------------------------------------------------------------ experiment A */

function readCommon(form) {
  const values = {};
  for (const element of form.elements) {
    if (element.name) values[element.name] = element.value;
  }
  return {
    N: Number(values.N),
    steps: Number(values.steps),
    seeds: Number(values.seeds),
    n: Number(values.n || 10),
    scenario: values.scenario,
    delta: Number(values.delta || 0.1),
  };
}

function renderReproduction(result, jobId, ui) {
  const dev = Number(result.max_deviation);
  const good = dev <= 0.02;
  const params = result.params;
  ui.status.textContent = `Hoàn tất · N=${params.N} · T=${params.steps} · ${params.seeds} seed`;
  ui.status.classList.remove('is-error');

  ui.metrics.innerHTML = [
    metric('Bầy đàn · final', fmt(result.finals.well_mixed, 3), 'best mass', 'good'),
    metric('P-MCL · final', fmt(result.finals.pmcl, 3), 'tác tử đơn'),
    metric('MRD · final', fmt(result.finals.mrd, 3), 'tham chiếu'),
    metric('Sai lệch cực đại', fmt(dev, 3), 'P-MCL vs MRD', good ? 'good' : 'warn'),
    metric('t₉₀ bầy đàn', stepLabel(result.t90.well_mixed), 'đường trung bình'),
    metric('t₉₀ P-MCL', stepLabel(result.t90.pmcl), 'đường trung bình'),
  ].join('');

  const plot = $('#rep-plot');
  plot.src = `/api/jobs/${jobId}/plots/curves.png?v=${Date.now()}`;
  $('#rep-figure').hidden = false;

  const rows = [
    ['Bầy đàn (weighted voter)', result.finals.well_mixed, result.t90.well_mixed],
    ['P-MCL', result.finals.pmcl, result.t90.pmcl],
    ['MRD', result.finals.mrd, result.t90.mrd],
  ];
  $('#rep-table tbody').innerHTML = rows.map(([name, final, t90]) => `
    <tr><td class="tag">${escapeHtml(name)}</td><td class="num">${fmt(final, 3)}</td><td class="num">${stepLabel(t90)}</td></tr>
  `).join('');
  $('#rep-table-wrap').hidden = false;

  const verdict = $('#rep-verdict');
  verdict.hidden = false;
  verdict.classList.toggle('is-warn', !good);
  verdict.innerHTML = good
    ? `<strong>Đạt.</strong> P-MCL bám MRD với sai lệch tối đa ${fmt(dev, 3)} ≤ 0,02; bầy đàn cho cùng đường học.`
    : `<strong>Chưa đạt ngưỡng 0,02.</strong> Sai lệch tối đa ${fmt(dev, 3)}. Preset nhỏ có phương sai lấy mẫu cao; ngưỡng được hiệu chuẩn ở N=1000, 300 bước, 100 seed.`;
}

function setupReproduction() {
  const ui = makeUi('rep');
  ui.form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const payload = readCommon(ui.form);
    if (!Object.values(payload).slice(0, 4).every(Number.isFinite)) {
      ui.error.textContent = 'N, bước, seed và n phải là số hợp lệ.';
      return;
    }
    const run = await runJob('reproduction', payload, ui);
    if (run) renderReproduction(run.result, run.id, ui);
  });
}

/* ----------------------------------------------------------- single graph */

function familyLabel(result) {
  return result.family_label || result.family;
}

function renderGraph(result, jobId, ui) {
  state.graphResult = result;
  state.graphJobId = jobId;
  const metrics = result.metrics;
  const summary = result.summary;
  const params = result.params;
  const connected = metrics.connected;
  const hitTone = summary.hit_rate >= 1 ? 'good' : summary.hit_rate >= 0.9 ? 'warn' : 'bad';
  const finalTone = summary.final95 >= 1 ? 'good' : summary.final95 >= 0.9 ? 'warn' : 'bad';

  ui.status.textContent = `Hoàn tất · ${familyLabel(result)} · N=${params.N}`;
  ui.status.classList.remove('is-error');

  ui.metrics.innerHTML = [
    metric('τ (relaxation)', fmtTau(metrics.tau), connected ? 'thời gian trộn' : 'không liên thông', connected ? '' : 'warn'),
    metric('L (đường đi)', fmt(metrics.path_length, 2), 'khoảng cách trung bình'),
    metric('t₉₀ trung vị', fmt(summary.t90_median, 1), 'bước', hitTone),
    metric('Seed đạt t₉₀', pct(summary.hit_rate), `${params.seeds} seed`, hitTone),
    metric('Final (trung vị)', fmt(summary.final_median, 3), 'best mass'),
    metric('Final ≥ 0,95', pct(summary.final95), 'tỉ lệ seed', finalTone),
    metric('Consensus', fmt(summary.consensus_median, 3), 'khối lượng lớn nhất'),
    metric('Degree CV', fmt(metrics.degree_cv, 3), `${metrics.edges} cạnh`),
  ].join('');

  const notes = [];
  if (!connected) notes.push('Đồ thị không liên thông: τ = ∞; L chỉ tính trên các cặp đạt tới được.');
  if (result.mode === 'well_mixed') notes.push('Chế độ đối chứng trộn đều: mọi cá thể quan sát toàn đàn.');
  if (summary.final95 < 1) notes.push(`${pct(1 - summary.final95)} seed chưa đạt final ≥ 0,95 — đuôi chậm đang lộ ra.`);
  if (summary.hit_rate < 1) notes.push(`${pct(1 - summary.hit_rate)} seed bị censor ở ngưỡng 0,9 trong ${params.steps} bước.`);
  const note = $('#graph-note');
  note.hidden = notes.length === 0;
  note.innerHTML = notes.map((text) => `<div>${escapeHtml(text)}</div>`).join('');

  $('#graph-plot-tabs').hidden = false;
  $('#graph-figure').hidden = false;
  showGraphPlot('curves');
}

function showGraphPlot(kind) {
  if (!state.graphResult || !state.graphJobId) return;
  const result = state.graphResult;
  const image = $('#graph-plot');
  const caption = $('#graph-caption');
  const src = `/api/jobs/${state.graphJobId}/plots/${kind === 'graph' ? 'graph' : 'curves'}.png?v=${Date.now()}`;
  image.src = src;
  if (kind === 'graph') {
    image.alt = 'Cấu trúc đồ thị với màu theo type ở trạng thái cuối';
    caption.textContent = `Trạng thái cuối của seed cuối. Vàng = type tốt nhất (#${result.graph.best}); các màu khác là type khác. ${result.metrics.edges} cạnh.`;
  } else {
    image.alt = 'Đường học trung bình trên đồ thị';
    caption.textContent = `Trung bình trên ${result.params.seeds} seed; dải tô là ±1 độ lệch chuẩn. MRD là tham chiếu trộn đều.`;
  }
  $$('#graph-plot-tabs .plot-tab').forEach((button) => {
    button.classList.toggle('is-active', button.dataset.plot === kind);
  });
}

function setupGraph() {
  const ui = makeUi('graph');
  ui.form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const common = readCommon(ui.form);
    const values = {};
    for (const element of ui.form.elements) if (element.name) values[element.name] = element.value;
    if (![common.N, common.steps, common.seeds, common.n].every(Number.isFinite)) {
      ui.error.textContent = 'N, bước, seed và n phải là số hợp lệ.';
      return;
    }
    const payload = {
      ...common,
      family: values.family,
      param: values.family === 'complete' ? 1 : Number(values.param),
      mode: values.mode,
    };
    const run = await runJob('graph', payload, ui);
    if (run) renderGraph(run.result, run.id, ui);
  });
  $$('#graph-plot-tabs .plot-tab').forEach((button) => {
    button.addEventListener('click', () => showGraphPlot(button.dataset.plot));
  });
}

/* --------------------------------------------------------------- comparison */

function renderCompare(result, jobId, ui) {
  const fit = result.fit;
  ui.status.textContent = `Hoàn tất · ${result.rows.length} cấu hình · ${result.params.seeds} seed`;
  ui.status.classList.remove('is-error');

  const fitBox = $('#compare-fit');
  const warningsBox = $('#compare-warnings');
  fitBox.hidden = false;
  if (fit.slope !== null && fit.slope !== undefined) {
    const slopeCi = fit.ci && fit.ci.slope;
    fitBox.innerHTML = `
      <span class="fit-title">Hiệu chuẩn trên cấu hình không can thiệp</span>
      <span>t₉₀ = <strong>${fmt(fit.intercept, 2)}</strong> + <strong>${fmt(fit.slope, 2)}</strong> · L</span>
      <span>R² = <strong>${fmt(fit.r2, 3)}</strong></span>
      <span>n = <strong>${fit.n_train}</strong></span>
      ${slopeCi ? `<span>CI 95% độ dốc = <strong>${fmt(slopeCi[0], 2)} – ${fmt(slopeCi[1], 2)}</strong></span>` : ''}
    `;
  } else {
    fitBox.innerHTML = '<span class="fit-title">Chưa đủ điểm để fit</span><span>Cần ít nhất 3 cấu hình không thuộc họ hai buồng.</span>';
  }

  const warnings = fit.warnings || [];
  warningsBox.hidden = warnings.length === 0 || fit.slope === null || fit.slope === undefined;
  warningsBox.innerHTML = warnings.map((text) => `<div>${viText(text)}</div>`).join('');

  const body = result.rows.map((row) => `
    <tr class="${row.family === 'two_room' ? 'is-intervention' : ''}">
      <td class="tag">${escapeHtml(row.label)}</td>
      <td class="num">${fmtTau(row.tau)}</td>
      <td class="num">${fmt(row.path_length, 2)}</td>
      <td class="num">${fmt(row.t90_median, 1)}</td>
      <td class="num">${pct(row.hit_rate)}</td>
      <td class="num">${pct(row.final95)}</td>
      <td>${row.connected ? 'có' : 'không'}</td>
    </tr>
  `).join('');
  $('#compare-table tbody').innerHTML = body;
  $('#compare-table-wrap').hidden = false;

  const image = $('#compare-plot');
  image.src = `/api/jobs/${jobId}/plots/compare.png?v=${Date.now()}`;
  $('#compare-figure').hidden = false;

  const notes = [];
  const censored = result.rows.filter((row) => row.hit_rate < 1);
  const intervention = result.rows.filter((row) => row.family === 'two_room');
  if (intervention.length) {
    const taus = intervention.map((row) => row.tau).filter((value) => value !== null);
    const spread = taus.length ? `${fmtTau(Math.min(...taus))} → ${fmtTau(Math.max(...taus))}` : '—';
    notes.push(`Hai buồng: τ trải ${spread} nhưng t₉₀ gần như đứng yên — kiểm định tách τ khỏi L.`);
  }
  if (censored.length) {
    notes.push(`${censored.length} cấu hình có seed bị censor ở ngưỡng 0,9; t₉₀ trung vị là cận dưới.`);
  }
  if (result.params.N < 200) {
    notes.push('N nhỏ: τ và L khác bảng trong báo cáo (N=500).');
  }
  const note = $('#compare-note');
  note.hidden = notes.length === 0;
  note.innerHTML = notes.map((text) => `<div>${escapeHtml(text)}</div>`).join('');
}

function setupCompare() {
  const ui = makeUi('compare');
  $('#compare-all').addEventListener('click', () => {
    $$('#compare-configs input').forEach((input) => { input.checked = true; });
  });
  $('#compare-none').addEventListener('click', () => {
    $$('#compare-configs input').forEach((input) => { input.checked = false; });
  });
  ui.form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const configs = $$('#compare-configs input:checked').map((input) => ({
      family: input.dataset.family,
      param: Number(input.dataset.param),
    }));
    if (!configs.length) {
      ui.error.textContent = 'Chọn ít nhất một cấu hình.';
      return;
    }
    const common = readCommon(ui.form);
    if (![common.N, common.steps, common.seeds, common.n].every(Number.isFinite)) {
      ui.error.textContent = 'N, bước, seed và n phải là số hợp lệ.';
      return;
    }
    const run = await runJob('compare', { ...common, configs }, ui);
    if (run) renderCompare(run.result, run.id, ui);
  });
}

/* --------------------------------------------------------------- stored CSV */

async function loadStoredList() {
  const container = $('#stored-list');
  try {
    const data = await api('/api/results');
    if (!data.items.length) {
      container.innerHTML = '<p class="muted">Chưa có CSV trong results/. Chạy scripts/run_all.py hoặc check_heavy_tail.py trước.</p>';
      return;
    }
    container.innerHTML = data.items.map((item) => `
      <div class="file-row">
        <div class="file-meta">
          <strong>${escapeHtml(item.name)}.csv</strong>
          <span>${escapeHtml(item.modified)} · ${fmtBytes(item.size)}</span>
        </div>
        <button class="btn" data-file="${escapeHtml(item.name)}">Xem</button>
        <a class="btn btn-ghost" href="/api/results/${encodeURIComponent(item.name)}.csv" download>Tải</a>
      </div>
    `).join('');
    container.addEventListener('click', (event) => {
      const button = event.target.closest('[data-file]');
      if (button) loadStoredFile(button.dataset.file);
    });
  } catch (error) {
    container.innerHTML = `<p class="muted">Không đọc được danh sách: ${escapeHtml(error.message)}</p>`;
  }
}

async function loadStoredFile(name) {
  const view = $('#stored-view');
  view.innerHTML = '<p class="muted">Đang tải…</p>';
  try {
    const data = await api(`/api/results/${encodeURIComponent(name)}`);
    const columns = data.columns;
    const head = columns.map((column) => `<th>${escapeHtml(column)}</th>`).join('');
    const rows = data.data.map((row) => `<tr>${columns.map((column) => {
      const value = row[column];
      return `<td>${value === null || value === undefined ? '—' : escapeHtml(value)}</td>`;
    }).join('')}</tr>`).join('');
    view.innerHTML = `
      <div class="result-head">
        <div>
          <span class="card-kicker">${escapeHtml(data.name)}.csv</span>
          <h3>${data.rows} dòng · cập nhật ${escapeHtml(data.modified)}</h3>
        </div>
        <a class="btn btn-ghost" href="/api/results/${encodeURIComponent(data.name)}.csv" download>Tải CSV</a>
      </div>
      ${data.truncated ? '<p class="hint">Chỉ hiển thị 500 dòng đầu; tải CSV để xem đầy đủ.</p>' : ''}
      <div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table></div>
    `;
  } catch (error) {
    view.innerHTML = `<p class="muted">Lỗi đọc tệp: ${escapeHtml(error.message)}</p>`;
  }
}

/* ------------------------------------------------------------ law summary */

async function loadLawDiagnostics() {
  const line = $('#law-line');
  if (!line) return;
  const tag = $('#law-tag');
  const warningsBox = $('#law-warnings');
  try {
    const data = await api('/api/law');
    if (data.error) throw new Error(data.error);
    line.textContent = `t₉₀ = ${fmt(data.intercept, 2)} + ${fmt(data.slope, 2)} · L`;
    $('#law-r2').textContent = fmt(data.r2, 3);
    $('#law-n').textContent = String(data.n);
    const slopeCi = data.ci && data.ci.slope;
    $('#law-ci').textContent = slopeCi ? `${fmt(slopeCi[0], 2)} – ${fmt(slopeCi[1], 2)}` : '—';
    const warnings = data.warnings || [];
    tag.textContent = warnings.length ? `${warnings.length} lưu ý` : 'ổn định';
    tag.className = `law-tag ${warnings.length ? 'warn' : 'ok'}`;
    warningsBox.innerHTML = warnings.length
      ? warnings.map((text) => `<div>${viText(text)}</div>`).join('')
      : '<div class="ok">Không có cảnh báo chẩn đoán trên tập huấn luyện hiện tại.</div>';
  } catch (error) {
    tag.textContent = 'chưa có dữ liệu';
    tag.className = 'law-tag err';
    line.textContent = 'Chưa đủ dữ liệu để hiệu chuẩn';
    warningsBox.innerHTML = `<div>${escapeHtml(error.message)}</div>`;
  }
}

/* -------------------------------------------------------------------- tests */

function setupTests() {
  const ui = {
    prefix: 'tests',
    form: null,
    output: null,
    status: $('#tests-status'),
    bar: $('#tests-bar'),
    progress: $('#tests-progress'),
    elapsed: $('#tests-elapsed'),
    error: $('#tests-error'),
    button: $('#run-tests'),
    originalText: null,
  };

  ui.button.addEventListener('click', async () => {
    if (ui.button.disabled) return;
    $('#tests-output').textContent = 'Đang chạy pytest…';
    $('#tests-badge').innerHTML = '<span class="badge-test pending">Đang chạy</span>';
    const run = await runJob('tests', {}, ui);
    if (!run) {
      $('#tests-badge').innerHTML = '<span class="badge-test fail">Lỗi</span>';
      return;
    }
    const result = run.result;
    const output = result.output || '';
    $('#tests-output').textContent = output || '(không có output)';
    const passedMatch = output.match(/(\d+)\s+passed/);
    const failedMatch = output.match(/(\d+)\s+failed/);
    if (result.passed) {
      $('#tests-status').textContent = 'Tất cả test đạt';
      $('#tests-badge').innerHTML = `<span class="badge-test pass">${passedMatch ? `${passedMatch[1]} passed` : 'passed'}</span>`;
      if (passedMatch) $('#test-count-badge').textContent = `${passedMatch[1]} test`;
      toast(`pytest: ${passedMatch ? passedMatch[1] + ' test đạt' : 'đạt'}`);
    } else {
      $('#tests-status').textContent = 'Có test thất bại';
      $('#tests-badge').innerHTML = `<span class="badge-test fail">${failedMatch ? `${failedMatch[1]} failed` : 'failed'}</span>`;
      toast('pytest báo lỗi — xem output', 'error');
    }
  });
}

/* --------------------------------------------------------------------- init */

async function init() {
  setupTabs();
  setupPresets();
  setupGraphFamily();
  setupReproduction();
  setupGraph();
  setupCompare();
  setupTests();

  document.addEventListener('keydown', (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
      const panel = $('.panel.is-active');
      const form = panel ? $('form', panel) : null;
      if (form) form.requestSubmit();
    }
  });

  try {
    const [health, options] = await Promise.all([api('/api/health'), api('/api/options')]);
    state.options = options;
    markServer(true);
    renderControls();
    document.body.dataset.ready = 'true';
    if (health.version) {
      $('#server-status').title = `HiveMindRL ${health.version} · ${health.root}`;
    }
    loadStoredList();
    loadLawDiagnostics();
  } catch (error) {
    markServer(false, 'không tải được cấu hình');
    toast(`Không kết nối được máy chủ: ${error.message}`, 'error');
  }

  activateTab(location.hash.slice(1) || 'overview');
  checkHealth();
  setInterval(checkHealth, 20000);
}

document.addEventListener('DOMContentLoaded', init);
