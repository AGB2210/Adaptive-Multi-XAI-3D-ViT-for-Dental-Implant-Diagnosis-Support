/* Implant Site Screening -- page logic. No framework, no build step.
 *
 * Every number shown comes from the server; this file only lays it out. Arrays
 * arrive as raw uint8 with their shape in an X-Shape header, indexed C-order
 * as (x, y, z) -- the voxel frame the model reads.
 */
"use strict";

// ---------------------------------------------------------------- helpers
const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "style") node.setAttribute("style", v);
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { const body = await res.json(); msg = body.detail || msg; } catch (_) { /* not JSON */ }
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return res;
}
const getJSON = (p) => api(p).then((r) => r.json());
const postJSON = (p, body) => api(p, {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body ?? {}),
}).then((r) => r.json());

async function getArray(p) {
  const res = await api(p);
  const shape = res.headers.get("X-Shape").split(",").map(Number);
  return { shape, data: new Uint8Array(await res.arrayBuffer()) };
}

async function waitJob(jobId, onProgress) {
  for (;;) {
    const job = await getJSON(`/api/jobs/${jobId}`);
    onProgress(job);
    if (job.status === "done") return job;
    if (job.status === "failed") throw new Error(job.error || "the job failed");
    await sleep(600);
  }
}

const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// Matplotlib's inferno as a polynomial fit -- perceptually uniform and
// monotonic in lightness, the colormap the research figures use.
function inferno(t) {
  const c = [
    [0.0002189403691192265, 0.001651004631001012, -0.01948089843709184],
    [0.1065134194856116, 0.5639564367884091, 3.932712388889277],
    [11.60249308247187, -3.972853965665698, -15.9423941062914],
    [-41.70399613139459, 17.43639888205313, 44.35414519872813],
    [77.162935699427, -33.40235894210092, -81.80730925738993],
    [-71.31942824499214, 32.62606426397723, 73.20951985803202],
    [25.13112622477341, -12.24266895238567, -23.07032500287172],
  ];
  return [0, 1, 2].map((i) => {
    let v = c[6][i];
    for (let k = 5; k >= 0; k--) v = c[k][i] + t * v;
    return Math.max(0, Math.min(255, Math.round(v * 255)));
  });
}
const LUT = Array.from({ length: 256 }, (_, i) => inferno(i / 255));
const OVERLAY_FLOOR = Math.round(0.2 * 255);   // the research figures' masking

const STATUS = {
  feasible: { label: "Needs implant · feasible", var: "--st-feasible" },
  not_feasible: { label: "Needs implant · not feasible", var: "--st-not-feasible" },
  borderline: { label: "Borderline · review", var: "--st-borderline" },
  not_needed: { label: "No implant needed", var: "--st-not-needed" },
  unmeasurable: { label: "Unmeasurable", var: "--st-none" },
  no_position: { label: "No position", var: "--st-none" },
};
const LOWER_RIGHT = [47, 46, 45, 44, 43, 42, 41];
const LOWER_LEFT = [31, 32, 33, 34, 35, 36, 37];
const OUTPUT_LABEL = {
  needs_implant: "P(needs implant)",
  available_height_mm: "Bone height",
  ridge_width_mm: "Ridge width",
};

// Grey-level presets, in z-scored units. The patch arrives as uint8 over the
// server's display window; these re-map it without another request.
const WINDOWS = {
  full: { label: "Full range", range: null },
  bone: { label: "Bone", range: [0.0, 4.0] },
  soft: { label: "Soft tissue", range: [-2.0, 2.0] },
};

// ---------------------------------------------------------------- state
const S = {
  status: null,
  models: [],
  scanId: null,
  body: null,          // GET /api/scans/{id}
  rules: null,
  tooth: null,
  patch: null,
  overview: null,
  explanation: null,
  maps: {},
  overlay: "",
  slice: { x: 48, y: 48, z: 72 },
  view: { flipX: false, anteriorLowY: true, known: false },
  window: "full",
  token: 0,            // guards against stale async renders
};

// ---------------------------------------------------------------- orientation
// Display orientation is decided by the anatomy, never the header -- the same
// rule the pipeline follows for z. Patient's right goes on the screen's left
// and the incisors at the top, matching the tooth chart. Which voxel direction
// that is varies by scan, so it is read off where the 4x and 3x teeth, and the
// incisors and the molars, actually sit.
function readAnatomy() {
  const sites = ((S.body && S.body.result && S.body.result.sites) || [])
    .filter((s) => s.site_x !== null && s.site_y !== null);
  const mean = (teeth, key) => {
    const v = sites.filter((s) => teeth.includes(s.tooth)).map((s) => s[key]);
    return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
  };
  const right = mean(LOWER_RIGHT, "site_x");
  const left = mean(LOWER_LEFT, "site_x");
  const front = mean([41, 42, 31, 32], "site_y");
  const back = mean([46, 47, 36, 37], "site_y");
  S.view = {
    flipX: right !== null && left !== null && right > left,
    anteriorLowY: front === null || back === null ? true : front < back,
    known: right !== null && left !== null && front !== null && back !== null,
  };
}

// Axial image of width W (x) and height H (y): voxel <-> screen.
const axialToScreen = (x, y, W, H) => [S.view.flipX ? W - 1 - x : x, S.view.anteriorLowY ? y : H - 1 - y];
const axialFromScreen = (c, r, W, H) => [S.view.flipX ? W - 1 - c : c, S.view.anteriorLowY ? r : H - 1 - r];

// ---------------------------------------------------------------- boot
async function boot() {
  S.status = await getJSON("/api/status");
  $("device-chip").textContent = S.status.device.toUpperCase();
  $("version-chip").textContent = `v${S.status.version}`;
  S.rules = { ...S.status.rules };
  syncRuleInputs();
  renderLegend();
  $("colorbar").style.background =
    `linear-gradient(90deg, ${[0, .25, .5, .75, 1].map((t) => `rgb(${inferno(t).join(",")})`).join(",")})`;
  $("window-select").replaceChildren(...Object.entries(WINDOWS).map(([k, w]) => el("option", { value: k }, w.label)));
  await refreshModels();
  await refreshScans();
}

function syncRuleInputs() {
  $("rule-height").value = S.rules.min_height_mandible_mm;
  $("rule-width").value = S.rules.min_width_mm;
}

function mark(status) {
  const st = STATUS[status] || STATUS.no_position;
  const hollow = status === "no_position" || status === "unmeasurable";
  return el("span", { class: `mark${hollow ? " hollow" : ""}`, style: hollow ? null : `background: var(${st.var})` });
}

function renderLegend() {
  $("legend").replaceChildren(...["feasible", "not_feasible", "borderline", "not_needed", "no_position"]
    .map((k) => el("li", {}, mark(k), STATUS[k].label)));
}

// File pickers show the chosen name instead of the browser's default control.
for (const input of document.querySelectorAll(".pick input")) {
  input.addEventListener("change", () => {
    const name = input.closest(".pick").querySelector(".pick-name");
    const files = [...input.files].map((f) => f.name);
    name.textContent = files.length ? files.join(", ") : name.dataset.empty;
    name.classList.toggle("set", files.length > 0);
  });
}
function resetPicks(form) {
  for (const name of form.querySelectorAll(".pick-name")) {
    name.textContent = name.dataset.empty;
    name.classList.remove("set");
  }
}

// ---------------------------------------------------------------- models
async function refreshModels() {
  S.models = await getJSON("/api/models");
  const site = S.models.find((m) => m.kind === "site" && m.active);
  $("top-model").textContent = site ? `Model: ${site.name}` : "No model";
  renderModelList();
}

function modelFlags(d) {
  const flag = (ok, good, bad) => el("span", { class: ok ? "" : "warn" }, ok ? good : bad);
  const mae = d.validation_mae_mm || {};
  return [
    flag(d.calibrated, `calibrated, T = ${fmt(d.temperature, 3)}`, "uncalibrated, T = 1"),
    " · ",
    flag(!!d.gate, "confidence gate fitted", "gate: app default"),
    " · ",
    flag(!!mae.available_height_mm,
      `validation MAE ${fmt(mae.available_height_mm)} / ${fmt(mae.ridge_width_mm)} mm`,
      "no validation error"),
  ];
}

function renderModelList() {
  const list = $("model-list");
  if (!S.models.length) {
    list.replaceChildren(el("li", {}, el("span", { class: "none" }, "No models yet. Add the .pt you trained, with its companions.")));
    return;
  }
  list.replaceChildren(...S.models.map((m) => {
    const d = m.description || {};
    const a = d.architecture || {};
    const lines = m.kind === "site" ? [
      `${(d.outputs || []).map((o) => o.name).join(", ")}`,
      `${a.depth} blocks × ${a.num_heads} heads, width ${a.embed_dim}, patch ${a.patch_size} · ${d.patch_mm} mm input, ${d.tokens} tokens at ${d.mm_per_token} mm`,
      `Architecture ${d.architecture_source}${d.epoch !== null && d.epoch !== undefined ? ` · epoch ${d.epoch}` : ""}${m.fold !== null && m.fold !== undefined ? ` · fold ${m.fold}` : ""}${m.has_folds ? " · fold partition known" : ""}`,
    ] : [
      `Lower-jaw sites ${(d.sites || []).join(", ")}`,
      `${d.grid_mm} mm grid ${(d.input_shape || []).join(" × ")} · ${(d.n_params / 1e6).toFixed(2)}M params${d.epoch !== null && d.epoch !== undefined ? ` · epoch ${d.epoch}` : ""}${d.fold !== null && d.fold !== undefined ? ` · fold ${d.fold}` : ""}`,
      d.validation && d.validation.median_error_mm
        ? `Validation: median position error ${fmt(d.validation.median_error_mm)} mm${d.validation.orientation_accuracy !== undefined ? `, orientation ${fmt(d.validation.orientation_accuracy * 100, 1)}% correct` : ""}`
        : "No validation numbers in this checkpoint",
    ];
    return el("li", {},
      el("div", {},
        el("div", { class: "name" }, m.name,
          el("span", { class: "kind" }, m.kind === "site" ? "Site model" : "Localiser"),
          m.active ? el("span", { class: "active-label" }, "Active") : null),
        ...lines.map((t) => el("div", { class: "line" }, t)),
        m.kind === "site" ? el("div", { class: "flags" }, modelFlags(d)) : null),
      el("div", { class: "actions" },
        m.active ? null : el("button", { class: "link", type: "button", onclick: () => activateModel(m.id) }, "Use"),
        el("button", { class: "link danger", type: "button", onclick: () => removeModel(m) }, "Remove")));
  }));
}

async function activateModel(id) {
  await api(`/api/models/${id}/activate`, { method: "POST" });
  await refreshModels();
  if (S.body) renderScan();
}

async function removeModel(m) {
  if (!confirm(`Remove ${m.name}? Scans it analysed keep their results but cannot be explained again.`)) return;
  await api(`/api/models/${m.id}`, { method: "DELETE" });
  await refreshModels();
}

$("open-models").addEventListener("click", () => { $("model-error").textContent = ""; $("models-dialog").showModal(); });

$("model-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const form = ev.currentTarget;
  const button = form.querySelector("button[type=submit]");
  $("model-error").textContent = "";
  button.disabled = true;
  button.textContent = "Validating…";
  try {
    await api("/api/models", { method: "POST", body: new FormData(form) });
    form.reset();
    resetPicks(form);
    await refreshModels();
  } catch (err) {
    $("model-error").textContent = err.message;
  } finally {
    button.disabled = false;
    button.textContent = "Validate and add";
  }
});

// ---------------------------------------------------------------- scans
async function refreshScans() {
  const scans = await getJSON("/api/scans");
  if (!scans.length) {
    $("scan-list").replaceChildren(el("li", {}, el("span", { class: "none" }, "No scans yet.")));
    return;
  }
  $("scan-list").replaceChildren(...scans.map((s) => el("li", {},
    el("button", { type: "button", "aria-current": s.id === S.scanId ? "true" : "false", onclick: () => openScan(s.id) },
      el("span", { class: "id" }, s.patient_id),
      el("span", { class: "sub" }, `${new Date(s.created * 1000).toLocaleString()} · ${s.mask_name ? "with mask" : "image only"}${s.analysed ? "" : " · not analysed"}`)))));
}

$("scan-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const form = ev.currentTarget;
  const data = new FormData(form);
  if (!form.mask.files.length) data.delete("mask");
  const button = form.querySelector("button[type=submit]");
  $("scan-error").textContent = "";
  button.disabled = true;
  button.textContent = "Uploading…";
  try {
    const res = await api("/api/scans", { method: "POST", body: data }).then((r) => r.json());
    form.reset();
    resetPicks(form);
    await refreshScans();
    await openScan(res.scan.id, res.job);
  } catch (err) {
    $("scan-error").textContent = err.message;
  } finally {
    button.disabled = false;
    button.textContent = "Analyse";
  }
});

function showProgress(boxId, job) {
  const box = $(boxId);
  box.hidden = false;
  box.classList.toggle("failed", job.status === "failed");
  box.querySelector(".track span").style.width = `${Math.round((job.progress || 0) * 100)}%`;
  box.querySelector(".progress-text").textContent = job.status === "failed"
    ? `Failed: ${job.error}` : `${job.message} · ${Math.round((job.progress || 0) * 100)}%`;
}

async function trackJob(job, boxId) {
  try {
    await waitJob(job.id, (j) => showProgress(boxId, j));
    $(boxId).hidden = true;
    return true;
  } catch (err) {
    showProgress(boxId, { status: "failed", error: err.message, progress: 1 });
    return false;
  }
}

async function openScan(id, job = null) {
  const token = ++S.token;
  if (S.scanId !== id) {
    S.tooth = null; S.patch = null; S.explanation = null; S.maps = {}; S.overlay = "";
    $("scan-progress").hidden = true;
  }
  S.scanId = id;
  $("empty-state").hidden = true;
  $("scan-view").hidden = false;
  await refreshScans();
  await loadScan(token);
  const running = job ? [job] : (S.body.jobs || []);
  for (const j of running) {
    const ok = await trackJob(j, "scan-progress");
    if (token !== S.token) return;
    if (ok) { await loadScan(token); await refreshScans(); }
  }
}

async function loadScan(token) {
  const body = await getJSON(`/api/scans/${S.scanId}`);
  if (token !== S.token) return;
  S.body = body;
  if (body.result && body.result.model) await rescore(token);   // at the rules on screen
  renderScan();
  if (body.result && body.result.overview) {
    S.overview = await getArray(`/api/scans/${S.scanId}/overview`).catch(() => null);
    if (token === S.token) drawOverview();
  } else {
    S.overview = null;
  }
  if (S.tooth !== null) selectTooth(S.tooth, true);
  else renderSiteCard(null);
}

async function rescore(token = S.token) {
  if (!S.body || !S.body.result || !S.body.result.model) return;
  const scored = await postJSON(`/api/scans/${S.scanId}/score`, S.rules);
  if (token !== S.token) return;
  S.body.result = scored;
}

function renderScan() {
  const { meta, result } = S.body;
  readAnatomy();
  $("scan-title").textContent = meta.patient_id;
  const parts = [meta.image_name + (meta.mask_name ? ` + ${meta.mask_name}` : "")];
  if (result && result.scan) {
    const sc = result.scan;
    parts.push(`${sc.shape.join(" × ")} at ${sc.spacing_mm[0]} mm`);
    parts.push(`orientation ${sc.orientation_sign > 0 ? "+1" : "−1"} from ${sc.orientation_source}`);
    parts.push(`sites from ${result.site_source}`);
  }
  if (result && result.model) parts.push(`model ${result.model.name}`);
  $("scan-meta").textContent = parts.join("  ·  ");

  const role = $("patient-role");
  const pr = result && result.patient_role;
  role.className = `role ${pr ? pr.role : ""}`;
  role.textContent = pr ? pr.detail.charAt(0).toUpperCase() + pr.detail.slice(1) : "";

  const activeSite = S.models.find((m) => m.kind === "site" && m.active);
  $("repredict").hidden = !(result && result.model && activeSite && activeSite.id !== result.model.id);
  $("download-csv").hidden = !(result && result.model);
  updateCsvLink();

  $("scan-warnings").replaceChildren(...((result && result.warnings) || []).map((w) => el("li", {}, w)));
  $("chart-caption").textContent = S.view.known
    ? "Patient's right on the left, in the image and the chart · FDI 47–41 | 31–37"
    : "FDI 47–41 | 31–37 · image orientation could not be read from the sites";
  renderChart();
}

function updateCsvLink() {
  if (!S.scanId) return;
  const q = new URLSearchParams({ min_height_mandible_mm: S.rules.min_height_mandible_mm, min_width_mm: S.rules.min_width_mm });
  $("download-csv").href = `/api/scans/${S.scanId}/report.csv?${q}`;
}

function siteOf(tooth) {
  const r = S.body && S.body.result;
  return r ? r.sites.find((s) => s.tooth === tooth) : null;
}

function statusOf(site) {
  return (site && site.verdict && site.verdict.status) || "no_position";
}

function renderChart() {
  const tile = (t) => {
    const st = statusOf(siteOf(t));
    const hollow = st === "no_position" || st === "unmeasurable";
    return el("button", {
      type: "button", class: "tooth", role: "option",
      "aria-selected": S.tooth === t ? "true" : "false",
      title: `${t}: ${STATUS[st].label}`, onclick: () => selectTooth(t),
    }, String(t), el("span", {
      class: "bar", style: hollow ? `box-shadow: inset 0 0 0 1px var(--rule-2)` : `background: var(${STATUS[st].var})`,
    }));
  };
  $("tooth-chart").replaceChildren(...LOWER_RIGHT.map(tile), el("span", { class: "gap" }), ...LOWER_LEFT.map(tile));
}

// ---------------------------------------------------------------- overview
const OV_SCALE = 2;

function drawOverview() {
  const canvas = $("overview");
  const ov = S.overview;
  if (!ov || !S.body || !S.body.result) return;
  const info = S.body.result.overview;
  const [W, H] = ov.shape;                 // (x, y) after the 2x step
  const off = document.createElement("canvas");
  off.width = W; off.height = H;
  const octx = off.getContext("2d");
  const img = octx.createImageData(W, H);
  for (let r = 0; r < H; r++) {
    for (let c = 0; c < W; c++) {
      const [x, y] = axialFromScreen(c, r, W, H);
      const v = ov.data[x * H + y];
      const i = (r * W + c) * 4;
      img.data[i] = img.data[i + 1] = img.data[i + 2] = v;
      img.data[i + 3] = 255;
    }
  }
  octx.putImageData(img, 0, 0);

  canvas.width = W * OV_SCALE; canvas.height = H * OV_SCALE;
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(off, 0, 0, canvas.width, canvas.height);

  ctx.font = `500 10px ${cssVar("--mono") || "monospace"}`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (const s of S.body.result.sites) {
    if (s.site_x === null || s.site_y === null) continue;
    const [sc, sr] = axialToScreen(s.site_x / info.step, s.site_y / info.step, W, H);
    const cx = sc * OV_SCALE; const cy = sr * OV_SCALE;
    const st = statusOf(s);
    const selected = S.tooth === s.tooth;
    ctx.beginPath();
    ctx.arc(cx, cy, selected ? 10 : 8, 0, Math.PI * 2);
    if (st === "no_position" || st === "unmeasurable") {
      ctx.fillStyle = "rgba(0,0,0,.6)";
    } else {
      ctx.fillStyle = cssVar(STATUS[st].var);
    }
    ctx.fill();
    ctx.lineWidth = selected ? 2 : 1;
    ctx.strokeStyle = selected ? "#ffffff" : "rgba(0,0,0,.7)";
    ctx.stroke();
    ctx.fillStyle = "#ffffff";
    ctx.fillText(String(s.tooth), cx, cy + 0.5);
  }
}

$("overview").addEventListener("click", (ev) => {
  if (!S.overview || !S.body) return;
  const canvas = ev.currentTarget;
  const rect = canvas.getBoundingClientRect();
  const px = ((ev.clientX - rect.left) / rect.width) * canvas.width;
  const py = ((ev.clientY - rect.top) / rect.height) * canvas.height;
  const [W, H] = S.overview.shape;
  const step = S.body.result.overview.step;
  let best = null; let bestD = Infinity;
  for (const s of S.body.result.sites) {
    if (s.site_x === null || s.site_y === null) continue;
    const [sc, sr] = axialToScreen(s.site_x / step, s.site_y / step, W, H);
    const d = Math.hypot(px - sc * OV_SCALE, py - sr * OV_SCALE);
    if (d < bestD) { bestD = d; best = s; }
  }
  if (best && bestD < 24) selectTooth(best.tooth);
});

// ---------------------------------------------------------------- site
async function selectTooth(tooth, keepView = false) {
  const changed = S.tooth !== tooth;
  S.tooth = tooth;
  renderChart();
  drawOverview();
  const site = siteOf(tooth);
  renderSiteCard(site);
  if (!site || !site.prediction || !S.body.result.model) {
    $("viewer-card").hidden = true;
    return;
  }
  if (changed || !keepView || !S.patch) {
    S.explanation = null; S.maps = {}; S.overlay = "";
    $("explain-progress").hidden = true;
    renderExplanation();
    const token = S.token;
    S.patch = await getArray(`/api/scans/${S.scanId}/sites/${tooth}/patch`);
    if (token !== S.token || S.tooth !== tooth) return;
    const [X, Y, Z] = S.patch.shape;
    // patch_centre shifts the box a quarter down, so the crest sits high in it.
    S.slice = { x: Math.floor(X / 2), y: Math.floor(Y / 2), z: Math.min(Z - 1, Math.floor(Z / 2) + Math.floor(Z / 4)) };
    setupTargets();
  }
  $("viewer-card").hidden = false;
  $("patch-caption").textContent = `· ${S.body.result.model.description.patch_mm} mm cube at tooth ${tooth}, crest near the top`;
  drawPlanes();
}

function figure(label, value, unit, lines, truth) {
  return el("div", {},
    el("dt", {}, label),
    el("dd", {}, value, unit ? el("small", {}, unit) : null),
    ...lines.filter(Boolean).map((t) => el("div", { class: "sub" }, t)),
    truth ? el("div", { class: "truth" }, truth) : null);
}

function renderSiteCard(site) {
  const status = $("site-status");
  if (!site) {
    $("site-title").textContent = "Tooth site";
    status.replaceChildren();
    $("site-body").hidden = true;
    $("site-empty").hidden = false;
    return;
  }
  const st = statusOf(site);
  $("site-title").textContent = `Tooth ${site.tooth}`;
  status.replaceChildren(mark(st), STATUS[st].label);
  $("site-body").hidden = false;
  $("site-empty").hidden = true;

  const v = site.verdict || {};
  const out = (site.prediction && site.prediction.outputs) || {};
  const truth = site.truth || null;
  const desc = (S.body.result.model && S.body.result.model.description) || {};
  const mae = desc.validation_mae_mm || {};
  const err = (name) => (mae[name] ? `± ${fmt(mae[name])} mm validation error` : "Error not measured");
  const measured = (name) => (truth && typeof truth[name] === "number" ? `Mask: ${fmt(truth[name])} mm` : null);

  const figs = [];
  if ("needs_implant" in out) {
    figs.push(figure("P(needs implant)", fmt(out.needs_implant, 2), "",
      [`Decision threshold ${fmt(v.decision_threshold, 2)}`],
      truth && truth.needs_implant !== null && truth.needs_implant !== undefined
        ? `Mask: ${truth.needs_implant ? "site is empty" : `occupied (${truth.occupied_by || "tooth"})`}` : null));
  }
  figs.push(figure("Bone height", fmt(out.available_height_mm), "mm",
    [err("available_height_mm"), `Rule ≥ ${fmt(S.rules.min_height_mandible_mm)} mm`], measured("available_height_mm")));
  figs.push(figure("Ridge width", fmt(out.ridge_width_mm), "mm",
    [err("ridge_width_mm"), `Rule ≥ ${fmt(S.rules.min_width_mm)} mm`], measured("ridge_width_mm")));
  $("site-figures").replaceChildren(...figs);
  $("site-reasons").replaceChildren(...(v.reasons || []).map((r) => el("li", {}, r)));
  const notes = [];
  if (site.source === "localiser") {
    notes.push(`Position from the localiser model (heatmap spread ± ${fmt(site.position_sd_mm)} mm), not from a segmentation, so no measured values are available.`);
  }
  if (site.note) notes.push(site.note.charAt(0).toUpperCase() + site.note.slice(1) + ".");
  if (truth && truth.limiting_structure) notes.push(`Mask: height limited by ${truth.limiting_structure.replace("_", " ")}.`);
  $("site-note").textContent = notes.join(" ");
}

// ---------------------------------------------------------------- viewer
const PLANES = ["axial", "coronal", "sagittal"];
const RENDER = 4;   // canvas pixels per voxel, so the crosshair stays thin

// Each plane: size, screen -> voxel (`at`) and voxel -> screen (`to`).
// Superior is always up; left-right and anterior follow S.view.
function planeGeometry(plane) {
  const [X, Y, Z] = S.patch.shape;
  const sx = (x) => (S.view.flipX ? X - 1 - x : x);
  const sy = (y) => (S.view.anteriorLowY ? y : Y - 1 - y);
  if (plane === "axial") {
    return { w: X, h: Y, at: (c, r) => [sx(c), sy(r), S.slice.z], to: (x, y) => [sx(x), sy(y)] };
  }
  if (plane === "coronal") {
    return { w: X, h: Z, at: (c, r) => [sx(c), S.slice.y, Z - 1 - r], to: (x, y, z) => [sx(x), Z - 1 - z] };
  }
  return { w: Y, h: Z, at: (c, r) => [S.slice.x, sy(c), Z - 1 - r], to: (x, y, z) => [sy(y), Z - 1 - z] };
}

function grayLut() {
  const [w0, w1] = S.status.window;
  const range = WINDOWS[S.window].range || [w0, w1];
  return Uint8ClampedArray.from({ length: 256 }, (_, g) => {
    const z = w0 + (g / 255) * (w1 - w0);
    return Math.round(((z - range[0]) / (range[1] - range[0])) * 255);
  });
}

// Per-map display scale: the 99th percentile of its non-zero values maps to
// the top of the colour scale. Gradient maps are spiky; scaled to their maximum
// almost every voxel falls under the 20% floor. Changes what is VISIBLE only.
function displayScale(map) {
  if (map.scale) return map.scale;
  const counts = new Uint32Array(256);
  let n = 0;
  for (const v of map.data) if (v > 0) { counts[v]++; n++; }
  let acc = 0; let p99 = 255;
  for (let v = 1; v < 256; v++) { acc += counts[v]; if (acc >= 0.99 * n) { p99 = v; break; } }
  map.scale = 255 / Math.max(p99, 1);
  return map.scale;
}

function drawPlanes() {
  if (!S.patch) return;
  const [, Y, Z] = S.patch.shape;
  const map = S.overlay ? S.maps[S.overlay] : null;
  const alpha = Number($("overlay-alpha").value) / 100;
  const scale = map ? displayScale(map) : 1;
  const gray = grayLut();
  for (const plane of PLANES) {
    const { w, h, at, to } = planeGeometry(plane);
    const off = document.createElement("canvas");
    off.width = w; off.height = h;
    const octx = off.getContext("2d");
    const img = octx.createImageData(w, h);
    for (let r = 0; r < h; r++) {
      for (let c = 0; c < w; c++) {
        const [x, y, z] = at(c, r);
        const idx = (x * Y + y) * Z + z;
        const g = gray[S.patch.data[idx]];
        let rr = g; let gg = g; let bb = g;
        if (map) {
          const m = Math.min(255, Math.round(map.data[idx] * scale));
          if (m >= OVERLAY_FLOOR) {
            // Fainter attribution is drawn fainter, so the dark low end of the
            // colour scale does not read as specks on bright bone.
            const a = alpha * (0.3 + 0.7 * (m - OVERLAY_FLOOR) / (255 - OVERLAY_FLOOR));
            const col = LUT[m];
            rr = rr * (1 - a) + col[0] * a;
            gg = gg * (1 - a) + col[1] * a;
            bb = bb * (1 - a) + col[2] * a;
          }
        }
        const i = (r * w + c) * 4;
        img.data[i] = rr; img.data[i + 1] = gg; img.data[i + 2] = bb; img.data[i + 3] = 255;
      }
    }
    octx.putImageData(img, 0, 0);
    const canvas = $(`plane-${plane}`);
    canvas.width = w * RENDER; canvas.height = h * RENDER;
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, canvas.width, canvas.height);

    const [lc, lr] = to(S.slice.x, S.slice.y, S.slice.z);
    ctx.strokeStyle = "rgba(255, 255, 255, .28)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo((lc + 0.5) * RENDER, 0); ctx.lineTo((lc + 0.5) * RENDER, canvas.height);
    ctx.moveTo(0, (lr + 0.5) * RENDER); ctx.lineTo(canvas.width, (lr + 0.5) * RENDER);
    ctx.stroke();
  }
  const mm = (v) => (v * S.status.spacing_mm).toFixed(1);
  const axes = S.view.known
    ? { axial: "R ← · A ↑", coronal: "R ← · S ↑", sagittal: "A ← · S ↑" } : { axial: "", coronal: "", sagittal: "" };
  $("cap-axial").textContent = `${axes.axial}   z ${mm(S.slice.z)} mm`;
  $("cap-coronal").textContent = `${axes.coronal}   y ${mm(S.slice.y)} mm`;
  $("cap-sagittal").textContent = `${axes.sagittal}   x ${mm(S.slice.x)} mm`;
}

for (const plane of PLANES) {
  const canvas = $(`plane-${plane}`);
  canvas.addEventListener("click", (ev) => {
    if (!S.patch) return;
    const rect = canvas.getBoundingClientRect();
    const { w, h, at } = planeGeometry(plane);
    const c = Math.max(0, Math.min(w - 1, Math.floor(((ev.clientX - rect.left) / rect.width) * w)));
    const r = Math.max(0, Math.min(h - 1, Math.floor(((ev.clientY - rect.top) / rect.height) * h)));
    const [x, y, z] = at(c, r);
    if (plane === "axial") { S.slice.x = x; S.slice.y = y; }
    if (plane === "coronal") { S.slice.x = x; S.slice.z = z; }
    if (plane === "sagittal") { S.slice.y = y; S.slice.z = z; }
    drawPlanes();
  });
  canvas.addEventListener("wheel", (ev) => {
    if (!S.patch) return;
    ev.preventDefault();
    const [X, Y, Z] = S.patch.shape;
    const d = ev.deltaY > 0 ? -1 : 1;
    if (plane === "axial") S.slice.z = Math.max(0, Math.min(Z - 1, S.slice.z + d));
    if (plane === "coronal") S.slice.y = Math.max(0, Math.min(Y - 1, S.slice.y + d));
    if (plane === "sagittal") S.slice.x = Math.max(0, Math.min(X - 1, S.slice.x + d));
    drawPlanes();
  }, { passive: false });
}

$("overlay-alpha").addEventListener("input", drawPlanes);
$("window-select").addEventListener("change", (ev) => { S.window = ev.target.value; drawPlanes(); });
$("overlay-select").addEventListener("change", (ev) => showOverlay(ev.target.value));

async function showOverlay(name) {
  S.overlay = name;
  $("overlay-select").value = name;
  $("goto-peak").disabled = !name;
  if (name && !S.maps[name] && S.explanation) {
    S.maps[name] = await getArray(`/api/scans/${S.scanId}/explain/${S.explanation.key}/${name}`);
  }
  drawPlanes();
}

$("goto-peak").addEventListener("click", () => {
  const map = S.maps[S.overlay];
  if (!map) return;
  let best = 0; let at = 0;
  for (let i = 0; i < map.data.length; i++) if (map.data[i] > best) { best = map.data[i]; at = i; }
  const [, Y, Z] = map.shape;
  S.slice = { x: Math.floor(at / (Y * Z)), y: Math.floor(at / Z) % Y, z: at % Z };
  drawPlanes();
});

$("save-png").addEventListener("click", () => {
  const site = siteOf(S.tooth);
  if (!site || !S.patch) return;
  const canvases = PLANES.map((p) => $(`plane-${p}`));
  const pad = 20; const header = 96;
  const w = canvases.reduce((a, c) => a + c.width, 0) + pad * 4;
  const h = Math.max(...canvases.map((c) => c.height)) + header + pad * 2;
  const out = document.createElement("canvas");
  out.width = w; out.height = h;
  const ctx = out.getContext("2d");
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, w, h);
  const o = site.prediction.outputs;
  ctx.fillStyle = "#e6e8ea"; ctx.font = "600 20px 'IBM Plex Sans', sans-serif";
  ctx.fillText(`${S.body.meta.patient_id}  ·  tooth ${site.tooth}  ·  ${STATUS[statusOf(site)].label}`, pad, 38);
  ctx.font = "15px 'IBM Plex Sans', sans-serif";
  ctx.fillText(`P(needs implant) ${fmt(o.needs_implant, 2)}   height ${fmt(o.available_height_mm)} mm   width ${fmt(o.ridge_width_mm)} mm   rules ${S.rules.min_height_mandible_mm} / ${S.rules.min_width_mm} mm`, pad, 62);
  ctx.fillStyle = "#8c949b";
  ctx.fillText(`${S.overlay ? `${S.overlay} attribution for ${S.explanation.target}` : "no overlay"}   ·   model ${S.body.result.model.name}   ·   screening aid, not a diagnosis`, pad, 84);
  let x = pad;
  for (const c of canvases) { ctx.drawImage(c, x, header); x += c.width + pad; }
  const a = el("a", { href: out.toDataURL("image/png"), download: `${S.body.meta.patient_id}_tooth${site.tooth}.png` });
  document.body.append(a); a.click(); a.remove();
});

// ---------------------------------------------------------------- explain
function setupTargets() {
  const outputs = S.body.result.model.description.outputs || [];
  const sel = $("explain-target");
  sel.replaceChildren(...outputs.map((o) => el("option", { value: o.name }, OUTPUT_LABEL[o.name] || o.name)));
  // Bone height by default: in the mandible it IS crest-to-canal distance, the
  // target whose evidence can be checked against real anatomy.
  if (outputs.some((o) => o.name === "available_height_mm")) sel.value = "available_height_mm";
}

$("explain-run").addEventListener("click", async () => {
  if (S.tooth === null) return;
  const tooth = S.tooth;
  const button = $("explain-run");
  button.disabled = true;
  $("explain-result").hidden = true;
  try {
    const res = await postJSON(`/api/scans/${S.scanId}/sites/${tooth}/explain`,
      { target: $("explain-target").value, force: $("explain-force").checked });
    let meta = res.explanation;
    if (!meta) {
      const job = await waitJob(res.job.id, (j) => showProgress("explain-progress", j));
      meta = job.result;
    }
    $("explain-progress").hidden = true;
    if (S.tooth !== tooth) return;
    S.explanation = meta; S.maps = {};
    renderExplanation();
    await showOverlay(meta.methods.includes("fused") ? "fused" : meta.methods[0]);
  } catch (err) {
    showProgress("explain-progress", { status: "failed", error: err.message, progress: 1 });
  } finally {
    button.disabled = false;
  }
});

const methodName = (m) => (m === "fused" ? "Fused (agreement-weighted)"
  : m.replaceAll("_", " ").replace(/^./, (c) => c.toUpperCase()));

function renderExplanation() {
  const e = S.explanation;
  $("overlay-select").replaceChildren(el("option", { value: "" }, "None"),
    ...(e ? e.methods.map((m) => el("option", { value: m }, methodName(m))) : []));
  $("goto-peak").disabled = true;
  $("explain-result").hidden = !e;
  if (!e) return;

  const r = e.routing;
  const why = r.forced ? "all four methods were requested"
    : r.uncertainty === null ? "the model has no binary output to be confident about"
      : r.decision === "ensemble" ? `uncertainty ${fmt(r.uncertainty, 3)} is at or above the gate (${fmt(r.threshold, 3)})`
        : `uncertainty ${fmt(r.uncertainty, 3)} is below the gate (${fmt(r.threshold, 3)})`;
  $("explain-routing").replaceChildren(
    "Explaining ", el("b", {}, OUTPUT_LABEL[e.target] || e.target), ". ",
    el("b", {}, r.decision === "ensemble" ? "Four methods with agreement-weighted fusion" : "Attention rollout only"),
    `: ${why}. Gate ${r.source.replace(" -- ", ", ")}.`);

  if (e.fusion) {
    const entries = Object.entries(e.fusion.weights).sort((a, b) => b[1] - a[1]);
    $("explain-weights").replaceChildren(
      el("span", { class: "head" }, "Method"), el("span", { class: "head" }, "Fusion weight"), el("span", {}),
      ...entries.flatMap(([name, w]) => [
        el("span", {}, methodName(name)),
        el("span", { class: "bar" }, el("span", { style: `width:${Math.round(w * 100)}%` })),
        el("span", { class: "val" }, w.toFixed(3))]));
  } else {
    $("explain-weights").replaceChildren();
  }

  const rows = [];
  const add = (k, v, warn = false) => rows.push(el("div", {}, el("dt", {}, k), el("dd", { class: warn ? "warn" : "" }, v)));
  if (e.fusion) {
    add(`Fused ${e.fusion.eval_metric.replace("_", " ")}`, fmt(e.fusion.fused_eval, 4));
    add(`Uniform ${e.fusion.eval_metric.replace("_", " ")}`, fmt(e.fusion.uniform_eval, 4));
    add("Beats best single method", e.fusion.beats_best_individual ? "yes" : "no");
    add("Beats uniform average", e.fusion.beats_uniform ? "yes" : "no");
    add("Weighted by", e.fusion.weight_metric.replace("_", " "));
  }
  if (e.ig_completeness) {
    const ce = e.ig_completeness.relative_error;
    add("IG completeness error", `${fmt(ce * 100, 2)}%${ce > 0.05 ? " (over 5%)" : ""}`, ce > 0.05);
  }
  if (e.shap_relative_se !== undefined && e.shap_relative_se !== null) add("SHAP relative SE", fmt(e.shap_relative_se, 3));
  for (const [k, v] of Object.entries(e.timings_s || {})) add(`${methodName(k)} time`, `${fmt(v, 2)} s`);
  $("explain-diagnostics").replaceChildren(...rows);
  $("explain-notes").replaceChildren(...(e.notes || []).map((n) => el("li", {}, n.charAt(0).toUpperCase() + n.slice(1) + ".")));
}

// ---------------------------------------------------------------- rules
let ruleTimer = null;
function onRuleChange() {
  const h = Number($("rule-height").value);
  const w = Number($("rule-width").value);
  if (!(h > 0 && h < 100 && w > 0 && w < 100)) return;
  S.rules = { ...S.rules, min_height_mandible_mm: h, min_width_mm: w };
  clearTimeout(ruleTimer);
  ruleTimer = setTimeout(async () => {
    if (!S.scanId || !S.body || !S.body.result || !S.body.result.model) return;
    await rescore();
    renderScan();
    drawOverview();
    if (S.tooth !== null) renderSiteCard(siteOf(S.tooth));
  }, 200);
}
$("rule-height").addEventListener("input", onRuleChange);
$("rule-width").addEventListener("input", onRuleChange);
$("rules-reset").addEventListener("click", () => {
  S.rules = { ...S.status.rules };
  syncRuleInputs();
  onRuleChange();
});

$("repredict").addEventListener("click", async () => {
  const res = await postJSON(`/api/scans/${S.scanId}/predict`);
  const token = S.token;
  S.patch = null; S.explanation = null; S.maps = {};
  if (await trackJob(res.job, "scan-progress") && token === S.token) await loadScan(token);
});

boot().catch((err) => {
  document.body.prepend(el("p", { class: "error", style: "padding:12px 24px" }, `Could not reach the server: ${err.message}`));
});
