/* EEG Alzheimer's Detection — frontend logic */
"use strict";

const API = "/api";
const GROUP_NAMES = {
  A: "Alzheimer's Disease",
  F: "Frontotemporal Dementia",
  C: "Healthy Control",
};
const GROUP_COLORS = { A: "#ef4444", F: "#f59e0b", C: "#22c55e" };

/* ----------------------------- utilities ----------------------------- */

async function api(path, options = {}) {
  const res = await fetch(API + path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch (_) { /* ignore */ }
    throw new Error(detail);
  }
  return res.json();
}

function toast(msg, isError = false) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.className = "toast show" + (isError ? " err" : "");
  clearTimeout(el._t);
  el._t = setTimeout(() => (el.className = "toast"), 3200);
}

function setStatus(ok, text) {
  const dot = document.getElementById("status-dot");
  const label = document.getElementById("status-text");
  dot.className = "dot " + (ok ? "ok" : "err");
  label.textContent = text;
}

function el(id) { return document.getElementById(id); }

function table(containerId, columns, rows) {
  const t = el(containerId);
  if (!rows || rows.length === 0) {
    t.innerHTML = "<tbody><tr><td>No data</td></tr></tbody>";
    return;
  }
  const head = "<thead><tr>" + columns.map((c) => `<th>${c}</th>`).join("") + "</tr></thead>";
  const body = "<tbody>" + rows.map((r) =>
    "<tr>" + columns.map((c) => {
      let v = r[c];
      if (typeof v === "number") v = Number.isInteger(v) ? v : v.toFixed(4);
      return `<td>${v === null || v === undefined ? "—" : v}</td>`;
    }).join("") + "</tr>"
  ).join("") + "</tbody>";
  t.innerHTML = head + body;
}

function busy(button, on, label) {
  if (on) {
    button.dataset._label = button.innerHTML;
    button.innerHTML = `<span class="spinner"></span>${label || "Working…"}`;
    button.disabled = true;
  } else {
    button.innerHTML = button.dataset._label || button.innerHTML;
    button.disabled = false;
  }
}

/* ----------------------------- navigation ----------------------------- */

function navigate(page) {
  document.querySelectorAll(".nav-item").forEach((b) =>
    b.classList.toggle("active", b.dataset.page === page));
  document.querySelectorAll(".page").forEach((p) =>
    p.classList.toggle("active", p.id === "page-" + page));
  const titles = {
    dashboard: "Dashboard", dataset: "Dataset Overview", viewer: "EEG Viewer",
    upload: "Upload Data", preprocess: "Preprocessing", features: "Feature Analysis",
    models: "Models", evaluation: "Evaluation", explain: "Explainability",
    predict: "Prediction",
  };
  el("page-title").textContent = titles[page] || page;
  if (page === "dataset") loadDataset();
  if (page === "viewer") initViewer();
  if (page === "upload") loadUploaded();
  if (page === "preprocess") initPreprocess();
  if (page === "features") loadFeatures();
  if (page === "models") loadModels();
  if (page === "evaluation") loadEvaluation();
  if (page === "explain") initExplain();
  if (page === "predict") initPredict();
}

/* ----------------------------- dashboard ----------------------------- */

async function loadStatus() {
  try {
    const s = await api("/status");
    setStatus(s.config_loaded, s.config_loaded ? "Ready" : "Config error");
    el("dash-models").textContent = s.models.length;

    const list = el("status-list");
    const rows = [
      ["Configuration loaded", s.config_loaded],
      ["EEG data directory found", s.data_available],
      ["Features extracted", s.features_available],
      ["Evaluation metrics present", s.metrics_available],
      ["Trained models", s.models.length > 0],
    ];
    list.innerHTML = rows.map(([label, ok]) =>
      `<li>${label}<span class="badge ${ok ? "ok" : "no"}">${ok ? "YES" : "NO"}</span></li>`
    ).join("");
    return s;
  } catch (e) {
    setStatus(false, "Offline");
    toast("Cannot reach backend: " + e.message, true);
    return null;
  }
}

async function loadDashboard() {
  const status = await loadStatus();
  try {
    const summary = await api("/dataset/summary");
    el("dash-subjects").textContent = summary.n_subjects;
    el("dash-recordings").textContent = summary.recording_available;
    const counts = summary.group_counts || {};
    const labels = Object.keys(counts).map((k) => GROUP_NAMES[k] || k);
    const values = Object.values(counts);
    const colors = Object.keys(counts).map((k) => GROUP_COLORS[k] || "#888");
    Plotly.newPlot("dash-chart", [{
      x: labels, y: values, type: "bar",
      marker: { color: colors }, text: values, textposition: "auto",
    }], {
      margin: { t: 10, l: 40, r: 10, b: 60 }, height: 300,
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    }, { displayModeBar: false, responsive: true });
  } catch (e) {
    toast("Failed to load summary: " + e.message, true);
  }
}

/* ----------------------------- dataset ----------------------------- */

async function loadDataset() {
  try {
    const summary = await api("/dataset/summary");
    el("dataset-cards").innerHTML = [
      ["Subjects", summary.n_subjects],
      ["Recordings", summary.recording_available],
      ["Age range", `${summary.age_range[0]}–${summary.age_range[1]}`],
      ["MMSE range", `${summary.mmse_range[0]}–${summary.mmse_range[1]}`],
      ["AD / FTD / HC", `${summary.group_counts.A} / ${summary.group_counts.F} / ${summary.group_counts.C}`],
    ].map(([l, v]) => `<div class="card"><div class="card-label">${l}</div><div class="card-value">${v}</div></div>`).join("");

    const meta = await api("/dataset/metadata");
    table("metadata-table", meta.columns, meta.rows);
  } catch (e) {
    toast("Failed to load dataset: " + e.message, true);
  }
}

/* ----------------------------- EEG viewer ----------------------------- */

let viewerSubjects = [];

async function initViewer() {
  if (viewerSubjects.length === 0) {
    try {
      const data = await api("/subjects");
      viewerSubjects = data.subjects;
      const opts = viewerSubjects.map((s) =>
        `<option value="${s.subject_id}">${s.subject_id} — ${GROUP_NAMES[s.group] || s.group}</option>`).join("");
      el("viewer-subject").innerHTML = opts;
      el("prep-subject").innerHTML = opts;
      el("feat-subject").innerHTML = opts;
      el("predict-subject").innerHTML = opts;
    } catch (e) {
      toast("Failed to load subjects: " + e.message, true);
      return;
    }
  }
  await loadViewerChannels();
}

async function loadViewerChannels() {
  const subject = el("viewer-subject").value;
  if (!subject) return;
  try {
    const data = await api(`/eeg/${subject}?duration=2`);
    el("viewer-channels").innerHTML = data.all_channels.map((c, i) =>
      `<label class="channel-chip ${i < 6 ? "checked" : ""}">
         <input type="checkbox" value="${c}" ${i < 6 ? "checked" : ""}/> ${c}
       </label>`
    ).join("");
    el("viewer-channels").querySelectorAll("input").forEach((cb) =>
      cb.addEventListener("change", (e) =>
        e.target.closest(".channel-chip").classList.toggle("checked", e.target.checked)));
    el("viewer-info").textContent = `${data.n_channels} channels · ${data.sfreq} Hz · ${data.duration.toFixed(1)}s total`;
  } catch (e) {
    toast("Failed to load channels: " + e.message, true);
  }
}

async function loadViewerSignal() {
  const subject = el("viewer-subject").value;
  const duration = parseFloat(el("viewer-duration").value) || 10;
  const start = parseFloat(el("viewer-start").value) || 0;
  const channels = Array.from(el("viewer-channels").querySelectorAll("input:checked")).map((c) => c.value);
  if (channels.length === 0) { toast("Select at least one channel", true); return; }

  const btn = el("viewer-load");
  busy(btn, true, "Loading…");
  try {
    const q = `channels=${encodeURIComponent(channels.join(","))}&start=${start}&duration=${duration}`;
    const data = await api(`/eeg/${subject}?${q}`);
    const traces = data.channels.map((name, idx) => ({
      x: data.times,
      y: data.data[idx],
      type: "scattergl",
      mode: "lines",
      name,
      line: { width: 1 },
    }));
    Plotly.newPlot("viewer-chart", traces, {
      margin: { t: 20, l: 50, r: 20, b: 45 },
      xaxis: { title: "Time (s)", gridcolor: "#eef2f7" },
      yaxis: { title: "Amplitude (µV)", gridcolor: "#eef2f7" },
      legend: { orientation: "h", y: -0.15 },
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      hovermode: "closest",
    }, { responsive: true, scrollZoom: true });
    el("viewer-info").textContent =
      `${data.channels.length} channels · ${duration}s window · ${data.times.length} points (stride ${data.stride})`;
  } catch (e) {
    toast("Failed to load signal: " + e.message, true);
  } finally {
    busy(btn, false);
  }
}

/* ----------------------------- upload ----------------------------- */

function setupDropzone(zoneId, inputId, handler) {
  const zone = el(zoneId);
  const input = el(inputId);
  zone.addEventListener("click", () => input.click());
  input.addEventListener("change", () => { if (input.files[0]) handler(input.files[0]); });
  ["dragover", "dragenter"].forEach((ev) =>
    zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((ev) =>
    zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove("dragover"); }));
  zone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) handler(file);
  });
}

async function uploadEEG(file) {
  const out = el("eeg-upload-result");
  out.innerHTML = `<p class="muted"><span class="spinner" style="border-color:#2f6fed;border-top-color:transparent"></span>Uploading ${file.name}…</p>`;
  const form = new FormData();
  form.append("file", file);
  try {
    const info = await api("/upload/eeg", { method: "POST", body: form });
    if (info.valid) {
      out.innerHTML = `<div class="panel" style="margin:0"><b>✅ ${info.filename}</b>
        <div class="info-line">${info.n_channels} channels · ${info.sfreq} Hz · ${info.duration.toFixed(1)}s</div>
        <div class="muted">Channels: ${info.channels.slice(0, 12).join(", ")}${info.channels.length > 12 ? "…" : ""}</div></div>`;
    } else {
      out.innerHTML = `<div class="panel" style="margin:0"><b>⚠️ ${info.filename}</b>
        <div class="info-line">Could not parse: ${info.error}</div></div>`;
    }
    loadUploaded();
  } catch (e) {
    out.innerHTML = `<p style="color:#ef4444">Upload failed: ${e.message}</p>`;
  }
}

async function uploadMetadata(file) {
  const out = el("meta-upload-result");
  out.innerHTML = `<p class="muted"><span class="spinner" style="border-color:#2f6fed;border-top-color:transparent"></span>Uploading ${file.name}…</p>`;
  const form = new FormData();
  form.append("file", file);
  try {
    const r = await api("/upload/metadata", { method: "POST", body: form });
    out.innerHTML = `<div class="panel" style="margin:0">
      <b>${r.is_valid ? "✅" : "⚠️"} ${r.filename}</b>
      <div class="info-line">${r.n_rows} rows · ${r.columns.length} columns · valid: ${r.is_valid}</div>
      <pre>${r.summary}</pre></div>`;
  } catch (e) {
    out.innerHTML = `<p style="color:#ef4444">Upload failed: ${e.message}</p>`;
  }
}

async function loadUploaded() {
  try {
    const r = await api("/uploaded");
    table("uploaded-table", ["filename", "size_bytes"], r.files);
    el("predict-file").innerHTML = r.files.map((f) =>
      `<option value="${f.filename}">${f.filename}</option>`).join("");
  } catch (e) { /* silent */ }
}

/* ----------------------------- preprocessing ----------------------------- */

async function initPreprocess() {
  await initViewer();
}

async function runPreprocess() {
  const btn = el("prep-run");
  const subject = el("prep-subject").value;
  busy(btn, true, "Processing…");
  try {
    const r = await api(`/preprocess?subject_id=${encodeURIComponent(subject)}`, { method: "POST" });
    el("prep-result").innerHTML = `<div class="panel" style="margin:0">
      <b>${r.subject_id}</b>
      <div class="info-line">Steps: ${r.steps_applied.join(", ") || "none"}</div>
      <div class="info-line">Channels: ${r.n_channels_original} → ${r.n_channels_final} ·
        ${r.sfreq_original} → ${r.sfreq_final} Hz ·
        ${r.duration_original.toFixed(1)}s → ${r.duration_final.toFixed(1)}s</div>
      <div class="info-line">Excluded: ${r.excluded}</div></div>`;
  } catch (e) {
    toast("Preprocessing failed: " + e.message, true);
  } finally { busy(btn, false); }
}

async function runSegmentation() {
  const btn = el("seg-run");
  const subject = el("prep-subject").value;
  busy(btn, true, "Segmenting…");
  try {
    const r = await api(`/segment?subject_id=${encodeURIComponent(subject)}`, { method: "POST" });
    el("prep-result").innerHTML = `<div class="panel" style="margin:0">
      <b>${r.subject_id}</b>
      <div class="info-line">Total epochs: ${r.n_epochs_total} ·
        Rejected: ${r.n_epochs_rejected} · Clean: ${r.n_epochs_clean}</div>
      <div class="info-line">Epoch length: ${r.epoch_length_sec}s · Overlap: ${r.overlap * 100}% · ${r.sfreq} Hz</div></div>`;
  } catch (e) {
    toast("Segmentation failed: " + e.message, true);
  } finally { busy(btn, false); }
}

/* ----------------------------- features ----------------------------- */

async function loadFeatures() {
  try {
    const r = await api("/features?limit=50");
    el("feature-cards").innerHTML = [
      ["Subjects", r.n_subjects], ["Features", r.n_features],
    ].map(([l, v]) => `<div class="card"><div class="card-label">${l}</div><div class="card-value">${v}</div></div>`).join("");
    table("feature-table", r.meta_columns.concat(r.columns.slice(0, 10)), r.rows);
  } catch (e) {
    el("feature-cards").innerHTML = "";
    el("feature-table").innerHTML = `<tbody><tr><td>${e.message}</td></tr></tbody>`;
  }
}

async function extractFeatures() {
  const btn = el("feat-run");
  const subject = el("feat-subject").value;
  busy(btn, true, "Extracting…");
  try {
    const r = await api(`/extract?subject_id=${encodeURIComponent(subject)}`, { method: "POST" });
    el("feat-result").innerHTML = `<div class="info-line">
      ${r.subject_id}: ${r.n_epochs} epochs · ${r.n_features_per_epoch} features/epoch</div>`;
    const bands = Object.keys(r.band_power);
    Plotly.newPlot("feat-chart", [{
      x: bands, y: bands.map((b) => r.band_power[b]), type: "bar",
      marker: { color: ["#3b82f6", "#8b5cf6", "#14b8a6", "#f59e0b", "#ef4444"] },
    }], {
      margin: { t: 20, l: 50, r: 20, b: 40 }, height: 300,
      yaxis: { title: "Mean band power" }, xaxis: { title: "Band" },
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    }, { responsive: true, displayModeBar: false });
  } catch (e) {
    toast("Extraction failed: " + e.message, true);
  } finally { busy(btn, false); }
}

/* ----------------------------- models ----------------------------- */

async function loadModels() {
  try {
    const r = await api("/models");
    const rows = r.models.map((m) => ({
      name: m.name, model: m.model_name,
      f1_macro: m.test_metrics.f1_macro ?? null,
      balanced_accuracy: m.test_metrics.balanced_accuracy ?? null,
      accuracy: m.test_metrics.accuracy ?? null,
      timestamp: m.timestamp,
    }));
    table("models-table", ["name", "model", "accuracy", "balanced_accuracy", "f1_macro", "timestamp"], rows);

    const opts = r.models.map((m) => `<option value="${m.name}">${m.name}</option>`).join("");
    el("explain-model").innerHTML = opts;
    el("predict-model").innerHTML = opts;
  } catch (e) {
    toast("Failed to load models: " + e.message, true);
  }
}

/* ----------------------------- evaluation ----------------------------- */

async function loadEvaluation() {
  try {
    const m = await api("/metrics");
    const best = m.best_model ? `<div class="info-line">Best model: <b>${m.best_model}</b></div>` : "";
    el("eval-content").innerHTML = best + `<pre>${JSON.stringify(m, null, 2)}</pre>`;

    // Find a model with a confusion matrix in results
    const results = m.results || [];
    if (results.length > 0 && results[0].test_metrics) {
      const names = results.map((r) => r.name);
      const f1s = results.map((r) => r.test_metrics.f1_macro ?? 0);
      Plotly.newPlot("eval-f1", [{
        x: names, y: f1s, type: "bar", marker: { color: "#2f6fed" },
        text: f1s.map((v) => v.toFixed(3)), textposition: "auto",
      }], {
        margin: { t: 10, l: 45, r: 10, b: 40 }, height: 300,
        yaxis: { title: "F1 macro", range: [0, 1] },
        paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      }, { responsive: true, displayModeBar: false });
    }
  } catch (e) {
    el("eval-content").innerHTML = `<p class="muted">${e.message}</p>`;
  }
}

/* ----------------------------- explainability ----------------------------- */

async function initExplain() {
  await loadModels();
}

async function runExplain() {
  const btn = el("explain-run");
  const model = el("explain-model").value;
  if (!model) { toast("No model available", true); return; }
  busy(btn, true, "Computing…");
  el("explain-info").textContent = "Computing permutation importance (this can take a few seconds)…";
  try {
    const r = await api(`/explainability?model=${encodeURIComponent(model)}&n_repeats=5&top_n=15`);
    const feats = r.ranked_features.slice().reverse();
    const vals = r.ranked_importance.slice().reverse();
    Plotly.newPlot("explain-chart", [{
      x: vals, y: feats, type: "bar", orientation: "h",
      marker: { color: "#14b8a6" },
    }], {
      margin: { t: 10, l: 180, r: 20, b: 40 }, height: 460,
      xaxis: { title: "Importance (mean decrease)" },
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    }, { responsive: true, displayModeBar: false });
    el("explain-info").textContent =
      "Feature importance reflects model behavior, not biological causation.";
  } catch (e) {
    toast("Explainability failed: " + e.message, true);
    el("explain-info").textContent = "";
  } finally { busy(btn, false); }
}

/* ----------------------------- prediction ----------------------------- */

async function initPredict() {
  await loadModels();
  await loadUploaded();
}

async function runPredict() {
  const btn = el("predict-run");
  const model = el("predict-model").value;
  const source = el("predict-source").value;
  if (!model) { toast("No trained model available", true); return; }

  let q = `model=${encodeURIComponent(model)}`;
  if (source === "subject") {
    q += `&subject_id=${encodeURIComponent(el("predict-subject").value)}`;
  } else {
    const file = el("predict-file").value;
    if (!file) { toast("No uploaded file available", true); return; }
    q += `&filename=${encodeURIComponent(file)}`;
  }

  busy(btn, true, "Predicting…");
  try {
    const r = await api(`/predict?${q}`, { method: "POST" });
    const probs = r.probabilities || {};
    const bars = Object.keys(probs).sort().map((k) =>
      `<div class="prob-row">
         <span class="name">${k}</span>
         <span class="prob-track"><span class="prob-fill" style="width:${(probs[k] * 100).toFixed(1)}%"></span></span>
         <span class="prob-val">${(probs[k] * 100).toFixed(1)}%</span>
       </div>`).join("");
    el("predict-result").innerHTML = `<div class="pred-card">
      <div class="pred-class">${r.predicted_class}</div>
      <div class="prob-bars">
        <div class="info-line"><b>${r.predicted_label}</b> · source: ${r.source} · ${r.n_epochs} epochs</div>
        ${bars}
      </div>
    </div>
    <p class="muted" style="margin-top:14px">Research prediction only — not a medical diagnosis.</p>`;
  } catch (e) {
    el("predict-result").innerHTML = `<p style="color:#ef4444">Prediction failed: ${e.message}</p>`;
  } finally { busy(btn, false); }
}

/* ----------------------------- wiring ----------------------------- */

function wireEvents() {
  document.querySelectorAll(".nav-item").forEach((b) =>
    b.addEventListener("click", () => navigate(b.dataset.page)));

  el("viewer-subject").addEventListener("change", loadViewerChannels);
  el("viewer-load").addEventListener("click", loadViewerSignal);
  el("prep-run").addEventListener("click", runPreprocess);
  el("seg-run").addEventListener("click", runSegmentation);
  el("feat-run").addEventListener("click", extractFeatures);
  el("explain-run").addEventListener("click", runExplain);
  el("predict-run").addEventListener("click", runPredict);

  el("predict-source").addEventListener("change", (e) => {
    const upload = e.target.value === "upload";
    el("predict-subject-wrap").style.display = upload ? "none" : "flex";
    el("predict-file-wrap").style.display = upload ? "flex" : "none";
  });

  setupDropzone("eeg-dropzone", "eeg-file", uploadEEG);
  setupDropzone("meta-dropzone", "meta-file", uploadMetadata);
}

document.addEventListener("DOMContentLoaded", () => {
  wireEvents();
  loadDashboard();
});
