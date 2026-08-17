const state = {
  datasets: [],
  files: [],
  activeDataset: null,
  fluorescence: {
    images: [],
    selected: new Set(),
    running: false,
    progressTimer: null,
  },
  results: {
    records: [],
    selectedId: null,
    recordedJobIds: new Set(),
  },
  agent: {
    sessionId: null,
    activeJobId: null,
    morphologyJobId: null,
    morphologyStatus: null,
    pollTimer: null,
    activeScript: "yichao",
    selectedDataset: "",
    availableFiles: [],
    selectedFiles: new Set(),
  },
};

const pipelineScripts = {
  fluorescence: {
    title: "Viability detection",
    databaseHint: "analysis-outputs/fluorescence_prediction",
    text: `AUTOAPPDEV_PIPELINE 1
TASK {"id":"fluorescence_viability","title":"Organoid viability detection","objective":"Read one or more images from datasets, segment organoids, predict viability, quantify morphology, sort samples, and write an evidence-bound report."}
STEP {"id":"inspect","block":"plan","title":"Resolve dataset inputs","instruction":"Resolve the exact requested image file or dataset directory under datasets. Do not substitute another dataset."}
ACTION {"type":"dataset","target":"datasets"}
STEP {"id":"predict","block":"work","title":"Predict organoid viability","instruction":"Run scripts/run_fluorescence_prediction.py once with the exact input path(s), an output directory under analysis-outputs/fluorescence_prediction, and the requested sort order."}
ACTION {"type":"script","target":"scripts/run_fluorescence_prediction.py"}
STEP {"id":"review","block":"debug","title":"Review segmentation and evidence","instruction":"Confirm results.json, results.csv, report.md, overlays, masks, crops, viability scores, and morphology evidence exist. Flag low YOLO or SAM confidence."}
STEP {"id":"report","block":"summary","title":"Report results","instruction":"Summarize ranked viability scores and measured image evidence without claiming biological causality."}`,
  },
  yichao: {
    title: "Yichao pix2pix differentiation prediction",
    databaseHint: "analysis-outputs/yichao_instance_pairs/database/instance_pairs.sqlite",
    text: `AUTOAPPDEV_PIPELINE 1
TASK {"id":"yichao_pix2pix","title":"Yichao fluorescence prediction dataset","objective":"Prepare paired brightfield and fluorescence instances for pix2pix training."}
STEP {"id":"inspect","block":"plan","title":"Inspect data","instruction":"Check Yichao 1/2/3/4/5/6 structure, channel mapping, and existing instance-pair database."}
ACTION {"type":"read","target":"references/Yichao"}
STEP {"id":"segment","block":"work","title":"Segment brightfield","instruction":"Run the multiscale Cellpose segmentation pipeline on brightfield channel c1 and save overlays/intermediates."}
ACTION {"type":"script","target":"analysis-tools/yichao_instance_pairs"}
STEP {"id":"pair","block":"work","title":"Build pix2pix pairs","instruction":"Crop matched c1 brightfield and c0 fluorescence instances, then resize or pad to 256x256."}
ACTION {"type":"dataset","target":"analysis-outputs/yichao_pix2pix_256"}
STEP {"id":"database","block":"work","title":"Maintain database","instruction":"Backfill edge padding flags, size quantiles, source image metadata, and resized-pair links into SQLite."}
ACTION {"type":"database","target":"analysis-outputs/yichao_instance_pairs/database/instance_pairs.sqlite"}
STEP {"id":"review","block":"summary","title":"Review quality","instruction":"Report edge padding, instance size quantiles, debris filtering risks, and preview paths."}`,
  },
  zhengyu: {
    title: "DEO quality-selected segmentation and metric pipeline",
    databaseHint: "analysis-outputs/density_growth",
    text: `AUTOAPPDEV_PIPELINE 1
TASK {"id":"zhengyu_deo_metrics","title":"DEO segmentation and morphology metrics","objective":"Run Cellpose first, compare a signal-recovery candidate only when Cellpose QC fails, select the objectively better segmentation, and compute metrics only from that selected mask."}
STEP {"id":"read_method","block":"plan","title":"Read canonical method","instruction":"Open the repository-local segmentation handoff and readable TeX method source before running scripts."}
ACTION {"type":"read","target":"references/codex_segmentation_handoff.md"}
ACTION {"type":"read","target":"references/deo_segmentation_metric_method_tex/main.tex"}
STEP {"id":"segment","block":"work","title":"Submit managed multiscale segmentation","instruction":"Do not run the Cellpose script in a Codex shell. POST one JSON request to http://127.0.0.1:8080/api/morphology/run with the exact selected relative paths in paths, workflow_id zhengyu, the user's instruction, and this Studio job id in parent_job_id. The backend owns the long-running Cellpose process. Report the accepted morphology job id and return without waiting for completion."}
ACTION {"type":"api","target":"http://127.0.0.1:8080/api/morphology/run"}
STEP {"id":"quantify","block":"work","title":"Compute DEO metrics","instruction":"Use the metrics produced by the same repository-local runner to summarize growth, fusion, compactness, and differentiation evidence for the selected images."}
ACTION {"type":"database","target":"analysis-outputs/density_growth"}
STEP {"id":"validate","block":"debug","title":"Validate against references","instruction":"Compare outputs with saved intermediate masks and metric catalogs, then report deviations."}
STEP {"id":"report","block":"summary","title":"Summarize run","instruction":"Write output paths, database status, and quality-control notes."}`,
  },
  y27632: {
    title: "Y-27632 dataset analysis",
    databaseHint: "analysis-outputs/morphology_jobs",
    text: `AUTOAPPDEV_PIPELINE 1
TASK {"id":"y27632_morphology","title":"Y-27632 dataset analysis","objective":"Analyze selected Y-27632 TIFFs by concentration and day using App80 stage parameters, Cellpose-first segmentation, quality-scored fallback comparison, morphology metrics, masks, overlays, and a concise report."}
STEP {"id":"read_method","block":"plan","title":"Read App80 method","instruction":"Read the repository segmentation handoff and App80 Y-27632 method notes before submission."}
ACTION {"type":"read","target":"references/codex_segmentation_handoff.md"}
ACTION {"type":"read","target":"references/app80_multiscale_cellpose_large_recovery_method.md"}
STEP {"id":"segment","block":"work","title":"Submit managed Y-27632 analysis","instruction":"POST exactly one request to http://127.0.0.1:8080/api/morphology/run with only the selected TIFF paths, workflow_id y27632, the user's instruction, and this Studio job id in parent_job_id. Do not launch Cellpose from a Codex shell and do not wait for completion."}
ACTION {"type":"api","target":"http://127.0.0.1:8080/api/morphology/run"}
STEP {"id":"quantify","block":"work","title":"Compute concentration-aware metrics","instruction":"Use only the selected best mask per image to quantify growth, fusion-context, roundness, edge, darkness, and concentration/day comparisons."}
STEP {"id":"report","block":"summary","title":"Report Y-27632 results","instruction":"Report the managed job id, output directory, QC method, and concentration/day analysis artifacts."}`,
  },
  sodium_alginate: {
    title: "Sodium alginate dataset analysis",
    databaseHint: "analysis-outputs/morphology_jobs",
    text: `AUTOAPPDEV_PIPELINE 1
TASK {"id":"sodium_alginate_morphology","title":"Sodium alginate dataset analysis","objective":"Analyze selected sodium alginate TIFFs by condition and day using App65 stage parameters, Cellpose-first segmentation, quality-scored fallback comparison, morphology metrics, masks, overlays, and a concise report."}
STEP {"id":"read_method","block":"plan","title":"Read App65 method","instruction":"Read the repository segmentation handoff and sodium alginate experiment-design notes before submission."}
ACTION {"type":"read","target":"references/codex_segmentation_handoff.md"}
ACTION {"type":"read","target":"references/experiment_design/06_figure3_alginate_design.md"}
STEP {"id":"segment","block":"work","title":"Submit managed sodium alginate analysis","instruction":"POST exactly one request to http://127.0.0.1:8080/api/morphology/run with only the selected TIFF paths, workflow_id sodium_alginate, the user's instruction, and this Studio job id in parent_job_id. Do not launch Cellpose from a Codex shell and do not wait for completion."}
ACTION {"type":"api","target":"http://127.0.0.1:8080/api/morphology/run"}
STEP {"id":"quantify","block":"work","title":"Compute condition-aware metrics","instruction":"Use only the selected best mask per image to quantify growth, fusion-context, differentiation morphology, edge, darkness, and condition/day comparisons."}
STEP {"id":"report","block":"summary","title":"Report sodium alginate results","instruction":"Report the managed job id, output directory, QC method, and condition/day analysis artifacts."}`,
  },
  compactness: {
    title: "../Compactness compactness analysis",
    databaseHint: "../Compactness",
    text: `AUTOAPPDEV_PIPELINE 1
TASK {"id":"compactness_analysis","title":"Compactness image-analysis pipeline","objective":"Build and maintain a compactness-focused organoid image analysis workflow."}
STEP {"id":"inspect","block":"plan","title":"Inspect Compactness repo","instruction":"Find the image sources, current scripts, database outputs, and expected compactness definitions."}
ACTION {"type":"read","target":"../Compactness"}
STEP {"id":"segment","block":"work","title":"Segment organoids","instruction":"Run or adapt multiscale brightfield segmentation and save instance masks and overlays."}
STEP {"id":"quantify","block":"work","title":"Quantify compactness","instruction":"Measure area, perimeter, solidity, eccentricity, texture, and compactness scores per instance."}
STEP {"id":"database","block":"work","title":"Maintain database","instruction":"Store per-image and per-instance metrics with source paths, crop geometry, and QC flags."}
ACTION {"type":"database","target":"../Compactness"}
STEP {"id":"review","block":"debug","title":"Review edge cases","instruction":"Sample high/low compactness instances and flag debris, edge clipping, and bad focus."}
STEP {"id":"report","block":"summary","title":"Report dataset readiness","instruction":"Summarize usable images, metric distributions, and recommended filtering."}`,
  },
  generic: {
    title: "Generic organoid agent workflow",
    databaseHint: "analysis-outputs",
    text: `AUTOAPPDEV_PIPELINE 1
TASK {"id":"organoid_generic","title":"Generic organoid analysis task","objective":"Inspect data, segment objects, quantify instances, maintain a database, and report results."}
STEP {"id":"inspect","block":"plan","title":"Inspect task","instruction":"Identify inputs, channels, imaging design, output folders, and existing references."}
STEP {"id":"segment","block":"work","title":"Segment images","instruction":"Run the appropriate segmentation pipeline and save masks, overlays, and intermediates."}
STEP {"id":"quantify","block":"work","title":"Quantify instances","instruction":"Extract per-image and per-instance metrics with QC flags."}
STEP {"id":"database","block":"work","title":"Maintain database","instruction":"Create or update the SQLite database and summary manifests."}
STEP {"id":"report","block":"summary","title":"Report outputs","instruction":"Document output paths, counts, filters, and next actions."}`,
  },
};

async function fetchJson(url, options = {}) {
  const resp = await fetch(url, options);
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(text || resp.statusText);
  }
  return resp.json();
}

async function postJson(url, payload) {
  return fetchJson(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

function setActiveTab(tabName) {
  document.querySelectorAll(".subtab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.tab === tabName);
  });
  document.querySelectorAll(".panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `panel-${tabName}`);
  });
}

const stageDefaults = {
  data: "datasets",
  workflow: "agent",
  results: "viability-results",
};

function setActiveStage(stageName, targetTab = null) {
  document.querySelectorAll(".stage-tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.stage === stageName);
  });
  document.querySelectorAll(".subtab-group").forEach((group) => {
    group.classList.toggle("active", group.dataset.stageGroup === stageName);
  });
  setActiveTab(targetTab || stageDefaults[stageName]);
}

function renderList(containerId, items, onClick) {
  const container = document.getElementById(containerId);
  if (!items.length) {
    container.innerHTML = "<div class='muted'>No items found.</div>";
    return;
  }
  container.innerHTML = "";
  items.forEach((item, idx) => {
    const div = document.createElement("div");
    div.className = "list-item";
    div.style.animationDelay = `${idx * 0.02}s`;
    div.innerHTML = `
      <div><strong>${item.name || item.path}</strong></div>
      <div class="meta">${item.size_human || ""} ${item.kind ? `• ${item.kind}` : ""}</div>
    `;
    div.addEventListener("click", () => onClick(item));
    container.appendChild(div);
  });
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function renderInlineMarkdown(text) {
  const pattern = /\[([^\]]+)\]\(([^)]+)\)/g;
  let result = "";
  let lastIndex = 0;
  let match;
  while ((match = pattern.exec(text)) !== null) {
    result += escapeHtml(text.slice(lastIndex, match.index));
    const label = escapeHtml(match[1]);
    const url = match[2];
    if (url.startsWith("http://") || url.startsWith("https://")) {
      result += `<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${label}</a>`;
    } else {
      result += `${label} (${escapeHtml(url)})`;
    }
    lastIndex = match.index + match[0].length;
  }
  result += escapeHtml(text.slice(lastIndex));
  return result;
}

function renderMarkdown(markdown) {
  const lines = markdown.split(/\r?\n/);
  let html = "";
  let inList = false;
  lines.forEach((line) => {
    const trimmed = line.trim();
    if (!trimmed) {
      if (inList) {
        html += "</ul>";
        inList = false;
      }
      return;
    }
    if (trimmed.startsWith("### ")) {
      if (inList) {
        html += "</ul>";
        inList = false;
      }
      html += `<h4>${renderInlineMarkdown(trimmed.slice(4))}</h4>`;
      return;
    }
    if (trimmed.startsWith("## ")) {
      if (inList) {
        html += "</ul>";
        inList = false;
      }
      html += `<h3>${renderInlineMarkdown(trimmed.slice(3))}</h3>`;
      return;
    }
    if (trimmed.startsWith("# ")) {
      if (inList) {
        html += "</ul>";
        inList = false;
      }
      html += `<h3>${renderInlineMarkdown(trimmed.slice(2))}</h3>`;
      return;
    }
    if (trimmed.startsWith("- ")) {
      if (!inList) {
        html += "<ul>";
        inList = true;
      }
      html += `<li>${renderInlineMarkdown(trimmed.slice(2))}</li>`;
      return;
    }
    if (inList) {
      html += "</ul>";
      inList = false;
    }
    html += `<p>${renderInlineMarkdown(trimmed)}</p>`;
  });
  if (inList) {
    html += "</ul>";
  }
  return html;
}

function renderPreview(containerId, payload) {
  const container = document.getElementById(containerId);
  if (!payload) {
    container.textContent = "No preview.";
    return;
  }

  if (payload.error) {
    container.textContent = payload.error;
    return;
  }

  if (payload.kind === "table" && payload.preview?.columns) {
    const rows = payload.preview.rows.slice(0, 15);
    const headers = payload.preview.columns;
    const html = [
      "<table><thead><tr>",
      ...headers.map((h) => `<th>${h}</th>`),
      "</tr></thead><tbody>",
      ...rows.map(
        (row) =>
          `<tr>${row.map((v) => `<td>${String(v)}</td>`).join("")}</tr>`
      ),
      "</tbody></table>",
    ].join("");
    container.innerHTML = html;
    return;
  }

  if (payload.kind === "analysis" && payload.preview) {
    const summary = payload.preview;
    const imageHtml = summary.preview_url
      ? `<img src="${summary.preview_url}" alt="${payload.name}" />`
      : "";
    const previewMeta = summary.preview_url
      ? `<p>Embedding: ${summary.preview_source} (${summary.preview_points} points)</p>`
      : summary.preview_error
      ? `<p class="muted">Preview: ${summary.preview_error}</p>`
      : "";
    container.innerHTML = `
      <div class="tag">AnnData</div>
      ${imageHtml}
      ${previewMeta}
      <p>Observations: ${summary.n_obs}</p>
      <p>Variables: ${summary.n_vars}</p>
      <p><strong>Obs columns:</strong> ${summary.obs_columns.join(", ") || "—"}</p>
      <p><strong>Var columns:</strong> ${summary.var_columns.join(", ") || "—"}</p>
      <p><strong>Uns keys:</strong> ${summary.uns_keys.join(", ") || "—"}</p>
    `;
    return;
  }

  if (payload.kind === "image" && payload.preview?.preview_url) {
    container.innerHTML = `<img src="${payload.preview.preview_url}" alt="${payload.name}" />`;
    return;
  }

  if (payload.kind === "archive" && payload.preview?.entries) {
    const entries = payload.preview.entries
      .map((entry) => `<div>${entry.name}</div>`)
      .join("");
    const previewImage = payload.preview.preview_url
      ? `<div class="muted">Preview: ${payload.preview.preview_entry}</div><img src="${payload.preview.preview_url}" alt="Archive preview" />`
      : "";
    container.innerHTML = `
      <div class="tag">Archive</div>
      <button class="tab" id="extract-btn">Extract</button>
      ${previewImage}
      <div class="preview-body" style="margin-top:10px">${entries || "No entries."}</div>
    `;
    const btn = container.querySelector("#extract-btn");
    if (btn) {
      btn.addEventListener("click", async () => {
        btn.textContent = "Extracting…";
        try {
          const res = await fetchJson(`/api/extract?path=${payload.path}`, {
            method: "POST",
          });
          btn.textContent = `Extracted: ${res.extracted_to}`;
        } catch (err) {
          btn.textContent = "Extract failed";
        }
      });
    }
    return;
  }

  if (payload.preview?.download_url) {
    container.innerHTML = `<a href="${payload.preview.download_url}" target="_blank">Download ${payload.name}</a>`;
    return;
  }

  if (payload.preview?.lines) {
    container.innerHTML = `<pre>${payload.preview.lines.join("\n")}</pre>`;
    return;
  }

  container.textContent = "Preview not available.";
}

function appendAgentMessage(role, content) {
  const container = document.getElementById("agent-messages");
  if (!container) {
    return;
  }
  const div = document.createElement("div");
  div.className = `chat-message ${role}`;
  div.innerHTML = `<div class="chat-role">${escapeHtml(role)}</div><div>${escapeHtml(content)}</div>`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function renderPipelineBlocks(ir) {
  const canvas = document.getElementById("pipeline-blocks");
  if (!canvas) {
    return;
  }
  canvas.innerHTML = "";
  ir.tasks.forEach((task) => {
    const taskEl = document.createElement("div");
    taskEl.className = "program-block task";
    taskEl.innerHTML = `<strong>${escapeHtml(task.title)}</strong><span>${escapeHtml(task.objective || task.id)}</span>`;
    canvas.appendChild(taskEl);
    task.steps.forEach((step) => {
      const stepEl = document.createElement("div");
      stepEl.className = `program-block ${step.block}`;
      stepEl.innerHTML = `
        <strong>${escapeHtml(step.title)}</strong>
        <span>${escapeHtml(step.instruction || step.id)}</span>
        <small>${escapeHtml(step.block)} • ${step.actions.length} actions</small>
      `;
      canvas.appendChild(stepEl);
    });
  });
}

function populateScriptSelect() {
  const select = document.getElementById("script-select");
  if (!select) {
    return;
  }
  select.innerHTML = Object.entries(pipelineScripts)
    .map(([key, script]) => `<option value="${escapeHtml(key)}">${escapeHtml(script.title)}</option>`)
    .join("");
  select.value = state.agent.activeScript;
}

function loadSelectedScript(forcePreset = false) {
  const select = document.getElementById("script-select");
  const editor = document.getElementById("pipeline-editor");
  if (!select || !editor) {
    return;
  }
  const key = select.value || "yichao";
  const localKey = `organoid-agent-script-${key}`;
  state.agent.activeScript = key;
  editor.value = !forcePreset && localStorage.getItem(localKey) ? localStorage.getItem(localKey) : pipelineScripts[key].text;
  parsePipeline();
}

function saveSelectedScript() {
  const editor = document.getElementById("pipeline-editor");
  if (!editor) {
    return;
  }
  const key = state.agent.activeScript || "yichao";
  localStorage.setItem(`organoid-agent-script-${key}`, editor.value);
  const status = document.getElementById("pipeline-status");
  if (status) {
    status.textContent = `Saved local edits for ${pipelineScripts[key].title}.`;
  }
}

function renderDatabases(databases) {
  const container = document.getElementById("database-registry");
  if (!container) {
    return;
  }
  if (!databases.length) {
    container.innerHTML = "<div class='muted'>No SQLite databases found.</div>";
    return;
  }
  container.innerHTML = databases
    .map((db) => {
      const tableRows = (db.tables || [])
        .map((table) => `<span class="db-table">${escapeHtml(table.table)}: ${escapeHtml(table.rows ?? "?")}</span>`)
        .join("");
      const summary = db.summary?.path ? `<div class="meta">summary: ${escapeHtml(db.summary.path)}</div>` : "";
      const error = db.error ? `<div class="meta danger">sqlite read error: ${escapeHtml(db.error)}</div>` : "";
      return `
        <div class="database-card">
          <div class="db-title">${escapeHtml(db.project)} · ${escapeHtml(db.name)}</div>
          <div class="meta">${escapeHtml(db.path)} · ${escapeHtml(db.size_human)}</div>
          <div class="db-tables">${tableRows || "<span class='db-table'>no tables read</span>"}</div>
          ${summary}
          ${error}
        </div>
      `;
    })
    .join("");
}

async function loadDatabaseRegistry() {
  const container = document.getElementById("database-registry");
  if (container) {
    container.textContent = "Loading databases...";
  }
  try {
    const data = await fetchJson("/api/agent/databases");
    renderDatabases(data.databases || []);
  } catch (err) {
    if (container) {
      container.textContent = `Database scan failed: ${err.message}`;
    }
  }
}

async function parsePipeline() {
  const editor = document.getElementById("pipeline-editor");
  const status = document.getElementById("pipeline-status");
  if (!editor || !status) {
    return null;
  }
  status.textContent = "Parsing pipeline...";
  try {
    const data = await postJson("/api/agent/pipeline/parse", { text: editor.value });
    renderPipelineBlocks(data.ir);
    const stepCount = data.ir.tasks.reduce((sum, task) => sum + task.steps.length, 0);
    const script = pipelineScripts[state.agent.activeScript] || pipelineScripts.generic;
    status.textContent = `Parsed ${data.ir.tasks.length} task(s), ${stepCount} step(s). Database hint: ${script.databaseHint}.`;
    return data.ir;
  } catch (err) {
    status.textContent = `Parse failed: ${err.message}`;
    return null;
  }
}

async function loadAgentState() {
  const grid = document.getElementById("agent-status-grid");
  if (!grid) {
    return;
  }
  try {
    const data = await fetchJson("/api/agent/state");
    const activeMorphology = (data.active_morphology_jobs || [])[0];
    if (activeMorphology) {
      state.agent.morphologyJobId = activeMorphology.id || activeMorphology.job_id;
      state.agent.morphologyStatus = activeMorphology.status;
      setStopJobEnabled(true);
      updateMorphologyProgressView(activeMorphology.progress || {}, activeMorphology.status);
    }
    const localMorphologyActive = state.agent.morphologyJobId && ["queued", "running", "pending"].includes(state.agent.morphologyStatus);
    const activeJobCount = Math.max(
      Number(data.active_job_count ?? (data.active_jobs || []).length),
      localMorphologyActive ? 1 : 0,
    );
    grid.innerHTML = `
      <div class="status-row"><span>Backend</span><strong>ok</strong></div>
      <div class="status-row"><span>Codex</span><strong>${data.codex_available ? "available" : "missing"}</strong></div>
      <div class="status-row"><span>Model</span><strong>${escapeHtml(data.default_model)}</strong></div>
      <div class="status-row"><span>Active Jobs</span><strong>${activeJobCount}</strong></div>
    `;
  } catch (err) {
    grid.innerHTML = `<div class="status-row"><span>Backend</span><strong>error</strong></div>`;
  }
}

async function ensureAgentSession() {
  if (state.agent.sessionId) {
    return state.agent.sessionId;
  }
  const data = await postJson("/api/agent/session", { title: "OrganoidAgent Studio" });
  state.agent.sessionId = data.session.id;
  return state.agent.sessionId;
}

function setAgentOutput(text) {
  const output = document.getElementById("agent-job-output");
  if (output) {
    output.textContent = text || "No output yet.";
  }
}

function setStopJobEnabled(enabled) {
  const button = document.getElementById("stop-agent-job");
  if (button) button.disabled = !enabled;
}

function morphologyProgressText(progress = {}, status = "pending") {
  const percent = Number(progress.percent || 0);
  const phase = progress.phase ? ` Phase: ${progress.phase}.` : "";
  const current = progress.current_file ? ` Current: ${progress.current_file}.` : "";
  const scale = progress.current_scale ? ` Cellpose scale ${progress.scale_index}/${progress.scale_total}: ${progress.current_scale}px.` : "";
  return `${progress.completed || 0}/${progress.total || 0} completed (${percent.toFixed(1)}%).${phase}${current}${scale} Elapsed: ${Number(progress.elapsed_seconds || 0).toFixed(0)} s. Status: ${status}.`;
}

function updateMorphologyProgressView(progress = {}, status = "pending") {
  const text = document.getElementById("morphology-progress");
  const bar = document.getElementById("morphology-progress-bar");
  if (bar) {
    bar.hidden = false;
    bar.value = Number(progress.percent || 0);
  }
  if (text) text.textContent = morphologyProgressText(progress, status);
}

async function pollAgentJob(jobId) {
  if (!jobId) {
    return;
  }
  try {
    const data = await fetchJson(`/api/agent/codex/job?id=${encodeURIComponent(jobId)}`);
    const job = data.job;
    const morphology = data.morphology_artifacts;
    const morphologyStatus = morphology?.status || "pending";
    const morphologyProgressData = morphology?.progress || {};
    const morphologyProgress = morphology?.total
      ? `\n\nMorphology ${morphology.managed_job_id || "job"}: ${morphologyProgressText(morphologyProgressData, morphologyStatus)}`
      : "";
    const body = (data.output_text || data.logs?.stderr_tail || data.logs?.stdout_tail || "Waiting for Codex output...") + morphologyProgress;
    setAgentOutput(`[${job.status}] ${job.id}\n\n${body}`);
    if (morphology?.managed_job_id) {
      state.agent.morphologyJobId = morphology.managed_job_id;
      state.agent.morphologyStatus = morphologyStatus;
      updateMorphologyProgressView(morphologyProgressData, morphologyStatus);
    }
    const codexTerminal = ["succeeded", "failed", "cancelled"].includes(job.status);
    const morphologyTerminal = ["succeeded", "failed", "cancelled"].includes(morphologyStatus);
    const waitingForMorphology = codexTerminal && morphology?.managed_job_id && !morphologyTerminal;
    if (waitingForMorphology) {
      setStopJobEnabled(true);
      setMorphologyRunning(true);
      loadAgentState();
      return;
    }
    if (codexTerminal) {
      clearInterval(state.agent.pollTimer);
      state.agent.pollTimer = null;
      setStopJobEnabled(false);
      setMorphologyRunning(false);
      loadAgentState();
      if (morphology?.rows?.length) {
        renderMorphologyResults(morphology);
      }
      if (!state.results.recordedJobIds.has(job.id)) {
        appendAgentMessage("assistant", data.output_text || `Job ${job.status}: ${job.id}`);
        state.results.recordedJobIds.add(job.id);
        state.results.records.unshift({
          id: `agent:${job.id}`,
          kind: "agent",
          title: `Morphology · ${job.id}`,
          status: morphology?.managed_job_id ? morphologyStatus : job.status,
          created_at: Date.now() / 1000,
          elapsed_seconds: job.elapsed_seconds,
          model: job.model,
          prompt_preview: job.prompt_preview || "",
          output_text: data.output_text || "",
          error: job.error,
          detail: job.detail,
          artifacts: morphology || {},
        });
        if (morphologyStatus === "succeeded" && morphology?.rows?.length) {
          setActiveStage("results", "viability-results");
        }
        renderSessionResultsSummary();
        renderResultsHistory();
      }
    }
  } catch (err) {
    setAgentOutput(`Poll failed: ${err.message}`);
  }
}

async function submitAgentJob(tool, prompt) {
  const pipelineText = document.getElementById("pipeline-editor")?.value || "";
  const sessionId = await ensureAgentSession();
  const data = await postJson("/api/agent/chat", {
    session_id: sessionId,
    message: prompt,
    tool,
    pipeline_text: pipelineText,
    allow_edits: tool === "assistant",
    workflow_id: state.agent.activeScript,
    selected_dataset: state.agent.selectedDataset,
    selected_files: Array.from(state.agent.selectedFiles),
  });
  state.agent.activeJobId = data.job.id;
  setStopJobEnabled(true);
  appendAgentMessage("user", prompt);
  appendAgentMessage("assistant", `Started ${tool} job ${data.job.id}.`);
  setAgentOutput(`[queued] ${data.job.id}`);
  if (state.agent.pollTimer) {
    clearInterval(state.agent.pollTimer);
  }
  state.agent.pollTimer = setInterval(() => pollAgentJob(data.job.id), 2500);
  pollAgentJob(data.job.id);
}

function renderAgentFileSelection() {
  const container = document.getElementById("agent-file-selection");
  const count = document.getElementById("agent-selection-count");
  if (!container || !count) return;
  if (!state.agent.selectedDataset) {
    container.innerHTML = '<span class="muted">Select a dataset to analyze the whole dataset or choose specific images.</span>';
    count.textContent = "No dataset selected";
    return;
  }
  count.textContent = state.agent.selectedFiles.size
    ? `${state.agent.selectedFiles.size} image(s) selected`
    : "Whole dataset selected";
  if (!state.agent.availableFiles.length) {
    container.innerHTML = '<span class="muted">No supported images found. The dataset directory remains selected.</span>';
    return;
  }
  container.innerHTML = state.agent.availableFiles.map((file) => `
    <label class="agent-file-option">
      <input type="checkbox" value="${escapeHtml(file.path)}" ${state.agent.selectedFiles.has(file.path) ? "checked" : ""} />
      <span>${escapeHtml(file.path)}</span>
      <small>${escapeHtml(file.size_human || "")}</small>
    </label>
  `).join("");
  container.querySelectorAll('input[type="checkbox"]').forEach((checkbox) => {
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) state.agent.selectedFiles.add(checkbox.value);
      else state.agent.selectedFiles.delete(checkbox.value);
      renderAgentFileSelection();
    });
  });
}

async function loadAgentDatasetFiles(dataset) {
  state.agent.selectedDataset = dataset;
  state.agent.selectedFiles.clear();
  state.agent.availableFiles = [];
  renderAgentFileSelection();
  if (!dataset) return;
  const data = await fetchJson(`/api/fluorescence/images?dataset=${encodeURIComponent(dataset)}`);
  state.agent.availableFiles = data.images || [];
  renderAgentFileSelection();
}

function initAgentDatasetSelector() {
  const select = document.getElementById("agent-dataset-select");
  if (!select) return;
  select.innerHTML = '<option value="">Select a dataset</option>' + state.datasets.map(
    (dataset) => `<option value="${escapeHtml(dataset.path)}">${escapeHtml(dataset.name)}</option>`
  ).join("");
  select.addEventListener("change", () => loadAgentDatasetFiles(select.value).catch((err) => {
    document.getElementById("agent-file-selection").textContent = err.message;
  }));
  document.getElementById("agent-select-all-files")?.addEventListener("click", () => {
    state.agent.selectedFiles = new Set(state.agent.availableFiles.map((file) => file.path));
    renderAgentFileSelection();
  });
  document.getElementById("agent-clear-files")?.addEventListener("click", () => {
    state.agent.selectedFiles.clear();
    renderAgentFileSelection();
  });
  renderAgentFileSelection();
}

function setMorphologyRunning(running) {
  const run = document.getElementById("morphology-run");
  const stop = document.getElementById("morphology-stop");
  if (run) run.disabled = running;
  if (stop) stop.disabled = !running;
}

function renderMorphologyResults(data) {
  if (!data) return;
  const card = document.getElementById("morphology-results-card");
  const summary = document.getElementById("morphology-summary");
  const results = document.getElementById("morphology-results");
  const report = document.getElementById("morphology-report");
  card.hidden = false;
  document.getElementById("morphology-result-job-id").textContent = data.job_id || "";
  const downloads = data.download_urls || {};
  summary.innerHTML = `
    <div class="status-grid">
      <div class="status-row"><span>Images</span><strong>${(data.rows || []).length}</strong></div>
      <div class="status-row"><span>GPU</span><strong>${data.used_gpu ? "CUDA" : "CPU"}</strong></div>
      <div class="status-row"><span>Workflow</span><strong>${escapeHtml(data.workflow || "")}</strong></div>
      <div class="status-row"><span>Downloads</span><strong>${downloads.csv ? `<a href="${escapeHtml(downloads.csv)}" target="_blank">CSV</a>` : ""} ${downloads.json ? `· <a href="${escapeHtml(downloads.json)}" target="_blank">JSON</a>` : ""} ${downloads.report ? `· <a href="${escapeHtml(downloads.report)}" target="_blank">Report</a>` : ""}</strong></div>
    </div>
    ${(downloads.gallery || downloads.summary_figure) ? `<div class="fluorescence-images">${downloads.gallery ? `<figure><img src="${escapeHtml(downloads.gallery)}" alt="Morphology comparison" /><figcaption>Comparison gallery</figcaption></figure>` : ""}${downloads.summary_figure ? `<figure><img src="${escapeHtml(downloads.summary_figure)}" alt="Morphology metrics" /><figcaption>Summary metrics</figcaption></figure>` : ""}</div>` : ""}`;
  results.innerHTML = (data.rows || []).map((row) => {
    const assets = row.asset_urls || {};
    return `<article class="fluorescence-result">
      <div class="fluorescence-result-head"><div><strong>${escapeHtml(row.condition)} · D${String(row.day).padStart(2, "0")}</strong><div class="muted">${escapeHtml(row.image_name || "")}</div></div><div class="viability-score">${Number(row.count || 0)} objects</div></div>
      <div class="fluorescence-images">
        ${assets.overlay ? `<figure><img src="${escapeHtml(assets.overlay)}" alt="Selected best overlay" /><figcaption>Selected best segmentation</figcaption></figure>` : ""}
        ${assets.mask ? `<figure><img src="${escapeHtml(assets.mask)}" alt="Selected best mask" /><figcaption>Selected best instance mask</figcaption></figure>` : ""}
      </div>
      <div class="fluorescence-metrics">
        <div class="fluorescence-metric"><span>Total area</span><strong>${Number(row.total_area_px || 0).toLocaleString()}</strong></div>
        <div class="fluorescence-metric"><span>Roundness</span><strong>${Number(row.roundness || 0).toFixed(3)}</strong></div>
        <div class="fluorescence-metric"><span>Largest mass fraction</span><strong>${Number(row.fused_mass_fraction || 0).toFixed(3)}</strong></div>
        <div class="fluorescence-metric"><span>Mean diameter</span><strong>${Number(row.mean_equivalent_diameter_px || 0).toFixed(1)} px</strong></div>
        <div class="fluorescence-metric"><span>Stage</span><strong>${escapeHtml(row.stage || "")}</strong></div>
        <div class="fluorescence-metric"><span>Selection</span><strong>${escapeHtml(row.segmentation_method || "legacy")}</strong></div>
        <div class="fluorescence-metric"><span>Quality score</span><strong>${Number(row.segmentation_quality_score || 0).toFixed(3)}</strong></div>
        <div class="fluorescence-metric"><span>Decision</span><strong>${escapeHtml(row.segmentation_selection_reason || "not recorded")}</strong></div>
      </div>
    </article>`;
  }).join("");
  report.innerHTML = renderMarkdown(data.report_text || "");
}

async function pollMorphologyJob() {
  const jobId = state.agent.morphologyJobId;
  if (!jobId) return;
  const text = document.getElementById("morphology-progress");
  const bar = document.getElementById("morphology-progress-bar");
  try {
    const data = await fetchJson(`/api/morphology/status?id=${encodeURIComponent(jobId)}`);
    const job = data.job || {};
    const progress = data.progress || {};
    state.agent.morphologyStatus = job.status || null;
    if (!progress.total && job.inputs?.length) progress.total = job.inputs.length;
    updateMorphologyProgressView(progress, job.status);
    if (["succeeded", "failed", "cancelled"].includes(job.status)) {
      clearInterval(state.agent.morphologyPollTimer);
      state.agent.morphologyPollTimer = null;
      setMorphologyRunning(false);
      setStopJobEnabled(false);
      loadAgentState();
      if (text && job.status !== "succeeded") text.textContent += ` ${job.error || data.stderr_tail || ""}`;
      if (job.status === "succeeded" && data.result) {
        renderMorphologyResults(data.result);
        setActiveStage("results", "viability-results");
      }
      const recordId = `morphology:${jobId}`;
      if (!state.results.records.some((item) => item.id === recordId)) {
        state.results.records.unshift({id: recordId, kind: "agent", title: `Morphology · ${jobId}`, status: job.status, created_at: job.created_at, elapsed_seconds: progress.elapsed_seconds, output_text: data.stdout_tail || data.stderr_tail || "", artifacts: {output_dir: job.output_dir}});
        renderSessionResultsSummary();
        renderResultsHistory();
      }
    }
  } catch (err) {
    if (text) text.textContent = `Morphology status failed: ${err.message}`;
  }
}

async function runMorphologyDirect() {
  const text = document.getElementById("morphology-progress");
  const bar = document.getElementById("morphology-progress-bar");
  const instruction = document.getElementById("workflow-analysis-instruction")?.value.trim() || "Analyze the selected images.";
  if (state.agent.activeScript === "fluorescence") {
    state.fluorescence.selected = new Set(state.agent.selectedFiles);
    const target = document.getElementById("fluorescence-instruction");
    if (target) target.value = instruction;
    updateFluorescenceSelectionCount();
    setActiveStage("workflow", "fluorescence");
    runFluorescencePrediction();
    return;
  }
  if (!["zhengyu", "y27632", "sodium_alginate"].includes(state.agent.activeScript)) {
    if (text) text.textContent = `Delegated ${pipelineScripts[state.agent.activeScript]?.title || "selected workflow"} to the Agent with the selected inputs and instruction.`;
    submitAgentJob("assistant", instruction);
    return;
  }
  if (!state.agent.selectedFiles.size) {
    if (text) text.textContent = "Select one or more TIFF images. Whole-dataset execution is disabled for this long-running workflow.";
    return;
  }
  setMorphologyRunning(true);
  if (bar) { bar.hidden = false; bar.value = 0; }
  if (text) text.textContent = "Submitting morphology job...";
  try {
    const data = await postJson("/api/morphology/run", {paths: Array.from(state.agent.selectedFiles), overwrite: false, workflow_id: state.agent.activeScript, instruction});
    state.agent.morphologyJobId = data.job_id;
    document.getElementById("morphology-job-id").textContent = data.job_id;
    state.agent.morphologyPollTimer = setInterval(pollMorphologyJob, 1500);
    pollMorphologyJob();
  } catch (err) {
    setMorphologyRunning(false);
    if (text) text.textContent = `Morphology submission failed: ${err.message}`;
  }
}

async function stopMorphologyDirect() {
  if (!state.agent.morphologyJobId) return;
  const text = document.getElementById("morphology-progress");
  try {
    await postJson("/api/morphology/cancel", {job_id: state.agent.morphologyJobId});
    if (text) text.textContent = "Stopping morphology job...";
    pollMorphologyJob();
  } catch (err) {
    if (text) text.textContent = `Stop failed: ${err.message}`;
  }
}

function insertPipelineTemplate(kind) {
  const editor = document.getElementById("pipeline-editor");
  if (!editor) {
    return;
  }
  const snippets = {
    inspect: 'STEP {"id":"inspect_next","block":"plan","title":"Inspect inputs","instruction":"Identify source folders, channels, time/depth design, references, scripts, and output requirements."}\nACTION {"type":"read","target":"references"}',
    segmentation: 'STEP {"id":"segment_next","block":"work","title":"Segment brightfield","instruction":"Run brightfield segmentation and save instance masks, overlays, and crops."}\nACTION {"type":"script","target":"analysis-tools/yichao_instance_pairs"}',
    tracking: 'STEP {"id":"tracking_next","block":"work","title":"Track positions over time","instruction":"Link objects across day/time/position folders and record monitoring metadata."}\nACTION {"type":"database","target":"analysis-outputs"}',
    pairing: 'STEP {"id":"pair_next","block":"work","title":"Create paired crops","instruction":"Match brightfield c1 with fluorescence c0 and write paired dataset records."}\nACTION {"type":"dataset","target":"analysis-outputs/yichao_instance_pairs"}',
    quantification: 'STEP {"id":"quantify_next","block":"work","title":"Quantify instances","instruction":"Measure instance size, padding, edge contact, fluorescence intensity, and debris flags."}',
    fluorescence: 'STEP {"id":"fluorescence_next","block":"work","title":"Detect organoid viability","instruction":"Run scripts/run_fluorescence_prediction.py on exact image paths under datasets and save overlays, masks, crops, scores, morphology evidence, CSV, JSON, and report."}\nACTION {"type":"script","target":"scripts/run_fluorescence_prediction.py"}',
    database: 'STEP {"id":"database_next","block":"work","title":"Maintain database","instruction":"Create or update SQLite tables, summary manifests, schema notes, source paths, and QC filter fields."}\nACTION {"type":"database","target":"analysis-outputs"}',
    review: 'STEP {"id":"review_next","block":"debug","title":"Human review gate","instruction":"Sample random pairs and identify debris, edge-padded crops, and wrong channel assignments."}',
    report: 'STEP {"id":"report_next","block":"summary","title":"Write report","instruction":"Summarize outputs, database paths, histograms, and training recommendations."}',
  };
  editor.value = `${editor.value.trim()}\n${snippets[kind] || ""}\n`;
  parsePipeline();
}

function initAgentStudio() {
  const editor = document.getElementById("pipeline-editor");
  if (!editor) {
    return;
  }
  populateScriptSelect();
  loadSelectedScript();
  parsePipeline();
  loadAgentState();
  loadDatabaseRegistry();
  document.getElementById("script-select")?.addEventListener("change", () => loadSelectedScript());
  document.getElementById("load-script-btn")?.addEventListener("click", () => loadSelectedScript(true));
  document.getElementById("save-script-btn")?.addEventListener("click", saveSelectedScript);
  document.getElementById("parse-pipeline-btn")?.addEventListener("click", parsePipeline);
  document.getElementById("refresh-agent-state")?.addEventListener("click", loadAgentState);
  document.getElementById("stop-agent-job")?.addEventListener("click", async () => {
    const morphologyIsActive = state.agent.morphologyJobId && ["queued", "running", "pending"].includes(state.agent.morphologyStatus);
    if (!morphologyIsActive && !state.agent.activeJobId) return;
    const button = document.getElementById("stop-agent-job");
    button.disabled = true;
    button.textContent = "Stopping...";
    try {
      if (morphologyIsActive) {
        const data = await postJson("/api/morphology/cancel", { job_id: state.agent.morphologyJobId });
        state.agent.morphologyStatus = data.job.status;
        setAgentOutput(`[${data.job.status}] ${data.job.id || state.agent.morphologyJobId}\n\nMorphology job stopped by user.`);
        pollAgentJob(state.agent.activeJobId);
        loadAgentState();
        return;
      }
      const data = await postJson("/api/agent/codex/cancel", { job_id: state.agent.activeJobId });
      setAgentOutput(`[${data.job.status}] ${data.job.id}\n\nJob stopped by user.`);
      pollAgentJob(state.agent.activeJobId);
    } catch (err) {
      setAgentOutput(`Stop failed: ${err.message}`);
      setStopJobEnabled(true);
    } finally {
      button.textContent = "Stop Job";
    }
  });
  document.getElementById("refresh-database-registry")?.addEventListener("click", loadDatabaseRegistry);
  document.getElementById("new-agent-chat")?.addEventListener("click", async () => {
    const data = await postJson("/api/agent/session", { title: "OrganoidAgent Studio" });
    state.agent.sessionId = data.session.id;
    document.getElementById("agent-messages").innerHTML = "";
    setAgentOutput(`New session ${data.session.id}`);
  });
  document.getElementById("send-plan-btn")?.addEventListener("click", () => {
    submitAgentJob("response", "Review this AAPS pipeline and suggest the next concrete OrganoidAgent implementation step.");
  });
  document.getElementById("run-assistant-btn")?.addEventListener("click", () => {
    submitAgentJob("assistant", "Use this AAPS pipeline as the current plan and make the next safe implementation change in this repository.");
  });
  document.getElementById("send-agent-chat")?.addEventListener("click", () => {
    const input = document.getElementById("agent-chat-input");
    const message = input.value.trim();
    if (!message) {
      return;
    }
    const tool = document.getElementById("assistant-mode-toggle")?.checked ? "assistant" : "response";
    input.value = "";
    submitAgentJob(tool, message);
  });
  document.querySelectorAll("[data-template]").forEach((button) => {
    button.addEventListener("click", () => insertPipelineTemplate(button.dataset.template));
  });
}

async function loadDatasetMetadata(dataset) {
  const container = document.getElementById("dataset-info");
  if (!container) {
    return;
  }
  container.textContent = "Loading metadata…";
  try {
    const data = await fetchJson(`/api/datasets/${dataset}/metadata`);
    if (!data.markdown) {
      container.textContent = "No metadata available.";
      return;
    }
    container.innerHTML = renderMarkdown(data.markdown);
  } catch (err) {
    container.textContent = "No metadata available.";
  }
}

async function loadDatasets() {
  const data = await fetchJson("/api/datasets");
  state.datasets = data.datasets;
  const totalSize = data.datasets
    .reduce((acc, ds) => acc + ds.size_bytes, 0);
  document.getElementById(
    "dataset-stats"
  ).textContent = `${data.datasets.length} datasets • ${(
    totalSize /
    (1024 * 1024 * 1024)
  ).toFixed(2)} GB`;

  renderList("dataset-list", data.datasets, (item) => {
    state.activeDataset = item.path;
    loadDatasetFiles(item.path);
  });

  if (data.datasets.length) {
    state.activeDataset = data.datasets[0].path;
    loadDatasetFiles(state.activeDataset);
  }
}

async function loadDatasetFiles(dataset) {
  const data = await fetchJson(`/api/datasets/${dataset}`);
  state.files = data.files;
  renderList("file-list", data.files, async (file) => {
    const preview = await fetchJson(`/api/preview?path=${file.path}`);
    renderPreview("preview-panel", preview);
  });
  loadDatasetMetadata(dataset);
}

async function loadCategory(category, listId, previewId) {
  const data = await fetchJson(`/api/category/${category}`);
  renderList(listId, data.files, async (file) => {
    const preview = await fetchJson(`/api/preview?path=${file.path}`);
    renderPreview(previewId, preview);
  });
}

function renderResultsHistory() {
  const list = document.getElementById("results-history-list");
  if (!list) return;
  if (!state.results.records.length) {
    list.innerHTML = '<div class="muted">No tasks have been run on this page yet.</div>';
    return;
  }
  list.innerHTML = state.results.records.map((record) => {
    const active = record.id === state.results.selectedId ? " active" : "";
    const date = record.created_at ? new Date(record.created_at * 1000).toLocaleString() : "Unknown time";
    const detail = record.kind === "viability"
      ? `${record.sample_count} sample(s)`
      : `${escapeHtml(record.model || "Codex")} · ${Number(record.elapsed_seconds || 0).toFixed(1)} s`;
    return `
      <button class="result-history-item${active}" data-result-id="${escapeHtml(record.id)}">
        <span><strong>${escapeHtml(record.title)}</strong><small>${escapeHtml(date)}</small></span>
        <span><span class="result-status ${escapeHtml(record.status)}">${escapeHtml(record.status)}</span><small>${detail}</small></span>
      </button>`;
  }).join("");
  list.querySelectorAll("[data-result-id]").forEach((button) => {
    button.addEventListener("click", () => openResultRecord(button.dataset.resultId));
  });
}

async function openResultRecord(recordId) {
  const record = state.results.records.find((item) => item.id === recordId);
  if (!record) return;
  state.results.selectedId = recordId;
  renderResultsHistory();
  const detail = document.getElementById("results-history-detail");
  if (record.kind === "viability") {
    const payload = record.payload;
    detail.innerHTML = `
      <h3>${escapeHtml(record.title)}</h3>
      <p><strong>Status:</strong> ${escapeHtml(record.status)} · <strong>Samples:</strong> ${record.sample_count}</p>
      <p><strong>Run:</strong> ${escapeHtml(payload.job_id || record.id)}</p>
      <p><a href="${escapeHtml(payload.results_url)}" target="_blank">JSON</a>${payload.csv_url ? ` · <a href="${escapeHtml(payload.csv_url)}" target="_blank">CSV</a>` : ""}${payload.report_url ? ` · <a href="${escapeHtml(payload.report_url)}" target="_blank">Report</a>` : ""}</p>`;
    renderFluorescenceResults(payload);
    document.getElementById("fluorescence-progress").textContent = `Loaded saved run ${record.title}.`;
    if (payload.report_url) {
      const response = await fetch(payload.report_url);
      document.getElementById("fluorescence-report").innerHTML = response.ok ? renderMarkdown(await response.text()) : "";
    }
    return;
  }
  if (record.artifacts?.rows?.length) {
    const morphology = record.artifacts;
    const downloads = morphology.download_urls || {};
    detail.innerHTML = `
      <h3>${escapeHtml(record.title)}</h3>
      <p><strong>Status:</strong> ${escapeHtml(record.status)} · <strong>Morphology outputs:</strong> ${morphology.completed || morphology.rows.length}/${morphology.total || morphology.rows.length}</p>
      <p>${downloads.json ? `<a href="${escapeHtml(downloads.json)}" target="_blank">JSON</a>` : ""}${downloads.csv ? ` · <a href="${escapeHtml(downloads.csv)}" target="_blank">CSV</a>` : ""}${downloads.report ? ` · <a href="${escapeHtml(downloads.report)}" target="_blank">Report</a>` : ""}</p>
      <pre class="result-output-text">${escapeHtml(record.output_text || "")}</pre>`;
    renderMorphologyResults(morphology);
    return;
  }
  const links = Object.entries(record.artifacts || {}).filter(([, url]) => url).map(
    ([name, url]) => `<a href="${escapeHtml(url)}" target="_blank">${escapeHtml(name)}</a>`
  ).join(" · ");
  detail.innerHTML = `
    <h3>${escapeHtml(record.title)}</h3>
    <p><strong>Status:</strong> ${escapeHtml(record.status)} · <strong>Model:</strong> ${escapeHtml(record.model || "")}</p>
    <p><strong>Prompt:</strong> ${escapeHtml(record.prompt_preview || "")}</p>
    ${record.error ? `<p class="error"><strong>Error:</strong> ${escapeHtml(record.error)} ${escapeHtml(record.detail || "")}</p>` : ""}
    <p>${links}</p>
    <pre class="result-output-text">${escapeHtml(record.output_text || "No output text was saved.")}</pre>`;
}

function renderSessionResultsSummary() {
  const summary = {
    total: state.results.records.length,
    viability: state.results.records.filter((item) => item.kind === "viability").length,
    agent: state.results.records.filter((item) => item.kind === "agent").length,
    succeeded: state.results.records.filter((item) => item.status === "succeeded").length,
    failed: state.results.records.filter((item) => item.status === "failed").length,
  };
  document.getElementById("results-history-summary").innerHTML = `
    <div><span>All runs</span><strong>${summary.total || 0}</strong></div>
    <div><span>Viability</span><strong>${summary.viability || 0}</strong></div>
    <div><span>Morphology jobs</span><strong>${summary.agent || 0}</strong></div>
    <div><span>Succeeded / Failed</span><strong>${summary.succeeded || 0} / ${summary.failed || 0}</strong></div>`;
}

function updateFluorescenceSelectionCount() {
  const count = state.fluorescence.selected.size;
  const target = document.getElementById("fluorescence-selection-count");
  if (target) {
    target.textContent = `${count} selected`;
  }
}

function renderFluorescenceImages() {
  const container = document.getElementById("fluorescence-image-list");
  if (!container) {
    return;
  }
  if (!state.fluorescence.images.length) {
    container.innerHTML = "<div class='muted' style='padding:12px'>No supported images found.</div>";
    updateFluorescenceSelectionCount();
    return;
  }
  container.innerHTML = state.fluorescence.images
    .map((item) => `
      <label class="fluorescence-image-option">
        <input type="checkbox" data-fluorescence-path="${escapeHtml(item.path)}" ${state.fluorescence.selected.has(item.path) ? "checked" : ""} />
        <span>${escapeHtml(item.path)}</span>
        <small class="muted">${escapeHtml(item.size_human)}</small>
      </label>
    `)
    .join("");
  container.querySelectorAll("[data-fluorescence-path]").forEach((checkbox) => {
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) {
        if (state.fluorescence.selected.size >= 30) {
          checkbox.checked = false;
          return;
        }
        state.fluorescence.selected.add(checkbox.dataset.fluorescencePath);
      } else {
        state.fluorescence.selected.delete(checkbox.dataset.fluorescencePath);
      }
      updateFluorescenceSelectionCount();
    });
  });
  updateFluorescenceSelectionCount();
}

async function loadFluorescenceImages(dataset) {
  const container = document.getElementById("fluorescence-image-list");
  container.textContent = "Loading images...";
  state.fluorescence.selected.clear();
  const data = await fetchJson(`/api/fluorescence/images?dataset=${encodeURIComponent(dataset)}`);
  state.fluorescence.images = data.images || [];
  renderFluorescenceImages();
}

function renderFluorescenceResults(data) {
  const summary = document.getElementById("fluorescence-summary");
  const results = document.getElementById("fluorescence-results");
  const job = document.getElementById("fluorescence-job-id");
  job.textContent = data.job_id || "";
  summary.innerHTML = `
    <div class="status-grid">
      <div class="status-row"><span>Successful</span><strong>${data.summary.succeeded}/${data.summary.total}</strong></div>
      <div class="status-row"><span>High / Medium / Low</span><strong>${data.summary.high} / ${data.summary.medium} / ${data.summary.low}</strong></div>
      <div class="status-row"><span>Elapsed</span><strong>${Number(data.summary.elapsed_seconds).toFixed(1)} s</strong></div>
      <div class="status-row"><span>Downloads</span><strong><a href="${escapeHtml(data.csv_url)}" target="_blank">CSV</a> · <a href="${escapeHtml(data.results_url)}" target="_blank">JSON</a></strong></div>
    </div>
  `;
  results.innerHTML = (data.samples || []).map((sample, index) => {
    const e = sample.evidence;
    return `
      <article class="fluorescence-result">
        <div class="fluorescence-result-head">
          <div><strong>#${index + 1} ${escapeHtml(sample.filename)}</strong><div class="muted">${escapeHtml(sample.level)} · YOLO ${sample.segmentation.yolo_confidence} · SAM ${sample.segmentation.sam_score}</div></div>
          <div class="viability-score">${Number(sample.viability_percent).toFixed(1)}%</div>
        </div>
        <div class="fluorescence-images">
          <figure><img src="${escapeHtml(sample.asset_urls.overlay)}" alt="Segmentation overlay" /><figcaption>YOLO + SAM overlay</figcaption></figure>
          <figure><img src="${escapeHtml(sample.asset_urls.crop)}" alt="Prediction crop" /><figcaption>Model input crop</figcaption></figure>
        </div>
        <div class="fluorescence-metrics">
          <div class="fluorescence-metric"><span>Circularity</span><strong>${Number(e.circularity).toFixed(3)}</strong></div>
          <div class="fluorescence-metric"><span>Area fraction</span><strong>${Number(e.area_fraction).toFixed(3)}</strong></div>
          <div class="fluorescence-metric"><span>Bounding-box fill</span><strong>${Number(e.bbox_fill_ratio).toFixed(3)}</strong></div>
          <div class="fluorescence-metric"><span>Internal contrast</span><strong>${Number(e.contrast).toFixed(3)}</strong></div>
          <div class="fluorescence-metric"><span>Edge density</span><strong>${Number(e.edge_density).toFixed(3)}</strong></div>
          <div class="fluorescence-metric"><span>Mean brightness</span><strong>${Number(e.mean_brightness).toFixed(3)}</strong></div>
          <div class="fluorescence-metric"><span>Aspect ratio</span><strong>${Number(e.aspect_ratio).toFixed(3)}</strong></div>
          <div class="fluorescence-metric"><span>Processing time</span><strong>${Number(sample.elapsed_seconds).toFixed(1)} s</strong></div>
        </div>
        <p>${escapeHtml(sample.explanation)}</p>
        <a href="${escapeHtml(sample.asset_urls.mask)}" target="_blank">Open mask</a>
      </article>
    `;
  }).join("");
}

async function runFluorescencePrediction() {
  if (state.fluorescence.running || !state.fluorescence.selected.size) {
    return;
  }
  const button = document.getElementById("fluorescence-run");
  const progress = document.getElementById("fluorescence-progress");
  const report = document.getElementById("fluorescence-report");
  state.fluorescence.running = true;
  button.disabled = true;
  button.textContent = "Running...";
  progress.textContent = `Running segmentation and viability prediction for ${state.fluorescence.selected.size} image(s). Do not submit another GPU job.`;
  const progressBar = document.getElementById("fluorescence-progress-bar");
  if (progressBar) {
    progressBar.hidden = false;
    progressBar.value = 0;
  }
  report.innerHTML = "";
  try {
    state.fluorescence.progressTimer = setInterval(async () => {
      try {
        const status = await fetchJson("/api/fluorescence/status");
        const current = status.progress;
        if (!current) return;
        if (progressBar) progressBar.value = Number(current.percent || 0);
        const currentFile = current.current_file ? ` Current: ${current.current_file}.` : "";
        progress.textContent = `${current.completed}/${current.total} completed (${Number(current.percent || 0).toFixed(1)}%); ${current.succeeded} succeeded, ${current.failed} failed.${currentFile} Elapsed: ${Number(current.elapsed_seconds || 0).toFixed(0)} s.`;
      } catch (_) {
        // Keep the last visible progress while a poll is temporarily unavailable.
      }
    }, 1000);
    const data = await postJson("/api/fluorescence/run", {
      paths: Array.from(state.fluorescence.selected),
      order: document.getElementById("fluorescence-order").value,
      instruction: document.getElementById("fluorescence-instruction").value,
    });
    renderFluorescenceResults(data);
    if (progressBar) progressBar.value = 100;
    progress.textContent = `Completed ${data.summary.succeeded} image(s); ${data.summary.failed} failed.`;
    const reportResponse = await fetch(data.report_url);
    report.innerHTML = reportResponse.ok ? renderMarkdown(await reportResponse.text()) : "";
    const recordId = `viability:${data.job_id}`;
    if (!state.results.records.some((item) => item.id === recordId)) {
      state.results.records.unshift({
        id: recordId,
        kind: "viability",
        title: `Viability · ${data.job_id}`,
        status: data.status || "succeeded",
        created_at: Date.now() / 1000,
        sample_count: data.samples?.length || 0,
        payload: data,
      });
      state.results.selectedId = recordId;
      renderSessionResultsSummary();
      renderResultsHistory();
    }
    setActiveStage("results", "viability-results");
  } catch (err) {
    progress.textContent = `Prediction failed: ${err.message}`;
  } finally {
    if (state.fluorescence.progressTimer) {
      clearInterval(state.fluorescence.progressTimer);
      state.fluorescence.progressTimer = null;
    }
    state.fluorescence.running = false;
    button.disabled = false;
    button.textContent = "Run prediction";
  }
}

async function initFluorescence() {
  const select = document.getElementById("fluorescence-dataset");
  if (!select) {
    return;
  }
  const statusEl = document.getElementById("fluorescence-status");
  select.innerHTML = state.datasets.map((dataset) => `<option value="${escapeHtml(dataset.path)}">${escapeHtml(dataset.name)}</option>`).join("");
  try {
    const status = await fetchJson("/api/fluorescence/status");
    statusEl.textContent = status.ready ? "ready" : "not ready";
  } catch (err) {
    statusEl.textContent = "status unavailable";
    statusEl.title = err.message;
  }
  select.addEventListener("change", () => loadFluorescenceImages(select.value));
  document.getElementById("fluorescence-select-all")?.addEventListener("click", () => {
    state.fluorescence.selected = new Set(state.fluorescence.images.slice(0, 30).map((item) => item.path));
    renderFluorescenceImages();
  });
  document.getElementById("fluorescence-clear")?.addEventListener("click", () => {
    state.fluorescence.selected.clear();
    renderFluorescenceImages();
  });
  document.getElementById("fluorescence-run")?.addEventListener("click", runFluorescencePrediction);
  if (state.datasets.length) {
    const demo = state.datasets.find((dataset) => dataset.path === "05_Fluorescence_demo");
    select.value = (demo || state.datasets[0]).path;
    await loadFluorescenceImages(select.value);
  }
}

document.querySelectorAll(".stage-tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    setActiveStage(tab.dataset.stage);
  });
});

document.querySelectorAll(".subtab").forEach((tab) => {
  tab.addEventListener("click", () => {
    setActiveTab(tab.dataset.tab);
  });
});

document.getElementById("clear-session-results")?.addEventListener("click", () => {
  state.results.records = [];
  state.results.selectedId = null;
  state.results.recordedJobIds.clear();
  renderSessionResultsSummary();
  renderResultsHistory();
  document.getElementById("results-history-detail").textContent = "Select a run to inspect its outputs.";
});

loadDatasets().then(() => {
  initFluorescence();
  initAgentDatasetSelector();
}).catch((err) => {
  document.getElementById("dataset-stats").textContent = err.message;
});

loadCategory("segmentation", "segmentation-list", "preview-panel").catch(() => {});
loadCategory("features", "features-list", "features-preview").catch(() => {});
loadCategory("analysis", "analysis-list", "analysis-preview").catch(() => {});
initAgentStudio();
renderSessionResultsSummary();
renderResultsHistory();

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("/static/sw.js", { scope: "/" });
}
