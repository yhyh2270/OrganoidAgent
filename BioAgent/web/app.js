const apiBase = window.BIOAGENT_API_BASE || (window.location.port === "8090" ? "" : "http://localhost:8090");

const eyesSelect = document.getElementById("eyes");
const mindSelect = document.getElementById("mind");
const handSelect = document.getElementById("hand");
const taskSelect = document.getElementById("task");
const promptField = document.getElementById("prompt");
const notesField = document.getElementById("notes");
const fileInput = document.getElementById("task-files");
const folderInput = document.getElementById("task-folder");
const fileLabel = document.getElementById("task-files-label");
const folderLabel = document.getElementById("task-folder-label");
const filesDropzone = document.getElementById("files-dropzone");
const folderDropzone = document.getElementById("folder-dropzone");
const browseFilesButton = document.getElementById("browse-files");
const browseFolderButton = document.getElementById("browse-folder");
const statusIndicator = document.getElementById("status-indicator");
const jobCount = document.getElementById("job-count");
const jobsContainer = document.getElementById("jobs");

const templateList = document.getElementById("template-list");
const templateLabel = document.getElementById("template-label");
const templateBody = document.getElementById("template-body");
const templateSave = document.getElementById("template-save");
const templateNew = document.getElementById("template-new");
const templateDelete = document.getElementById("template-delete");
const templateColumn = document.getElementById("template-column");
const taskColumn = document.getElementById("task-column");

const chatHistory = document.getElementById("chat-history");
const chatInput = document.getElementById("chat-input");
const chatSend = document.getElementById("chat-send");
const bench = document.getElementById("bench");

let templates = [];
let activeTemplateId = null;
let selectedFiles = [];
let selectedFolderFiles = [];
let baseTemplateListHeight = null;

function setStatus(text, state) {
  statusIndicator.textContent = text;
  statusIndicator.classList.remove("neutral", "good", "bad");
  statusIndicator.classList.add(state || "neutral");
}

function fillSelect(select, items, defaultId) {
  select.innerHTML = "";
  items.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = item.label;
    select.appendChild(option);
  });
  if (defaultId) {
    select.value = defaultId;
  }
}

function slugify(value) {
  return value
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)+/g, "");
}

function storeTemplates() {
  localStorage.setItem("bioagent_templates", JSON.stringify(templates));
}

function loadStoredTemplates() {
  const raw = localStorage.getItem("bioagent_templates");
  if (!raw) {
    return null;
  }
  try {
    return JSON.parse(raw);
  } catch (error) {
    return null;
  }
}

function renderTemplateList() {
  templateList.innerHTML = "";
  if (!templates.length) {
    templateList.innerHTML = "<p>No templates yet.</p>";
    return;
  }

  templates.forEach((template) => {
    const item = document.createElement("div");
    item.className = "template-item";
    if (template.id === activeTemplateId) {
      item.classList.add("active");
    }

    const title = document.createElement("h4");
    title.textContent = template.label;
    item.appendChild(title);

    const preview = document.createElement("p");
    preview.textContent = template.prompt.slice(0, 80);
    item.appendChild(preview);

    item.addEventListener("click", () => {
      selectTemplate(template.id);
    });

    templateList.appendChild(item);
  });

  requestAnimationFrame(syncTemplateListHeight);
}

function renderTemplateEditor(template) {
  if (!template) {
    templateLabel.value = "";
    templateBody.value = "";
    return;
  }
  templateLabel.value = template.label || "";
  templateBody.value = template.prompt || "";
  requestAnimationFrame(syncTemplateListHeight);
}

function refreshTaskSelect() {
  fillSelect(taskSelect, templates, activeTemplateId || (templates[0] && templates[0].id));
  const selected = templates.find((item) => item.id === taskSelect.value);
  if (selected) {
    promptField.value = selected.prompt;
  }
  requestAnimationFrame(syncTemplateListHeight);
}

function selectTemplate(templateId) {
  activeTemplateId = templateId;
  const template = templates.find((item) => item.id === templateId);
  renderTemplateList();
  renderTemplateEditor(template);
  refreshTaskSelect();
}

function saveTemplate() {
  const label = templateLabel.value.trim();
  const prompt = templateBody.value.trim();
  if (!label || !prompt) {
    alert("Template name and prompt are required.");
    return;
  }

  const existingIndex = templates.findIndex((item) => item.id === activeTemplateId);
  if (existingIndex >= 0) {
    templates[existingIndex] = {
      ...templates[existingIndex],
      label,
      prompt,
    };
  } else {
    const id = slugify(label) || `template-${Date.now()}`;
    templates.push({ id, label, prompt });
    activeTemplateId = id;
  }

  storeTemplates();
  renderTemplateList();
  renderTemplateEditor(templates.find((item) => item.id === activeTemplateId));
  refreshTaskSelect();
}

function newTemplate() {
  activeTemplateId = null;
  renderTemplateList();
  renderTemplateEditor(null);
}

function deleteTemplate() {
  if (!activeTemplateId) {
    return;
  }
  templates = templates.filter((item) => item.id !== activeTemplateId);
  activeTemplateId = templates[0] ? templates[0].id : null;
  storeTemplates();
  renderTemplateList();
  renderTemplateEditor(templates.find((item) => item.id === activeTemplateId));
  refreshTaskSelect();
}

function syncTemplateListHeight() {
  if (!templateColumn || !taskColumn || !templateList) {
    return;
  }
  if (baseTemplateListHeight === null) {
    baseTemplateListHeight = Math.max(templateList.offsetHeight, 320);
  }
  templateList.style.maxHeight = `${baseTemplateListHeight}px`;
  const leftHeight = templateColumn.offsetHeight;
  const rightHeight = taskColumn.offsetHeight;
  if (rightHeight > leftHeight) {
    const diff = rightHeight - leftHeight;
    const targetHeight = baseTemplateListHeight + diff;
    templateList.style.maxHeight = `${targetHeight}px`;
  }
}

function addChatMessage(role, content) {
  const message = document.createElement("div");
  message.className = `chat-message ${role}`;

  const meta = document.createElement("strong");
  meta.textContent = role === "user" ? "You" : "BioAgent";
  message.appendChild(meta);

  const body = document.createElement("div");
  body.textContent = content;
  message.appendChild(body);

  chatHistory.appendChild(message);
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

function getRelativePath(file) {
  return file._relativePath || file.webkitRelativePath || file.name;
}

function describeSelection(list, emptyText) {
  if (!list || list.length === 0) {
    return emptyText;
  }
  const names = Array.from(list)
    .slice(0, 3)
    .map((file) => getRelativePath(file));
  const more = list.length > 3 ? ` +${list.length - 3} more` : "";
  return `${list.length} selected: ${names.join(", ")}${more}`;
}

function describeFolderSelection(list, emptyText) {
  if (!list || list.length === 0) {
    return emptyText;
  }
  const paths = Array.from(list).map((file) => getRelativePath(file));
  const root = paths[0].includes("/") ? paths[0].split("/")[0] : null;
  const sameRoot = root && paths.every((path) => path.startsWith(`${root}/`));
  if (sameRoot) {
    return `Folder: ${root} (${list.length} files)`;
  }
  return `${list.length} files selected.`;
}

function collectInputs() {
  const inputs = [];
  selectedFiles.forEach((file) => {
    inputs.push({
      name: file.name,
      path: getRelativePath(file),
      size: file.size,
      kind: "file",
    });
  });
  selectedFolderFiles.forEach((file) => {
    inputs.push({
      name: file.name,
      path: getRelativePath(file),
      size: file.size,
      kind: "folder",
    });
  });
  return inputs;
}

function renderBench(jobs) {
  bench.innerHTML = "";
  if (!jobs.length) {
    bench.innerHTML = "<div class=\"bench-card\">Visualizations, tables, and artifacts will appear here.</div>";
    return;
  }

  const latest = jobs[0];
  const card = document.createElement("div");
  card.className = "bench-card";
  card.innerHTML = `
    <strong>Latest job</strong>
    <div>${latest.task || "Untitled"}</div>
    <div>${latest.eyes} • ${latest.mind} • ${latest.hand}</div>
    <div>Status: ${latest.status}</div>
    <div>${latest.summary || ""}</div>
  `;
  bench.appendChild(card);
}

async function readAllEntries(reader) {
  const entries = [];
  let batch;
  do {
    batch = await new Promise((resolve) => reader.readEntries(resolve));
    entries.push(...batch);
  } while (batch.length);
  return entries;
}

async function collectEntryFiles(entry) {
  if (entry.isFile) {
    return new Promise((resolve) => {
      entry.file((file) => {
        file._relativePath = entry.fullPath.replace(/^\//, "");
        resolve([file]);
      });
    });
  }
  if (entry.isDirectory) {
    const reader = entry.createReader();
    const entries = await readAllEntries(reader);
    const results = [];
    for (const child of entries) {
      const childFiles = await collectEntryFiles(child);
      results.push(...childFiles);
    }
    return results;
  }
  return [];
}

async function extractFilesFromDrop(event) {
  const items = event.dataTransfer.items;
  if (items && items.length && items[0].webkitGetAsEntry) {
    const entries = Array.from(items)
      .map((item) => item.webkitGetAsEntry())
      .filter(Boolean);
    const results = [];
    for (const entry of entries) {
      const entryFiles = await collectEntryFiles(entry);
      results.push(...entryFiles);
    }
    return results;
  }
  return Array.from(event.dataTransfer.files || []);
}

function setFilesSelection(files) {
  selectedFiles = files;
  fileLabel.textContent = describeSelection(selectedFiles, "No files selected.");
}

function setFolderSelection(files) {
  selectedFolderFiles = files;
  folderLabel.textContent = describeFolderSelection(selectedFolderFiles, "No folder selected.");
}

function handleDragEvents(dropzone) {
  dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragging");
  });
  dropzone.addEventListener("drop", async (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragging");
  });
}

function renderJobs(jobs) {
  jobsContainer.innerHTML = "";
  if (!jobs.length) {
    jobsContainer.innerHTML = "<p>No jobs yet. Submit a task to queue one.</p>";
    jobCount.textContent = "0 queued";
    renderBench([]);
    return;
  }

  jobCount.textContent = `${jobs.length} queued`;
  jobs.forEach((job) => {
    const card = document.createElement("div");
    card.className = "job-card";

    const title = document.createElement("h3");
    title.textContent = job.task || "Untitled task";
    card.appendChild(title);

    const meta = document.createElement("div");
    meta.className = "job-meta";
    meta.textContent = `${job.eyes} • ${job.mind} • ${job.hand} • ${job.status}`;
    card.appendChild(meta);

    const summary = document.createElement("div");
    summary.textContent = job.summary || "";
    card.appendChild(summary);

    jobsContainer.appendChild(card);
  });

  renderBench(jobs);
}

async function fetchJobs() {
  try {
    const response = await fetch(`${apiBase}/api/jobs`);
    const data = await response.json();
    renderJobs(data.jobs || []);
  } catch (error) {
    renderJobs([]);
  }
}

async function runJob() {
  const payload = {
    eyes: eyesSelect.value,
    mind: mindSelect.value,
    hand: handSelect.value,
    task: taskSelect.value,
    prompt: promptField.value,
    notes: notesField.value,
    inputs: collectInputs(),
  };

  addChatMessage("user", `Run ${payload.task} using ${payload.eyes}/${payload.mind}/${payload.hand}.`);

  const response = await fetch(`${apiBase}/api/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (response.ok) {
    const job = await response.json();
    addChatMessage("assistant", `Queued job ${job.id}. Awaiting SDK execution.`);
    await fetchJobs();
  } else {
    addChatMessage("assistant", "Unable to queue job. Check backend status.");
  }
}

async function loadOptions() {
  try {
    const response = await fetch(`${apiBase}/api/options`);
    const data = await response.json();

    fillSelect(eyesSelect, data.eyes, data.defaults.eyes);
    fillSelect(mindSelect, data.minds, data.defaults.mind);
    fillSelect(handSelect, data.hands, data.defaults.hand);

    const stored = loadStoredTemplates();
    templates = stored && stored.length ? stored : data.tasks;
    activeTemplateId = templates[0] ? templates[0].id : null;
    renderTemplateList();
    renderTemplateEditor(templates.find((item) => item.id === activeTemplateId));
    refreshTaskSelect();

    setStatus("Online", "good");
  } catch (error) {
    setStatus("Offline", "bad");
  }
}

taskSelect.addEventListener("change", () => {
  const selected = templates.find((item) => item.id === taskSelect.value);
  if (selected) {
    promptField.value = selected.prompt;
  }
});

templateSave.addEventListener("click", () => {
  saveTemplate();
});

templateNew.addEventListener("click", () => {
  newTemplate();
});

templateDelete.addEventListener("click", () => {
  deleteTemplate();
});

document.getElementById("run").addEventListener("click", () => {
  runJob();
});

document.getElementById("refresh").addEventListener("click", () => {
  fetchJobs();
});

chatSend.addEventListener("click", () => {
  const message = chatInput.value.trim();
  if (!message) {
    return;
  }
  addChatMessage("user", message);
  addChatMessage("assistant", "Message received. Use Task Studio to run jobs." );
  chatInput.value = "";
});

fileInput.addEventListener("change", () => {
  setFilesSelection(Array.from(fileInput.files || []));
});

folderInput.addEventListener("change", () => {
  setFolderSelection(Array.from(folderInput.files || []));
});

browseFilesButton.addEventListener("click", () => {
  fileInput.click();
});

browseFolderButton.addEventListener("click", () => {
  folderInput.click();
});

handleDragEvents(filesDropzone);
handleDragEvents(folderDropzone);

filesDropzone.addEventListener("drop", async (event) => {
  const files = await extractFilesFromDrop(event);
  setFilesSelection(files);
});

folderDropzone.addEventListener("drop", async (event) => {
  const files = await extractFilesFromDrop(event);
  setFolderSelection(files);
});

loadOptions();
fetchJobs();

window.addEventListener("resize", () => {
  requestAnimationFrame(syncTemplateListHeight);
});
