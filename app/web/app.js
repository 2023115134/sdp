const senderForm = document.getElementById("sender-form");
const senderButton = document.getElementById("run-sender");
const resetButton = document.getElementById("reset-demo");
const deliverButton = document.getElementById("deliver-carrier");
const receiverButton = document.getElementById("run-receiver");
const carrierText = document.getElementById("carrier-text");
const receivedCarrier = document.getElementById("received-carrier");
const senderOutput = document.getElementById("sender-stage-output");
const receiverOutput = document.getElementById("receiver-stage-output");
const logList = document.getElementById("execution-log");
const modeSelect = document.getElementById("key-mode");
const senderPassword = document.getElementById("sender-password");
const receiverPassword = document.getElementById("receiver-password");
let sessionMetadata = null;
let currentEmbeddingEvent = {};
let lastEmbeddingProgress = {};
let ecdheRounds = new Map();

const ecdheStatusLabels = {
  generating: "Generating",
  delivered: "Delivered",
  received: "Received",
  verified: "Verified",
  failed: "Failed",
};

function resetEcdheExchange() {
  ecdheRounds.clear();
  const list = document.getElementById("ecdhe-carrier-list");
  list.replaceChildren();
  const empty = document.createElement("li");
  empty.className = "ecdhe-empty";
  empty.textContent = modeSelect.value === "ECDHE"
    ? "Waiting for the backend to begin the key exchange."
    : "ECDHE carrier exchanges will appear here when ECDHE mode is run.";
  list.append(empty);
  const status = document.getElementById("ecdhe-exchange-status");
  status.textContent = modeSelect.value === "ECDHE" ? "Ready" : "Waiting for ECDHE mode";
  status.className = "status-label pending";
  const error = document.getElementById("ecdhe-handshake-error");
  error.textContent = "";
  error.classList.add("hidden");
}

function setEcdheHandshakeFailure(message) {
  const status = document.getElementById("ecdhe-exchange-status");
  status.textContent = "Handshake failed";
  status.className = "status-label failed";
  const error = document.getElementById("ecdhe-handshake-error");
  error.textContent = message;
  error.classList.remove("hidden");
}

function updateEcdheCarrier(data) {
  const round = { ...(ecdheRounds.get(data.round_number) || {}), ...data };
  ecdheRounds.set(data.round_number, round);
  const list = document.getElementById("ecdhe-carrier-list");
  list.querySelector(".ecdhe-empty")?.remove();

  const item = document.createElement("li");
  item.className = `ecdhe-round ${round.status || "pending"}`;
  item.dataset.roundNumber = String(round.round_number);
  const heading = document.createElement("div");
  heading.className = "ecdhe-round-heading";
  const title = document.createElement("strong");
  title.textContent = `Exchange ${round.round_number}/${round.total_rounds} · ${round.participant}`;
  const status = document.createElement("span");
  status.className = `stage-state ${round.status || "pending"}`;
  status.textContent = ecdheStatusLabels[round.status] || "Pending";
  heading.append(title, status);

  const direction = document.createElement("p");
  direction.className = "ecdhe-round-meta";
  direction.textContent = `${round.direction || ""} · ${round.purpose || ""}`;
  item.append(heading, direction);

  if (round.progress) {
    const progress = round.progress;
    const details = document.createElement("p");
    details.className = "ecdhe-round-progress";
    const parts = [
      `${progress.completed_characters}/${progress.total_characters} characters`,
      progress.current_character === undefined
        ? ""
        : `character ${progress.current_character}${progress.current_character_value === undefined
          ? ""
          : ` · ${progress.current_character_value === " " ? "SPACE" : progress.current_character_value}`}`,
      progress.target_position === undefined ? "" : `position ${progress.target_position}`,
      progress.retry_number === undefined
        ? ""
        : `retry ${progress.retry_number}${progress.retry_limit ? `/${progress.retry_limit}` : ""}`,
      progress.last_rejection ? `rejected: ${progress.last_rejection}` : "",
      progress.error ? `error: ${progress.error}` : "",
    ].filter(Boolean);
    details.textContent = `${progress.current_stage || "Embedding"} · ${parts.join(" · ")}`;
    item.append(details);
  }

  if (round.carrier_text !== undefined) {
    const label = document.createElement("span");
    label.className = "ecdhe-carrier-label";
    label.textContent = `Outgoing carrier · ${round.direction || ""}`;
    const text = document.createElement("pre");
    text.className = "ecdhe-carrier-text";
    text.textContent = round.carrier_text;
    item.append(label, text);
    if (["received", "verified"].includes(round.status)) {
      const incomingLabel = document.createElement("span");
      incomingLabel.className = "ecdhe-carrier-label";
      const receiver = round.participant === "Sender" ? "Receiver" : "Sender";
      incomingLabel.textContent = `Incoming carrier · received by ${receiver}`;
      const incomingText = document.createElement("pre");
      incomingText.className = "ecdhe-carrier-text";
      incomingText.textContent = round.carrier_text;
      item.append(incomingLabel, incomingText);
    }
  }
  if (round.error) {
    const error = document.createElement("p");
    error.className = "ecdhe-round-error";
    error.setAttribute("role", "alert");
    error.textContent = round.error;
    item.append(error);
  }

  const existing = list.querySelector(`[data-round-number="${round.round_number}"]`);
  if (existing) existing.replaceWith(item);
  else list.append(item);

  const exchangeStatus = document.getElementById("ecdhe-exchange-status");
  if (round.status === "failed") {
    exchangeStatus.textContent = `Exchange ${round.round_number}/${round.total_rounds} failed`;
    exchangeStatus.className = "status-label failed";
    setEcdheHandshakeFailure(round.error || `Exchange ${round.round_number} failed.`);
    return;
  }
  exchangeStatus.className = "status-label running";
  if (round.status === "verified") {
    const verified = [...ecdheRounds.values()].filter((entry) => entry.status === "verified").length;
    exchangeStatus.textContent = `${verified}/${round.total_rounds} carriers verified`;
  } else {
    exchangeStatus.textContent = `Exchange ${round.round_number}/${round.total_rounds} · ${ecdheStatusLabels[round.status] || "Pending"}`;
  }
}

function updateHandshakeStage(name, data) {
  if (name === "ecdhe_carrier") {
    updateEcdheCarrier(data);
    return;
  }
  if (name === "ecdhe_carrier_progress") {
    updateEcdheCarrier(data);
    const progress = data.progress || {};
    const message = `ECDHE exchange ${data.round_number}/${data.total_rounds}: ${progress.current_stage || "Embedding public-key carrier"}.`;
    updateEmbeddingProgress({ ...progress, message });
    setOverallStatus(message, "running");
    return;
  }
  updateSenderStage(name, data);
  if (name === "ecdhe_key_confirmation" && data.status === "success") {
    const status = document.getElementById("ecdhe-exchange-status");
    status.textContent = "Key confirmation verified";
    status.className = "status-label success";
  }
  if (name === "ecdhe_key_derivation" && data.status === "success") {
    const totalRounds = Number([...ecdheRounds.values()][0]?.total_rounds || 0);
    const verifiedRounds = [...ecdheRounds.values()].filter((entry) => entry.status === "verified").length;
    const status = document.getElementById("ecdhe-exchange-status");
    status.textContent = `Handshake complete · ${verifiedRounds}/${totalRounds} carriers verified`;
    status.className = "status-label success";
  }
  if (name.startsWith("ecdhe_") && data.status === "failed") {
    setEcdheHandshakeFailure(data.message || "ECDHE handshake failed.");
  }
}

function setBusy(busy) {
  senderButton.disabled = busy;
  resetButton.disabled = busy;
  modeSelect.disabled = busy;
  deliverButton.disabled = busy || !sessionMetadata;
  receiverButton.disabled = busy || !sessionMetadata;
}

function clearPipelineResults() {
  senderOutput.replaceChildren();
  senderOutput.className = "stage-grid";
  const senderEmpty = document.createElement("div");
  senderEmpty.className = "empty-state";
  senderEmpty.textContent = "Sender stages will appear here after a successful run.";
  senderOutput.append(senderEmpty);

  receiverOutput.replaceChildren();
  receiverOutput.className = "receiver-results empty-state";
  receiverOutput.textContent = "Receiver results will appear here. Plaintext is shown only after successful AES-GCM authentication.";
  receivedCarrier.value = "";
  document.getElementById("carrier-empty").classList.remove("hidden");
  document.getElementById("carrier-result").classList.add("hidden");
  carrierText.textContent = "";
  document.getElementById("carrier-length").textContent = "0";
  document.getElementById("channel-status").textContent = "Waiting for sender output.";
  document.getElementById("receiver-status").textContent = "Waiting for carrier delivery.";
  document.getElementById("receiver-status").className = "receiver-status pending";
  currentEmbeddingEvent = {};
  lastEmbeddingProgress = {};
  updateEmbeddingProgress({});
  resetEcdheExchange();
}

const senderStageTitles = {
  ecdhe_key_pair_generation: "ECDHE · Ephemeral X25519 key pairs",
  ecdhe_public_key_exchange: "ECDHE · Public-key exchange",
  ecdhe_shared_secret: "ECDHE · Shared-secret agreement",
  ecdhe_key_confirmation: "ECDHE · Key confirmation",
  ecdhe_key_derivation: "ECDHE · PBKDF2 DK1/DK2 derivation",
  key_establishment: "A · Key establishment",
  pbkdf2: "B · PBKDF2 key derivation",
  encryption: "C · AES-256-GCM encryption",
  payload: "D · Payload construction",
  mapping: "E · h4 character mapping",
  positions: "F · SHAKE-128 positions",
  embedding: "G · EmbedderLLM generation",
  carrier: "H · Final carrier",
  sender_verification: "Sender-side extraction and verification",
};
const senderStageOrder = Object.keys(senderStageTitles);

const receiverStageTitles = {
  key_derivation: "J · Receiver key derivation",
  extraction: "K · Position regeneration and extraction",
  inverse_mapping: "L · Inverse mapping and reconstruction",
  authentication: "M · Authenticated decryption",
};

const senderFields = {
  ecdhe_key_pair_generation: [
    ["Algorithm", "algorithm"],
    ["Handshake progress", "message"],
  ],
  ecdhe_public_key_exchange: [
    ["Transport", "transport"],
    ["Handshake progress", "message"],
  ],
  ecdhe_shared_secret: [
    ["Endpoint agreement", "match"],
    ["Handshake progress", "message"],
  ],
  ecdhe_key_confirmation: [
    ["Handshake progress", "message"],
  ],
  ecdhe_key_derivation: [
    ["Algorithm", "algorithm"],
    ["Iterations", "iterations"],
    ["Public salt (hex)", "salt_hex"],
    ["DK1 length", "dk1_length", "bytes"],
    ["DK2 length", "dk2_length", "bytes"],
    ["Handshake progress", "message"],
  ],
  key_establishment: [
    ["Selected mode", "mode"],
    ["Explanation", "explanation"],
  ],
  pbkdf2: [
    ["Algorithm", "algorithm"],
    ["Iterations", "iterations"],
    ["Public salt (hex)", "salt_hex"],
    ["Derived key length", "derived_key_length", "bytes"],
    ["DK1 length", "dk1_length", "bytes"],
    ["DK2 length", "dk2_length", "bytes"],
    ["Key purposes", "explanation"],
  ],
  encryption: [
    ["Original plaintext", "plaintext"],
    ["Algorithm", "algorithm"],
    ["Nonce (hex)", "nonce_hex"],
    ["AAD (hex)", "aad_hex"],
    ["Ciphertext (hex)", "ciphertext_hex"],
    ["Authentication tag (hex)", "tag_hex"],
    ["Ciphertext length", "ciphertext_length", "bytes"],
    ["Tag length", "tag_length", "bytes"],
  ],
  payload: [
    ["Payload order", "order"],
    ["Combined payload (hex)", "hex"],
    ["Payload length", "byte_length", "bytes"],
    ["Hexadecimal characters", "hex_character_length"],
  ],
  mapping: [
    ["Input payload (hex)", "input_hex"],
    ["h4 output", "output"],
    ["Input bytes", "input_byte_length"],
    ["Mapped character count", "output_character_length"],
    ["How it works", "explanation"],
  ],
  positions: [
    ["Algorithm", "algorithm"],
    ["Number of targets", "count"],
    ["Position key", "key_source"],
  ],
  embedding: [
    ["Progress", "message"],
    ["Current stage", "current_stage"],
    ["Characters embedded", "completed_characters"],
    ["Total characters", "total_characters"],
    ["Current character", "current_character_value"],
    ["Target position", "target_position"],
    ["Retry", "retry_number"],
    ["Retry limit", "retry_limit"],
    ["Candidate rejection", "last_rejection"],
    ["Error", "error"],
    ["Topic", "topic"],
    ["Model", "model"],
    ["Temperature", "temperature"],
    ["Top-k", "top_k"],
    ["Hidden characters", "mapped_character_count"],
    ["Embedder attempts", "attempts"],
    ["LLM calls", "llm_calls"],
    ["Candidate evaluations", "candidate_evaluations"],
    ["Retries", "retries"],
  ],
  carrier: [
    ["Length", "length", "characters"],
    ["Generated carrier text", "text"],
    ["Embedding verification", "verified"],
    ["Caution", "notice"],
  ],
  sender_verification: [
    ["Extracted characters", "extracted"],
    ["AES-GCM authentication", "authentication"],
    ["Verification detail", "message"],
  ],
};

const receiverFields = {
  key_derivation: [
    ["Algorithm", "algorithm"],
    ["Iterations", "iterations"],
    ["DK1 length", "dk1_length", "bytes"],
    ["DK2 length", "dk2_length", "bytes"],
  ],
  extraction: [
    ["Regenerated target positions", "positions"],
    ["Extracted h4 characters", "extracted"],
  ],
  inverse_mapping: [
    ["Recovered payload (hex)", "payload_hex"],
    ["Authentication tag (hex)", "tag_hex"],
    ["Ciphertext (hex)", "ciphertext_hex"],
    ["Tag length", "tag_length", "bytes"],
    ["Ciphertext length", "ciphertext_length", "bytes"],
  ],
  authentication: [
    ["Verification", "message"],
    ["Recovered plaintext", "plaintext"],
  ],
};

function addLog(message) {
  const item = document.createElement("li");
  const time = document.createElement("time");
  const text = document.createElement("span");
  time.textContent = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  text.textContent = message;
  item.append(time, text);
  logList.prepend(item);
}

function setOverallStatus(text, status) {
  const badge = document.getElementById("overall-status");
  badge.textContent = text;
  badge.className = `status-label ${status}`;
}

function updateEmbeddingProgress(data) {
  const total = Number(data.total_characters ?? data.mapped_character_count ?? 0);
  const completed = Number(data.completed_characters ?? 0);
  const current = Number(data.current_character ?? 0);
  const meter = document.getElementById("embedding-meter");
  meter.max = Math.max(total, 1);
  meter.value = Math.min(completed, total);
  document.getElementById("embedding-count").textContent =
    `${completed} / ${total} characters`;
  document.getElementById("embedding-progress-label").textContent =
    data.message || data.current_stage || "No embedding started";

  const character = data.current_character_value;
  document.getElementById("embedding-character").textContent = current
    ? `${current}/${total}${character === undefined ? "" : ` · ${character === " " ? "SPACE" : character}`}`
    : "—";
  const position = data.target_position ?? data.failed_position;
  document.getElementById("embedding-position").textContent =
    position === undefined ? "—" : String(position);
  const retry = data.retry_number === undefined
    ? "—"
    : `${data.retry_number}${data.retry_limit ? ` / ${data.retry_limit}` : ""}`
      + (data.temperature === undefined ? "" : ` · T ${data.temperature}`)
      + (data.top_k === undefined ? "" : ` · k ${data.top_k}`);
  document.getElementById("embedding-retry").textContent = retry;

  const error = document.getElementById("embedding-error");
  const errorText = data.error || (data.status === "failed" ? data.message : "");
  error.textContent = errorText || "";
  error.classList.toggle("hidden", !errorText);
}

function setPipelineStage(name, status) {
  const item = document.querySelector(`[data-step="${name}"]`);
  if (!item) return;
  item.classList.remove("success", "failed", "running");
  if (status && status !== "pending") item.classList.add(status);
}

function resetPipeline() {
  document.querySelectorAll(".pipeline-track li").forEach((item) => {
    item.classList.remove("success", "failed", "running");
  });
  setOverallStatus("Waiting for input", "pending");
  updateEmbeddingProgress({});
}

function appendField(container, label, value, copyable = false) {
  if (value === undefined || value === null) return;
  const field = document.createElement("div");
  field.className = "stage-field";
  const caption = document.createElement("span");
  caption.textContent = label;
  const content = document.createElement("div");
  content.className = `stage-value${copyable ? " has-copy" : ""}`;
  content.textContent = typeof value === "string" ? value : JSON.stringify(value);
  if (copyable && typeof value === "string") {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "copy-small";
    button.textContent = "Copy";
    button.addEventListener("click", () => copyText(value, button));
    content.append(button);
  }
  field.append(caption, content);
  container.append(field);
}

function createStageCard(title, values, fields, statusOverride) {
  const card = document.createElement("article");
  card.className = "stage-card";
  const heading = document.createElement("div");
  heading.className = "stage-title";
  const titleElement = document.createElement("h3");
  titleElement.textContent = title;
  const state = document.createElement("span");
  const status = statusOverride || values.status || "pending";
  state.className = `stage-state ${status}`;
  state.textContent = status;
  heading.append(titleElement, state);
  card.append(heading);

  const fieldContainer = document.createElement("div");
  fieldContainer.className = "stage-fields";
  for (const [label, key, unit] of fields) {
    if (values[key] === undefined || values[key] === null) continue;
    const rendered = unit ? `${values[key]} ${unit}` : values[key];
    appendField(fieldContainer, label, rendered, typeof rendered === "string" && rendered.length < 12000);
  }
  if (Object.keys(values).includes("rows")) {
    const table = document.createElement("div");
    table.className = "data-table-wrap";
    const tableElement = document.createElement("table");
    tableElement.className = "data-table";
    const head = document.createElement("thead");
    const headerRow = document.createElement("tr");
    for (const label of ["Index", "Mapped character", "Target position"]) {
      const cell = document.createElement("th");
      cell.textContent = label;
      headerRow.append(cell);
    }
    head.append(headerRow);
    const body = document.createElement("tbody");
    const rows = document.createDocumentFragment();
    for (const row of values.rows) {
      const tr = document.createElement("tr");
      for (const value of [row.index, row.character === " " ? "SPACE" : row.character, row.position]) {
        const td = document.createElement("td");
        td.textContent = String(value);
        tr.append(td);
      }
      rows.append(tr);
    }
    body.append(rows);
    tableElement.append(head, body);
    table.append(tableElement);
    appendField(fieldContainer, "Mapped character → target position", "");
    fieldContainer.lastElementChild.append(table);
  }
  card.append(fieldContainer);
  return card;
}

function renderStages(target, stages, titles, fields) {
  target.replaceChildren();
  for (const [key, title] of Object.entries(titles)) {
    const data = stages[key];
    if (!data) continue;
    target.append(createStageCard(title, data, fields[key] || []));
  }
}

function updateSenderStage(name, data) {
  const index = senderStageOrder.indexOf(name);
  if (index === -1) return;
  if (name === "embedding") {
    data = { ...lastEmbeddingProgress, ...data };
    lastEmbeddingProgress = data;
  }

  const existingEmpty = senderOutput.querySelector(".empty-state");
  if (existingEmpty) existingEmpty.remove();
  const card = createStageCard(
    senderStageTitles[name],
    data,
    senderFields[name] || [],
  );
  card.dataset.outputStage = name;
  const existing = senderOutput.querySelector(`[data-output-stage="${name}"]`);
  if (existing) {
    existing.replaceWith(card);
  } else {
    const nextCard = [...senderOutput.children].find(
      (child) => senderStageOrder.indexOf(child.dataset.outputStage) > index,
    );
    senderOutput.insertBefore(card, nextCard || null);
  }

  const pipelineStep = {
    ecdhe_key_pair_generation: "key_establishment",
    ecdhe_public_key_exchange: "key_establishment",
    ecdhe_shared_secret: "key_establishment",
    ecdhe_key_derivation: "key_establishment",
    key_establishment: "key_establishment",
    pbkdf2: "pbkdf2",
    encryption: "encryption",
    payload: "payload",
    mapping: "mapping",
    positions: "positions",
    embedding: "embedding",
    carrier: "carrier",
  }[name];
  if (pipelineStep) setPipelineStage(pipelineStep, data.status);
  if (name === "embedding") {
    updateEmbeddingProgress(data);
    if (data.status === "failed") {
      setOverallStatus("Embedding failed", "failed");
    } else if (data.status === "success") {
      setOverallStatus("Embedding complete", "success");
    }
  }
  if (name.startsWith("ecdhe_") && data.status === "running") {
    setOverallStatus(data.message || "ECDHE handshake in progress", "running");
  }
  if (name === "embedding" && data.status === "running" && data.current_character) {
    const current = data.current_character;
    const total = data.total_characters || data.mapped_character_count;
    const status = data.current_stage === "Character embedded"
      ? `Embedded ${current}/${total}`
      : `Generating ${current}/${total}`;
    setOverallStatus(status, "running");
  }
  if (name === "sender_verification" && data.status === "success") {
    for (const step of ["extraction", "inverse_mapping", "authentication", "plaintext"]) {
      setPipelineStage(step, "success");
    }
  }
  if (name === "carrier" && data.status === "success") {
    document.getElementById("carrier-empty").classList.add("hidden");
    document.getElementById("carrier-result").classList.remove("hidden");
    carrierText.textContent = data.text;
    document.getElementById("carrier-length").textContent = String(data.length);
    document.getElementById("channel-status").textContent = "Carrier generated. Sender verification is finishing.";
  }
}

function updateReceiverStage(name, data) {
  const title = receiverStageTitles[name];
  if (!title) return;
  const card = createStageCard(title, data, receiverFields[name] || []);
  card.dataset.receiverStage = name;
  const existing = receiverOutput.querySelector(`[data-receiver-stage="${name}"]`);
  if (existing) {
    existing.replaceWith(card);
  } else {
    const emptyState = receiverOutput.querySelector(".empty-state");
    if (emptyState) emptyState.remove();
    const order = Object.keys(receiverStageTitles);
    const index = order.indexOf(name);
    const nextCard = [...receiverOutput.children].find(
      (child) => order.indexOf(child.dataset.receiverStage) > index,
    );
    receiverOutput.classList.remove("empty-state");
    receiverOutput.insertBefore(card, nextCard || null);
  }
  if (["extraction", "inverse_mapping", "authentication"].includes(name)) {
    setPipelineStage(name, data.status);
  }
  if (name === "key_derivation" && data.status === "running") {
    setPipelineStage("key_establishment", "running");
  } else if (name === "key_derivation" && data.status === "success") {
    setPipelineStage("key_establishment", "success");
  }
  if (data.status === "running" && data.message) {
    document.getElementById("receiver-status").textContent = data.message;
    document.getElementById("receiver-status").className = "receiver-status pending";
    setOverallStatus(data.message, "running");
  }
}

function updateModeUI() {
  const ecdhe = modeSelect.value === "ECDHE";
  document.getElementById("ecdhe-exchange").classList.toggle("hidden", !ecdhe);
  document.getElementById("sender-password-field").classList.toggle("hidden", ecdhe);
  document.getElementById("receiver-password-field").classList.toggle("hidden", ecdhe);
  senderPassword.required = !ecdhe;
  receiverPassword.required = !ecdhe;
  if (ecdhe) {
    senderPassword.value = "";
    receiverPassword.value = "";
  }
  document.getElementById("mode-pill").textContent = `${modeSelect.value} MODE`;
  document.getElementById("key-mode-step").textContent = ecdhe
    ? "ECDHE · local handshake"
    : "PSK";
  document.getElementById("mode-explanation").textContent = ecdhe
    ? "No password is used. The backend runs a real X25519 exchange between two logical endpoints on this computer; the exchange is not peer-authenticated."
    : "Sender and receiver independently derive keys from the same pre-shared password.";
  document.getElementById("receiver-key-help").textContent = ecdhe
    ? "The receiver uses its own backend-held ECDHE-derived keys from this local session; no password or key bytes are sent."
    : "Enter the same password used by the sender. The receiver re-derives its own keys; no key bytes are transferred.";
  resetEcdheExchange();
}

async function copyText(text, button) {
  try {
    await navigator.clipboard.writeText(text);
    const previous = button.textContent;
    button.textContent = "Copied";
    window.setTimeout(() => { button.textContent = previous; }, 1200);
  } catch {
    button.textContent = "Copy unavailable";
    window.setTimeout(() => { button.textContent = "Copy"; }, 1500);
  }
}

function showSenderResult(data) {
  sessionMetadata = data.session;
  for (const [name, stage] of Object.entries(data.stages)) {
    if (!senderOutput.querySelector(`[data-output-stage="${name}"]`)) {
      updateSenderStage(name, stage);
    }
  }
  deliverButton.disabled = false;
  receiverButton.disabled = false;
  document.getElementById("channel-status").textContent = "Carrier generated. Deliver it when ready.";
  document.getElementById("receiver-status").textContent = "Carrier ready. Deliver or edit the text, then run extraction.";
  document.getElementById("receiver-status").className = "receiver-status pending";
  for (const name of ["key_establishment", "pbkdf2", "encryption", "payload", "mapping", "positions", "embedding", "carrier"]) {
    setPipelineStage(name, "success");
  }
  for (const name of ["extraction", "inverse_mapping", "authentication", "plaintext"]) setPipelineStage(name, null);
}

function resetDemo() {
  senderForm.reset();
  receiverPassword.value = "";
  sessionMetadata = null;
  clearPipelineResults();
  updateModeUI();
  setBusy(false);
  resetPipeline();
  logList.replaceChildren();
  addLog("Demo reset. No password or key material was retained.");
}

async function postEventStream(url, payload, onStage, requestName) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    cache: "no-store",
    body: JSON.stringify(payload),
  });
  if (!response.ok || !response.body) {
    let errorData = null;
    try {
      errorData = await response.json();
    } catch {
      // Keep the transport error when the server did not return JSON.
    }
    const failure = new Error(errorData && errorData.error || `The ${requestName} request failed.`);
    failure.stage = errorData && errorData.stage;
    throw failure;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let completed = null;
  const consumeLine = (line) => {
    if (!line.trim()) return;
    const event = JSON.parse(line);
    if (event.type === "stage") {
      onStage(event.key, event.data);
    } else if (event.type === "error") {
      const error = new Error(event.error || `The ${requestName} request failed.`);
      error.stage = event.stage;
      throw error;
    } else if (event.type === "complete") {
      completed = event.data;
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop();
    for (const line of lines) consumeLine(line);
    if (done) break;
  }
  if (buffer.trim()) consumeLine(buffer);
  if (!completed) throw new Error(`The ${requestName} stream ended before completion.`);
  return completed;
}

function postSenderStream(payload) {
  return postEventStream("/api/sender/stream", payload, updateSenderStage, "sender");
}

function postHandshakeStream(topic) {
  return postEventStream(
    "/api/ecdhe/handshake/stream",
    { topic },
    updateHandshakeStage,
    "ECDHE handshake",
  );
}

function postReceiverStream(payload) {
  return postEventStream(
    "/api/receiver/stream",
    payload,
    updateReceiverStage,
    "receiver",
  );
}

senderForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!senderForm.reportValidity()) return;
  sessionMetadata = null;
  clearPipelineResults();
  setBusy(true);
  senderButton.querySelector("span").textContent = "Running real sender pipeline…";
  resetPipeline();
  setOverallStatus("Sender running", "running");
  setPipelineStage("key_establishment", "running");
  document.getElementById("channel-status").textContent = "Waiting for real LLM generation to finish.";
  addLog("Sender pipeline started. Password and plaintext are not written to the execution log.");
  let ecdheHandshakeComplete = false;
  try {
    const mode = modeSelect.value;
    let ecdheSessionId;
    if (mode === "ECDHE") {
      senderButton.querySelector("span").textContent = "Establishing ECDHE session…";
      addLog("ECDHE handshake started between two logical endpoints on this computer; no password is used.");
      const handshake = await postHandshakeStream(document.getElementById("topic").value);
      ecdheSessionId = handshake.session_id;
      ecdheHandshakeComplete = true;
      addLog("ECDHE handshake completed. Public-key exchange is unauthenticated; secret key bytes were not exposed.");
    }
    senderButton.querySelector("span").textContent = "Running real sender pipeline…";
    const data = await postSenderStream({
      message: document.getElementById("secret-message").value,
      ...(mode === "PSK" ? { password: senderPassword.value } : {}),
      topic: document.getElementById("topic").value,
      mode,
      ...(ecdheSessionId ? { ecdhe_session_id: ecdheSessionId } : {}),
      temperature: Number(document.getElementById("temperature").value),
      top_k: Number(document.getElementById("top-k").value),
    });
    showSenderResult(data);
    setOverallStatus("Sender verified · ready to deliver", "success");
    addLog("Sender completed: encryption, real embedding, extraction, and AEAD verification succeeded.");
  } catch (error) {
    if (modeSelect.value === "ECDHE" && !ecdheHandshakeComplete) {
      setEcdheHandshakeFailure(error.message);
    }
    const failedStage = error.stage === "verification" ? "sender_verification" : error.stage;
    if (failedStage && senderStageTitles[failedStage]) {
      updateSenderStage(failedStage, { status: "failed", message: error.message });
    }
    const progressStep = {
      cryptography: "encryption",
      embedding: "embedding",
      verification: "authentication",
    }[error.stage];
    if (progressStep) setPipelineStage(progressStep, "failed");
    setOverallStatus("Sender failed", "failed");
    document.getElementById("channel-status").textContent = error.message;
    addLog(`Sender failed at ${error.stage || "pipeline"} stage: ${error.message}`);
  } finally {
    senderPassword.value = "";
    setBusy(false);
    senderButton.querySelector("span").textContent = "Run complete sender";
  }
});

modeSelect.addEventListener("change", () => {
  sessionMetadata = null;
  clearPipelineResults();
  resetPipeline();
  updateModeUI();
  setBusy(false);
  addLog(`${modeSelect.value} mode selected. Previous session metadata was discarded from the browser.`);
});

deliverButton.addEventListener("click", () => {
  if (!sessionMetadata) return;
  receivedCarrier.value = carrierText.textContent;
  document.getElementById("channel-status").textContent = "Carrier text delivered to the receiver editor.";
  document.getElementById("receiver-status").textContent = modeSelect.value === "ECDHE"
    ? "Carrier delivered. Run receiver extraction with the established local ECDHE session."
    : "Carrier delivered. Enter the receiver PSK to run extraction.";
  if (modeSelect.value === "PSK") receiverPassword.focus();
  addLog("Carrier text delivered through the public-channel simulation. Session metadata stayed separate.");
});

receiverButton.addEventListener("click", async () => {
  if (!sessionMetadata) {
    document.getElementById("receiver-status").textContent = "Run the sender first to establish public session metadata.";
    return;
  }
  const mode = modeSelect.value;
  if (mode === "PSK" && !receiverPassword.value) {
    receiverPassword.focus();
    document.getElementById("receiver-status").textContent = "Enter the receiver PSK.";
    return;
  }
  receiverOutput.replaceChildren();
  receiverOutput.className = "receiver-results";
  setBusy(true);
  receiverButton.textContent = "Running receiver pipeline…";
  document.getElementById("receiver-status").textContent = "Regenerating positions and verifying the extracted payload…";
  document.getElementById("receiver-status").className = "receiver-status pending";
  addLog(mode === "ECDHE"
    ? "Receiver started using its backend-held ECDHE-derived keys; no key bytes were transferred."
    : "Receiver started with independently derived PSK keys; no key bytes were transferred.");
  try {
    const result = await postReceiverStream({
      carrier_text: receivedCarrier.value,
      ...(mode === "PSK" ? { password: receiverPassword.value } : {}),
      mode,
      salt_hex: sessionMetadata.salt_hex,
      nonce_hex: sessionMetadata.nonce_hex,
      mapped_length: sessionMetadata.mapped_length,
      ...(mode === "ECDHE"
        ? { ecdhe_session_id: sessionMetadata.ecdhe_session_id }
        : {}),
    });
    const stages = result.stages;
    renderStages(receiverOutput, stages, receiverStageTitles, receiverFields);
    receiverOutput.className = "receiver-results";
    const authenticated = stages.authentication.status === "success";
    if (authenticated) {
      const plaintext = document.createElement("div");
      plaintext.className = "plaintext-result";
      plaintext.textContent = stages.authentication.plaintext;
      appendField(receiverOutput, "Authenticated plaintext", "");
      receiverOutput.lastElementChild.append(plaintext);
      document.getElementById("receiver-status").textContent = "Authentication succeeded. Plaintext recovered.";
      document.getElementById("receiver-status").className = "receiver-status success";
      setOverallStatus("End-to-end demo verified", "success");
      for (const name of ["extraction", "inverse_mapping", "authentication", "plaintext"]) setPipelineStage(name, "success");
      addLog("Receiver authenticated the payload and recovered plaintext.");
    } else {
      document.getElementById("receiver-status").textContent = stages.authentication.message;
      document.getElementById("receiver-status").className = "receiver-status failed";
      for (const name of ["extraction", "inverse_mapping"]) setPipelineStage(name, "success");
      setPipelineStage("authentication", "failed");
      setPipelineStage("plaintext", "failed");
      setOverallStatus("Authentication failed · no plaintext", "failed");
      addLog("Receiver rejected the payload during AEAD authentication. No plaintext was returned.");
    }
  } catch (error) {
    receiverOutput.replaceChildren();
    receiverOutput.className = "receiver-results";
    const failure = document.createElement("div");
    failure.className = "stage-card";
    const title = document.createElement("h3");
    title.textContent = `Receiver stopped at ${error.stage || "input"} stage`;
    const detail = document.createElement("p");
    detail.className = "stage-copy";
    detail.textContent = error.message;
    failure.append(title, detail);
    receiverOutput.append(failure);
    document.getElementById("receiver-status").textContent = error.message;
    document.getElementById("receiver-status").className = "receiver-status failed";
    if (error.stage === "key_derivation") setPipelineStage("key_establishment", "failed");
    else if (error.stage === "extraction") setPipelineStage("extraction", "failed");
    else if (error.stage === "inverse_mapping") {
      setPipelineStage("extraction", "success");
      setPipelineStage("inverse_mapping", "failed");
    } else {
      setPipelineStage("authentication", "failed");
    }
    setPipelineStage("plaintext", "failed");
    setOverallStatus("Receiver failed · no plaintext", "failed");
    addLog(`Receiver failed at ${error.stage || "input"} stage: ${error.message}`);
  } finally {
    receiverPassword.value = "";
    setBusy(false);
    receiverButton.innerHTML = 'Run receiver pipeline <span aria-hidden="true">→</span>';
  }
});

document.getElementById("copy-carrier").addEventListener("click", (event) => {
  copyText(carrierText.textContent, event.currentTarget);
});

document.getElementById("download-carrier").addEventListener("click", () => {
  const blob = new Blob([carrierText.textContent], { type: "text/plain;charset=utf-8" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "llm-shield-carrier.txt";
  link.click();
  URL.revokeObjectURL(link.href);
  addLog("Carrier text downloaded.");
});

resetButton.addEventListener("click", resetDemo);
updateModeUI();

async function loadConfig() {
  try {
    const response = await fetch("/api/config", { cache: "no-store" });
    if (!response.ok) throw new Error("Configuration request failed.");
    const config = await response.json();
    document.getElementById("model-label").textContent = config.model;
    document.getElementById("secret-message").maxLength = Math.floor(config.max_mapped_characters / 2) - 16;
  } catch {
    document.getElementById("model-label").textContent = "Backend configuration unavailable";
    addLog("Could not load backend configuration. Confirm the local server is running.");
  }
}

document.getElementById("receiver-password").addEventListener("keydown", (event) => {
  if (event.key === "Enter") receiverButton.click();
});

loadConfig();
