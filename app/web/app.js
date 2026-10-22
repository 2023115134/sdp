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
let sessionMetadata = null;

const senderStageTitles = {
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

const receiverStageTitles = {
  key_derivation: "J · Receiver key derivation",
  extraction: "K · Position regeneration and extraction",
  inverse_mapping: "L · Inverse mapping and reconstruction",
  authentication: "M · Authenticated decryption",
};

const senderFields = {
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
    ["Embedding verification", "verified"],
    ["Caution", "notice"],
  ],
  sender_verification: [
    ["Extracted characters", "extracted"],
    ["AES-GCM authentication", "authentication"],
    ["Recovered plaintext", "recovered_plaintext"],
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
  badge.className = `status-badge ${status}`;
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
  renderStages(senderOutput, data.stages, senderStageTitles, senderFields);
  document.getElementById("carrier-empty").classList.add("hidden");
  document.getElementById("carrier-result").classList.remove("hidden");
  carrierText.textContent = data.stages.carrier.text;
  document.getElementById("carrier-length").textContent = String(data.stages.carrier.length);
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
  document.getElementById("receiver-password").value = "";
  receivedCarrier.value = "";
  sessionMetadata = null;
  senderOutput.replaceChildren();
  senderOutput.className = "stage-grid";
  const empty = document.createElement("div");
  empty.className = "empty-state";
  empty.textContent = "Sender stages will appear here after a successful run.";
  senderOutput.append(empty);
  receiverOutput.replaceChildren();
  receiverOutput.className = "receiver-results empty-state";
  receiverOutput.textContent = "Receiver results will appear here. Plaintext is shown only after successful AES-GCM authentication.";
  document.getElementById("carrier-empty").classList.remove("hidden");
  document.getElementById("carrier-result").classList.add("hidden");
  carrierText.textContent = "";
  document.getElementById("carrier-length").textContent = "0";
  deliverButton.disabled = true;
  receiverButton.disabled = true;
  document.getElementById("channel-status").textContent = "Waiting for sender output.";
  document.getElementById("receiver-status").textContent = "Waiting for carrier delivery.";
  document.getElementById("receiver-status").className = "receiver-status pending";
  resetPipeline();
  logList.replaceChildren();
  addLog("Demo reset. No password or key material was retained.");
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    cache: "no-store",
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) {
    const error = new Error(data.error || "The request failed.");
    error.stage = data.stage;
    throw error;
  }
  return data;
}

senderForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!senderForm.reportValidity()) return;
  senderButton.disabled = true;
  senderButton.querySelector("span").textContent = "Running real sender pipeline…";
  resetPipeline();
  setOverallStatus("Sender running", "running");
  setPipelineStage("key_establishment", "running");
  document.getElementById("channel-status").textContent = "Waiting for real LLM generation to finish.";
  addLog("Sender pipeline started. Password and plaintext are not written to the execution log.");
  try {
    const data = await postJson("/api/sender", {
      message: document.getElementById("secret-message").value,
      password: document.getElementById("sender-password").value,
      topic: document.getElementById("topic").value,
      temperature: Number(document.getElementById("temperature").value),
      top_k: Number(document.getElementById("top-k").value),
    });
    showSenderResult(data);
    setOverallStatus("Sender verified · ready to deliver", "success");
    addLog("Sender completed: encryption, real embedding, extraction, and AEAD verification succeeded.");
  } catch (error) {
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
    document.getElementById("sender-password").value = "";
    senderButton.disabled = false;
    senderButton.querySelector("span").textContent = "Run complete sender";
  }
});

deliverButton.addEventListener("click", () => {
  if (!sessionMetadata) return;
  receivedCarrier.value = carrierText.textContent;
  document.getElementById("channel-status").textContent = "Carrier text delivered to the receiver editor.";
  document.getElementById("receiver-status").textContent = "Carrier delivered. Enter the receiver PSK to run extraction.";
  document.getElementById("receiver-password").focus();
  addLog("Carrier text delivered through the public-channel simulation. Session metadata stayed separate.");
});

receiverButton.addEventListener("click", async () => {
  if (!sessionMetadata) {
    document.getElementById("receiver-status").textContent = "Run the sender first to establish public session metadata.";
    return;
  }
  if (!document.getElementById("receiver-password").value) {
    document.getElementById("receiver-password").focus();
    document.getElementById("receiver-status").textContent = "Enter the receiver PSK.";
    return;
  }
  receiverButton.disabled = true;
  receiverButton.textContent = "Running receiver pipeline…";
  document.getElementById("receiver-status").textContent = "Regenerating positions and verifying the extracted payload…";
  document.getElementById("receiver-status").className = "receiver-status pending";
  addLog("Receiver started with independently derived keys; no key bytes were transferred.");
  try {
    const result = await postJson("/api/receiver", {
      carrier_text: receivedCarrier.value,
      password: document.getElementById("receiver-password").value,
      salt_hex: sessionMetadata.salt_hex,
      nonce_hex: sessionMetadata.nonce_hex,
      mapped_length: sessionMetadata.mapped_length,
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
    if (error.stage === "extraction") setPipelineStage("extraction", "failed");
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
    document.getElementById("receiver-password").value = "";
    receiverButton.disabled = false;
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

async function loadConfig() {
  try {
    const response = await fetch("/api/config", { cache: "no-store" });
    if (!response.ok) throw new Error("Configuration request failed.");
    const config = await response.json();
    document.getElementById("model-label").textContent = config.model;
    document.getElementById("kdf-label").textContent = config.pbkdf2_algorithm;
    document.getElementById("kdf-iterations").textContent = `${Number(config.pbkdf2_iterations).toLocaleString()} iterations · ${config.dk_length} derived bytes`;
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
