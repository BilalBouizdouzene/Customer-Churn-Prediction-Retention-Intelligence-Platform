const form = document.querySelector("#customer-form");
const customerSelect = document.querySelector("#customer-select");
const predictionCard = document.querySelector("#prediction-card");
const formError = document.querySelector("#form-error");
const localDrivers = document.querySelector("#local-drivers");
const localCustomerLabel = document.querySelector("#local-customer-label");
const queueBody = document.querySelector("#queue-table tbody");
const queueCount = document.querySelector("#queue-count");

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#039;", '"': "&quot;",
  }[character]));
}

function humanFeature(name) {
  return String(name || "")
    .replace("num__", "")
    .replace("cat__", "")
    .replaceAll("_", " ")
    .replaceAll("-", " ");
}

function setFormValues(values) {
  Object.entries(values).forEach(([key, value]) => {
    const field = form.elements.namedItem(key);
    if (!field || value === null || value === undefined) return;
    field.value = value;
  });
}

function showDrivers(result) {
  const positives = result.top_risk_drivers || [];
  const negatives = result.protective_factors || [];
  const positiveRows = positives.map((item) => `<li class="driver-positive"><span>+</span>${escapeHtml(humanFeature(item.feature))}<b>${Number(item.shap_value).toFixed(3)}</b></li>`).join("");
  const negativeRows = negatives.map((item) => `<li class="driver-negative"><span>−</span>${escapeHtml(humanFeature(item.feature))}<b>${Number(item.shap_value).toFixed(3)}</b></li>`).join("");
  localDrivers.innerHTML = `<div class="driver-group"><h4>Risk drivers</h4><ul>${positiveRows || "<li class='muted-cell'>Aucun facteur positif identifié.</li>"}</ul></div><div class="driver-group"><h4>Protective factors</h4><ul>${negativeRows || "<li class='muted-cell'>Aucun facteur protecteur identifié.</li>"}</ul></div>`;
}

function showPrediction(result) {
  predictionCard.classList.remove("empty-state");
  predictionCard.innerHTML = `<div class="prediction-top"><div><p class="eyebrow">PREDICTION RESULT</p><div class="risk-number">${(result.churn_probability * 100).toFixed(1)}<small>%</small></div></div><span class="pill ${String(result.risk_level).toLowerCase()}">${escapeHtml(result.risk_level)}</span></div><div class="prediction-meta"><div><span>Retention priority</span><b>${Number(result.retention_priority).toFixed(1)}</b></div><div><span>Operating threshold</span><b>${Number(result.threshold).toFixed(2)}</b></div></div><div class="recommendation"><span>Recommended action</span><p>${escapeHtml(result.recommended_action)}</p></div>`;
  showDrivers(result);
  localCustomerLabel.textContent = "Latest prediction";
  document.querySelector("#api-result").textContent = `Last response: ${result.risk_level} · ${(result.churn_probability * 100).toFixed(1)}% · ${result.recommended_action}`;
}

function numericPayload(payload) {
  ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"].forEach((field) => {
    if (payload[field] !== undefined && payload[field] !== "") payload[field] = Number(payload[field]);
  });
  return payload;
}

async function submitPrediction(event) {
  event.preventDefault();
  formError.textContent = "";
  const payload = numericPayload(Object.fromEntries(new FormData(form).entries()));
  try {
    const response = await fetch("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "La requête a échoué.");
    showPrediction(data);
  } catch (error) {
    formError.textContent = error.message;
  }
}

async function loadCustomer(customerId) {
  if (!customerId) return;
  const response = await fetch(`/api/customer/${encodeURIComponent(customerId)}`);
  const data = await response.json();
  if (!response.ok) return;
  setFormValues(data.customer);
  if (data.score) {
    localCustomerLabel.textContent = `${data.score.customer_id} · existing score`;
  }
  form.requestSubmit();
}

function queueRow(row) {
  const risk = String(row.risk_level || "").toLowerCase();
  return `<tr><td><b>${Number(row.priority_score).toFixed(1)}</b></td><td><b>${escapeHtml(row.customer_id)}</b></td><td class="probability">${(Number(row.churn_probability) * 100).toFixed(1)}%</td><td><span class="pill ${risk}">${escapeHtml(row.risk_level)}</span></td><td>${escapeHtml(row.Contract || "—")}</td><td>${escapeHtml(row.InternetService || "—")}</td><td class="muted-cell">${escapeHtml(row.recommended_action)}</td></tr>`;
}

async function loadQueue() {
  const params = new URLSearchParams({ limit: "1000" });
  const risk = document.querySelector("#risk-filter").value;
  const contract = document.querySelector("#contract-filter").value;
  const internet = document.querySelector("#internet-filter").value;
  if (risk) params.set("risk", risk);
  if (contract) params.set("contract", contract);
  if (internet) params.set("internet", internet);
  const response = await fetch(`/api/scores?${params.toString()}`);
  const rows = await response.json();
  queueBody.innerHTML = rows.map(queueRow).join("") || `<tr><td colspan="7" class="empty-table">Aucun client ne correspond à ces filtres.</td></tr>`;
  queueCount.textContent = `${rows.length} client${rows.length > 1 ? "s" : ""}`;
}

async function loadCustomerOptions() {
  const response = await fetch("/api/scores?limit=7043");
  const rows = await response.json();
  rows.slice(0, 250).forEach((row) => {
    const option = document.createElement("option");
    option.value = row.customer_id;
    option.textContent = `${row.customer_id} · ${(Number(row.churn_probability) * 100).toFixed(1)}% · ${row.risk_level}`;
    customerSelect.appendChild(option);
  });
}

form.addEventListener("submit", submitPrediction);
customerSelect.addEventListener("change", (event) => loadCustomer(event.target.value));
["risk-filter", "contract-filter", "internet-filter"].forEach((id) => document.querySelector(`#${id}`).addEventListener("change", loadQueue));

loadQueue();
loadCustomerOptions();
