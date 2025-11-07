const startBtn = document.getElementById("startBtn");
const stepsEl = document.getElementById("steps");
const resultEl = document.getElementById("result");
const errorEl = document.getElementById("error");
const barEl = document.getElementById("bar");

const rowsEl = document.getElementById("rows");
const colsEl = document.getElementById("cols");
const featuresEl = document.getElementById("features");
const targetEl = document.getElementById("target");
const mseEl = document.getElementById("mse");
const r2El = document.getElementById("r2");
const interceptEl = document.getElementById("intercept");
const coefsEl = document.getElementById("coefs");
const tableEl = document.getElementById("previewTable");
const rmseUsdEl = document.getElementById("rmseUsd");
const unitsNoteEl = document.getElementById("unitsNote");

const fmtUSD0 = (n) =>
  Number(n).toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });

function markStepDone(idx) {
  const step = document.querySelectorAll(".step")[idx];
  if (step) step.classList.add("done");
  const progress = ((idx + 1) / document.querySelectorAll(".step").length) * 100;
  barEl.style.width = progress + "%";
}

function fillPreviewTable(rows) {
  if (!rows || rows.length === 0) return;
  const cols = Object.keys(rows[0]);

  const thead = tableEl.querySelector("thead");
  const tbody = tableEl.querySelector("tbody");
  thead.innerHTML = "";
  tbody.innerHTML = "";

  // Header
  const trHead = document.createElement("tr");
  cols.forEach((c) => {
    const th = document.createElement("th");
    th.textContent = c;
    trHead.appendChild(th);
  });
  thead.appendChild(trHead);

  // Body
  rows.forEach((r) => {
    const tr = document.createElement("tr");
    cols.forEach((c) => {
      const td = document.createElement("td");
      let v = r[c];

      // Formatear dólares solo para la columna convertida
      if (c === "MedHouseVal_USD") v = fmtUSD0(v);
      tr.appendChild(Object.assign(td, { textContent: v }));
    });
    tbody.appendChild(tr);
  });
}

async function startPipeline() {
  startBtn.disabled = true;
  errorEl.classList.add("hidden");
  resultEl.classList.add("hidden");
  stepsEl.classList.remove("hidden");
  barEl.style.width = "0%";
  document.querySelectorAll(".step").forEach((s) => s.classList.remove("done"));

  // Animación de pasos (tiempos simulados)
  const tick = (i, delay) =>
    new Promise((res) => setTimeout(() => { markStepDone(i); res(); }, delay));

  await tick(0, 600);
  await tick(1, 500);
  await tick(2, 500);
  await tick(3, 700);

  try {
    const resp = await fetch("/start");
    const data = await resp.json();
    await tick(4, 500);

    if (!data.ok) throw new Error(data.error || "Error desconocido");
    const r = data.result;

    // Dataset info
    rowsEl.textContent = r.dataset_info.rows;
    colsEl.textContent = r.dataset_info.cols;
    featuresEl.textContent = r.dataset_info.features.join(", ");
    targetEl.textContent = r.dataset_info.target;
    unitsNoteEl.textContent = r.dataset_info.units_note;

    // Métricas
    mseEl.textContent = r.metrics.mse_native.toFixed(4);
    rmseUsdEl.textContent = fmtUSD0(r.metrics.rmse_usd);
    r2El.textContent = r.metrics.r2.toFixed(3);
    interceptEl.textContent = r.metrics.intercept.toFixed(4);

    // Coefs
    coefsEl.innerHTML = "";
    Object.entries(r.metrics.coefs).forEach(([k, v]) => {
      const div = document.createElement("div");
      div.className = "pill";
      div.textContent = `${k}: ${v.toFixed(4)}`;
      coefsEl.appendChild(div);
    });

    // Preview
    fillPreviewTable(r.preview);

    resultEl.classList.remove("hidden");
  } catch (err) {
    errorEl.textContent = "⚠️ " + err.message;
    errorEl.classList.remove("hidden");
  } finally {
    startBtn.disabled = false;
  }
}

startBtn.addEventListener("click", startPipeline);
