const startBtn = document.getElementById("startBtn");
const rowsInput = document.getElementById("rowsInput");
const forceChk = document.getElementById("forceChk");
const btnCsv = document.getElementById("btnCsv");
const btnJson = document.getElementById("btnJson");

const stepsEl = document.getElementById("steps");
const resultEl = document.getElementById("result");
const errorEl = document.getElementById("error");
const barEl = document.getElementById("bar");

// Resumen
const rowsEl = document.getElementById("rows");
const colsEl = document.getElementById("cols");
const xcolsEl = document.getElementById("xcols");
const ycolsEl = document.getElementById("ycols");
const noteEl = document.getElementById("note");
const baseMaeEl = document.getElementById("baseMae");
const baseRmseEl = document.getElementById("baseRmse");
const baseMapeEl = document.getElementById("baseMape");
const baseR2El = document.getElementById("baseR2");

// Tablas y canvases
const modelsTable = document.getElementById("modelsTable");
const tableX = document.getElementById("tableX");
const tableY = document.getElementById("tableY");
const heatmapEl = document.getElementById("heatmap");

// Charts refs to destroy on rerun
let chartModels, chartCoefs, chartScatter, chartResiduals;

function markStepDone(idx) {
  const step = document.querySelectorAll(".step")[idx];
  if (step) step.classList.add("done");
  const progress = ((idx + 1) / document.querySelectorAll(".step").length) * 100;
  barEl.style.width = progress + "%";
}

function fillTable(table, rows) {
  if (!rows || rows.length === 0) return;
  const cols = Object.keys(rows[0]);
  const thead = table.querySelector("thead");
  const tbody = table.querySelector("tbody");
  thead.innerHTML = ""; tbody.innerHTML = "";

  const trHead = document.createElement("tr");
  cols.forEach(c => { const th = document.createElement("th"); th.textContent = c; trHead.appendChild(th); });
  thead.appendChild(trHead);

  rows.forEach(r => {
    const tr = document.createElement("tr");
    cols.forEach(c => { const td = document.createElement("td"); td.textContent = r[c]; tr.appendChild(td); });
    tbody.appendChild(tr);
  });
}

function fillModelsTable(models) {
  const thead = modelsTable.querySelector("thead");
  const tbody = modelsTable.querySelector("tbody");
  thead.innerHTML = ""; tbody.innerHTML = "";

  const headRow = document.createElement("tr");
  ["Modelo","MAE","RMSE","MAPE%","R²","CV R² (μ±σ)","CV MAE (μ±σ)"].forEach(h => {
    const th = document.createElement("th"); th.textContent = h; headRow.appendChild(th);
  });
  thead.appendChild(headRow);

  models.forEach(m => {
    const r = document.createElement("tr");
    const met = m.metrics;
    const cells = [
      m.name,
      met.mae.toFixed(3),
      met.rmse.toFixed(3),
      met.mape.toFixed(2),
      met.r2.toFixed(3),
      `${met.cv_r2_mean.toFixed(3)} ± ${met.cv_r2_std.toFixed(3)}`,
      `${met.cv_mae_mean.toFixed(3)} ± ${met.cv_mae_std.toFixed(3)}`
    ];
    cells.forEach(v => { const td = document.createElement("td"); td.textContent = v; r.appendChild(td); });
    tbody.appendChild(r);
  });
}

function renderHeatmap(labels, matrix) {
  heatmapEl.innerHTML = "";
  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  headRow.appendChild(document.createElement("th")); // corner
  labels.forEach(l => { const th = document.createElement("th"); th.textContent = l; headRow.appendChild(th); });
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  for (let i = 0; i < matrix.length; i++) {
    const tr = document.createElement("tr");
    const rowHead = document.createElement("th"); rowHead.textContent = labels[i];
    tr.appendChild(rowHead);
    for (let j = 0; j < matrix[i].length; j++) {
      const v = matrix[i][j];
      const td = document.createElement("td");
      td.textContent = v.toFixed(2);
      // color scale -1 -> rojo, 0 -> neutro, 1 -> azul
      const hue = v >= 0 ? 210 : 0; // azul / rojo
      const alpha = Math.min(1, Math.abs(v));
      td.style.background = `rgba(${hue === 210 ? "59,130,246" : "248,113,113"}, ${alpha * 0.25})`;
      tbody.appendChild(tr);
      tr.appendChild(td);
    }
  }
  table.appendChild(tbody);
  heatmapEl.appendChild(table);
}

function destroyIf(chart) { if (chart) chart.destroy(); }

async function startPipeline() {
  startBtn.disabled = true;
  errorEl.classList.add("hidden");
  resultEl.classList.add("hidden");
  stepsEl.classList.remove("hidden");
  barEl.style.width = "0%";
  document.querySelectorAll(".step").forEach(s => s.classList.remove("done"));

  const n = Math.max(1000, parseInt(rowsInput.value || "10000", 10));
  const force = forceChk.checked ? 1 : 0;
  btnCsv.href = `/download/dataset?n=${n}&force=${force}`;
  btnJson.href = `/download/results?n=${n}&force=${force}`;

  const tick = (i, d) => new Promise(res => setTimeout(() => { markStepDone(i); res(); }, d));
  await tick(0, 250); await tick(1, 200); await tick(2, 200); await tick(3, 200);

  try {
    const resp = await fetch(`/start?n=${n}&force=${force}`);
    const data = await resp.json();
    await tick(4, 200);

    if (!data.ok) throw new Error(data.error || "Error desconocido");
    const r = data.result;

    // Resumen
    rowsEl.textContent = r.dataset_info.rows;
    colsEl.textContent = r.dataset_info.cols;
    xcolsEl.textContent = r.dataset_info.X_cols.join(", ");
    ycolsEl.textContent = r.dataset_info.Y_cols.join(", ");
    noteEl.textContent = r.dataset_info.note;

    baseMaeEl.textContent = r.baseline.mae.toFixed(3);
    baseRmseEl.textContent = r.baseline.rmse.toFixed(3);
    baseMapeEl.textContent = r.baseline.mape.toFixed(2);
    baseR2El.textContent = r.baseline.r2.toFixed(3);

    // Tabla de modelos
    fillModelsTable(r.models);

    // Chart modelos (R²)
    destroyIf(chartModels);
    const ctxM = document.getElementById("chartModels").getContext("2d");
    chartModels = new Chart(ctxM, {
      type: "bar",
      data: {
        labels: r.models.map(m => m.name),
        datasets: [{ label: "R²", data: r.models.map(m => m.metrics.r2) }]
      },
      options: { responsive: true, scales: { y: { beginAtZero: true, max: 1 } } }
    });

    // Mejor modelo por R²
    const best = r.models.reduce((a,b)=> a.metrics.r2 > b.metrics.r2 ? a : b);

    // Coefs / Importancia
    destroyIf(chartCoefs);
    const ctxC = document.getElementById("chartCoefs").getContext("2d");
    const obj = best.feature_importance || best.coefs || {};
    chartCoefs = new Chart(ctxC, {
      type: "bar",
      data: { labels: Object.keys(obj), datasets: [{ label: best.feature_importance ? "Importancia" : "Coeficientes", data: Object.values(obj) }] },
      options: { responsive: true }
    });

    // Scatter Real vs Predicho
    destroyIf(chartScatter);
    const ctxS = document.getElementById("chartScatter").getContext("2d");
    chartScatter = new Chart(ctxS, {
      type: "scatter",
      data: { datasets: [{ label: "Puntos", pointRadius: 2, data: best.y_true.map((y,i)=>({x:y, y:best.y_pred[i]})) }] },
      options: { plugins:{legend:{display:false}}, scales: { x:{ title:{display:true, text:"Real"}, min:1, max:5 }, y:{ title:{display:true, text:"Predicho"}, min:1, max:5 } } }
    });

    // Histograma residuales
    destroyIf(chartResiduals);
    const res = best.residuals;
    const bins = 30, min = Math.min(...res), max = Math.max(...res);
    const step = (max - min) / bins || 1;
    const hist = Array(bins).fill(0);
    res.forEach(v => { const i = Math.min(bins-1, Math.max(0, Math.floor((v-min)/step))); hist[i]++; });
    const ctxR = document.getElementById("chartResiduals").getContext("2d");
    chartResiduals = new Chart(ctxR, {
      type: "bar",
      data: { labels: hist.map((_,i)=> (min + i*step).toFixed(2)), datasets: [{ label: "Frecuencia", data: hist }] },
      options: { responsive: true }
    });

    // Heatmap correlación (tabla coloreada)
    renderHeatmap(r.corr.labels, r.corr.matrix);

    // Tablas X / Y
    fillTable(tableX, r.preview_X);
    fillTable(tableY, r.preview_Y);

    resultEl.classList.remove("hidden");
  } catch (err) {
    errorEl.textContent = "⚠️ " + err.message;
    errorEl.classList.remove("hidden");
  } finally {
    startBtn.disabled = false;
  }
}

startBtn.addEventListener("click", startPipeline);
