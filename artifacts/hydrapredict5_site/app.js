async function loadReport() {
  const response = await fetch("./data/report.json");
  if (!response.ok) {
    throw new Error(`Failed to load report.json (${response.status})`);
  }
  return response.json();
}

function pct(v) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "N/A";
  return `${(Number(v) * 100).toFixed(2)}%`;
}

function num(v, d = 2) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "N/A";
  return Number(v).toFixed(d);
}

function money(v) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "N/A";
  return Number(v).toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 });
}

function verdictClass(status) {
  const s = String(status || "").toUpperCase();
  if (s === "CLEARED") return "good";
  if (s === "CANDIDATE") return "warn";
  return "bad";
}

function addKeyCards(data) {
  const root = document.getElementById("key-cards");
  const oos = data.oos_metrics || {};
  const isMetrics = data.is_metrics || {};
  const verdict = data.verdict || {};

  const cards = [
    ["OOS Sharpe", num(oos.sharpe, 2)],
    ["OOS Max Drawdown", pct(oos.max_drawdown)],
    ["OOS Annual Return", pct(oos.annualized_return)],
    ["IS Sharpe", num(isMetrics.sharpe, 2)],
    ["OOS Ending Equity", money(oos.ending_equity)],
    ["Verdict", String(verdict.status || "N/A")],
  ];

  cards.forEach(([label, value]) => {
    const card = document.createElement("article");
    card.className = "card";
    const h = document.createElement("h3");
    h.textContent = label;
    const v = document.createElement("div");
    v.className = `value ${label === "Verdict" ? verdictClass(value) : ""}`;
    v.textContent = value;
    card.appendChild(h);
    card.appendChild(v);
    root.appendChild(card);
  });
}

function setConfig(data) {
  const root = document.getElementById("config-grid");
  const items = [
    ["Product", data.product_name],
    ["Timeframe", data.timeframe],
    ["Engine", data.engine_label],
    ["Universe Mode", data.universe_mode],
    ["Universe", (data.universe || []).join(", ")],
    ["Benchmark", data.benchmark_symbol],
    ["Train Period", `${data.train_period?.start || "N/A"} to ${data.train_period?.end || "N/A"}`],
    ["Test Period", `${data.test_period?.start || "N/A"} to ${data.test_period?.end || "N/A"}`],
    ["Generated UTC", data.generated_utc],
  ];

  items.forEach(([k, v]) => {
    const box = document.createElement("div");
    box.className = "kv";
    box.innerHTML = `<div class="k">${k}</div><div class="v">${v || "N/A"}</div>`;
    root.appendChild(box);
  });
}

function renderChart(data) {
  const oos = data.oos_metrics || {};
  const isMetrics = data.is_metrics || {};
  const benchmarkOos = data.benchmark_metrics?.out_of_sample || {};
  const ctx = document.getElementById("metrics-chart");
  new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["IS Sharpe", "OOS Sharpe", "Benchmark OOS Sharpe", "OOS Return %", "OOS MDD %"],
      datasets: [{
        label: "HydraPredict 5",
        data: [
          Number(isMetrics.sharpe || 0),
          Number(oos.sharpe || 0),
          Number(benchmarkOos.sharpe || 0),
          Number(oos.total_return || 0) * 100.0,
          Number(oos.max_drawdown || 0) * 100.0,
        ],
        backgroundColor: ["#6ea8fe", "#1f6fd8", "#8a96aa", "#1b9e77", "#cc3d3d"],
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { y: { ticks: { callback: (val) => `${val}` } } },
    },
  });
}

async function setReportText(assetPath) {
  const el = document.getElementById("report-text");
  if (!assetPath) {
    el.textContent = "No text report bundled.";
    return;
  }
  try {
    const resp = await fetch(`./${assetPath}`);
    if (!resp.ok) {
      el.textContent = "Text report could not be loaded.";
      return;
    }
    el.textContent = await resp.text();
  } catch (_err) {
    el.textContent = "Text report could not be loaded.";
  }
}

async function setDiffText(assetPath) {
  const el = document.getElementById("diff-text");
  if (!assetPath) {
    el.textContent = "No diff artifact bundled.";
    return;
  }
  try {
    const resp = await fetch(`./${assetPath}`);
    if (!resp.ok) {
      el.textContent = "Diff artifact could not be loaded.";
      return;
    }
    el.textContent = await resp.text();
  } catch (_err) {
    el.textContent = "Diff artifact could not be loaded.";
  }
}

function setDownloads(assets) {
  const root = document.getElementById("downloads");
  const links = [
    ["Report TXT", assets.report_txt],
    ["Report PDF", assets.report_pdf],
    ["Curves PNG", assets.curves_png],
    ["Report Diff", assets.report_diff],
  ];
  links.forEach(([label, href]) => {
    if (!href) return;
    const a = document.createElement("a");
    a.href = `./${href}`;
    a.textContent = label;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    root.appendChild(a);
  });
  if (!root.children.length) {
    root.textContent = "No downloadable assets bundled.";
  }
}

function setCurvesImage(assets) {
  const img = document.getElementById("curves-image");
  const missing = document.getElementById("curves-missing");
  if (!assets.curves_png) {
    img.classList.add("hidden");
    missing.classList.remove("hidden");
    return;
  }
  img.src = `./${assets.curves_png}`;
  img.classList.remove("hidden");
  missing.classList.add("hidden");
}

function setHeader(data) {
  const verdict = data.verdict || {};
  const oos = data.oos_metrics || {};
  document.getElementById("title").textContent = `${data.product_name} Backtest Portal`;
  document.getElementById("subtitle").textContent =
    `OOS Sharpe ${num(oos.sharpe, 2)} | OOS MDD ${pct(oos.max_drawdown)} | Verdict ${String(verdict.status || "N/A")}`;
}

function setRawJson(data) {
  document.getElementById("raw-json").textContent = JSON.stringify(data.payload_sanitized || {}, null, 2);
}

async function main() {
  try {
    const data = await loadReport();
    setHeader(data);
    addKeyCards(data);
    setConfig(data);
    renderChart(data);
    setCurvesImage(data.assets || {});
    await setReportText((data.assets || {}).report_txt);
    await setDiffText((data.assets || {}).report_diff);
    setDownloads(data.assets || {});
    setRawJson(data);
  } catch (err) {
    document.getElementById("subtitle").textContent = `Failed to load report: ${err}`;
  }
}

main();
