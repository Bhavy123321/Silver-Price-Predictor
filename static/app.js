const $ = (id) => document.getElementById(id);

function showError(msg) {
  const banner = $("errorBanner");
  const text = $("errorText");
  text.textContent = msg || "Something went wrong.";
  banner.classList.add("show");
  setTimeout(() => banner.classList.remove("show"), 3500);
}

function setLoading(on) {
  const btn = $("predictBtn");
  const spin = btn.querySelector(".btn-spin");
  const txt = btn.querySelector(".btn-text");
  btn.disabled = on;
  spin.style.display = on ? "block" : "none";
  txt.textContent = on ? "Running..." : "Run Prediction";
}

function setChartMuted(muted) {
  const card = $("chartCard");
  if (!card) return;
  card.style.opacity = muted ? "0.72" : "1";
  card.style.filter = muted ? "saturate(0.85)" : "saturate(1)";
}

/* ---------- Custom Select (works 100%) ---------- */
function setupSelect(selectEl, hiddenInputId, defaultValueLabel, defaultValueHidden) {
  const btn = selectEl.querySelector(".select-btn");
  const valueEl = selectEl.querySelector("[data-value]");
  const optionsWrap = selectEl.querySelector("[data-options]");
  const search = selectEl.querySelector(".select-search");
  const hidden = $(hiddenInputId);

  function closeAll() {
    document.querySelectorAll(".select.open").forEach(s => s.classList.remove("open"));
  }

  function setValue(label, hiddenVal) {
    valueEl.textContent = label;
    hidden.value = hiddenVal;
    optionsWrap.querySelectorAll(".opt").forEach(o => {
      o.classList.toggle("active", o.getAttribute("data-opt") === hiddenVal);
    });
  }

  // Default
  setValue(defaultValueLabel, defaultValueHidden);

  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    const willOpen = !selectEl.classList.contains("open");
    closeAll();
    if (willOpen) {
      selectEl.classList.add("open");
      if (search) {
        search.value = "";
        search.focus();
        optionsWrap.querySelectorAll(".opt").forEach(o => o.style.display = "");
      }
    }
  });

  optionsWrap.addEventListener("click", (e) => {
    const opt = e.target.closest(".opt");
    if (!opt) return;

    // For state: label==hidden
    // For purity/horizon: label differs from hidden sometimes
    const hiddenVal = opt.getAttribute("data-opt");
    const label = opt.textContent.trim();
    setValue(label, hiddenVal);

    selectEl.classList.remove("open");
  });

  if (search) {
    search.addEventListener("input", () => {
      const q = search.value.toLowerCase().trim();
      optionsWrap.querySelectorAll(".opt").forEach(o => {
        o.style.display = o.textContent.toLowerCase().includes(q) ? "" : "none";
      });
    });
  }

  document.addEventListener("click", () => selectEl.classList.remove("open"));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") selectEl.classList.remove("open");
  });
}

/* ---------- Chart ---------- */
let trendChart;

async function loadTrendChart() {
  const canvas = $("trendChart");
  if (!canvas) return;

  const res = await fetch("/api/trend");
  const out = await res.json();

  const ctx = canvas.getContext("2d");
  trendChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: out.labels,
      datasets: [{
        data: out.values,
        tension: 0.35,
        borderWidth: 2,
        pointRadius: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } }
    }
  });
}

/* ---------- Prediction ---------- */
async function runPrediction() {
  const state = $("stateInput").value;
  const purity = $("purityInput").value;
  const horizon = $("horizonInput").value;

  if (!state || !purity || !horizon) {
    showError("Please select State/UT, Purity and Horizon.");
    return;
  }

  setLoading(true);

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ state, purity, horizon })
    });

    const out = await res.json();

    if (!res.ok || !out.ok) {
      showError(out.error || "Prediction failed. Try again.");
      return;
    }

    $("resultBox").classList.add("show");

    const badge = $("dirBadge");
    badge.textContent = out.direction;
    badge.classList.remove("up", "down");
    badge.classList.add(out.direction === "UP" ? "up" : "down");

    const conf = Math.round(out.confidence);
    $("confFill").style.width = `${conf}%`;
    $("confText").textContent = `${conf}%`;

    const showPrices = $("showPrices").checked;
    $("pricesBox").style.display = showPrices ? "grid" : "none";

    $("p1g").textContent = `₹ ${out.prices["1g"]}`;
    $("p10g").textContent = `₹ ${out.prices["10g"]}`;
    $("p100g").textContent = `₹ ${out.prices["100g"]}`;

    const showTrend = $("showTrend").checked;
    $("chartCard").style.display = showTrend ? "block" : "none";

    setChartMuted(false);

  } catch (e) {
    showError("Network error. Try again.");
  } finally {
    setLoading(false);
  }
}

/* ---------- Init ---------- */
document.addEventListener("DOMContentLoaded", async () => {
  $("year").textContent = new Date().getFullYear();
  setChartMuted(true);

  // Setup selects
  setupSelect(document.querySelector('[data-select="state"]'), "stateInput", "Choose State/UT", "");
  setupSelect(document.querySelector('[data-select="purity"]'), "purityInput", "999 (Fine Silver)", "999");
  setupSelect(document.querySelector('[data-select="horizon"]'), "horizonInput", "Next Hour", "1h");

  await loadTrendChart();

  $("predictBtn").addEventListener("click", runPrediction);

  $("showTrend").addEventListener("change", (e) => {
    $("chartCard").style.display = e.target.checked ? "block" : "none";
  });
});
