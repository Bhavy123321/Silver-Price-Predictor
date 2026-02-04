// ---------- Helpers ----------
const $ = (id) => document.getElementById(id);

function showError(msg) {
  const banner = $("errorBanner");
  const text = $("errorText");
  text.textContent = msg || "Something went wrong.";
  banner.classList.add("show");
  // auto-hide
  window.clearTimeout(window.__errTimer);
  window.__errTimer = window.setTimeout(() => {
    banner.classList.remove("show");
  }, 3800);
}

function setLoading(isLoading) {
  const btn = $("predictBtn");
  const spin = btn.querySelector(".btn-spin");
  const txt = btn.querySelector(".btn-text");
  btn.disabled = isLoading;
  spin.style.display = isLoading ? "block" : "none";
  txt.textContent = isLoading ? "Running..." : "Run Prediction";
}

function setChartMuted(muted) {
  const card = $("chartCard");
  if (!card) return;
  card.style.opacity = muted ? "0.72" : "1";
  card.style.filter = muted ? "saturate(0.85)" : "saturate(1)";
}

// ---------- Custom Select ----------
function setupSelect(selectEl, hiddenInputId, defaultValue) {
  const btn = selectEl.querySelector(".select-btn");
  const valueEl = selectEl.querySelector("[data-value]");
  const menu = selectEl.querySelector(".select-menu");
  const optionsWrap = selectEl.querySelector("[data-options]");
  const search = selectEl.querySelector(".select-search");
  const hidden = $(hiddenInputId);

  function closeAll() {
    document.querySelectorAll(".select.open").forEach(s => s.classList.remove("open"));
  }

  function setValue(v) {
    hidden.value = v;
    valueEl.textContent = v;
    // active class
    optionsWrap.querySelectorAll(".opt").forEach(o => {
      o.classList.toggle("active", o.getAttribute("data-opt") === v);
    });
  }

  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    const willOpen = !selectEl.classList.contains("open");
    closeAll();
    if (willOpen) {
      selectEl.classList.add("open");
      if (search) {
        search.value = "";
        search.focus();
        // show all initially
        optionsWrap.querySelectorAll(".opt").forEach(o => o.style.display = "");
      }
    }
  });

  optionsWrap.addEventListener("click", (e) => {
    const opt = e.target.closest(".opt");
    if (!opt) return;
    setValue(opt.getAttribute("data-opt"));
    selectEl.classList.remove("open");
  });

  if (search) {
    search.addEventListener("input", () => {
      const q = search.value.toLowerCase().trim();
      optionsWrap.querySelectorAll(".opt").forEach(o => {
        const t = o.textContent.toLowerCase();
        o.style.display = t.includes(q) ? "" : "none";
      });
    });
  }

  document.addEventListener("click", () => selectEl.classList.remove("open"));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") selectEl.classList.remove("open");
  });

  // default
  if (defaultValue) setValue(defaultValue);
}

// ---------- Chart ----------
let trendChart;

async function loadTrendChart() {
  const res = await fetch("/api/trend");
  const data = await res.json();

  const ctx = $("trendChart").getContext("2d");
  trendChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: data.labels,
      datasets: [{
        label: "Silver",
        data: data.values,
        tension: 0.35,
        borderWidth: 2,
        pointRadius: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { enabled: true }
      },
      scales: {
        x: { ticks: { maxTicksLimit: 8 } },
        y: { ticks: { maxTicksLimit: 6 } }
      },
      animation: {
        duration: 700
      }
    }
  });
}

// ---------- Prediction ----------
async function runPrediction() {
  const state = $("stateInput").value;
  const purity = $("purityInput").value;
  const horizon = $("horizonInput").value;

  // UX: no error banner on load; show only on click
  if (!state || !purity || !horizon) {
    showError("Please select State/UT, Purity, and Horizon.");
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
      showError(out.error || "Something went wrong while calculating prediction. Please try again.");
      return;
    }

    // Show result box
    const box = $("resultBox");
    box.classList.add("show");

    // Direction badge
    const badge = $("dirBadge");
    badge.textContent = out.direction;
    badge.classList.remove("up", "down");
    badge.classList.add(out.direction === "UP" ? "up" : "down");

    // Confidence
    const conf = Math.round((out.confidence || 0) * 100);
    $("confFill").style.width = `${conf}%`;
    $("confText").textContent = `${conf}%`;

    // Prices toggle
    const showPrices = $("showPrices").checked;
    const pricesBox = $("pricesBox");
    pricesBox.style.display = showPrices ? "grid" : "none";

    if (out.prices) {
      $("p1g").textContent = `₹ ${out.prices["1g"]}`;
      $("p10g").textContent = `₹ ${out.prices["10g"]}`;
      $("p100g").textContent = `₹ ${out.prices["100g"]}`;
    }

    // Trend toggle / focus flow
    const showTrend = $("showTrend").checked;
    $("chartCard").style.display = showTrend ? "block" : "none";

    setChartMuted(false);
  } catch (e) {
    showError("Network error. Please try again.");
  } finally {
    setLoading(false);
  }
}

// ---------- Theme toggle (simple label change) ----------
function setupThemeToggle() {
  const t = $("themeToggle");
  const label = t?.querySelector(".toggle-text");
  if (!t) return;

  // Just a label toggle (your UI already dark; keep simple)
  t.addEventListener("click", () => {
    const current = label.textContent.trim();
    label.textContent = current === "Dark" ? "Dark" : "Dark";
  });
}

// ---------- Links ----------
function setupLinks() {
  // put your real links here
  const github = "https://github.com/";
  const linkedin = "https://www.linkedin.com/";

  ["githubLink", "githubLink2"].forEach(id => {
    const el = $(id);
    if (el) el.href = github;
  });

  ["linkedinLink", "linkedinLink2"].forEach(id => {
    const el = $(id);
    if (el) el.href = linkedin;
  });
}

// ---------- Init ----------
document.addEventListener("DOMContentLoaded", async () => {
  $("year").textContent = new Date().getFullYear();

  setupLinks();
  setupThemeToggle();

  // form defaults
  setupSelect(document.querySelector('[data-select="state"]'), "stateInput", "");
  setupSelect(document.querySelector('[data-select="purity"]'), "purityInput", "999");
  setupSelect(document.querySelector('[data-select="horizon"]'), "horizonInput", "Next Hour");

  // Initially keep chart slightly muted until prediction runs
  setChartMuted(true);

  await loadTrendChart();

  $("predictBtn").addEventListener("click", runPrediction);

  // If user toggles trend off, hide chart
  $("showTrend").addEventListener("change", (e) => {
    $("chartCard").style.display = e.target.checked ? "block" : "none";
  });
});
