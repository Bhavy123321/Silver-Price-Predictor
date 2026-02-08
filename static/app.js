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
