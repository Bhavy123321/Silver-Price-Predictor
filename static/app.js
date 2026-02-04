// ========== THEME TOGGLE ==========
(function initTheme() {
  const root = document.documentElement;
  const saved = localStorage.getItem("theme");
  if (saved === "light" || saved === "dark") root.dataset.theme = saved;

  const btn = document.getElementById("themeToggle");
  const icon = document.getElementById("themeIcon");
  const text = document.getElementById("themeText");

  function sync() {
    const t = root.dataset.theme || "dark";
    if (icon) icon.textContent = (t === "dark") ? "🌙" : "☀️";
    if (text) text.textContent = (t === "dark") ? "Dark" : "Light";
  }
  sync();

  if (btn) {
    btn.addEventListener("click", () => {
      const next = (root.dataset.theme === "dark") ? "light" : "dark";
      root.dataset.theme = next;
      localStorage.setItem("theme", next);
      sync();
    });
  }
})();

// ========== CUSTOM DROPDOWNS ==========
(function initDropdowns() {
  function closeAll(except) {
    document.querySelectorAll(".dd.open").forEach(dd => {
      if (dd !== except) dd.classList.remove("open");
    });
  }

  document.querySelectorAll(".dd").forEach(dd => {
    const targetId = dd.getAttribute("data-target");
    const hidden = document.getElementById(targetId);
    const btn = dd.querySelector(".dd-btn");
    const label = dd.querySelector(".dd-value");

    btn.addEventListener("click", (e) => {
      e.preventDefault();
      const open = dd.classList.contains("open");
      closeAll(dd);
      dd.classList.toggle("open", !open);
    });

    dd.querySelectorAll(".dd-item").forEach(item => {
      item.addEventListener("click", () => {
        const val = item.getAttribute("data-value");
        if (hidden) hidden.value = val;
        if (label) label.textContent = item.textContent.trim();
        dd.classList.remove("open");
      });
    });
  });

  document.addEventListener("click", (e) => {
    const inside = e.target.closest(".dd");
    if (!inside) closeAll(null);
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeAll(null);
  });
})();
