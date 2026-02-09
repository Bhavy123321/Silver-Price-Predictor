(function () {
  const root = document.documentElement;

  function applyTheme(theme) {
    root.setAttribute("data-theme", theme);

    const icon = document.getElementById("themeIcon");
    const text = document.getElementById("themeText");

    if (icon && text) {
      if (theme === "dark") {
        icon.textContent = "☀️";
        text.textContent = "Light";
      } else {
        icon.textContent = "🌙";
        text.textContent = "Dark";
      }
    }
  }

  function getSavedTheme() {
    const saved = localStorage.getItem("sp_theme");
    if (saved === "light" || saved === "dark") return saved;
    return "light"; // default
  }

  document.addEventListener("DOMContentLoaded", () => {
    // Apply saved/default theme
    applyTheme(getSavedTheme());

    // Toggle button
    const btn = document.getElementById("themeToggle");
    if (btn) {
      btn.addEventListener("click", () => {
        const current = root.getAttribute("data-theme") || "light";
        const next = current === "dark" ? "light" : "dark";
        localStorage.setItem("sp_theme", next);
        applyTheme(next);
      });
    }
  });
})();
