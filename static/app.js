// Small client helpers (safe even if you don't use JS-heavy UI)
document.addEventListener("DOMContentLoaded", () => {
  // Auto-hide flash messages (success/error) after 3.5s
  const flashes = document.querySelectorAll(".flash-area .flash");
  if (flashes && flashes.length) {
    setTimeout(() => {
      const area = document.querySelector(".flash-area");
      if (area) area.style.display = "none";
    }, 3500);
  }
});
