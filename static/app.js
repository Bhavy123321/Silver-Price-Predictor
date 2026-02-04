// Custom dropdown logic (smooth + search + click outside close)
(function () {
  const dropdowns = document.querySelectorAll(".dd");

  function closeAll(except = null) {
    dropdowns.forEach(dd => {
      if (dd !== except) dd.classList.remove("open");
    });
  }

  dropdowns.forEach(dd => {
    const btn = dd.querySelector(".dd-btn");
    const panel = dd.querySelector(".dd-panel");
    const list = dd.querySelector(".dd-list");
    const items = dd.querySelectorAll(".dd-item");
    const search = dd.querySelector(".dd-search-input");

    btn.addEventListener("click", () => {
      const isOpen = dd.classList.contains("open");
      closeAll(dd);
      dd.classList.toggle("open", !isOpen);

      if (!isOpen && search) {
        search.value = "";
        filterItems(items, "");
        setTimeout(() => search.focus(), 60);
      }
    });

    items.forEach(item => {
      item.addEventListener("click", () => {
        const type = dd.getAttribute("data-dd");
        const value = item.getAttribute("data-value");
        const text = item.textContent.trim();

        // highlight active
        items.forEach(i => i.classList.remove("active"));
        item.classList.add("active");

        // Update visible label + hidden input used by Flask
        if (type === "state") {
          document.getElementById("stateLabel").textContent = text;
          document.getElementById("stateHidden").value = value;
        }
        if (type === "horizon") {
          document.getElementById("horizonLabel").textContent = text;
          document.getElementById("horizonHidden").value = value;
        }
        if (type === "purity") {
          document.getElementById("purityLabel").textContent = text;
          document.getElementById("purityHidden").value = value;
        }

        dd.classList.remove("open");
      });
    });

    if (search) {
      search.addEventListener("input", (e) => {
        filterItems(items, e.target.value);
      });
    }
  });

  function filterItems(items, q) {
    const query = (q || "").toLowerCase().trim();
    items.forEach(it => {
      const show = it.textContent.toLowerCase().includes(query);
      it.style.display = show ? "block" : "none";
    });
  }

  // Close on outside click
  document.addEventListener("click", (e) => {
    const isInside = e.target.closest(".dd");
    if (!isInside) closeAll();
  });

  // ESC closes dropdown
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeAll();
  });
})();
