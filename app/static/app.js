(function () {
    function getOverlay() {
        return document.getElementById("app-loader");
    }

    function hideLoader() {
        var overlay = getOverlay();
        if (!overlay) {
            return;
        }
        overlay.hidden = true;
        overlay.setAttribute("aria-hidden", "true");
        // Жёстко снимаем перехват кликов (редирект / bfcache / долгий USB).
        overlay.style.setProperty("display", "none", "important");
        overlay.style.setProperty("pointer-events", "none", "important");
        overlay.style.setProperty("visibility", "hidden", "important");
    }

    function showLoader(message) {
        var overlay = getOverlay();
        if (!overlay) {
            return;
        }
        var text = overlay.querySelector(".loader-text");
        if (text && message) {
            text.textContent = message;
        }
        overlay.style.removeProperty("display");
        overlay.style.removeProperty("pointer-events");
        overlay.style.removeProperty("visibility");
        overlay.hidden = false;
        overlay.setAttribute("aria-hidden", "false");
    }

    hideLoader();
    window.addEventListener("pageshow", hideLoader);

    document.querySelectorAll("[data-show-loader]").forEach(function (form) {
        form.addEventListener("submit", function () {
            if (form.getAttribute("action") === "/export/csv") {
                form.querySelectorAll('input[type="date"], input[type="hidden"][name^="date_"]').forEach(function (input) {
                    if (!input.value) {
                        input.removeAttribute("name");
                    }
                });
            }
            showLoader(form.getAttribute("data-loader-message") || "Загрузка…");
        });
    });
})();
