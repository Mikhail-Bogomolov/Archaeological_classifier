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
        overlay.hidden = false;
        overlay.setAttribute("aria-hidden", "false");
    }

    // После редиректа (ошибка камеры и т.п.) и при bfcache оверлей
    // мог остаться видимым и перехватывать клики.
    hideLoader();
    window.addEventListener("pageshow", function () {
        hideLoader();
    });

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
