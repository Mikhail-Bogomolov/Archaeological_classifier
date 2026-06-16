(function () {
    function showLoader(message) {
        var overlay = document.getElementById("app-loader");
        if (!overlay) {
            return;
        }
        var text = overlay.querySelector(".loader-text");
        if (text && message) {
            text.textContent = message;
        }
        overlay.hidden = false;
    }

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
