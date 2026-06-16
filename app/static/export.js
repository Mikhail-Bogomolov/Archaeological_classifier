(function () {
    var MONTHS = [
        "Январь", "Февраль", "Март", "Апрель", "Май", "Июнь",
        "Июль", "Август", "Сентябрь", "Октябрь", "Ноябрь", "Декабрь"
    ];
    var WEEKDAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"];

    var bounds = window.EXPORT_BOUNDS || {};
    var startDate = null;
    var endDate = null;
    var leftMonth = new Date();
    var rightMonth = new Date();

    function parseIso(s) {
        if (!s) return null;
        var p = s.split("-");
        return new Date(Number(p[0]), Number(p[1]) - 1, Number(p[2]));
    }

    function toIso(d) {
        var y = d.getFullYear();
        var m = String(d.getMonth() + 1).padStart(2, "0");
        var day = String(d.getDate()).padStart(2, "0");
        return y + "-" + m + "-" + day;
    }

    function toDisplay(d) {
        var day = String(d.getDate()).padStart(2, "0");
        var m = String(d.getMonth() + 1).padStart(2, "0");
        return day + "." + m + "." + d.getFullYear();
    }

    function sameDay(a, b) {
        return a && b &&
            a.getFullYear() === b.getFullYear() &&
            a.getMonth() === b.getMonth() &&
            a.getDate() === b.getDate();
    }

    function inRange(d) {
        if (!startDate || !endDate) return false;
        var t = d.getTime();
        return t >= startDate.getTime() && t <= endDate.getTime();
    }

    function clampDate(d) {
        var min = parseIso(bounds.min);
        var max = parseIso(bounds.max);
        if (min && d < min) return new Date(min);
        if (max && d > max) return new Date(max);
        return d;
    }

    function isDisabled(d) {
        var min = parseIso(bounds.min);
        var max = parseIso(bounds.max);
        if (min && d < min) return true;
        if (max && d > max) return true;
        return false;
    }

    function daysBetween(a, b) {
        var ms = Math.abs(b.getTime() - a.getTime());
        return Math.floor(ms / 86400000) + 1;
    }

    function updateFields() {
        var fromInput = document.getElementById("date_from");
        var toInput = document.getElementById("date_to");
        var dispFrom = document.getElementById("display_from");
        var dispTo = document.getElementById("display_to");
        var summary = document.getElementById("export-summary");

        if (startDate) {
            fromInput.value = toIso(startDate);
            dispFrom.value = toDisplay(startDate);
        } else {
            fromInput.value = "";
            dispFrom.value = "";
        }
        if (endDate) {
            toInput.value = toIso(endDate);
            dispTo.value = toDisplay(endDate);
        } else {
            toInput.value = "";
            dispTo.value = "";
        }

        if (startDate && endDate) {
            var a = startDate < endDate ? startDate : endDate;
            var b = startDate < endDate ? endDate : startDate;
            summary.textContent =
                "Выбрано: " + toDisplay(a) + " – " + toDisplay(b) +
                " (" + daysBetween(a, b) + " " + pluralDays(daysBetween(a, b)) + ")";
        } else if (startDate) {
            summary.textContent = "Выбрано начало: " + toDisplay(startDate) + ". Укажите конец периода.";
        } else {
            summary.textContent = "Оставьте период пустым и скачайте — выгрузятся все объекты.";
        }
    }

    function pluralDays(n) {
        var m10 = n % 10;
        var m100 = n % 100;
        if (m10 === 1 && m100 !== 11) return "день";
        if (m10 >= 2 && m10 <= 4 && (m100 < 10 || m100 >= 20)) return "дня";
        return "дней";
    }

    function onDayClick(d, calRole) {
        d = clampDate(d);
        if (isDisabled(d)) return;

        if (calRole === "start") {
            startDate = new Date(d);
            if (endDate && startDate > endDate) {
                endDate = new Date(startDate);
            }
        } else {
            endDate = new Date(d);
            if (startDate && endDate < startDate) {
                startDate = new Date(endDate);
            }
        }
        updateFields();
        renderCalendars();
    }

    function buildCalendar(container, monthDate, calRole) {
        container.innerHTML = "";
        var year = monthDate.getFullYear();
        var month = monthDate.getMonth();

        var head = document.createElement("div");
        head.className = "mini-cal-head";
        var prev = document.createElement("button");
        prev.type = "button";
        prev.className = "mini-cal-nav";
        prev.textContent = "‹";
        prev.addEventListener("click", function () {
            if (calRole === "start") {
                leftMonth = new Date(leftMonth.getFullYear(), leftMonth.getMonth() - 1, 1);
            } else {
                rightMonth = new Date(rightMonth.getFullYear(), rightMonth.getMonth() - 1, 1);
            }
            renderCalendars();
        });
        var title = document.createElement("div");
        title.className = "mini-cal-title";
        title.textContent = MONTHS[month] + " " + year;
        var next = document.createElement("button");
        next.type = "button";
        next.className = "mini-cal-nav";
        next.textContent = "›";
        next.addEventListener("click", function () {
            if (calRole === "start") {
                leftMonth = new Date(leftMonth.getFullYear(), leftMonth.getMonth() + 1, 1);
            } else {
                rightMonth = new Date(rightMonth.getFullYear(), rightMonth.getMonth() + 1, 1);
            }
            renderCalendars();
        });
        head.appendChild(prev);
        head.appendChild(title);
        head.appendChild(next);
        container.appendChild(head);

        var weekdays = document.createElement("div");
        weekdays.className = "mini-cal-weekdays";
        WEEKDAYS.forEach(function (w) {
            var s = document.createElement("span");
            s.textContent = w;
            weekdays.appendChild(s);
        });
        container.appendChild(weekdays);

        var daysGrid = document.createElement("div");
        daysGrid.className = "mini-cal-days";

        var first = new Date(year, month, 1);
        var startWd = (first.getDay() + 6) % 7;
        var daysInMonth = new Date(year, month + 1, 0).getDate();

        for (var i = 0; i < startWd; i++) {
            var empty = document.createElement("button");
            empty.type = "button";
            empty.className = "mini-cal-day";
            empty.disabled = true;
            daysGrid.appendChild(empty);
        }

        for (var day = 1; day <= daysInMonth; day++) {
            var d = new Date(year, month, day);
            var btn = document.createElement("button");
            btn.type = "button";
            btn.className = "mini-cal-day";
            btn.textContent = String(day);

            if (isDisabled(d)) {
                btn.disabled = true;
                btn.style.color = "#d1d5db";
            } else {
                if (sameDay(d, startDate)) btn.classList.add("is-start");
                if (sameDay(d, endDate)) btn.classList.add("is-end");
                if (inRange(d) && !sameDay(d, startDate) && !sameDay(d, endDate)) {
                    btn.classList.add("is-in-range");
                }
                btn.addEventListener("click", (function (dateCopy) {
                    return function () { onDayClick(dateCopy, calRole); };
                })(d));
            }
            daysGrid.appendChild(btn);
        }

        container.appendChild(daysGrid);
    }

    function renderCalendars() {
        var row = document.getElementById("calendars-row");
        if (!row) return;
        row.innerHTML = "";

        var left = document.createElement("div");
        left.className = "mini-cal";
        var right = document.createElement("div");
        right.className = "mini-cal";

        buildCalendar(left, leftMonth, "start");
        buildCalendar(right, rightMonth, "end");

        row.appendChild(left);
        row.appendChild(right);
    }

    function initDefaults() {
        var min = parseIso(bounds.min);
        var max = parseIso(bounds.max);
        if (min && max) {
            startDate = new Date(min);
            endDate = new Date(max);
            leftMonth = new Date(min.getFullYear(), min.getMonth(), 1);
            rightMonth = new Date(max.getFullYear(), max.getMonth(), 1);
        } else {
            var now = new Date();
            leftMonth = new Date(now.getFullYear(), now.getMonth(), 1);
            rightMonth = new Date(now.getFullYear(), now.getMonth(), 1);
        }
    }

    document.addEventListener("DOMContentLoaded", function () {
        initDefaults();
        updateFields();
        renderCalendars();

        var form = document.getElementById("export-form");
        if (form) {
            form.addEventListener("submit", function () {
                if (!startDate) {
                    document.getElementById("date_from").removeAttribute("name");
                }
                if (!endDate) {
                    document.getElementById("date_to").removeAttribute("name");
                }
            });
        }

        var clearBtn = document.getElementById("clear-range");
        if (clearBtn) {
            clearBtn.addEventListener("click", function () {
                startDate = null;
                endDate = null;
                updateFields();
                renderCalendars();
            });
        }
    });
})();
