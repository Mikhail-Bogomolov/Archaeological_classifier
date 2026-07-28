"""Нормализация значений в Excel — по «Маппинг признаков.md»."""

from __future__ import annotations

import re
from pathlib import Path

NOT_SPECIFIED_VALUES = frozenset({
    "не указано",
    "не определимо",
    "не нужны",
    "не нужен",
    "",
    "nan",
    "none",
    "—",
    "-",
})

# Префиксы предметов, где сеть 1 чаще путает классы (уд85_4-2, уд85_14-8 …).
CONFUSED_ITEM_PREFIXES: tuple[str, ...] = (
    "уд85_4-2",
    "уд85_14-8",
    "уд85_8-26",
    "уд85_8-49",
    "уд85_14-11",
    "уд85_4-15",
    "уд85_8-62",
    "уд85_25-7",
    "уд85_14-2",
    "уд85_42-42",
)

# Строки «класс» в таблицах, которые не относятся к целевому типу объекта.
EXCLUDED_ROW_CLASSES: dict[str, frozenset[str]] = {
    "удила": frozenset({"псалии", "подвеска"}),
    "наконечники стрел": frozenset({"копье", "копьё", "наконечник копья"}),
    "кельты": frozenset({"литейная форма"}),
}

# Колонки, которые убираем из таблиц / схемы (неинформативные).
DROP_FEATURE_COLUMNS: dict[str, frozenset[str]] = {
    "кельты": frozenset({"форма"}),
    "накладки": frozenset({"материал"}),
    "удила": frozenset({"шарнирность"}),
    "наконечники стрел": frozenset({"ребра"}),
}


def _clean(text: object) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def is_not_specified(text: object) -> bool:
    return _clean(text) in NOT_SPECIFIED_VALUES


def is_excluded_row_class(table_class: str, row_class: object) -> bool:
    """True, если строка в таблице не должна участвовать в обучении."""
    text = _clean(row_class)
    if not text:
        return False
    excluded = EXCLUDED_ROW_CLASSES.get(table_class, frozenset())
    if text in excluded:
        return True
    # «наконечник копья» / «копьевидный …» как тип объекта в колонке класс
    if table_class == "наконечники стрел" and ("копь" in text or "копье" in text):
        return True
    return False


def coarse_sohrannost(raw: object, class_name: str | None = None) -> str | None:
    """Сохранность → целый / сломан (по маппингу для всех классов)."""
    text = _clean(raw)
    if not text or text in NOT_SPECIFIED_VALUES:
        return None

    has_whole = bool(re.search(r"\bцел", text))
    has_broken = any(
        x in text
        for x in (
            "сломан",
            "обломан",
            "облом",
            "отлом",
            "фрагмент",
            "погнут",
            "половин",
        )
    )
    # Удила/ножи/накладки/стрелы/кельты: любой «не целый» → сломан
    if has_broken:
        return "сломан"
    if has_whole or text.startswith("цел"):
        return "целый"
    return "сломан"


def coarse_material(raw: object, class_name: str | None = None) -> str | None:
    text = _clean(raw)
    if not text or text in NOT_SPECIFIED_VALUES:
        return None
    # Мусорные перечисления (ножи) — не материал предмета
    if class_name == "ножи" and (
        "латунь" in text or "пластмасс" in text or "алюминий" in text
    ):
        return None
    if class_name == "накладки":
        # материал у накладок удаляем целиком
        return None
    if text.startswith("бронза") or text == "бронза":
        return "бронза"
    if text.startswith("железо") or text == "железо":
        return "железо"
    return None


def coarse_kreplenie(raw: object) -> str | None:
    """Накладки:крепление → внутреннее / внешнее."""
    text = _clean(raw)
    if not text or text in NOT_SPECIFIED_VALUES:
        return None
    if text in ("внутреннее", "внешнее"):
        return text
    if "шпен" in text or "отверст" in text:
        return "внутреннее"
    if any(x in text for x in ("петель", "перемыч", "клепк")):
        return "внешнее"
    return None


def coarse_nakladka_forma(raw: object) -> str | None:
    """Накладки:форма → округлая / прямоугольная / фигурная; ажурная/двухчастная — drop."""
    text = _clean(raw)
    if not text or text in NOT_SPECIFIED_VALUES:
        return None
    if text in ("округлая", "прямоугольная", "фигурная"):
        return text
    if "ажур" in text or "двухчаст" in text or "двучаст" in text:
        return None
    if "зубчат" in text or "подпрямоуг" in text:
        return "прямоугольная"
    if "прямоуг" in text:
        return "прямоугольная"
    if any(
        x in text
        for x in ("лепест", "фигур", "меч", "конус", "капле", "полусфер")
    ):
        return "фигурная"
    if "кругл" in text or "овал" in text or "дисков" in text or "подоваль" in text:
        return "округлая"
    return "фигурная"


def coarse_tip_nasada(raw: object) -> str | None:
    """Наконечники:крепление (тип_насада) → втульчатый / черешковый."""
    text = _clean(raw)
    if not text or text in NOT_SPECIFIED_VALUES:
        return None
    if text in ("втульчатый", "черешковый"):
        return text
    if "втул" in text:
        return "втульчатый"
    return "черешковый"


def coarse_forma_pera(raw: object) -> str | None:
    """Наконечники:форма → заостренный / прямой / расщепленный."""
    text = _clean(raw)
    if not text or text in NOT_SPECIFIED_VALUES:
        return None
    if text in ("заостренный", "прямой", "расщепленный"):
        return text
    if "вильчат" in text or "трехлопаст" in text or "двухлопаст" in text:
        return "расщепленный"
    if "прямоуголь" in text or "подквадрат" in text:
        return "прямой"
    if any(
        x in text
        for x in (
            "треуголь",
            "пулевид",
            "пылевид",
            "шипаст",
            "ромбовид",
            "ромбическ",
            "трапециевид",
            "четырехгран",
        )
    ):
        return "заостренный"
    return "заостренный"


def coarse_sechenie(raw_sechenie: object, raw_rebra: object = None) -> str | None:
    """Сечение через пересечение «сечение» и «ребра» → плоское / объемное."""
    sec = _clean(raw_sechenie)
    reb = _clean(raw_rebra) if raw_rebra is not None else ""
    if sec in ("плоское", "объемное"):
        return sec
    sec_ok = bool(sec) and sec not in NOT_SPECIFIED_VALUES
    reb_ok = bool(reb) and reb not in NOT_SPECIFIED_VALUES
    if not sec_ok and not reb_ok:
        return None
    if (reb_ok and reb == "нет") or (sec_ok and "плоск" in sec):
        return "плоское"
    return "объемное"


def coarse_knife_tip(raw: object) -> str | None:
    text = _clean(raw)
    if not text or text in NOT_SPECIFIED_VALUES:
        return None
    if text in ("прямой", "изогнутый"):
        return text
    if text in ("прямолезвийный", "листовидный") or "прямолезв" in text or "листовид" in text:
        return "прямой"
    if any(
        x in text
        for x in ("выпуклообуш", "змейчатообуш", "коленчат")
    ):
        return "изогнутый"
    return None


def coarse_knife_rukoyat(raw: object) -> str | None:
    text = _clean(raw)
    if not text or text in NOT_SPECIFIED_VALUES:
        return None
    if text in ("выделенная", "невыделенная"):
        return text
    if text in ("есть",) or "петельчат" in text or "грибовид" in text:
        return "выделенная"
    if text == "нет" or "без выделен" in text:
        return "невыделенная"
    return None


def coarse_tip_okonchania(raw: object) -> str | None:
    """Удила:тип_окончания → кольчато-овальное / подтреугольное."""
    text = _clean(raw)
    if not text or text in NOT_SPECIFIED_VALUES:
        return None
    if text in ("кольчато-овальное", "подтреугольное"):
        return text
    # Подтреугольное семейство (включая подтреугольно-кольчатое и трапециевидное кольчатое)
    if "подтреугол" in text or "трапец" in text:
        return "подтреугольное"
    # Кольчато-овальное: овальные/кольчатые/стремечковидные/с рамкой
    if any(
        x in text
        for x in (
            "кольч",
            "овальн",
            "кольцев",
            "стремеч",
            "рамк",
        )
    ):
        return "кольчато-овальное"
    return None


def coarse_kelty_tulya(raw: object) -> str | None:
    text = _clean(raw)
    if not text or text in NOT_SPECIFIED_VALUES:
        return None
    if text in ("овальный", "прямоугольный"):
        return text
    if "овал" in text or "округл" in text:
        return "овальный"
    return "прямоугольный"


def coarse_yes_no(raw: object) -> str | None:
    text = _clean(raw)
    if not text or text in NOT_SPECIFIED_VALUES:
        return None
    if text in ("да", "есть") or text.startswith("на одной"):
        return "да"
    if text == "нет":
        return "нет"
    return None


def normalize_cell_value(class_name: str, feature_name: str, raw: object) -> str | None:
    """Нормализация одной ячейки признака для Excel и обучения."""
    text = _clean(raw)
    if not text or text in NOT_SPECIFIED_VALUES:
        return None

    if feature_name in DROP_FEATURE_COLUMNS.get(class_name, frozenset()):
        return None

    if feature_name == "сохранность":
        return coarse_sohrannost(text, class_name)

    if feature_name == "материал":
        return coarse_material(text, class_name)

    if feature_name == "тулья" and class_name == "кельты":
        return coarse_kelty_tulya(text)

    if feature_name in ("ушки", "орнамент"):
        return coarse_yes_no(text)

    if feature_name == "крепление" and class_name == "накладки":
        return coarse_kreplenie(text)

    if feature_name == "форма" and class_name == "накладки":
        return coarse_nakladka_forma(text)

    if feature_name == "тип_насада" and class_name == "наконечники стрел":
        return coarse_tip_nasada(text)

    if feature_name == "форма_пера" and class_name == "наконечники стрел":
        return coarse_forma_pera(text)

    if feature_name == "сечение" and class_name == "наконечники стрел":
        # одиночная ячейка (после merge); «ребра» уже учтены в apply-скрипте
        return coarse_sechenie(text, None)

    if feature_name == "тип" and class_name == "ножи":
        return coarse_knife_tip(text)

    if feature_name == "рукоять" and class_name == "ножи":
        return coarse_knife_rukoyat(text)

    if feature_name == "тип_окончания" and class_name == "удила":
        return coarse_tip_okonchania(text)

    return text


def fix_image_path(raw: object, photos_dir: Path) -> str | None:
    """Связь строки таблицы с файлом в data/dataset/photos."""
    if raw is None:
        return None
    name = Path(str(raw).strip()).name
    if not name:
        return None
    lower = name.lower()
    candidate = photos_dir / name
    if candidate.is_file():
        return name
    # уд80 → уд85 (опечатка в разметке полевых фото)
    if lower.startswith("уд80_"):
        fixed = "уд85_" + lower[5:]
        if (photos_dir / fixed).is_file():
            return fixed
    # только имя файла без пути
    if (photos_dir / lower).is_file():
        return lower
    return name


def is_confused_item(image_path: str) -> bool:
    stem = Path(image_path).stem.lower()
    item_key = stem.split("_field_")[0] if "_field_" in stem else stem.rsplit("_", 1)[0]
    return any(item_key.startswith(p) for p in CONFUSED_ITEM_PREFIXES)
