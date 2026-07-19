"""Нормализация значений в Excel — синонимы и укрупнение без смены смысла."""

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
    "уд85_8-62",
    "уд85_25-7",
)

GENERIC_ALIASES: dict[str, str] = {
    "бронза": "бронза",
    "железо": "железо",
    "целый": "целый",
    "целая": "целый",
    "целые": "целый",
    "целое": "целый",
    "сломан": "сломан",
    "сломана": "сломан",
    "сломаны": "сломан",
    "обломан": "сломан",
    "обломаны": "сломан",
    "обломок": "фрагмент",
    "обломки": "фрагмент",
    "фрагмент": "фрагмент",
    "половина": "половина",
    "целый и фрагмент": "целый и фрагмент",
    "целые и сломаны": "смешанный",
    "целые и в обломках": "смешанный",
    "целые и два обломанных": "смешанный",
    "целый, сломан": "смешанный",
    "целый и сломанные": "смешанный",
    "два целых и три обломанных": "смешанный",
    "да": "да",
    "нет": "нет",
    "не определимо": "не указано",
    "не нужны": "не указано",
    "не нужен": "не указано",
    "шпенек": "шпеньки",
    "отверстие": "через отверстия",
    "цветной металл": "цветной металл",
    "цветной металл, пластик. стекло": "цветной металл, пластик. стекло",
    "железо, латунь, медь, алюминий, свинец, пластмасса": (
        "железо, латунь, медь, алюминий, свинец, пластмасса"
    ),
    "бронза, олово, медь, и другой металл, камень": (
        "бронза, олово, медь, и другой металл, камень"
    ),
}

KREPLENIE_ALIASES: dict[str, str] = {
    "шпеньки": "шпеньки",
    "шпенек": "шпеньки",
    "петелька": "петелька",
    "через отверстия": "через отверстия",
    "отверстие": "через отверстия",
    "перемычка": "перемычка",
    "клепка": "клепка",
    "нет": "нет",
    "через отверстия, одна отверстие и шпеньки": "шпеньки",
}


def _clean(text: object) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def is_not_specified(text: object) -> bool:
    return _clean(text) in NOT_SPECIFIED_VALUES


def coarse_forma_pera(raw: object) -> str | None:
    """Укрупнение формы пера наконечника (сохраняем тип, убираем детали отверстий)."""
    text = _clean(raw)
    if not text or text in NOT_SPECIFIED_VALUES:
        return None
    if "трехлопаст" in text:
        return "трехлопастной"
    if "двухлопаст" in text:
        return "двухлопастной"
    if "вильчат" in text:
        return "вильчатый"
    if "ромбовид" in text or "ромбическ" in text or "ассиметрично ромб" in text:
        return "ромбовидный"
    if "четырехгран" in text:
        return "четырехгранный"
    if "трапециевид" in text:
        return "трапециевидный"
    if "пулевид" in text:
        return "пулевидный"
    if "подквадрат" in text:
        return "подквадратный"
    if "прямоуголь" in text:
        return "прямоугольный"
    if "треуголь" in text:
        return "треугольный"
    if "шипаст" in text:
        return "шипастый"
    return text


def coarse_sohrannost(raw: object) -> str | None:
    """Схлопывает опечатки и свободный текст сохранности к канону."""
    text = _clean(raw)
    if not text or text in NOT_SPECIFIED_VALUES:
        return None
    if text in GENERIC_ALIASES:
        return GENERIC_ALIASES[text]

    has_whole = bool(re.search(r"\bцел", text))
    has_broken = any(
        x in text
        for x in ("сломан", "обломан", "облом", "отлом", "фрагмент", "погнут")
    )
    mixed_markers = (
        " и сломан",
        " и облом",
        " и в облом",
        ", сломан",
        "смешан",
        "два цел",
        "целые и",
        "целый и сломан",
        "целый, сломан",
    )
    if any(m in text for m in mixed_markers):
        return "смешанный"
    if has_whole and has_broken and "и фрагмент" not in text:
        # «тело целое, ушко отломано» и т.п.
        return "смешанный"
    if "и фрагмент" in text:
        return "целый и фрагмент"
    if "половин" in text:
        return "половина"
    if "фрагмент" in text or "облом" in text:
        return "фрагмент"
    if "сломан" in text or "обломан" in text or "погнут" in text:
        return "сломан"
    # «целый, на одной сторон отверстие (недолив)» / окрашен / склеен → целый
    if has_whole or text.startswith("цел"):
        return "целый"
    return text


def normalize_cell_value(class_name: str, feature_name: str, raw: object) -> str | None:
    """Нормализация одной ячейки признака для Excel и обучения."""
    text = _clean(raw)
    if not text or text in NOT_SPECIFIED_VALUES:
        return None

    if feature_name == "сохранность":
        return coarse_sohrannost(text)

    if feature_name == "форма_пера" and class_name == "наконечники стрел":
        return coarse_forma_pera(text)

    if feature_name == "крепление" and class_name == "накладки":
        return KREPLENIE_ALIASES.get(text, text)

    if feature_name == "материал":
        if "бронза" in text and "олово" in text:
            return "бронза, олово, медь, и другой металл, камень"
        if "цветной металл" in text and "пластик" in text:
            return "цветной металл, пластик. стекло"
        if "железо" in text and any(
            x in text for x in ("латунь", "медь", "алюминий", "свинец", "пластмасса")
        ):
            return "железо, латунь, медь, алюминий, свинец, пластмасса"
        if text.startswith("бронза"):
            return "бронза"
        if text.startswith("железо"):
            return "железо"

    return GENERIC_ALIASES.get(text, text)


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
