"""
Схема классов и признаков (заглушка до появления датасета).
При появлении данных — правьте списки и перезапустите обучение.
"""

from dataclasses import dataclass

# --- Сеть 1: тип объекта (multiclass, one-hot на выходе обучения) ---
OBJECT_CLASSES: list[str] = [
    "керамика",
    "металл",
    "кость",
    "стекло",
    "камень",
    "неопределено",
]

# --- Сеть 2: признаки зависят от типа объекта (multi-label) ---
FEATURE_SCHEMA: dict[str, list[str]] = {
    "керамика": [
        "орнамент_есть",
        "глазурь",
        "скол_края",
        "фрагмент_донышко",
        "фрагмент_горлышко",
    ],
    "металл": [
        "коррозия",
        "патина",
        "слой_окисла",
        "сохранность_высокая",
    ],
    "кость": [
        "окраска",
        "трещины",
        "обработка_края",
    ],
    "стекло": [
        "прозрачность",
        "пузыри_в_массе",
        "скол",
    ],
    "камень": [
        "шлифовка",
        "сколы",
        "отложения",
    ],
    "неопределено": [],
}

INPUT_SIZE = (224, 224)  # MobileNet standard
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MODELS_DIR = "models/archaeology"
OBJECT_MODEL_FILE = "object_classifier.pt"
FEATURE_MODEL_FILE = "feature_classifier.pt"


@dataclass(frozen=True)
class FeatureIndex:
    """Глобальный индекс multi-label вектора признаков."""

    name: str
    object_class: str
    local_index: int
    global_index: int


def build_feature_index() -> tuple[list[str], list[FeatureIndex], dict[str, list[int]]]:
    """Плоский список признаков и индексы по классу объекта."""
    flat: list[str] = []
    meta: list[FeatureIndex] = []
    by_class: dict[str, list[int]] = {c: [] for c in OBJECT_CLASSES}

    global_i = 0
    for obj_class in OBJECT_CLASSES:
        for local_i, feat in enumerate(FEATURE_SCHEMA.get(obj_class, [])):
            flat.append(f"{obj_class}:{feat}")
            meta.append(
                FeatureIndex(
                    name=feat,
                    object_class=obj_class,
                    local_index=local_i,
                    global_index=global_i,
                )
            )
            by_class[obj_class].append(global_i)
            global_i += 1

    return flat, meta, by_class


FEATURE_LABELS, FEATURE_INDEX, FEATURE_INDICES_BY_CLASS = build_feature_index()
NUM_OBJECT_CLASSES = len(OBJECT_CLASSES)
NUM_FEATURES = len(FEATURE_LABELS)
