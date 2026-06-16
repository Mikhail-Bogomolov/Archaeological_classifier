"""Классы и пути — датасет Канск 2023."""

from dataclasses import dataclass

# Тип объекта (сеть 1)
OBJECT_CLASSES: list[str] = [
    "кельты",
    "ножи",
    "удила",
    "наконечники стрел",
    "накладки",
]

# Признаки по типу (сеть 2)
FEATURE_SCHEMA: dict[str, list[str]] = {
    "кельты": [
        "материал",
        "сохранность",
        "форма",
        "тулья",
        "ушки",
        "орнамент",
    ],
    "ножи": [
        "материал",
        "сохранность",
        "тип",
        "рукоять",
        "орнамент",
    ],
    "удила": [
        "материал",
        "сохранность",
        "тип_окончания",
        "шарнирность",
    ],
    "наконечники стрел": [
        "материал",
        "сохранность",
        "тип_насадки",
        "форма_лезвия",
        "сечение",
        "ребра",
    ],
    "накладки": [
        "материал",
        "сохранность",
        "форма",
        "крепление",
        "орнамент",
    ],
}

# Путь к датасету Канск 2023
KANSK_DATASET_DIR = "data/perdataset/kansk_2023"
KANSK_PHOTOS_DIR = f"{KANSK_DATASET_DIR}/photos"
KANSK_TABLES_DIR = f"{KANSK_DATASET_DIR}/tables"

# Колонки шаблона разметки / экспорта
MARKUP_COLUMNS: list[str] = ["номер", "название"] + [
    f"признак {i}" for i in range(1, 6)
]

INPUT_SIZE = (224, 224)
CV_TARGET_SHAPE = 50  # размер после обработки OpenCV
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MODELS_DIR = "models/archaeology"
OBJECT_MODEL_FILE = "object_classifier.pt"
FEATURE_MODEL_FILE = "feature_classifier.pt"
USE_TEXTURE_FEATURES = True

MARKUP_TEMPLATE_PATH = "data/templates/markup_template.xlsx"

# Тестовые фото в корне (если понадобятся вручную)
TEST_IMAGE_PATHS: list[str] = [
    "test_img_02.jpg",
    "test_img_10.jpg",
]


@dataclass(frozen=True)
class FeatureIndex:
    name: str
    object_class: str
    local_index: int
    global_index: int


def build_feature_index() -> tuple[list[str], list[FeatureIndex], dict[str, list[int]]]:
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
