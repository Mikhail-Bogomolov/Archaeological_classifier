"""Сборка orange_pi_deploy.zip — минимум для запуска на Orange Pi."""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "orange_pi_deploy.zip"

DEPLOY_FILES = [
    "requirements-orange-pi.txt",
    "app/main.py",
    "app/inference.py",
    "app/camera_capture.py",
    "app/db.py",
    "app/export_markup.py",
    "app/usb_export.py",
    "app/templates/index.html",
    "app/templates/scan.html",
    "app/templates/object_detail.html",
    "app/templates/export_csv.html",
    "app/static/app.js",
    "app/static/export.js",
    "app/static/home.css",
    "app/static/small-screen.css",
    "app/ml/__init__.py",
    "app/ml/pipeline.py",
    "app/ml/preprocess.py",
    "app/ml/config.py",
    "app/ml/encoders.py",
    "app/ml/texture_features.py",
    "app/ml/image_processing.py",
    "app/ml/feature_vocab.py",
    "app/ml/models/__init__.py",
    "app/ml/models/backbone.py",
    "app/ml/models/object_classifier.py",
    "app/ml/models/feature_classifier.py",
    "data/templates/markup_template.xlsx",
    "data/templates/markup_template_csv.zip",
    "models/archaeology/object_classifier.pt",
    "models/archaeology/feature_classifier.pt",
    "models/archaeology/feature_classifier.vocab.json",
]

README = """Orange Pi — минимальный деплой Archaeological Classifier
============================================================

1. Распаковать:
   unzip orange_pi_deploy.zip -d ~/Archaeological_classifier

2. Python-окружение:
   cd ~/Archaeological_classifier
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip

   # PyTorch CPU-only (CUDA на одноплатнике нет — не ставить cu118/cu121 wheels).
   # Вариант A — официальный CPU-индекс (часто x86_64):
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

   # Вариант B — ARM (Orange Pi / Raspberry Pi), если A не сработал:
   # pip install torch torchvision --extra-index-url https://www.piwheels.org/simple

   pip install -r requirements-orange-pi.txt

   Проверка:
   python -c "import torch; print(torch.__version__, 'cuda=', torch.cuda.is_available())"
   # cuda должен быть False

3. Запуск:
   uvicorn app.main:app --host 0.0.0.0 --port 8000

4. Браузер (телефон/ПК в той же Wi-Fi):
   http://IP_ПЛАТЫ:8000/scan

IP платы: hostname -I

Сканирование:
- Можно загрузить фото вручную.
- Или нажать «Начать сканирование» без файла — снимок с USB-камеры.
- Если камера не та: export CAMERA_INDEX=0  (или 1) перед запуском uvicorn.
- Снимки сохраняются в data/photos_from_camera/ (папка создаётся автоматически).

Примечания:
- В архиве requirements-orange-pi.txt — без torch/torchvision (чтобы pip не тянул CUDA-сборки).
- На ПК для разработки используйте обычный requirements.txt.
- Датасет для обучения в архив не входит — только веса и веб-приложение.
"""


def main() -> None:
    missing = [rel for rel in DEPLOY_FILES if not (ROOT / rel).is_file()]
    if missing:
        raise SystemExit(f"Не найдены файлы:\n" + "\n".join(missing))

    if OUT.exists():
        OUT.unlink()

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("ORANGE_PI_README.txt", README)
        for rel in DEPLOY_FILES:
            zf.write(ROOT / rel, rel)

    size_mb = OUT.stat().st_size / (1024 * 1024)
    print(f"Created {OUT} ({size_mb:.1f} MB, {len(DEPLOY_FILES) + 1} entries)")


if __name__ == "__main__":
    main()
