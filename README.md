# Archaeological Classifier — Канск 2023

Двухэтапный классификатор археологических находок: **сеть 1** (тип объекта) → **сеть 2** (признаки).

## Структура

| Путь | Назначение |
|------|------------|
| `app/` | FastAPI UI, инференс, камера |
| `app/ml/` | обучение, оценка, препроцессинг |
| `data/dataset/` | фото + Excel (не в git) |
| `models/archaeology/` | веса `.pt` |
| `reports/` | бенчмарки и логи обучения |

## Установка (ПК)

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Порядок работы

### 1. Подготовка данных

```powershell
py scripts/process_kansk_dataset.py
py scripts/audit_dataset.py
```

`audit_dataset.py` проверяет утечки `item_key`, сохраняет `data/dataset/split_manifest.json`.

### 2. Обучение

```powershell
# Сеть 1 — тип объекта (val только для early stopping)
py -m app.ml.train_classifier --epochs 50 --batch-size 16

# Сеть 2 — признаки
py -m app.ml.train_feature_classifier --epochs 40
```

Лог эпох: `reports/training/object_classifier.csv`.  
Гиперпараметры сохраняются в чекпоинте (`training` в `.pt`).

### 3. Оценка (только test holdout)

```powershell
py -m app.ml.evaluate_classifier --split test --json-out reports/object_test.json
py -m app.ml.evaluate_classifier --split val --calibrate
py -m app.ml.evaluate_feature_classifier --split test --json-out reports/feature_test.json
py -m app.ml.evaluate_feature_classifier --split test --class ножи
```

Сеть 1: один отчёт с `per_class` (кельты, ножи, …).  
Сеть 2: головы вида `класс:признак` (например `ножи:материал`); в JSON и консоли есть сводка **`by_class`** — accuracy и macro-F1 по каждому типу объекта и по каждому признаку внутри типа.

`--calibrate` пишет `extra.calibration` (в т.ч. `suggested_threshold`) в `object_classifier.pt`; pipeline подхватывает порог при загрузке.  
В отчёте сети 1 также печатается список проблемных `item_key` (не все ракурсы верны).

Рядом с JSON сохраняются PNG:
- `reports/object_test.png` — confusion heatmap
- `reports/object_test_bars.png` — precision/recall/F1 по классам
- для сети 2 — папка `reports/feature_test_heatmaps/`

**Важно:** `val` используется при обучении (выбор лучшей эпохи).  
Итоговые цифры для отчёта — только **`--split test`**.

### 4. Веб-приложение

```powershell
uvicorn app.main:app --reload
```

Сканирование: http://127.0.0.1:8000/scan

### 5. Orange Pi

```powershell
py scripts/build_orange_pi_deploy.py
```

На плате: PyTorch CPU + `requirements-orange-pi.txt` (см. `ORANGE_PI_README.txt` в zip).

## Методология сплитов

- Разбивка **по `item_key`** (все ракурсы одного предмета в одном сплите).
- По умолчанию: **70% train / 15% val / 15% test**, seed=42.
- Для устойчивости оценки — повторите бенчмарк с разными `--seed` (42, 43, …).

## Тесты

```powershell
py -m unittest discover -s tests -v
```

## Конфигурация

Гиперпараметры: `app/ml/training_config.py`  
Порог «низкая уверенность»: из `extra.calibration` в `.pt`, иначе `DEFAULT_INFERENCE.object_low_conf_threshold`.
