# Дополнения к MAPF-GPT

Здесь лежат скрипты и эталонные правки к ядру, которые не входят в оригинальный репозиторий [CognitiveAISystems/MAPF-GPT](https://github.com/CognitiveAISystems/MAPF-GPT).

## Запуск

Рабочая директория — **корень клона** (где лежат `mapf_gpt/`, `example.py`, `weights/`):

```bash
cd /path/to/MAPF-GPT
source .venv/bin/activate   # или ваше окружение
```

## Скрипты в `extras/`

| Файл | Назначение |
|------|------------|
| `trim_model.py` | Усечь глубину трансформера: оставить первые `n_layer` блоков из чекпоинта (без дообучения). |
| `quantize_checkpoint.py` | Динамическая int8-квантизация `Linear` → меньший `.pt`; инференс в PyTorch на CPU. |
| `compare_three_metrics.py` | Один сценарий, три веса (2M / trim / int8), метрики и PNG-график. |

Примеры:

```bash
python extras/trim_model.py -i weights/model-2M.pt -o weights/model-2M-L3.pt --n_layer 3
python extras/quantize_checkpoint.py -i weights/model-2M.pt -o weights/model-2M-int8.pt
python extras/compare_three_metrics.py --map_name validation-mazes-seed-000 -o svg/compare-three-metrics.png
```

## Правки в основном коде

В этой копии репозитория уже применены изменения:

- **`mapf_gpt/inference.py`** — загрузка int8-чекпоинтов (`quantized`), `strip_prefix` с сохранением `_metadata`, поле **`infer_dtype`** (`float32` / `bfloat16` / `float16`) для GPU.
- **`example.py`** — флаги **`--weights`**, **`--dtype`**.

Эталонные копии для переноса на чистый апстрим лежат в **`extras/integration/`**:

```bash
cp extras/integration/mapf_gpt/inference.py mapf_gpt/inference.py
cp extras/integration/example.py example.py
```

## Зависимости

Те же, что у проекта (`docker/requirements.txt`): PyTorch, `pogema-toolbox`, `matplotlib` для графика сравнения.
