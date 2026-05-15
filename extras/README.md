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
| `clean_checkpoint.py` | Очистка метаданных модели (optimizer state, config) для инференса |
| `compare_metrics.py` | Один сценарий, шесть весов (2M / L3 (trim) / L3 (trim, ft) / L4 (trim) / L4 (trim, ft) / int8), метрики и PNG-график. |

Примеры:

```bash
python extras/trim_model.py -i weights/model-2M.pt -o weights/model-2M-L3.pt --n_layer 3
python extras/quantize_checkpoint.py -i weights/model-2M.pt -o weights/model-2M-int8.pt
python extras/compare_three_metrics.py --map_name validation-mazes-seed-000 -o svg/compare-three-metrics.png

python extras/clean_checkpoint.py -i out_finetuned/L4/ckpt.pt -o weights/L4-ft.pt
python extras/compare_metrics.py --map_name validation-mazes-seed-000 -o svg/compare-metrics.png
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

## Дообучение после тримминга

Для дообучения необходимо добавить конфиг для модели после тримминга в папку `mapf_gpt`. Для L3 и L4 уже созданы соответствующие конфиги: `config_finetune_L3.py` и `config_finetune_L4.py`. Внутри этих конфигов в том числе заданы пути к тренировочному и валидационному датасетам, а также путь к выходному файлу (настроить пути при необходимости).

Нужно добавить чекпоинт для дообучения по пути параметра `out_dir`:
```bash
cp weights/model-2M-L4.pt out_finetuned/L4/ckpt.pt
```

Само дообучение запускается скриптом `finetune.py`, который является немного видоизмененной версией файла `train.py`. Запуск дообучения:

```bash
python finetune.py mapf_gpt/config_finetune_L4.py
```

Результаты дообучения лежат в папке `out_finetuned`.

Эти чекпоинты содержат служебные данные, что приводит к увеличению веса файла. Чтобы очистить их и подготовить к инференсу, запускаем скрипт:

```bash
python extras/clean_checkpoint.py -i out_finetuned/L4/ckpt.pt -o weights/L4-ft.pt
```

## Изменения для бенчмаркинга

Для оценки кастомных моделей требуется изменить следующие файлы:

```bash
MAPF-GPT/
└── eval_configs/
    ├── 01-random
    │   └── 01-random.yaml
    ├── 02-mazes
    │   └── 02-mazes.yaml
    ├── 03-warehouse
    │   └── 03-warehouse.yaml
    ├── 04-moningai
    │   └──  04-moningai.yaml
    └── 05-puzzles
        └── 05-puzzles.yaml
```

В них изменить `algoritms` (уже сделано):

```yaml
algorithms:
  # Original
  MAPF-GPT-2M:
    name: MAPF-GPT
    path_to_weights: weights/model-2M.pt
  
  # L4 trimmed
  MAPF-GPT-2M-L4:
    name: MAPF-GPT
    path_to_weights: weights/model-2M-L4.pt
  
  # L4 fine-tuned
  MAPF-GPT-2M-L4-ft:
    name: MAPF-GPT
    path_to_weights: weights/model-2M-L4-finetuned.pt
  
  # L3 trimmed
  MAPF-GPT-2M-L3:
    name: MAPF-GPT
    path_to_weights: weights/model-2M-L3.pt

  # L3 fine-tuned
  MAPF-GPT-2M-L3-ft:
    name: MAPF-GPT
    path_to_weights: weights/model-2M-L3-finetuned.pt
```

Запуск бенчмаркинга такой же, как в оригинале:

```bash
python benchmark.py
```

Результаты бенчмаркинга будут в папках карт в папке `eval_configs`.

Визуализация результатов:

```bash
python extras/view_results.py
```

Результаты визуализации будут в папке `plots/by_map`.