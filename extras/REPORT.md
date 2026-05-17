# LightWeight MAPF-GPT — финальный отчёт по экспериментам

**Проект:** сжатие и ускорение политики [MAPF-GPT](https://github.com/CognitiveAISystems/MAPF-GPT) при сохранении качества на бенчмарке POGEMA.  
**Код:** каталог `extras/` (пакет `lmgpt/`, CLI в `extras/scripts/`).  
**Дата экспериментов:** 15–17 мая 2026.  
**Окружение:** `conda` env `mapf-gpt`, PyTorch 2.10, CUDA 12.8, GPU NVIDIA H200 140GB.

---

## Содержание

1. [Резюме](#1-резюме)
2. [Цели и постановка](#2-цели-и-постановка)
3. [Важно: номенклатура моделей](#3-важно-номенклатура-моделей)
4. [Инфраструктура и воспроизводимость](#4-инфраструктура-и-воспроизводимость)
5. [Методология](#5-методология)
6. [План экспериментов (ID)](#6-план-экспериментов-id)
7. [Протокол обучения](#7-протокол-обучения)
8. [Протокол оценки](#8-протокол-оценки)
9. [Результаты](#9-результаты)
10. [Анализ и выводы](#10-анализ-и-выводы)
11. [Ограничения и дальнейшая работа](#11-ограничения-и-дальнейшая-работа)
12. [Публикация на GitHub и раздача весов](#12-публикация-на-github-и-раздача-весов)

---

## 1. Резюме

Мы реализовали конвейер **LightWeight MAPF-GPT**: структурное сжатие (trim), восстановление качества (fine-tune / self-distillation), дистилляция знаний от больших учителей (6M, 85M), квантизация AWQ (W4A16), единый **POGEMA-eval** и отдельный **latency-bench** (только forward `model.act`).

**Главный практический результат:** для замены **author 2M** (базовая продакшен-модель) лучший компромисс — **trim + fine-tune (T1/T2)**:

| Метод | Параметры | CSR | Ускорение forward (B=2048, p50) |
|--------|-----------|-----|----------------------------------|
| 2M author (R0) | 1.55M | **0.850** | 28.8 ms (bf16) |
| **T1: L4 trim + CE+KL** | 1.24M (−20%) | **0.849** | **14.8 ms (~1.9×)** |
| **T2: L3 trim + CE+KL** | 0.93M (−40%) | 0.832 | **11.5 ms (~2.5×)** |
| C3: AWQ W4A16 | 1.55M (сжатые веса) | 0.845 | 18.2 ms |

**Дистилляция в 1M** даёт ещё меньший размер и ~12 ms latency, но **CSR 0.79–0.83** — заметно ниже R0. Эксперимент **D5** (`student-2M.json`) по факту обучил студента **8×256 (~6.4M)** — ту же ширину, что у **6M author**, а не у маленькой 2M; он улучшает качество относительно 6M, но **не ускоряет** author-2M.

---

## 2. Цели и постановка

### 2.1. Задача

MAPF-GPT — autoregressive GPT-политика для multi-agent path finding. Author предоставляет чекпойнты **2M**, **6M**, **85M**. Для развёртывания важны:

- **Качество:** CSR / ISR на стандартном POGEMA-наборе сценариев.
- **Размер и скорость инференса:** параметры и latency forward-pass при батче агентов.

### 2.2. Гипотезы (что проверяли)

| ID | Гипотеза |
|----|----------|
| **C1–C2** | Удаление верхних слоёв (trim) уменьшает модель и ускоряет inference, но ломает политику без дообучения. |
| **T1–T3** | Fine-tune (CE или CE+KL от полной 2M) восстанавливает качество trim-модели. |
| **C3** | AWQ (4-bit веса) сохраняет CSR при умеренном ускорении на GPU. |
| **D1–D5** | Дистилляция от 6M/85M в меньших студентов даёт лучший quality/size, чем naive trim. |
| — | Hidden-state KD + multi-position logits сильнее, чем KL только на последнем токене. |

### 2.3. Референс (R0)

**R0** — author **2M** (`weights/model-2M.pt`), FP32 на POGEMA. **Верхние границы качества** — 6M и 85M author.

---

## 3. Важно: номенклатура моделей

В репозитории MAPF-GPT метки **«2M», «6M», «1M»** — **имена чекпойнтов**, а не всегда точное число параметров или одна архитектура.

### 3.1. Фактические архитектуры (из `model_args` в `.pt`)

| Имя | Слои × `n_embd` × головы | Параметры (подсчёт) |
|-----|--------------------------|---------------------|
| **2M author** | 5 × 160 × 5 | **1.55M** |
| **6M author** | 8 × 256 × 8 | **6.31M** |
| **85M author** | 12 × 768 × 12 | **85.0M** |
| **2M-L4-trim** | 4 × 160 × 5 | **1.24M** |
| **2M-L3-trim** | 3 × 160 × 5 | **0.93M** |
| **student-1M** | 4 × 128 × 4 | **0.84M** |
| **student-1.5M** | 4 × 160 × 5 | **1.29M** |
| **student-2M** (конфиг JSON) | 8 × 256 × 8 | **6.40M** (= ширина **6M**, не author 2M) |

**Вывод для отчёта:** сравнивать **D5** с «сжатием 2M author» некорректно — это студент **класса 6M**. Сжатие **именно author-2M** — линейка **C1 → T1/T2**.

Конфиги студентов: `extras/configs/students/*.json`.

---

## 4. Инфраструктура и воспроизводимость

### 4.1. Структура кода

```
extras/
  lmgpt/           # checkpoints, compression, distillation, eval, utils
  scripts/         # CLI-обёртки
  configs/         # students, distill, awq
  runs/            # артефакты (в .gitignore)
  REPORT.md        # этот документ
```

### 4.2. Запуск

Все команды — из **корня** `MAPF-GPT/`:

```bash
source /data/conda/miniconda3/etc/profile.d/conda.sh
conda activate mapf-gpt
cd /path/to/MAPF-GPT
```

### 4.3. Данные

- **Обучение:** `dataset/train` (Arrow, streaming).
- **Валидация при обучении:** `dataset/validation`.
- **POGEMA-eval:** все 5 групп сценариев (см. §8).

Датасет и author-веса **не в git** (см. `.gitignore`: `dataset/`, `weights/`).

### 4.4. Логирование

- **Comet ML:** проект `LightWeight-MAPF-GPT` (train distill, finetune trim). Ключ: `extras/.key_comet` или `COMET_API_KEY`. Отключение: `--no-comet`.
- **Локально:** каждый run → `run_config.json`, `metrics.jsonl`, `environment.txt`, `summary_latest.json`.

### 4.5. Канонические каталоги run-ов

| Категория | Путь (пример) |
|-----------|----------------|
| R0 baselines | `extras/runs/baselines/2026-05-15_20-23-07` |
| Trim eval | `extras/runs/trim/2026-05-15_20-27-01` (L4), `.../22-32-19` (L3) |
| AWQ eval | `extras/runs/awq/2026-05-15_20-31-50` |
| Trim+FT train | `extras/runs/trim_ft/2026-05-15_20-36-05_L4_cekl`, `..._L3_cekl`, `..._L4_ce_only` |
| Trim+FT eval | `extras/runs/trim_ft/2026-05-17_03-02-42`, `03-33-50`, `04-02-47` |
| Distill train | `extras/runs/distill/2026-05-15_21-10-21_1M_from_6M`, … (см. §6) |
| Distill eval | `extras/runs/distill/2026-05-17_04-34-34`, … |
| Сводный отчёт | `extras/runs/reports/final_2026-05-17/` |

Сборка таблиц:

```bash
python extras/scripts/compare_runs.py \
  --runs extras/runs/baselines/2026-05-15_20-23-07 \
         extras/runs/trim/2026-05-15_20-27-01 \
         extras/runs/trim/2026-05-15_22-32-19 \
         extras/runs/awq/2026-05-15_20-31-50 \
         extras/runs/trim_ft/2026-05-17_03-02-42 \
         extras/runs/trim_ft/2026-05-17_03-33-50 \
         extras/runs/trim_ft/2026-05-17_04-02-47 \
         extras/runs/distill/2026-05-17_04-34-34 \
         extras/runs/distill/2026-05-17_05-03-13 \
         extras/runs/distill/2026-05-17_05-31-58 \
         extras/runs/distill/2026-05-17_11-10-38 \
         extras/runs/distill/2026-05-17_11-43-45 \
  --baseline-model 2M \
  --out-dir extras/runs/reports/final_2026-05-17
```

---

## 5. Методология

### 5.1. Structural trim (C1, C2)

**Идея:** оставить только первые `n_layer` блоков трансформера; эмбеддинги и LM head без изменений.

**Реализация:** `extras/lmgpt/compression/trim.py` → `extras/scripts/trim.py`.

```bash
# L4: 5 → 4 слоя
python extras/scripts/trim.py -i weights/model-2M.pt -o weights/model-2M-L4.pt --n_layer 4
# L3: 5 → 3 слоя
python extras/scripts/trim.py -i weights/model-2M.pt -o weights/model-2M-L3.pt --n_layer 3
```

**Ожидание:** резкое падение CSR без fine-tune (проверено на C1/C2).

### 5.2. Fine-tune trimmed checkpoint (T1–T3)

**Идея:** восстановить политику на том же Arrow-датасете, что и pretrain.

**Режимы** (`extras/lmgpt/distillation/finetune.py`):

| Режим | `alpha_kl` | `alpha_ce` | Учитель |
|-------|------------|------------|---------|
| `ce_only` (T3) | 0 | 1 | не используется |
| `ce_kl` (T1, T2) | 0.7 | 0.3 | **полная 2M** до trim (`weights/model-2M.pt`) |

**Self-distillation:** студент — trimmed веса (`init_from`), учитель — оригинальная 5-слойная 2M. Температура KL: **T=2.0**.

**Скрипт:** `extras/scripts/finetune_trimmed.py`.

Гиперпараметры финального прогона (T1): `max_iters=30000`, `batch_size=2048`, `lr=1e-4`, `warmup=500`, `bf16`, полный train split.

### 5.3. Knowledge distillation (D1–D5)

**Общий loss** (логиты действий, первые 5 индексов vocab — как в `GPT.act`):

\[
\mathcal{L} = \alpha_{kl} \cdot T^2 \cdot \mathrm{KL}(p_s^{1:T} \| p_t^{1:T}) + \alpha_{ce} \cdot \mathrm{CE}(y)
\]

По умолчанию: \(\alpha_{kl}=0.7\), \(\alpha_{ce}=0.3\), \(T=2\).

**Расширения** (`extras/lmgpt/distillation/losses.py`):

1. **Multi-position (`multipos=True`):** KL по всем позициям контекста; вес последнего токена 1.0, остальных 0.1.
2. **Hidden-state KD (`alpha_hid=0.5`):** MSE между спроецированными hidden states студента и учителя; линейные проекторы `R^{d_s} \to R^{d_t}` при разной ширине.

**Конфиги:**

| Файл | Учитель | `alpha_hid` | `multipos` |
|------|---------|-------------|------------|
| `configs/distill/from_6M.json` | 6M | 0 | false |
| `configs/distill/default.json` | 85M | 0 | false |
| `configs/distill/with_hidden.json` | 85M | 0.5 | true |

**Скрипт:** `extras/scripts/train_distill.py --config ... --student ... --teacher ...`

**Валидация при обучении:** `num_val_batches=50`; опционально mini-rollout на одной POGEMA-карте (`with_hidden.json`).

### 5.4. AWQ quantization (C3)

**Идея:** 4-bit веса, 16-bit активации (W4A16), калибровка на батчах из `dataset/validation`.

**Конфиг:** `extras/configs/awq/default.json` — `group_size=64`, asymmetric, grid search `alpha` на [0, 1].

**Скрипт:** `extras/scripts/quantize_awq.py` → `weights/model-2M-awq.pt`.

**Инференс:** требуется `torchao` для GPU kernels.

### 5.5. Что сознательно не вошло в финальные цифры

- **SmoothQuant / dynamic int8** — реализованы в `extras/lmgpt/compression/`, но не входят в финальную таблицу POGEMA.
- **Неудачные distill-run-ы** (`2026-05-15_21-10-36`, `21-10-41` — OOM) без `ckpt_best.pt`.
- **Дубликат D4** (`2026-05-16_13-56-16`) — заменён каноническим `13-58-44`.

---

## 6. План экспериментов (ID)

| ID | Метод | Обучение (каталог) | Eval (каталог) | Статус |
|----|--------|-------------------|----------------|--------|
| **R0** | Author 2M / 6M / 85M | — (готовые веса) | `baselines/2026-05-15_20-23-07` | ✓ |
| **C1** | L4 trim, без FT | — | `trim/2026-05-15_20-27-01` | ✓ |
| **C2** | L3 trim, без FT | — | `trim/2026-05-15_22-32-19` | ✓ |
| **C3** | 2M AWQ W4A16 | — | `awq/2026-05-15_20-31-50` | ✓ |
| **T1** | L4 trim + CE+KL | `trim_ft/2026-05-15_20-36-05_L4_cekl` | `trim_ft/2026-05-17_03-02-42` | ✓ |
| **T2** | L3 trim + CE+KL | `trim_ft/2026-05-15_20-45-27_L3_cekl` | `trim_ft/2026-05-17_03-33-50` | ✓ |
| **T3** | L4 trim + CE only | `trim_ft/2026-05-15_20-47-39_L4_ce_only` | `trim_ft/2026-05-17_04-02-47` | ✓ |
| **D1** | 1M ← 6M (logits) | `distill/2026-05-15_21-10-21_1M_from_6M` | `distill/2026-05-17_04-34-34` | ✓ |
| **D2** | 1M ← 85M (default) | `distill/2026-05-15_21-10-26_1M_from_85M_default` | `distill/2026-05-17_05-03-13` | ✓ |
| **D3** | 1M ← 85M (hidden+mp) | `distill/2026-05-15_21-10-31_1M_from_85M_hidden` | `distill/2026-05-17_05-31-58` | ✓ |
| **D4** | 1.5M ← 85M (hidden+mp) | `distill/2026-05-16_13-58-44_1.5M_from_85M_hidden` | `distill/2026-05-17_11-10-38` | ✓ |
| **D5** | «2M» ← 85M (hidden+mp) | `distill/2026-05-16_13-59-24_2M_from_85M_hidden` | `distill/2026-05-17_11-43-45` | ✓ |

**Чекпойнты для коллег** (лучшие веса):

| ID | Файл |
|----|------|
| T1 | `extras/runs/trim_ft/2026-05-15_20-36-05_L4_cekl/ckpt_best.pt` |
| T2 | `extras/runs/trim_ft/2026-05-15_20-45-27_L3_cekl/ckpt_best.pt` |
| T3 | `extras/runs/trim_ft/2026-05-15_20-47-39_L4_ce_only/ckpt_best.pt` |
| D1–D5 | `extras/runs/distill/<run>/ckpt_best.pt` |
| C1/C2 | `weights/model-2M-L4.pt`, `weights/model-2M-L3.pt` |
| C3 | `weights/model-2M-awq.pt` |

---

## 7. Протокол обучения

### 7.1. Общие настройки

| Параметр | Trim+FT | Distillation |
|----------|---------|--------------|
| `max_iters` | 30 000 | 50 000 |
| `batch_size` | 2048 | 2048 |
| `dtype` | bfloat16 | bfloat16 |
| `optimizer` | AdamW (β1=0.9, β2=0.95) | то же |
| `weight_decay` | 0.1 | 0.1 |
| `grad_clip` | 1.0 | 1.0 |
| `lr` | **1e-4** (FT) | **3e-4** (distill) |
| `warmup` | 500 | 1000 |
| LR schedule | cosine → `min_lr_ratio=0.1` | cosine |

**Важно:** обучение — **фиксированное число итераций** по streaming Arrow, а не полные эпохи по 1B токенов. При `batch_size=2048` и 50k iters это порядка **~10%** одной «эпохи» относительно полного pretrain-объёма (оценка из плана проекта).

### 7.2. Данные

- Без `--max-train-files` — весь `dataset/train`.
- Валидация: `dataset/validation`, 50 батчей на eval step.

---

## 8. Протокол оценки

### 8.1. POGEMA benchmark

**Скрипт:** `extras/scripts/eval_pogema.py`.

| Параметр | Значение |
|----------|----------|
| Группы | `01-random`, `02-mazes`, `03-warehouse`, `04-movingai`, `05-puzzles` |
| Режим | полный (без `--quick`) |
| Эпизодов на модель | **3296** |
| Device | CUDA |
| Dtype инференса | **float32** (все финальные eval) |
| Метрики | **CSR**, **ISR**, **SR**, **SoC** (sum of costs), `runtime` (end-to-end эпизод) |

**Определения (POGEMA):**

- **CSR** (Collision Success Rate) — доля эпизодов, решённых без коллизий/конфликтов (выше лучше).
- **ISR** — доля агентов, успешно достигших цели (выше лучше).
- **SoC** — суммарная стоимость пути; в таблицах ниже **`SoC_mean`** по всем эпизодам с валидным SoC (ниже лучше для эффективности).

### 8.2. Latency benchmark

**Скрипт:** `extras/scripts/bench_latency.py`.

| Параметр | Значение |
|----------|----------|
| Что меряем | только `model.act(idx)`, **без** среды POGEMA |
| Batch sizes | 1, 32, 128, 512, **2048** |
| dtype | **bfloat16** (основной столбец отчёта) |
| Статистика | p50, p95, mean (CUDA events, 50 warmup + 200 measure) |
| `seq_len` | 256 |

**Сравнение с POGEMA `runtime`:** latency — чистый forward; `runtime` в eval включает симуляцию, планирование батчей агентов и т.д. (например, trimmed-модели после FT имеют `runtime_s` ~0.4–0.5 vs 1.5 у R0 из-за меньшой сети).

---

## 9. Результаты

### 9.1. Главная таблица (POGEMA + latency B=2048)

| Method | ID | Params | CSR ↑ | ISR ↑ | SoC ↓ | Latency p50 @2048 (ms) | Notes |
|--------|-----|--------|-------|-------|-------|------------------------|-------|
| 2M author | R0 | 1.55M | **0.850** | 0.964 | 3408 | 28.8 | ref; POGEMA FP32 |
| 6M author | R0 | 6.31M | 0.866 | 0.968 | 3299 | — | quality ceiling (mid) |
| 85M author | R0 | 85.0M | **0.925** | 0.983 | 3070 | — | quality ceiling (high) |
| L4 trim | C1 | 1.24M | 0.561 | 0.899 | 4613 | 14.9 | без FT — непригодно |
| L3 trim | C2 | 0.93M | 0.259 | 0.802 | 9508 | 11.5 | без FT — непригодно |
| L4 trim + CE+KL | **T1** | 1.24M | **0.849** | 0.964 | 3401 | 14.8 | **рекомендуемый заменитель 2M** |
| L3 trim + CE+KL | **T2** | 0.93M | 0.832 | 0.958 | 3515 | 11.5 | агрессивное сжатие |
| L4 trim + CE only | T3 | 1.24M | 0.848 | 0.961 | 3530 | 14.8 | ablation: KL почти не нужен |
| 2M AWQ W4A16 | C3 | 1.55M* | 0.845 | 0.965 | 3422 | 18.2 | *размер файла ↓, param count тот же |
| 1M ← 6M | D1 | 0.84M | 0.829 | 0.954 | 3545 | ~12.2† | logits KD |
| 1M ← 85M default | D2 | 0.84M | 0.796 | 0.942 | 3646 | ~12.2† | logits KD |
| 1M ← 85M hidden | D3 | 0.84M | 0.788 | 0.942 | 3829 | 12.2 | hidden + multipos |
| 1.5M ← 85M hidden | D4 | 1.29M | 0.819 | 0.951 | 3682 | 14.9 | hidden + multipos |
| «2M» ← 85M hidden | D5 | **6.40M** | 0.881 | 0.969 | 3255 | 38.8 | студент = 6M-width, не author 2M |

† D1/D2 отдельно не бенчмаркались; оценка по архитектуре 1M ≈ D3.

**Источник POGEMA:** `extras/runs/reports/final_2026-05-17/global_by_model.md`  
**Источник latency:** `extras/runs/reports/final_2026-05-17/latency_summary.md`

### 9.2. Полная таблица POGEMA (все модели)

| model_name | CSR | ISR | SR | SoC_mean | runtime_s | source_run |
|------------|-----|-----|-----|----------|-----------|------------|
| 2M | 0.8504 | 0.9643 | 0.8504 | 3407.87 | 1.465 | baselines/2026-05-15_20-23-07 |
| 6M | 0.8656 | 0.9678 | 0.8656 | 3298.90 | 2.404 | baselines/2026-05-15_20-23-07 |
| 85M | 0.9245 | 0.9825 | 0.9245 | 3069.99 | 8.852 | baselines/2026-05-15_20-23-07 |
| 2M-L4-trim | 0.5610 | 0.8985 | 0.5610 | 4612.91 | 2.128 | trim/2026-05-15_20-27-01 |
| 2M-L3-trim | 0.2591 | 0.8017 | 0.2591 | 9508.37 | 1.546 | trim/2026-05-15_22-32-19 |
| 2M-AWQ-W4A16 | 0.8453 | 0.9646 | 0.8453 | 3422.35 | 1.793 | awq/2026-05-15_20-31-50 |
| 2M-L4-ce_kl | 0.8486 | 0.9636 | 0.8486 | 3400.69 | 0.458 | trim_ft/2026-05-17_03-02-42 |
| 2M-L3-ce_kl | 0.8316 | 0.9581 | 0.8316 | 3515.35 | 0.411 | trim_ft/2026-05-17_03-33-50 |
| 2M-L4-ce_only | 0.8477 | 0.9610 | 0.8477 | 3530.44 | 0.467 | trim_ft/2026-05-17_04-02-47 |
| 1M-distilled-6M | 0.8292 | 0.9542 | 0.8292 | 3544.71 | 0.406 | distill/2026-05-17_04-34-34 |
| 1M-distilled-85M-default | 0.7961 | 0.9422 | 0.7961 | 3646.06 | 0.408 | distill/2026-05-17_05-03-13 |
| 1M-distilled-85M-hidden | 0.7882 | 0.9424 | 0.7882 | 3828.87 | 0.421 | distill/2026-05-17_05-31-58 |
| 1.5M-distilled-85M-hidden | 0.8189 | 0.9506 | 0.8189 | 3682.40 | 0.486 | distill/2026-05-17_11-10-38 |
| 2M-distilled-85M-hidden | 0.8805 | 0.9691 | 0.8805 | 3254.93 | 1.059 | distill/2026-05-17_11-43-45 |

### 9.3. Latency (forward only), p50 ms

| label | B=1 | B=32 | B=128 | B=512 | **B=2048** |
|-------|-----|------|-------|-------|------------|
| 2M_BF16 (author) | 1.20 | 2.46 | 1.69 | 8.46 | **28.82** |
| 2M_FP32 (author) | 1.26 | 1.97 | 9.09 | 31.33 | **130.35** |
| 2M-L4-ce_kl (T1) | 0.99 | 1.02 | 1.33 | 4.13 | **14.83** |
| 2M-L3-ce_kl (T2) | 0.83 | 0.83 | 1.08 | 3.22 | **11.47** |
| 2M-L4-trim (C1) | 1.05 | 1.07 | 1.35 | 4.12 | 14.85 |
| 2M-AWQ (C3) | 1.25 | 1.23 | 1.59 | 5.00 | 18.17 |
| 1M-distill-hidden (D3) | 1.02 | 1.06 | 1.16 | 3.41 | 12.16 |
| 1.5M-distill-hidden (D4) | 1.00 | 1.01 | 1.34 | 4.12 | 14.85 |
| 2M-distill-hidden (D5) | 1.64 | 1.66 | 3.01 | 10.35 | 38.84 |

### 9.4. Ablation: нужен ли KL при trim+FT?

| T3 (CE only) | T1 (CE+KL) | Δ CSR |
|--------------|------------|-------|
| 0.8477 | 0.8486 | +0.0009 |

**Вывод:** при L4 trim разница в пределах шума; CE-only достаточен, CE+KL — чуть стабильнее на валидации.

### 9.5. Ablation: teacher для 1M

| Student | Teacher | Recipe | CSR |
|---------|---------|--------|-----|
| 1M | **6M** | logits (D1) | **0.829** |
| 1M | 85M | logits (D2) | 0.796 |
| 1M | 85M | hidden+mp (D3) | 0.788 |

**Вывод:** для маленького студента учитель **6M** лучше, чем 85M (меньший capacity gap). Hidden KD на 1M не помог vs logits.

### 9.6. Относительно R0 (2M author)

| Method | Δ CSR | Speedup @B=2048 |
|--------|-------|-----------------|
| T1 | −0.0018 | **1.94×** |
| T2 | −0.0188 | **2.51×** |
| C3 AWQ | −0.0051 | 1.59× |
| D1 | −0.0212 | ~2.37× |
| D5 | +0.0301 | 0.74× (медленнее) |

---

## 10. Анализ и выводы

### 10.1. Trim + fine-tune — основной успех

1. **Trim без обучения (C1)** сильно ломает политику — структурное сжатие не сохраняет знания в весах.
2. **Fine-tune (T1)** восстанавливает CSR до уровня R0 при **~2×** ускорении forward и **−20%** параметров.
3. **T2** — ещё агрессивнее (−40% params, **~2.5×** speed), плата **~1.8 pp CSR**.

### 10.2. AWQ

Качество близко к R0 (ΔCSR ≈ −0.5 pp), ускорение скромнее trim+FT, но **не требует переобучения** — хороший вариант, если нельзя трогать training pipeline.

### 10.3. Distillation

- **D1** — лучший «маленький» вариант (0.84M, CSR 0.83), но **не догоняет R0**.
- **D3/D2:** hidden KD и 85M-учитель на 1M **не выигрывают** у D1.
- **D5:** конфиг `student-2M.json` задаёт **8×256**, фактически класс **6M**; CSR выше 6M author, но **не является сжатием author-2M** и **медленнее** по latency bench (большая сеть + возможные overhead).

### 10.4. Рекомендации для продакшена

| Приоритет | Модель | Когда |
|-----------|--------|-------|
| 1 | **T1** (`2M-L4-ce_kl`) | Нужны почти те же CSR, что у 2M, и минимальный риск |
| 2 | **T2** (`2M-L3-ce_kl`) | Нужен максимум скорости, допустима потеря ~2% CSR |
| 3 | **C3 AWQ** | Нельзя переобучать, только post-training quant |
| 4 | **D1** | Жёсткий лимит <1M параметров, готовы к −2% CSR |

---

## 11. Ограничения и дальнейшая работа

1. **Номенклатура D5:** переименовать конфиг / добавить `student-6M-width.json`; для сжатия 2M distill в **5×160**, а не 8×256.
2. **POGEMA vs latency dtype:** eval в FP32, bench в bf16 — для отчёта сравнивать осознанно; при необходимости прогнать POGEMA в bf16.
3. **Обучение по итерациям**, не по полным эпохам — возможен недо/переобучение; стоит сверить кривые val в Comet.
4. **INT8 / SmoothQuant** — не включены в финальные цифры.
5. **Статистическая значимость:** Wilson CI доступны в `compare_runs.py`, но в таблице не приведены; для публикации можно добавить.
6. **End-to-end latency** в проде включает POGEMA; смотреть и `runtime_s` из eval.

---

## 12. Публикация на GitHub и раздача весов

### 12.1. Pull Request (код)

В PR логично включить:

- `extras/lmgpt/`, `extras/scripts/`, `extras/configs/`
- `extras/README.md`, **`extras/REPORT.md`** (этот документ)
- `extras/integration/` (если есть патчи inference)
- **Не включать:** `extras/runs/`, `dataset/`, `weights/`, `extras/.key_comet` (уже в `.gitignore`)

В описании PR — ссылка на §9.1 и команды воспроизведения из §4.5.

### 12.2. Можно ли залить веса на Git?

**Да, но не обычным `git add` в тот же репозиторий без подготовки.**

| Артефакт | Размер (порядок) | GitHub без LFS |
|----------|------------------|----------------|
| T1/T2/T3 ckpt | ~5–10 MB | ✓ OK |
| D1–D4 ckpt | ~10–20 MB | ✓ OK |
| D5 ckpt | ~86 MB | ✓ OK (лимит файла 100 MB) |
| Author 2M | ~19 MB | ✓ OK |
| Author 6M | ~74 MB | ✓ OK |
| **Author 85M** | **~976 MB** | ✗ превышает 100 MB на файл |

**Ограничения GitHub:**

- Жёсткий лимит **100 MB на файл** (без LFS push отклонится).
- Репозиторий >1–2 GB неудобен для клонирования.
- `weights/` сейчас в `.gitignore` — это правильно для кода.

### 12.3. Рекомендуемые варианты раздачи весов

**Вариант A — Hugging Face Hub (предпочтительно)**

```text
org/lightweight-mapf-gpt/
  model-2M-author/          # или ссылка на оригинал MAPF-GPT
  trim_ft/T1-L4-ce_kl/
  trim_ft/T2-L3-ce_kl/
  distill/D1-1M-from-6M/
  ...
```

Плюсы: большие файлы, `huggingface-cli download`, версионирование, README на Hub.

**Вариант B — GitHub Releases**

Прикрепить `.pt` как assets к релизу `v1.0.0-experiments`. Лимит на asset ~2 GB на файл для Releases (выше, чем в tree). 85M влезет. Минус: нет удобного partial clone.

**Вариант C — Отдельный репозиторий + Git LFS**

```bash
git lfs track "*.pt"
git add .gitattributes weights/
```

Трекать только нужные чекпойнты (~200–300 MB суммарно для наших экспериментов, без 85M или с 85M отдельно на HF).

**Вариант D — Только manifest в git**

В `extras/weights_manifest.json` — URL, SHA256, arch, метрики; коллеги качают скриптом:

```bash
python extras/scripts/download_weights.py --manifest extras/weights_manifest.json
```

### 12.4. Практическая рекомендация для команды

1. **PR #1:** код + `REPORT.md` + manifest (без бинарников).
2. **Релиз весов:** HF Hub или GitHub Release с архивом:
   - минимум: `T1`, `T2`, `C3-awq`, `D1` (+ при желании остальные D*).
   - author 2M/6M — ссылка на официальный MAPF-GPT; **85M** — только HF/Release, не в основной ветке git.
3. В README добавить секцию **«Downloading trained checkpoints»** с одной командой.

### 12.5. Лицензия

Убедитесь, что лицензия upstream MAPF-GPT разрешает redistribution ваших производных весов; при публикации на HF укажите базовую модель и commit hash кода обучения.

---

## Приложение A. Скрипты (шпаргалка)

```bash
# Eval одной модели
python extras/scripts/eval_pogema.py \
  --custom-weights extras/runs/trim_ft/2026-05-15_20-36-05_L4_cekl/ckpt_best.pt \
  --custom-model-name 2M-L4-ce_kl \
  --output-root extras/runs/trim_ft

# Latency
python extras/scripts/bench_latency.py \
  --weights extras/runs/trim_ft/2026-05-15_20-36-05_L4_cekl/ckpt_best.pt \
  --label 2M-L4-ce_kl --dtype bfloat16

# Train distill
python extras/scripts/train_distill.py \
  --config extras/configs/distill/with_hidden.json \
  --student extras/configs/students/student-1M.json \
  --teacher weights/model-85M.pt \
  --out-dir extras/runs/distill/my_run
```

---

## Приложение B. Ссылки

- [MAPF-GPT](https://github.com/CognitiveAISystems/MAPF-GPT)
- [POGEMA](https://arxiv.org/abs/2407.14931)
- Локальные отчёты: `extras/runs/reports/final_2026-05-17/`

---

*Документ сгенерирован по финальным run-ам от 2026-05-15 — 2026-05-17. При обновлении экспериментов пересоберите таблицы через `compare_runs.py` и обновите §9.*
