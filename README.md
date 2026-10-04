# Beam-Surrogate

## CPU-суррогат пучка

Проект считает пучок C⁶⁺ 480 кэВ через два соленоида модулем Beam Tracing и обучает MLP предсказывать остаток поперечного движения относительно свободного пролёта на 1.5 м. Сеть и нормировка совпадают с ноутбуком `4 курс/Progect_V2_1_ipynb_.ipynb`. Расчёт и обучение идут на CPU, не больше чем на 12 логических ядрах.

Батчи считаются по очереди. До импорта Numba и PyTorch выставляются `NUMBA_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS` и `OPENBLAS_NUM_THREADS`. У PyTorch 12 intra-op потоков и 1 inter-op поток.

## Установка

Из этой папки, с CPU-колёсами PyTorch:

```bash
uv sync --extra dev
```

Без uv:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e "../Beam Tracing"
pip install -e ".[dev]"
```

## Запуск

Короткий профиль (2×64 частицы, урезанная сетка, 1 эпоха):

```bash
python -m beam_surrogate.generate --config configs/smoke.yaml
python -m beam_surrogate.train --config configs/smoke.yaml
python -m beam_surrogate.evaluate --config configs/smoke.yaml
```

Рабочий профиль: 40×2000 частиц, 1200 шагов, сетка поля 300×2000, 25 эпох. Файл датасета `data/structured_beam_data.h5`, веса `artifacts/best_beam_model.pth`.

```bash
python -m beam_surrogate.generate --config configs/cpu.yaml
python -m beam_surrogate.train --config configs/cpu.yaml
python -m beam_surrogate.evaluate --config configs/cpu.yaml
```

Потолок ядер задаётся ключом `max_cpus` (по умолчанию 12). Если у машины меньше ядер, берётся фактическое число.

## MLflow

Генерация, обучение и проверка пишут параметры, время и метрики в локальную базу `mlflow.db` (эксперимент `beam-surrogate`). Отдельные запуски называются `generate`, `train` и `evaluate`. Веса и история лежат в `mlartifacts`.

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Интерфейс открывается на http://127.0.0.1:5000.
