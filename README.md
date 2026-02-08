# Kodik Lab

Небольшая Streamlit-песочница для анализа графов (метрики, 3D, нулевые модели, атаки/устойчивость). 

## Как запустить

1) Установить зависимости:

```bash
pip install -r requirements.txt
```

2) Запустить:

```bash
streamlit run app.py
```

## Разработка (lint + тесты)

```bash
pip install -r requirements.txt -r requirements-dev.txt
ruff check .
pytest -q
```

## Данные

Можно:
- загрузить CSV / Excel с рёбрами
- или создать демо-граф из сайдбара

Ожидаемый минимальный набор колонок:
- `src` — источник
- `dst` — цель

Опционально (если есть):
- `weight` — вес ребра (число)
- `confidence` — уверенность (0–100)

Если файл "нестандартный" — после загрузки появится блок **"Сопоставление колонок"**, где можно вручную выбрать колонки.

### Политика весов (важно)

В приложении веса рёбер интерпретируются как *сила связи* / *пропускная способность*.
Ряд алгоритмов требует **строго положительных** весов (например, когда расстояние
определяется как `dist = 1/weight`).

Если исходные данные содержат нулевые/отрицательные веса (типичный случай —
корреляционные матрицы), поведение задаётся настройками:

- `WEIGHT_POLICY` в `src/config.py`: `drop_nonpositive` (по умолчанию), `abs`, `clip`, `shift`.
- `WEIGHT_EPS` — эпсилон для режимов `clip/shift`.
- `WEIGHT_SHIFT` — константа сдвига для режима `shift`.

Для строгого исследовательского режима рекомендуется `drop_nonpositive`.
Для корреляционных весов обычно используют `abs` (с потерей знака) либо `shift/clip`
(в зависимости от смысла весов).

## Design parameters and heuristics

This project is an interactive research tool. Some numerical and runtime-related
parameters are heuristic by design. They are centralized in `src/config.py`.

### Numerical stability
- `EPS_W`, `EPS_LOG` protect against division/log singularities.

### Ricci / W₁ approximation
- `RICCI_MASS_SCALE` discretizes neighbor measures for min-cost flow.
- `RICCI_MAX_SUPPORT`, `RICCI_CUTOFF`, `RICCI_SAMPLE_EDGES` limit runtime on large graphs.

### Phase-transition detection (recommended)
We detect jumps in graph metrics using a null-model-based threshold:

1. Generate a null ensemble (degree-preserving rewires).
2. Compute jump scores on the same trajectory.
3. Set threshold as the `1 - JUMP_ALPHA` quantile of null jumps.
4. Flag transition if `jump_fraction >= threshold`.

`0.35` is used only as a fallback quick-screening value when null-model
statistics are not available.

## Что внутри

- **Dashboard**: быстрые метрики + распределения.
- **Energy**: динамика/потоки (тут всё ещё есть спорные решения).
- **3D**: сцена графа и подсветки.
- **Null**: генерация нулевых графов (для сравнения).
- **Attack**: удаление узлов/рёбер и графики деградации.
- **Compare**: сравнение графов в воркспейсе.

## Экспорт

На Dashboard есть:
- **Node metrics CSV** (degree, strength, clustering)
- экспорт графа в **GraphML/GEXF** (удобно для Gephi)

Для экспорта графиков в PNG нужна библиотека `kaleido`. Она добавлена в requirements.

## Known bugs / странности

- На очень больших графах часть метрик может "залипать" (ограничения по CPU/RAM, плюс Streamlit кэшируется не всегда предсказуемо).
- Ricci считается долго. Прогрессбар есть, но счёт идёт последовательно (иначе joblib не даёт адекватный прогресс).
- Если накидать десяток графов/экспов, браузер может начать жрать память: есть кнопка **"Trim state"** и лимиты в `src/config.py`.

## Почему местами неидеально

Потому что проект живой: что-то делалось ради скорости исследования, что-то — ради UI. Если хочется "идеальной архитектуры", это отдельная задача.


## Воспроизводимость

- В местах, где используется рандомизация (нулевые модели, аппроксимации), фиксируй `seed` в UI.
- Для научных прогонов сохраняй параметры эксперимента вместе с результатами (экспорт в CSV/JSON).

## Ограничения и допущения

- Веса рёбер (`weight`) интерпретируются как **неотрицательная** мера силы/пропускной способности связи.
- Для некоторых метрик используется преобразование расстояний `dist = 1/weight` — это допущение. Для корреляционных весов (в т.ч. отрицательных) нужна отдельная политика препроцессинга.
- Часть расчётов на больших графах использует аппроксимации/ограничения по времени (например, sampling в betweenness). Это может менять ranking, но ускоряет интерактивный UI.
