# План реализации новых метрик для Ayase

## Общие сведения об архитектуре Ayase

Перед описанием каждой метрики -- краткое резюме архитектурных паттернов, выявленных при анализе кодовой базы.

### Структура модуля

Каждый модуль -- это файл `src/ayase/modules/<module_name>.py`, содержащий один класс, наследуемый от одного из базовых:
- `PipelineModule` (в `ayase.pipeline`) -- основной базовый класс для per-sample метрик
- `ReferenceBasedModule` (в `ayase.base_modules`) -- для full-reference метрик (VMAF, DISTS и т.д.)
- `BatchMetricModule` (в `ayase.base_modules`) -- для dataset-level distribution метрик (FVD, KVD и т.д.)
- `NoReferenceModule` (в `ayase.base_modules`) -- для no-reference метрик с упрощённым API

### Обязательные элементы класса

```python
class MyModule(PipelineModule):
    name = "my_module"                    # Уникальное имя, используется для авто-регистрации
    description = "..."                    # Краткое описание
    default_config = {"param": value}      # Конфигурация по умолчанию

    def __init__(self, config=None):       # Инициализация с _ml_available = False
    def setup(self):                       # Загрузка моделей (tiered backends)
    def process(self, sample: Sample) -> Sample:  # Основная логика
```

### Регистрация

1. Класс авто-регистрируется через `ModuleRegistry` при установке `name != "unnamed_module"` (в `__init_subclass__`)
2. Добавить запись в `src/ayase/modules/__init__.py` в список `_IMPORTS`
3. Для dataset-level метрик: добавить поле в `DatasetStats` (в `models.py`)
4. Для per-sample метрик: добавить поле(я) в `QualityMetrics` (в `models.py`) + запись в `_FIELD_GROUPS`

### Паттерн tiered backends

Каждый модуль поддерживает несколько бэкендов (от лучшего к fallback), выбираемых в `setup()`. При отсутствии ML-зависимостей модуль корректно возвращает sample без изменений.

### Конфигурация extra="forbid"

`QualityMetrics` использует `model_config = ConfigDict(extra="forbid")`, то есть все новые поля **обязательно** нужно объявить в модели.

### Тестирование

Тесты хранятся в `tests/modules/per_module/test_<module_name>.py`. Используются фикстуры из `tests/modules/conftest.py`: `image_sample`, `video_sample`, `synthetic_image`, `synthetic_video`. Стандартная функция `_test_module_basics(ModuleCls, "name")` для базовой валидации.

---

## Метрика 1: KID (Kernel Inception Distance)

### Назначение

KID измеряет расстояние между распределениями сгенерированных и референсных изображений через Maximum Mean Discrepancy (MMD) в пространстве Inception features. В отличие от FID, KID имеет несмещённую оценку и работает корректно на малых выборках (от 50 изображений). Это dataset-level метрика.

### Файл и расположение

`F:/projects/ayase/src/ayase/modules/kid.py`

### Структура класса

```python
class KIDModule(BatchMetricModule):
    name = "kid"
    description = "Kernel Inception Distance for image generation evaluation (batch metric)"
    default_config = {
        "feature_layer": "2048",          # Inception feature layer
        "subset_size": 100,               # Подвыборка для KID
        "num_subsets": 100,               # Количество подвыборок
        "degree": 3,                      # Степень полиномиального ядра
        "gamma": None,                    # Gamma для ядра (None = auto)
        "coef0": 1.0,                     # Коэффициент полиномиального ядра
        "device": "auto",
        "batch_size": 32,
        "resize": 299,                    # Размер для Inception
    }
```

Наследуется от `BatchMetricModule` (аналогично `FVDModule`). Реализует:
- `setup()`: загрузка InceptionV3 (уже используется в `inception_score.py`, можно переиспользовать)
- `extract_features(sample)`: извлечение Inception features из изображения (или кадров видео)
- `compute_distribution_metric(features, reference_features)`: вычисление KID через polynomial kernel MMD

### Tiered backends

1. **clean-fid** (`cleanfid`) -- библиотека с оптимизированной реализацией KID
2. **torch-fidelity** (`torch_fidelity`) -- альтернативная реализация
3. **Нативная реализация** -- InceptionV3 + numpy polynomial kernel MMD (аналогично FVD, но с MMD вместо Frechet distance)

### Зависимости

Добавить в `pyproject.toml`:
- Новая optional-dependency group `v-kid`:
  ```toml
  v-kid = [
      "torch>=2.1.0,<3.0",
      "torchvision>=0.16.0,<1.0",
      "clean-fid>=0.1.35",
  ]
  ```
- Также добавить `clean-fid` в группу `ml`

### Изменения в models.py

В `DatasetStats` добавить:
```python
kid: Optional[float] = None  # Kernel Inception Distance (lower=better)
kid_std: Optional[float] = None  # KID standard deviation
```

### Конфигурация

```toml
[modules.kid]
subset_size = 100
num_subsets = 100
degree = 3
device = "auto"
batch_size = 32
```

### Вход/выход

- **Вход**: набор изображений (через pipeline scan) + опционально reference_path для каждого sample
- **Выход**: dataset-level метрика `kid` в `DatasetStats`, записывается через `self.pipeline.add_dataset_metric("kid", score)`

### План тестирования

Файл: `tests/modules/per_module/test_kid.py`

1. `test_kid_basics()` -- базовая проверка name/description/config
2. `test_kid_skip_without_ml()` -- корректный пропуск при отсутствии torch
3. `test_kid_feature_extraction(image_sample)` -- извлечение features из синтетического изображения
4. `test_kid_compute_same_distribution()` -- KID между двумя идентичными наборами features должен быть ~0
5. `test_kid_compute_different_distributions()` -- KID между разными должен быть > 0
6. `test_kid_dataset_stats_field()` -- проверка существования поля в DatasetStats

### Оценка трудозатрат

**2-3 дня**. Основная сложность -- корректная реализация polynomial MMD с подвыборками. При использовании clean-fid большая часть логики делегируется библиотеке.

---

## Метрика 2: ImageReward

### Назначение

ImageReward -- модель предсказания человеческих предпочтений для text-to-image генерации. На вход принимает изображение + текстовый prompt, на выходе -- reward score, коррелирующий с человеческой оценкой качества. Это per-sample метрика.

### Файл и расположение

`F:/projects/ayase/src/ayase/modules/image_reward.py`

### Структура класса

```python
class ImageRewardModule(PipelineModule):
    name = "image_reward"
    description = "Human preference prediction for text-to-image quality (ImageReward)"
    default_config = {
        "model_name": "ImageReward-v1.0",
        "num_frames": 5,                  # Для видео: сколько кадров оценивать
        "warning_threshold": 0.0,         # Порог для warning (ImageReward 0 = нейтрально)
    }
```

Наследуется от `PipelineModule` (аналогично `AestheticModule` и `SemanticAlignmentModule`). Объединяет паттерны обоих: требует caption (как semantic_alignment) и выдаёт скалярную оценку (как aesthetic).

### Tiered backends

1. **image-reward** (`ImageReward`) -- основная библиотека
2. **Heuristic fallback** -- CLIP score * aesthetic score (грубое приближение). Если оба уже вычислены другими модулями, можно использовать как proxy.

### Зависимости

Добавить в `pyproject.toml`:
- Новая optional-dependency group `v-reward`:
  ```toml
  v-reward = [
      "torch>=2.1.0,<3.0",
      "torchvision>=0.16.0,<1.0",
      "transformers>=4.36.0,<5.0",
      "image-reward>=1.5",
  ]
  ```
- Добавить `image-reward>=1.5` в группу `ml`

### Изменения в models.py

В `QualityMetrics`:
```python
image_reward_score: Optional[float] = None  # Human preference reward (-2..+2, higher=better)
```

В `_FIELD_GROUPS`:
```python
"image_reward_score": "alignment",
```

### Конфигурация

```toml
[modules.image_reward]
model_name = "ImageReward-v1.0"
num_frames = 5
warning_threshold = 0.0
```

### Вход/выход

- **Вход**: изображение/видео (sample.path) + текстовый prompt (sample.caption.text или .txt файл)
- **Выход**: `sample.quality_metrics.image_reward_score` (float, типичный диапазон -2..+2, выше = лучше)
- При отсутствии caption -- модуль корректно пропускает sample (аналогично `SemanticAlignmentModule`)

### Ключевые моменты реализации

Логика `process()`:
1. Получить caption из `sample.caption.text` или `.txt` файла рядом (паттерн из `semantic_alignment.py`, строки 59-68)
2. Если caption отсутствует -- `return sample`
3. Загрузить кадры через `_load_frames(sample)` (стандартный паттерн)
4. Для каждого кадра: `score = self._model.score(prompt, image)` (API ImageReward)
5. Усреднить scores, записать в `sample.quality_metrics.image_reward_score`

### План тестирования

Файл: `tests/modules/per_module/test_image_reward.py`

1. `test_image_reward_basics()` -- базовая проверка
2. `test_image_reward_skip_no_caption(image_sample)` -- пропуск при отсутствии caption
3. `test_image_reward_skip_no_ml(image_sample)` -- пропуск при отсутствии библиотеки
4. `test_image_reward_field_exists()` -- проверка поля в QualityMetrics
5. `test_image_reward_field_group()` -- проверка группировки в "alignment"

### Оценка трудозатрат

**1-2 дня**. Библиотека image-reward предоставляет простой API. Основная работа -- интеграция с паттернами Ayase и обработка краевых случаев.

---

## Метрика 3: Face Cross-Similarity Matrix

### Назначение

Вычисление матрицы попарной косинусной схожести лиц (ArcFace embeddings) между N изображениями. Используется для оценки консистентности идентичности персонажей в наборах изображений/видео.

### Файл и расположение

`F:/projects/ayase/src/ayase/modules/face_cross_similarity.py`

Рассматривалась возможность расширения существующего `identity_loss.py`, однако предпочтительнее создать отдельный модуль, поскольку:
- `identity_loss` -- per-sample метрика (1 sample vs 1 reference)
- `face_cross_similarity` -- batch/cross-sample метрика (NxN матрица)
- Разная гранулярность вывода (скаляр vs матрица)

При этом модуль будет переиспользовать тот же InsightFace/ArcFace бэкенд.

### Структура класса

```python
class FaceCrossSimilarityModule(PipelineModule):
    name = "face_cross_similarity"
    description = "Pairwise ArcFace cosine similarity matrix across dataset faces"
    default_config = {
        "model_name": "buffalo_l",
        "max_faces_per_image": 5,          # Максимум лиц на изображение
        "similarity_threshold": 0.3,       # Порог для считания identity match
        "device": "auto",
    }
```

Наследуется от `PipelineModule`, но использует `post_process(all_samples)` для вычисления cross-sample метрик после обработки всех samples.

### Tiered backends

Идентично `IdentityLossModule` (строки 52-86 в `identity_loss.py`):
1. **InsightFace** (buffalo_l ArcFace)
2. **DeepFace** (ArcFace)
3. **MediaPipe FaceMesh** (геометрические landmarks) -- менее точный, но работает без тяжёлых зависимостей

### Зависимости

Использует те же зависимости, что уже объявлены в `v-identity`:
```toml
v-identity = [
    "insightface>=0.7.0",
    "onnxruntime>=1.14.0",
]
```

Дополнительных зависимостей не требуется.

### Изменения в models.py

В `QualityMetrics`:
```python
face_cross_similarity: Optional[float] = None  # Avg pairwise face similarity (0-1, higher=more consistent)
face_identity_count: Optional[int] = None  # Number of unique identities detected
```

В `_FIELD_GROUPS`:
```python
"face_cross_similarity": "face",
"face_identity_count": "face",
```

В `DatasetStats`:
```python
face_similarity_matrix: Optional[List[List[float]]] = None  # NxN pairwise similarity
avg_face_cross_similarity: Optional[float] = None  # Dataset-level average
identity_cluster_count: Optional[int] = None  # Number of identity clusters
```

### Конфигурация

```toml
[modules.face_cross_similarity]
model_name = "buffalo_l"
max_faces_per_image = 5
similarity_threshold = 0.3
```

### Вход/выход

- **Вход**: набор изображений/видео (каждый sample обрабатывается в `process()`)
- **Per-sample выход**: `face_cross_similarity` (среднее сходство с остальными samples), `face_identity_count` (количество лиц)
- **Dataset-level выход**: NxN матрица, кластеры идентичностей -- через `post_process(all_samples)` и `pipeline.add_dataset_metric()`

### Ключевые моменты реализации

1. В `process()`: извлечь face embeddings и кэшировать в `self._embeddings_cache[sample.path]`
2. В `post_process(all_samples)`:
   - Построить NxN матрицу попарной косинусной схожести
   - Кластеризация идентичностей (agglomerative clustering по порогу)
   - Для каждого sample записать средний cross-similarity score
   - Записать dataset-level метрики

### План тестирования

Файл: `tests/modules/per_module/test_face_cross_similarity.py`

1. `test_face_cross_similarity_basics()` -- базовая проверка
2. `test_face_cross_similarity_empty_cache()` -- post_process с пустым кэшем
3. `test_face_cross_similarity_cosine_math()` -- unit-тест вычисления косинусного сходства
4. `test_face_cross_similarity_clustering()` -- тест кластеризации с синтетическими embeddings
5. `test_face_cross_similarity_fields()` -- проверка полей в QualityMetrics и DatasetStats

### Оценка трудозатрат

**3-4 дня**. Модуль сложнее остальных: cross-sample логика, кластеризация, кэширование embeddings, поддержка multi-face.

---

## Метрика 4: LPIPS (image-to-image)

### Назначение

Перцептуальное расстояние между парами изображений. В Ayase уже есть LPIPS для видео (`st_lpips`, `flolpips`, `i2v_similarity`), но нет чистого image-to-image LPIPS. Дополнительно: diversity metric -- среднее попарное LPIPS по набору изображений.

### Файл и расположение

`F:/projects/ayase/src/ayase/modules/image_lpips.py`

Имя `image_lpips` для отличия от существующих `st_lpips` и `flolpips`, а также от `i2v_lpips`.

### Структура класса

```python
class ImageLPIPSModule(PipelineModule):
    name = "image_lpips"
    description = "LPIPS perceptual distance between image pairs and diversity metric"
    default_config = {
        "net": "alex",                     # "alex", "vgg", "squeeze"
        "resize": 256,                     # Размер для LPIPS
        "diversity_max_pairs": 500,        # Максимум пар для diversity
    }
```

Наследуется от `PipelineModule`. Для per-sample full-reference LPIPS логика аналогична `ReferenceBasedModule`, но нам также нужна diversity-метрика через `post_process()`.

### Tiered backends

1. **lpips** (уже в зависимостях `v-perceptual`) -- основной бэкенд
2. **Heuristic** -- SSIM-based proxy через OpenCV (грубое приближение)

### Зависимости

`lpips>=0.1.4` уже включён в `v-perceptual` (строка 55 в `pyproject.toml`). Дополнительных зависимостей не требуется.

### Изменения в models.py

В `QualityMetrics`:
```python
image_lpips: Optional[float] = None  # LPIPS perceptual distance vs reference (0-1, lower=more similar)
```

В `_FIELD_GROUPS`:
```python
"image_lpips": "fr_quality",
```

В `DatasetStats`:
```python
lpips_diversity: Optional[float] = None  # Average pairwise LPIPS across dataset (higher=more diverse)
```

### Конфигурация

```toml
[modules.image_lpips]
net = "alex"
resize = 256
diversity_max_pairs = 500
```

### Вход/выход

- **Per-sample (full-reference)**: если есть `sample.reference_path`, вычисляется LPIPS distance между sample и reference, записывается в `sample.quality_metrics.image_lpips`
- **Dataset-level (diversity)**: в `post_process(all_samples)` вычисляется среднее попарное LPIPS по набору (подвыборка до `diversity_max_pairs` пар). Записывается через `pipeline.add_dataset_metric("lpips_diversity", value)`

### Ключевые моменты реализации

1. В `setup()`: загрузить LPIPS модель (аналогично `st_lpips.py`, строки 56-69)
2. В `process()`:
   - Если есть reference_path: вычислить LPIPS(sample, reference), записать в `image_lpips`
   - Кэшировать LPIPS-compatible тензор для каждого sample (для diversity)
3. В `post_process()`:
   - Взять случайную подвыборку пар из кэша
   - Вычислить средний LPIPS distance
   - Записать в DatasetStats

### План тестирования

Файл: `tests/modules/per_module/test_image_lpips.py`

1. `test_image_lpips_basics()` -- базовая проверка
2. `test_image_lpips_same_image()` -- LPIPS между изображением и самим собой = ~0
3. `test_image_lpips_skip_no_reference(image_sample)` -- пропуск без reference
4. `test_image_lpips_diversity_empty()` -- diversity с недостаточно samples
5. `test_image_lpips_field_exists()` -- проверка полей

### Оценка трудозатрат

**2 дня**. LPIPS API простой, основная работа -- diversity metric и кэширование тензоров.

---

## Метрика 5: Concept Presence Detection

### Назначение

Обнаружение присутствия заданного концепта в изображении: лица (через face detection + подсчёт), объекты (через CLIP-based detection), стили (через CLIP-based классификацию). Выход: булево присутствие + confidence score.

### Файл и расположение

`F:/projects/ayase/src/ayase/modules/concept_presence.py`

### Структура класса

```python
class ConceptPresenceModule(PipelineModule):
    name = "concept_presence"
    description = "Detect concept presence via face detection, CLIP-based object/style detection"
    default_config = {
        "detection_mode": "auto",          # "auto", "face", "clip", "combined"
        "clip_model": "openai/clip-vit-base-patch32",
        "clip_threshold": 0.25,            # Порог CLIP similarity для наличия концепта
        "face_detection_confidence": 0.5,  # Порог уверенности face detection
        "concepts": [],                    # Список концептов для проверки (пустой = из caption)
        "num_frames": 5,                   # Для видео
    }
```

Наследуется от `PipelineModule`.

### Tiered backends

**Face detection sub-module**:
1. InsightFace detector (из `v-identity`)
2. MediaPipe FaceDetection (из `v-face`)
3. OpenCV Haar cascade (всегда доступен, в core dependencies)

**CLIP-based detection sub-module**:
1. Transformers CLIP (аналогично `semantic_alignment.py`)
2. open-clip-torch (альтернатива)
3. Heuristic -- template matching / color histogram (очень грубо, только для fallback)

### Зависимости

Переиспользует существующие:
- `v-perceptual` (для open-clip) или `v-text` (для transformers CLIP)
- `v-face` (для MediaPipe)
- `v-identity` (для InsightFace)

Дополнительных зависимостей не требуется.

### Изменения в models.py

В `QualityMetrics`:
```python
concept_presence: Optional[float] = None  # Concept presence confidence (0-1, higher=more confident)
concept_count: Optional[int] = None  # Number of detected instances of target concept
concept_face_count: Optional[int] = None  # Number of faces detected
```

В `_FIELD_GROUPS`:
```python
"concept_presence": "scene",
"concept_count": "scene",
"concept_face_count": "face",
```

### Конфигурация

```toml
[modules.concept_presence]
detection_mode = "auto"
clip_model = "openai/clip-vit-base-patch32"
clip_threshold = 0.25
face_detection_confidence = 0.5
concepts = ["a person wearing a hat", "sunset"]
num_frames = 5
```

### Вход/выход

- **Вход**: изображение/видео + опционально список концептов (через config или caption)
- **Выход**:
  - `concept_presence` -- максимальная confidence среди проверенных концептов
  - `concept_count` -- количество обнаруженных экземпляров
  - `concept_face_count` -- количество обнаруженных лиц
  - `sample.validation_issues` -- ValidationIssue при отсутствии ожидаемого концепта

### Ключевые моменты реализации

1. В `setup()`: инициализировать face detector и CLIP модель (tiered)
2. В `process()`:
   - Определить режим (`detection_mode`):
     - `"auto"`: если concepts содержат face-related terms -> face, иначе -> clip
     - `"face"`: только face detection
     - `"clip"`: только CLIP similarity
     - `"combined"`: оба
   - Face detection: подсчёт лиц, запись в `concept_face_count`
   - CLIP detection: вычислить cosine similarity между изображением и каждым concept text, сравнить с threshold
   - Записать максимальную confidence в `concept_presence`, общий count в `concept_count`

### План тестирования

Файл: `tests/modules/per_module/test_concept_presence.py`

1. `test_concept_presence_basics()` -- базовая проверка
2. `test_concept_presence_skip_no_ml(image_sample)` -- пропуск без ML
3. `test_concept_presence_face_counting()` -- face detection на синтетическом изображении
4. `test_concept_presence_clip_threshold()` -- проверка порога CLIP similarity
5. `test_concept_presence_auto_mode()` -- auto-определение режима
6. `test_concept_presence_fields()` -- проверка полей

### Оценка трудозатрат

**3-4 дня**. Комбинированный модуль с несколькими sub-backends. Основная сложность -- auto-detection режима и корректное объединение результатов face + CLIP.

---

## Порядок реализации и зависимости

Рекомендуемый порядок (от простого к сложному, с учётом зависимостей):

| # | Метрика | Трудозатраты | Зависит от |
|---|---------|-------------|------------|
| 1 | LPIPS (image-to-image) | 2 дня | -- (lpips уже в зависимостях) |
| 2 | ImageReward | 1-2 дня | -- |
| 3 | KID | 2-3 дня | -- |
| 4 | Concept Presence | 3-4 дня | -- (но переиспользует CLIP из semantic_alignment) |
| 5 | Face Cross-Similarity | 3-4 дня | -- (но переиспользует ArcFace из identity_loss) |

**Итого: 11-15 рабочих дней**

### Общие файлы, требующие изменений

Для каждой метрики:

1. **`src/ayase/models.py`** -- добавить поля в `QualityMetrics` (+ `_FIELD_GROUPS`) и/или `DatasetStats`
2. **`src/ayase/modules/__init__.py`** -- добавить строку в `_IMPORTS`
3. **`pyproject.toml`** -- добавить зависимости (только для KID и ImageReward)
4. **Новый файл модуля** -- `src/ayase/modules/<name>.py`
5. **Новый файл теста** -- `tests/modules/per_module/test_<name>.py`

### Сводная таблица новых полей

**QualityMetrics (per-sample)**:
| Поле | Тип | Группа | Модуль |
|------|-----|--------|--------|
| `image_reward_score` | float | alignment | image_reward |
| `face_cross_similarity` | float | face | face_cross_similarity |
| `face_identity_count` | int | face | face_cross_similarity |
| `image_lpips` | float | fr_quality | image_lpips |
| `concept_presence` | float | scene | concept_presence |
| `concept_count` | int | scene | concept_presence |
| `concept_face_count` | int | face | concept_presence |

**DatasetStats (dataset-level)**:
| Поле | Тип | Модуль |
|------|-----|--------|
| `kid` | float | kid |
| `kid_std` | float | kid |
| `lpips_diversity` | float | image_lpips |
| `face_similarity_matrix` | List[List[float]] | face_cross_similarity |
| `avg_face_cross_similarity` | float | face_cross_similarity |
| `identity_cluster_count` | int | face_cross_similarity |

### Сводная таблица новых записей в __init__.py

```python
# В список _IMPORTS добавить:
("KIDModule", ".kid"),
("ImageRewardModule", ".image_reward"),
("FaceCrossSimilarityModule", ".face_cross_similarity"),
("ImageLPIPSModule", ".image_lpips"),
("ConceptPresenceModule", ".concept_presence"),
```

### Потенциальные риски и вызовы

1. **KID**: библиотека `clean-fid` может иметь конфликты версий с `torch-fidelity` (если присутствует). Рекомендуется выбрать одну из двух.

2. **ImageReward**: библиотека `image-reward` тянёт `transformers` и `torch` определённых версий. Необходимо проверить совместимость с уже объявленными ограничениями в pyproject.toml.

3. **Face Cross-Similarity**: кэширование embeddings для всего датасета может потребовать значительной памяти. Необходимо реализовать ограничение на максимальное количество кэшированных embeddings.

4. **Concept Presence**: auto-detection режима по ключевым словам ("face", "person", "people") может быть ненадёжным. Рекомендуется предоставить пользователю возможность явно указать режим через config.

5. **LPIPS diversity**: для больших датасетов (10000+ изображений) подсчёт всех пар невозможен (O(N^2)). Подвыборка `diversity_max_pairs` решает проблему, но нужно обеспечить репрезентативность выборки.

---

### Critical Files for Implementation
- `F:/projects/ayase/src/ayase/models.py` - Must add new fields to QualityMetrics (_FIELD_GROUPS + field definitions) and DatasetStats for all 5 metrics
- `F:/projects/ayase/src/ayase/modules/__init__.py` - Must register all 5 new module classes in the _IMPORTS list
- `F:/projects/ayase/src/ayase/base_modules.py` - Contains BatchMetricModule base class that KID must extend, and ReferenceBasedModule pattern for LPIPS
- `F:/projects/ayase/src/ayase/modules/identity_loss.py` - Reference implementation for Face Cross-Similarity (reuse ArcFace tiered backend pattern from lines 52-86)
- `F:/projects/ayase/src/ayase/modules/semantic_alignment.py` - Reference implementation for ImageReward and Concept Presence (caption loading pattern from lines 59-68, CLIP model loading from lines 33-48)