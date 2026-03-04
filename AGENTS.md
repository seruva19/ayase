# AGENTS.md — Coding LLM Instructions for Ayase

## Project Overview

Ayase is a modular media quality metrics toolkit. It provides 238 pipeline modules across 198 files for analyzing video/image quality, text-video alignment, motion, temporal consistency, safety, audio, and more.

- **Language:** Python 3.9+
- **Build:** Hatchling (`pyproject.toml`)
- **License:** MIT
- **Entry point:** `ayase = "ayase.cli:app"` (Typer CLI)
- **Source layout:** `src/ayase/` (src-layout)
- **Version:** defined in `src/ayase/__init__.py` AND `pyproject.toml` — keep both in sync

## Directory Structure

```
src/ayase/
├── __init__.py          # Public API: AyasePipeline, PipelineProfile, load_profile, instantiate_profile_modules
├── __main__.py          # python -m ayase
├── cli.py               # Typer CLI (scan, run, filter, stats, tui, modules, config)
├── config.py            # AyaseConfig (pydantic-settings), resolve_model_path, download_model_file
├── models.py            # Sample, QualityMetrics (~175 fields), ValidationIssue, DatasetStats
├── pipeline.py          # Pipeline, PipelineModule (ABC), ModuleRegistry, AyasePipeline
├── profile.py           # PipelineProfile, load_profile, instantiate_profile_modules
├── scanner.py           # DatasetScanner, scan_dataset
├── base_modules.py      # ReferenceBasedModule, BatchMetricModule, NoReferenceModule
├── tui.py               # Textual TUI (6 screens)
├── video.py             # Video utilities
├── audio.py             # Audio utilities
├── modules/             # 198 module files, 238 PipelineModule subclasses
│   ├── __init__.py      # Explicit imports of ~90 key modules + __all__
│   └── *.py             # All auto-discovered at runtime via ModuleRegistry
├── third_party/         # Vendored code (DOVER, FastVQA, Kandinsky)
│   ├── dover/
│   ├── fastvqa/
│   └── kandinsky/
└── utils/
    └── sampling.py
tests/
├── test_golden_values.py
├── test_integration_synthetic.py
├── test_module_smoke.py
├── test_profiles.py
├── test_readme_contract.py
├── test_regressions.py
├── test_tui.py
└── modules/
    ├── conftest.py      # Shared fixtures: synthetic_image, synthetic_video, image_sample, video_sample
    └── test_*.py        # Per-category module tests (18 files)
```

## Critical Rules

### 1. Module Auto-Registration

Every class that inherits from `PipelineModule` and sets `name` to something other than `"unnamed_module"` is **automatically registered** via `__init_subclass__`. Do NOT manually register modules. Just subclass and set `name`.

### 2. Always Return the Sample

`process(sample) -> Sample` must **always return the sample**, even on failure. Modules must degrade gracefully — never raise, never return None.

### 3. QualityMetrics Fields Must Be Declared

Before setting `sample.quality_metrics.my_field = value`, the field must exist in the `QualityMetrics` Pydantic model in `models.py`. If you add a new metric, you must:
1. Add the field to `QualityMetrics` (as `Optional[float] = None`)
2. Add it to `_FIELD_GROUPS` dict with the correct category
3. The field must follow the naming convention: `snake_case`, suffix usually `_score` for scores

### 4. Never Break Existing Tests

768 tests must pass. Run `pytest tests/ -x -q` before considering any change complete. The TUI has 39 tests with specific widget ID contracts — see "TUI Contracts" section below.

### 5. Module `__init__.py` vs Auto-Discovery

`modules/__init__.py` explicitly imports ~90 key modules for convenience. The remaining ~105 are auto-discovered at runtime by `ModuleRegistry.discover_modules()`. If you create a new module, it works without touching `__init__.py`, but add it there if it's a commonly-used module.

## Code Style

### Formatting
- **Line length:** 100
- **Formatter:** Black (but `src/ayase/modules/` is excluded from Black)
- **Linter:** Ruff — only `E` + `F` rules; `E501`, `F401`, `F403`, `F541`, `F841` are ignored
- **Type checking:** MyPy strict on core files only (`config.py`, `models.py`, `pipeline.py`, `profile.py`)

### Conventions
- All modules use `logger = logging.getLogger(__name__)` — never `print()` in library code
- Config values accessed via `self.config.get("key", default)` in `__init__`
- Imports of heavy ML libraries (`torch`, `transformers`, etc.) are done inside methods, not at module top level
- Error handling: `try/except ImportError` for missing deps, `try/except Exception` with `logger.warning` in `process()`

## Writing a New Module

### Minimal Template

```python
import logging
from ayase.models import Sample, QualityMetrics, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class MyMetricModule(PipelineModule):
    name = "my_metric"                          # Registry key — must be unique
    description = "What this module measures"
    default_config = {
        "threshold": 50.0,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.threshold = self.config.get("threshold", 50.0)
        self._ml_available = False

    def setup(self) -> None:
        try:
            import torch  # heavy import inside method
            self._ml_available = True
        except ImportError:
            logger.warning("torch not installed. MyMetric disabled.")
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample                       # graceful skip

        try:
            score = self._compute(sample)
            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.my_metric_score = score  # field must exist in QualityMetrics

            if score < self.threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low my_metric: {score:.1f}",
                    )
                )
        except Exception as e:
            logger.warning(f"MyMetric failed: {e}")

        return sample                           # ALWAYS return sample
```

### Checklist for New Modules

- [ ] Set unique `name` class attribute
- [ ] Set `description`
- [ ] `default_config` for all tunable parameters
- [ ] Heavy imports inside `setup()` or `on_mount()`, not at top level
- [ ] Tiered backend pattern: real model → proxy → heuristic (see below)
- [ ] `process()` always returns sample
- [ ] Add `Optional[float] = None` field to `QualityMetrics` in `models.py`
- [ ] Add field to `_FIELD_GROUPS` in `models.py`
- [ ] Write tests in `tests/modules/` — use `_test_module_basics()` from conftest
- [ ] Add to `modules/__init__.py` if it's a key module (grouped by category, not alphabetical)
- [ ] Update hardcoded `README_METRICS` list and field count in `tests/test_readme_contract.py`

### Tiered Backend Pattern

All modules use automatic backend detection instead of an `enable_ml` flag. The pattern:

```python
def __init__(self, config=None):
    super().__init__(config)
    self._backend = None  # "insightface" | "clip" | "heuristic" | None

def setup(self):
    # Tier 1: Best quality
    try:
        from heavy_lib import Model
        self._model = Model(...)
        self._backend = "tier1"
        return
    except Exception:
        pass
    # Tier 2: Lighter fallback
    try:
        ...
        self._backend = "tier2"
        return
    except Exception:
        pass
    # Tier 3: Always-available heuristic
    self._backend = "heuristic"

def process(self, sample):
    if self._backend is None:
        return sample  # graceful skip
    ...
```

### Video Frame Subsampling

Standard pattern for video processing — extract uniformly spaced frames:

```python
self.subsample = self.config.get("subsample", 8)  # default: 8 frames
n = min(self.subsample, total_frames)
indices = np.linspace(0, total_frames - 1, n, dtype=int)
# Read frames via cv2.VideoCapture, convert BGR→RGB
```

### Caption and Reference Access

- **Caption:** `sample.caption.text` (primary), sidecar `.txt` file next to sample (fallback)
- **Reference image:** `getattr(sample, "reference_path", None)` — `reference_path` is NOT a Pydantic field on Sample, use `getattr` with default

### Module Ordering in `__init__.py`

Modules in `modules/__init__.py` are grouped by **functional category**, not alphabetically:
Core → Aesthetics → Text/OCR → Motion → Temporal → Alignment → NR-Quality → FR-Quality → SOTA Video → Generation → Face → Scene → Safety → Audio → HDR/Codec → Dataset → Utility

### Module Lifecycle

```
__init__(config)     # Config merge, field init, no heavy work
    ↓
on_mount()           # Called once: check deps → download weights → setup()
    ↓
on_execute()         # Called before each pipeline run
    ↓
process(sample) ×N   # Called per sample
    ↓
post_process(all)    # Called after all samples (cross-sample analysis)
    ↓
on_dispose()         # Cleanup, release GPU memory
```

- `setup()` is called by `on_mount()`. Most existing modules override `setup()` — this is the standard pattern despite being technically deprecated.
- `on_mount()` is idempotent — guarded by `self._mounted` flag.

### Specialized Base Classes

`base_modules.py` provides `ReferenceBasedModule`, `BatchMetricModule`, and `NoReferenceModule` — optional helpers used by ~25 modules. See existing modules (e.g., `vmaf.py`, `fvd.py`, `niqe.py`) for usage examples. Most modules just subclass `PipelineModule` directly.

## Model Weight Resolution

Modules that load ML models should use these utilities from `ayase.config`:

```python
from ayase.config import resolve_model_path, download_model_file

# For HuggingFace models — checks local cache first
models_dir = self.config.get("models_dir", "models")
resolved = resolve_model_path("openai/clip-vit-base-patch32", models_dir)
# Returns: "models/openai/clip-vit-base-patch32" or "models/openai--clip-vit-base-patch32" or original name

# For individual weight files — downloads if not cached
path = download_model_file("dover/DOVER.pth", "https://github.com/.../DOVER.pth", models_dir)
```

The `models_dir` value comes from `AyaseConfig.general.models_dir` and is injected into module config by `instantiate_profile_modules()`.

## Configuration System

### AyaseConfig (pydantic-settings)

Loads from: `ayase.toml` (CWD) → `~/.config/ayase/config.toml` → env vars (`AYASE_` prefix) → defaults.

```toml
[general]
models_dir = "models"
parallel_jobs = 8
cache_enabled = true

[quality]
enable_blur_detection = true
blur_threshold = 100.0

[output]
default_format = "json"
artifacts_dir = "reports"

[pipeline]
modules = []
plugin_folders = ["plugins"]

[filter]
default_mode = "list"
min_score_threshold = 60
```

### Profile System

Profiles define which modules to run and per-module config overrides:

```json
{
  "name": "t2v_evaluation",
  "modules": ["dover", "clip_temporal", "semantic_alignment", "motion"],
  "module_config": {
    "dover": {"warning_threshold": 0.3},
    "semantic_alignment": {"clip_model": "openai/clip-vit-large-patch14"}
  }
}
```

Load with `load_profile(path_or_dict)`, instantiate with `instantiate_profile_modules(profile, config)`.

## Plugin System

External plugins are auto-discovered from directories listed in `config.pipeline.plugin_folders` (default: `["plugins"]`).

Rules for plugin files:
- Place `.py` files in the plugins directory
- Files prefixed with `_` are skipped
- Subclass `PipelineModule` and set `name` — registration is automatic
- No need to modify any ayase source files

## TUI Contracts (for test compatibility)

All 39 TUI tests depend on these. Do NOT rename or remove:

**Widget IDs:** `#btn_folder`, `#btn_config`, `#module_list`, `#btn_start`, `#path_label`, `#config_panel`, `#progress`, `#log`, `#status_title`, `#btn_results`, `#results_table`, `#btn_export`, `#btn_back`, `#tree`, `#select_btn`, `#cancel_btn`, `#btn_continue`, `#readiness_table`, `#btn_json`, `#btn_csv`, `#btn_html`, `#btn_cancel`

**CSS class:** `.no-config`

**Exported classes:** `AyaseApp`, `WelcomeScreen`, `ConfigScreen`, `ExecutionScreen`, `ResultsScreen`, `FolderSelectionScreen`, `ReadinessScreen`, `ExportDialog`, `ModuleConfigWidget`

**App config:** `AyaseApp.MODES = {"welcome", "config", "execution", "results"}`, `app.theme == "monokai"`

**Results table columns:** must include `"FILE"`, `"SCORE"`, `"ISSUES"`

## Testing

### Running Tests

```bash
# Full suite (fast — excludes model downloads)
pytest tests/ -x -q

# Specific category
pytest tests/modules/test_motion_scene_semantic_metrics.py -v

# TUI tests only
pytest tests/test_tui.py -v

# Smoke test all modules (slow — may download models)
pytest tests/test_module_smoke.py -v --timeout=300
```

### Test Conventions

- Test files: `test_*.py`, functions: `test_*`
- Use `_test_module_basics(ModuleClass, "expected_name")` from `tests/modules/conftest.py` for basic attribute checks
- Use `synthetic_image`, `synthetic_video`, `image_sample`, `video_sample` fixtures from conftest
- Import modules locally inside test functions, not at file top level
- Tests must not require GPU or model downloads to pass (except `test_module_smoke.py`)

### Writing Tests for a New Module

```python
from tests.modules.conftest import _test_module_basics

def test_my_metric_basics():
    from ayase.modules.my_metric import MyMetricModule
    _test_module_basics(MyMetricModule, "my_metric")

def test_my_metric_video(video_sample):
    from ayase.modules.my_metric import MyMetricModule
    mod = MyMetricModule()
    # Don't call setup() in unit tests — test without ML deps
    result = mod.process(video_sample)
    assert result is video_sample  # always returns same object

def test_my_metric_with_ml(video_sample):
    """Requires ML deps — may be skipped in CI."""
    import pytest
    try:
        import torch
    except ImportError:
        pytest.skip("torch not available")
    from ayase.modules.my_metric import MyMetricModule
    mod = MyMetricModule()
    mod.setup()
    if not mod._ml_available:
        pytest.skip("model not loaded")
    result = mod.process(video_sample)
    assert result.quality_metrics is not None
    assert result.quality_metrics.my_metric_score is not None
```

## Downstream Integration

Ayase is used as a backend by downstream projects (e.g., vigen_metrics). Key integration patterns:

### Passing Config from Downstream

```python
from ayase.config import AyaseConfig
from ayase.profile import instantiate_profile_modules, PipelineProfile
from ayase.pipeline import Pipeline

config = AyaseConfig.load(Path("ayase.toml"))
profile = PipelineProfile(
    name="my_evaluation",
    modules=["dover", "clip_temporal", "semantic_alignment"],
    module_config={
        "ocr_fidelity": {"expected_text": "Hello World"},      # explicit values
        "motion_amplitude": {"expected_motion": "fast"},        # instead of heuristics
        "action_recognition": {"expected_action": "dancing"},
    }
)
modules = instantiate_profile_modules(profile, config=config)
pipeline = Pipeline(modules)
pipeline.start()
```

### Evaluation Modules with Explicit Config

Three modules accept explicit expected values for evaluation use cases (bypassing caption-based heuristics):
- `ocr_fidelity`: `expected_text` (str or list) — text that should appear in video
- `motion_amplitude`: `expected_motion` (`"large"`/`"fast"`/`"medium"` or `"slow"`/`"small"`) — expected motion class
- `action_recognition`: `expected_action` (str) — expected action description for CLIP matching

## Vendored Third-Party Code

`src/ayase/third_party/` contains vendored copies of:
- **DOVER** (ICCV 2023) — video quality assessment models
- **FastVQA** — fast video quality assessment
- **Kandinsky** — video motion predictor

These have known issues:
- 49 `print()` statements (use stdout instead of logging)
- 6 bare `except:` clauses (swallow KeyboardInterrupt)
- Do NOT run Black/Ruff on these — they are external code

## Release Checklist

After implementing any change (new modules, new metrics, bug fixes), complete ALL applicable steps before considering done:

### 1. Code

- [ ] Implement modules in `src/ayase/modules/`
- [ ] Add `Optional[float] = None` fields to `QualityMetrics` in `models.py`
- [ ] Add fields to `_FIELD_GROUPS` in `models.py`
- [ ] Register in `modules/__init__.py` if it's a key module
- [ ] Add optional deps to `pyproject.toml` if needed (and update `all` extras)

### 2. Tests

- [ ] Write tests in `tests/modules/test_*.py`
- [ ] Update `tests/test_readme_contract.py`: add fields to `README_METRICS` list, update field count (currently 231)
- [ ] Run `pytest tests/ -x -q` — full regression must pass

### 3. README.md

- [ ] Update metrics table — 5-column format: `#`, `Metric`, `Module`, `Input`, `Description`
- [ ] Metric names must exactly match `QualityMetrics` field names (no backticks)
- [ ] Input shorthand: `img/vid` = image or video, `+ref` = needs `reference_path`, `+cap` = needs caption, `batch` = dataset-level
- [ ] Renumber rows after insertions
- [ ] Update overview module/metric counts
- [ ] Validated by `tests/modules/test_metrics_table.py` — one row per `QualityMetrics` field, in order

### 4. MODELS.md

- [ ] Add new model weights with Used By, License, Size
- [ ] Update "Used By" column for existing models reused by new modules
- [ ] Add to tiered fallback table (Section 7) if module has multi-backend
- [ ] Update license summary and storage estimates

### 5. CHANGELOG.md

- [ ] Add entries under `[Unreleased]` — follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format
- [ ] Sections: Added, Changed, Removed

### 6. Lint & Commit

- [ ] `ruff check` on new/changed files
- [ ] Commit with a concise message describing what was added/changed

### 7. PyPI Publish

- [ ] Bump version in both `pyproject.toml` and `src/ayase/__init__.py` (keep in sync)
- [ ] `python -m build && twine upload dist/*`

## Contract Tests

`tests/test_readme_contract.py` enforces structural invariants. When adding metrics, you MUST update:

1. **Field count** — `test_quality_metrics_has_231_fields()` asserts exact count
2. **`README_METRICS` list** — hardcoded list of all field names in order; used by:
   - `test_readme_table_count()` — list length matches field count
   - `test_readme_metric_exists_in_model()` — every listed name is a QualityMetrics field
   - `test_no_unlisted_fields()` — every QualityMetrics field appears in the list
3. **`tests/modules/test_metrics_table.py`** — parses README `## Metrics` table and verifies 1:1 match with QualityMetrics fields in order

Other contract tests: golden values (`test_golden_values.py`, ±2% tolerance), public API surface (`TestPipelineAPI`, `TestProfileAPI`, `TestSampleModel`), CLI commands, TUI widget IDs.

## Common Pitfalls

1. **Don't import torch at module top level** — it slows down `import ayase` for users who don't need ML
2. **Don't forget `if sample.quality_metrics is None: sample.quality_metrics = QualityMetrics()`** — samples start with `quality_metrics = None`
3. **Don't raise exceptions in `process()`** — catch and log, always return sample
4. **Don't hardcode model paths** — use `resolve_model_path()` with `self.config.get("models_dir", "models")`
5. **Don't add fields to QualityMetrics without adding to `_FIELD_GROUPS`** — `to_grouped_dict()` will put them in "other"
6. **Don't modify `third_party/`** unless absolutely necessary — it's vendored external code
7. **Version is in two places** — `__init__.py` and `pyproject.toml` — update both
8. **Don't forget `tests/test_readme_contract.py`** — has a hardcoded `README_METRICS` list and field count that must be updated when adding new QualityMetrics fields
9. **Don't use `sample.reference_path` directly** — it's not a Pydantic field; use `getattr(sample, "reference_path", None)`
10. **Don't use `enable_ml` flag** — removed; use tiered backend auto-detection pattern instead
11. **Don't add modules alphabetically to `__init__.py`** — they're grouped by functional category
