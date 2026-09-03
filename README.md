# Ayase

Modular media quality metrics for video, image, and audio datasets.

> **Work in progress** - APIs and module interfaces may change before 1.0.

## What It Does

Ayase runs quality assessment modules over a dataset and writes structured per-sample metrics. 375 modules produce 511 metrics across 21 categories (NR-IQA, FR-IQA, NR-VQA, temporal, motion, audio, face, safety, aesthetics, text-video alignment, and more). Modules are independent - pick only what you need.

Full metric catalog: [METRICS.md](METRICS.md). Pretrained model catalog: [MODELS.md](MODELS.md).

## Install

```bash
pip install ayase
```

Ayase is distributed as a single install. Runtime dependencies are managed by the
project itself, and model weights are downloaded and cached on first use.

### Isolated installation

Applications that already have their own PyTorch, NumPy, or media dependency
constraints can keep Ayase completely outside their environment:

```bash
pip install ayase --no-deps
```

```python
from ayase_client import AyasePipeline

pipeline = AyasePipeline(modules=["basic", "metadata", "motion"])
results = pipeline.run("./my_dataset")
pipeline.export("report.json")
pipeline.close()
```

This installs the same Ayase wheel without ML dependencies. On first use the
client creates a private full-runtime venv and starts a loopback worker. Media is
passed by local path. Regular `pip install ayase` remains unchanged.

Optional check: `ayase-client doctor`. To use an existing runtime, set
`AYASE_RUNTIME_PYTHON`.

## CLI

```bash
ayase scan ./dataset                                    # default balanced pipeline
ayase scan ./dataset --deep                             # run every discovered module
ayase scan ./dataset --modules metadata,basic_quality   # selected modules
ayase help                                              # list every metric and provider
ayase help rqvqa_score                                  # metric/module models, config, and usage
ayase modules list                                      # show all 375 modules
ayase modules check                                     # import/dependency readiness
ayase filter ./dataset --min-score 70 --output ./good   # filter by quality
ayase stats ./dataset                                   # dataset statistics for images/video
ayase tui                                               # terminal UI
```

## Python API

```python
from ayase import AyasePipeline

pipeline = AyasePipeline(modules=["basic", "metadata", "motion"])
results = pipeline.run("./my_dataset")

for path, sample in results.items():
    qm = sample.quality_metrics
    if qm:
        print(f"{sample.path.name}: technical={qm.technical_score} blur={qm.blur_score}")

pipeline.export("report.json")   # also: report.csv, report.html
```

## Configuration

`ayase.toml` in project root:

```toml
[general]
parallel_jobs = 8  # concurrency hint passed to capable modules/backends

[pipeline]
modules = ["metadata", "basic_quality", "motion"]

[output]
default_format = "json"
artifacts_dir = "reports"
```

## Custom Modules

```python
from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule
import cv2

class BlurCheck(PipelineModule):
    name = "blur_check"
    description = "Flag blurry frames via Laplacian variance"
    default_config = {"threshold": 100.0}

    def process(self, sample: Sample) -> Sample:
        img = cv2.imread(str(sample.path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return sample
        score = float(cv2.Laplacian(img, cv2.CV_64F).var())
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.blur_score = score
        if score < self.config.get("threshold", 100.0):
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Blurry ({score:.0f})",
                )
            )
        return sample
```

Modules auto-register via `__init_subclass__`. Config is available as `self.config`.

## Development

```bash
git clone <repo-url> && cd ayase
pip install -e ".[dev]"
pytest                    # 8000+ tests, ~4 min
pytest tests/ --full      # with ML model loading
```

## License

Ayase's own code is MIT. Model weights downloaded at runtime carry their own licenses -
see [MODELS.md](MODELS.md).

Some metrics run research code vendored under `ayase/vendor` (see
[the inventory](src/ayase/vendor/README.md)), and four of those components are not
permissive. **Running one of these metrics places its component's licence on your use
of the result**, whatever Ayase's own licence says. Each affected module declares what
it runs and logs a notice at setup:

| Metric | Vendored component | Licence |
|---|---|---|
| `chronomagic`, `dynamics_controllability`, `physics`, `video_edit_motion_fidelity`, `vmbench_pas`, `vmbench_tcs` | CoTracker | CC BY-NC 4.0 - non-commercial only |
| `mj_video` | MJ-Video | no licence file upstream - no grant is stated |
| `vbench2` | VBench 2.0 with its vendored YOLO-World and CoTracker | Apache-2.0, plus GPL-3.0 and CC BY-NC 4.0 inside |

Every other vendored component is MIT, BSD or Apache-2.0 and imposes nothing beyond
attribution, which the retained licence files provide.

This is a disclosure, not a resolution. The plan for 1.0 is to replace these
components with implementations Ayase can license itself, so that the whole
distribution is MIT in substance and not only in name.
