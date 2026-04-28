# Ayase

Modular media quality metrics for video, image, and audio datasets.

> **Work in progress** - APIs and module interfaces may change before 1.0.

## What It Does

Ayase runs quality assessment modules over a dataset and writes structured per-sample metrics. 327 modules produce 368 metrics across 21 categories (NR-IQA, FR-IQA, NR-VQA, temporal, motion, audio, face, safety, aesthetics, text-video alignment, and more). Modules are independent - pick only what you need.

Full metric catalog: [METRICS.md](METRICS.md). Pretrained model catalog: [MODELS.md](MODELS.md).

## Install

```bash
pip install ayase
```

Ayase is distributed as a single install. Runtime dependencies are managed by the
project itself, and model weights are downloaded and cached on first use.

## CLI

```bash
ayase scan ./dataset                                    # default balanced pipeline
ayase scan ./dataset --deep                             # run every discovered module
ayase scan ./dataset --modules metadata,basic_quality   # selected modules
ayase modules list                                      # show all 327 modules
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

MIT. Model weights downloaded at runtime carry their own licenses - see [MODELS.md](MODELS.md).
