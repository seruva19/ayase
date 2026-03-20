# Ayase

Modular media quality metrics toolkit for video, image, and audio datasets.

## Overview

- **312 modules**, **349 quality metrics** across visual, temporal, audio, perceptual, and safety categories
- Modular pipeline - modules compute raw values, downstream apps decide what to do with them
- CLI and Python API with profile-based configuration
- See [METRICS.md](METRICS.md) for the full reference, [MODELS.md](MODELS.md) for all pretrained weights

## Installation

```bash
pip install ayase                # Core (metadata/structural checks only)
pip install ayase[ml]            # Everything
pip install ayase[v-iqa]         # PyIQA, BRISQUE, NIQE, scikit-video
pip install ayase[v-perceptual]  # CLIP, LPIPS, open-clip, timm
pip install ayase[v-motion]      # RAFT optical flow, decord
pip install ayase[dev]           # pytest, black, ruff, mypy
```

Models are downloaded and cached automatically on first use.

## Quick Start

```bash
ayase scan ./my_dataset
ayase scan ./my_dataset --modules metadata,basic_quality,motion
ayase modules list
ayase modules check
ayase filter ./my_dataset --min-score 70 --output ./filtered
```

```python
from ayase import AyasePipeline

ayase = AyasePipeline(modules=["basic", "aesthetic", "motion"])
results = ayase.run("./my_dataset")

for path, sample in results.items():
    if sample.quality_metrics:
        print(f"{sample.path.name}: score={sample.quality_metrics.technical_score}")

ayase.export("report.json")
```

## Configuration

Create `ayase.toml` in your project root:

```toml
[general]
parallel_jobs = 8

[pipeline]
modules = ["metadata", "basic_quality", "motion"]

[output]
default_format = "json"
artifacts_dir = "reports"
```

## Writing Plugins

```python
from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

class MyCheck(PipelineModule):
    name = "my_check"
    description = "Custom quality check"
    default_config = {"threshold": 0.5}

    def process(self, sample: Sample) -> Sample:
        # Your logic here
        return sample
```

```bash
ayase scan ./data --modules metadata,my_check
```

## Development

```bash
git clone <repo-url> && cd ayase
pip install -e ".[dev]"
pytest
```

## License

MIT - see [LICENSE](LICENSE).

**Model licenses:** ML models downloaded at runtime have their own licenses (Apache 2.0, BSD, CC-BY-NC, etc.). Users are responsible for compliance. See [MODELS.md](MODELS.md) for the full catalog with license info. No model weights are bundled with this package.
