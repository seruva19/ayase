# Ayase

Modular media quality metrics toolkit.  

⚠️ Work in progress. Some features may not work as expected.

## Overview

- 246 quality metrics across visual, temporal, audio, perceptual, and safety categories.
- Modular pipeline - modules compute raw values, downstream apps decide what to do with them.
- CLI and Python API.
- Profile-based pipeline configuration.

## Installation

Core (no ML models, metadata/structural checks only):

```bash
pip install ayase
```

With ML-based quality metrics:

```bash
pip install ayase[ml]           # Everything
pip install ayase[v-perceptual] # CLIP, LPIPS, open-clip, timm
pip install ayase[v-iqa]        # PyIQA, BRISQUE, NIQE, scikit-video
pip install ayase[v-motion]     # RAFT optical flow, decord
pip install ayase[v-ocr]        # PaddleOCR text recognition
pip install ayase[v-face]       # MediaPipe face detection
pip install ayase[v-audio]      # librosa audio analysis
```

Development:

```bash
pip install ayase[dev]          # pytest, black, ruff, mypy
pip install ayase[all]          # Everything including dev + TUI
```

See [MODELS.md](MODELS.md) for the complete inventory of all pretrained weights used by every module. Models are downloaded and cached automatically on first use via HuggingFace Hub, Torch Hub, and PyIQA.

## Metrics

208 modules across image, video, and audio quality assessment. See **[METRICS.md](METRICS.md)** for the full reference with config parameters.

Categories: visual quality (NR/FR), perceptual similarity, motion, temporal consistency, audio, text alignment, HDR, safety, face, depth, codec.


## Quick Start

### CLI

```bash
# Scan a dataset and get a report
ayase scan ./my_dataset

# Scan with specific modules
ayase scan ./my_dataset --modules metadata,basic_quality,motion

# List all available modules
ayase modules list

# Check which modules can be loaded (dependencies installed)
ayase modules check

# Filter dataset by quality score
ayase filter ./my_dataset --min-score 70 --output ./filtered
```

### Python API (recommended)

```python
from ayase import AyasePipeline

ayase = AyasePipeline(modules=["basic"])
results = ayase.run("./my_dataset")

for path, sample in results.items():
    if sample.quality_metrics:
        print(f"{sample.path.name}: technical={sample.quality_metrics.technical_score}")

print(f"Total: {ayase.stats.total_samples}, Valid: {ayase.stats.valid_samples}")
ayase.export("report.json")
```

`AyasePipeline` accepts three ways to configure modules:

```python
# By module names
ayase = AyasePipeline(modules=["metadata", "basic_quality", "motion"])

# By profile dict
ayase = AyasePipeline(profile={
    "name": "my_check",
    "modules": ["basic", "aesthetic"],
    "module_config": {
        "aesthetic": {"model_name": "openai/clip-vit-large-patch14"},
    },
})

# By profile file
ayase = AyasePipeline(profile="my_profile.toml")

# With custom config
from ayase.config import AyaseConfig
ayase = AyasePipeline(config=AyaseConfig(general={"parallel_jobs": 16}), modules=["basic"])
```

### Low-level Pipeline API

```python
import asyncio
from pathlib import Path
from ayase.pipeline import Pipeline, ModuleRegistry
from ayase.scanner import scan_dataset

ModuleRegistry.discover_modules()
module_names = ["metadata", "basic_quality", "semantic_alignment"]
modules = [ModuleRegistry.get_module(n)() for n in module_names]

pipeline = Pipeline(modules)
pipeline.start()

samples = scan_dataset(Path("./my_dataset"), recursive=True)
for sample in samples:
    processed = asyncio.run(pipeline.process_sample(sample))

pipeline.stop()
pipeline.export_report("report.json", format="json")
```

### Profile-based pipelines

```python
from ayase import load_profile, instantiate_profile_modules

profile = load_profile("my_profile.toml")
modules = instantiate_profile_modules(profile)
# modules is a list of PipelineModule instances ready for Pipeline()
```

## Configuration

Create `ayase.toml` in your project root:

```toml
[general]
parallel_jobs = 8
cache_enabled = true

[quality]
enable_blur_detection = true
blur_threshold = 100.0

[pipeline]
dataset_path = "./my_dataset"
modules = ["metadata", "basic_quality", "motion"]
plugin_folders = ["plugins"]

[output]
default_format = "markdown"
artifacts_dir = "reports"
artifacts_format = "json"

[filter]
default_mode = "list"
min_score_threshold = 60
```

Ayase looks for config in: `./ayase.toml` -> `~/.config/ayase/config.toml` -> built-in defaults.

## Writing Plugins

Create a `.py` file in your `plugins/` folder:

```python
from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

class MyCustomCheck(PipelineModule):
    name = "my_check"
    description = "Custom quality check"
    default_config = {"threshold": 0.5}

    def process(self, sample: Sample) -> Sample:
        # Your logic here
        if some_score < self.config["threshold"]:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message="Quality below threshold",
                )
            )
        return sample
```

Then run:

```bash
ayase scan ./data --modules metadata,my_check
```

## Development

```bash
git clone <repo-url>
cd ayase
pip install -e ".[dev]"

# Run tests
pytest

# Lint and format
ruff check src/ tests/
black src/ tests/

# Type check
mypy src/ayase
```

## License

MIT -- see [LICENSE](LICENSE).
