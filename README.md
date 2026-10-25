# Ayase

Modular media quality metrics for video, image, and audio datasets.

Ayase scans a dataset, runs the metrics you choose, and exports per-sample
results and validation issues. 397 modules produce 596 metrics across 22 categories,
covering perceptual quality, text alignment, motion, temporal consistency,
audio, safety, and more.

## Install

Requires Python 3.9+.

```bash
pip install ayase
```

One install provides the metric runtime infrastructure. Ayase downloads weights
and provisions private Python environments or native tools as needed on first
use, then caches them for later runs. No per-metric extras are required.
Uncached setup needs internet access; hardware requirements depend on the metric.

## Quick start

Scan a dataset with a small, explicit set of modules and save a report:

```bash
ayase scan ./dataset --modules metadata,basic_quality --format json --output report.json
```

Find and inspect metrics from the command line:

```bash
ayase help                      # browse metrics and providers
ayase help rqvqa_score           # inspect a metric's models, config, and usage
```

### Python

```python
from ayase import AyasePipeline

pipeline = AyasePipeline(modules=["metadata", "basic_quality"])
results = pipeline.run("./dataset")

for sample in results.values():
    if sample.quality_metrics is not None:
        print(sample.path.name, sample.quality_metrics.blur_score)

pipeline.export("report.json")   # also supports CSV and HTML
```

## Choose metrics

Use the [metric catalog](METRICS.md) to compare inputs, score ranges,
configuration, and backend requirements. The [model catalog](MODELS.md) lists
weights and their licenses. Metrics use different scales and may require
captions, reference media, or a compatible GPU; check the selected metric's entry.

Video-generation evaluation includes [JEDi](docs/jedi_backend.md),
[VideoPhy-2](docs/videophy2_backend.md), and all seven
[T2V-CompBench categories](docs/t2v_compbench_backend.md), with automatically
managed backend environments and assets.

## Guides

| Task | Guide |
| --- | --- |
| Scan, filter, export, or configure a dataset run | [Usage and configuration](docs/usage.md) |
| Understand downloads, caches, and backend environments | [Managed backends](docs/managed_backends.md) |
| Integrate without changing an application's ML dependencies | [Isolated client](docs/isolated_client.md) |
| Add a metric or external plugin | [Custom modules](docs/custom_modules.md) |
| Work on Ayase itself | [Contributing](CONTRIBUTING.md) |
| Review changes between versions | [Changelog](CHANGELOG.md) |

APIs and module interfaces may change before 1.0.

## License

Ayase's own code is MIT. Bundled components and downloaded model weights retain
their own terms; see [third-party notices](src/ayase/third_party/NOTICES.md)
and the [model catalog](MODELS.md).
