# Ayase

Ayase measures the quality and content of images and videos. Choose the metrics
that matter to you, run them on your files, and save a report with scores and
validation issues. 390 modules produce 618 metrics across 22 categories,
covering visual quality, text alignment, motion, audio, safety, and more.

## Install

Requires Python 3.9 or newer:

```bash
pip install ayase
```

Ayase automatically downloads weights and provisions backend environments or
native tools when needed, then caches them for later runs. No per-metric extras
are required. First use of an uncached metric needs internet access; hardware
requirements depend on the metric.

## Quick start

Put images or videos in a folder, then run:

```bash
ayase scan ./media --modules metadata,basic_quality --format json --output report.json
```

The JSON report contains `run_status`, overall `stats`, and `samples` with
per-file results. A missing score is not zero: the metric may not have been
selected, may not apply to the file, or may have failed or been unavailable.
Check the run status and logs for details.

## Choose metrics

Browse available metrics and inspect one:

```bash
ayase help
ayase help blur_score
```

`blur_score` is a metric field; `basic_quality` is a module that produces it.
Pass module names to `--modules`. Scores use different scales and directions,
so check the metric description before comparing results. The
[metric catalog](METRICS.md) describes inputs, ranges, requirements, and
readiness: an approximate metric uses a substitute or partial method, while an
unimplemented metric has no supported complete backend in a standard install.
The [model catalog](MODELS.md) lists weights and licenses.

## Inputs

The scanner finds images and videos, including in subfolders. For metrics that
need a caption, place a `.txt`, `.caption`, or caption `.json` file beside the
media file with the same base name, such as `clip.mp4` and `clip.txt`.
Reference-based metrics need an
explicit `reference_path` on a `Sample`; the scanner does not pair references
automatically. The Python API also accepts prepared `Sample` objects. See the
[usage guide](docs/usage.md) for input and configuration details.

## Python

```python
from ayase import AyasePipeline

pipeline = AyasePipeline(modules=["metadata", "basic_quality"])
results = pipeline.run("./media")

for sample in results.values():
    if sample.quality_metrics is not None:
        print(sample.path.name, sample.quality_metrics.blur_score)

pipeline.export("report.json")
pipeline.export("report.csv", format="csv")
```

## Guides

| Task | Guide |
| --- | --- |
| Run checks, filter files, or configure metrics | [Usage and configuration](docs/usage.md) |
| Understand downloads, caches, and backend requirements | [Managed backends](docs/managed_backends.md) |
| Integrate through an isolated environment | [Isolated client](docs/isolated_client.md) |
| Add a metric or plugin | [Custom modules](docs/custom_modules.md) |
| Contribute to Ayase | [Contributing](CONTRIBUTING.md) |
| Review changes between versions | [Changelog](CHANGELOG.md) |

APIs and module interfaces may change before 1.0.

## License

Ayase's own code is MIT. Bundled components and downloaded model weights retain
their own terms; see [third-party notices](src/ayase/third_party/NOTICES.md)
and the [model catalog](MODELS.md).
