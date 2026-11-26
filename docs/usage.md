# Using Ayase

[Back to README](../README.md)

## Command-line reference

```bash
ayase scan ./dataset                                    # default balanced pipeline
ayase scan ./dataset --deep                             # run default-eligible packaged metrics
ayase scan ./dataset --modules metadata,basic_quality   # selected modules
ayase help                                              # list every metric and provider
ayase help rqvqa_score                                  # metric/module models, config, and usage
ayase modules list                                      # show registry provenance and backend requirements
ayase modules check                                     # import/dependency readiness
ayase filter ./dataset --min-score 70 --output ./good   # filter by quality
ayase stats ./dataset                                   # dataset statistics for images/video
```

Use `ayase scan --help` for scan options and `ayase help <metric-or-module>`
for a metric's configuration, models, and usage. JSON, CSV, Markdown, and HTML
reports can be selected with `--format` and written with `--output`:

```bash
ayase scan ./dataset --modules metadata,basic_quality --format json --output report.json
```

`--deep` runs default-eligible packaged metrics and can require substantial
model downloads and compute. Select modules explicitly for a targeted run.
See the [metric catalog](../METRICS.md) for inputs, score directions, provenance,
and backend requirements. Scores from different metrics use different scales.

## Pairwise model comparison

Run `ayase compare judgments.jsonl --draws 1000 --seed 0 -o comparison.json`.
Supply a JSON array or one JSON object per line in a `.jsonl` file.

```json
{"prompt": "prompt-001", "model_a": "model-a", "model_b": "model-b", "outcome": "tie"}
```

Use `a` or `b` for a win, or `tie` for half a win per model. The
Bradley–Terry fit uses a symmetric half-win prior per observed pair to keep
ratings finite when one model wins every comparison. Log-strengths are
centered within each connected component. If `connected` is false, the ranks
apply within their respective components.

Confidence intervals resample complete prompts with replacement. A bootstrap
draw contributes to a component only when it retains that component's
connectivity. The report includes the number of accepted draws and uses
`null` for unavailable intervals.

## Configuration

`ayase.toml` in project root:

```toml
[general]
parallel_jobs = 8  # concurrency hint passed to capable modules/backends
prefetch_workers = 0  # opt-in: number of background video readers
video_decoder = "opencv"
assets_repo = "AkaneTendo25/ayase-assets"  # owner/repo or its huggingface.co URL

[pipeline]
modules = ["metadata", "basic_quality", "motion"]

[output]
default_format = "json"
artifacts_dir = "reports"
```

`video_decoder = "decord"` opts into Decord for video metrics that use
Ayase's frame readers. OpenCV is the default and the current behavior of
`"auto"`. A Decord decode failure leaves the affected frame request empty.
Resume state records the effective decoder and rejects reuse across different
decoders.

Set `prefetch_workers` above zero to overlap decoding one upcoming batch with
the current batch. Prefetch reads the fixed frame grids declared by compatible
modules and requires the shared frame cache. Module processing, hooks and
checkpoints run sequentially in input order. For video with no reliable frame
count, the normal reader handles the request when the metric runs.

For repeated Python API calls, `AyasePipeline(..., reuse_model_weights=True)`
keeps supported CLIP, FID and FVD inference weights between runs. Each run
has fresh module instances, accumulators and results. Call
`release_model_weights()` between runs to free the retained weights.

GPU profiles can set `sample_batch_size` to 4–16 where device memory allows.
The default is 1. FVD batches I3D feature extraction and computes one score
over the complete dataset. Its default `dtype = "auto"` keeps I3D in FP32.
Choose `"fp16"` or `"bf16"` explicitly to enable I3D autocast on CUDA.

`assets_repo` selects the Hugging Face repository used for Ayase-managed
checkpoints and source archives. It can also be set with
`AYASE_GENERAL__ASSETS_REPO`; a module profile's `assets_repo` value takes
precedence for that module.

See [managed backends](managed_backends.md) for runtime caching and expert
backend overrides, or [the isolated client](isolated_client.md) to keep Ayase
outside an application's dependency environment.
