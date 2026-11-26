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

## Video editing inputs

For an edited video, set `Sample.source_video` to the original video path
and `Sample.edit_instruction` to the requested edit. Use the separate
`reference_path` field when a metric expects a target reference.

`Sample.video_alignment` accepts a `VideoAlignment` policy. Pass the actual
frame timestamps in seconds to `source_indices(edited_pts, source_pts)`.
The default policy, `timestamp`, matches elapsed time. `normalized` maps
the edited span onto the source span. `anchors` interpolates between
`(edited_seconds, source_seconds)` pairs and supports held frames.
Each timeline starts at its first frame. The result uses the nearest
source frame, breaking ties toward the earlier frame, and uses `None`
outside the covered range.

The edit inputs are serialized with pipeline state. A new instruction,
alignment policy, or changed source video invalidates cached results.

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

FVD uses FP32 by default. Set `dtype = "fp16"` or `dtype = "bf16"` in the
FVD module settings to enable CUDA autocast. On an RTX 5090 with four
16-frame clips, the pretrained I3D feature pass measured 323 clips/s in
FP32 and 440 in FP16 after warmup, excluding decode and distribution
aggregation. FVD differed from FP32 by 0.0023% in FP16 and 0.30% in BF16
on that input set.

Set `execution_mode = "staged"` under `[general]` to process an input cohort
one module at a time and release each module before mounting the next.
The selected stack may include sample-local modules and supported dataset
accumulators such as FID and FVD. The accumulators finalize at the end of
the run and rebuild their features from the full input cohort after resume.
Stacks with hooks or other modules use sample-first execution. Staged runs
hold the cohort in memory and publish results only after the final stage.
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
