# Official JEDi backend

Ayase's `jedi_official` module calls the public `videojedi.JEDiMetric` API. It
passes a loader for reference videos as `train_loader`, a loader for generated
videos as `test_loader`, calls `load_features(...)`, and returns the package's
`compute_metric()` result. The upstream calculation is fine-tuned V-JEPA
features followed by degree-2 polynomial-kernel MMD with `coef0=0`, multiplied
by 100. Lower is better.

The integration is pinned to `videojedi==1.1.0`, source commit
`e64f3fe8572b89ea904d161def44b419024e31a0` (2024-12-08). The upstream code is
Apache-2.0. Its runtime dependencies are `vjepa==0.1.2`, scikit-learn, PyAV,
PyTorch and the V-JEPA encoder/probe checkpoints `vith16.pth.tar` and
`ssv2-probe.pth.tar`.

Run the backend in a dedicated Python environment and configure `backend_python`
to that environment's interpreter. Install the official `videojedi==1.1.0` and
its pinned dependencies only in that environment. Do not install it into Ayase's
main environment or weaken its dependency pins: `vjepa==0.1.2` requires an older
Torch/torchvision pair and pins pandas. The worker receives pre-resolved checkpoint
and YAML paths, so it never downloads weights.

Example Ayase profile (replace both paths):

```json
{
  "name": "jedi_distribution",
  "modules": ["jedi_official"],
  "module_config": {
    "jedi_official": {
      "backend_python": "/path/to/jedi-env/bin/python",
      "reference_dir": "/path/to/reference-videos",
      "num_samples": 5000
    }
  }
}
```

Load it with `AyasePipeline(profile="jedi-profile.json")` and run the generated
collection. On Windows, `backend_python` points to the environment's
`Scripts/python.exe`. `backend_cuda_visible_devices` optionally selects a
physical GPU for the child without changing the main process environment.

The upstream `vjepa` package declares the Creative Commons Attribution-NonCommercial
4.0 license. The isolated environment uses the published package without redistributing
it inside Ayase; users remain responsible for following upstream model and package terms.
The backend is marked `adapted` because Ayase defines its own loader policy.

Upstream delegates video decoding and sampling to caller-provided loaders. The
Ayase loader fixes that otherwise-unspecified part of the protocol: decode with
PyAV, sample 16 frames uniformly including both endpoints, resize each frame to
224 x 224 with bilinear interpolation, scale RGB pixels to `[0, 1]`, and emit
`T,C,H,W`. `jedi_preprocessing_sha256` identifies this policy. The ordered
reference-file manifest is identified by `jedi_reference_manifest_sha256`; the
generated and reference counts are also recorded.

References can be supplied per sample through `reference_path`, or once through
`reference_dir`. A directory is recursively enumerated and sorted. Per-sample
references are deduplicated and sorted. The two distributions are truncated to
the same `min(num_samples, generated_count, reference_count)` size before the
official API is called.

`mmd_selfsplit` remains unchanged. It compares two insertion-order halves of a
single collection with a Gaussian kernel and is not an alias for this published
two-distribution metric. For backward compatibility, old `jedi` payloads still
refer to `mmd_selfsplit`; the official result is stored as `jedi_score`.

An optional `feature_path` is a cache root. Subdirectories are keyed by both
collections' paths, sizes, modification times, and preprocessing policy, so
upstream's fixed `train.npy`/`test.npy` names cannot reuse another run's features.
Without a cache root, a fresh temporary directory is removed after scoring.
Each namespace is protected by a worker process lock because upstream writes NumPy
feature files directly. Before reuse and after extraction, the worker verifies that
both arrays are numeric, finite, two-dimensional, and contain the requested number
of samples. A corrupt or incomplete derived array is removed and recomputed while a
valid array for the other distribution is preserved.

Upstream currently moves input batches to CUDA unconditionally and initializes
distributed state while loading V-JEPA. Ayase therefore launches `_jedi_worker.py`
as a fresh process using the configured `backend_python`. A CUDA-capable runtime is therefore the
supported path for real feature extraction, despite parts of the model loader
having CPU branches. Unit tests mock only this heavyweight boundary and verify
the real public API contract and Ayase's deterministic collection handling.
An existing distributed process group is rejected because upstream cleans up
its group globally. The worker communicates exclusively through JSON input and output files;
upstream stdout is redirected to the child stderr log.
The upstream encoder hardcodes `cuda:0`; other CUDA indices are rejected instead
of silently using another device. Select a physical GPU with `CUDA_VISIBLE_DEVICES`
in the worker's environment. The fixed encoder configuration requires 16 frames
at 224 pixels. Feature caches also include the package version, configuration
hash, and checkpoint path/size/mtime identity. Any group created by a failed
extraction is cleaned up so the worker can retry.
