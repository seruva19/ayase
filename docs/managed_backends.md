# Managed metric infrastructure

Install Ayase once:

```bash
pip install ayase
```

There are no per-metric installation extras. Compatible libraries are included in
the normal installation. When a metric needs conflicting library versions, Ayase
creates a private environment on its first real mount. It also acquires the
required Python version; the application does not need another Python installed.
Native binary dependencies use private conda-forge environments where necessary.
These operations do not install packages into the application's Python environment
or change its global PATH or Python registration.

The default cache is below `models_dir/.ayase`:

| Directory | Contents |
| --- | --- |
| `python` | Managed Python interpreters |
| `environments` | Cached isolated backend installations |
| `sources` | Pinned upstream source snapshots |
| `tools`, `mamba`, `conda-packages` | Native package manager and binary cache |
| `workers` | Per-module worker configuration, protocol files and logs |
| `logs` | Dependency installation logs |

Environments are keyed by their Python and dependency specifications. Installation
locks prevent competing setup processes; completion manifests are written only
after installation succeeds. Interrupted installations are retried on the next
mount. Identical dependency sets share an environment, while each running module
has its own worker and state.

The worker forwards the normal module lifecycle, per-sample results, dataset
metrics and validation failures. Samples keep their identity, and metric writes
retain provenance even when a module writes the same numeric value as an earlier
module. Backend setup and processing failures remain visible in pipeline failure
reporting. Test mode skips provisioning, model downloads and backend setup.

JEDi and T2V-CompBench use their dedicated managed evaluators. VideoPhy-2 uses its
upstream Transformers stack in isolation. VMAF automatically selects FFmpeg with
libvmaf, including the binary supplied with Ayase's imageio-ffmpeg dependency.
ViSQOL uses a labelled numerical transcription validated against the official
conformance data; see [its backend notes](visqol_backend.md).

Internet access is needed for uncached dependencies and assets. Hardware
requirements belong to the selected metric: CUDA-only author implementations
still require a compatible NVIDIA GPU. User data, captions and reference datasets
remain inputs supplied by the caller.

Backend paths and `managed_runtime=false` remain expert overrides for deployments
that intentionally supply their own prepared runtime. They are not installation
prerequisites. Development tools use the separate `dev` dependency group:

```bash
python -m pip install --upgrade pip
pip install -e . --group dev
```
