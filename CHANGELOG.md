# Changelog

All notable changes to Ayase will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.10]

### Added

- **dover**: ONNX backend (tier 2) with configurable `preferred_backend` ("native" / "onnx" / "pyiqa")
- **action_recognition**: `matching_mode` config ("weighted" top-K or "top1" direct similarity); open_clip as preferred CLIP backend
- **motion_smoothness**: bundled RIFE HD v3 (vendored in `third_party/rife/`) with auto-download from HuggingFace and padding fix for non-32-aligned resolutions

### Fixed

- **clip_temporal**: `face_consistency` changed from first-frame comparison to rolling window (consecutive pairs), matching VBench methodology
- **motion_amplitude**: added `scoring_mode` config ("binary" / "continuous") for smooth 0-100 scoring
- **dover**: fixed aesthetic/technical output order in `DOVERModel.forward()` (dict key iteration)
- **dover**: ConvNeXt backbone `pretrained=False` to match original DOVER training procedure

## [0.1.8]

### Fixed

- **dover**, **fastvqa**: third-party model source code was excluded from package builds by overly broad `.gitignore` rule
- **i2v_similarity**: replaced `torch.hub.load` with `timm.create_model()` for DINOv2 — eliminates network requests when local weights are available
- **i2v_similarity**: removed `os.environ["TORCH_HOME"]` side effect that broke DOVER weight resolution on pipeline re-initialization
- `download_model_file()`: added socket timeout (300s) to prevent indefinite hangs on restricted networks

### Changed

- **dover**, **i2v_similarity**, **aesthetic_scoring**, **fast_vqa**: model download URLs migrated to [HuggingFace Hub](https://huggingface.co/AkaneTendo25/ayase-models) for reliable access; original URLs preserved in source comments

## [0.1.4]

### Fixed

- Pipeline `_mounted` guard: modules with missing dependencies now stay unmounted and are skipped during processing
- 8 modules had broken ML dispatch: loaded real models in `setup()` but never invoked them in `process()` — fixed for **kvq**, **rqvqa**, **p1203**, **t2v_score**, **st_lpips**, **psnr_hvs**, **hdr_sdr_vqa**, **dynamics_controllability**
- **temporal_flickering**: added `max_frames` config (default 300) to prevent OOM on long videos
- **fvd**: fixed docstring/variable naming to match actual R3D-18 backbone

### Changed

- `QualityMetrics` now uses `extra="forbid"` — typo'd field names raise `ValidationError`
- `pyiqa` minimum version bumped to `>=0.1.13`

## [Unreleased]

### Added

- 6 new modules: **identity_loss**, **tifa**, **tonal_dynamic_range**, **nemo_curator**, **umap_projection**, **vlm_judge presets**
- `resolve_model_path()` and `download_model_file()` utilities in `config.py`
- Explicit config params for evaluation: `ocr_fidelity.expected_text`, `motion_amplitude.expected_motion`, `action_recognition.expected_action`
- `v-identity` optional dependency group

### Changed

- README metrics table redesigned as 5-column API reference
- Removed `enable_ml` flag from all modules — ML auto-detected via tiered backend pattern
- TUI: Windows drive letter support, `Path.home()` as default start directory

### Removed

- `quality.py` and `video.py` legacy files

## [0.1.0] - 2024-12-01

Initial release.

- Modular pipeline architecture with 235 modules across 198 files
- CLI (`ayase scan/run/filter/stats/tui/modules/config`)
- TUI built with Textual (6 screens)
- Profile system (JSON/TOML) for configurable module sets
- `AyaseConfig` via pydantic-settings (TOML + env vars)
- Export: JSON, CSV, HTML, Markdown + state save/resume
- `QualityMetrics` data model with ~175 metric fields in 18 groups
- Plugin auto-discovery from configurable directories
- 15 optional dependency groups for selective installation
- Python 3.9–3.12, MIT license
