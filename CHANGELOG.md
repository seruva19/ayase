# Changelog

All notable changes to Ayase will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.30]

### Added

- **unified_vqa**: added a dedicated `unified_vqa_score` metric while preserving the legacy `dover_score` compatibility alias when unset.
- **kandinsky_motion**: added declared camera, object, and dynamics motion metrics and model metadata.

### Changed

- Generated docs now use packaged-only module discovery by default, document DatasetStats outputs, and keep README counts aligned with the docs generator.
- `METRICS.md` now always includes static per-module test coverage links, even when generated without live pytest status collection.
- Model docs now classify CLIP variants separately from HuggingFace repositories and support offline deterministic regeneration.

### Fixed

- Pipeline module and hook execution now rejects non-`Sample` returns without corrupting the current sample.
- Dataset-level modules and vendored/third-party model modules now declare required `metric_info` / `models` metadata for docs generation.
- Core config and pipeline typing now pass the configured strict MyPy target.

### Fixed

- Core config loading now lets `AYASE_*` environment overrides take precedence over TOML values
- Pipeline cache and resume state now validate caption/reference context, reject stale persisted entries, and keep aggregate stats consistent when entries are replaced or skipped
- Resume state now records a pipeline fingerprint, rejects incompatible or legacy untrusted caches, and rolls back partial module registrations from failed imports
- Resume loading now replaces prior in-memory state, fingerprints effective `test_mode`, makes `modules check` mount modules with the loaded runtime config for real readiness, hard-fails unknown `run`/`stats` format values, guarantees TUI pipeline cleanup after execution errors, and resets `AyasePipeline.run()` state between runs
- Corrupt state files now leave the current in-memory pipeline state intact, and plugin readiness no longer keeps stale entries after broken plugin files are removed
- External plugins now reload when their source files change, unregister when removed, and repeated low-level `Pipeline.start()`/`stop()` cycles begin with a fresh run state
- `AyasePipeline.run()` now preserves caller-installed pipeline hooks and public module-config overrides across fresh run rebuilds, and external plugin readiness entries are namespaced by plugin file path so same-named plugins in different folders no longer overwrite each other
- Required model/weight downloads now reject path-escaping filenames and save atomically
- TUI and CLI file execution paths now attach sidecar captions consistently and inject the same runtime config (`models_dir`, `parallel_jobs`) as the profile/API path
- `ayase stats` now counts image-only datasets, `filter --mode list` no longer requires `--output`, and `scan`/`run` no longer create hidden artifact reports when stdout or explicit `--output` is used
- Duplicate module names now fail fast during auto-registration instead of silently overwriting each other
- Install/runtime docs no longer reference nonexistent extras, and the legacy `requirements-lock.txt` workflow was removed to keep `pip install ayase` as the single supported install path

## [0.1.29]

### Changed

- Replace all heuristic fallback backends with real ML implementations across 79 modules
- Modules now gracefully skip when ML backend is unavailable instead of computing proxy values
- CLIP weights on AkaneTendo25/ayase-models converted from .pt to .safetensors
- Fix model references: KVQ (lero233/KVQ), AIGV-Assessor (IntMeGroup/), SenseVoice (FunAudioLLM/)

### Added

- **dino_face_identity**: DINOv2 face identity fields in QualityMetrics
- **test_docs_integrity**: 327 new tests verifying module documentation, field writes, model references, and no-heuristic enforcement
- Paper-accurate implementations: VSFA (quality-aware temporal pooling), VIDEVAL (60 hand-crafted features), VIIDEO/V-BLIINDS (scikit-video backend), face IQA (CR-FIQA/MagFace/SER-FIQ/GraFIQs via InsightFace), ModularBVQA (Laplacian+SlowFast rectifiers), Zoom-VQA (dual-branch IQA+VQA)

## [0.1.18]

### Added

- **kid**: Kernel Inception Distance batch distribution metric (clean-fid/native)
- **image_reward**: ImageReward human preference scoring for text-to-image
- **image_lpips**: LPIPS perceptual distance + dataset diversity metric
- **concept_presence**: Concept detection via face detection + CLIP
- **face_cross_similarity**: Pairwise ArcFace cosine similarity matrix across dataset

### Fixed

- Fix 156 audit issues across 312 modules and core framework
- Fix 14 field collisions (each module now writes to unique QualityMetrics field)
- Fix 3 wrong HuggingFace model IDs (kandinsky, VideoReward, vjepa)
- Fix on_mount vs setup lifecycle in 12 modules
- Fix pyiqa device detection in 27 modules (fragile next(parameters).device)
- Fix 7 format string crashes on None values
- Fix algorithm bugs: t2v_compbench CLIP sim, commonsense scoring, chronomagic inversion
- Fix deepfake FFT spectral check (was triggering on all images)
- Fix SSIM negative variance in ws_ssim/pu_metrics
- Fix CPP-PSNR (was fabricated, now proper projection)
- Fix HDR metadata PQ EOTF applied to SDR content
- Add frame limits to 7 modules that read entire videos
- Add try/finally for VideoCapture in 5 modules
- Add audio extraction from video for audio_estoi, audio_si_sdr
- Remove async from process_sample (no await, adds overhead)
- Fix stale cache in pipeline load_state
- Fix path traversal in config.py
- Security: replace eval() with getattr(), default trust_remote_code=False

### Changed

- METRICS.md: metric-centric info panels, seaborn charts, clickable nav, source links
- MODELS.md: info panels, pyiqa as table, URL validation, weight file grouping
- Remove 21 orphaned QualityMetrics fields (batch-only moved to DatasetStats)
- Remove deprecated property aliases (fid_score, kid_score, etc.)
- Remove test_golden_values.py (fragile, superseded by integration tests)

## [0.1.14]

### Added

- **audio_estoi**: ESTOI speech intelligibility (full-reference, pystoi)
- **audio_mcd**: Mel Cepstral Distortion for TTS/voice conversion (librosa)
- **audio_si_sdr**: Scale-Invariant Signal-to-Distortion Ratio (numpy, no ML deps)
- **audio_lpdist**: Log-Power Spectral Distance (librosa)
- **audio_utmos**: UTMOS no-reference MOS prediction (SpeechMOS)

## [0.1.13]

### Fixed

- **temporal_flickering**: RAFT padding fix — pad frames to multiple of 8 before inference, crop results back. Fixes crash on 1080p video (540px height not divisible by 8)
- **ocr_fidelity**: added CER (Character Error Rate) and WER (Word Error Rate) alongside NED

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

- **videoscore2**: VideoScore2 generative video evaluation with visual quality, text-video alignment, and physical/common-sense consistency outputs
- 3 new QualityMetrics fields for VideoScore2: `videoscore2_visual`, `videoscore2_alignment`, `videoscore2_physical`
- **verse_bench**: native Ayase Verse-Bench aggregation over vendored inferencers, with dataset-level outputs stored in `DatasetStats` when a materialized benchmark dataset is provided
- 3 new DatasetStats fields for Verse-Bench: `verse_bench_overall`, `verse_bench_metrics`, `verse_bench_breakdown`
- `models` and `metric_info` class-level declarations on `PipelineModule` for explicit model/metadata documentation in `MODELS.md` and `METRICS.md`
- Module-level docstrings added to 46 modules that were missing them
- Module docstring requirement documented in AGENTS.md (Section 7)
- Model/metric declaration rules documented in AGENTS.md (Section 8)

### Changed

- **verse_bench**: added missing runtime dependencies to the base install (`moviepy`, `pyloudnorm`, `python_speech_features`, `wget`) so `pip install ayase` includes the vendored benchmark inferencer requirements
- `PipelineModule.get_metadata()` now returns `models` and `metric_info` fields
- `MODELS.md` generator reads `cls.models` declarations in addition to regex scanning
- `METRICS.md` generator merges `cls.metric_info` descriptions into auto-inferred output fields
- Removed unused vendor files from `verse_bench`: `aesthetic/musiq/` training code, `aesthetic/manica_utils/process.py`

### Fixed

- Config precedence now applies `AYASE_*` environment overrides on top of TOML/default values instead of silently letting file values win
- Pipeline cache reuse now respects caption/reference context instead of reusing stale results solely by file path
- Required model-file downloads now reject path-escaping targets and use atomic `.part` writes before replacing the final file
- CLI `stats` now counts image-only datasets, and `filter --mode list` no longer requires `--output`
- CLI `scan`/`run` no longer create surprise report artifacts when the user already chose explicit stdout or `--output`
- Module registry now rejects duplicate module names instead of silently overwriting the first registration
- Docker/TUI docs no longer reference non-existent install extras in the single-install distribution

## [0.1.19] - 2026-03-28

### Added

- **pickscore**: PickScore prompt-conditioned preference scoring
- **hpsv3**: HPSv3 prompt-conditioned preference scoring
- **chipqa**: ChipQA no-reference video quality scoring
- **hdr_chipqa**: HDR-ChipQA no-reference HDR video quality scoring
- **hdrmax**: HDRMAX full-reference HDR video quality scoring
- **brightrate**: BrightRate no-reference HDR video quality scoring
- 2 new QualityMetrics fields for prompt-conditioned reward scoring: `pickscore_score`, `hpsv3_score`
- 4 new QualityMetrics fields for ChipQA, HDR-ChipQA, HDRMAX, and BrightRate: `chipqa_score`, `hdr_chipqa_score`, `hdrmax_score`, `brightrate_score`
- 3 new modules: **creativity** (VBench-2.0 artistic novelty), **chronomagic** (ChronoMagic-Bench MTScore + CHScore), **t2v_compbench** (T2V-CompBench 7 compositional sub-metrics)
- 13 new QualityMetrics fields for VBench-2.0 faithfulness, ChronoMagic-Bench, and T2V-CompBench coverage
- 4 upgraded modules with tiered backends and QM scoring: **physics** (`physics_score`), **human_fidelity** (`human_fidelity_score`), **commonsense** (`commonsense_score`), **dynamics_controllability** (CoTracker + camera motion classification)
- 6 new modules: **identity_loss**, **tifa**, **tonal_dynamic_range**, **nemo_curator**, **umap_projection**, **vlm_judge presets**
- `resolve_model_path()` and `download_model_file()` utilities in `config.py`
- Explicit config params for evaluation: `ocr_fidelity.expected_text`, `motion_amplitude.expected_motion`, `action_recognition.expected_action`

### Changed

- Base installation now includes the shared runtime dependencies used by bundled metrics
- HPSv3 loads directly through the Qwen2-VL reward path used by the bundled inference code
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
