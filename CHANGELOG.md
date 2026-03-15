# Changelog

All notable changes to Ayase will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.4]

### Fixed

- Pipeline `_mounted` guard: modules with missing dependencies now stay unmounted and are skipped during processing instead of silently running with broken state
- TUI had the same `_mounted` force-set bug — fixed
- **psnr_hvs**: was computing plain PSNR via `piq.psnr()` instead of CSF-weighted PSNR-HVS — now uses the correct DCT-based algorithm
- **kvq**: real KVQ model was loaded in `setup()` but never invoked in `process()` — added dispatch
- **rqvqa**: same issue — real RQ-VQA model loaded but never called — added dispatch
- **p1203**: official ITU-T P.1203 backend loaded but never called — added `_compute_official()` dispatch
- **t2v_score**: never attempted to load the real T2VScore model — added Tier 1 `AutoModel.from_pretrained` before CLIP fallback
- **st_lpips**: spatial quality component always used heuristic even when LPIPS/ST-LPIPS models were loaded — added model-based spatial dispatch
- **temporal_flickering**: loaded all video frames into memory with no limit (OOM risk on long videos) — added `max_frames` config (default 300) with uniform subsampling
- **hdr_sdr_vqa**: HDR detection relied solely on pixel dtype, missing actual HDR video content — now probes ffprobe color space metadata (bt2020, smpte2084)
- **dynamics_controllability**: returned 0.5 on failure, indistinguishable from a real score — now returns `None` and skips storing the metric
- **fvd**: docstring claimed I3D but code uses R3D-18 — fixed docstring and renamed `_i3d_model` → `_r3d_model`

### Changed

- `QualityMetrics` now uses `extra="forbid"` — typo'd field names raise `ValidationError` instead of being silently ignored
- `pyiqa` minimum version bumped from `0.1.7` to `0.1.13` in both `v-iqa` and `ml` dependency groups
- **c3dvqa**: documented ML feature-mapping as experimental heuristic proxy (renamed `activation_score` → `activation_proxy`)
- **tlvqm**: documented CNN pretrained feature-mapping as uncalibrated heuristic proxy

## [Unreleased]

### Added

- **Identity Loss** module (`identity_loss`) — reference-based face identity preservation metric with 4-tier backend: InsightFace ArcFace → DeepFace ArcFace → MediaPipe FaceMesh → skip. Outputs `identity_loss` (cosine distance, lower=better) and `face_recognition_score` (cosine similarity, higher=better). Used by IP-Adapter, DreamBooth, InstantID evaluation pipelines
- **TIFA** module (`tifa`) — Text-to-Image Faithfulness Assessment (ICCV 2023) with 3-tier backend: rule-based question generation + ViLT VQA → CLIP similarity proxy → heuristic. Outputs `tifa_score` (0–1, higher=better)
- **Tonal Dynamic Range** module (`tonal_dynamic_range`) — luminance histogram percentile range (0–100), with video frame subsampling
- **NeMo Curator Quality** module (`nemo_curator`) — caption text quality scoring with 3-tier backend: nvidia/quality-classifier-deberta → FastText → heuristic
- **UMAP Projection** module (`umap_projection`) — dataset-level embedding spread and coverage via CLIP features + 4-tier dimensionality reduction (UMAP → t-SNE → sklearn PCA → numpy PCA)
- **VLM Classification Presets** — new `"presets"` mode for `vlm_judge` module with 5 predefined label sets (shot_scale, time_of_day, clothing_style, mood, expression), VLM inference + heuristic fallback
- `QualityMetrics` fields: `identity_loss`, `face_recognition_score`, `tifa_score`, `tonal_dynamic_range`, `nemo_quality_score`, `nemo_quality_label`
- `DatasetStats` fields: `umap_spread`, `umap_coverage`
- `v-identity` optional dependency group: `insightface>=0.7.0`, `onnxruntime>=1.14.0`
- `resolve_model_path(model_name, models_dir)` in `config.py` — resolves HuggingFace model names to local paths with `--`-delimited fallback, or falls back to Hub download
- `download_model_file(relative_path, url, models_dir)` in `config.py` — downloads model weights to local cache with atomic `.part` temp file
- Explicit config params for downstream integration in evaluation modules:
  - `ocr_fidelity`: `expected_text` — pass OCR target directly instead of extracting from caption
  - `motion_amplitude`: `expected_motion` — pass `"large"`/`"fast"`/`"slow"` directly instead of keyword heuristics
  - `action_recognition`: `expected_action` — pass action description directly instead of relying on caption text

### Changed

- README metrics table redesigned as API reference: 5-column format (`#`, `Metric`, `Module`, `Input`, `Description`) replacing the old 4-column format — maps every QualityMetrics field to its producing module, required input type, and output range
- Removed `enable_ml` flag from all 235 modules — ML is now always enabled when dependencies are available
- TUI: `FolderSelectionScreen` starts at `Path.home()` instead of `"/"` (fixes Windows drive switching)
- TUI: added Windows drive letter selector, path input, and visual polish across all screens

### Removed

- `quality.py` and `video.py` legacy files

## [0.1.0] - 2024-12-01

### Core

- Modular pipeline architecture with `PipelineModule` base class and lifecycle hooks (`on_mount` → `process` → `post_process` → `on_dispose`)
- `AyasePipeline` high-level API with `run(dataset_path)` and `export(path, format)`
- `ModuleRegistry` with auto-discovery via `pkgutil.iter_modules`
- `DatasetScanner` for media file discovery with caption association (`.txt`, `.caption`, `.json`)
- Supported media: `.mp4`, `.webm`, `.avi`, `.mov`, `.mkv`, `.flv`, `.wmv`, `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tiff`

### Modules — 235 pipeline modules across 195 files

**Core / basic (5)** — metadata extraction, structural validation, exposure analysis, compression artifact detection

**Aesthetics (4)** — NIMA, LAION aesthetic predictor, general aesthetic scoring

**Text / OCR (3)** — PaddleOCR text detection, OCR fidelity (EvalCrafter NED score), BLIP/BLIP-2 captioning

**Motion & flow (6)** — RAFT optical flow, motion smoothness, motion amplitude classification, camera motion analysis, Kandinsky VideoMAE-V2 motion predictor

**Temporal consistency (8)** — CLIP temporal score, DINOv2 subject consistency, CLIP background consistency, object permanence, color consistency, flicker detection, temporal/style consistency

**Text-video alignment (4)** — CLIP score, BLIP-BLEU, VQAScore, T2VScore

**No-reference image/video quality (10)** — DOVER (aesthetic + technical), FastVQA, Q-Align, TOPIQ, MUSIQ, MANIQA, BRISQUE, NIQE, CLIP-IQA, imaging quality

**Full-reference quality (3)** — VMAF, DISTS, perceptual FR metrics (LPIPS, SSIM, MS-SSIM)

**SOTA video quality — CVPR/NeurIPS/EMNLP 2024-2025 (9)** — VideoScore, VideoReward, RQ-VQA, AIGV-Assessor, FineVQ, KVQ, JEDi, COVER, VIDEVAL

**Generation metrics (5)** — FVD, FVMD, Inception Score, I2V similarity (CLIP + DINOv2 + LPIPS sliding window), Stable Diffusion reference comparison

**Face & human (4)** — MediaPipe face/hand/pose, face consistency, face landmark quality, Face-IQA

**Scene & content (9)** — scene detection (TransNetV2), scene tagging, object detection (YOLO), action recognition (VideoMAE), spatial relationships, physics plausibility, commonsense validation, multi-object tracking

**Safety & ethics (5)** — NSFW classification, deepfake detection, harmful content, watermark detection, bias detection

**Audio (3)** — librosa audio analysis, PESQ speech quality, DNSMOS

**HDR / codec (2)** — HDR metadata validation, production quality checks

**Dataset operations (6)** — deduplication, embedding generation, diversity-based selection, dataset analytics, resolution bucketing (Wan 2.1, HunyuanVideo, CogVideoX, SVD bucket presets), LLM-assisted advisory

**Additional modules (~105)** — auto-discovered at runtime: ARNIQA, Butteraugli, C3D-VQA, CAMBI, CW-SSIM, DreamSim, FLIP, FloLPIPS, FUNQUE, HDR-VDP, HyperIQA, ILNIQE, LIQE, MAD, NLPD, NRQM, PaQ-2-PiQ, PieAPP, PIQE, PSNR-HVS, SSIMULACRA2, ST-GREED, ST-LPIPS, STRRED, TLVQM, VIF, VisQOL, VMAF-4K, VMAF-NEG, VMAF-Phone, XPSNR, depth estimation (Depth Anything), depth consistency, semantic segmentation consistency, and more

### CLI

- `ayase scan` — scan dataset with quality metrics report (`--format json/csv/markdown/html`, `--quick`, `--deep`)
- `ayase run` — run specific pipeline on target paths (`--pipeline` with inline module config syntax)
- `ayase filter` — filter dataset by quality thresholds (`--mode symlink/copy/list`, `--min-score`, `--metric`, `--aspect-ratio`, `--resolution`)
- `ayase stats` — distribution analysis with optional charts (`--format text/json/html`, `--chart`)
- `ayase tui` — launch Terminal User Interface
- `ayase modules list` — list all discovered modules (built-in + plugins)
- `ayase modules check` — verify module loading
- `ayase config init|show|edit|validate` — configuration management
- Auto-artifact export after every `scan` and `run`

### TUI

- Built with Textual — 6 screens: Welcome, Config (module selection + ordering + per-module config), Execution (progress + live log), Results (DataTable + export), Folder Selection (modal), Readiness Report (modal)
- Export dialog: JSON (full data), CSV (summary), HTML (readable)
- Module reordering with `u`/`d` keybindings

### Profile System

- `PipelineProfile` (Pydantic model): `name`, `modules` list, `module_config` per-module overrides
- Load from JSON or TOML files via `load_profile()`
- `instantiate_profile_modules(profile, config)` — creates configured module instances with model path resolution

### Configuration

- `AyaseConfig` via pydantic-settings: TOML files (`ayase.toml` or `~/.config/ayase/config.toml`) + environment variables (`AYASE_` prefix)
- Sections: `general` (parallel_jobs, cache, models_dir), `quality` (blur/compression thresholds), `output` (format, artifacts), `pipeline` (modules, plugin_folders), `filter` (mode, threshold)
- Plugin auto-discovery from configurable folders (default: `plugins/`)

### Export

- JSON (full metric data), CSV (summary table), HTML (styled report), Markdown
- `Pipeline.save_state()` / `Pipeline.load_state()` for resume support

### Data Model

- `QualityMetrics` with ~175 metric fields organized into 18 field groups: basic, aesthetic, alignment, motion, temporal, nr_quality, fr_quality, face, text, i2v, safety, audio, scene, distribution, hdr, codec, spatial, production, meta
- `Sample` with path, metadata, caption, detections, validation issues, quality metrics
- `ValidationIssue` with severity levels (INFO, WARNING, ERROR)

### Dependencies

- Core: typer, rich, pydantic, opencv-python, pillow, numpy, tqdm, pandas, pyarrow, imageio
- 15 optional dependency groups for selective installation (`ml`, `tui`, `v-perceptual`, `v-iqa`, `v-motion`, `v-ocr`, `v-face`, `v-object`, `v-watermark`, `v-audio`, `v-text`, `v-align`, `v-scene`, `dev`, `all`)
- Python 3.9–3.12, MIT license
