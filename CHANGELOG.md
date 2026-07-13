# Changelog

All notable changes to Ayase will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **vbench2_official**: native Apache-2.0 VBench 2.0 evaluation with setup-time checkpoint resolution, all 18 intrinsic-faithfulness dimensions, and five category aggregates.
- **worldmodelbench**: native VILA-judge evaluation for instruction following, physical adherence, commonsense, and seven component scores, with the official judge downloaded during setup.
- **mj_video**: native MJ-VIDEO-2B inference for the learned overall preference reward, five aspect rewards, and raw 28-criterion training diagnostics, with weights downloaded during setup.
- Pipeline JSON reports now expose requested, mounted, and failed module coverage through `run_status`.
- Interactive `ayase help [metric-or-module]` catalog with output semantics, owning modules, model assets, automatic-download status, dependencies, defaults, and runnable CLI/profile examples.

### Changed

- Dataset scanning indexes media and caption sidecars in one directory traversal.
- **rqvqa**: reports the released PLCC-trained raw regression output instead of clipping it to 0–1; the published score is unbounded and higher is better.
- **videoscore2**: soft ratings use the probability-weighted expectation on the documented 1–5 scale; the existing sampling default is unchanged.
- **vbench2_official**, **worldmodelbench**, and **mj_video**: external source archives and non-Hugging-Face checkpoints are resolved through the `AkaneTendo25/ayase-models` mirror; native Hugging Face model repositories remain unchanged.
- Mirrored model files now use flat module directories at the repository root; the duplicate `weights/` hierarchy and all runtime references to it were removed.
- The `dev` extra now installs `pytest-asyncio`, which is required by the TUI test suite.

### Fixed

- **rqvqa**: replaced the invalid Transformers loader with the published ten-fold Swin-B/BoT ensemble and its real SlowFast, Q-Align, LIQE, and FAST-VQA feature streams; all non-Hugging-Face checkpoints are resolved through the Ayase model mirror, and stochastic FAST-VQA fragment selection is isolated for reproducible scoring.
- **depth_consistency/depth_map_quality**: strict-load all supported MiDaS Small, Hybrid, and Large checkpoints from the Ayase model mirror.
- **song_eval**: register the published attention head as a real PyTorch module and strict-load its mirrored checkpoint instead of silently leaving its parameters unusable.
- Pipeline state fingerprints now include module source, runtime versions, model declarations, and score-affecting runtime configuration; sample cache signatures include structured metadata and annotations.
- Requested modules and hooks that fail initialization or execution now mark affected samples incomplete instead of producing cacheable partial results silently.
- **videophy**: VLM judgments ignore unrelated pre-existing issues and isolate physical and semantic issue ranges from each other.

## [0.1.64] - 2026-07-10

### Added

- **VMBench motion axes** — the five perception-aligned motion metrics from VMBench (AMAP-ML, ICCV 2025) are ported faithfully and run from a plain `pip install ayase`:
  - `object_integrity` (`object_integrity_score`): bone-length / joint-angle temporal integrity on the rtmlib RTMPose backend.
  - `vmbench_mss`: motion smoothness via Q-Align per-frame quality-jump detection over centred sliding windows.
  - `vmbench_pas` (`perceptible_amplitude_score`): subject-vs-background tracked-point motion amplitude (GroundingDINO + SAM + CoTracker3).
  - `vmbench_cas` (`commonsense_adherence_score`): VideoMAEv2 ViT-giant 5-level ordinal commonsense-plausibility rating.
  - `vmbench_tcs` (`temporal_coherence_score`): implausible object vanish/emerge over SAM2-propagated masks (GroundingDINO + SAM2 + CoTracker3).
- Self-contained, pure-torch vendored backends (no new dependencies; weights auto-downloaded from Hugging Face): GroundingDINO SwinB, SAM ViT-H, SAM 2.1 Hiera-L, VideoMAEv2 ViT-giant, Q-Align (mPLUG-Owl2), and CoTracker3 (offline).

## [0.1.63] - 2026-07-10

### Added

- **clip_image_similarity**: video support — CLIP-I is averaged over sampled frames (previously image-only, videos were skipped).

## [0.1.62] - 2026-07-10

### Changed

- **lip_sync**: LSE-C / LSE-D are now computed by a bundled reference-free SyncNet (mirrored weights), removing the external `syncnet` package dependency and the need for an external evaluation dataset.

## [0.1.61] - 2026-07-10

### Added

- **rtmpose_fidelity**: RTMPose keypoint-confidence pose-plausibility metric (rtmlib backend), plus `LSE-D` and `person_count` outputs.

## [0.1.60] - 2026-07-06

### Fixed

- **dreamsim**: the reference path (`image` + `reference_path`) now accepts video inputs. It opened both paths as still images, so a video reference or a video sample raised `cannot identify image file`; it now samples frames from videos (single frame for images), pairs them by position, and averages — matching how `dino_face_identity` handles video.

## [0.1.59] - 2026-07-06

### Fixed

- **dreamsim**: the perceptual-similarity metric never populated for real inputs. `dreamsim(pretrained=True)`'s `preprocess` already returns a batched `(1, 3, H, W)` tensor, so the extra `unsqueeze(0)` made it 5-D and the model's forward raised; the CPU input tensors were also never moved onto the model's (CUDA) device. Both the reference (`image` + `reference_path`) and inter-frame (video) paths now compute a value. The failure was silent because `process()` swallows the exception and the module tests only checked that the sample was returned — those tests now assert the metric is populated when the backend is available.

## [0.1.58] - 2026-07-04

### Changed

- **dreamsim**: the base DINO ViT-B/16 backbone pulled by `dreamsim(pretrained=True)` via `torch.hub` is now pre-cached from the `AkaneTendo25/ayase-models` HF mirror into the torch hub checkpoints directory before the model loads, matching the mirror path `dover` uses for its ConvNeXt backbone. The checkpoint (originally `dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth`) then loads offline. Metric values are unchanged — identical checkpoint.

## [0.1.57] - 2026-07-04

### Changed

- **dino_face_identity**: the DINOv2 backbone weights are now fetched from the `AkaneTendo25/ayase-models` HF mirror instead of the torch.hub entrypoint's fbaipublicfiles original; the architecture still comes from the torch.hub repo code. This aligns the checkpoint download with the mirror the other modules already use. Metric values are unchanged — identical architecture and identical weights. Adds a `models_dir` config key (default `"models"`) so the weights land in the standard Ayase model cache.

## [0.1.56] - 2026-07-03

### Changed

- **deps**: raised the torch stack ceiling to `<2.11` (`torch>=2.1.0,<2.11`, `torchvision>=0.16.0,<0.26`, `torchaudio>=2.1.0,<2.11`; previously `torch<2.8`). torch 2.10 is the newest release whose default PyPI wheels still ship CUDA 12.x builds, so a plain `pip install ayase` keeps a GPU-working torch on CUDA 12.x drivers while allowing modern torch. torch 2.11+ default to CUDA 13 wheels, which silently fall back to CPU on 12.x drivers, so they stay excluded until the target hosts move to CUDA 13.

## [0.1.54] - 2026-06-10

### Added

- **evoquality**: added ByteDance EvoQuality self-evolving VLM no-reference quality rating (1-5) with transformers and OpenAI-compatible endpoint backends.
- **qwen_image_bench**: added Qwen-Image-Bench text-to-image judge metrics for Quality, Aesthetics, Alignment, Real-world Fidelity, Creative Generation, and overall scoring.
- **hpsv2** / **unified_reward_2** / **unified_reward_edit**: added DiffSynth image-quality metric parity for standalone HPSv2, UnifiedReward 2.0, and UnifiedReward Edit scoring.
- **release**: added `ayase release prepare VERSION` to bump both version files, promote `[Unreleased]` changelog entries, and regenerate release docs/counts in one command.

### Changed

- **metrics**: every module now declares the category of the metrics it produces via a `metric_groups` class attribute, folded into `QualityMetrics._FIELD_GROUPS` at discovery time. The central grouping table is now empty — adding a self-describing module needs no edit to `models.py`. Added consistency tests that fail loudly if a metric field is ungrouped or a module declares a metric/group for a non-existent field.

### Removed

- **models**: dropped three placeholder `QualityMetrics` fields that no module produced (`human_preference_score`, `engagement_score`, `perceptual_hash`). Resume/state caches written by older versions still load — unknown metric keys are dropped with a warning instead of discarding the cached sample.

## [0.1.45]

### Fixed

- **_compat.ensure_paddle_gpu**: probe paddle presence via `importlib.metadata` instead of `import paddle` — the latter cached the CPU build in `sys.modules` and prevented the lazy GPU swap from taking effect in the same process.

## [0.1.44]

### Changed

- **deps**: only pin CPU `paddlepaddle>=3.0` (PyPI forbids direct URLs to GPU wheels). The OCR modules detect CUDA at first `setup()` and lazily install the matching `paddlepaddle-gpu==3.3.1` wheel from paddle.org index via subprocess — so `pip install ayase` followed by OCR usage just works on CUDA hosts without extra flags.

## [0.1.43]

### Changed

- **deps**: GPU paddle wheel is now pinned via PEP 508 direct URL for Linux + Python 3.10 + x86_64 (CUDA 12.6 build of paddlepaddle-gpu 3.3.1). Other platforms fall back to plain `paddlepaddle>=3.0`. Removes the manual extra step required to get GPU OCR working.

### Removed

- **ayase-bootstrap** CLI (introduced in 0.1.42) — no longer needed now that paddle-gpu is resolved by pip directly.

## [0.1.42]

### Changed

- **deps**: require `paddleocr>=3.0` and `paddlepaddle>=3.0`; removed paddleocr 2.x compatibility branches in `ocr_fidelity` and `text_detection`.
- **ocr_fidelity** / **text_detection**: accept `text_recognition_model_name` in module config to bypass `lang`-based auto-routing (e.g. force `cyrillic_PP-OCRv5_mobile_rec` instead of the default `eslav_PP-OCRv5_mobile_rec`, which hangs on init in PaddleOCR 3.5.0).

### Added

- **ayase-bootstrap** CLI: `ayase-bootstrap --gpu` auto-detects CUDA and installs the matching `paddlepaddle-gpu` wheel from paddle.org index (CUDA 11.8 / 12.0 / 12.3 / 12.6). Required for GPU OCR inference, which standard `pip install` cannot resolve from `pyproject.toml`.

## [0.1.41]

### Fixed

- **text_detection**: PaddleOCR 2.x is now supported alongside 3.x (the module previously called `.predict()` unconditionally and silently produced `None` on 2.x).

## [0.1.40]

### Changed

- **text_detection**: `lang` is now read from module config (defaults to `"en"`); previously hard-coded to English.

## [0.1.39]

### Fixed

- **ocr_fidelity**: aggregate frames via per-frame best NED (EvalCrafter "rendered correctly at least once" semantics).

## [0.1.38]

### Fixed

- **background_consistency**: fixed `F.cosine_similarity` 1D crash on the 2D embedding tensor returned by `cached_clip_image_features`; the metric reports values again instead of silently falling back to `None`.
- **i2v_similarity**: stripped OpenAI-CLIP metadata keys (`context_length`, `input_resolution`, `vocab_size`) before `open_clip` `load_state_dict`, allowing the bundled `.safetensors` checkpoint to load.
- **clip_temporal**: reverted `face_consistency` to consecutive-pair averaging (EvalCrafter rolling window) after the 0.1.37 switch to first-frame similarity caused unexpected drops on long clips with camera motion.

## [0.1.37]

### Added

- **pipeline**: added runtime timing, per-sample frame/feature caching, shared pipeline-scoped model resources, and optional attention backend config for faster inference.
- **pipeline**: added optional dataset-level sample batching via `sample_batch_size`, `Pipeline.process_samples()`, and overridable module `process_batch()` hooks.
- **examples**: added `examples/benchmark_inference.py` for comparing inference throughput across `sample_batch_size` values.
- **audio_isc** and **audio_kl**: added optional dataset-level audio distribution metrics with PANNs/PaSST backends.
- **clap_score**, **imagebind_score**, and **nima_legacy_onnx**: added first-class optional CLAP/ImageBind audio-text alignment and legacy NIMA ONNX modules.

### Changed

- **video modules**: batched selected CLIP/DINO/VLM frame inference paths and reused shared frame sampling to reduce duplicate video decoding.
- **CLIP VQA modules**: extended shared CLIP model, frame, and feature reuse to Q-CLIP, VQA^2, UGVQ, VQAThinker, VQ-Insight, VideoReward, and background consistency.
- **CLIP modules**: added true cross-sample CLIP frame batching for semantic alignment, CLIP temporal consistency, and the shared-cache CLIP VQA/reward modules.
- **CLIP modules**: extended shared HuggingFace CLIP/X-CLIP loading, feature caching, and batch feature paths across temporal, safety, compositional, distribution, and dataset-analytics modules.
- **OpenAI CLIP modules**: shared `clip.load()` resources and cached OpenAI CLIP text/image features across LMM-VQA, PreResQ, VQAScore fallback, MD-VQA, ModularBVQA, and Unified-VQA.
- **fad**: added PANNs CNN14 and PaSST backbone support while preserving the legacy VGGish-compatible dataset metric aliases.

### Fixed

- **OpenAI CLIP modules**: fixed RGB/BGR frame conversion on LMM-VQA and PreResQ CLIP paths.
- **metrics docs**: declared CLAP, ImageBind, legacy NIMA, FAD-backbone, audio ISC, and audio KL outputs so generated docs and dataset stats match module behavior.

## [0.1.36]

### Added

- **fvd**: added `content_debiased` and `dinov2` backbones (Ge et al. CVPR 2024 + rFVD); legacy `r3d18` remains the default.
- **audio_nisqa**: added NISQA multidimensional speech quality (MOS, noisiness, coloration, discontinuity, loudness) with vendored upstream source and auto-downloaded weights to avoid the `pip install nisqa` dependency cascade.
- **audio_peaq**: added reference-based ITU-R BS.1387 audio codec quality (ODG, DI) with peaqb/Bark-band/log-spectral-distance tiers.
- **geneval**: added GenEval T2I compositional benchmark (NeurIPS 2024) with mmdetection/YOLO/CLIP tiers.
- **tc_bench**: added TC-Bench temporal compositionality scoring for text-to-video.
- **videophy**: added VideoPhy-2 VLM-based physics adherence (LLaVA-NeXT-Video / vlm_judge / trajectory tiers).
- **entitybench**: added cross-shot identity/appearance consistency batch metric (DINOv2+InsightFace / CLIP / histogram tiers).

### Changed

- **deps**: tightened pins so fresh installs get GPU-working torch on CUDA 12.x drivers - `torch<2.8`, added `torchaudio<2.8`, capped `transformers<5.0` and `huggingface-hub<2.0`, dropped deprecated `typer[all]` extra.

## [0.1.31]

### Added

- **blip_score**: added BLIP caption-alignment scoring for visual samples.
- **cmmd**: added dataset CMMD distribution scoring with model and proxy backends.
- **prdc_dinov2**: added precision, recall, density, and coverage dataset metrics.
- **fid**: added dataset FID scoring with Inception and proxy feature backends.
- **asr_cer** and **asr_wer**: added cached ASR transcript error-rate metrics.
- **scoreq**, **ttsds2**, and **audio_utmos_v2**: added speech-quality scoring modules with graceful proxy fallbacks.
- **human_clap** and **pam**: added CLAP-based audio-text relevance and anti-prompt metrics.
- **kad**: added Kernel Audio Distance dataset scoring.
- **aqascore**: added opt-in audio question-answering quality scoring.
- **image**: added public `ayase.image` loading and frame-sampling helpers for downstream image adapters.

### Changed

- **semantic_alignment**: added configurable OpenCLIP backend support for legacy image-to-image adapter compatibility.
- **fad**: added optional FAD∞ extrapolation mode alongside the existing FAD dataset metric.

### Fixed

- **dnsmos**: fixed torchmetrics backend detection to use `DeepNoiseSuppressionMeanOpinionScore`.
- **audio_visual_sync**: added configurable segment sizing and an optional Synchformer backend hook while preserving the energy-correlation default.

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
