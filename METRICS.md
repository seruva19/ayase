# Ayase Metrics Reference

194 modules, 225 output fields.

## Audio

| Module | Outputs | Description | Config |
|--------|---------|-------------|--------|
| `audio_estoi` | `estoi_score`  -  ESTOI intelligibility (0-1, higher=better) | ESTOI speech intelligibility (full-reference) | `target_sr=10000`, `warning_threshold=0.5` |
| `audio_lpdist` | `lpdist_score`  -  Log-Power Spectral Distance (lower=better) | Log-Power Spectral Distance (full-reference audio) | `target_sr=16000`, `n_mels=80`, +1 |
| `audio_mcd` | `mcd_score`  -  Mel Cepstral Distortion (dB, lower=better) | Mel Cepstral Distortion for TTS/VC quality (full-reference) | `target_sr=16000`, `n_mfcc=13`, +1 |
| `audio_pesq` | `pesq_score`  -  PESQ (-0.5 to 4.5, higher=better) | PESQ speech quality (full-reference, ITU-T P.862) | `target_sr=16000`, `warning_threshold=3.0` |
| `audio_si_sdr` | `si_sdr_score`  -  Scale-Invariant SDR (dB, higher=better) | Scale-Invariant SDR for audio quality (full-reference) | `target_sr=16000`, `warning_threshold=0.0` |
| `audio_text_alignment` | - | Multimodal alignment check (Audio-Text) using CLAP | `alignment_threshold=0.2`, `model_name=laion/clap-htsat-fused` |
| `audio_utmos` | `utmos_score`  -  UTMOS predicted MOS (1-5, higher=better) | UTMOS no-reference MOS prediction for speech quality | `target_sr=16000`, `warning_threshold=3.0` |
| `audio_visual_sync` | `av_sync_offset`  -  Audio-video sync offset in ms | Audio-video synchronisation offset detection | `max_frames=600`, `warning_threshold_ms=80.0` |
| `dnsmos` | `dnsmos_overall`  -  DNSMOS overall MOS (1-5, higher=better); `dnsmos_sig`  -  DNSMOS signal quality (1-5, higher=better); `dnsmos_bak`  -  DNSMOS background quality (1-5, higher=better) | DNSMOS non-intrusive audio quality (Microsoft, 1-5 MOS) | - |
| `p1203` | `p1203_mos`  -  ITU-T P.1203 streaming QoE MOS (1-5) | ITU-T P.1203 streaming QoE estimation (1-5 MOS) | `display_size=phone` |
| `visqol` | `visqol`  -  ViSQOL audio quality MOS (1-5, higher=better) | ViSQOL audio quality MOS (Google, 1-5, higher=better) | `mode=audio` |

## Depth

| Module | Outputs | Description | Config |
|--------|---------|-------------|--------|
| `depth_anything` | `depth_anything_score`  -  Monocular depth quality; `depth_anything_consistency`  -  Temporal depth consistency | Depth Anything V2 monocular depth estimation and consistency | `model_name=depth-anything/Depth-Anything-V2-Small-hf`, `subsample=8` |
| `depth_consistency` | `depth_temporal_consistency`  -  Depth map correlation 0-1 (higher=better) | Monocular depth temporal consistency | `model_type=MiDaS_small`, `device=auto`, +3 |
| `depth_map_quality` | `depth_quality`  -  Depth map quality 0-100 (higher=better) | Monocular depth map quality (sharpness, completeness, edge alignment) | `model_type=MiDaS_small`, `device=auto`, +2 |

## Face Quality

| Module | Outputs | Description | Config |
|--------|---------|-------------|--------|
| `face_fidelity` | `face_count`; `face_quality_score`  -  Composite face quality 0-100 (higher=better) | Face detection and per-face quality assessment | `backend=haar`, `subsample=5`, +4 |
| `face_iqa` | `face_iqa_score`  -  TOPIQ-face face quality (higher=better) | Face-specific IQA via TOPIQ-face (GFIQA-trained, higher=better) | `subsample=8` |
| `face_landmark_quality` | `face_landmark_jitter`  -  Landmark jitter 0-100 (lower=better); `face_identity_consistency`  -  Temporal face identity stability (0-1) | Facial landmark jitter, expression smoothness, identity consistency | `subsample=2`, `max_frames=300`, +1 |

## Full-Reference

| Module | Outputs | Description | Config |
|--------|---------|-------------|--------|
| `ahiq` | `ahiq`  -  Attention Hybrid IQA (higher=better) | Attention-based Hybrid IQA full-reference (higher=better) | `subsample=8` |
| `butteraugli` | `butteraugli`  -  Butteraugli perceptual distance (lower=better) | Butteraugli perceptual distance (Google/JPEG XL, lower=better) | `subsample=5`, `warning_threshold=2.0` |
| `cgvqm` | `cgvqm`  -  CGVQM gaming quality (higher=better) | CGVQM gaming/rendering quality metric (Intel, higher=better) | `subsample=5` |
| `ciede2000` | `ciede2000`  -  CIEDE2000 perceptual color difference (lower=better) | CIEDE2000 perceptual color difference (lower=better) | `subsample=5` |
| `ckdn` | `ckdn_score`  -  CKDN knowledge distillation FR | CKDN knowledge distillation FR image quality | `subsample=4` |
| `cw_ssim` | `cw_ssim`  -  Complex Wavelet SSIM (0-1, higher=better) | Complex Wavelet SSIM full-reference metric (0-1, higher=better) | `subsample=8` |
| `deepwsd` | `deepwsd_score`  -  DeepWSD Wasserstein distance FR | DeepWSD Wasserstein distance FR image quality | `subsample=4` |
| `delta_ictcp` | `delta_ictcp`  -  Delta ICtCp HDR color difference (lower=better) | Delta ICtCp HDR perceptual color difference (lower=better) | `subsample=5` |
| `dmm` | `dmm`  -  DMM Detail Model Metric FR (higher=better) | DMM detail model metric full-reference (higher=better) | `subsample=8` |
| `dreamsim_metric` | `dreamsim`  -  DreamSim CLIP+DINO similarity (lower=more similar) | DreamSim foundation model perceptual similarity (CLIP+DINO ensemble) | `subsample=8`, `model_type=ensemble` |
| `flip_metric` | `flip_score`  -  NVIDIA FLIP perceptual metric (0-1, lower=better) | NVIDIA FLIP perceptual difference (0-1, lower=better) | `subsample=5`, `warning_threshold=0.3` |
| `flolpips` | `flolpips`  -  FloLPIPS flow-based perceptual FR | Flow-compensated perceptual distance (RAFT+LPIPS, Farneback+LPIPS, or MSE fallback) | `subsample=8` |
| `funque` | `funque_score`  -  FUNQUE unified quality (beats VMAF) | Fused quality evaluator (FUNQUE package, handcrafted FR, or NR fallback) | `subsample=8` |
| `hdr_vdp` | `hdr_vdp`  -  HDR-VDP visual difference predictor (higher=better) | HDR-VDP visual difference predictor (higher=better) | `subsample=5` |
| `hdr_vqm` | `hdr_vqm`  -  HDR-VQM HDR video quality FR | HDR-aware video quality (PU21+wavelet FR or gamma heuristic fallback) | `subsample=8` |
| `mad_metric` | `mad`  -  Most Apparent Distortion (lower=better) | Most Apparent Distortion full-reference metric (lower=better) | `subsample=8` |
| `movie` | `movie_score`  -  MOVIE motion trajectory FR | Video quality via spatiotemporal Gabor decomposition (FR or NR fallback) | `subsample=8` |
| `ms_ssim` | `ms_ssim`  -  Multi-Scale SSIM (0-1) | Multi-Scale SSIM perceptual similarity metric (full-reference) | `scales=5`, `weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333]`, +3 |
| `nlpd_metric` | `nlpd`  -  Normalized Laplacian Pyramid Distance (lower=better) | Normalized Laplacian Pyramid Distance full-reference (lower=better) | `subsample=8` |
| `pieapp` | `pieapp`  -  PieAPP pairwise preference (lower=better) | PieAPP full-reference perceptual error via pairwise preference (lower=better) | `subsample=8` |
| `psnr_hvs` | `psnr_hvs`  -  PSNR-HVS perceptually weighted (dB, higher=better) | PSNR-HVS + PSNR-HVS-M perceptually weighted PSNR (dB, higher=better) | `subsample=5` |
| `ssimc` | `ssimc`  -  Complex Wavelet SSIM-C FR (higher=better) | SSIM-C complex wavelet structural similarity FR (higher=better) | `subsample=8` |
| `st_greed` | `st_greed_score`  -  ST-GREED variable frame rate FR | Spatial-temporal entropic quality (FR entropic difference or NR heuristic fallback) | `subsample=16` |
| `st_lpips` | `st_lpips`  -  ST-LPIPS spatiotemporal perceptual FR | Spatiotemporal perceptual video quality (ST-LPIPS model, LPIPS, or heuristic fallback) | `subsample=8` |
| `topiq_fr` | `topiq_fr`  -  TOPIQ full-reference (higher=better) | TOPIQ full-reference top-down semantics-to-distortion IQA (higher=better) | `subsample=8` |
| `wadiqam_fr` | `wadiqam_fr`  -  WaDIQaM full-reference (higher=better) | WaDIQaM full-reference deep quality metric (higher=better) | `subsample=8` |

## HDR

| Module | Outputs | Description | Config |
|--------|---------|-------------|--------|
| `hdr_metadata` | `max_fall`  -  MaxFALL frame average light level (nits); `max_cll`  -  MaxCLL content light level (nits) | MaxFALL + MaxCLL HDR static metadata analysis | `subsample=3`, `peak_nits=10000.0` |
| `pu_metrics` | `pu_psnr`  -  PU-PSNR perceptually uniform HDR (dB, higher=better); `pu_ssim`  -  PU-SSIM perceptually uniform HDR (0-1, higher=better) | PU-PSNR + PU-SSIM for HDR content (perceptually uniform) | `subsample=5`, `assume_nits_range=10000.0` |

## Image-to-Video Reference

| Module | Outputs | Description | Config |
|--------|---------|-------------|--------|
| `i2v_similarity` | `i2v_clip`  -  CLIP image-video similarity (0-1); `i2v_dino`  -  DINOv2 image-video similarity (0-1); `i2v_lpips`  -  LPIPS image-video distance (0-1, lower=better); `i2v_quality`  -  Aggregated I2V quality (0-100) | Image-to-Video reference similarity using CLIP, DINOv2, and LPIPS (sliding window) | `window_size=16`, `stride=8`, +7 |

## Image: No-Reference

| Module | Outputs | Description | Config |
|--------|---------|-------------|--------|
| `4k_vqa` | `hdr_quality`  -  HDR-specific quality; `sdr_quality`  -  SDR-specific quality; `technical_score`  -  Composite technical score | Memory-efficient quality assessment for 4K+ videos | `tile_size=512`, `subsample=10` |
| `action_recognition` | `action_confidence`  -  Top-1 action confidence (0-100); `action_score`  -  Caption-action fidelity (0-100) | Recognizes human actions (VideoMAE / UMT) - Supports Heavy Models | `model_name=MCG-NJU/videomae-large-finetuned-kinetics`, `caption_matching=False`, +3 |
| `advanced_flow` | `flow_score` | RAFT optical flow: flow_score (all consecutive pairs) | `use_large_model=True`, `max_frames=150` |
| `aesthetic` | `aesthetic_score`  -  0-10, from aesthetic predictor; `vqa_a_score` | Estimates aesthetic quality using Aesthetic Predictor V2.5 | `num_frames=5`, `trust_remote_code=True`, +1 |
| `aesthetic_scoring` | `aesthetic_score`  -  0-10, from aesthetic predictor | Calculates aesthetic score (1-10) using LAION-Aesthetics MLP | `models_dir=models` |
| `afine` | `afine_score`  -  A-FINE fidelity-naturalness (CVPR 2025) | A-FINE adaptive fidelity-naturalness IQA (CVPR 2025) | `subsample=4` |
| `arniqa` | `arniqa_score`  -  ARNIQA (higher=better) | ARNIQA no-reference image quality assessment | `subsample=8` |
| `audio` | - | Validates audio stream quality and presence | `require_audio=False`, `min_sample_rate=44100`, +4 |
| `background_diversity` | - | Checks background complexity (entropy) to detect concept bleeding | `min_entropy_threshold=3.0`, `use_rembg=True` |
| `basic` | `blur_score`  -  Laplacian variance; `brightness`; `contrast`; `saturation`  -  Advanced metrics; `noise_score`; `artifacts_score`; `technical_score`  -  Composite technical score; `vqa_t_score`; `gradient_detail`  -  Sobel gradient detail (0-100) | Comprehensive technical quality assessment (blur, noise, artifacts, contrast) | `threshold=40.0`, `blur_threshold=100.0`, +1 |
| `bd_rate` | - | BD-Rate codec comparison (dataset-level, negative%=better) | `quality_metric=psnr` |
| `brisque` | `brisque`  -  BRISQUE (0-100, lower=better) | BRISQUE no-reference image quality (lower=better) | `subsample=3`, `warning_threshold=50.0` |
| `cambi` | `cambi`  -  CAMBI banding index (0-24, lower=better) | CAMBI banding/contouring detector (Netflix, 0-24, lower=better) | `warning_threshold=5.0` |
| `celebrity_id` | `celebrity_id_score` | Face identity verification using DeepFace (EvalCrafter celebrity_id_score) | `reference_dir=`, `num_frames=8`, +2 |
| `cnniqa` | `cnniqa_score`  -  CNNIQA blind CNN IQA | CNNIQA blind CNN-based image quality assessment | `subsample=4` |
| `codec_compatibility` | - | Validates codec, pixel format, and container for ML dataloader compatibility | `min_bitrate_kbps=500`, `min_bpp=0.02` |
| `codec_specific_quality` | `codec_efficiency`  -  Quality-per-bit efficiency 0-100 (higher=better); `gop_quality`  -  GOP structure appropriateness 0-100 (higher=better); `codec_artifacts`  -  Block artifact severity 0-100 (lower=better) | Codec-level efficiency, GOP quality, and artifact detection | `max_frames=100`, `subsample=10`, +2 |
| `color_consistency` | `color_score` | Verifies color attributes in prompt vs video content | - |
| `commonsense` | `commonsense_score`  -  Common sense adherence (0-1, higher=better) | Common sense adherence (VLM / ViLT VQA / heuristic) | `model_name=dandelin/vilt-b32-finetuned-vqa`, `vlm_model=llava-hf/llava-1.5-7b-hf` |
| `compare2score` | `compare2score`  -  Compare2Score comparison-based | Compare2Score comparison-based NR image quality | `subsample=4` |
| `contrique` | `contrique_score`  -  CONTRIQUE contrastive IQA (higher=better) | Contrastive no-reference IQA | `subsample=5` |
| `cpbd` | `blur_score`  -  Laplacian variance | Cumulative Probability of Blur Detection (Perceptual Blur) | `threshold_cpbd=0.65`, `threshold_heuristic=10.0` |
| `creativity` | `creativity_score`  -  Artistic novelty (0-1, higher=better) | Artistic novelty assessment (VLM / CLIP / heuristic) | `vlm_model=llava-hf/llava-1.5-7b-hf` |
| `dbcnn` | `dbcnn_score`  -  DBCNN bilinear CNN (higher=better) | DBCNN deep bilinear CNN for no-reference IQA | `subsample=8` |
| `decoder_stress` | - | Random access decoder stress test | `num_probes=5`, `check_integrity=True` |
| `dedup` | - | Detects duplicates using Perceptual Hashing (pHash) | - |
| `dists` | `dists`  -  DISTS (0-1, lower=more similar) | Deep Image Structure and Texture Similarity (full-reference) | `subsample=5`, `warning_threshold=0.3`, +1 |
| `diversity_selection` | - | Flags redundant samples using embedding similarity (Deduplication) | `similarity_threshold=0.95`, `priority_metric=aesthetic_score` |
| `dynamics_controllability` | `dynamics_controllability`  -  Motion control fidelity | Assesses motion controllability based on text-motion alignment | `subsample=16` |
| `dynamics_range` | `dynamics_range`  -  Extent of content variation | Measures extent of motion and content variation (DEVIL protocol) | `scene_change_threshold=30.0` |
| `embedding` | - | Calculates X-CLIP embeddings for similarity search | `model_name=microsoft/xclip-base-patch32`, `num_frames=8` |
| `exposure` | - | Checks for overexposure, underexposure, and low contrast using histograms | `overexposure_threshold=0.3`, `underexposure_threshold=0.3`, +1 |
| `human_fidelity` | `human_fidelity_score`  -  Body/hand/face quality (0-1, higher=better) | Human body/hand/face fidelity (DWPose / MediaPipe / heuristic) | - |
| `hyperiqa` | `hyperiqa_score`  -  HyperIQA adaptive NR-IQA | HyperIQA adaptive hypernetwork NR image quality | `subsample=4` |
| `identity_loss` | `identity_loss`  -  Face identity cosine distance (0-1, lower=better); `face_recognition_score`  -  Face identity cosine similarity (0-1, higher=better) | Face identity preservation metric (cosine distance/similarity vs reference) | `model_name=buffalo_l`, `subsample=8`, +1 |
| `ilniqe` | `ilniqe`  -  IL-NIQE Integrated Local NIQE (lower=better) | IL-NIQE integrated local no-reference quality (lower=better) | `subsample=3`, `warning_threshold=50.0` |
| `imaging_quality` | `noise_score`; `artifacts_score` | Assesses technical quality (Noise, Blockiness) - Proxy for MUSIQ/DOVER | `noise_threshold=20.0` |
| `inception_score` | `is_score` | Inception Score (IS) using InceptionV3  -  EvalCrafter quality metric | `num_frames=16`, `splits=1` |
| `jedi_metric` | - | JEDi distribution metric (V-JEPA + MMD, ICLR 2025) | `num_frames=16`, `batch_size=8`, +2 |
| `judder_stutter` | `judder_score`  -  Judder severity 0-100 (lower=better); `stutter_score`  -  Duplicate/dropped frames 0-100 (lower=better) | Detects judder (uneven cadence) and stutter (duplicate frames) | `max_frames=600`, `duplicate_threshold=1.0`, +1 |
| `knowledge_graph` | - | Generates a conceptual knowledge graph of the video dataset | `output_file=knowledge_graph.json`, `min_confidence=0.5`, +4 |
| `laion_aesthetic` | `laion_aesthetic`  -  LAION Aesthetics V2 (0-10) | LAION Aesthetics V2 predictor (0-10, industry standard) | `subsample=4` |
| `letterbox` | `letterbox_ratio`  -  Border/letterbox fraction (0-1, 0=no borders) | Border/letterbox detection (0-1, 0=no borders) | `threshold=16`, `subsample=4` |
| `liqe` | `liqe_score`  -  LIQE lightweight IQA (higher=better) | LIQE lightweight no-reference IQA | `subsample=5`, `warning_threshold=2.5` |
| `llm_advisor` | - | Rule-based improvement recommendations derived from quality metrics (no LLM used) | `severity_level=INFO` |
| `llm_descriptive_qa` | `confidence_score`  -  Prediction confidence | LMM-based interpretable quality assessment with explanations | `model_name=llava-hf/llava-v1.6-mistral-7b-hf`, `use_openai=False`, +3 |
| `maclip` | `maclip_score`  -  MACLIP multi-attribute CLIP NR-IQA (higher=better) | MACLIP multi-attribute CLIP no-reference quality (higher=better) | `subsample=3` |
| `maniqa` | `maniqa_score`  -  MANIQA multi-attention (higher=better) | MANIQA multi-dimension attention no-reference IQA | `subsample=8` |
| `metadata` | - | Checks video/image metadata (resolution, FPS, duration, integrity) | `min_resolution=720`, `min_fps=15`, +4 |
| `multi_view_consistency` | `multiview_consistency`  -  Geometric consistency 0-1 (higher=better) | Geometric multi-view consistency via epipolar analysis | `subsample=5`, `max_pairs=30`, +1 |
| `multiple_objects` | - | Verifies object count matches caption (VBench multiple_objects dimension) | `tolerance=1` |
| `musiq` | `musiq_score`  -  MUSIQ multi-scale IQA (higher=better) | Multi-Scale Image Quality Transformer (no-reference) | `variant=musiq`, `subsample=5`, +1 |
| `naturalness` | `naturalness_score`  -  Natural scene statistics | Measures naturalness of content (natural vs synthetic) | `use_pyiqa=True`, `subsample=2`, +1 |
| `nima` | `nima_score`  -  NIMA aesthetic+technical (1-10, higher=better) | NIMA aesthetic and technical image quality (1-10 scale) | `subsample=8` |
| `niqe` | `niqe`  -  Natural Image Quality Evaluator (lower=better) | Natural Image Quality Evaluator (no-reference) | `subsample=2`, `warning_threshold=7.0` |
| `nrqm` | `nrqm`  -  NRQM No-Reference Quality Metric (higher=better) | NRQM no-reference quality metric for super-resolution (higher=better) | `subsample=3` |
| `object_detection` | `detection_score`; `count_score`; `is_score` | Detects objects (GRiT / YOLOv8) - Supports Heavy Models | `model_name=yolov8n.pt`, `use_yolo_world=False`, +1 |
| `paq2piq` | `paq2piq_score`  -  PaQ-2-PiQ patch-to-picture (CVPR 2020) | PaQ-2-PiQ patch-to-picture NR quality (CVPR 2020) | `subsample=4` |
| `paranoid_decoder` | - | Deep bitstream validation using FFmpeg (Paranoid Mode) | `timeout=60`, `strict_mode=True` |
| `perceptual_fr` | `fsim`  -  Feature Similarity Index (0-1, higher=better); `gmsd`  -  Gradient Magnitude Similarity Deviation (lower=better); `vsi_score`  -  Visual Saliency Index (0-1, higher=better) | FSIM + GMSD + VSI full-reference perceptual metrics | `subsample=5`, `device=auto` |
| `physics` | `physics_score`  -  Physics plausibility (0-1, higher=better) | Physics plausibility via trajectory analysis (CoTracker / LK / heuristic) | `subsample=16`, `accel_threshold=50.0` |
| `pi_metric` | `pi_score`  -  Perceptual Index (PIRM challenge, lower=better) | Perceptual Index (PIRM challenge metric, lower=better) | `subsample=3` |
| `piqe` | `piqe`  -  PIQE perception-based NR-IQA (lower=better) | PIQE perception-based no-reference quality (lower=better) | `subsample=3`, `warning_threshold=50.0` |
| `production_quality` | `white_balance_score`  -  White balance accuracy 0-100; `focus_quality`  -  Sharpness/focus quality 0-100; `banding_severity`  -  Colour banding 0-100 (lower=better); `color_grading_score`  -  Colour consistency 0-100; `exposure_consistency`  -  Exposure stability 0-100 | Professional production quality (colour, exposure, focus, banding) | `max_frames=150` |
| `promptiqa` | `promptiqa_score`  -  Few-shot NR-IQA score | Prompt-guided NR-IQA (PromptIQA via pyiqa, TOPIQ-NR, or CLIP-IQA+ fallback) | `subsample=4` |
| `q_align` | `qalign_quality`  -  Q-Align technical quality (1-5, higher=better); `qalign_aesthetic`  -  Q-Align aesthetic quality (1-5, higher=better) | Q-Align unified quality + aesthetic assessment (ICML 2024) | `model_name=q-future/one-align`, `dtype=float16`, +6 |
| `qcn` | `qcn_score`  -  Geometric order blind IQA | Blind IQA (QCN via pyiqa, or HyperIQA fallback) | `subsample=4` |
| `qualiclip` | `qualiclip_score`  -  QualiCLIP opinion-unaware (higher=better) | QualiCLIP opinion-unaware CLIP-based no-reference IQA | `subsample=8` |
| `resolution_bucketing` | - | Validates resolution/aspect-ratio fit for training buckets | `max_crop_ratio=0.15`, `max_scale_factor=2.0`, +3 |
| `scene` | - | Detects scene cuts and shots using PySceneDetect | `threshold=27.0`, `min_scene_len=15`, +2 |
| `scene_complexity` | `scene_complexity`  -  Visual complexity score | Spatial and temporal scene complexity analysis | `subsample=2`, `spatial_weight=0.5`, +1 |
| `scene_tagging` | - | Tags scene context (Proxy for Tag2Text/RAM using CLIP) | `models_dir=models` |
| `spatial_relationship` | - | Verifies spatial relations (left/right/top/bottom) in prompt vs detections | - |
| `spectral` | `spectral_entropy`  -  DINOv2 spectral entropy; `spectral_rank`  -  DINOv2 effective rank ratio | Analyzes spectral complexity (Effective Rank) of video features (DINOv2) | `model_type=dinov2_vits14`, `sample_rate=8`, +2 |
| `spectral_upscaling` | - | Detection of upscaled/fake high-resolution content | `energy_threshold=0.05`, `sample_rate=20` |
| `stereoscopic_quality` | `stereo_comfort_score`  -  Stereo viewing comfort 0-100 (higher=better) | Stereo 3D comfort and quality assessment | `stereo_format=auto`, `subsample=10`, +3 |
| `structural` | - | Checks structural integrity (scene cuts, black bars) | `detect_cuts=True`, `detect_black_bars=True` |
| `style_consistency` | - | Appearance Style verification (Gram Matrix Consistency) | - |
| `text` | `ocr_area_ratio`  -  0-1; `text_overlay_score`  -  Text overlay severity (0-1) | Detects text/watermarks using OCR (PaddleOCR / Tesseract) | `use_paddle=True`, `max_text_area=0.05` |
| `ti_si` | `spatial_information`  -  ITU-T P.910 SI (higher=more detail); `temporal_information`  -  ITU-T P.910 TI (higher=more motion) | ITU-T P.910 Temporal & Spatial Information | `max_frames=300` |
| `tonal_dynamic_range` | `tonal_dynamic_range`  -  Luminance histogram span (0-100) | Luminance histogram tonal range (0-100) | `low_percentile=1`, `high_percentile=99`, +1 |
| `topiq` | `topiq_score`  -  TOPIQ transformer-based IQA (higher=better) | TOPIQ transformer-based no-reference IQA | `variant=topiq_nr`, `subsample=5`, +1 |
| `trajan` | `trajan_score`  -  Point track motion consistency | Motion consistency via point tracking (CoTracker or Lucas-Kanade fallback) | `num_frames=16`, `num_points=256` |
| `tres` | `tres_score`  -  TReS transformer IQA (WACV 2022) | TReS transformer-based NR image quality (WACV 2022) | `subsample=4` |
| `unique_iqa` | `unique_score`  -  UNIQUE unified NR-IQA (TIP 2021) | UNIQUE unified NR image quality (TIP 2021) | `subsample=4` |
| `usability_rate` | `usability_rate`  -  Percentage of usable frames | Computes percentage of usable frames based on quality thresholds | `quality_threshold=50.0` |
| `vfr_detection` | - | Variable Frame Rate (VFR) and jitter detection | `jitter_threshold_ms=2.0` |
| `vlm_judge` | - | Advanced semantic verification using VLM (e.g. LLaVA) | `model_name=llava-hf/llava-1.5-7b-hf`, `max_new_tokens=256`, +4 |
| `vtss` | `vtss`  -  Video Training Suitability Score (0-1) | Video Training Suitability Score (0-1, meta-metric) | `weights={'aesthetic': 0.15, 'technical': 0.15, 'motion': 0.1, 'temporal_consistency': 0.15, 'blur': 0.1, 'noise': 0.1, 'scene_stability': 0.1, 'resolution': 0.15}` |
| `wadiqam` | `wadiqam_score`  -  WaDIQaM-NR (higher=better) | WaDIQaM-NR weighted averaging deep image quality mapper | `subsample=8` |

## Safety & Content

| Module | Outputs | Description | Config |
|--------|---------|-------------|--------|
| `bias_detection` | `bias_score`  -  Representation imbalance indicator 0-1 | Demographic representation analysis (face count, age distribution) | `subsample=10`, `max_frames=30`, +1 |
| `deepfake_detection` | `deepfake_probability`  -  Synthetic/deepfake likelihood 0-1 | Synthetic media / deepfake likelihood estimation | `subsample=10`, `max_frames=60`, +1 |
| `harmful_content` | `harmful_content_score`  -  Violence/gore severity 0-1 | Violence, gore, and disturbing content detection | `subsample=10`, `max_frames=60`, +1 |
| `nsfw` | `nsfw_score`  -  0-1, likelihood of being NSFW | Detects NSFW (adult/violent) content using ViT | `model_name=Falconsai/nsfw_image_detection`, `threshold=0.5`, +1 |
| `watermark_classifier` | `ai_generated_probability`  -  AI-generated content likelihood 0-1; `watermark_probability`  -  0-1 | Classifies video for watermarks using a pretrained model or custom ResNet-50 weights | `model_weights_path=`, `hf_model=umm-maybe/AI-image-detector`, +1 |
| `watermark_robustness` | `watermark_strength`  -  Invisible watermark strength 0-1 | Invisible watermark detection and strength estimation | `subsample=15`, `max_frames=30` |

## Text & Semantic

| Module | Outputs | Description | Config |
|--------|---------|-------------|--------|
| `captioning` | `blip_bleu`; `auto_caption`  -  Generated caption | Generates captions using BLIP + computes BLEU score (EvalCrafter blip_bleu) | `model_name=Salesforce/blip-image-captioning-base`, `num_frames=5` |
| `clip_iqa` | `clip_iqa_score`  -  CLIP-IQA semantic quality (0-1, higher=better) | CLIP-based no-reference image quality assessment | `subsample=5`, `warning_threshold=0.4` |
| `compression_artifacts` | `compression_artifacts`  -  Artifact severity (0-100) | Detects compression artifacts (blocking, ringing, mosquito noise) | `subsample=3`, `warning_threshold=40.0` |
| `nemo_curator` | `nemo_quality_score`  -  Caption text quality (0-1); `nemo_quality_label`  -  Quality label (Low/Medium/High) | Caption text quality scoring (DeBERTa/FastText/heuristic) | `backend=auto`, `model_name=nvidia/quality-classifier-deberta`, +2 |
| `ocr_fidelity` | `ocr_fidelity`  -  OCR text accuracy vs caption (0-100, higher=better); `ocr_score`; `ocr_cer`  -  Character Error Rate (0-1, lower=better); `ocr_wer`  -  Word Error Rate (0-1, lower=better) | Checks whether text requested in the caption actually appears in video frames (EvalCrafter OCR) | `num_frames=8`, `lang=en` |
| `ram_tagging` | `ram_tags`  -  Comma-separated RAM auto-tags | RAM (Recognize Anything Model) auto-tagging for video frames | `model_name=xinyu1205/recognize-anything-plus-model`, `subsample=4`, +2 |
| `sd_reference` | `sd_score`  -  SD-reference similarity (0-1) | SD Score  -  CLIP similarity between video frames and SDXL-generated reference images | `clip_model=openai/clip-vit-base-patch32`, `sdxl_model=stabilityai/stable-diffusion-xl-base-1.0`, +4 |
| `semantic_alignment` | `clip_score`  -  Caption-image alignment | Checks alignment between video and caption (CLIP Score) | `model_name=openai/clip-vit-base-patch32`, `max_frames=32`, +1 |
| `semantic_segmentation_consistency` | `semantic_consistency`  -  Segmentation temporal IoU 0-1 (higher=better) | Temporal stability of semantic segmentation | `backend=auto`, `device=auto`, +4 |
| `semantic_selection` | - | Selects diverse samples based on VLM-extracted semantic traits | `num_to_select=10`, `uniqueness_weight=0.7`, +1 |
| `text_overlay` | `text_overlay_score`  -  Text overlay severity (0-1) | Text overlay / subtitle detection in video frames | `subsample=4`, `edge_threshold=0.15` |
| `tifa` | `tifa_score`  -  VQA faithfulness (0-1, higher=better) | TIFA text-to-image faithfulness via VQA question answering (ICCV 2023) | `vqa_model=dandelin/vilt-b32-finetuned-vqa`, `num_questions=8`, +1 |
| `video_text_matching` | `clip_score`  -  Caption-image alignment; `clip_temp` | ViCLIP / X-CLIP (Temporal alignment) or Frame-averaged CLIP | `use_xclip=False`, `model_name=openai/clip-vit-base-patch32`, +3 |
| `vqa_score` | - | VQAScore text-visual alignment via VQA probability (0-1, higher=better) | `model=clip-flant5-xxl`, `subsample=4` |

## Video: Generation & AI

| Module | Outputs | Description | Config |
|--------|---------|-------------|--------|
| `aigv_assessor` | `aigv_static`  -  AI video static quality; `aigv_temporal`  -  AI video temporal smoothness; `aigv_dynamic`  -  AI video dynamic degree; `aigv_alignment`  -  AI video text-video alignment | AI-generated video quality (AIGV-Assessor model, CLIP+heuristic, or OpenCV fallback) | `subsample=8`, `trust_remote_code=True`, +1 |
| `chronomagic` | `chronomagic_mt_score`  -  Metamorphic temporal (0-1, higher=better); `chronomagic_ch_score`  -  Chrono-hallucination (0-1, lower=fewer) | ChronoMagic-Bench MTScore + CHScore (CLIP / heuristic) | `subsample=16`, `hallucination_threshold=2.0` |
| `t2v_compbench` | - | T2V-CompBench compositional metrics (YOLO+Depth+CLIP / CLIP / heuristic) | `subsample=8`, `enable_attribute=True`, +6 |
| `t2v_score` | `t2v_score`  -  T2VScore alignment + quality; `t2v_alignment`  -  Text-video semantic alignment; `t2v_quality`  -  Video production quality | Text-to-Video alignment and quality scoring | `model_name=TIGER-Lab/T2VScore`, `use_clip_fallback=True`, +5 |
| `video_memorability` | `video_memorability`  -  Memorability prediction | Content memorability approximation (CLIP/DINOv2 feature statistics, not a trained predictor) | `subsample=5` |
| `video_reward` | `video_reward_score`  -  Human preference reward | VideoAlign human preference reward model (NeurIPS 2025) | `model_name=KlingTeam/VideoAlign-Reward`, `subsample=8`, +2 |
| `video_type_classifier` | `video_type`  -  Content type (real, animated, game, etc.); `video_type_confidence`  -  Classification confidence | CLIP zero-shot video content type classification | `subsample=4` |
| `videoscore` | `videoscore_visual`  -  VideoScore visual quality; `videoscore_temporal`  -  VideoScore temporal consistency; `videoscore_dynamic`  -  VideoScore dynamic degree; `videoscore_alignment`  -  VideoScore text-video alignment; `videoscore_factual`  -  VideoScore factual consistency | VideoScore 5-dimensional video quality assessment (1-4 scale) | `model_name=TIGER-Lab/VideoScore`, `num_frames=8`, +2 |

## Video: Motion & Temporal

| Module | Outputs | Description | Config |
|--------|---------|-------------|--------|
| `background_consistency` | - | Background consistency using CLIP (all pairwise frame similarity) | `model_name=openai/clip-vit-base-patch32`, `max_frames=16`, +1 |
| `camera_jitter` | `camera_jitter_score`  -  Camera stability (0-1, 1=stable) | Camera jitter/shake detection (0-1, 1=stable) | `subsample=16` |
| `camera_motion` | - | Analyzes camera motion stability (VMBench) using Homography | - |
| `clip_temporal` | `clip_temp`; `face_consistency` | CLIP temporal consistency + face/identity consistency (EvalCrafter clip_temp & face_consistency) | `model_name=openai/clip-vit-base-patch32`, `max_frames=32`, +2 |
| `flicker_detection` | `flicker_score`  -  Flicker severity 0-100 (lower=better) | Detects temporal luminance flicker | `max_frames=600`, `warning_threshold=30.0` |
| `flow_coherence` | `flow_coherence`  -  Bidirectional optical flow consistency (0-1) | Bidirectional optical flow consistency (0-1, higher=coherent) | `subsample=8` |
| `jump_cut` | `jump_cut_score`  -  Jump cut absence (0-1, 1=no cuts) | Jump cut / abrupt transition detection (0-1, 1=no cuts) | `threshold=40.0` |
| `kandinsky_motion` | - | Video/Camera Motion Analysis using Kandinsky Video Tools (VideoMAE-V2) | `models_dir=models` |
| `motion` | `motion_score`  -  Scene motion intensity | Analyzes motion dynamics (optical flow, flickering) | `sample_rate=5`, `low_motion_threshold=0.5`, +1 |
| `motion_amplitude` | `motion_ac_score` | Motion amplitude classification vs caption (motion_ac_score via RAFT) | `amplitude_threshold=5.0`, `max_frames=150`, +1 |
| `motion_smoothness` | `motion_smoothness`  -  Motion smoothness (0-1, higher=better) | Motion smoothness via RIFE VFI reconstruction error (VBench) | `vfi_error_threshold=0.08`, `max_frames=64` |
| `object_permanence` | - | Object tracking consistency (ID switches, disappearances) | `backend=auto`, `subsample=2`, +3 |
| `playback_speed` | `playback_speed_score`  -  Normal speed (1.0=normal) | Playback speed normality detection (1.0=normal) | `subsample=16` |
| `ptlflow_motion` | `ptlflow_motion_score`  -  ptlflow optical flow magnitude | ptlflow optical flow motion scoring (dpflow model) | `model_name=dpflow`, `ckpt_path=things`, +1 |
| `raft_motion` | `raft_motion_score`  -  RAFT optical flow magnitude | RAFT optical flow motion scoring (torchvision) | `subsample=8` |
| `scene_detection` | `avg_scene_duration`  -  Average scene duration in seconds | Scene stability metric  -  penalises rapid cuts (0-1, higher=more stable) | `threshold=0.5` |
| `stabilized_motion` | `motion_score`  -  Scene motion intensity; `camera_motion_score`  -  Camera motion intensity | Calculates motion scores with camera stabilization (ORB+Homography) | `step=2`, `threshold_px=0.5`, +3 |
| `subject_consistency` | `subject_consistency`  -  Subject identity consistency (0-1, higher=better) | Subject consistency using DINOv2-base (all pairwise frame similarity) | `model_name=facebook/dinov2-base`, `max_frames=16`, +1 |
| `temporal_flickering` | `warping_error` | Warping Error using RAFT optical flow with occlusion masking | `warning_threshold=0.02`, `max_frames=300` |
| `temporal_style` | - | Analyzes temporal style (Slow Motion, Timelapse, Speed) | - |

## Video: Quality Assessment

| Module | Outputs | Description | Config |
|--------|---------|-------------|--------|
| `c3dvqa` | `c3dvqa_score`  -  C3DVQA 3D CNN spatiotemporal FR | 3D CNN spatiotemporal video quality assessment | `clip_length=16`, `subsample=4` |
| `cover` | `cover_technical`  -  COVER technical branch; `cover_aesthetic`  -  COVER aesthetic branch; `cover_semantic`  -  COVER semantic branch; `cover_score`  -  COVER overall (higher=better) | COVER 3-branch comprehensive video quality (semantic + aesthetic + technical) | `subsample=8`, `quality_threshold=30.0` |
| `dover` | `dover_score`  -  DOVER overall (higher=better); `dover_aesthetic`  -  DOVER aesthetic quality; `dover_technical`  -  DOVER technical quality | DOVER disentangled technical + aesthetic VQA (ICCV 2023) | `warning_threshold=0.4`, `weights_path=None`, +1 |
| `fast_vqa` | `fast_vqa_score`  -  0-100 | Deep Learning Video Quality Assessment (FAST-VQA) | `model_type=FasterVQA` |
| `finevq` | `finevq_score`  -  FineVQ fine-grained UGC VQA (CVPR 2025) | Fine-grained video quality (FineVQ model, TOPIQ+handcrafted, or heuristic fallback) | `subsample=8`, `trust_remote_code=True`, +2 |
| `kvq` | `kvq_score`  -  KVQ saliency-guided VQA (CVPR 2025) | Saliency-guided video quality (KVQ model, TOPIQ+saliency, or heuristic fallback) | `subsample=8`, `trust_remote_code=True`, +1 |
| `mdtvsfa` | `mdtvsfa_score`  -  MDTVSFA fragment-based VQA (higher=better) | Multi-Dimensional fragment-based VQA | `subsample=5` |
| `rqvqa` | `rqvqa_score`  -  RQ-VQA rich quality-aware (CVPR 2024 winner) | Multi-attribute video quality (RQ-VQA model, CLIP-IQA+, or heuristic fallback) | `subsample=8`, `trust_remote_code=True`, +2 |
| `tlvqm` | `tlvqm_score`  -  TLVQM two-level video quality | Two-level video quality model (CNN-TLVQM or handcrafted fallback) | `subsample=8` |
| `videval` | `videval_score`  -  VIDEVAL 60-feature fusion NR-VQA | Feature-fusion NR-VQA (VIDEVAL-style SVR or heuristic linear mapping) | `subsample=8` |
