# Ayase Metrics Reference

208 modules.

## Audio

| Module | Description | Config |
|--------|-------------|--------|
| `audio_estoi` | ESTOI speech intelligibility (full-reference) | target_sr=10000, warning_threshold=0.5 |
| `audio_lpdist` | Log-Power Spectral Distance (full-reference audio) | target_sr=16000, n_mels=80, warning_threshold=4.0 |
| `audio_mcd` | Mel Cepstral Distortion for TTS/VC quality (full-reference) | target_sr=16000, n_mfcc=13, warning_threshold=8.0 |
| `audio_pesq` | PESQ speech quality (full-reference, ITU-T P.862) | target_sr=16000, warning_threshold=3.0 |
| `audio_si_sdr` | Scale-Invariant SDR for audio quality (full-reference) | target_sr=16000, warning_threshold=0.0 |
| `audio_text_alignment` | Multimodal alignment check (Audio-Text) using CLAP | alignment_threshold=0.2, model_name=laion/clap-htsat-fused |
| `audio_utmos` | UTMOS no-reference MOS prediction for speech quality | target_sr=16000, warning_threshold=3.0 |
| `audio_visual_sync` | Audio-video synchronisation offset detection | max_frames=600, warning_threshold_ms=80.0 |
| `dnsmos` | DNSMOS non-intrusive audio quality (Microsoft, 1-5 MOS) | - |
| `p1203` | ITU-T P.1203 streaming QoE estimation (1-5 MOS) | display_size=phone |
| `visqol` | ViSQOL audio quality MOS (Google, 1-5, higher=better) | mode=audio |

## Audio-Visual

| Module | Description | Config |
|--------|-------------|--------|
| `av_sync` | Audio-video synchronisation offset detection | max_frames=600, warning_threshold_ms=80.0 |

## Depth

| Module | Description | Config |
|--------|-------------|--------|
| `depth_anything` | Depth Anything V2 monocular depth estimation and consistency | model_name=depth-anything/Depth-Anything-V2-Small-hf, subsample=8 |
| `depth_consistency` | Monocular depth temporal consistency | model_type=MiDaS_small, device=auto, subsample=3 (+2) |
| `depth_map_quality` | Monocular depth map quality (sharpness, completeness, edge alignment) | model_type=MiDaS_small, device=auto, subsample=10 (+1) |

## Face Quality

| Module | Description | Config |
|--------|-------------|--------|
| `face_fidelity` | Face detection and per-face quality assessment | backend=haar, subsample=5, max_frames=60 (+3) |
| `face_iqa` | Face-specific IQA via TOPIQ-face (GFIQA-trained, higher=better) | subsample=8 |
| `face_landmark_quality` | Facial landmark jitter, expression smoothness, identity consistency | subsample=2, max_frames=300, jitter_warning=30.0 |

## Full-Reference (image/video)

| Module | Description | Config |
|--------|-------------|--------|
| `ahiq` | Attention-based Hybrid IQA full-reference (higher=better) | subsample=8 |
| `butteraugli` | Butteraugli perceptual distance (Google/JPEG XL, lower=better) | subsample=5, warning_threshold=2.0 |
| `cgvqm` | CGVQM gaming/rendering quality metric (Intel, higher=better) | subsample=5 |
| `ciede2000` | CIEDE2000 perceptual color difference (lower=better) | subsample=5 |
| `ckdn` | CKDN knowledge distillation FR image quality | subsample=4 |
| `cw_ssim` | Complex Wavelet SSIM full-reference metric (0-1, higher=better) | subsample=8 |
| `deepwsd` | DeepWSD Wasserstein distance FR image quality | subsample=4 |
| `delta_ictcp` | Delta ICtCp HDR perceptual color difference (lower=better) | subsample=5 |
| `dmm` | DMM detail model metric full-reference (higher=better) | subsample=8 |
| `dreamsim` | DreamSim foundation model perceptual similarity (CLIP+DINO ensemble) | subsample=8, model_type=ensemble |
| `dreamsim_metric` | DreamSim foundation model perceptual similarity (CLIP+DINO ensemble) | subsample=8, model_type=ensemble |
| `flip` | NVIDIA FLIP perceptual difference (0-1, lower=better) | subsample=5, warning_threshold=0.3 |
| `flip_metric` | NVIDIA FLIP perceptual difference (0-1, lower=better) | subsample=5, warning_threshold=0.3 |
| `flolpips` | Flow-compensated perceptual distance (RAFT+LPIPS, Farneback+LPIPS, or MSE fallback) | subsample=8 |
| `funque` | Fused quality evaluator (FUNQUE package, handcrafted FR, or NR fallback) | subsample=8 |
| `hdr_vdp` | HDR-VDP visual difference predictor (higher=better) | subsample=5 |
| `hdr_vqm` | HDR-aware video quality (PU21+wavelet FR or gamma heuristic fallback) | subsample=8 |
| `mad` | Most Apparent Distortion full-reference metric (lower=better) | subsample=8 |
| `mad_metric` | Most Apparent Distortion full-reference metric (lower=better) | subsample=8 |
| `movie` | Video quality via spatiotemporal Gabor decomposition (FR or NR fallback) | subsample=8 |
| `ms_ssim` | Multi-Scale SSIM perceptual similarity metric (full-reference) | scales=5, weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333], subsample=1 (+2) |
| `nlpd` | Normalized Laplacian Pyramid Distance full-reference (lower=better) | subsample=8 |
| `nlpd_metric` | Normalized Laplacian Pyramid Distance full-reference (lower=better) | subsample=8 |
| `pieapp` | PieAPP full-reference perceptual error via pairwise preference (lower=better) | subsample=8 |
| `psnr_hvs` | PSNR-HVS + PSNR-HVS-M perceptually weighted PSNR (dB, higher=better) | subsample=5 |
| `ssimc` | SSIM-C complex wavelet structural similarity FR (higher=better) | subsample=8 |
| `st_greed` | Spatial-temporal entropic quality (FR entropic difference or NR heuristic fallback) | subsample=16 |
| `st_lpips` | Spatiotemporal perceptual video quality (ST-LPIPS model, LPIPS, or heuristic fallback) | subsample=8 |
| `topiq_fr` | TOPIQ full-reference top-down semantics-to-distortion IQA (higher=better) | subsample=8 |
| `wadiqam_fr` | WaDIQaM full-reference deep quality metric (higher=better) | subsample=8 |

## HDR

| Module | Description | Config |
|--------|-------------|--------|
| `hdr_metadata` | MaxFALL + MaxCLL HDR static metadata analysis | subsample=3, peak_nits=10000.0 |
| `hdr_sdr_vqa` | HDR/SDR-aware video quality assessment | subsample=5 |
| `pu_metrics` | PU-PSNR + PU-SSIM for HDR content (perceptually uniform) | subsample=5, assume_nits_range=10000.0 |

## Image-to-Video Reference

| Module | Description | Config |
|--------|-------------|--------|
| `i2v_similarity` | Image-to-Video reference similarity using CLIP, DINOv2, and LPIPS (sliding window) | window_size=16, stride=8, max_frames=256 (+6) |

## Image: No-Reference Quality

| Module | Description | Config |
|--------|-------------|--------|
| `4k_vqa` | Memory-efficient quality assessment for 4K+ videos | tile_size=512, subsample=10 |
| `action_recognition` | Recognizes human actions (VideoMAE / UMT) - Supports Heavy Models | model_name=MCG-NJU/videomae-large-finetuned-kinetics, caption_matching=False, matching_mode=weighted (+2) |
| `advanced_flow` | RAFT optical flow: flow_score (all consecutive pairs) | use_large_model=True, max_frames=150 |
| `aesthetic` | Estimates aesthetic quality using Aesthetic Predictor V2.5 | num_frames=5, trust_remote_code=True, model_revision=None |
| `aesthetic_scoring` | Calculates aesthetic score (1-10) using LAION-Aesthetics MLP | models_dir=models |
| `afine` | A-FINE adaptive fidelity-naturalness IQA (CVPR 2025) | subsample=4 |
| `arniqa` | ARNIQA no-reference image quality assessment | subsample=8 |
| `audio` | Validates audio stream quality and presence | require_audio=False, min_sample_rate=44100, min_bit_rate=128000 (+3) |
| `background_diversity` | Checks background complexity (entropy) to detect concept bleeding | min_entropy_threshold=3.0, use_rembg=True |
| `basic` | Comprehensive technical quality assessment (blur, noise, artifacts, contrast) | threshold=40.0, blur_threshold=100.0, noise_threshold=50.0 |
| `basic_quality` | Comprehensive technical quality assessment (blur, noise, artifacts, contrast) | threshold=40.0, blur_threshold=100.0, noise_threshold=50.0 |
| `bd_rate` | BD-Rate codec comparison (dataset-level, negative%=better) | quality_metric=psnr |
| `brisque` | BRISQUE no-reference image quality (lower=better) | subsample=3, warning_threshold=50.0 |
| `cambi` | CAMBI banding/contouring detector (Netflix, 0-24, lower=better) | warning_threshold=5.0 |
| `celebrity_id` | Face identity verification using DeepFace (EvalCrafter celebrity_id_score) | reference_dir=, num_frames=8, consistency_threshold=0.4 (+1) |
| `cnniqa` | CNNIQA blind CNN-based image quality assessment | subsample=4 |
| `codec_compatibility` | Validates codec, pixel format, and container for ML dataloader compatibility | min_bitrate_kbps=500, min_bpp=0.02 |
| `codec_specific_quality` | Codec-level efficiency, GOP quality, and artifact detection | max_frames=100, subsample=10, warning_efficiency=30.0 (+1) |
| `color_consistency` | Verifies color attributes in prompt vs video content | - |
| `commonsense` | Common sense adherence (VLM / ViLT VQA / heuristic) | model_name=dandelin/vilt-b32-finetuned-vqa, vlm_model=llava-hf/llava-1.5-7b-hf |
| `compare2score` | Compare2Score comparison-based NR image quality | subsample=4 |
| `contrique` | Contrastive no-reference IQA | subsample=5 |
| `cpbd` | Cumulative Probability of Blur Detection (Perceptual Blur) | threshold_cpbd=0.65, threshold_heuristic=10.0 |
| `creativity` | Artistic novelty assessment (VLM / CLIP / heuristic) | vlm_model=llava-hf/llava-1.5-7b-hf |
| `dbcnn` | DBCNN deep bilinear CNN for no-reference IQA | subsample=8 |
| `decoder_stress` | Random access decoder stress test | num_probes=5, check_integrity=True |
| `dedup` | Detects duplicates using Perceptual Hashing (pHash) | - |
| `deduplication` | Detects duplicates using Perceptual Hashing (pHash) | - |
| `dists` | Deep Image Structure and Texture Similarity (full-reference) | subsample=5, warning_threshold=0.3, device=auto |
| `diversity` | Flags redundant samples using embedding similarity (Deduplication) | similarity_threshold=0.95, priority_metric=aesthetic_score |
| `diversity_selection` | Flags redundant samples using embedding similarity (Deduplication) | similarity_threshold=0.95, priority_metric=aesthetic_score |
| `dynamics_controllability` | Assesses motion controllability based on text-motion alignment | subsample=16 |
| `dynamics_range` | Measures extent of motion and content variation (DEVIL protocol) | scene_change_threshold=30.0 |
| `embedding` | Calculates X-CLIP embeddings for similarity search | model_name=microsoft/xclip-base-patch32, num_frames=8 |
| `exposure` | Checks for overexposure, underexposure, and low contrast using histograms | overexposure_threshold=0.3, underexposure_threshold=0.3, contrast_threshold=30.0 |
| `human_fidelity` | Human body/hand/face fidelity (DWPose / MediaPipe / heuristic) | - |
| `hyperiqa` | HyperIQA adaptive hypernetwork NR image quality | subsample=4 |
| `identity_loss` | Face identity preservation metric (cosine distance/similarity vs reference) | model_name=buffalo_l, subsample=8, warning_threshold=0.5 |
| `ilniqe` | IL-NIQE integrated local no-reference quality (lower=better) | subsample=3, warning_threshold=50.0 |
| `imaging_quality` | Assesses technical quality (Noise, Blockiness) - Proxy for MUSIQ/DOVER | noise_threshold=20.0 |
| `inception_score` | Inception Score (IS) using InceptionV3 — EvalCrafter quality metric | num_frames=16, splits=1 |
| `jedi` | JEDi distribution metric (V-JEPA + MMD, ICLR 2025) | num_frames=16, batch_size=8, trust_remote_code=True (+1) |
| `jedi_metric` | JEDi distribution metric (V-JEPA + MMD, ICLR 2025) | num_frames=16, batch_size=8, trust_remote_code=True (+1) |
| `judder_stutter` | Detects judder (uneven cadence) and stutter (duplicate frames) | max_frames=600, duplicate_threshold=1.0, warning_threshold=20.0 |
| `knowledge_graph` | Generates a conceptual knowledge graph of the video dataset | output_file=knowledge_graph.json, min_confidence=0.5, include_similarity_edges=False (+3) |
| `laion_aesthetic` | LAION Aesthetics V2 predictor (0-10, industry standard) | subsample=4 |
| `letterbox` | Border/letterbox detection (0-1, 0=no borders) | threshold=16, subsample=4 |
| `liqe` | LIQE lightweight no-reference IQA | subsample=5, warning_threshold=2.5 |
| `llm_advisor` | Rule-based improvement recommendations derived from quality metrics (no LLM used) | severity_level=INFO |
| `llm_descriptive_qa` | LMM-based interpretable quality assessment with explanations | model_name=llava-hf/llava-v1.6-mistral-7b-hf, use_openai=False, openai_api_key=None (+2) |
| `maclip` | MACLIP multi-attribute CLIP no-reference quality (higher=better) | subsample=3 |
| `maniqa` | MANIQA multi-dimension attention no-reference IQA | subsample=8 |
| `metadata` | Checks video/image metadata (resolution, FPS, duration, integrity) | min_resolution=720, min_fps=15, min_duration=2.0 (+3) |
| `multi_view_consistency` | Geometric multi-view consistency via epipolar analysis | subsample=5, max_pairs=30, min_matches=20 |
| `multiple_objects` | Verifies object count matches caption (VBench multiple_objects dimension) | tolerance=1 |
| `musiq` | Multi-Scale Image Quality Transformer (no-reference) | variant=musiq, subsample=5, warning_threshold=40.0 |
| `naturalness` | Measures naturalness of content (natural vs synthetic) | use_pyiqa=True, subsample=2, warning_threshold=0.4 |
| `nima` | NIMA aesthetic and technical image quality (1-10 scale) | subsample=8 |
| `niqe` | Natural Image Quality Evaluator (no-reference) | subsample=2, warning_threshold=7.0 |
| `nrqm` | NRQM no-reference quality metric for super-resolution (higher=better) | subsample=3 |
| `object_detection` | Detects objects (GRiT / YOLOv8) - Supports Heavy Models | model_name=yolov8n.pt, use_yolo_world=False, use_grit=False |
| `paq2piq` | PaQ-2-PiQ patch-to-picture NR quality (CVPR 2020) | subsample=4 |
| `paranoid_decoder` | Deep bitstream validation using FFmpeg (Paranoid Mode) | timeout=60, strict_mode=True |
| `perceptual_fr` | FSIM + GMSD + VSI full-reference perceptual metrics | subsample=5, device=auto |
| `physics` | Physics plausibility via trajectory analysis (CoTracker / LK / heuristic) | subsample=16, accel_threshold=50.0 |
| `pi` | Perceptual Index (PIRM challenge metric, lower=better) | subsample=3 |
| `pi_metric` | Perceptual Index (PIRM challenge metric, lower=better) | subsample=3 |
| `piqe` | PIQE perception-based no-reference quality (lower=better) | subsample=3, warning_threshold=50.0 |
| `production_quality` | Professional production quality (colour, exposure, focus, banding) | max_frames=150 |
| `promptiqa` | Prompt-guided NR-IQA (PromptIQA via pyiqa, TOPIQ-NR, or CLIP-IQA+ fallback) | subsample=4 |
| `q_align` | Q-Align unified quality + aesthetic assessment (ICML 2024) | model_name=q-future/one-align, dtype=float16, device=auto (+5) |
| `qcn` | Blind IQA (QCN via pyiqa, or HyperIQA fallback) | subsample=4 |
| `qualiclip` | QualiCLIP opinion-unaware CLIP-based no-reference IQA | subsample=8 |
| `resolution_bucketing` | Validates resolution/aspect-ratio fit for training buckets | max_crop_ratio=0.15, max_scale_factor=2.0, divisibility=8 (+2) |
| `scene` | Detects scene cuts and shots using PySceneDetect | threshold=27.0, min_scene_len=15, warn_on_high_shot_count=True (+1) |
| `scene_complexity` | Spatial and temporal scene complexity analysis | subsample=2, spatial_weight=0.5, temporal_weight=0.5 |
| `scene_tagging` | Tags scene context (Proxy for Tag2Text/RAM using CLIP) | models_dir=models |
| `spatial_relationship` | Verifies spatial relations (left/right/top/bottom) in prompt vs detections | - |
| `spectral` | Analyzes spectral complexity (Effective Rank) of video features (DINOv2) | model_type=dinov2_vits14, sample_rate=8, min_rank_ratio=0.05 (+1) |
| `spectral_complexity` | Analyzes spectral complexity (Effective Rank) of video features (DINOv2) | model_type=dinov2_vits14, sample_rate=8, min_rank_ratio=0.05 (+1) |
| `spectral_upscaling` | Detection of upscaled/fake high-resolution content | energy_threshold=0.05, sample_rate=20 |
| `stereoscopic_quality` | Stereo 3D comfort and quality assessment | stereo_format=auto, subsample=10, max_frames=30 (+2) |
| `structural` | Checks structural integrity (scene cuts, black bars) | detect_cuts=True, detect_black_bars=True |
| `style_consistency` | Appearance Style verification (Gram Matrix Consistency) | - |
| `text` | Detects text/watermarks using OCR (PaddleOCR / Tesseract) | use_paddle=True, max_text_area=0.05 |
| `ti_si` | ITU-T P.910 Temporal & Spatial Information | max_frames=300 |
| `tonal_dynamic_range` | Luminance histogram tonal range (0-100) | low_percentile=1, high_percentile=99, subsample=8 |
| `topiq` | TOPIQ transformer-based no-reference IQA | variant=topiq_nr, subsample=5, warning_threshold=0.4 |
| `trajan` | Motion consistency via point tracking (CoTracker or Lucas-Kanade fallback) | num_frames=16, num_points=256 |
| `tres` | TReS transformer-based NR image quality (WACV 2022) | subsample=4 |
| `unique` | UNIQUE unified NR image quality (TIP 2021) | subsample=4 |
| `unique_iqa` | UNIQUE unified NR image quality (TIP 2021) | subsample=4 |
| `usability_rate` | Computes percentage of usable frames based on quality thresholds | quality_threshold=50.0 |
| `vfr_detection` | Variable Frame Rate (VFR) and jitter detection | jitter_threshold_ms=2.0 |
| `vlm_judge` | Advanced semantic verification using VLM (e.g. LLaVA) | model_name=llava-hf/llava-1.5-7b-hf, max_new_tokens=256, mode=verify (+3) |
| `vtss` | Video Training Suitability Score (0-1, meta-metric) | weights={'aesthetic': 0.15, 'technical': 0.15, 'motion': 0.1, 'temporal_consistency': 0.15, 'blur': 0.1, 'noise': 0.1, 'scene_stability': 0.1, 'resolution': 0.15} |
| `wadiqam` | WaDIQaM-NR weighted averaging deep image quality mapper | subsample=8 |

## Safety & Content

| Module | Description | Config |
|--------|-------------|--------|
| `bias_detection` | Demographic representation analysis (face count, age distribution) | subsample=10, max_frames=30, warning_threshold=0.7 |
| `deepfake_detection` | Synthetic media / deepfake likelihood estimation | subsample=10, max_frames=60, warning_threshold=0.6 |
| `harmful_content` | Violence, gore, and disturbing content detection | subsample=10, max_frames=60, warning_threshold=0.4 |
| `nsfw` | Detects NSFW (adult/violent) content using ViT | model_name=Falconsai/nsfw_image_detection, threshold=0.5, num_frames=8 |
| `watermark_classifier` | Classifies video for watermarks using a pretrained model or custom ResNet-50 weights | model_weights_path=, hf_model=umm-maybe/AI-image-detector, threshold=0.5 |
| `watermark_robustness` | Invisible watermark detection and strength estimation | subsample=15, max_frames=30 |

## Text & Semantic Alignment

| Module | Description | Config |
|--------|-------------|--------|
| `captioning` | Generates captions using BLIP + computes BLEU score (EvalCrafter blip_bleu) | model_name=Salesforce/blip-image-captioning-base, num_frames=5 |
| `clip_iqa` | CLIP-based no-reference image quality assessment | subsample=5, warning_threshold=0.4 |
| `compression_artifacts` | Detects compression artifacts (blocking, ringing, mosquito noise) | subsample=3, warning_threshold=40.0 |
| `nemo_curator` | Caption text quality scoring (DeBERTa/FastText/heuristic) | backend=auto, model_name=nvidia/quality-classifier-deberta, min_length=10 (+1) |
| `ocr_fidelity` | Checks whether text requested in the caption actually appears in video frames (EvalCrafter OCR) | num_frames=8, lang=en |
| `ram_tagging` | RAM (Recognize Anything Model) auto-tagging for video frames | model_name=xinyu1205/recognize-anything-plus-model, subsample=4, trust_remote_code=True (+1) |
| `sd_reference` | SD Score — CLIP similarity between video frames and SDXL-generated reference images | clip_model=openai/clip-vit-base-patch32, sdxl_model=stabilityai/stable-diffusion-xl-base-1.0, num_sd_images=5 (+3) |
| `semantic_alignment` | Checks alignment between video and caption (CLIP Score) | model_name=openai/clip-vit-base-patch32, max_frames=32, warning_threshold=0.2 |
| `semantic_segmentation_consistency` | Temporal stability of semantic segmentation | backend=auto, device=auto, subsample=3 (+3) |
| `semantic_selection` | Selects diverse samples based on VLM-extracted semantic traits | num_to_select=10, uniqueness_weight=0.7, quality_weight=0.3 |
| `text_detection` | Detects text/watermarks using OCR (PaddleOCR / Tesseract) | use_paddle=True, max_text_area=0.05 |
| `text_overlay` | Text overlay / subtitle detection in video frames | subsample=4, edge_threshold=0.15 |
| `tifa` | TIFA text-to-image faithfulness via VQA question answering (ICCV 2023) | vqa_model=dandelin/vilt-b32-finetuned-vqa, num_questions=8, subsample=4 |
| `video_text_matching` | ViCLIP / X-CLIP (Temporal alignment) or Frame-averaged CLIP | use_xclip=False, model_name=openai/clip-vit-base-patch32, xclip_model_name=microsoft/xclip-base-patch32 (+2) |
| `vqa_score` | VQAScore text-visual alignment via VQA probability (0-1, higher=better) | model=clip-flant5-xxl, subsample=4 |

## Video: Generation & AI

| Module | Description | Config |
|--------|-------------|--------|
| `aigv_assessor` | AI-generated video quality (AIGV-Assessor model, CLIP+heuristic, or OpenCV fallback) | subsample=8, trust_remote_code=True, model_revision=None |
| `chronomagic` | ChronoMagic-Bench MTScore + CHScore (CLIP / heuristic) | subsample=16, hallucination_threshold=2.0 |
| `t2v_compbench` | T2V-CompBench compositional metrics (YOLO+Depth+CLIP / CLIP / heuristic) | subsample=8, enable_attribute=True, enable_object_rel=True (+5) |
| `t2v_score` | Text-to-Video alignment and quality scoring | model_name=TIGER-Lab/T2VScore, use_clip_fallback=True, num_frames=8 (+4) |
| `video_memorability` | Content memorability approximation (CLIP/DINOv2 feature statistics, not a trained predictor) | subsample=5 |
| `video_reward` | VideoAlign human preference reward model (NeurIPS 2025) | model_name=KlingTeam/VideoAlign-Reward, subsample=8, trust_remote_code=True (+1) |
| `video_type_classifier` | CLIP zero-shot video content type classification | subsample=4 |
| `videoscore` | VideoScore 5-dimensional video quality assessment (1-4 scale) | model_name=TIGER-Lab/VideoScore, num_frames=8, trust_remote_code=True (+1) |

## Video: Motion & Temporal

| Module | Description | Config |
|--------|-------------|--------|
| `background_consistency` | Background consistency using CLIP (all pairwise frame similarity) | model_name=openai/clip-vit-base-patch32, max_frames=16, warning_threshold=0.5 |
| `camera_jitter` | Camera jitter/shake detection (0-1, 1=stable) | subsample=16 |
| `camera_motion` | Analyzes camera motion stability (VMBench) using Homography | - |
| `clip_temporal` | CLIP temporal consistency + face/identity consistency (EvalCrafter clip_temp & face_consistency) | model_name=openai/clip-vit-base-patch32, max_frames=32, temp_threshold=0.9 (+1) |
| `flicker_detection` | Detects temporal luminance flicker | max_frames=600, warning_threshold=30.0 |
| `flow_coherence` | Bidirectional optical flow consistency (0-1, higher=coherent) | subsample=8 |
| `jump_cut` | Jump cut / abrupt transition detection (0-1, 1=no cuts) | threshold=40.0 |
| `kandinsky_motion` | Video/Camera Motion Analysis using Kandinsky Video Tools (VideoMAE-V2) | models_dir=models |
| `motion` | Analyzes motion dynamics (optical flow, flickering) | sample_rate=5, low_motion_threshold=0.5, high_motion_threshold=20.0 |
| `motion_amplitude` | Motion amplitude classification vs caption (motion_ac_score via RAFT) | amplitude_threshold=5.0, max_frames=150, scoring_mode=binary |
| `motion_smoothness` | Motion smoothness via RIFE VFI reconstruction error (VBench) | vfi_error_threshold=0.08, max_frames=64 |
| `object_permanence` | Object tracking consistency (ID switches, disappearances) | backend=auto, subsample=2, max_frames=300 (+2) |
| `playback_speed` | Playback speed normality detection (1.0=normal) | subsample=16 |
| `ptlflow_motion` | ptlflow optical flow motion scoring (dpflow model) | model_name=dpflow, ckpt_path=things, subsample=8 |
| `raft_motion` | RAFT optical flow motion scoring (torchvision) | subsample=8 |
| `scene_detection` | Scene stability metric — penalises rapid cuts (0-1, higher=more stable) | threshold=0.5 |
| `stabilized_motion` | Calculates motion scores with camera stabilization (ORB+Homography) | step=2, threshold_px=0.5, stabilize=True (+2) |
| `subject_consistency` | Subject consistency using DINOv2-base (all pairwise frame similarity) | model_name=facebook/dinov2-base, max_frames=16, warning_threshold=0.6 |
| `temporal_flickering` | Warping Error using RAFT optical flow with occlusion masking | warning_threshold=0.02, max_frames=300 |
| `temporal_style` | Analyzes temporal style (Slow Motion, Timelapse, Speed) | - |

## Video: Quality Assessment

| Module | Description | Config |
|--------|-------------|--------|
| `c3dvqa` | 3D CNN spatiotemporal video quality assessment | clip_length=16, subsample=4 |
| `cover` | COVER 3-branch comprehensive video quality (semantic + aesthetic + technical) | subsample=8, quality_threshold=30.0 |
| `dover` | DOVER disentangled technical + aesthetic VQA (ICCV 2023) | warning_threshold=0.4, weights_path=None, preferred_backend=None |
| `fast_vqa` | Deep Learning Video Quality Assessment (FAST-VQA) | model_type=FasterVQA |
| `finevq` | Fine-grained video quality (FineVQ model, TOPIQ+handcrafted, or heuristic fallback) | subsample=8, trust_remote_code=True, model_revision=None (+1) |
| `kvq` | Saliency-guided video quality (KVQ model, TOPIQ+saliency, or heuristic fallback) | subsample=8, trust_remote_code=True, model_revision=None |
| `mdtvsfa` | Multi-Dimensional fragment-based VQA | subsample=5 |
| `rqvqa` | Multi-attribute video quality (RQ-VQA model, CLIP-IQA+, or heuristic fallback) | subsample=8, trust_remote_code=True, model_revision=None (+1) |
| `tlvqm` | Two-level video quality model (CNN-TLVQM or handcrafted fallback) | subsample=8 |
| `videval` | Feature-fusion NR-VQA (VIDEVAL-style SVR or heuristic linear mapping) | subsample=8 |

