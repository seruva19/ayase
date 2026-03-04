# Ayase

Modular media quality metrics toolkit.

## Overview

- 231 quality metrics across visual, temporal, audio, perceptual, and safety categories.
- Modular pipeline - modules compute raw values, downstream apps decide what to do with them.
- CLI and Python API.
- Profile-based pipeline configuration.

## Installation

Core (no ML models, metadata/structural checks only):

```bash
pip install ayase
```

With ML-based quality metrics:

```bash
pip install ayase[ml]           # Everything
pip install ayase[v-perceptual] # CLIP, LPIPS, open-clip, timm
pip install ayase[v-iqa]        # PyIQA, BRISQUE, NIQE, scikit-video
pip install ayase[v-motion]     # RAFT optical flow, decord
pip install ayase[v-ocr]        # PaddleOCR text recognition
pip install ayase[v-face]       # MediaPipe face detection
pip install ayase[v-audio]      # librosa audio analysis
```

Development:

```bash
pip install ayase[dev]          # pytest, black, ruff, mypy
pip install ayase[all]          # Everything including dev + TUI
```

See [MODELS.md](MODELS.md) for the complete inventory of all pretrained weights used by every module. Models are downloaded and cached automatically on first use via HuggingFace Hub, Torch Hub, and PyIQA.

## Metrics

**Input key:** `image/video` = either, `+ref` = needs `reference_path`, `+caption` = needs caption text, `audio` = audio stream, `batch` = dataset-level.

| # | Metric | Module | Input | Description |
|---:|--------|--------|-------|-------------|
| 1 | `blur_score` | `basic_quality` | image/video | Laplacian variance |
| 2 | `compression_score` | `basic_quality` | image/video | Compression quality (0-100) |
| 3 | `aesthetic_score` | `aesthetic` | image/video | 0-10, from aesthetic predictor |
| 4 | `clip_score` | `semantic_alignment` | image/video +caption | Caption-image alignment |
| 5 | `brightness` | `basic_quality` | image/video | Mean brightness (0-255) |
| 6 | `contrast` | `basic_quality` | image/video | Contrast level (0-100) |
| 7 | `saturation` | `basic_quality` | image/video | Color saturation (0-100) |
| 8 | `fast_vqa_score` | `fast_vqa` | image/video | 0-100 |
| 9 | `motion_score` | `motion` | image/video | Scene motion intensity |
| 10 | `camera_motion_score` | `stabilized_motion` | image/video | Camera motion intensity |
| 11 | `temporal_consistency` | `clip_temporal` | image/video | Frame consistency |
| 12 | `technical_score` | `basic_quality` | image/video | Composite technical score |
| 13 | `noise_score` | `basic_quality` | image/video | Noise level (0-100, lower=cleaner) |
| 14 | `artifacts_score` | `basic_quality` | image/video | Artifact severity (0-100, lower=better) |
| 15 | `watermark_probability` | `watermark_classifier` | image/video | 0-1 |
| 16 | `ocr_area_ratio` | `text_detection` | image | 0-1 |
| 17 | `face_count` | `face_fidelity` | image/video | Integer face count |
| 18 | `nsfw_score` | `nsfw` | image | 0-1, likelihood of being NSFW |
| 19 | `audio_quality_score` | `audio` | audio | 0-100 |
| 20 | `perceptual_hash` | `dedup` | image/video | dHash or similar |
| 21 | `depth_score` | `depth_map_quality` | image/video | Scene depth complexity |
| 22 | `auto_caption` | `captioning` | image/video +caption | Generated caption |
| 23 | `vqa_a_score` | `aesthetic` | image/video | VQA aesthetic sub-score |
| 24 | `vqa_t_score` | `basic_quality` | image/video | VQA technical sub-score |
| 25 | `is_score` | `inception_score` | image/video | Inception Score |
| 26 | `sd_score` | `sd_reference` | image/video +caption | SD-reference similarity (0-1) |
| 27 | `gradient_detail` | `basic_quality` | image/video | Sobel gradient detail (0-100) |
| 28 | `blip_bleu` | `captioning` | image/video +caption | BLIP caption BLEU score |
| 29 | `detection_score` | `object_detection` | image +caption | Object detection confidence |
| 30 | `count_score` | `object_detection` | image +caption | Object count accuracy |
| 31 | `color_score` | `color_consistency` | image/video +caption | Color consistency |
| 32 | `celebrity_id_score` | `celebrity_id` | image/video | Celebrity identity distance |
| 33 | `identity_loss` | `identity_loss` | image/video +ref | Face identity cosine distance (0-1, lower=better) |
| 34 | `face_recognition_score` | `identity_loss` | image/video +ref | Face identity cosine similarity (0-1, higher=better) |
| 35 | `ocr_score` | `ocr_fidelity` | image/video +caption | OCR confidence |
| 36 | `ocr_fidelity` | `ocr_fidelity` | image/video +caption | OCR text accuracy vs caption (0-100, higher=better) |
| 37 | `i2v_clip` | `i2v_similarity` | image/video +ref | CLIP image-video similarity (0-1) |
| 38 | `i2v_dino` | `i2v_similarity` | image/video +ref | DINOv2 image-video similarity (0-1) |
| 39 | `i2v_lpips` | `i2v_similarity` | image/video +ref | LPIPS image-video distance (0-1, lower=better) |
| 40 | `i2v_quality` | `i2v_similarity` | image/video +ref | Aggregated I2V quality (0-100) |
| 41 | `action_score` | `action_recognition` | image/video +caption | Caption-action fidelity (0-100) |
| 42 | `action_confidence` | `action_recognition` | image/video +caption | Top-1 action confidence (0-100) |
| 43 | `flow_score` | `advanced_flow` | image/video | Optical flow magnitude |
| 44 | `motion_ac_score` | `motion_amplitude` | image/video +caption | Motion amplitude |
| 45 | `warping_error` | `temporal_flickering` | image/video | Warping error |
| 46 | `clip_temp` | `clip_temporal` | image/video | CLIP temporal consistency |
| 47 | `face_consistency` | `clip_temporal` | image/video | Face identity temporal consistency |
| 48 | `psnr` | `structural` | image/video +ref | Peak Signal-to-Noise Ratio (dB, higher=better) |
| 49 | `ssim` | `structural` | image/video +ref | Structural Similarity (0-1, higher=better) |
| 50 | `lpips` | `perceptual_fr` | image/video +ref | Learned Perceptual distance (0-1, lower=better) |
| 51 | `spectral_entropy` | `spectral_complexity` | image/video | DINOv2 spectral entropy |
| 52 | `spectral_rank` | `spectral_complexity` | image/video | DINOv2 effective rank ratio |
| 53 | `fvd` | `fvd` | batch | Frechet Video Distance |
| 54 | `kvd` | `fvd` | batch | Kernel Video Distance |
| 55 | `fvmd` | `fvmd` | batch | Frechet Video Motion Distance |
| 56 | `vmaf` | `vmaf` | image/video +ref | VMAF (0-100, higher=better) |
| 57 | `ms_ssim` | `ms_ssim` | image/video +ref | Multi-Scale SSIM (0-1) |
| 58 | `vif` | `vif` | image +ref | Visual Information Fidelity |
| 59 | `niqe` | `niqe` | image | Natural Image Quality Evaluator (lower=better) |
| 60 | `t2v_score` | `t2v_score` | image/video +caption | T2VScore alignment + quality |
| 61 | `t2v_alignment` | `t2v_score` | image/video +caption | Text-video semantic alignment |
| 62 | `t2v_quality` | `t2v_score` | image/video +caption | Video production quality |
| 63 | `dynamics_range` | `dynamics_range` | image/video | Extent of content variation |
| 64 | `dynamics_controllability` | `dynamics_controllability` | image/video +caption | Motion control fidelity |
| 65 | `scene_complexity` | `scene_complexity` | image/video | Visual complexity score |
| 66 | `compression_artifacts` | `compression_artifacts` | image/video | Artifact severity (0-100) |
| 67 | `naturalness_score` | `naturalness` | image | Natural scene statistics |
| 68 | `video_memorability` | `video_memorability` | image/video | Memorability prediction |
| 69 | `usability_rate` | `usability_rate` | image | Percentage of usable frames |
| 70 | `confidence_score` | `llm_descriptive_qa` | image/video | Prediction confidence |
| 71 | `human_preference_score` | `llm_advisor` | image/video | Human preference (0-1, higher=better) |
| 72 | `engagement_score` | `llm_advisor` | image/video | Engagement prediction (0-1, higher=better) |
| 73 | `usability_score` | `usability_rate` | image/video | Usability estimate (0-100, higher=better) |
| 74 | `hdr_quality` | `hdr_sdr_vqa` | image/video | HDR-specific quality |
| 75 | `sdr_quality` | `hdr_sdr_vqa` | image/video | SDR-specific quality |
| 76 | `temporal_information` | `ti_si` | image/video | ITU-T P.910 TI (higher=more motion) |
| 77 | `spatial_information` | `ti_si` | image/video | ITU-T P.910 SI (higher=more detail) |
| 78 | `flicker_score` | `flicker_detection` | image/video | Flicker severity 0-100 (lower=better) |
| 79 | `judder_score` | `judder_stutter` | image/video | Judder severity 0-100 (lower=better) |
| 80 | `stutter_score` | `judder_stutter` | image/video | Duplicate/dropped frames 0-100 (lower=better) |
| 81 | `dists` | `dists` | image/video +ref | DISTS (0-1, lower=more similar) |
| 82 | `fsim` | `perceptual_fr` | image/video +ref | Feature Similarity Index (0-1, higher=better) |
| 83 | `gmsd` | `perceptual_fr` | image/video +ref | Gradient Magnitude Similarity Deviation (lower=better) |
| 84 | `vsi_score` | `perceptual_fr` | image/video +ref | Visual Saliency Index (0-1, higher=better) |
| 85 | `brisque` | `brisque` | image/video | BRISQUE (0-100, lower=better) |
| 86 | `pesq_score` | `audio_pesq` | audio +ref | PESQ (-0.5 to 4.5, higher=better) |
| 87 | `av_sync_offset` | `av_sync` | audio | Audio-video sync offset in ms |
| 88 | `dover_score` | `dover` | image/video | DOVER overall (higher=better) |
| 89 | `dover_technical` | `dover` | image/video | DOVER technical quality |
| 90 | `dover_aesthetic` | `dover` | image/video | DOVER aesthetic quality |
| 91 | `topiq_score` | `topiq` | image/video | TOPIQ transformer-based IQA (higher=better) |
| 92 | `liqe_score` | `liqe` | image/video | LIQE lightweight IQA (higher=better) |
| 93 | `clip_iqa_score` | `clip_iqa` | image/video | CLIP-IQA semantic quality (0-1, higher=better) |
| 94 | `color_grading_score` | `production_quality` | image/video | Colour consistency 0-100 |
| 95 | `white_balance_score` | `production_quality` | image/video | White balance accuracy 0-100 |
| 96 | `exposure_consistency` | `production_quality` | image/video | Exposure stability 0-100 |
| 97 | `focus_quality` | `production_quality` | image/video | Sharpness/focus quality 0-100 |
| 98 | `banding_severity` | `production_quality` | image/video | Colour banding 0-100 (lower=better) |
| 99 | `qalign_quality` | `q_align` | image/video | Q-Align technical quality (1-5, higher=better) |
| 100 | `qalign_aesthetic` | `q_align` | image/video | Q-Align aesthetic quality (1-5, higher=better) |
| 101 | `face_quality_score` | `face_fidelity` | image/video | Composite face quality 0-100 (higher=better) |
| 102 | `face_identity_consistency` | `face_landmark_quality` | image/video | Temporal face identity stability (0-1) |
| 103 | `face_expression_smoothness` | `face_landmark_quality` | image/video | Expression smoothness 0-100 (higher=better) |
| 104 | `face_landmark_jitter` | `face_landmark_quality` | image/video | Landmark jitter 0-100 (lower=better) |
| 105 | `object_permanence_score` | `object_permanence` | image/video | Object tracking consistency 0-100 (higher=better) |
| 106 | `semantic_consistency` | `semantic_segmentation_consistency` | image/video | Segmentation temporal IoU 0-1 (higher=better) |
| 107 | `depth_temporal_consistency` | `depth_consistency` | image/video | Depth map correlation 0-1 (higher=better) |
| 108 | `subject_consistency` | `subject_consistency` | image/video | Subject identity consistency (0-1, higher=better) |
| 109 | `background_consistency` | `background_consistency` | image/video | Background stability (0-1, higher=better) |
| 110 | `motion_smoothness` | `motion_smoothness` | image/video | Motion smoothness (0-1, higher=better) |
| 111 | `codec_efficiency` | `codec_specific_quality` | image/video | Quality-per-bit efficiency 0-100 (higher=better) |
| 112 | `gop_quality` | `codec_specific_quality` | image/video | GOP structure appropriateness 0-100 (higher=better) |
| 113 | `codec_artifacts` | `codec_specific_quality` | image/video | Block artifact severity 0-100 (lower=better) |
| 114 | `deepfake_probability` | `deepfake_detection` | image/video | Synthetic/deepfake likelihood 0-1 |
| 115 | `ai_generated_probability` | `watermark_classifier` | image/video | AI-generated content likelihood 0-1 |
| 116 | `harmful_content_score` | `harmful_content` | image/video | Violence/gore severity 0-1 |
| 117 | `watermark_strength` | `watermark_robustness` | image/video | Invisible watermark strength 0-1 |
| 118 | `bias_score` | `bias_detection` | image/video | Representation imbalance indicator 0-1 |
| 119 | `depth_quality` | `depth_map_quality` | image/video | Depth map quality 0-100 (higher=better) |
| 120 | `multiview_consistency` | `multi_view_consistency` | image/video | Geometric consistency 0-1 (higher=better) |
| 121 | `stereo_comfort_score` | `stereoscopic_quality` | image/video | Stereo viewing comfort 0-100 (higher=better) |
| 122 | `musiq_score` | `musiq` | image/video | MUSIQ multi-scale IQA (higher=better) |
| 123 | `contrique_score` | `contrique` | image/video | CONTRIQUE contrastive IQA (higher=better) |
| 124 | `mdtvsfa_score` | `mdtvsfa` | image/video | MDTVSFA fragment-based VQA (higher=better) |
| 125 | `nima_score` | `nima` | image/video | NIMA aesthetic+technical (1-10, higher=better) |
| 126 | `dbcnn_score` | `dbcnn` | image/video | DBCNN bilinear CNN (higher=better) |
| 127 | `wadiqam_score` | `wadiqam` | image/video | WaDIQaM-NR (higher=better) |
| 128 | `maniqa_score` | `maniqa` | image/video | MANIQA multi-attention (higher=better) |
| 129 | `arniqa_score` | `arniqa` | image/video | ARNIQA (higher=better) |
| 130 | `qualiclip_score` | `qualiclip` | image/video | QualiCLIP opinion-unaware (higher=better) |
| 131 | `pieapp` | `pieapp` | image/video +ref | PieAPP pairwise preference (lower=better) |
| 132 | `cw_ssim` | `cw_ssim` | image/video +ref | Complex Wavelet SSIM (0-1, higher=better) |
| 133 | `nlpd` | `nlpd` | image/video +ref | Normalized Laplacian Pyramid Distance (lower=better) |
| 134 | `mad` | `mad` | image/video +ref | Most Apparent Distortion (lower=better) |
| 135 | `ahiq` | `ahiq` | image/video +ref | Attention Hybrid IQA (higher=better) |
| 136 | `topiq_fr` | `topiq_fr` | image/video +ref | TOPIQ full-reference (higher=better) |
| 137 | `dreamsim` | `dreamsim` | image/video +ref | DreamSim CLIP+DINO similarity (lower=more similar) |
| 138 | `cover_score` | `cover` | image/video | COVER overall (higher=better) |
| 139 | `cover_technical` | `cover` | image/video | COVER technical branch |
| 140 | `cover_aesthetic` | `cover` | image/video | COVER aesthetic branch |
| 141 | `cover_semantic` | `cover` | image/video | COVER semantic branch |
| 142 | `vqa_score_alignment` | `vqa_score` | image/video +caption | VQAScore text-visual alignment (0-1, higher=better) |
| 143 | `videoscore_visual` | `videoscore` | image/video +caption | VideoScore visual quality |
| 144 | `videoscore_temporal` | `videoscore` | image/video +caption | VideoScore temporal consistency |
| 145 | `videoscore_dynamic` | `videoscore` | image/video +caption | VideoScore dynamic degree |
| 146 | `videoscore_alignment` | `videoscore` | image/video +caption | VideoScore text-video alignment |
| 147 | `videoscore_factual` | `videoscore` | image/video +caption | VideoScore factual consistency |
| 148 | `face_iqa_score` | `face_iqa` | image/video | TOPIQ-face face quality (higher=better) |
| 149 | `scene_stability` | `scene_detection` | image/video | Scene stability (0-1, 1=single continuous scene) |
| 150 | `avg_scene_duration` | `scene_detection` | image/video | Average scene duration in seconds |
| 151 | `raft_motion_score` | `raft_motion` | image/video | RAFT optical flow magnitude |
| 152 | `ram_tags` | `ram_tagging` | image/video | Comma-separated RAM auto-tags |
| 153 | `depth_anything_score` | `depth_anything` | image/video | Monocular depth quality |
| 154 | `depth_anything_consistency` | `depth_anything` | image/video | Temporal depth consistency |
| 155 | `video_type` | `video_type_classifier` | image/video | Content type (real, animated, game, etc.) |
| 156 | `video_type_confidence` | `video_type_classifier` | image/video | Classification confidence |
| 157 | `jedi` | `jedi_metric` | batch | Per-sample V-JEPA feature (batch-computed) |
| 158 | `trajan_score` | `trajan` | image/video | Point track motion consistency |
| 159 | `promptiqa_score` | `promptiqa` | image/video | Few-shot NR-IQA score |
| 160 | `aigv_static` | `aigv_assessor` | image/video +caption | AI video static quality |
| 161 | `aigv_temporal` | `aigv_assessor` | image/video +caption | AI video temporal smoothness |
| 162 | `aigv_dynamic` | `aigv_assessor` | image/video +caption | AI video dynamic degree |
| 163 | `aigv_alignment` | `aigv_assessor` | image/video +caption | AI video text-video alignment |
| 164 | `video_reward_score` | `video_reward` | image/video +caption | Human preference reward |
| 165 | `tifa_score` | `tifa` | image/video +caption | VQA faithfulness (0-1, higher=better) |
| 166 | `text_overlay_score` | `text_detection` | image | Text overlay severity (0-1) |
| 167 | `ptlflow_motion_score` | `ptlflow_motion` | image/video | ptlflow optical flow magnitude |
| 168 | `qcn_score` | `qcn` | image/video | Geometric order blind IQA |
| 169 | `finevq_score` | `finevq` | image/video | FineVQ fine-grained UGC VQA (CVPR 2025) |
| 170 | `kvq_score` | `kvq` | image/video | KVQ saliency-guided VQA (CVPR 2025) |
| 171 | `rqvqa_score` | `rqvqa` | image/video | RQ-VQA rich quality-aware (CVPR 2024 winner) |
| 172 | `videval_score` | `videval` | image/video | VIDEVAL 60-feature fusion NR-VQA |
| 173 | `tlvqm_score` | `tlvqm` | image/video | TLVQM two-level video quality |
| 174 | `funque_score` | `funque` | image/video +ref | FUNQUE unified quality (beats VMAF) |
| 175 | `movie_score` | `movie` | image/video +ref | MOVIE motion trajectory FR |
| 176 | `st_greed_score` | `st_greed` | image/video +ref | ST-GREED variable frame rate FR |
| 177 | `c3dvqa_score` | `c3dvqa` | image/video | C3DVQA 3D CNN spatiotemporal FR |
| 178 | `flolpips` | `flolpips` | image/video | FloLPIPS flow-based perceptual FR |
| 179 | `hdr_vqm` | `hdr_vqm` | image/video +ref | HDR-VQM HDR video quality FR |
| 180 | `st_lpips` | `st_lpips` | image/video | ST-LPIPS spatiotemporal perceptual FR |
| 181 | `camera_jitter_score` | `camera_jitter` | image/video | Camera stability (0-1, 1=stable) |
| 182 | `jump_cut_score` | `jump_cut` | image/video | Jump cut absence (0-1, 1=no cuts) |
| 183 | `playback_speed_score` | `playback_speed` | image/video | Normal speed (1.0=normal) |
| 184 | `flow_coherence` | `flow_coherence` | image/video | Bidirectional optical flow consistency (0-1) |
| 185 | `letterbox_ratio` | `letterbox` | image/video | Border/letterbox fraction (0-1, 0=no borders) |
| 186 | `tonal_dynamic_range` | `tonal_dynamic_range` | image/video | Luminance histogram span (0-100) |
| 187 | `vtss` | `vtss` | image | Video Training Suitability Score (0-1) |
| 188 | `cnniqa_score` | `cnniqa` | image/video | CNNIQA blind CNN IQA |
| 189 | `hyperiqa_score` | `hyperiqa` | image/video | HyperIQA adaptive NR-IQA |
| 190 | `paq2piq_score` | `paq2piq` | image/video | PaQ-2-PiQ patch-to-picture (CVPR 2020) |
| 191 | `tres_score` | `tres` | image/video | TReS transformer IQA (WACV 2022) |
| 192 | `unique_score` | `unique` | image/video | UNIQUE unified NR-IQA (TIP 2021) |
| 193 | `laion_aesthetic` | `laion_aesthetic` | image/video | LAION Aesthetics V2 (0-10) |
| 194 | `compare2score` | `compare2score` | image/video | Compare2Score comparison-based |
| 195 | `afine_score` | `afine` | image/video | A-FINE fidelity-naturalness (CVPR 2025) |
| 196 | `ckdn_score` | `ckdn` | image +ref | CKDN knowledge distillation FR |
| 197 | `deepwsd_score` | `deepwsd` | image +ref | DeepWSD Wasserstein distance FR |
| 198 | `ssimulacra2` | `ssimulacra2` | image/video +ref | SSIMULACRA 2 (0-100, lower=better, JPEG XL standard) |
| 199 | `butteraugli` | `butteraugli` | image/video +ref | Butteraugli perceptual distance (lower=better) |
| 200 | `flip_score` | `flip` | image/video +ref | NVIDIA FLIP perceptual metric (0-1, lower=better) |
| 201 | `vmaf_neg` | `vmaf_neg` | image/video +ref | VMAF NEG (no enhancement gain, 0-100, higher=better) |
| 202 | `ilniqe` | `ilniqe` | image/video | IL-NIQE Integrated Local NIQE (lower=better) |
| 203 | `nrqm` | `nrqm` | image/video | NRQM No-Reference Quality Metric (higher=better) |
| 204 | `pi_score` | `pi` | image/video | Perceptual Index (PIRM challenge, lower=better) |
| 205 | `piqe` | `piqe` | image/video | PIQE perception-based NR-IQA (lower=better) |
| 206 | `maclip_score` | `maclip` | image/video | MACLIP multi-attribute CLIP NR-IQA (higher=better) |
| 207 | `dmm` | `dmm` | image/video +ref | DMM Detail Model Metric FR (higher=better) |
| 208 | `wadiqam_fr` | `wadiqam_fr` | image/video +ref | WaDIQaM full-reference (higher=better) |
| 209 | `ssimc` | `ssimc` | image/video +ref | Complex Wavelet SSIM-C FR (higher=better) |
| 210 | `cambi` | `cambi` | image/video | CAMBI banding index (0-24, lower=better) |
| 211 | `xpsnr` | `xpsnr` | image +ref | XPSNR perceptual PSNR (dB, higher=better) |
| 212 | `vmaf_phone` | `vmaf_phone` | image/video +ref | VMAF phone model (0-100, higher=better) |
| 213 | `vmaf_4k` | `vmaf_4k` | image/video +ref | VMAF 4K model (0-100, higher=better) |
| 214 | `visqol` | `visqol` | audio +ref | ViSQOL audio quality MOS (1-5, higher=better) |
| 215 | `dnsmos_overall` | `dnsmos` | audio | DNSMOS overall MOS (1-5, higher=better) |
| 216 | `dnsmos_sig` | `dnsmos` | audio | DNSMOS signal quality (1-5, higher=better) |
| 217 | `dnsmos_bak` | `dnsmos` | audio | DNSMOS background quality (1-5, higher=better) |
| 218 | `pu_psnr` | `pu_metrics` | image/video +ref | PU-PSNR perceptually uniform HDR (dB, higher=better) |
| 219 | `pu_ssim` | `pu_metrics` | image/video +ref | PU-SSIM perceptually uniform HDR (0-1, higher=better) |
| 220 | `max_fall` | `hdr_metadata` | image/video | MaxFALL frame average light level (nits) |
| 221 | `max_cll` | `hdr_metadata` | image/video | MaxCLL content light level (nits) |
| 222 | `hdr_vdp` | `hdr_vdp` | image/video +ref | HDR-VDP visual difference predictor (higher=better) |
| 223 | `delta_ictcp` | `delta_ictcp` | image/video +ref | Delta ICtCp HDR color difference (lower=better) |
| 224 | `ciede2000` | `ciede2000` | image/video +ref | CIEDE2000 perceptual color difference (lower=better) |
| 225 | `psnr_hvs` | `psnr_hvs` | image/video +ref | PSNR-HVS perceptually weighted (dB, higher=better) |
| 226 | `psnr_hvs_m` | `psnr_hvs` | image +ref | PSNR-HVS-M with masking (dB, higher=better) |
| 227 | `cgvqm` | `cgvqm` | image/video +ref | CGVQM gaming quality (higher=better) |
| 228 | `strred` | `strred` | image +ref | STRRED reduced-reference temporal (lower=better) |
| 229 | `p1203_mos` | `p1203` | audio | ITU-T P.1203 streaming QoE MOS (1-5) |
| 230 | `nemo_quality_score` | `nemo_curator` | image +caption | Caption text quality (0-1) |
| 231 | `nemo_quality_label` | `nemo_curator` | image +caption | Quality label (Low/Medium/High) |

## Quick Start

### CLI

```bash
# Scan a dataset and get a report
ayase scan ./my_dataset

# Scan with specific modules
ayase scan ./my_dataset --modules metadata,basic_quality,motion

# List all available modules
ayase modules list

# Check which modules can be loaded (dependencies installed)
ayase modules check

# Filter dataset by quality score
ayase filter ./my_dataset --min-score 70 --output ./filtered
```

### Python API (recommended)

```python
from ayase import AyasePipeline

ayase = AyasePipeline(modules=["basic"])
results = ayase.run("./my_dataset")

for path, sample in results.items():
    if sample.quality_metrics:
        print(f"{sample.path.name}: technical={sample.quality_metrics.technical_score}")

print(f"Total: {ayase.stats.total_samples}, Valid: {ayase.stats.valid_samples}")
ayase.export("report.json")
```

`AyasePipeline` accepts three ways to configure modules:

```python
# By module names
ayase = AyasePipeline(modules=["metadata", "basic_quality", "motion"])

# By profile dict
ayase = AyasePipeline(profile={
    "name": "my_check",
    "modules": ["basic", "aesthetic"],
    "module_config": {
        "aesthetic": {"model_name": "openai/clip-vit-large-patch14"},
    },
})

# By profile file
ayase = AyasePipeline(profile="my_profile.toml")

# With custom config
from ayase.config import AyaseConfig
ayase = AyasePipeline(config=AyaseConfig(general={"parallel_jobs": 16}), modules=["basic"])
```

### Low-level Pipeline API

```python
import asyncio
from pathlib import Path
from ayase.pipeline import Pipeline, ModuleRegistry
from ayase.scanner import scan_dataset

ModuleRegistry.discover_modules()
module_names = ["metadata", "basic_quality", "semantic_alignment"]
modules = [ModuleRegistry.get_module(n)() for n in module_names]

pipeline = Pipeline(modules)
pipeline.start()

samples = scan_dataset(Path("./my_dataset"), recursive=True)
for sample in samples:
    processed = asyncio.run(pipeline.process_sample(sample))

pipeline.stop()
pipeline.export_report("report.json", format="json")
```

### Profile-based pipelines

```python
from ayase import load_profile, instantiate_profile_modules

profile = load_profile("my_profile.toml")
modules = instantiate_profile_modules(profile)
# modules is a list of PipelineModule instances ready for Pipeline()
```

## Configuration

Create `ayase.toml` in your project root:

```toml
[general]
parallel_jobs = 8
cache_enabled = true

[quality]
enable_blur_detection = true
blur_threshold = 100.0

[pipeline]
dataset_path = "./my_dataset"
modules = ["metadata", "basic_quality", "motion"]
plugin_folders = ["plugins"]

[output]
default_format = "markdown"
artifacts_dir = "reports"
artifacts_format = "json"

[filter]
default_mode = "list"
min_score_threshold = 60
```

Ayase looks for config in: `./ayase.toml` -> `~/.config/ayase/config.toml` -> built-in defaults.

## Writing Plugins

Create a `.py` file in your `plugins/` folder:

```python
from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

class MyCustomCheck(PipelineModule):
    name = "my_check"
    description = "Custom quality check"
    default_config = {"threshold": 0.5}

    def process(self, sample: Sample) -> Sample:
        # Your logic here
        if some_score < self.config["threshold"]:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message="Quality below threshold",
                )
            )
        return sample
```

Then run:

```bash
ayase scan ./data --modules metadata,my_check
```

## Development

```bash
git clone <repo-url>
cd ayase
pip install -e ".[dev]"

# Run tests
pytest

# Lint and format
ruff check src/ tests/
black src/ tests/

# Type check
mypy src/ayase
```

## License

MIT -- see [LICENSE](LICENSE).
