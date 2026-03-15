# Ayase

Modular media quality metrics toolkit.  

⚠️ Work in progress. Some features may not work as expected.

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

**Input:** `img/vid` = image or video, `+ref` = needs `reference_path`, `+cap` = needs caption, `batch` = dataset-level.

| # | Metric | Module | Input | Description |
|---:|--------|--------|-------|-------------|
| 1 | blur_score | basic_quality | img/vid | Laplacian variance |
| 2 | compression_score | basic_quality | img/vid | Compression quality (0-100) |
| 3 | aesthetic_score | aesthetic | img/vid | 0-10, from aesthetic predictor |
| 4 | clip_score | semantic_alignment | img/vid +cap | Caption-image alignment |
| 5 | brightness | basic_quality | img/vid | Mean brightness (0-255) |
| 6 | contrast | basic_quality | img/vid | Contrast level (0-100) |
| 7 | saturation | basic_quality | img/vid | Color saturation (0-100) |
| 8 | fast_vqa_score | fast_vqa | img/vid | 0-100 |
| 9 | motion_score | motion | img/vid | Scene motion intensity |
| 10 | camera_motion_score | stabilized_motion | img/vid | Camera motion intensity |
| 11 | temporal_consistency | clip_temporal | img/vid | Frame consistency |
| 12 | technical_score | basic_quality | img/vid | Composite technical score |
| 13 | noise_score | basic_quality | img/vid | Noise level (0-100, lower=cleaner) |
| 14 | artifacts_score | basic_quality | img/vid | Artifact severity (0-100, lower=better) |
| 15 | watermark_probability | watermark_classifier | img/vid | 0-1 |
| 16 | ocr_area_ratio | text_detection | img | 0-1 |
| 17 | face_count | face_fidelity | img/vid | Integer face count |
| 18 | nsfw_score | nsfw | img | 0-1, likelihood of being NSFW |
| 19 | audio_quality_score | audio | audio | 0-100 |
| 20 | perceptual_hash | dedup | img/vid | dHash or similar |
| 21 | depth_score | depth_map_quality | img/vid | Scene depth complexity |
| 22 | auto_caption | captioning | img/vid +cap | Generated caption |
| 23 | vqa_a_score | aesthetic | img/vid | VQA aesthetic sub-score |
| 24 | vqa_t_score | basic_quality | img/vid | VQA technical sub-score |
| 25 | is_score | inception_score | img/vid | Inception Score |
| 26 | sd_score | sd_reference | img/vid +cap | SD-reference similarity (0-1) |
| 27 | gradient_detail | basic_quality | img/vid | Sobel gradient detail (0-100) |
| 28 | blip_bleu | captioning | img/vid +cap | BLIP caption BLEU score |
| 29 | detection_score | object_detection | img +cap | Object detection confidence |
| 30 | count_score | object_detection | img +cap | Object count accuracy |
| 31 | color_score | color_consistency | img/vid +cap | Color consistency |
| 32 | celebrity_id_score | celebrity_id | img/vid | Celebrity identity distance |
| 33 | identity_loss | identity_loss | img/vid +ref | Face identity distance (0-1, lower=better) |
| 34 | face_recognition_score | identity_loss | img/vid +ref | Face identity similarity (0-1, higher=better) |
| 35 | ocr_score | ocr_fidelity | img/vid +cap | OCR confidence |
| 36 | ocr_fidelity | ocr_fidelity | img/vid +cap | OCR text accuracy vs caption (0-100, higher=better) |
| 37 | i2v_clip | i2v_similarity | img/vid +ref | CLIP image-video similarity (0-1) |
| 38 | i2v_dino | i2v_similarity | img/vid +ref | DINOv2 image-video similarity (0-1) |
| 39 | i2v_lpips | i2v_similarity | img/vid +ref | LPIPS image-video distance (0-1, lower=better) |
| 40 | i2v_quality | i2v_similarity | img/vid +ref | Aggregated I2V quality (0-100) |
| 41 | action_score | action_recognition | img/vid +cap | Caption-action fidelity (0-100) |
| 42 | action_confidence | action_recognition | img/vid +cap | Top-1 action confidence (0-100) |
| 43 | flow_score | advanced_flow | img/vid | Optical flow magnitude |
| 44 | motion_ac_score | motion_amplitude | img/vid +cap | Motion amplitude |
| 45 | warping_error | temporal_flickering | img/vid | Warping error |
| 46 | clip_temp | clip_temporal | img/vid | CLIP temporal consistency |
| 47 | face_consistency | clip_temporal | img/vid | Face identity temporal consistency |
| 48 | psnr | structural | img/vid +ref | Peak Signal-to-Noise Ratio (dB, higher=better) |
| 49 | ssim | structural | img/vid +ref | Structural Similarity (0-1, higher=better) |
| 50 | lpips | perceptual_fr | img/vid +ref | Learned Perceptual distance (0-1, lower=better) |
| 51 | spectral_entropy | spectral_complexity | img/vid | DINOv2 spectral entropy |
| 52 | spectral_rank | spectral_complexity | img/vid | DINOv2 effective rank ratio |
| 53 | fvd | fvd | batch | Frechet Video Distance |
| 54 | kvd | fvd | batch | Kernel Video Distance |
| 55 | fvmd | fvmd | batch | Frechet Video Motion Distance |
| 56 | vmaf | vmaf | img/vid +ref | VMAF (0-100, higher=better) |
| 57 | ms_ssim | ms_ssim | img/vid +ref | Multi-Scale SSIM (0-1) |
| 58 | vif | vif | img +ref | Visual Information Fidelity |
| 59 | niqe | niqe | img | Natural Image Quality Evaluator (lower=better) |
| 60 | t2v_score | t2v_score | img/vid +cap | T2VScore alignment + quality |
| 61 | t2v_alignment | t2v_score | img/vid +cap | Text-video semantic alignment |
| 62 | t2v_quality | t2v_score | img/vid +cap | Video production quality |
| 63 | dynamics_range | dynamics_range | img/vid | Extent of content variation |
| 64 | dynamics_controllability | dynamics_controllability | img/vid +cap | Motion control fidelity |
| 65 | scene_complexity | scene_complexity | img/vid | Visual complexity score |
| 66 | compression_artifacts | compression_artifacts | img/vid | Artifact severity (0-100) |
| 67 | naturalness_score | naturalness | img | Natural scene statistics |
| 68 | video_memorability | video_memorability | img/vid | Memorability prediction |
| 69 | usability_rate | usability_rate | img | Percentage of usable frames |
| 70 | confidence_score | llm_descriptive_qa | img/vid | Prediction confidence |
| 71 | human_preference_score | llm_advisor | img/vid | Human preference (0-1, higher=better) |
| 72 | engagement_score | llm_advisor | img/vid | Engagement prediction (0-1, higher=better) |
| 73 | usability_score | usability_rate | img/vid | Usability estimate (0-100, higher=better) |
| 74 | hdr_quality | hdr_sdr_vqa | img/vid | HDR-specific quality |
| 75 | sdr_quality | hdr_sdr_vqa | img/vid | SDR-specific quality |
| 76 | temporal_information | ti_si | img/vid | ITU-T P.910 TI (higher=more motion) |
| 77 | spatial_information | ti_si | img/vid | ITU-T P.910 SI (higher=more detail) |
| 78 | flicker_score | flicker_detection | img/vid | Flicker severity 0-100 (lower=better) |
| 79 | judder_score | judder_stutter | img/vid | Judder severity 0-100 (lower=better) |
| 80 | stutter_score | judder_stutter | img/vid | Duplicate/dropped frames 0-100 (lower=better) |
| 81 | dists | dists | img/vid +ref | DISTS (0-1, lower=more similar) |
| 82 | fsim | perceptual_fr | img/vid +ref | Feature Similarity Index (0-1, higher=better) |
| 83 | gmsd | perceptual_fr | img/vid +ref | Gradient Magnitude Similarity Deviation (lower=better) |
| 84 | vsi_score | perceptual_fr | img/vid +ref | Visual Saliency Index (0-1, higher=better) |
| 85 | brisque | brisque | img/vid | BRISQUE (0-100, lower=better) |
| 86 | pesq_score | audio_pesq | audio +ref | PESQ (-0.5 to 4.5, higher=better) |
| 87 | av_sync_offset | av_sync | audio | Audio-video sync offset in ms |
| 88 | dover_score | dover | img/vid | DOVER overall (higher=better) |
| 89 | dover_technical | dover | img/vid | DOVER technical quality |
| 90 | dover_aesthetic | dover | img/vid | DOVER aesthetic quality |
| 91 | topiq_score | topiq | img/vid | TOPIQ transformer-based IQA (higher=better) |
| 92 | liqe_score | liqe | img/vid | LIQE lightweight IQA (higher=better) |
| 93 | clip_iqa_score | clip_iqa | img/vid | CLIP-IQA semantic quality (0-1, higher=better) |
| 94 | color_grading_score | production_quality | img/vid | Colour consistency 0-100 |
| 95 | white_balance_score | production_quality | img/vid | White balance accuracy 0-100 |
| 96 | exposure_consistency | production_quality | img/vid | Exposure stability 0-100 |
| 97 | focus_quality | production_quality | img/vid | Sharpness/focus quality 0-100 |
| 98 | banding_severity | production_quality | img/vid | Colour banding 0-100 (lower=better) |
| 99 | qalign_quality | q_align | img/vid | Q-Align technical quality (1-5, higher=better) |
| 100 | qalign_aesthetic | q_align | img/vid | Q-Align aesthetic quality (1-5, higher=better) |
| 101 | face_quality_score | face_fidelity | img/vid | Composite face quality 0-100 (higher=better) |
| 102 | face_identity_consistency | face_landmark_quality | img/vid | Temporal face identity stability (0-1) |
| 103 | face_expression_smoothness | face_landmark_quality | img/vid | Expression smoothness (0-100) |
| 104 | face_landmark_jitter | face_landmark_quality | img/vid | Landmark jitter 0-100 (lower=better) |
| 105 | object_permanence_score | object_permanence | img/vid | Object tracking (0-100) |
| 106 | semantic_consistency | semantic_segmentation_consistency | img/vid | Segmentation IoU (0-1) |
| 107 | depth_temporal_consistency | depth_consistency | img/vid | Depth map correlation 0-1 (higher=better) |
| 108 | subject_consistency | subject_consistency | img/vid | Subject identity (0-1) |
| 109 | background_consistency | background_consistency | img/vid | Background stability (0-1, higher=better) |
| 110 | motion_smoothness | motion_smoothness | img/vid | Motion smoothness (0-1, higher=better) |
| 111 | codec_efficiency | codec_specific_quality | img/vid | Quality-per-bit efficiency 0-100 (higher=better) |
| 112 | gop_quality | codec_specific_quality | img/vid | GOP structure appropriateness 0-100 (higher=better) |
| 113 | codec_artifacts | codec_specific_quality | img/vid | Block artifact severity 0-100 (lower=better) |
| 114 | deepfake_probability | deepfake_detection | img/vid | Synthetic/deepfake likelihood 0-1 |
| 115 | ai_generated_probability | watermark_classifier | img/vid | AI-generated content likelihood 0-1 |
| 116 | harmful_content_score | harmful_content | img/vid | Violence/gore severity 0-1 |
| 117 | watermark_strength | watermark_robustness | img/vid | Invisible watermark strength 0-1 |
| 118 | bias_score | bias_detection | img/vid | Representation imbalance indicator 0-1 |
| 119 | depth_quality | depth_map_quality | img/vid | Depth map quality 0-100 (higher=better) |
| 120 | multiview_consistency | multi_view_consistency | img/vid | Geometric consistency 0-1 (higher=better) |
| 121 | stereo_comfort_score | stereoscopic_quality | img/vid | Stereo viewing comfort 0-100 (higher=better) |
| 122 | musiq_score | musiq | img/vid | MUSIQ multi-scale IQA (higher=better) |
| 123 | contrique_score | contrique | img/vid | CONTRIQUE contrastive IQA (higher=better) |
| 124 | mdtvsfa_score | mdtvsfa | img/vid | MDTVSFA fragment-based VQA (higher=better) |
| 125 | nima_score | nima | img/vid | NIMA aesthetic+technical (1-10, higher=better) |
| 126 | dbcnn_score | dbcnn | img/vid | DBCNN bilinear CNN (higher=better) |
| 127 | wadiqam_score | wadiqam | img/vid | WaDIQaM-NR (higher=better) |
| 128 | maniqa_score | maniqa | img/vid | MANIQA multi-attention (higher=better) |
| 129 | arniqa_score | arniqa | img/vid | ARNIQA (higher=better) |
| 130 | qualiclip_score | qualiclip | img/vid | QualiCLIP opinion-unaware (higher=better) |
| 131 | pieapp | pieapp | img/vid +ref | PieAPP pairwise preference (lower=better) |
| 132 | cw_ssim | cw_ssim | img/vid +ref | Complex Wavelet SSIM (0-1, higher=better) |
| 133 | nlpd | nlpd | img/vid +ref | Normalized Laplacian Pyramid Distance (lower=better) |
| 134 | mad | mad | img/vid +ref | Most Apparent Distortion (lower=better) |
| 135 | ahiq | ahiq | img/vid +ref | Attention Hybrid IQA (higher=better) |
| 136 | topiq_fr | topiq_fr | img/vid +ref | TOPIQ full-reference (higher=better) |
| 137 | dreamsim | dreamsim | img/vid +ref | DreamSim CLIP+DINO similarity (lower=more similar) |
| 138 | cover_score | cover | img/vid | COVER overall (higher=better) |
| 139 | cover_technical | cover | img/vid | COVER technical branch |
| 140 | cover_aesthetic | cover | img/vid | COVER aesthetic branch |
| 141 | cover_semantic | cover | img/vid | COVER semantic branch |
| 142 | vqa_score_alignment | vqa_score | img/vid +cap | VQAScore text-visual alignment (0-1, higher=better) |
| 143 | videoscore_visual | videoscore | img/vid +cap | VideoScore visual quality |
| 144 | videoscore_temporal | videoscore | img/vid +cap | VideoScore temporal consistency |
| 145 | videoscore_dynamic | videoscore | img/vid +cap | VideoScore dynamic degree |
| 146 | videoscore_alignment | videoscore | img/vid +cap | VideoScore text-video alignment |
| 147 | videoscore_factual | videoscore | img/vid +cap | VideoScore factual consistency |
| 148 | face_iqa_score | face_iqa | img/vid | TOPIQ-face face quality (higher=better) |
| 149 | scene_stability | scene_detection | img/vid | Scene stability (0-1, 1=single continuous scene) |
| 150 | avg_scene_duration | scene_detection | img/vid | Average scene duration in seconds |
| 151 | raft_motion_score | raft_motion | img/vid | RAFT optical flow magnitude |
| 152 | ram_tags | ram_tagging | img/vid | Comma-separated RAM auto-tags |
| 153 | depth_anything_score | depth_anything | img/vid | Monocular depth quality |
| 154 | depth_anything_consistency | depth_anything | img/vid | Temporal depth consistency |
| 155 | video_type | video_type_classifier | img/vid | Content type (real, animated, game, etc.) |
| 156 | video_type_confidence | video_type_classifier | img/vid | Classification confidence |
| 157 | jedi | jedi_metric | batch | Per-sample V-JEPA feature (batch-computed) |
| 158 | trajan_score | trajan | img/vid | Point track motion consistency |
| 159 | promptiqa_score | promptiqa | img/vid | Few-shot NR-IQA score |
| 160 | aigv_static | aigv_assessor | img/vid +cap | AI video static quality |
| 161 | aigv_temporal | aigv_assessor | img/vid +cap | AI video temporal smoothness |
| 162 | aigv_dynamic | aigv_assessor | img/vid +cap | AI video dynamic degree |
| 163 | aigv_alignment | aigv_assessor | img/vid +cap | AI video text-video alignment |
| 164 | video_reward_score | video_reward | img/vid +cap | Human preference reward |
| 165 | tifa_score | tifa | img/vid +cap | VQA faithfulness (0-1, higher=better) |
| 166 | text_overlay_score | text_detection | img | Text overlay severity (0-1) |
| 167 | ptlflow_motion_score | ptlflow_motion | img/vid | ptlflow optical flow magnitude |
| 168 | qcn_score | qcn | img/vid | Geometric order blind IQA |
| 169 | finevq_score | finevq | img/vid | FineVQ fine-grained UGC VQA (CVPR 2025) |
| 170 | kvq_score | kvq | img/vid | KVQ saliency-guided VQA (CVPR 2025) |
| 171 | rqvqa_score | rqvqa | img/vid | RQ-VQA rich quality-aware (CVPR 2024 winner) |
| 172 | videval_score | videval | img/vid | VIDEVAL 60-feature fusion NR-VQA |
| 173 | tlvqm_score | tlvqm | img/vid | TLVQM two-level video quality |
| 174 | funque_score | funque | img/vid +ref | FUNQUE unified quality (beats VMAF) |
| 175 | movie_score | movie | img/vid +ref | MOVIE motion trajectory FR |
| 176 | st_greed_score | st_greed | img/vid +ref | ST-GREED variable frame rate FR |
| 177 | c3dvqa_score | c3dvqa | img/vid | C3DVQA 3D CNN spatiotemporal FR |
| 178 | flolpips | flolpips | img/vid | FloLPIPS flow-based perceptual FR |
| 179 | hdr_vqm | hdr_vqm | img/vid +ref | HDR-VQM HDR video quality FR |
| 180 | st_lpips | st_lpips | img/vid | ST-LPIPS spatiotemporal perceptual FR |
| 181 | camera_jitter_score | camera_jitter | img/vid | Camera stability (0-1, 1=stable) |
| 182 | jump_cut_score | jump_cut | img/vid | Jump cut absence (0-1, 1=no cuts) |
| 183 | playback_speed_score | playback_speed | img/vid | Normal speed (1.0=normal) |
| 184 | flow_coherence | flow_coherence | img/vid | Bidirectional optical flow consistency (0-1) |
| 185 | letterbox_ratio | letterbox | img/vid | Border/letterbox fraction (0-1, 0=no borders) |
| 186 | tonal_dynamic_range | tonal_dynamic_range | img/vid | Luminance histogram span (0-100) |
| 187 | vtss | vtss | img | Video Training Suitability Score (0-1) |
| 188 | cnniqa_score | cnniqa | img/vid | CNNIQA blind CNN IQA |
| 189 | hyperiqa_score | hyperiqa | img/vid | HyperIQA adaptive NR-IQA |
| 190 | paq2piq_score | paq2piq | img/vid | PaQ-2-PiQ patch-to-picture (CVPR 2020) |
| 191 | tres_score | tres | img/vid | TReS transformer IQA (WACV 2022) |
| 192 | unique_score | unique | img/vid | UNIQUE unified NR-IQA (TIP 2021) |
| 193 | laion_aesthetic | laion_aesthetic | img/vid | LAION Aesthetics V2 (0-10) |
| 194 | compare2score | compare2score | img/vid | Compare2Score comparison-based |
| 195 | afine_score | afine | img/vid | A-FINE fidelity-naturalness (CVPR 2025) |
| 196 | ckdn_score | ckdn | img +ref | CKDN knowledge distillation FR |
| 197 | deepwsd_score | deepwsd | img +ref | DeepWSD Wasserstein distance FR |
| 198 | ssimulacra2 | ssimulacra2 | img/vid +ref | SSIMULACRA 2 (0-100, lower=better, JPEG XL standard) |
| 199 | butteraugli | butteraugli | img/vid +ref | Butteraugli perceptual distance (lower=better) |
| 200 | flip_score | flip | img/vid +ref | NVIDIA FLIP perceptual metric (0-1, lower=better) |
| 201 | vmaf_neg | vmaf_neg | img/vid +ref | VMAF NEG (no enhancement gain, 0-100, higher=better) |
| 202 | ilniqe | ilniqe | img/vid | IL-NIQE Integrated Local NIQE (lower=better) |
| 203 | nrqm | nrqm | img/vid | NRQM No-Reference Quality Metric (higher=better) |
| 204 | pi_score | pi | img/vid | Perceptual Index (PIRM challenge, lower=better) |
| 205 | piqe | piqe | img/vid | PIQE perception-based NR-IQA (lower=better) |
| 206 | maclip_score | maclip | img/vid | MACLIP multi-attribute CLIP NR-IQA (higher=better) |
| 207 | dmm | dmm | img/vid +ref | DMM Detail Model Metric FR (higher=better) |
| 208 | wadiqam_fr | wadiqam_fr | img/vid +ref | WaDIQaM full-reference (higher=better) |
| 209 | ssimc | ssimc | img/vid +ref | Complex Wavelet SSIM-C FR (higher=better) |
| 210 | cambi | cambi | img/vid | CAMBI banding index (0-24, lower=better) |
| 211 | xpsnr | xpsnr | img +ref | XPSNR perceptual PSNR (dB, higher=better) |
| 212 | vmaf_phone | vmaf_phone | img/vid +ref | VMAF phone model (0-100, higher=better) |
| 213 | vmaf_4k | vmaf_4k | img/vid +ref | VMAF 4K model (0-100, higher=better) |
| 214 | visqol | visqol | audio +ref | ViSQOL audio quality MOS (1-5, higher=better) |
| 215 | dnsmos_overall | dnsmos | audio | DNSMOS overall MOS (1-5, higher=better) |
| 216 | dnsmos_sig | dnsmos | audio | DNSMOS signal quality (1-5, higher=better) |
| 217 | dnsmos_bak | dnsmos | audio | DNSMOS background quality (1-5, higher=better) |
| 218 | pu_psnr | pu_metrics | img/vid +ref | PU-PSNR perceptually uniform HDR (dB, higher=better) |
| 219 | pu_ssim | pu_metrics | img/vid +ref | PU-SSIM perceptually uniform HDR (0-1, higher=better) |
| 220 | max_fall | hdr_metadata | img/vid | MaxFALL frame average light level (nits) |
| 221 | max_cll | hdr_metadata | img/vid | MaxCLL content light level (nits) |
| 222 | hdr_vdp | hdr_vdp | img/vid +ref | HDR-VDP visual difference predictor (higher=better) |
| 223 | delta_ictcp | delta_ictcp | img/vid +ref | Delta ICtCp HDR color difference (lower=better) |
| 224 | ciede2000 | ciede2000 | img/vid +ref | CIEDE2000 perceptual color difference (lower=better) |
| 225 | psnr_hvs | psnr_hvs | img/vid +ref | PSNR-HVS perceptually weighted (dB, higher=better) |
| 226 | psnr_hvs_m | psnr_hvs | img +ref | PSNR-HVS-M with masking (dB, higher=better) |
| 227 | cgvqm | cgvqm | img/vid +ref | CGVQM gaming quality (higher=better) |
| 228 | strred | strred | img +ref | STRRED reduced-reference temporal (lower=better) |
| 229 | p1203_mos | p1203 | audio | ITU-T P.1203 streaming QoE MOS (1-5) |
| 230 | nemo_quality_score | nemo_curator | img +cap | Caption text quality (0-1) |
| 231 | nemo_quality_label | nemo_curator | img +cap | Quality label (Low/Medium/High) |

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
